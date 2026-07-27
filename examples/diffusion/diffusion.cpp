#include "diffusion.h"

#include "log.h"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <thread>
#include <utility>
#include <vector>

static float calculate_confidence(const llama_token_data_array & cur_p,
                                  diffusion_algorithm            algorithm,
                                  std::mt19937 &                 rng) {
    switch (algorithm) {
        case DIFFUSION_ALGORITHM_CONFIDENCE_BASED:
            return cur_p.data[cur_p.selected].p;  // Selected token probability

        case DIFFUSION_ALGORITHM_ENTROPY_BASED:
            {
                float       entropy = 0.0f;
                const float epsilon = 1e-10f;
                for (size_t i = 0; i < cur_p.size; i++) {
                    float prob = cur_p.data[i].p;
                    entropy += prob * logf(prob + epsilon);
                }
                return -entropy;  // Higher entropy = lower confidence
            }

        case DIFFUSION_ALGORITHM_MARGIN_BASED:
            return (cur_p.size > 1) ? cur_p.data[0].p - cur_p.data[1].p : cur_p.data[0].p;

        case DIFFUSION_ALGORITHM_RANDOM:
            {
                std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
                return uniform(rng);  // Random confidence
            }

        case DIFFUSION_ALGORITHM_ORIGIN:
            return cur_p.data[cur_p.selected].p;

        default:
            return 0.0f;
    }
}

// Unified transfer count calculation function
static int32_t calculate_transfer_count(int32_t                      step,
                                        int32_t                      total_steps,
                                        int32_t                      remaining_masked,
                                        diffusion_transfer_schedule  schedule,
                                        float                        eps,
                                        const std::vector<int32_t> & num_transfer_tokens = {}) {
    switch (schedule) {
        case DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED:
            {
                float t          = 1.0f - (float) step / total_steps * (1.0f - eps);
                float s          = 1.0f - (float) (step + 1) / total_steps * (1.0f - eps);
                float p_transfer = (step < total_steps - 1) ? (1.0f - s / t) : 1.0f;
                return (int32_t) (remaining_masked * p_transfer);
            }

        case DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED:
            if (!num_transfer_tokens.empty() && step < (int32_t) num_transfer_tokens.size()) {
                return num_transfer_tokens[step];
            }
            return remaining_masked / (total_steps - step);  // Fallback

        default:
            return remaining_masked / (total_steps - step);
    }
}

static void add_gumbel_noise(float * logits, int32_t n_vocab, float temperature, std::mt19937 & rng) {
    if (temperature == 0.0f) {
        return;
    }

    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    for (int32_t i = 0; i < n_vocab; i++) {
        double noise        = uniform(rng);
        // Prevent log(0)
        noise               = std::max(noise, 1e-20);
        double gumbel_noise = std::pow(-std::log(noise), temperature);
        logits[i]           = std::exp(logits[i]) / gumbel_noise;
    }
}

static std::vector<int32_t> get_num_transfer_tokens(int32_t mask_count, int32_t steps) {
    std::vector<int32_t> num_transfer_tokens(steps);

    int32_t base      = mask_count / steps;
    int32_t remainder = mask_count % steps;

    for (int32_t i = 0; i < steps; i++) {
        num_transfer_tokens[i] = base + (i < remainder ? 1 : 0);
    }

    return num_transfer_tokens;
}

void diffusion_generate(llama_context *          ctx,
                        const llama_token *      input_tokens,
                        llama_token *            output_tokens,
                        int32_t                  n_input,
                        const diffusion_params & params,
                        int32_t &                n_generated) {
    n_generated = 0;
    if (!ctx || !input_tokens || !output_tokens || n_input <= 0 || params.max_length <= n_input ||
        params.steps <= 0 || (uint32_t) params.max_length > llama_n_ctx(ctx) ||
        (params.suppress_mask_token && params.mask_token_id == LLAMA_TOKEN_NULL)) {
        return;
    }
    if (params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED &&
        (params.block_length <= 0 || params.max_length % params.block_length != 0 ||
         params.steps % (params.max_length / params.block_length) != 0)) {
        LOG_ERR("%s: block length/step count do not divide the diffusion buffer\n", __func__);
        return;
    }

    const llama_model * model = llama_get_model(ctx);

    // Initialize with input and pad with mask tokens
    std::copy(input_tokens, input_tokens + n_input, output_tokens);
    std::fill(output_tokens + n_input, output_tokens + params.max_length, params.mask_token_id);

    std::mt19937 rng(params.seed);

    llama_set_causal_attn(ctx, false);

    int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (n_vocab <= 0) {
        return;
    }

    std::vector<llama_token_data> candidates(n_vocab);
    std::vector<llama_token_data> conf_candidates;
    conf_candidates.reserve(params.max_length);
    std::vector<int32_t> mask_positions;
    mask_positions.reserve(params.max_length);

    // Setup sampler chain
    struct llama_sampler * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (params.top_k > 0) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(params.top_k));
    }
    if (params.top_p < 1.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(params.top_p, 1));
    }
    if (params.temperature > 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(params.temperature));
    }
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(params.seed));

    struct llama_sampler * dist_sampler = llama_sampler_init_dist(params.seed);

    llama_batch batch = llama_batch_init(params.max_length, 0, 1);
    batch.n_tokens    = params.max_length;

    // Self-conditioning (DiffusionGemma): feed the previous step's canvas logits back into the graph.
    llama_model *      sc_model = const_cast<llama_model *>(llama_get_model(ctx));
    const int32_t      sc_canvas = params.max_length - n_input;
    std::vector<float> sc_buffer;
    if (params.self_conditioning) {
        sc_buffer.assign((size_t) sc_canvas * n_vocab, 0.0f);
    }

    // Pre-allocate buffers for CFG if needed
    int32_t                  logits_size = n_vocab * params.max_length;
    std::vector<float>       cond_logits_buffer;
    std::vector<llama_token> un_x_buffer;
    if (params.cfg_scale > 0.0f) {
        cond_logits_buffer.resize(logits_size);
        un_x_buffer.resize(params.max_length);
    }

    // For block-based processing
    std::vector<int32_t> num_transfer_tokens;
    int32_t              num_blocks      = 1;
    int32_t              steps_per_block = params.steps;

    if (params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED) {
        num_blocks     = params.max_length / params.block_length;
        steps_per_block = params.steps / num_blocks;
    }

    std::vector<float> confidence(params.max_length);

    int64_t total_sampling_time = 0;
    int64_t total_time          = 0;
    int64_t time_start          = ggml_time_us();

    bool stop_requested = false;
    for (int block_num = 0; block_num < num_blocks && !stop_requested; block_num++) {
        int32_t block_start = (params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED) ? n_input + block_num * params.block_length : 0;
        int32_t block_end   = (params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED) ?
                                  std::min(n_input + (block_num + 1) * params.block_length, params.max_length) :
                                  params.max_length;

        // Count masked tokens in current block for block-based processing
        if (params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED) {
            int32_t block_mask_count = 0;
            for (int i = block_start; i < block_end; i++) {
                if (output_tokens[i] == params.mask_token_id) {
                    block_mask_count++;
                }
            }
            num_transfer_tokens = get_num_transfer_tokens(block_mask_count, steps_per_block);
        }

        for (int32_t step = 0; step < steps_per_block; step++) {
            int32_t global_step = block_num * steps_per_block + step;


            // Setup batch
            for (int32_t i = 0; i < params.max_length; i++) {
                batch.token[i]     = output_tokens[i];
                batch.pos[i]       = i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = 1;
            }

            if (params.self_conditioning) {
                llama_diffusion_set_sc(sc_model, sc_buffer.data(), global_step == 0 ? 0.0f : 1.0f, 1.0f, true);
            }

            float * logits = nullptr;

            if (params.cfg_scale > 0.0f) {
                int ret = llama_decode(ctx, batch);
                if (ret != 0) {
                    LOG_ERR("Failed to generate conditional");
                    break;
                }
                float * cond_logits_ptr = llama_get_logits(ctx);
                std::memcpy(cond_logits_buffer.data(), cond_logits_ptr, logits_size * sizeof(float));

                // Unconditional generation (mask input)
                std::copy(output_tokens, output_tokens + params.max_length, un_x_buffer.begin());
                for (int32_t i = 0; i < n_input; i++) {
                    un_x_buffer[i] = params.mask_token_id;
                }

                for (int32_t i = 0; i < params.max_length; i++) {
                    batch.token[i] = un_x_buffer[i];
                }
                ret = llama_decode(ctx, batch);
                if (ret != 0) {
                    LOG_ERR("Failed to generate unconditional");
                    break;
                }
                float * uncond_logits = llama_get_logits(ctx);

                // Apply CFG
                for (int32_t i = 0; i < logits_size; i++) {
                    cond_logits_buffer[i] =
                        uncond_logits[i] + (params.cfg_scale + 1.0f) * (cond_logits_buffer[i] - uncond_logits[i]);
                }
                logits = cond_logits_buffer.data();
            } else {
                int ret = llama_decode(ctx, batch);
                if (ret != 0) {
                    LOG_ERR("%s: failed to decode at step %d, ret = %d\n", __func__, global_step, ret);
                    break;
                }
                logits = llama_get_logits(ctx);
            }

            if (!logits) {
                LOG_ERR("%s: failed to get logits at step %d\n", __func__, global_step);
                break;
            }

            if (params.self_conditioning) {
                std::memcpy(sc_buffer.data(), logits + (size_t) n_input * n_vocab,
                            (size_t) sc_canvas * n_vocab * sizeof(float));
            }

            auto get_logits_for_pos = [&](int32_t pos) -> const float * {
                if (params.shift_logits) {
                    return pos == 0 ? logits : logits + (pos - 1) * n_vocab;
                }
                return logits + pos * n_vocab;
            };

            int64_t time_start_sampling = ggml_time_us();

            mask_positions.clear();
            for (int32_t i = 0; i < params.max_length; i++) {
                if (output_tokens[i] == params.mask_token_id) {
                    // For block-based, only consider current block
                    if (params.schedule != DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED || (i >= block_start && i < block_end)) {
                        mask_positions.push_back(i);
                    }
                }
            }

            if (mask_positions.empty()) {
                break;
            }

            if (params.add_gumbel_noise && params.temperature > 0.0f) {
                add_gumbel_noise(logits, n_vocab, params.temperature, rng);
            }

            if (params.algorithm == DIFFUSION_ALGORITHM_ORIGIN) {
                int32_t transfer_count = calculate_transfer_count(
                    step, steps_per_block, mask_positions.size(), params.schedule, params.eps, num_transfer_tokens);
                float p_transfer = (float) transfer_count / mask_positions.size();

                for (int32_t pos : mask_positions) {
                    if (std::uniform_real_distribution<float>(0.0f, 1.0f)(rng) < p_transfer) {
                        const float * pos_logits = get_logits_for_pos(pos);
                        for (int32_t token_id = 0; token_id < n_vocab; token_id++) {
                            candidates[token_id].id    = token_id;
                            candidates[token_id].logit = pos_logits[token_id];
                            candidates[token_id].p     = 0.0f;
                        }
                        if (params.suppress_mask_token) {
                            candidates[params.mask_token_id].logit = -INFINITY;
                        }

                        llama_token_data_array cur_p = {
                            candidates.data(),
                            (size_t) n_vocab,
                            -1,
                            false,
                        };

                        llama_sampler_apply(sampler, &cur_p);
                        output_tokens[pos] = cur_p.data[cur_p.selected].id;
                    }
                }
            } else {
                std::vector<std::pair<float, int32_t>> confidences;
                std::vector<llama_token>               sampled_tokens(mask_positions.size());

                for (size_t i = 0; i < mask_positions.size(); i++) {
                    int32_t       pos        = mask_positions[i];
                    const float * pos_logits = get_logits_for_pos(pos);

                    for (int32_t token_id = 0; token_id < n_vocab; token_id++) {
                        candidates[token_id].logit = pos_logits[token_id];
                        candidates[token_id].p     = 0.0f;
                        candidates[token_id].id    = token_id;
                    }
                    if (params.suppress_mask_token) {
                        candidates[params.mask_token_id].logit = -INFINITY;
                    }

                    llama_token_data_array cur_p = {
                        candidates.data(),
                        candidates.size(),
                        -1,
                        false,
                    };

                    llama_sampler_apply(sampler, &cur_p);
                    llama_token sampled_token = cur_p.data[cur_p.selected].id;

                    float conf = calculate_confidence(cur_p, params.algorithm, rng);

                    sampled_tokens[i] = sampled_token;
                    confidences.emplace_back(conf, i);
                }

                int32_t transfer_count = calculate_transfer_count(
                    step, steps_per_block, mask_positions.size(), params.schedule, params.eps, num_transfer_tokens);

                if (transfer_count > 0) {
                    if (params.alg_temp == 0.0f) {
                        std::partial_sort(confidences.begin(),
                                          confidences.begin() + std::min(transfer_count, (int32_t) confidences.size()),
                                          confidences.end(),
                                          [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
                                              if (a.first != b.first) {
                                                  return a.first > b.first;
                                              }
                                              return a.second < b.second;
                                          });

                        for (int32_t i = 0; i < std::min(transfer_count, (int32_t) confidences.size()); i++) {
                            int32_t mask_idx   = confidences[i].second;
                            int32_t pos        = mask_positions[mask_idx];
                            output_tokens[pos] = sampled_tokens[mask_idx];
                        }
                    } else {
                        conf_candidates.clear();
                        for (size_t i = 0; i < confidences.size(); i++) {
                            float conf_logit = confidences[i].first / params.alg_temp;
                            conf_candidates.emplace_back(llama_token_data{ (int32_t) i, conf_logit, 0.0f });
                        }

                        llama_token_data_array conf_array = {
                            conf_candidates.data(),
                            conf_candidates.size(),
                            -1,
                            false,
                        };

                        for (int32_t i = 0; i < std::min(transfer_count, (int32_t) confidences.size()); i++) {
                            llama_sampler_apply(dist_sampler, &conf_array);
                            int32_t selected_idx = conf_array.selected;
                            int32_t mask_idx     = selected_idx;
                            int32_t pos          = mask_positions[mask_idx];
                            output_tokens[pos]   = sampled_tokens[mask_idx];

                            conf_candidates[selected_idx].p = 0.0f;
                            conf_array.selected             = -1;
                        }
                    }
                }
            }

            int64_t time_end_sampling = ggml_time_us();
            total_sampling_time += time_end_sampling - time_start_sampling;

            if (params.step_callback &&
                !params.step_callback(global_step, params.steps, output_tokens,
                                      params.max_length, params.step_callback_user_data)) {
                stop_requested = true;
                break;
            }
        }
    }

    int64_t time_end = ggml_time_us();
    total_time += time_end - time_start;

    LOG_INF("\ntotal time: %0.2fms, time per step: %0.2fms, sampling time per step: %0.2fms\n",
            total_time / 1000.0,
            total_time / 1000.0 / params.steps,
            total_sampling_time / 1000.0 / params.steps);

    if (params.self_conditioning) {
        llama_diffusion_set_sc(sc_model, nullptr, 0.0f, 1.0f, false);
    }

    llama_batch_free(batch);
    llama_sampler_free(sampler);
    llama_sampler_free(dist_sampler);

    n_generated = params.max_length;
}

void diffusion_generate_entropy_bound(llama_context *             ctx,
                                      const llama_token *         input_tokens,
                                      llama_token *               output_tokens,
                                      int32_t                     n_input,
                                      const diffusion_eb_params & params,
                                      int32_t &                   n_generated) {
    n_generated = 0;
    if (!ctx || !input_tokens || !output_tokens || n_input <= 0 || params.max_length <= n_input ||
        (uint32_t) params.max_length > llama_n_ctx(ctx) || params.max_denoising_steps <= 0 ||
        !std::isfinite(params.t_min) || !std::isfinite(params.t_max) ||
        !std::isfinite(params.entropy_bound) || !std::isfinite(params.confidence_threshold) ||
        params.t_min <= 0.0f || params.t_max <= 0.0f || params.entropy_bound < 0.0f ||
        params.confidence_threshold < 0.0f || params.stability_threshold < 0) {
        return;
    }

    llama_model * model   = const_cast<llama_model *>(llama_get_model(ctx));
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int32_t n_canvas = params.max_length - n_input;
    if (n_vocab <= 0 || n_canvas <= 0) {
        return;
    }

    struct diffusion_state_restore {
        llama_model * model;
        ~diffusion_state_restore() {
            llama_diffusion_set_phase(model, 0, 0, 0);
            llama_diffusion_set_device_sc(model, false);
            llama_diffusion_set_sc(model, nullptr, 0.0f, 1.0f, false);
        }
    } state_restore{model};

    const bool dev_sc            = params.gpu_sampling;
    const bool gpu_sample_reduce = params.gpu_sample_reduce && dev_sc;
    llama_diffusion_set_phase(model, 0, 0, 0);
    llama_diffusion_set_device_sc(model, dev_sc);
    llama_set_causal_attn(ctx, false);
    std::copy(input_tokens, input_tokens + n_input, output_tokens);

    std::mt19937                           rng(params.seed);
    std::uniform_real_distribution<float> uni01(0.0f, 1.0f);
    std::uniform_int_distribution<int32_t> vocab_dist(0, n_vocab - 1);

    std::vector<llama_token> current_canvas(n_canvas);
    for (llama_token & token : current_canvas) {
        token = vocab_dist(rng);
    }

    std::vector<float>       sc_buffer((size_t) (dev_sc ? 0 : n_canvas) * n_vocab, 0.0f);
    std::vector<llama_token> argmax_canvas(n_canvas, 0);
    std::vector<llama_token> prev_argmax(n_canvas, -1);
    std::vector<float>       entropy(n_canvas);
    std::vector<llama_token> denoiser(n_canvas);
    std::vector<int32_t>     order(n_canvas);
    std::vector<float>       uniforms(n_canvas);
    std::vector<llama_token> renoise(n_canvas);
    std::vector<char>        accepted(n_canvas);

    const unsigned hw  = std::thread::hardware_concurrency();
    const unsigned nth = std::max(1u, std::min(hw ? hw : 1u, 32u));

    struct batch_owner {
        llama_batch batch;
        ~batch_owner() {
            llama_batch_free(batch);
        }
    } owner{llama_batch_init(params.max_length, 0, 1)};
    llama_batch & batch = owner.batch;

    // Request only canvas logits in either phase; llama_decode packs selected rows contiguously.
    const int32_t logit_off = 0;
    if (params.kv_cache) {
        llama_diffusion_set_sc(model, nullptr, 0.0f, 1.0f, false);

        const int32_t chunk_size = std::max(1, (int32_t) llama_n_ubatch(ctx));
        for (int32_t offset = 0; offset < n_input; offset += chunk_size) {
            const int32_t n_chunk = std::min(chunk_size, n_input - offset);
            llama_diffusion_set_phase(model, 1, n_input, offset);
            batch.n_tokens = n_chunk;
            for (int32_t i = 0; i < n_chunk; ++i) {
                batch.token[i]     = input_tokens[offset + i];
                batch.pos[i]       = offset + i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = i == n_chunk - 1;
            }
            if (llama_decode(ctx, batch) != 0) {
                LOG_ERR("%s: prefill chunk [%d,%d) failed\n", __func__, offset, offset + n_chunk);
                return;
            }
        }
    }

    float prev_temp_inv = 1.0f;
    int32_t held        = 0;
    bool finished       = false;
    bool decoded_any    = false;
    bool device_sample_ok = gpu_sample_reduce;

    for (int32_t cur_step = params.max_denoising_steps; cur_step >= 1 && !finished; --cur_step) {
        const int32_t step_idx = params.max_denoising_steps - cur_step;
        const float t = params.t_min +
                        (params.t_max - params.t_min) *
                            ((float) cur_step / (float) params.max_denoising_steps);
        const float temp_inv = 1.0f / t;

        if (params.kv_cache) {
            llama_diffusion_set_phase(model, 2, n_input, 0);
            batch.n_tokens = n_canvas;
            for (int32_t i = 0; i < n_canvas; ++i) {
                batch.token[i]     = current_canvas[i];
                batch.pos[i]       = n_input + i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = 1;
            }
        } else {
            batch.n_tokens = params.max_length;
            for (int32_t i = 0; i < params.max_length; ++i) {
                batch.token[i]     = i < n_input ? input_tokens[i] : current_canvas[i - n_input];
                batch.pos[i]       = i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = i >= n_input;
            }
        }

        llama_diffusion_set_sc(model, dev_sc ? nullptr : sc_buffer.data(),
                               step_idx == 0 ? 0.0f : 1.0f, prev_temp_inv, true);
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: failed to decode at step %d\n", __func__, step_idx);
            break;
        }

        const bool gpu_reduce = dev_sc && device_sample_ok;
        const float * logits  = gpu_reduce ? nullptr : llama_get_logits(ctx);
        if (!gpu_reduce && !logits) {
            LOG_ERR("%s: failed to get logits at step %d\n", __func__, step_idx);
            break;
        }
        if (gpu_reduce) {
            llama_synchronize(ctx);
        }

        for (int32_t pos = 0; pos < n_canvas; ++pos) {
            uniforms[pos] = uni01(rng);
            renoise[pos]  = vocab_dist(rng);
        }

        auto host_worker = [&](int32_t p0, int32_t p1) {
            for (int32_t pos = p0; pos < p1; ++pos) {
                const float * row = logits + (size_t) (logit_off + pos) * n_vocab;
                float max_logit = -INFINITY;
                int32_t argmax  = 0;
                for (int32_t v = 0; v < n_vocab; ++v) {
                    const float z = row[v] * temp_inv;
                    if (z > max_logit) {
                        max_logit = z;
                        argmax    = v;
                    }
                }

                float normalizer = 0.0f;
                for (int32_t v = 0; v < n_vocab; ++v) {
                    normalizer += expf(row[v] * temp_inv - max_logit);
                }

                const float target = uniforms[pos] * normalizer;
                float cumulative = 0.0f;
                float h          = 0.0f;
                int32_t sampled  = n_vocab - 1;
                bool picked      = false;
                for (int32_t v = 0; v < n_vocab; ++v) {
                    const float e = expf(row[v] * temp_inv - max_logit);
                    const float p = e / normalizer;
                    if (p > 0.0f) {
                        h -= p * logf(p);
                    }
                    cumulative += e;
                    if (!picked && cumulative >= target) {
                        sampled = v;
                        picked  = true;
                    }
                }

                entropy[pos]       = h;
                argmax_canvas[pos] = argmax;
                denoiser[pos]      = sampled;
                if (!dev_sc) {
                    std::memcpy(sc_buffer.data() + (size_t) pos * n_vocab, row,
                                (size_t) n_vocab * sizeof(float));
                }
            }
        };

        auto run_host_workers = [&]() {
            std::vector<std::thread> pool;
            pool.reserve(nth);
            const int32_t chunk = (n_canvas + (int32_t) nth - 1) / (int32_t) nth;
            for (unsigned ti = 0; ti < nth; ++ti) {
                const int32_t p0 = (int32_t) ti * chunk;
                const int32_t p1 = std::min(p0 + chunk, n_canvas);
                if (p0 < p1) {
                    pool.emplace_back(host_worker, p0, p1);
                }
            }
            for (std::thread & thread : pool) {
                thread.join();
            }
        };

        if (gpu_reduce) {
            if (!llama_diffusion_device_sample(model, uniforms.data(), argmax_canvas.data(), entropy.data(),
                                               denoiser.data(), n_canvas, temp_inv)) {
                LOG_WRN("%s: on-device sampling unsupported on this backend; using host sampling\n", __func__);
                device_sample_ok = false;
                logits = llama_get_logits(ctx);
                if (!logits) {
                    LOG_ERR("%s: failed to get logits for host sampling at step %d\n", __func__, step_idx);
                    break;
                }
                run_host_workers();
            }
        } else {
            run_host_workers();
        }

        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
            return entropy[a] < entropy[b];
        });
        std::fill(accepted.begin(), accepted.end(), 0);
        double cumulative_entropy = 0.0;
        for (int32_t rank = 0; rank < n_canvas; ++rank) {
            const int32_t pos = order[rank];
            cumulative_entropy += entropy[pos];
            if (cumulative_entropy - entropy[pos] <= params.entropy_bound) {
                accepted[pos] = 1;
            }
        }

        float entropy_sum = 0.0f;
        for (int32_t pos = 0; pos < n_canvas; ++pos) {
            current_canvas[pos]          = accepted[pos] ? denoiser[pos] : renoise[pos];
            output_tokens[n_input + pos] = argmax_canvas[pos];
            entropy_sum += entropy[pos];
        }
        decoded_any = true;

        held = prev_argmax == argmax_canvas ? held + 1 : 0;
        const bool confident = entropy_sum / (float) n_canvas < params.confidence_threshold;
        finished      = held >= params.stability_threshold && confident;
        prev_argmax   = argmax_canvas;
        prev_temp_inv = temp_inv;

        if (params.step_callback &&
            !params.step_callback(step_idx, params.max_denoising_steps, output_tokens,
                                  params.max_length, params.step_callback_user_data)) {
            break;
        }
    }

    if (decoded_any) {
        n_generated = params.max_length;
    }
}
