#include "cli-diffusion.h"
#include "cli-ui.h"
#include "chat.h"
#include "common.h"
#include "diffusion.h"
#include "ggml-backend.h"
#include "llama.h"
#include "log.h"
#include "unicode.h"

#include <limits.h>
#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
#        define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#    endif
#else
#    include <sys/ioctl.h>
#    include <unistd.h>
#endif

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

class visual_terminal {
public:
    explicit visual_terminal(bool requested) {
        if (!requested) {
            return;
        }
#if defined(_WIN32)
        handle_ = GetStdHandle(STD_OUTPUT_HANDLE);
        if (handle_ == INVALID_HANDLE_VALUE || !GetConsoleMode(handle_, &mode_)) {
            return;
        }
        restore_mode_ = (mode_ & ENABLE_VIRTUAL_TERMINAL_PROCESSING) == 0;
        if (restore_mode_ && !SetConsoleMode(handle_, mode_ | ENABLE_VIRTUAL_TERMINAL_PROCESSING)) {
            return;
        }
        enabled_ = true;
#else
        enabled_ = isatty(STDOUT_FILENO);
#endif
    }

    ~visual_terminal() {
#if defined(_WIN32)
        if (enabled_ && restore_mode_) {
            SetConsoleMode(handle_, mode_);
        }
#endif
    }

    bool enabled() const {
        return enabled_;
    }

private:
    bool enabled_ = false;
#if defined(_WIN32)
    HANDLE handle_      = INVALID_HANDLE_VALUE;
    DWORD  mode_        = 0;
    bool   restore_mode_ = false;
#endif
};

struct callback_data {
    diffusion_params *  diff_params;
    const llama_vocab * vocab;
    int32_t             n_input;
    bool                show_progress;
    int32_t             visual_interval;
    int32_t             steps_seen;
    int32_t             blocks_seen;
    int32_t             term_rows;
    int32_t             term_cols;
    int32_t             vis_rows;
};

static void get_terminal_size(int32_t & rows, int32_t & cols) {
    rows = 24;
    cols = 80;
#if defined(_WIN32)
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi)) {
        const int r = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
        const int c = csbi.srWindow.Right  - csbi.srWindow.Left + 1;
        if (r > 1) {
            rows = r;
        }
        if (c > 0) {
            cols = c;
        }
    }
#else
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_row > 1) {
        rows = ws.ws_row;
        cols = ws.ws_col > 0 ? ws.ws_col : 80;
    }
#endif
}

static int visual_codepoint_width(uint32_t codepoint) {
    if ((codepoint >= 0x0300 && codepoint <= 0x036f) ||
        (codepoint >= 0x1ab0 && codepoint <= 0x1aff) ||
        (codepoint >= 0x1dc0 && codepoint <= 0x1dff) ||
        (codepoint >= 0x20d0 && codepoint <= 0x20ff) ||
        (codepoint >= 0xfe00 && codepoint <= 0xfe0f) ||
        (codepoint >= 0xfe20 && codepoint <= 0xfe2f) ||
        (codepoint >= 0x1f3fb && codepoint <= 0x1f3ff) ||
        codepoint == 0x200d) {
        return 0;
    }

    if ((codepoint >= 0x1100 && codepoint <= 0x115f) ||
        codepoint == 0x2329 || codepoint == 0x232a ||
        (codepoint >= 0x2e80 && codepoint <= 0xa4cf && codepoint != 0x303f) ||
        (codepoint >= 0xac00 && codepoint <= 0xd7a3) ||
        (codepoint >= 0xf900 && codepoint <= 0xfaff) ||
        (codepoint >= 0xfe10 && codepoint <= 0xfe19) ||
        (codepoint >= 0xfe30 && codepoint <= 0xfe6f) ||
        (codepoint >= 0xff00 && codepoint <= 0xff60) ||
        (codepoint >= 0xffe0 && codepoint <= 0xffe6) ||
        (codepoint >= 0x1f300 && codepoint <= 0x1faff) ||
        (codepoint >= 0x20000 && codepoint <= 0x3fffd)) {
        return 2;
    }

    return 1;
}

static std::vector<std::string> visual_canvas_lines(
        const callback_data * data,
        const llama_token *   tokens,
        int32_t               n_tokens,
        int                   cols,
        int                   max_lines) {
    std::string text;
    text.reserve((size_t) std::max(0, n_tokens - data->n_input) * 4);
    const llama_token mask = llama_vocab_mask(data->vocab);
    for (int32_t i = data->n_input; i < n_tokens; ++i) {
        text += tokens[i] == mask ? "." : common_token_to_piece(data->vocab, tokens[i], false);
    }

    std::vector<std::string> lines(1);
    std::vector<int> widths(1, 0);
    size_t offset = 0;
    while (offset < text.size()) {
        const utf8_parse_result parsed = common_parse_utf8_codepoint(text, offset);
        uint32_t codepoint = '?';
        size_t consumed = 1;
        bool preserve = false;
        if (parsed.status == utf8_parse_result::SUCCESS) {
            codepoint = parsed.codepoint;
            consumed  = parsed.bytes_consumed;
            preserve  = true;
        }

        if (codepoint == '\r') {
            offset += consumed;
            continue;
        }
        if (codepoint == '\n') {
            if ((int) lines.size() >= max_lines) {
                break;
            }
            lines.emplace_back();
            widths.push_back(0);
            offset += consumed;
            continue;
        }
        if (codepoint == '\t') {
            codepoint = ' ';
            preserve = false;
        } else if (codepoint < 0x20 || (codepoint >= 0x7f && codepoint <= 0x9f)) {
            codepoint = '?';
            preserve = false;
        }

        int width = visual_codepoint_width(codepoint);
        if (width > cols) {
            codepoint = '?';
            preserve = false;
            width = 1;
        }
        if (widths.back() + width > cols && !lines.back().empty()) {
            if ((int) lines.size() >= max_lines) {
                break;
            }
            lines.emplace_back();
            widths.push_back(0);
        }

        if (preserve) {
            lines.back().append(text, offset, consumed);
        } else {
            lines.back().push_back((char) codepoint);
        }
        widths.back() += width;
        offset += consumed;
    }

    return lines;
}

static std::string visual_progress_line(const callback_data * data, int32_t step, int32_t total_steps, int cols) {
    total_steps = std::max(1, total_steps);
    const int32_t completed = std::min(total_steps, std::max(0, step + 1));
    const int percent = completed * 100 / total_steps;
    const std::string prefix = "diffusion block " + std::to_string(std::max(1, data->blocks_seen)) +
                               ", step " + std::to_string(completed) + "/" + std::to_string(total_steps);
    const std::string suffix = " " + std::to_string(percent) + "%";
    const int bar_width = std::min(50, cols - (int) prefix.size() - (int) suffix.size() - 3);

    std::string line;
    if (bar_width >= 4) {
        const int filled = completed * bar_width / total_steps;
        line = prefix + " [" + std::string(filled, '=') + std::string(bar_width - filled, ' ') + "]" + suffix;
    } else {
        line = prefix + suffix;
    }
    if ((int) line.size() > cols) {
        line.resize(cols);
    }
    return line;
}

static bool diffusion_step_callback(int32_t             step,
                                    int32_t             total_steps,
                                    const llama_token * tokens,
                                    int32_t             n_tokens,
                                    void *              user_data) {
    callback_data * data = static_cast<callback_data *>(user_data);

    data->steps_seen++;
    if (step == 0) {
        data->blocks_seen++;
    }

    total_steps = std::max(1, total_steps);
    const int32_t completed = std::min(total_steps, std::max(0, step + 1));
    if (!data->diff_params->visual_mode) {
        const int progress_percent = completed * 100 / total_steps;
        const int progress_bars    = completed * 50 / total_steps;
        LOG("\rdiffusion step: %d/%d [%s%s] %d%%",
                completed,
                total_steps,
                std::string(progress_bars, '=').c_str(),
                std::string(50 - progress_bars, ' ').c_str(),
                progress_percent);
        return true;
    }

    const bool final_step = completed == total_steps;
    if (data->visual_interval > 1 && step % data->visual_interval != 0 && !final_step) {
        return true;
    }

    const int max_rows = std::max(1, data->term_rows - 2);
    const int cols     = std::max(1, data->term_cols);
    std::vector<std::string> lines;
    if (data->show_progress && max_rows > 1) {
        lines.push_back(visual_progress_line(data, step, total_steps, cols));
    }
    const int canvas_rows = std::max(1, max_rows - (int) lines.size());
    std::vector<std::string> canvas = visual_canvas_lines(data, tokens, n_tokens, cols, canvas_rows);
    lines.insert(lines.end(), canvas.begin(), canvas.end());

    const int needed_rows = std::min(max_rows, std::max(1, (int) lines.size()));
    const int rows = std::max(data->vis_rows, needed_rows);
    std::string frame = "\033[?2026h";
    if (data->vis_rows == 0) {
        for (int r = 1; r < rows; ++r) {
            frame += "\r\n";
        }
    } else if (rows > data->vis_rows) {
        for (int r = data->vis_rows; r < rows; ++r) {
            frame += "\r\n";
        }
    }
    if (rows > 1) {
        frame += "\033[" + std::to_string(rows - 1) + "A";
    }
    for (int r = 0; r < rows; ++r) {
        frame += "\r";
        if (r < (int) lines.size()) {
            frame += lines[r];
        }
        frame += "\033[K";
        if (r < rows - 1) {
            frame += "\r\n";
        }
    }
    frame += "\033[?2026l";
    data->vis_rows = rows;
    fwrite(frame.data(), 1, frame.size(), stdout);
    fflush(stdout);
    return true;
}

int llama_cli_diffusion(common_params & params) {
    ui::init(params);
    llama_backend_init();
    ggml_backend_load_all();

    llama_model_params model_params = common_model_params_to_llama(params);

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), model_params);
    if (!model) {
        LOG_ERR("error: failed to load model '%s'\n", params.model.path.c_str());
        llama_backend_free();
        return 1;
    }

    if (!llama_model_is_diffusion(model)) {
        LOG_ERR("error: unsupported model for diffusion\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // DiffusionGemma block diffusion exposes diffusion.canvas_length: the prompt is followed by a fixed
    // canvas of that many masked positions, which overrides the generic n_ubatch generation length.
    char    canvas_str[32] = {};
    int64_t canvas_length = 0;
    if (llama_model_meta_val_str(model, "diffusion.canvas_length", canvas_str, sizeof(canvas_str)) >= 0) {
        char * end = nullptr;
        canvas_length = strtoll(canvas_str, &end, 10);
        if (end == canvas_str || *end != '\0' || canvas_length <= 0 || canvas_length > INT_MAX) {
            LOG_ERR("error: invalid diffusion.canvas_length metadata: '%s'\n", canvas_str);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
    }

    // Canvas models self-condition (a full-vocab soft-embedding in the graph). Enable it before context
    // creation so the reserve sizes the compute buffer; the real logits buffer is supplied per step.
    if (canvas_length > 0) {
        llama_diffusion_set_sc(model, nullptr, 0.0f, 1.0f, true);

        const int64_t cl = canvas_length;
        int64_t blocks = params.diffusion.blocks;
        if (params.n_predict > 0) {
            blocks = ((int64_t) params.n_predict + cl - 1) / cl;
            params.diffusion.blocks = (int32_t) blocks;
        }
        if (blocks <= 0 || blocks > INT_MAX) {
            LOG_ERR("error: --diffusion-blocks must be positive\n");
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        const int64_t needed = blocks * cl + 2048;
        if (needed > INT_MAX) {
            LOG_ERR("error: requested diffusion output is too large (%lld context tokens)\n",
                    (long long) needed);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        params.n_ubatch = std::max(params.n_ubatch, (int32_t) needed);
        params.n_batch  = std::max(params.n_batch, params.n_ubatch);
        params.n_ctx    = std::max(params.n_ctx, (int32_t) needed);
        LOG_INF("diffusion: %d blocks, n_ubatch=%d n_batch=%d n_ctx=%d (canvas_length=%d)\n",
                params.diffusion.blocks, params.n_ubatch, params.n_batch, params.n_ctx,
                (int) canvas_length);
    }

    // --fit (auto context/layer fitting) runs inside common_init_from_params, which this runner does not
    // use: it sizes context from -n and the canvas above. Tell the user so --fit is not silently ignored.
    if (params.fit_params) {
        LOG_INF("diffusion: --fit has no effect here; context is sized from -n and the canvas. "
                "Set -ngl / --n-cpu-moe to control device memory.\n");
    }

    llama_context_params ctx_params = common_context_params_to_llama(params);

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        LOG_ERR("error: failed to create context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    llama_set_n_threads(ctx, params.cpuparams.n_threads, params.cpuparams_batch.n_threads);

    const llama_vocab * vocab = llama_model_get_vocab(model);

    auto chat_templates = common_chat_templates_init(model, "");

    llama_token mask_token_id = llama_vocab_mask(vocab);
    if (mask_token_id == LLAMA_TOKEN_NULL) {
        LOG_ERR("error: diffusion model has no mask token\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    visual_terminal terminal(params.diffusion.visual_mode);
    const bool visual_mode = terminal.enabled();

    if (params.n_ubatch <= 0) {
        LOG_ERR("error: n_ubatch must be positive\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // reused across turns; canvas models fill only n_input + canvas_length of it
    std::vector<llama_token> output_tokens(params.n_ubatch);

    struct diffusion_params diff_params;

    char shift_logits_str[8];
    if (llama_model_meta_val_str(model, "diffusion.shift_logits", shift_logits_str, sizeof(shift_logits_str)) >= 0) {
        diff_params.shift_logits = (strcmp(shift_logits_str, "true") == 0);
    } else {
        // canvas block-diffusion is not autoregressive (logit i predicts token i); Dream defaults to shifted
        diff_params.shift_logits = (canvas_length == 0);
    }

    if (canvas_length > 0) {
        // Denoise the whole canvas with the timestep schedule; block scheduling asserts
        // max_length % block_length == 0, which a prompt + canvas layout does not satisfy.
        diff_params.schedule = DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED;
        diff_params.eps      = params.diffusion.eps > 0 ? params.diffusion.eps : 1e-3f;
        // these models put probability mass on the mask token at still-masked positions, so a reveal must
        // sample from the non-mask distribution (unlike Dream/LLaDA, which never emit their mask token).
        diff_params.suppress_mask_token = true;
        // bootstrap the denoise from all-mask: feed each step's canvas logits into the next step
        diff_params.self_conditioning = true;
    } else {
        //Use either eps or block length, but not both
        GGML_ASSERT((params.diffusion.eps == 0) ^ (params.diffusion.block_length == 0));

        if (params.diffusion.eps) {
            diff_params.schedule = DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED;
            diff_params.eps      = params.diffusion.eps;
        } else if (params.diffusion.block_length) {
            diff_params.schedule     = DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED;
            diff_params.block_length = params.diffusion.block_length;
        }
    }

    diff_params.mask_token_id    = mask_token_id;
    diff_params.seed             = params.sampling.seed;
    diff_params.temperature      = params.sampling.temp;
    diff_params.steps            = params.diffusion.steps;
    diff_params.algorithm        = static_cast<diffusion_algorithm>(params.diffusion.algorithm);
    diff_params.top_p            = params.sampling.top_p;
    diff_params.top_k            = params.sampling.top_k;
    diff_params.visual_mode      = visual_mode;
    diff_params.alg_temp         = params.diffusion.alg_temp;
    diff_params.cfg_scale        = params.diffusion.cfg_scale;
    diff_params.add_gumbel_noise = params.diffusion.add_gumbel_noise;

    callback_data cb_data               = { &diff_params, vocab, 0, params.diffusion.visual_progress,
                                            std::max(1, params.diffusion.visual_interval), 0, 0, 24, 80, 0 };
    diff_params.step_callback           = diffusion_step_callback;
    diff_params.step_callback_user_data = &cb_data;

    const char * alg_names[] = {
        "DIFFUSION_ALGORITHM_ORIGIN",
        "DIFFUSION_ALGORITHM_ENTROPY_BASED",
        "DIFFUSION_ALGORITHM_MARGIN_BASED",
        "DIFFUSION_ALGORITHM_RANDOM",
        "DIFFUSION_ALGORITHM_CONFIDENCE_BASED",
    };
    const char * sched_names[] = {
        "DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED",
        "DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED",
    };
    const int alg_index = (int) diff_params.algorithm;
    const int sched_index = (int) diff_params.schedule;
    const char * alg_name = alg_index >= 0 && alg_index < 5 ? alg_names[alg_index] : "UNKNOWN";
    const char * sched_name = sched_index >= 0 && sched_index < 2 ? sched_names[sched_index] : "UNKNOWN";
    LOG_INF("diffusion_params: - %-25s llama_token = %d\n", "mask_token_id", mask_token_id);
    LOG_INF("diffusion_params: - %-25s u32         = %d\n", "steps", diff_params.steps);
    LOG_INF("diffusion_params: - %-25s enum        = %d (%s)\n", "algorithm", alg_index, alg_name);
    LOG_INF("diffusion_params: - %-25s enum        = %d (%s)\n", "schedule", sched_index, sched_name);
    LOG_INF("diffusion_params: - %-25s f32         = %.3f\n", "temperature", diff_params.temperature);
    LOG_INF("diffusion_params: - %-25s f32         = %.6f\n", "eps", diff_params.eps);
    LOG_INF("diffusion_params: - %-25s f32         = %.3f\n", "alg_temp", diff_params.alg_temp);
    LOG_INF("diffusion_params: - %-25s f32         = %.3f\n", "cfg_scale", diff_params.cfg_scale);

    // Entropy-bound decoder: the real DiffusionGemma sampler (random-init canvas, MI-bounded acceptance,
    // renoise, temperature schedule, adaptive stop). Auto-enabled for canvas models; --diffusion-eb forces
    // it on/off. Params come from GGUF metadata (diffusion.eb_*), then reference defaults, then CLI override.
    const bool use_eb = canvas_length > 0 && params.diffusion.eb_mode != 2;

    struct diffusion_eb_params eb_params;
    if (use_eb) {
        auto meta_f = [&](const char * key, float def) -> float {
            char buf[32];
            return llama_model_meta_val_str(model, key, buf, sizeof(buf)) >= 0 ? strtof(buf, nullptr) : def;
        };
        auto meta_i = [&](const char * key, int32_t def) -> int32_t {
            char buf[32];
            return llama_model_meta_val_str(model, key, buf, sizeof(buf)) >= 0 ? (int32_t) strtol(buf, nullptr, 10) : def;
        };
        eb_params.max_denoising_steps  = meta_i("diffusion.eb_max_steps", 48);
        eb_params.t_min                = meta_f("diffusion.eb_t_min", 0.4f);
        eb_params.t_max                = meta_f("diffusion.eb_t_max", 0.8f);
        eb_params.entropy_bound        = meta_f("diffusion.eb_entropy_bound", 0.1f);
        eb_params.stability_threshold  = meta_i("diffusion.eb_stability_threshold", 1);
        eb_params.confidence_threshold = meta_f("diffusion.eb_confidence_threshold", 0.005f);
        if (params.diffusion.eb_t_min         >= 0) { eb_params.t_min                = params.diffusion.eb_t_min; }
        if (params.diffusion.eb_t_max         >= 0) { eb_params.t_max                = params.diffusion.eb_t_max; }
        if (params.diffusion.eb_entropy_bound >= 0) { eb_params.entropy_bound        = params.diffusion.eb_entropy_bound; }
        if (params.diffusion.eb_stability     >= 0) { eb_params.stability_threshold  = params.diffusion.eb_stability; }
        if (params.diffusion.eb_confidence    >= 0) { eb_params.confidence_threshold = params.diffusion.eb_confidence; }
        if (params.diffusion.eb_max_steps     >  0) { eb_params.max_denoising_steps  = params.diffusion.eb_max_steps; }
        eb_params.seed                    = params.sampling.seed;
        eb_params.step_callback           = diffusion_step_callback;
        eb_params.step_callback_user_data = &cb_data;

        // Prefix KV caching and device sampling require a single accelerator because their persistent
        // buffers are not replicated across devices.
        int accelerator_devs = 0;
        int sampling_devs = 0;
        auto count_device = [&](ggml_backend_dev_t dev) {
            const auto type = ggml_backend_dev_type(dev);
            if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
                return;
            }
            ++accelerator_devs;
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
            const char * reg_name = reg ? ggml_backend_reg_name(reg) : nullptr;
            if (reg_name && (strcmp(reg_name, "CUDA") == 0 || strcmp(reg_name, "ROCm") == 0 ||
                             strcmp(reg_name, "MUSA") == 0 ||
                             ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_diffusion_sample") != nullptr)) {
                ++sampling_devs;
            }
        };
        if (params.devices.empty()) {
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                count_device(ggml_backend_dev_get(i));
            }
        } else {
            for (ggml_backend_dev_t dev : params.devices) {
                if (dev == nullptr) {
                    break;
                }
                count_device(dev);
            }
        }
        if (params.diffusion.eb_kv_cache == 1) {
            eb_params.kv_cache = true;
        } else if (params.diffusion.eb_kv_cache == 2) {
            eb_params.kv_cache = false;
        } else {
            eb_params.kv_cache = accelerator_devs <= 1;
            if (accelerator_devs > 1) {
                LOG_INF("diffusion_eb: kv cache auto-off (%d accelerator devices)\n", accelerator_devs);
            }
        }

        const bool can_device_sample = accelerator_devs == 1 && sampling_devs == 1 && params.n_gpu_layers != 0;
        eb_params.gpu_sampling = params.diffusion.eb_gpu_sampling != 2 && can_device_sample;
        if (params.diffusion.eb_gpu_sampling != 2 && !can_device_sample) {
            LOG_INF("diffusion_eb: device sampling off (found %d compatible CUDA/ROCm/MUSA devices)\n",
                    sampling_devs);
        }

        eb_params.gpu_sample_reduce = params.diffusion.eb_gpu_sample_reduce != 2 && eb_params.gpu_sampling;
        if (params.diffusion.eb_gpu_sample_reduce != 2 && !eb_params.gpu_sampling) {
            LOG_INF("diffusion_eb: device sample reduction off (device sampling unavailable)\n");
        }

        LOG_INF("diffusion_eb: max_steps=%d t=[%.3f,%.3f] entropy_bound=%.4f stability=%d confidence=%.4f kv_cache=%s gpu_sampling=%s sample_reduce=%s\n",
                eb_params.max_denoising_steps, eb_params.t_min, eb_params.t_max, eb_params.entropy_bound,
                eb_params.stability_threshold, eb_params.confidence_threshold, eb_params.kv_cache ? "on" : "off",
                eb_params.gpu_sampling ? "on" : "off", eb_params.gpu_sample_reduce ? "on" : "off");
    }

    // Trim a denoised canvas: cut at the first end-of-generation token, or (checkpoints often emit no stop
    // token) at the onset of a repetition loop (a token recurring at stride 1-2 for >= 6 steps).
    auto trim_canvas = [&](const llama_token * canvas, size_t n) -> size_t {
        size_t cut = n;
        for (size_t i = 0; i < n; i++) {
            if (llama_vocab_is_eog(vocab, canvas[i])) {
                cut = i;
                break;
            }
        }
        for (size_t i = 0; i + 1 < cut; i++) {
            bool loop = false;
            for (size_t stride = 1; stride <= 2 && !loop; stride++) {
                size_t reps = 0;
                for (size_t j = i; j + stride < n && canvas[j] == canvas[j + stride]; j += stride) {
                    reps++;
                }
                loop = reps >= 6;
            }
            if (loop) {
                cut = i;
                break;
            }
        }
        return cut;
    };

    // Generate one response for a chat-formatted prompt. Canvas models denoise a fixed canvas_length block
    // per pass; with --diffusion-blocks > 1 we run block-autoregressively, committing each block to the
    // prefix and denoising the next until an end token, a repetition loop, the block budget, or the ubatch
    // limit (the whole [prefix | canvas] must fit in one non-causal ubatch). Returns the trimmed text.
    auto run_turn = [&](const std::string & formatted_prompt) -> std::string {
        std::vector<llama_token> prefix = common_tokenize(vocab, formatted_prompt,
                                                          /*add special*/ true, /*parse special*/ true);
        if (prefix.size() > INT_MAX || prefix.size() >= llama_n_ctx(ctx)) {
            LOG_ERR("error: input too long (%zu tokens), max context is %d\n",
                    prefix.size(), (int) llama_n_ctx(ctx));
            return "";
        }
        const int32_t n_input = (int32_t) prefix.size();

        // non-canvas models (Dream/LLaDA): single fixed-length pass, blocks ignored
        if (canvas_length <= 0) {
            diff_params.max_length = std::min(params.n_ubatch, (int32_t) llama_n_ctx(ctx));
            if (diff_params.max_length <= n_input) {
                LOG_ERR("error: no output space (%d input tokens, %d-token diffusion buffer)\n",
                        n_input, diff_params.max_length);
                return "";
            }
            cb_data.n_input = n_input;
            int32_t n_generated = 0;
            diffusion_generate(ctx, prefix.data(), output_tokens.data(), n_input, diff_params, n_generated);
            if (n_generated <= n_input) {
                LOG_INF("Error: diffusion generation failed\n");
                return "";
            }
            return common_detokenize(vocab,
                std::vector<llama_token>(output_tokens.begin() + n_input, output_tokens.begin() + n_generated), false);
        }

        const int32_t max_ub   = std::min((int32_t) params.n_ubatch, (int32_t) llama_n_ctx(ctx));
        const int     n_blocks = std::max(1, params.diffusion.blocks);
        std::vector<llama_token> response;

        for (int b = 0; b < n_blocks; b++) {
            if (prefix.size() > (size_t) INT_MAX) {
                LOG_ERR("error: accumulated diffusion prefix is too large\n");
                break;
            }
            const int32_t prefix_len = (int32_t) prefix.size();
            const int64_t needed = (int64_t) prefix_len + canvas_length;
            if (needed > max_ub) {
                if (b == 0) {
                    LOG_ERR("error: this diffusion model needs the whole [prompt | canvas] in one ubatch; "
                            "set -ub and -c >= n_input + canvas_length = %d + %d = %lld\n",
                            prefix_len, (int) canvas_length, (long long) needed);
                    return "";
                }
                break;
            }
            const int32_t max_length = (int32_t) needed;

            diff_params.max_length = max_length;
            eb_params.max_length   = max_length;
            cb_data.n_input        = prefix_len;

            int32_t n_generated = 0;
            if (use_eb) {
                diffusion_generate_entropy_bound(ctx, prefix.data(), output_tokens.data(), prefix_len, eb_params, n_generated);
            } else {
                diffusion_generate(ctx, prefix.data(), output_tokens.data(), prefix_len, diff_params, n_generated);
            }
            if (n_generated <= prefix_len) {
                if (b == 0) {
                    LOG_INF("Error: diffusion generation failed\n");
                    return "";
                }
                break;
            }

            const llama_token * canvas = output_tokens.data() + prefix_len;
            const size_t        cut    = trim_canvas(canvas, (size_t) canvas_length);
            response.insert(response.end(), canvas, canvas + cut);
            if (cut < (size_t) canvas_length) {
                break;  // end token or repetition loop: answer complete
            }
            prefix.insert(prefix.end(), canvas, canvas + cut);  // commit the block, denoise the next
        }

        return common_detokenize(vocab, response, false);
    };

    auto make_msg = [](const std::string & role, const std::string & content) {
        common_chat_msg m;
        m.role    = role;
        m.content = content;
        return m;
    };

    auto apply_template = [&](const std::vector<common_chat_msg> & messages) -> std::string {
        common_chat_templates_inputs inputs;
        inputs.messages              = messages;
        inputs.add_generation_prompt = true;
        return common_chat_templates_apply(chat_templates.get(), inputs).prompt;
    };

    // A live frame grows only as tall as its wrapped canvas needs. The callback repaints that fixed region,
    // then this wrapper clears it before printing the final reply into normal scrollback.
    auto run_turn_reply = [&](const std::string & formatted_prompt) -> std::string {
        cb_data.steps_seen  = 0;
        cb_data.blocks_seen = 0;
        if (visual_mode) {
            get_terminal_size(cb_data.term_rows, cb_data.term_cols);
            cb_data.vis_rows = 0;
            common_log_flush(common_log_main());
            fwrite("\033[?25l", 1, 6, stdout);
            fflush(stdout);
        }
        const int64_t t0 = ggml_time_us();
        std::string response = run_turn(formatted_prompt);
        const int64_t turn_us = ggml_time_us() - t0;
        if (visual_mode) {
            std::string fin;
            if (cb_data.vis_rows > 1) {
                fin += "\033[" + std::to_string(cb_data.vis_rows - 1) + "A";
            }
            fin += "\r\033[J\033[?25h";
            fwrite(fin.data(), 1, fin.size(), stdout);
            fflush(stdout);
        }
        {
            ui::assistant_turn turn;
            turn.push(ui::ASSISTANT_DISPLAY_MODE_CONTENT, "\n" + response);
        }
        if (params.show_timings && cb_data.steps_seen > 0 && turn_us > 0) {
            const double total_ms = turn_us / 1000.0;
            const double per_step = total_ms / cb_data.steps_seen;
            const int64_t step_tokens = canvas_length > 0 ?
                canvas_length : std::max<int64_t>(1, diff_params.max_length - cb_data.n_input);
            const int64_t generated_tokens = step_tokens * std::max(1, cb_data.blocks_seen);
            const double prompt_tps = step_tokens * 1000.0 / per_step;
            const double generation_tps = generated_tokens * 1000.0 / total_ms;
            ui::show_info(string_format(
                "\n[ Prompt: %.1f t/s | Generation: %.1f t/s ]",
                prompt_tps,
                generation_tps));
        }
        return response;
    };

    if (params.conversation_mode != COMMON_CONVERSATION_MODE_DISABLED) {
        if (!params.enable_chat_template) {
            LOG_ERR("error: conversation mode requires a chat template\n");
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }

        // Multi-turn: each turn re-applies the template to the full history and denoises a fresh canvas
        // (no state is kept across turns). History is bounded by the ubatch cap (run_turn reports overflow).
        std::vector<common_chat_msg> messages;
        if (!params.system_prompt.empty()) {
            messages.push_back(make_msg("system", params.system_prompt));
        }

        LOG("conversation mode: /help for commands, /clear to reset, /exit to quit\n");

        std::string pending = params.prompt;  // optional first user turn supplied via -p
        while (true) {
            std::string user;
            {
                ui::user_turn turn;
                if (!pending.empty()) {
                    user = pending;
                    pending.clear();
                    turn.echo(user);
                } else {
                    user = turn.read_input(params.multiline_input);
                }
            }
            if (user == "/exit" || user == "/quit") {
                break;
            }
            if (user == "/help" || user == "/?") {
                LOG("commands:\n"
                    "  /help, /?      show this message\n"
                    "  /clear         clear the conversation history (keeps the system prompt)\n"
                    "  /exit, /quit   end the session\n");
                continue;
            }
            if (user == "/clear") {
                messages.clear();
                if (!params.system_prompt.empty()) {
                    messages.push_back(make_msg("system", params.system_prompt));
                }
                LOG("conversation history cleared\n");
                continue;
            }
            if (user.empty()) {
                continue;
            }

            messages.push_back(make_msg("user", user));
            const std::string response = run_turn_reply(apply_template(messages));
            messages.push_back(make_msg("assistant", response));
            if (params.single_turn) {
                break;
            }
        }
    } else {
        if (params.display_prompt) {
            ui::user_turn turn;
            turn.echo(params.prompt);
        }
        std::string formatted = params.prompt;
        if (params.enable_chat_template) {
            std::vector<common_chat_msg> messages;
            if (!params.system_prompt.empty()) {
                messages.push_back(make_msg("system", params.system_prompt));
            }
            messages.push_back(make_msg("user", params.prompt));
            formatted = apply_template(messages);
        }
        run_turn_reply(formatted);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
