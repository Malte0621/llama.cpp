#include "ggml-alloc.h"
#include "llama-impl.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-model-loader.h"
#include "llama-ext.h"
#include "llama.h"
#include "llama-parquet.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cinttypes>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <fstream>
#include <future>
#include <mutex>
#include <limits>
#include <memory>
#include <regex>
#include <thread>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <type_traits>


// result of parsing --tensor-type option
// (changes to this struct must be reflected in tools/quantize/quantize.cpp)
struct tensor_type_option {
    std::string name;
    ggml_type type = GGML_TYPE_COUNT;
};

// tensor categorization - used to avoid repeated string matching in quantization logic.
// this is different from LLM_TN - we want broad categories, not specific tensor names per arch.
enum class tensor_category {
    TOKEN_EMBD,
    ATTENTION_Q,
    ATTENTION_V,
    ATTENTION_K,
    ATTENTION_QKV,
    ATTENTION_KV_B,
    ATTENTION_OUTPUT,
    FFN_UP,
    FFN_GATE,
    FFN_DOWN,
    OUTPUT,
    OTHER
};

static void zeros(std::ofstream & file, size_t n) {
    char zero = 0;
    for (size_t i = 0; i < n; ++i) {
        file.write(&zero, 1);
    }
}

static std::string remap_layer(const std::string & orig_name, const std::vector<int> & prune, std::map<int, std::string> & mapped, int & next_id) {
    if (prune.empty()) {
        return orig_name;
    }

    static const std::regex pattern(R"(blk\.(\d+)\.)");
    if (std::smatch match; std::regex_search(orig_name, match, pattern)) {
        const int blk = std::stoi(match[1]);
        std::string new_name = orig_name;

        if (mapped.count(blk)) {
            // Already mapped, do nothing
        } else if (std::find(prune.begin(), prune.end(), blk) != prune.end()) {
            mapped[blk] = "";
        } else if (blk < prune.front()) {
            mapped[blk] = std::to_string(blk);
            next_id = blk + 1;
        } else {
            mapped[blk] = std::to_string(next_id);
            ++next_id;
        }

        return mapped[blk].empty() ? mapped[blk] : new_name.replace(match.position(1), match.length(1), mapped[blk]);
    }

    return orig_name;
}

static std::string remap_imatrix(const std::string & orig_name, const std::map<int, std::string> & mapped) {
    if (mapped.empty()) {
        return orig_name;
    }

    static const std::regex pattern(R"(blk\.(\d+)\.)");
    if (std::smatch match; std::regex_search(orig_name, match, pattern)) {
        const std::string blk(match[1]);
        std::string new_name = orig_name;

        for (const auto & p : mapped) {
            if (p.second == blk) {
                return new_name.replace(match.position(1), match.length(1), std::to_string(p.first));
            }
        }
        GGML_ABORT("\n%s: imatrix mapping error for %s\n", __func__, orig_name.c_str());
    }

    return orig_name;
}

//
// helper functions for tensor name matching
//

static bool tensor_name_match_token_embd(const char * tensor_name) {
    return std::strcmp(tensor_name, "token_embd.weight") == 0 ||
           std::strcmp(tensor_name, "per_layer_token_embd.weight") == 0;
}

static bool tensor_name_match_output_weight(const char * tensor_name) {
    return std::strcmp(tensor_name, "output.weight") == 0;
}

//
// tensor categorization for quantization
//
// (this is different from LLM_TN - we want broad categories, not specific tensor names per arch)
//

static tensor_category tensor_get_category(const std::string & tensor_name) {
    if (tensor_name_match_output_weight(tensor_name.c_str())) {
        return tensor_category::OUTPUT;
    }
    if (tensor_name_match_token_embd(tensor_name.c_str())) {
        return tensor_category::TOKEN_EMBD;
    }
    if (tensor_name.find("attn_qkv.weight") != std::string::npos) {
        return tensor_category::ATTENTION_QKV;
    }
    if (tensor_name.find("attn_kv_b.weight") != std::string::npos) {
        return tensor_category::ATTENTION_KV_B;
    }
    if (tensor_name.find("attn_v.weight") != std::string::npos) {
        return tensor_category::ATTENTION_V;
    }
    if (tensor_name.find("attn_k.weight") != std::string::npos) {
        return tensor_category::ATTENTION_K;
    }
    if (tensor_name.find("attn_q.weight") != std::string::npos) {
        return tensor_category::ATTENTION_Q;
    }
    if (tensor_name.find("attn_output.weight") != std::string::npos) {
        return tensor_category::ATTENTION_OUTPUT;
    }
    if (tensor_name.find("ffn_up") != std::string::npos) {
        return tensor_category::FFN_UP;
    }
    if (tensor_name.find("ffn_gate") != std::string::npos) {
        return tensor_category::FFN_GATE;
    }
    if (tensor_name.find("ffn_down") != std::string::npos) {
        return tensor_category::FFN_DOWN;
    }
    return tensor_category::OTHER;
}

// check if category is for attention-v-like tensors (more sensitive to quantization)
static bool category_is_attn_v(tensor_category cat) {
    return cat == tensor_category::ATTENTION_V     ||
           cat == tensor_category::ATTENTION_QKV   ||
           cat == tensor_category::ATTENTION_KV_B;
}

//
// quantization state
//

struct quantize_state_impl {
    const llama_model                 & model;
    const llama_model_quantize_params * params;

    int n_attention_wv = 0;
    int n_ffn_down     = 0;
    int n_ffn_gate     = 0;
    int n_ffn_up       = 0;
    int i_attention_wv = 0;
    int i_ffn_down     = 0;
    int i_ffn_gate     = 0;
    int i_ffn_up       = 0;

    int n_fallback    = 0;

    bool has_imatrix = false;

    // used to figure out if a model has tied embeddings (tok_embd shares weights with output)
    bool has_tied_embeddings = true; // assume tied until we see output.weight

    // tensor type override patterns (compiled once, used twice)
    std::vector<std::pair<std::regex, ggml_type>> tensor_type_patterns;

    quantize_state_impl(const llama_model & model, const llama_model_quantize_params * params):
        model(model), params(params)
    {
        // compile regex patterns once - they are expensive
        if (params->tt_overrides) {
            for (const auto * p = params->tt_overrides; p->pattern != nullptr; p++) {
                tensor_type_patterns.emplace_back(std::regex(p->pattern), p->type);
            }
        }
    }
};

// per-tensor metadata, computed in the preliminary loop and used in the main loop
struct tensor_metadata {
    std::string     name;
    ggml_type       target_type;
    tensor_category category;
    std::string     remapped_imatrix_name;
    bool            allows_quantization;
    bool            requires_imatrix;
};

//
// dequantization
//

static void llama_tensor_dequantize_to_f32(
    ggml_tensor * tensor, float * output, std::vector<std::thread> & workers,
    const size_t nelements, const int nthread
) {
    const ggml_type_traits * qtype = ggml_get_type_traits(tensor->type);
    if (ggml_is_quantized(tensor->type)) {
        if (qtype->to_float == NULL) {
            throw std::runtime_error(format("type %s unsupported for integer quantization: no dequantization available", ggml_type_name(tensor->type)));
        }
    } else if (tensor->type != GGML_TYPE_F16 &&
               tensor->type != GGML_TYPE_BF16) {
        throw std::runtime_error(format("cannot dequantize/convert tensor type %s", ggml_type_name(tensor->type)));
    }

    if (nthread < 2) {
        if (tensor->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((ggml_fp16_t *)tensor->data, output, nelements);
        } else if (tensor->type == GGML_TYPE_BF16) {
            ggml_bf16_to_fp32_row((ggml_bf16_t *)tensor->data, output, nelements);
        } else if (ggml_is_quantized(tensor->type)) {
            qtype->to_float(tensor->data, output, nelements);
        } else {
            GGML_ABORT("fatal error"); // unreachable
        }
        return;
    }

    size_t block_size;
    if (tensor->type == GGML_TYPE_F16 ||
        tensor->type == GGML_TYPE_BF16) {
        block_size = 1;
    } else {
        block_size = (size_t)ggml_blck_size(tensor->type);
    }

    size_t block_size_bytes = ggml_type_size(tensor->type);

    GGML_ASSERT(nelements % block_size == 0);
    size_t nblocks = nelements / block_size;
    size_t blocks_per_thread = nblocks / nthread;
    size_t spare_blocks = nblocks - (blocks_per_thread * nthread); // if blocks aren't divisible by thread count

    size_t in_buff_offs = 0;
    size_t out_buff_offs = 0;

    for (int tnum = 0; tnum < nthread; tnum++) {
        size_t thr_blocks = blocks_per_thread + (tnum == nthread - 1 ? spare_blocks : 0); // num blocks for this thread
        size_t thr_elems = thr_blocks * block_size; // number of elements for this thread
        size_t thr_block_bytes = thr_blocks * block_size_bytes; // number of input bytes for this thread

        auto compute = [qtype] (ggml_type typ, uint8_t * inbuf, float * outbuf, int nels) {
            if (typ == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row((ggml_fp16_t *)inbuf, outbuf, nels);
            } else if (typ == GGML_TYPE_BF16) {
                ggml_bf16_to_fp32_row((ggml_bf16_t *)inbuf, outbuf, nels);
            } else {
                qtype->to_float(inbuf, outbuf, nels);
            }
        };
        workers.emplace_back(compute, tensor->type, (uint8_t *) tensor->data + in_buff_offs, output + out_buff_offs, thr_elems);
        in_buff_offs += thr_block_bytes;
        out_buff_offs += thr_elems;
    }
    for (auto & w : workers) { w.join(); }
    workers.clear();
}

static void llama_tensor_dequantize_impl(
    ggml_tensor * tensor, std::vector<no_init<float>> & output, std::vector<std::thread> & workers,
    const size_t nelements, const int nthread
) {
    if (output.size() < nelements) {
        output.resize(nelements);
    }
    llama_tensor_dequantize_to_f32(
            tensor, reinterpret_cast<float *>(output.data()), workers, nelements, nthread);
}

//
// do we allow this tensor to be quantized?
//

static bool tensor_allows_quantization(const llama_model_quantize_params * params, llm_arch arch, const ggml_tensor * tensor) {
    // trivial checks first -- no string ops needed
    if (params->only_copy)       return false;

    // quantize only 2D and 3D tensors (experts)
    if (ggml_n_dims(tensor) < 2) return false;

    const std::string name = ggml_get_name(tensor);

    // This used to be a regex, but <regex> has an extreme cost to compile times.
    bool quantize = name.rfind("weight") == name.size() - 6; // ends with 'weight'?

    // do not quantize norm tensors
    quantize &= name.find("_norm.weight") == std::string::npos;

    quantize &= params->quantize_output_tensor || name != "output.weight";

    // do not quantize expert gating tensors
    // NOTE: can't use LLM_TN here because the layer number is not known
    quantize &= name.find("ffn_gate_inp.weight") == std::string::npos;

    // do not quantize the i32 token-id -> expert-id routing table (DeepSeek-V4)
    quantize &= name.find("ffn_gate_tid2eid.weight") == std::string::npos;

    // these are very small (e.g. 4x4)
    quantize &= name.find("altup")  == std::string::npos;
    quantize &= name.find("laurel") == std::string::npos;

    // these are not too big so keep them as it is
    quantize &= name.find("per_layer_model_proj") == std::string::npos;

    // do not quantize positional embeddings and token types (BERT)
    quantize &= name != LLM_TN(arch)(LLM_TENSOR_POS_EMBD,    "weight");
    quantize &= name != LLM_TN(arch)(LLM_TENSOR_TOKEN_TYPES, "weight");

    // do not quantize Mamba/Kimi's small conv1d weights
    // NOTE: can't use LLM_TN here because the layer number is not known
    quantize &= name.find("ssm_conv1d") == std::string::npos;
    quantize &= name.find("shortconv.conv.weight") == std::string::npos;

    // do not quantize MiniMax's indexer projection weights, they are tiny
    quantize &= name.find("indexer.k_proj.weight") == std::string::npos;
    quantize &= name.find("indexer.q_proj.weight") == std::string::npos;

    // do not quantize RWKV's small yet 2D weights
    quantize &= name.find("time_mix_first.weight") == std::string::npos;
    quantize &= name.find("time_mix_w0.weight") == std::string::npos;
    quantize &= name.find("time_mix_w1.weight") == std::string::npos;
    quantize &= name.find("time_mix_w2.weight") == std::string::npos;
    quantize &= name.find("time_mix_v0.weight") == std::string::npos;
    quantize &= name.find("time_mix_v1.weight") == std::string::npos;
    quantize &= name.find("time_mix_v2.weight") == std::string::npos;
    quantize &= name.find("time_mix_a0.weight") == std::string::npos;
    quantize &= name.find("time_mix_a1.weight") == std::string::npos;
    quantize &= name.find("time_mix_a2.weight") == std::string::npos;
    quantize &= name.find("time_mix_g1.weight") == std::string::npos;
    quantize &= name.find("time_mix_g2.weight") == std::string::npos;
    quantize &= name.find("time_mix_decay_w1.weight") == std::string::npos;
    quantize &= name.find("time_mix_decay_w2.weight") == std::string::npos;
    quantize &= name.find("time_mix_lerp_fused.weight") == std::string::npos;

    // do not quantize relative position bias (T5)
    quantize &= name.find("attn_rel_b.weight") == std::string::npos;

    // do not quantize specific multimodal tensors
    quantize &= name.find(".position_embd") == std::string::npos;
    quantize &= name.find("sam.pos_embd")   == std::string::npos;
    quantize &= name.find("sam.neck.")      == std::string::npos;
    quantize &= name.find("sam.net_")       == std::string::npos;
    quantize &= name.find(".rel_pos")       == std::string::npos;
    quantize &= name.find(".patch_embd")    == std::string::npos;
    quantize &= name.find(".patch_merger")  == std::string::npos;

    // audio codebook
    quantize &= name.find("a.rvq.codebook")  == std::string::npos;
    quantize &= name.find("mm.a.code_embd")  == std::string::npos;

    return quantize;
}

//
// tensor type selection
//

// incompatible tensor shapes are handled here - fallback to a compatible type
static ggml_type tensor_type_fallback(quantize_state_impl & qs, const ggml_tensor * t, const ggml_type target_type) {
    ggml_type return_type = target_type;

    const int64_t ncols = t->ne[0];
    const int64_t qk_k = ggml_blck_size(target_type);

    if (ncols % qk_k != 0) { // this tensor's shape is incompatible with this quant
        LLAMA_LOG_WARN("warning: %-36s - ncols %6" PRId64 " not divisible by %3" PRId64 " (required for type %7s) ",
                        t->name, ncols, qk_k, ggml_type_name(target_type));
        ++qs.n_fallback;

        switch (target_type) {
            // types on the left: block size 256
            case GGML_TYPE_IQ1_S:
            case GGML_TYPE_IQ1_M:
            case GGML_TYPE_IQ2_XXS:
            case GGML_TYPE_IQ2_XS:
            case GGML_TYPE_IQ2_S:
            case GGML_TYPE_IQ3_XXS:
            case GGML_TYPE_IQ3_S:   // types on the right: block size 32
            case GGML_TYPE_IQ4_XS:  return_type = GGML_TYPE_IQ4_NL; break;
            case GGML_TYPE_Q2_0:
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_TQ1_0:
            case GGML_TYPE_TQ2_0:   return_type = GGML_TYPE_Q4_0;   break;
            case GGML_TYPE_Q4_K:    return_type = GGML_TYPE_Q5_0;   break;
            case GGML_TYPE_Q5_K:    return_type = GGML_TYPE_Q5_1;   break;
            case GGML_TYPE_Q6_K:    return_type = GGML_TYPE_Q8_0;   break;
            default:
                throw std::runtime_error(format("no tensor type fallback is defined for type %s",
                                                ggml_type_name(target_type)));
        }
        if (ncols % ggml_blck_size(return_type) != 0) {
            //
            // the fallback return type is still not compatible for this tensor!
            //
            // most likely, this tensor's first dimension is not divisible by 32.
            // this is very rare. we can either abort the quantization, or
            // fallback to F16 / F32.
            //
            LLAMA_LOG_WARN("(WARNING: must use F16 due to unusual shape) ");
            return_type = GGML_TYPE_F16;
        }
        LLAMA_LOG_WARN("-> falling back to %7s\n", ggml_type_name(return_type));
    }
    return return_type;
}

// internal standard logic for selecting the target tensor type based on tensor category, ftype, and model arch
static ggml_type llama_tensor_get_type_impl(quantize_state_impl & qs, ggml_type new_type, const ggml_tensor * tensor, llama_ftype ftype, tensor_category category) {
    const std::string name = ggml_get_name(tensor);

    // TODO: avoid hardcoded tensor names - use the TN_* constants
    const llm_arch arch = qs.model.arch;

    auto use_more_bits = [](int i_layer, int n_layers) -> bool {
        return i_layer < n_layers/8 || i_layer >= 7*n_layers/8 || (i_layer - n_layers/8)%3 == 2;
    };
    const int n_expert = std::max(1, (int)qs.model.hparams.n_expert);
    auto layer_info = [n_expert] (int i_layer, int n_layer, const char * name) {
        if (n_expert > 1) {
            // Believe it or not, "experts" in the FFN of Mixtral-8x7B are not consecutive, but occasionally randomly
            // sprinkled in the model. Hence, simply dividing i_ffn_down by n_expert does not work
            // for getting the current layer as I initially thought, and we need to resort to parsing the
            // tensor name.
            if (sscanf(name, "blk.%d.", &i_layer) != 1) {
                throw std::runtime_error(format("Failed to determine layer for tensor %s", name));
            }
            if (i_layer < 0 || i_layer >= n_layer) {
                throw std::runtime_error(format("Bad layer %d for tensor %s. Must be in [0, %d)", i_layer, name, n_layer));
            }
        }
        return std::make_pair(i_layer, n_layer);
    };

    // for arches that share the same tensor between the token embeddings and the output, we quantize the token embeddings
    // with the quantization of the output tensor
    if (category == tensor_category::OUTPUT || (qs.has_tied_embeddings && category == tensor_category::TOKEN_EMBD)) {
        if (qs.params->output_tensor_type < GGML_TYPE_COUNT) {
            new_type = qs.params->output_tensor_type;
        } else {
            const int64_t nx = tensor->ne[0];
            const int64_t qk_k = ggml_blck_size(new_type);

            if (ftype == LLAMA_FTYPE_MOSTLY_MXFP4_MOE) {
                new_type = GGML_TYPE_Q8_0;
            }
            else if (arch == LLM_ARCH_FALCON || nx % qk_k != 0) {
                new_type = GGML_TYPE_Q8_0;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS ||
                     ftype == LLAMA_FTYPE_MOSTLY_IQ1_S   || ftype == LLAMA_FTYPE_MOSTLY_IQ2_S  || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M   ||
                     ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) {
                new_type = GGML_TYPE_Q5_K;
            }
            else if (new_type != GGML_TYPE_Q8_0) {
                new_type = GGML_TYPE_Q6_K;
            }
        }
    } else if (ftype == LLAMA_FTYPE_MOSTLY_MXFP4_MOE) {
        // MoE   tensors -> MXFP4
        // other tensors -> Q8_0
        if (tensor->ne[2] > 1) {
            new_type = GGML_TYPE_MXFP4;
        } else {
            new_type = GGML_TYPE_Q8_0;
        }
    } else if (category == tensor_category::TOKEN_EMBD) {
        if (qs.params->token_embedding_type < GGML_TYPE_COUNT) {
            new_type = qs.params->token_embedding_type;
        } else {
            if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS ||
                ftype == LLAMA_FTYPE_MOSTLY_IQ1_S   || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) {
                new_type = GGML_TYPE_Q2_K;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M) {
                new_type = GGML_TYPE_IQ3_S;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
                new_type = GGML_TYPE_IQ3_S;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_TQ1_0 || ftype == LLAMA_FTYPE_MOSTLY_TQ2_0 || ftype == LLAMA_FTYPE_MOSTLY_Q2_0) {
                new_type = GGML_TYPE_Q4_K;
            }
        }
    } else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ1_S ||
               ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M    || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) {
        if (category_is_attn_v(category)) {
            if (qs.model.hparams.n_gqa() >= 4 || qs.model.hparams.n_expert >= 4) new_type = GGML_TYPE_Q4_K;
            else new_type = ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M ? GGML_TYPE_IQ3_S : GGML_TYPE_Q2_K;
            ++qs.i_attention_wv;
        }
        else if (qs.model.hparams.n_expert == 8 && category == tensor_category::ATTENTION_K) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (category == tensor_category::FFN_DOWN) {
            if (qs.i_ffn_down < qs.n_ffn_down/8) {
                new_type = ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M ? GGML_TYPE_IQ3_S : GGML_TYPE_Q2_K;
            }
            ++qs.i_ffn_down;
        }
        else if (category == tensor_category::ATTENTION_OUTPUT) {
            if (qs.model.hparams.n_expert == 8) {
                new_type = GGML_TYPE_Q5_K;
            } else {
                if (ftype == LLAMA_FTYPE_MOSTLY_IQ1_S || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) new_type = GGML_TYPE_IQ2_XXS;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M) new_type = GGML_TYPE_IQ3_S;
            }
        }
    } else if (category_is_attn_v(category)) {
        if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) {
            new_type = qs.model.hparams.n_gqa() >= 4 ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S && qs.model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = qs.model.hparams.n_gqa() >= 4 ? GGML_TYPE_Q4_K : !qs.has_imatrix ? GGML_TYPE_IQ3_S : GGML_TYPE_IQ3_XXS;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_S) && qs.model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) {
            new_type = qs.i_attention_wv < 2 ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q5_K;
        else if ((ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) && qs.model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) &&
                use_more_bits(qs.i_attention_wv, qs.n_attention_wv)) new_type = GGML_TYPE_Q6_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && qs.i_attention_wv < 4) new_type = GGML_TYPE_Q5_K;
        if (qs.model.type == LLM_TYPE_70B) {
            // In the 70B model we have 8 heads sharing the same attn_v weights. As a result, the attn_v.weight tensor is
            // 8x smaller compared to attn_q.weight. Hence, we can get a nice boost in quantization accuracy with
            // nearly negligible increase in model size by quantizing this tensor with more bits:
            if (new_type == GGML_TYPE_Q3_K || new_type == GGML_TYPE_Q4_K) new_type = GGML_TYPE_Q5_K;
        }
        if (qs.model.hparams.n_expert == 8) {
            // for the 8-expert model, bumping this to Q8_0 trades just ~128MB
            // TODO: explore better strategies
            new_type = GGML_TYPE_Q8_0;
        }
        ++qs.i_attention_wv;
    } else if (category == tensor_category::ATTENTION_K) {
        if (qs.model.hparams.n_expert == 8) {
            // for the 8-expert model, bumping this to Q8_0 trades just ~128MB
            // TODO: explore better strategies
            new_type = GGML_TYPE_Q8_0;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = GGML_TYPE_IQ2_S;
        }
    } else if (category == tensor_category::ATTENTION_Q) {
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = GGML_TYPE_IQ2_S;
        }
    } else if (category == tensor_category::FFN_DOWN) {
        auto info = layer_info(qs.i_ffn_down, qs.n_ffn_down, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) new_type = GGML_TYPE_Q3_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S) {
            if (i_layer < n_layer/8) new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS && !qs.has_imatrix) {
            new_type = i_layer < n_layer/8 ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) {
            new_type = i_layer < n_layer/16 ? GGML_TYPE_Q5_K
                     : arch != LLM_ARCH_FALCON || use_more_bits(i_layer, n_layer) ? GGML_TYPE_Q4_K
                     : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M && (i_layer < n_layer/8 ||
                    (qs.model.hparams.n_expert == 8 && use_more_bits(i_layer, n_layer)))) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) {
            new_type = arch == LLM_ARCH_FALCON ? GGML_TYPE_Q4_K : GGML_TYPE_Q5_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M) {
            if (arch == LLM_ARCH_FALCON) {
                new_type = i_layer < n_layer/16 ? GGML_TYPE_Q6_K :
                           use_more_bits(i_layer, n_layer) ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
            } else {
                if (use_more_bits(i_layer, n_layer)) new_type = GGML_TYPE_Q6_K;
            }
        }
        else if (i_layer < n_layer/8 && (ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) && !qs.has_imatrix) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M && use_more_bits(i_layer, n_layer)) new_type = GGML_TYPE_Q6_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && arch != LLM_ARCH_FALCON && i_layer < n_layer/8) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_0 || ftype == LLAMA_FTYPE_MOSTLY_Q5_0)
                && qs.has_imatrix && i_layer < n_layer/8) {
            // Guard against craziness in the first few ffn_down layers that can happen even with imatrix for Q4_0/Q5_0.
            // We only do it when an imatrix is provided because a) we want to make sure that one can always get the
            // same quantization as before imatrix stuff, and b) Q4_1/Q5_1 do go crazy on ffn_down without an imatrix.
            new_type = ftype == LLAMA_FTYPE_MOSTLY_Q4_0 ? GGML_TYPE_Q4_1 : GGML_TYPE_Q5_1;
        }
        ++qs.i_ffn_down;
    } else if (category == tensor_category::ATTENTION_OUTPUT) {
        if (arch != LLM_ARCH_FALCON) {
            if (qs.model.hparams.n_expert == 8) {
                if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K   || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS ||
                    ftype == LLAMA_FTYPE_MOSTLY_Q3_K_S || ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL  ||
                    ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S || ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ3_S  ||
                    ftype == LLAMA_FTYPE_MOSTLY_IQ3_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) {
                    new_type = GGML_TYPE_Q5_K;
                }
            } else {
                if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K   ) new_type = GGML_TYPE_Q3_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) new_type = GGML_TYPE_IQ3_S;
                else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M ) new_type = GGML_TYPE_Q4_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L ) new_type = GGML_TYPE_Q5_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M  ) new_type = GGML_TYPE_Q4_K;
            }
        } else {
            if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q4_K;
        }
    }
    else if (category == tensor_category::ATTENTION_QKV) {
        if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L || ftype == LLAMA_FTYPE_MOSTLY_IQ3_M) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M) new_type = GGML_TYPE_Q5_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) new_type = GGML_TYPE_Q6_K;
    }
    else if (category == tensor_category::FFN_GATE) {
        auto info = layer_info(qs.i_ffn_gate, qs.n_ffn_gate, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS && (i_layer >= n_layer/8 && i_layer < 7*n_layer/8)) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        ++qs.i_ffn_gate;
    }
    else if (category == tensor_category::FFN_UP) {
        auto info = layer_info(qs.i_ffn_up, qs.n_ffn_up, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS && (i_layer >= n_layer/8 && i_layer < 7*n_layer/8)) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        ++qs.i_ffn_up;
    }

    return new_type;
}

// outer wrapper: determine the ggml_type that this tensor should be quantized to
static ggml_type llama_tensor_get_type(quantize_state_impl & qs, const llama_model_quantize_params * params, const ggml_tensor * tensor, ggml_type default_type, const tensor_metadata & tm) {
    if (!tensor_allows_quantization(params, qs.model.arch, tensor)) {
        return tensor->type;
    }
    if (params->token_embedding_type < GGML_TYPE_COUNT && tm.category == tensor_category::TOKEN_EMBD) {
        return params->token_embedding_type;
    }
    if (params->output_tensor_type < GGML_TYPE_COUNT && tm.category == tensor_category::OUTPUT) {
        return params->output_tensor_type;
    }

    ggml_type new_type = default_type;

    // get more optimal quantization type based on the tensor shape, layer, etc.
    if (ggml_is_quantized(default_type)) {
        // if the user provided tensor types - use those
        bool manual = false;
        if (!qs.tensor_type_patterns.empty()) {
            const std::string tensor_name(tensor->name);
            for (const auto & [pattern, qtype] : qs.tensor_type_patterns) {
                if (std::regex_search(tensor_name, pattern)) {
                    if (qtype != new_type) {
                        LLAMA_LOG_WARN("%s: %-36s - applying manual override: %s -> %s\n",
                                       __func__, tensor_name.c_str(), ggml_type_name(new_type), ggml_type_name(qtype));
                        new_type = qtype;
                    }
                    manual = true;
                    break;
                }
            }
        }

        // if not manual - use the standard logic for choosing the quantization type based on the selected mixture
        if (!manual && !params->pure) {
            new_type = llama_tensor_get_type_impl(qs, new_type, tensor, params->ftype, tm.category);
        }

        // incompatible tensor shapes are handled here - fallback to a compatible type
        new_type = tensor_type_fallback(qs, tensor, new_type);
    }

    return new_type;
}

//
// quantization implementation
//

static size_t llama_tensor_quantize_impl(enum ggml_type new_type, const float * f32_data, void * new_data, const int64_t chunk_size, int64_t nrows, int64_t n_per_row, const float * imatrix, std::vector<std::thread> & workers, const int nthread) {
    if (nthread < 2) {
        // single-thread
        size_t new_size = ggml_quantize_chunk(new_type, f32_data, new_data, 0, nrows, n_per_row, imatrix);
        if (!ggml_validate_row_data(new_type, new_data, new_size)) {
            throw std::runtime_error("quantized data validation failed");
        }
        return new_size;
    }

    std::mutex mutex;
    int64_t counter = 0;
    size_t new_size = 0;
    bool valid = true;
    auto compute = [&mutex, &counter, &new_size, &valid, new_type, f32_data, new_data, chunk_size,
            nrows, n_per_row, imatrix]() {
        const int64_t nrows_per_chunk = chunk_size / n_per_row;
        size_t local_size = 0;
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            int64_t first_row = counter; counter += nrows_per_chunk;
            if (first_row >= nrows) {
                if (local_size > 0) {
                    new_size += local_size;
                }
                break;
            }
            lock.unlock();
            const int64_t this_nrow = std::min(nrows - first_row, nrows_per_chunk);
            size_t this_size = ggml_quantize_chunk(new_type, f32_data, new_data, first_row * n_per_row, this_nrow, n_per_row, imatrix);
            local_size += this_size;

            // validate the quantized data
            const size_t row_size  = ggml_row_size(new_type, n_per_row);
            void * this_data = (char *) new_data + first_row * row_size;
            if (!ggml_validate_row_data(new_type, this_data, this_size)) {
                std::unique_lock<std::mutex> lock(mutex);
                valid = false;
                break;
            }
        }
    };
    for (int it = 0; it < nthread - 1; ++it) {
        workers.emplace_back(compute);
    }
    compute();
    for (auto & w : workers) { w.join(); }
    workers.clear();
    if (!valid) {
        throw std::runtime_error("quantized data validation failed");
    }
    return new_size;
}

//
// imatrix requirement check
//

static bool tensor_requires_imatrix(const char * tensor_name, const ggml_type dst_type, const llama_ftype ftype) {
    if (tensor_name_match_token_embd(tensor_name) || tensor_name_match_output_weight(tensor_name)) {
        return false;
    }
    switch (dst_type) {
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ1_S:
            return true;
        case GGML_TYPE_Q2_K:
            // as a general rule, the k-type quantizations don't require imatrix data.
            // the only exception is Q2_K tensors that are part of a Q2_K_S file.
            return ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S;
        default:
            return false;
    }
}

//
// given a file type, get the default tensor type
//

ggml_type llama_ftype_get_default_type(llama_ftype ftype) {
    switch (ftype) {
        case LLAMA_FTYPE_MOSTLY_Q4_0: return GGML_TYPE_Q4_0;
        case LLAMA_FTYPE_MOSTLY_Q4_1: return GGML_TYPE_Q4_1;
        case LLAMA_FTYPE_MOSTLY_Q5_0: return GGML_TYPE_Q5_0;
        case LLAMA_FTYPE_MOSTLY_Q5_1: return GGML_TYPE_Q5_1;
        case LLAMA_FTYPE_MOSTLY_Q8_0: return GGML_TYPE_Q8_0;
        case LLAMA_FTYPE_MOSTLY_F16:  return GGML_TYPE_F16;
        case LLAMA_FTYPE_MOSTLY_BF16: return GGML_TYPE_BF16;
        case LLAMA_FTYPE_ALL_F32:     return GGML_TYPE_F32;
        case LLAMA_FTYPE_MOSTLY_Q1_0: return GGML_TYPE_Q1_0;
        case LLAMA_FTYPE_MOSTLY_Q2_0: return GGML_TYPE_Q2_0;

        case LLAMA_FTYPE_MOSTLY_MXFP4_MOE: return GGML_TYPE_MXFP4;

        // K-quants
        case LLAMA_FTYPE_MOSTLY_Q2_K_S:
        case LLAMA_FTYPE_MOSTLY_Q2_K:    return GGML_TYPE_Q2_K;
        case LLAMA_FTYPE_MOSTLY_IQ3_XS:  return GGML_TYPE_IQ3_S;
        case LLAMA_FTYPE_MOSTLY_Q3_K_S:
        case LLAMA_FTYPE_MOSTLY_Q3_K_M:
        case LLAMA_FTYPE_MOSTLY_Q3_K_L:  return GGML_TYPE_Q3_K;
        case LLAMA_FTYPE_MOSTLY_Q4_K_S:
        case LLAMA_FTYPE_MOSTLY_Q4_K_M:  return GGML_TYPE_Q4_K;
        case LLAMA_FTYPE_MOSTLY_Q5_K_S:
        case LLAMA_FTYPE_MOSTLY_Q5_K_M:  return GGML_TYPE_Q5_K;
        case LLAMA_FTYPE_MOSTLY_Q6_K:    return GGML_TYPE_Q6_K;
        case LLAMA_FTYPE_MOSTLY_TQ1_0:   return GGML_TYPE_TQ1_0;
        case LLAMA_FTYPE_MOSTLY_TQ2_0:   return GGML_TYPE_TQ2_0;
        case LLAMA_FTYPE_MOSTLY_TQ3_1S:  return GGML_TYPE_TQ3_1S;
        case LLAMA_FTYPE_MOSTLY_TQ4_1S:  return GGML_TYPE_TQ4_1S;
        // NanoQuant has no dense default type; its specialized writer emits sidecar groups.
        case LLAMA_FTYPE_MOSTLY_NANOQUANT: return GGML_TYPE_COUNT;
        case LLAMA_FTYPE_MOSTLY_IQ2_XXS: return GGML_TYPE_IQ2_XXS;
        case LLAMA_FTYPE_MOSTLY_IQ2_XS:  return GGML_TYPE_IQ2_XS;
        case LLAMA_FTYPE_MOSTLY_IQ2_S:   return GGML_TYPE_IQ2_XS;
        case LLAMA_FTYPE_MOSTLY_IQ2_M:   return GGML_TYPE_IQ2_S;
        case LLAMA_FTYPE_MOSTLY_IQ3_XXS: return GGML_TYPE_IQ3_XXS;
        case LLAMA_FTYPE_MOSTLY_IQ1_S:   return GGML_TYPE_IQ1_S;
        case LLAMA_FTYPE_MOSTLY_IQ1_M:   return GGML_TYPE_IQ1_M;
        case LLAMA_FTYPE_MOSTLY_IQ4_NL:  return GGML_TYPE_IQ4_NL;
        case LLAMA_FTYPE_MOSTLY_IQ4_XS:  return GGML_TYPE_IQ4_XS;
        case LLAMA_FTYPE_MOSTLY_IQ3_S:
        case LLAMA_FTYPE_MOSTLY_IQ3_M:   return GGML_TYPE_IQ3_S;

        default: return GGML_TYPE_COUNT;
    }
}


static void init_quantize_state_counters(quantize_state_impl & qs, std::vector<tensor_metadata> & metadata) {
    for (auto & tm : metadata) {
        tensor_category cat = tensor_get_category(tm.name);
        tm.category = cat;

        if (category_is_attn_v(cat)) {
            ++qs.n_attention_wv;
        }

        if (cat == tensor_category::OUTPUT) {
            qs.has_tied_embeddings = false;
        }
    }
    qs.n_ffn_down = qs.n_ffn_gate = qs.n_ffn_up = (int)qs.model.hparams.n_layer_all;
}

namespace nanoquant {

static constexpr uint32_t CHECKPOINT_VERSION = 6;
static constexpr size_t PROJECTION_MEMORY_BUDGET = 256u * 1024u * 1024u;
static constexpr size_t GRADIENT_MEMORY_BUDGET = 256u * 1024u * 1024u;
static constexpr size_t MODEL_DEVICE_RESERVE = size_t(2u) * 1024u * 1024u * 1024u;
static constexpr int32_t ADMM_LOG_INTERVAL = 25;
static constexpr float CALIBRATION_SHRINKAGE = 0.4f;
static constexpr float ADMM_REGULARIZATION = 3.0e-2f;
static constexpr float NUMERIC_EPSILON = 1.0e-8f;
static bool source_type_supports_f32(ggml_type type) {
    return type == GGML_TYPE_F32 ||
           type == GGML_TYPE_F16 ||
           type == GGML_TYPE_BF16 ||
           (ggml_is_quantized(type) && ggml_get_type_traits(type)->to_float != nullptr);
}


using hash256 = std::array<uint64_t, 4>;

struct hash_builder {
    hash256 value = {
        UINT64_C(1469598103934665603),
        UINT64_C(1099511628211),
        UINT64_C(7809847782465536322),
        UINT64_C(9650029242287828579),
    };

    void update(const void * data, size_t size) {
        static constexpr uint64_t primes[4] = {
            UINT64_C(1099511628211),
            UINT64_C(14029467366897019727),
            UINT64_C(1609587929392839161),
            UINT64_C(9650029242287828579),
        };
        const auto * bytes = static_cast<const uint8_t *>(data);
        for (size_t i = 0; i < size; ++i) {
            for (size_t lane = 0; lane < value.size(); ++lane) {
                value[lane] ^= uint64_t(bytes[i]) + uint64_t(lane * 0x9d);
                value[lane] *= primes[lane];
                value[lane] ^= value[lane] >> (17 + lane);
            }
        }
    }

    template<class T>
    void update_pod(const T & item) {
        static_assert(std::is_trivially_copyable<T>::value, "hash input must be POD");
        update(&item, sizeof(item));
    }

    void update_string(const std::string & text) {
        const uint64_t size = text.size();
        update_pod(size);
        update(text.data(), text.size());
    }
};

static hash256 hash_file(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(format("NanoQuant: cannot open '%s' for identity hashing", path.c_str()));
    }
    hash_builder hash;
    std::vector<char> buffer(1024u * 1024u);
    while (input) {
        input.read(buffer.data(), buffer.size());
        const std::streamsize count = input.gcount();
        if (count > 0) {
            hash.update(buffer.data(), size_t(count));
        }
    }
    if (!input.eof()) {
        throw std::runtime_error(format("NanoQuant: failed while hashing '%s'", path.c_str()));
    }
    return hash.value;
}

static std::string hash_hex(const hash256 & hash) {
    static const char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(64);
    for (uint64_t lane : hash) {
        for (int shift = 60; shift >= 0; shift -= 4) {
            result.push_back(digits[(lane >> shift) & 0x0f]);
        }
    }
    return result;
}

struct deterministic_rng {
    uint64_t state;

    explicit deterministic_rng(uint64_t seed) :
        state(seed ? seed : UINT64_C(0x9e3779b97f4a7c15)) {
    }

    uint64_t next_u64() {
        uint64_t x = state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        state = x;
        return x * UINT64_C(2685821657736338717);
    }

    float uniform_open() {
        return float((next_u64() >> 40) + 1) / float((UINT64_C(1) << 24) + 1);
    }

    float normal() {
        const float u1 = uniform_open();
        const float u2 = uniform_open();
        return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853071795864769f * u2);
    }
};

static bool has_suffix(const std::string & text, const char * suffix) {
    const size_t size = std::strlen(suffix);
    return text.size() >= size && text.compare(text.size() - size, size, suffix) == 0;
}

static int decoder_block(const std::string & name) {
    int block = -1;
    int consumed = 0;
    if (std::sscanf(name.c_str(), "blk.%d.%n", &block, &consumed) != 1 || consumed <= 0 || block < 0) {
        return -1;
    }
    return block;
}

static std::string sidecar_name(const std::string & weight_name, const char * suffix) {
    if (!has_suffix(weight_name, ".weight")) {
        throw std::runtime_error(format("NanoQuant: '%s' is not a .weight tensor", weight_name.c_str()));
    }
    std::string result = weight_name.substr(0, weight_name.size() - std::strlen(".weight")) + suffix;
    if (result.size() >= GGML_MAX_NAME) {
        throw std::runtime_error(format("NanoQuant: sidecar name for '%s' exceeds GGML_MAX_NAME", weight_name.c_str()));
    }
    return result;
}


struct group {
    const llama_model_loader::llama_tensor_weight * weight = nullptr;
    std::string name;
    std::string name_v;
    std::string name_u;
    std::string name_scale_pre;
    std::string name_scale_post;
    int block = -1;
    int64_t n_in = 0;
    int64_t n_out = 0;
    int64_t n_expert = 1;
    int64_t n_expert_used = 1;
    int64_t rank = 0;

    size_t v_size() const {
        return size_t((n_in + 31) / 32) * size_t(rank) * size_t(n_expert) * sizeof(uint32_t);
    }

    size_t u_size() const {
        return size_t((rank + 31) / 32) * size_t(n_out) * size_t(n_expert) * sizeof(uint32_t);
    }

    size_t scale_pre_size() const {
        return size_t(n_in) * size_t(n_expert) * sizeof(ggml_fp16_t);
    }

    size_t scale_post_size() const {
        return size_t(n_out) * size_t(n_expert) * sizeof(ggml_fp16_t);
    }

    size_t payload_size() const {
        return v_size() + u_size() + scale_pre_size() + scale_post_size();
    }

    size_t physical_size(size_t alignment) const {
        return GGML_PAD(v_size(), alignment) +
               GGML_PAD(u_size(), alignment) +
               GGML_PAD(scale_pre_size(), alignment) +
               GGML_PAD(scale_post_size(), alignment);
    }
};
static size_t allocate_ranks(
        std::vector<group> & groups,
        size_t alignment,
        size_t budget) {
    size_t used = 0;
    for (group & item : groups) {
        item.rank = 1;
        const size_t size = item.physical_size(alignment);
        if (size > std::numeric_limits<size_t>::max() - used) {
            throw std::runtime_error("NanoQuant: physical size overflow");
        }
        used += size;
    }
    if (used > budget) {
        throw std::runtime_error("NanoQuant: projection budget is below the rank-one layout");
    }

    while (true) {
        size_t best = SIZE_MAX;
        size_t best_size = 0;
        long double best_bits = std::numeric_limits<long double>::infinity();
        for (size_t i = 0; i < groups.size(); ++i) {
            group & item = groups[i];
            if (item.rank >= 2*std::min(item.n_in, item.n_out)) {
                continue;
            }
            const size_t current_size = item.physical_size(alignment);
            ++item.rank;
            const size_t next_size = item.physical_size(alignment);
            --item.rank;
            const size_t increment = next_size - current_size;
            if (increment > budget - used) {
                continue;
            }
            const long double bits =
                    (long double) next_size*8.0L/
                    ((long double) item.n_in*(long double) item.n_out*(long double) item.n_expert);
            if (bits < best_bits || (bits == best_bits && i < best)) {
                best = i;
                best_size = increment;
                best_bits = bits;
            }
        }
        if (best == SIZE_MAX) {
            break;
        }
        ++groups[best].rank;
        used += best_size;
    }
    LLAMA_LOG_INFO(
            "NanoQuant: projection rank allocation budget=%zu bytes used=%zu bytes slack=%zu bytes\n",
            budget, used, budget - used);
    return used;
}


enum class checkpoint_stage : uint32_t {
    NONE = 0,
    NONFACTOR = 1,
    ADMM = 2,
    FACTOR = 3,
    EXPERT_DONE = 4,
    GROUP_DONE = 5,
    MODEL = 6,
    MODEL_DONE = 7,
};

struct checkpoint_state {
    checkpoint_stage stage = checkpoint_stage::NONE;
    uint32_t progress = 0;
    uint64_t rng_state = 0;
    uint64_t optimizer_step = 0;
    uint32_t expert = 0;
    std::vector<float> weight;
    std::vector<float> u;
    std::vector<float> v;
    std::vector<float> z_u;
    std::vector<float> z_v;
    std::vector<float> dual_u;
    std::vector<float> dual_v;
    std::vector<float> scale_pre;
    std::vector<float> scale_post;
    std::vector<float> weight_first_moment;
    std::vector<float> weight_second_moment;
    std::vector<float> u_first_moment;
    std::vector<float> u_second_moment;
    std::vector<float> v_first_moment;
    std::vector<float> v_second_moment;
    std::vector<float> scale_pre_first_moment;
    std::vector<float> scale_pre_second_moment;
    std::vector<float> scale_post_first_moment;
    std::vector<float> scale_post_second_moment;
    std::vector<uint32_t> packed_u;
    std::vector<uint32_t> packed_v;
    std::vector<float> completed_scale_pre;
    std::vector<float> completed_scale_post;
    std::vector<uint32_t> completed_packed_u;
    std::vector<uint32_t> completed_packed_v;
};

template<class T>
static void release_vector(std::vector<T> & values) {
    std::vector<T>().swap(values);
}

static void release_completed_state(checkpoint_state & state) {
    release_vector(state.weight);
    release_vector(state.u);
    release_vector(state.v);
    release_vector(state.z_u);
    release_vector(state.z_v);
    release_vector(state.dual_u);
    release_vector(state.dual_v);
    release_vector(state.weight_first_moment);
    release_vector(state.weight_second_moment);
    release_vector(state.u_first_moment);
    release_vector(state.u_second_moment);
    release_vector(state.v_first_moment);
    release_vector(state.v_second_moment);
    release_vector(state.scale_pre_first_moment);
    release_vector(state.scale_pre_second_moment);
    release_vector(state.scale_post_first_moment);
    release_vector(state.scale_post_second_moment);
}
static void release_attached_state(checkpoint_state & state) {
    release_vector(state.packed_u);
    release_vector(state.packed_v);
}

template<class T>
static void write_pod(std::ostream & output, const T & value) {
    static_assert(std::is_trivially_copyable<T>::value, "checkpoint item must be POD");
    output.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

template<class T>
static T read_pod(std::istream & input) {
    static_assert(std::is_trivially_copyable<T>::value, "checkpoint item must be POD");
    T result;
    input.read(reinterpret_cast<char *>(&result), sizeof(result));
    return result;
}

static void write_string(std::ostream & output, const std::string & value) {
    write_pod(output, uint64_t(value.size()));
    output.write(value.data(), value.size());
}

static std::string read_string(std::istream & input) {
    const uint64_t size = read_pod<uint64_t>(input);
    if (size > GGML_MAX_NAME * 4u) {
        throw std::runtime_error("NanoQuant: invalid checkpoint string length");
    }
    std::string result(size, '\0');
    input.read(result.data(), result.size());
    return result;
}

template<class T>
static void write_vector(std::ostream & output, const std::vector<T> & values) {
    write_pod(output, uint64_t(values.size()));
    if (!values.empty()) {
        output.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(T));
    }
}

template<class T>
static std::vector<T> read_vector(std::istream & input) {
    const uint64_t size = read_pod<uint64_t>(input);
    if (size > (UINT64_C(1) << 31)) {
        throw std::runtime_error("NanoQuant: invalid checkpoint vector length");
    }
    std::vector<T> result(size);
    if (!result.empty()) {
        input.read(reinterpret_cast<char *>(result.data()), result.size() * sizeof(T));
    }
    return result;
}

static std::filesystem::path checkpoint_path(const std::filesystem::path & directory, const std::string & name) {
    hash_builder hash;
    hash.update_string(name);
    return directory / (hash_hex(hash.value) + ".nqckpt");
}

static void atomic_replace(const std::filesystem::path & path, const std::function<void(std::ostream &)> & writer) {
    const std::filesystem::path temporary = path.string() + ".tmp";
    const std::filesystem::path backup = path.string() + ".bak";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        output.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        writer(output);
        output.flush();
    }
    std::error_code ec;
    std::filesystem::remove(backup, ec);
    ec.clear();
    if (std::filesystem::exists(path)) {
        std::filesystem::rename(path, backup, ec);
        if (ec) {
            throw std::runtime_error(format("NanoQuant: cannot rotate checkpoint '%s': %s",
                    path.string().c_str(), ec.message().c_str()));
        }
    }
    std::filesystem::rename(temporary, path, ec);
    if (ec) {
        if (std::filesystem::exists(backup)) {
            std::error_code restore_ec;
            std::filesystem::rename(backup, path, restore_ec);
        }
        throw std::runtime_error(format("NanoQuant: cannot install checkpoint '%s': %s",
                path.string().c_str(), ec.message().c_str()));
    }
    std::filesystem::remove(backup, ec);
}

static std::filesystem::path expert_checkpoint_path(const std::filesystem::path & path) {
    return path.string() + ".experts";
}

static size_t expert_packed_u_count(const group & item) {
    return size_t((item.rank + 31)/32)*size_t(item.n_out);
}

static size_t expert_packed_v_count(const group & item) {
    return size_t((item.n_in + 31)/32)*size_t(item.rank);
}

static size_t expert_checkpoint_record_size(const group & item) {
    return sizeof(uint32_t) + sizeof(hash256) +
            (size_t(item.n_in) + size_t(item.n_out))*sizeof(float) +
            (expert_packed_u_count(item) + expert_packed_v_count(item))*sizeof(uint32_t);
}

static hash256 expert_checkpoint_hash(
        const group & item,
        uint32_t expert,
        const float * scale_pre,
        const float * scale_post,
        const uint32_t * packed_u,
        const uint32_t * packed_v) {
    hash_builder hash;
    hash.update_string("llama.cpp-nanoquant-expert-v1");
    hash.update_pod(expert);
    hash.update(scale_pre, size_t(item.n_in)*sizeof(float));
    hash.update(scale_post, size_t(item.n_out)*sizeof(float));
    hash.update(packed_u, expert_packed_u_count(item)*sizeof(uint32_t));
    hash.update(packed_v, expert_packed_v_count(item)*sizeof(uint32_t));
    return hash.value;
}

static uint64_t validate_expert_checkpoint(
        std::istream & input,
        const std::filesystem::path & path,
        const hash256 & source_hash,
        const hash256 & config_hash,
        const group & item) {
    char magic[8];
    input.read(magic, sizeof(magic));
    static const char expected_magic[8] = { 'N', 'Q', 'E', 'X', 'P', 'R', '1', '\0' };
    if (std::memcmp(magic, expected_magic, sizeof(magic)) != 0 ||
        read_pod<uint32_t>(input) != CHECKPOINT_VERSION ||
        read_pod<uint32_t>(input) != UINT32_C(0x01020304)) {
        throw std::runtime_error(format(
                "NanoQuant: invalid expert checkpoint '%s'", path.string().c_str()));
    }
    hash256 stored_source;
    hash256 stored_config;
    input.read(reinterpret_cast<char *>(stored_source.data()), sizeof(stored_source));
    input.read(reinterpret_cast<char *>(stored_config.data()), sizeof(stored_config));
    const std::string stored_name = read_string(input);
    const int64_t stored_n_in = read_pod<int64_t>(input);
    const int64_t stored_n_out = read_pod<int64_t>(input);
    const int64_t stored_n_expert = read_pod<int64_t>(input);
    const int64_t stored_rank = read_pod<int64_t>(input);
    if (stored_source != source_hash || stored_config != config_hash ||
        stored_name != item.name || stored_n_in != item.n_in ||
        stored_n_out != item.n_out || stored_n_expert != item.n_expert ||
        stored_rank != item.rank) {
        throw std::runtime_error(format(
                "NanoQuant: expert checkpoint identity mismatch for '%s'", item.name.c_str()));
    }
    const std::streampos position = input.tellg();
    if (position < 0) {
        throw std::runtime_error(format(
                "NanoQuant: invalid expert checkpoint header for '%s'", item.name.c_str()));
    }
    return uint64_t(position);
}

static uint64_t create_expert_checkpoint(
        const std::filesystem::path & path,
        const hash256 & source_hash,
        const hash256 & config_hash,
        const group & item) {
    atomic_replace(path, [&](std::ostream & output) {
        static const char magic[8] = { 'N', 'Q', 'E', 'X', 'P', 'R', '1', '\0' };
        output.write(magic, sizeof(magic));
        write_pod(output, CHECKPOINT_VERSION);
        write_pod(output, UINT32_C(0x01020304));
        output.write(reinterpret_cast<const char *>(source_hash.data()), sizeof(source_hash));
        output.write(reinterpret_cast<const char *>(config_hash.data()), sizeof(config_hash));
        write_string(output, item.name);
        write_pod(output, item.n_in);
        write_pod(output, item.n_out);
        write_pod(output, item.n_expert);
        write_pod(output, item.rank);
        const size_t data_size =
                expert_checkpoint_record_size(item)*size_t(item.n_expert);
        if (data_size > 0) {
            output.seekp(std::streamoff(data_size - 1), std::ios::cur);
            output.put('\0');
        }
    });
    std::ifstream input(path, std::ios::binary);
    input.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    return validate_expert_checkpoint(
            input, path, source_hash, config_hash, item);
}

static uint64_t open_expert_checkpoint(
        const std::filesystem::path & path,
        const hash256 & source_hash,
        const hash256 & config_hash,
        const group & item) {
    std::ifstream input(path, std::ios::binary);
    input.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    const uint64_t data_offset = validate_expert_checkpoint(
            input, path, source_hash, config_hash, item);
    const uint64_t expected_size = data_offset +
            uint64_t(expert_checkpoint_record_size(item))*uint64_t(item.n_expert);
    if (std::filesystem::file_size(path) != expected_size) {
        throw std::runtime_error(format(
                "NanoQuant: expert checkpoint size mismatch for '%s'", item.name.c_str()));
    }
    return data_offset;
}

static void write_expert_checkpoint(
        const std::filesystem::path & checkpoint,
        const hash256 & source_hash,
        const hash256 & config_hash,
        const group & item,
        uint32_t expert,
        const float * scale_pre,
        const float * scale_post,
        const uint32_t * packed_u,
        const uint32_t * packed_v) {
    const std::filesystem::path path = expert_checkpoint_path(checkpoint);
    const uint64_t data_offset = std::filesystem::exists(path) ?
            open_expert_checkpoint(path, source_hash, config_hash, item) :
            create_expert_checkpoint(path, source_hash, config_hash, item);
    const size_t record_size = expert_checkpoint_record_size(item);
    const uint64_t record_offset = data_offset + uint64_t(expert)*record_size;
    const hash256 digest = expert_checkpoint_hash(
            item, expert, scale_pre, scale_post, packed_u, packed_v);

    std::fstream output(path, std::ios::binary | std::ios::in | std::ios::out);
    output.exceptions(std::fstream::failbit | std::fstream::badbit);
    output.seekp(std::streamoff(record_offset + sizeof(uint32_t) + sizeof(hash256)));
    output.write(reinterpret_cast<const char *>(scale_pre), size_t(item.n_in)*sizeof(float));
    output.write(reinterpret_cast<const char *>(scale_post), size_t(item.n_out)*sizeof(float));
    output.write(reinterpret_cast<const char *>(packed_u),
            expert_packed_u_count(item)*sizeof(uint32_t));
    output.write(reinterpret_cast<const char *>(packed_v),
            expert_packed_v_count(item)*sizeof(uint32_t));
    output.seekp(std::streamoff(record_offset + sizeof(uint32_t)));
    output.write(reinterpret_cast<const char *>(digest.data()), sizeof(digest));
    output.flush();
    output.seekp(std::streamoff(record_offset));
    write_pod(output, expert + 1);
    output.flush();
}

static void restore_completed_experts(
        const std::filesystem::path & checkpoint,
        const hash256 & source_hash,
        const hash256 & config_hash,
        const group & item,
        checkpoint_state & state,
        bool retain) {
    if (item.n_expert <= 1 || state.expert == 0 ||
        state.stage >= checkpoint_stage::GROUP_DONE) {
        return;
    }
    const size_t n_completed = state.expert;
    const size_t n_scale_pre = n_completed*size_t(item.n_in);
    const size_t n_scale_post = n_completed*size_t(item.n_out);
    const size_t n_packed_u = n_completed*expert_packed_u_count(item);
    const size_t n_packed_v = n_completed*expert_packed_v_count(item);
    const bool populated =
            state.completed_scale_pre.size() == n_scale_pre &&
            state.completed_scale_post.size() == n_scale_post &&
            state.completed_packed_u.size() == n_packed_u &&
            state.completed_packed_v.size() == n_packed_v;
    const bool empty =
            state.completed_scale_pre.empty() &&
            state.completed_scale_post.empty() &&
            state.completed_packed_u.empty() &&
            state.completed_packed_v.empty();
    if (!populated && !empty) {
        throw std::runtime_error(format(
                "NanoQuant: incomplete expert checkpoint state for '%s'", item.name.c_str()));
    }

    const std::filesystem::path path = expert_checkpoint_path(checkpoint);
    if (populated) {
        for (uint32_t expert = 0; expert < state.expert; ++expert) {
            write_expert_checkpoint(
                    checkpoint, source_hash, config_hash, item, expert,
                    state.completed_scale_pre.data() + size_t(expert)*size_t(item.n_in),
                    state.completed_scale_post.data() + size_t(expert)*size_t(item.n_out),
                    state.completed_packed_u.data() + size_t(expert)*expert_packed_u_count(item),
                    state.completed_packed_v.data() + size_t(expert)*expert_packed_v_count(item));
        }
        if (!retain) {
            release_vector(state.completed_scale_pre);
            release_vector(state.completed_scale_post);
            release_vector(state.completed_packed_u);
            release_vector(state.completed_packed_v);
        }
        return;
    }
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error(format(
                "NanoQuant: missing expert checkpoint for '%s'", item.name.c_str()));
    }

    const uint64_t data_offset = open_expert_checkpoint(
            path, source_hash, config_hash, item);
    state.completed_scale_pre.resize(n_scale_pre);
    state.completed_scale_post.resize(n_scale_post);
    state.completed_packed_u.resize(n_packed_u);
    state.completed_packed_v.resize(n_packed_v);
    std::ifstream input(path, std::ios::binary);
    input.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    const size_t record_size = expert_checkpoint_record_size(item);
    for (uint32_t expert = 0; expert < state.expert; ++expert) {
        input.seekg(std::streamoff(data_offset + uint64_t(expert)*record_size));
        if (read_pod<uint32_t>(input) != expert + 1) {
            throw std::runtime_error(format(
                    "NanoQuant: incomplete expert %u checkpoint for '%s'",
                    expert + 1, item.name.c_str()));
        }
        hash256 stored_digest;
        input.read(reinterpret_cast<char *>(stored_digest.data()), sizeof(stored_digest));
        float * scale_pre =
                state.completed_scale_pre.data() + size_t(expert)*size_t(item.n_in);
        float * scale_post =
                state.completed_scale_post.data() + size_t(expert)*size_t(item.n_out);
        uint32_t * packed_u =
                state.completed_packed_u.data() + size_t(expert)*expert_packed_u_count(item);
        uint32_t * packed_v =
                state.completed_packed_v.data() + size_t(expert)*expert_packed_v_count(item);
        input.read(reinterpret_cast<char *>(scale_pre), size_t(item.n_in)*sizeof(float));
        input.read(reinterpret_cast<char *>(scale_post), size_t(item.n_out)*sizeof(float));
        input.read(reinterpret_cast<char *>(packed_u),
                expert_packed_u_count(item)*sizeof(uint32_t));
        input.read(reinterpret_cast<char *>(packed_v),
                expert_packed_v_count(item)*sizeof(uint32_t));
        if (stored_digest != expert_checkpoint_hash(
                item, expert, scale_pre, scale_post, packed_u, packed_v)) {
            throw std::runtime_error(format(
                    "NanoQuant: corrupt expert %u checkpoint for '%s'",
                    expert + 1, item.name.c_str()));
        }
    }
    if (!retain) {
        release_vector(state.completed_scale_pre);
        release_vector(state.completed_scale_post);
        release_vector(state.completed_packed_u);
        release_vector(state.completed_packed_v);
    }
}

static void save_checkpoint(
        const std::filesystem::path & directory,
        const hash256 & source_hash,
        const hash256 & config_hash,
        const group & item,
        const checkpoint_state & state) {
    const auto path = checkpoint_path(directory, item.name);
    atomic_replace(path, [&](std::ostream & output) {
        static const char magic[8] = { 'N', 'Q', 'C', 'K', 'P', 'T', '1', '\0' };
        output.write(magic, sizeof(magic));
        write_pod(output, CHECKPOINT_VERSION);
        write_pod(output, UINT32_C(0x01020304));
        output.write(reinterpret_cast<const char *>(source_hash.data()), sizeof(source_hash));
        output.write(reinterpret_cast<const char *>(config_hash.data()), sizeof(config_hash));
        write_string(output, item.name);
        write_pod(output, item.n_in);
        write_pod(output, item.n_out);
        write_pod(output, item.n_expert);
        write_pod(output, item.rank);
        write_pod(output, uint32_t(state.stage));
        write_pod(output, state.progress);
        write_pod(output, state.rng_state);
        write_pod(output, state.optimizer_step);
        write_pod(output, state.expert);
        write_vector(output, state.weight);
        write_vector(output, state.u);
        write_vector(output, state.v);
        write_vector(output, state.z_u);
        write_vector(output, state.z_v);
        write_vector(output, state.dual_u);
        write_vector(output, state.dual_v);
        write_vector(output, state.scale_pre);
        write_vector(output, state.scale_post);
        write_vector(output, state.weight_first_moment);
        write_vector(output, state.weight_second_moment);
        write_vector(output, state.u_first_moment);
        write_vector(output, state.u_second_moment);
        write_vector(output, state.v_first_moment);
        write_vector(output, state.v_second_moment);
        write_vector(output, state.scale_pre_first_moment);
        write_vector(output, state.scale_pre_second_moment);
        write_vector(output, state.scale_post_first_moment);
        write_vector(output, state.scale_post_second_moment);
        write_vector(output, state.packed_u);
        write_vector(output, state.packed_v);
        if (item.n_expert > 1) {
            write_pod(output, UINT64_C(0));
            write_pod(output, UINT64_C(0));
            write_pod(output, UINT64_C(0));
            write_pod(output, UINT64_C(0));
        } else {
            write_vector(output, state.completed_scale_pre);
            write_vector(output, state.completed_scale_post);
            write_vector(output, state.completed_packed_u);
            write_vector(output, state.completed_packed_v);
        }
    });
}

static bool load_checkpoint_file(
        const std::filesystem::path & path,
        const hash256 & source_hash,
        const hash256 & config_hash,
        const group & item,
        checkpoint_state & state) {
    std::filesystem::path selected = path;
    if (!std::filesystem::exists(selected)) {
        selected = path.string() + ".bak";
        if (!std::filesystem::exists(selected)) {
            return false;
        }
    }
    std::ifstream input(selected, std::ios::binary);
    input.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    char magic[8];
    input.read(magic, sizeof(magic));
    static const char expected_magic[8] = { 'N', 'Q', 'C', 'K', 'P', 'T', '1', '\0' };
    if (std::memcmp(magic, expected_magic, sizeof(magic)) != 0 ||
        read_pod<uint32_t>(input) != CHECKPOINT_VERSION ||
        read_pod<uint32_t>(input) != UINT32_C(0x01020304)) {
        throw std::runtime_error(format("NanoQuant: invalid checkpoint '%s'", selected.string().c_str()));
    }
    hash256 stored_source;
    hash256 stored_config;
    input.read(reinterpret_cast<char *>(stored_source.data()), sizeof(stored_source));
    input.read(reinterpret_cast<char *>(stored_config.data()), sizeof(stored_config));
    const std::string stored_name = read_string(input);
    const int64_t stored_n_in = read_pod<int64_t>(input);
    const int64_t stored_n_out = read_pod<int64_t>(input);
    const int64_t stored_n_expert = read_pod<int64_t>(input);
    const int64_t stored_rank = read_pod<int64_t>(input);
    if (stored_source != source_hash || stored_config != config_hash ||
        stored_name != item.name || stored_n_in != item.n_in ||
        stored_n_out != item.n_out || stored_n_expert != item.n_expert || stored_rank != item.rank) {
        throw std::runtime_error(format("NanoQuant: checkpoint identity mismatch for '%s'", item.name.c_str()));
    }
    const uint32_t stored_stage = read_pod<uint32_t>(input);
    if (stored_stage > uint32_t(checkpoint_stage::MODEL_DONE)) {
        throw std::runtime_error(format("NanoQuant: invalid checkpoint stage for '%s'", item.name.c_str()));
    }
    state.stage = checkpoint_stage(stored_stage);
    state.progress = read_pod<uint32_t>(input);
    state.rng_state = read_pod<uint64_t>(input);
    state.optimizer_step = read_pod<uint64_t>(input);
    state.expert = read_pod<uint32_t>(input);
    state.weight = read_vector<float>(input);
    state.u = read_vector<float>(input);
    state.v = read_vector<float>(input);
    state.z_u = read_vector<float>(input);
    state.z_v = read_vector<float>(input);
    state.dual_u = read_vector<float>(input);
    state.dual_v = read_vector<float>(input);
    state.scale_pre = read_vector<float>(input);
    state.scale_post = read_vector<float>(input);
    state.weight_first_moment = read_vector<float>(input);
    state.weight_second_moment = read_vector<float>(input);
    state.u_first_moment = read_vector<float>(input);
    state.u_second_moment = read_vector<float>(input);
    state.v_first_moment = read_vector<float>(input);
    state.v_second_moment = read_vector<float>(input);
    state.scale_pre_first_moment = read_vector<float>(input);
    state.scale_pre_second_moment = read_vector<float>(input);
    state.scale_post_first_moment = read_vector<float>(input);
    state.scale_post_second_moment = read_vector<float>(input);
    state.packed_u = read_vector<uint32_t>(input);
    state.packed_v = read_vector<uint32_t>(input);
    state.completed_scale_pre = read_vector<float>(input);
    state.completed_scale_post = read_vector<float>(input);
    state.completed_packed_u = read_vector<uint32_t>(input);
    state.completed_packed_v = read_vector<uint32_t>(input);
    if (input.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error(format("NanoQuant: trailing data in checkpoint for '%s'", item.name.c_str()));
    }
    restore_completed_experts(
            path, source_hash, config_hash, item, state, false);
    return true;
}

static void validate_state_shapes(
        const group & item,
        const checkpoint_state & state,
        bool require_packed = true) {
    const size_t weight_size = size_t(item.n_in)*size_t(item.n_out);
    const size_t u_size = size_t(item.n_out)*size_t(item.rank);
    const size_t v_size = size_t(item.rank)*size_t(item.n_in);
    const bool transposed_admm =
            state.stage == checkpoint_stage::ADMM && item.n_out < item.n_in;
    const size_t admm_u_size = transposed_admm ?
            size_t(item.n_in)*size_t(item.rank) : u_size;
    const size_t admm_v_size = transposed_admm ?
            size_t(item.rank)*size_t(item.n_out) : v_size;
    auto require_size = [&](size_t actual, size_t expected, const char * field) {
        if (actual != expected) {
            throw std::runtime_error(format("NanoQuant: checkpoint %s size mismatch for '%s' (%zu != %zu)",
                    field, item.name.c_str(), actual, expected));
        }
    };
    const bool group_done = state.stage >= checkpoint_stage::GROUP_DONE;
    if ((!group_done && state.expert >= uint32_t(item.n_expert)) ||
        (group_done && state.expert != uint32_t(item.n_expert))) {
        throw std::runtime_error(format(
                "NanoQuant: checkpoint expert index mismatch for '%s' (%u of %" PRId64 ")",
                item.name.c_str(), state.expert, item.n_expert));
    }
    const size_t completed_experts = group_done ? 0 : state.expert;
    const bool streamed_experts = item.n_expert > 1 &&
            state.completed_scale_pre.empty() &&
            state.completed_scale_post.empty() &&
            state.completed_packed_u.empty() &&
            state.completed_packed_v.empty();
    if (!streamed_experts) {
        require_size(state.completed_scale_pre.size(), completed_experts*size_t(item.n_in),
                "completed scale_pre");
        require_size(state.completed_scale_post.size(), completed_experts*size_t(item.n_out),
                "completed scale_post");
        require_size(state.completed_packed_u.size(),
                completed_experts*size_t((item.rank + 31)/32)*size_t(item.n_out),
                "completed packed_U");
        require_size(state.completed_packed_v.size(),
                completed_experts*size_t((item.n_in + 31)/32)*size_t(item.rank),
                "completed packed_V");
    }
    if (state.stage == checkpoint_stage::NONFACTOR ||
        state.stage == checkpoint_stage::ADMM) {
        require_size(state.weight.size(), weight_size, "weight");
    }
    if (state.stage == checkpoint_stage::NONFACTOR && state.progress > 0) {
        require_size(state.weight_first_moment.size(), weight_size, "weight first moment");
        require_size(state.weight_second_moment.size(), weight_size, "weight second moment");
    }
    if (state.stage == checkpoint_stage::ADMM) {
        require_size(state.u.size(), admm_u_size, "U");
        require_size(state.v.size(), admm_v_size, "V");
        require_size(state.z_u.size(), admm_u_size, "Z_U");
        require_size(state.z_v.size(), admm_v_size, "Z_V");
        require_size(state.dual_u.size(), admm_u_size, "dual_U");
        require_size(state.dual_v.size(), admm_v_size, "dual_V");
    } else if (state.stage == checkpoint_stage::FACTOR) {
        require_size(state.u.size(), u_size, "U");
        require_size(state.v.size(), v_size, "V");
    }
    if (state.stage >= checkpoint_stage::FACTOR) {
        const size_t scale_experts = group_done ? size_t(item.n_expert) : 1;
        require_size(state.scale_pre.size(), scale_experts*size_t(item.n_in), "scale_pre");
        require_size(state.scale_post.size(), scale_experts*size_t(item.n_out), "scale_post");
    }
    if (state.stage == checkpoint_stage::FACTOR && state.progress > 0) {
        require_size(state.u_first_moment.size(), u_size, "U first moment");
        require_size(state.u_second_moment.size(), u_size, "U second moment");
        require_size(state.v_first_moment.size(), v_size, "V first moment");
        require_size(state.v_second_moment.size(), v_size, "V second moment");
        require_size(state.scale_pre_first_moment.size(), item.n_in, "scale_pre first moment");
        require_size(state.scale_pre_second_moment.size(), item.n_in, "scale_pre second moment");
        require_size(state.scale_post_first_moment.size(), item.n_out, "scale_post first moment");
        require_size(state.scale_post_second_moment.size(), item.n_out, "scale_post second moment");
    }
    if (require_packed && state.stage == checkpoint_stage::EXPERT_DONE) {
        require_size(state.packed_u.size(),
                size_t((item.rank + 31)/32)*size_t(item.n_out), "packed_U");
        require_size(state.packed_v.size(),
                size_t((item.n_in + 31)/32)*size_t(item.rank), "packed_V");
    } else if (require_packed && state.stage >= checkpoint_stage::GROUP_DONE) {
        require_size(state.packed_u.size(), item.u_size()/sizeof(uint32_t), "packed_U");
        require_size(state.packed_v.size(), item.v_size()/sizeof(uint32_t), "packed_V");
    }
}

static hash256 fingerprint_model_file(const std::string & path) {
    std::error_code ec;
    const uintmax_t raw_size = std::filesystem::file_size(path, ec);
    if (ec || raw_size > uintmax_t(std::numeric_limits<uint64_t>::max())) {
        throw std::runtime_error(format(
                "NanoQuant: cannot determine the size of '%s' for identity fingerprinting",
                path.c_str()));
    }
    const auto write_time = std::filesystem::last_write_time(path, ec);
    if (ec) {
        throw std::runtime_error(format(
                "NanoQuant: cannot determine the modification time of '%s' for identity fingerprinting",
                path.c_str()));
    }

    const uint64_t file_size = uint64_t(raw_size);
    const int64_t write_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            write_time.time_since_epoch()).count();
    hash_builder hash;
    hash.update_string("llama.cpp-nanoquant-source-file-v2");
    hash.update_pod(file_size);
    hash.update_pod(write_time_ns);

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(format(
                "NanoQuant: cannot open '%s' for identity fingerprinting", path.c_str()));
    }
    static constexpr uint64_t CHUNK_SIZE = 1024u * 1024u;
    static constexpr uint64_t CHUNK_COUNT = 4;
    static constexpr uint64_t FULL_HASH_LIMIT = CHUNK_SIZE * CHUNK_COUNT;
    std::vector<char> buffer(size_t(std::min(file_size, CHUNK_SIZE)));
    if (file_size <= FULL_HASH_LIMIT) {
        buffer.resize(size_t(file_size));
        if (!buffer.empty()) {
            input.read(buffer.data(), std::streamsize(buffer.size()));
            if (input.gcount() != std::streamsize(buffer.size())) {
                throw std::runtime_error(format(
                        "NanoQuant: failed while fingerprinting '%s'", path.c_str()));
            }
            hash.update(buffer.data(), buffer.size());
        }
    } else {
        const uint64_t last_offset = file_size - CHUNK_SIZE;
        for (uint64_t chunk = 0; chunk < CHUNK_COUNT; ++chunk) {
            const uint64_t offset =
                    (last_offset / (CHUNK_COUNT - 1))*chunk +
                    ((last_offset % (CHUNK_COUNT - 1))*chunk)/(CHUNK_COUNT - 1);
            input.clear();
            input.seekg(static_cast<std::streamoff>(offset));
            input.read(buffer.data(), std::streamsize(buffer.size()));
            if (input.gcount() != std::streamsize(buffer.size())) {
                throw std::runtime_error(format(
                        "NanoQuant: failed while fingerprinting '%s'", path.c_str()));
            }
            hash.update_pod(offset);
            hash.update(buffer.data(), buffer.size());
        }
    }

    const uintmax_t final_size = std::filesystem::file_size(path, ec);
    if (ec) {
        throw std::runtime_error(format(
                "NanoQuant: cannot recheck '%s' after identity fingerprinting", path.c_str()));
    }
    const auto final_write_time = std::filesystem::last_write_time(path, ec);
    if (ec || final_size != raw_size || final_write_time != write_time) {
        throw std::runtime_error(format(
                "NanoQuant: source model file '%s' changed while it was being fingerprinted",
                path.c_str()));
    }
    return hash.value;
}

static hash256 hash_model_files(
        const std::string & input_path,
        const std::vector<std::string> & splits) {
    hash_builder hash;
    hash.update_string("llama.cpp-nanoquant-source-v2");
    const uint64_t file_count = splits.empty() ? 1 : splits.size();
    hash.update_pod(file_count);
    if (splits.empty()) {
        const hash256 digest = fingerprint_model_file(input_path);
        hash.update(digest.data(), sizeof(digest));
    } else {
        for (const std::string & split : splits) {
            const hash256 digest = fingerprint_model_file(split);
            hash.update(digest.data(), sizeof(digest));
        }
    }
    return hash.value;
}
static hash256 make_config_hash(
        const llama_model_quantize_params * params,
        int effective_nthread,
        const hash256 & dataset_hash,
        const std::vector<group> & groups) {
    hash_builder hash;
    hash.update_string("llama.cpp-native-nanoquant-v13-experts");
    hash.update(dataset_hash.data(), sizeof(dataset_hash));
    hash.update_pod(effective_nthread);
    hash.update_pod(params->nanoquant_sequence_length);
    hash.update_pod(params->nanoquant_sample_count);
    hash.update_pod(params->nanoquant_n_gpu_layers);
    hash.update_string(params->nanoquant_calibration_column != nullptr ?
            params->nanoquant_calibration_column : "");
    hash.update_string(params->nanoquant_device != nullptr ?
            params->nanoquant_device : "");
    hash.update_pod(params->nanoquant_target_bits);
    hash.update_pod(params->nanoquant_admm_outer_iterations);
    hash.update_pod(params->nanoquant_admm_inner_iterations);
    hash.update_pod(params->nanoquant_nonfactor_epochs);
    hash.update_pod(params->nanoquant_factor_epochs);
    hash.update_pod(params->nanoquant_model_epochs);
    hash.update_pod(params->nanoquant_nonfactor_learning_rate);
    hash.update_pod(params->nanoquant_factor_learning_rate);
    hash.update_pod(params->nanoquant_model_learning_rate);
    hash.update_pod(params->nanoquant_seed);
    hash.update_pod(params->allow_requantize);
    if (params->kv_overrides != nullptr) {
        for (const llama_model_kv_override * override = params->kv_overrides;
             override->key[0] != '\0';
             ++override) {
            hash.update_string(override->key);
            hash.update_pod(override->tag);
            switch (override->tag) {
                case LLAMA_KV_OVERRIDE_TYPE_INT:
                    hash.update_pod(override->val_i64);
                    break;
                case LLAMA_KV_OVERRIDE_TYPE_FLOAT:
                    hash.update_pod(override->val_f64);
                    break;
                case LLAMA_KV_OVERRIDE_TYPE_BOOL:
                    hash.update_pod(override->val_bool);
                    break;
                case LLAMA_KV_OVERRIDE_TYPE_STR:
                    hash.update_string(override->val_str);
                    break;
            }
        }
    }
    for (const group & item : groups) {
        hash.update_string(item.name);
        hash.update_pod(item.n_in);
        hash.update_pod(item.n_out);
        hash.update_pod(item.n_expert);
        hash.update_pod(item.rank);
    }
    return hash.value;
}

static void write_manifest(
        const std::filesystem::path & directory,
        const hash256 & source_hash,
        const hash256 & dataset_hash,
        const hash256 & config_hash,
        const std::vector<group> & groups) {
    const auto path = directory / "manifest.nq";
    atomic_replace(path, [&](std::ostream & output) {
        static const char magic[8] = { 'N', 'Q', 'M', 'A', 'N', '1', '\0', '\0' };
        output.write(magic, sizeof(magic));
        write_pod(output, CHECKPOINT_VERSION);
        write_pod(output, UINT32_C(0x01020304));
        output.write(reinterpret_cast<const char *>(source_hash.data()), sizeof(source_hash));
        output.write(reinterpret_cast<const char *>(dataset_hash.data()), sizeof(dataset_hash));
        output.write(reinterpret_cast<const char *>(config_hash.data()), sizeof(config_hash));
        write_pod(output, uint64_t(groups.size()));
        for (const group & item : groups) {
            write_string(output, item.name);
            write_pod(output, item.n_in);
            write_pod(output, item.n_out);
            write_pod(output, item.n_expert);
            write_pod(output, item.rank);
        }
    });
}

static void validate_manifest(
        const std::filesystem::path & directory,
        const hash256 & source_hash,
        const hash256 & dataset_hash,
        const hash256 & config_hash,
        const std::vector<group> & groups) {
    const auto path = directory / "manifest.nq";
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error(format("NanoQuant: --nanoquant-resume requires '%s'", path.string().c_str()));
    }
    std::ifstream input(path, std::ios::binary);
    input.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    char magic[8];
    input.read(magic, sizeof(magic));
    static const char expected_magic[8] = { 'N', 'Q', 'M', 'A', 'N', '1', '\0', '\0' };
    if (std::memcmp(magic, expected_magic, sizeof(magic)) != 0 ||
        read_pod<uint32_t>(input) != CHECKPOINT_VERSION ||
        read_pod<uint32_t>(input) != UINT32_C(0x01020304)) {
        throw std::runtime_error(format("NanoQuant: invalid checkpoint manifest '%s'", path.string().c_str()));
    }
    hash256 stored_source;
    hash256 stored_dataset;
    hash256 stored_config;
    input.read(reinterpret_cast<char *>(stored_source.data()), sizeof(stored_source));
    input.read(reinterpret_cast<char *>(stored_dataset.data()), sizeof(stored_dataset));
    input.read(reinterpret_cast<char *>(stored_config.data()), sizeof(stored_config));
    if (stored_source != source_hash) {
        throw std::runtime_error("NanoQuant: checkpoint source model hash mismatch");
    }
    if (stored_dataset != dataset_hash) {
        throw std::runtime_error("NanoQuant: checkpoint calibration dataset hash mismatch");
    }
    if (stored_config != config_hash) {
        throw std::runtime_error("NanoQuant: checkpoint configuration hash mismatch");
    }
    const uint64_t count = read_pod<uint64_t>(input);
    if (count != groups.size()) {
        throw std::runtime_error("NanoQuant: checkpoint group count mismatch");
    }
    for (const group & item : groups) {
        const std::string name = read_string(input);
        const int64_t n_in = read_pod<int64_t>(input);
        const int64_t n_out = read_pod<int64_t>(input);
        const int64_t n_expert = read_pod<int64_t>(input);
        const int64_t rank = read_pod<int64_t>(input);
        if (name != item.name || n_in != item.n_in || n_out != item.n_out ||
            n_expert != item.n_expert || rank != item.rank) {
            throw std::runtime_error(format("NanoQuant: checkpoint group mismatch at '%s'", item.name.c_str()));
        }
    }
}

struct calibration_data {
    int64_t rows = 0;
    bool repeated_use = false;
    std::vector<float> inputs;
    std::vector<float> teacher_outputs;
    std::vector<float> nonfactor_outputs;
    std::vector<float> input_norm;
    std::vector<float> output_norm;
};

static std::string graph_weight_name(const char * name) {
    if (name == nullptr) {
        return {};
    }
    const char * begin = std::strchr(name, '#');
    if (begin == nullptr) {
        return name;
    }
    ++begin;
    const char * end = std::strchr(begin, '#');
    return end == nullptr ? std::string(begin) : std::string(begin, end);
}

struct projection_reachability {
    const std::unordered_map<std::string, size_t> & group_by_name;
    std::vector<bool> & reachable;

    bool collect(ggml_tensor * tensor, bool ask) {
        if (!ask || tensor == nullptr ||
            (tensor->op != GGML_OP_MUL_MAT && tensor->op != GGML_OP_MUL_MAT_ID) ||
            tensor->src[0] == nullptr) {
            return false;
        }
        const auto found = group_by_name.find(graph_weight_name(tensor->src[0]->name));
        if (found != group_by_name.end()) {
            reachable[found->second] = true;
        }
        return false;
    }
};

static bool projection_reachability_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    return static_cast<projection_reachability *>(user_data)->collect(tensor, ask);
}

struct calibration_collector {
    std::mutex mutex;
    const group * target = nullptr;
    int64_t expert = 0;
    int64_t expected_rows = 0;
    int64_t observed_rows = 0;
    size_t capacity = 0;
    size_t selection_cursor = 0;
    uint64_t reservoir_seed = 0;
    bool captured_evaluation = false;
    bool repeated_use = false;
    std::vector<int64_t> selected_rows;
    std::vector<float> inputs;
    std::vector<float> outputs;
    std::vector<double> input_squares;
    std::vector<double> output_squares;
    double input_clip = 0.0;
    std::vector<float> output_row_norms;
    std::string error;

    bool routed() const {
        return target != nullptr && target->n_expert > 1;
    }

    static uint64_t mix64(uint64_t value) {
        value += UINT64_C(0x9e3779b97f4a7c15);
        value = (value ^ (value >> 30))*UINT64_C(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27))*UINT64_C(0x94d049bb133111eb);
        return value ^ (value >> 31);
    }

    void reset(
            const group & item,
            int64_t expert_index,
            int64_t total_rows,
            size_t memory_budget) {
        std::lock_guard<std::mutex> lock(mutex);
        target = &item;
        expert = expert_index;
        expected_rows = item.n_expert > 1 ? -1 : total_rows;
        observed_rows = 0;
        selection_cursor = 0;
        captured_evaluation = false;
        repeated_use = false;
        input_clip = 0.0;
        error.clear();
        const size_t bytes_per_row = sizeof(float)*size_t(item.n_in + item.n_out);
        capacity = size_t(std::max<int64_t>(1, std::min<int64_t>(
                total_rows, int64_t(memory_budget/std::max<size_t>(bytes_per_row, 1)))));
        selected_rows.clear();
        if (!routed()) {
            selected_rows.resize(capacity);
            for (size_t i = 0; i < capacity; ++i) {
                selected_rows[i] = ((2*int64_t(i) + 1)*total_rows)/(2*int64_t(capacity));
            }
        }
        reservoir_seed = UINT64_C(1469598103934665603) ^ uint64_t(expert);
        for (unsigned char value : item.name) {
            reservoir_seed = (reservoir_seed ^ value)*UINT64_C(1099511628211);
        }
        inputs.clear();
        outputs.clear();
        inputs.reserve(capacity*size_t(item.n_in));
        outputs.reserve(capacity*size_t(item.n_out));
        input_squares.assign(item.n_in, 0.0);
        output_squares.assign(item.n_out, 0.0);
        output_row_norms.clear();
        if (!routed()) {
            output_row_norms.reserve(total_rows);
        }
    }

    void begin_evaluation() {
        std::lock_guard<std::mutex> lock(mutex);
        captured_evaluation = false;
    }

    bool matches(const ggml_tensor * tensor) const {
        if (target == nullptr || tensor == nullptr || tensor->src[0] == nullptr) {
            return false;
        }
        const ggml_op expected_op = routed() ? GGML_OP_MUL_MAT_ID : GGML_OP_MUL_MAT;
        return tensor->op == expected_op &&
               graph_weight_name(tensor->src[0]->name) == target->name;
    }

    static const uint8_t * tensor_bytes(
            const ggml_tensor * tensor,
            std::vector<uint8_t> & staging) {
        if (tensor->buffer == nullptr || ggml_backend_buffer_is_host(tensor->buffer)) {
            return static_cast<const uint8_t *>(tensor->data);
        }
        staging.resize(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, staging.data(), 0, staging.size());
        return staging.data();
    }

    bool collect_data(
            ggml_tensor * tensor,
            const uint8_t * input_data,
            const uint8_t * output_data,
            const uint8_t * ids_data) {
        std::lock_guard<std::mutex> lock(mutex);
        if (captured_evaluation) {
            repeated_use = true;
            return true;
        }
        captured_evaluation = true;
        const ggml_tensor * src = tensor->src[1];
        if (src == nullptr || src->type != GGML_TYPE_F32 || tensor->type != GGML_TYPE_F32) {
            error = format("projection '%s' did not expose F32 input/output activations", target->name.c_str());
            return false;
        }
        if (src->ne[0] != target->n_in || tensor->ne[0] != target->n_out ||
            src->nb[0] != sizeof(float) || tensor->nb[0] != sizeof(float)) {
            error = format("projection shape mismatch for '%s'", target->name.c_str());
            return false;
        }

        std::vector<const float *> x_rows;
        std::vector<const float *> y_rows;
        if (!routed()) {
            if (src->ne[1] != tensor->ne[1] || src->ne[2] != tensor->ne[2] ||
                src->ne[3] != tensor->ne[3]) {
                error = format("projection shape mismatch for '%s'", target->name.c_str());
                return false;
            }
            const int64_t n_rows = src->ne[1]*src->ne[2]*src->ne[3];
            x_rows.reserve(n_rows);
            y_rows.reserve(n_rows);
            for (int64_t row = 0; row < n_rows; ++row) {
                const int64_t i1 = row % src->ne[1];
                const int64_t i2 = (row/src->ne[1]) % src->ne[2];
                const int64_t i3 = row/(src->ne[1]*src->ne[2]);
                x_rows.push_back(reinterpret_cast<const float *>(
                        input_data + i1*src->nb[1] + i2*src->nb[2] + i3*src->nb[3]));
                y_rows.push_back(reinterpret_cast<const float *>(
                        output_data + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3]));
            }
        } else {
            const ggml_tensor * ids = tensor->src[2];
            if (ids == nullptr || ids->type != GGML_TYPE_I32 || ids_data == nullptr ||
                ids->ne[0] != tensor->ne[1] || ids->ne[1] != tensor->ne[2] ||
                ids->ne[2] != 1 || ids->ne[3] != 1 ||
                src->ne[2] != ids->ne[1] || ids->ne[0] % src->ne[1] != 0) {
                error = format("routed projection shape mismatch for '%s'", target->name.c_str());
                return false;
            }
            x_rows.reserve(ids->ne[1]);
            y_rows.reserve(ids->ne[1]);
            for (int64_t token = 0; token < ids->ne[1]; ++token) {
                for (int64_t slot = 0; slot < ids->ne[0]; ++slot) {
                    const int32_t selected = *reinterpret_cast<const int32_t *>(
                            ids_data + slot*ids->nb[0] + token*ids->nb[1]);
                    if (selected < 0 || selected >= target->n_expert) {
                        error = format("projection '%s' selected invalid expert %d",
                                target->name.c_str(), selected);
                        return false;
                    }
                    if (selected != expert) {
                        continue;
                    }
                    x_rows.push_back(reinterpret_cast<const float *>(
                            input_data + (slot % src->ne[1])*src->nb[1] + token*src->nb[2]));
                    y_rows.push_back(reinterpret_cast<const float *>(
                            output_data + slot*tensor->nb[1] + token*tensor->nb[2]));
                }
            }
        }
        if (x_rows.empty()) {
            return true;
        }

        std::vector<double> input_norms(x_rows.size(), 0.0);
        for (size_t row = 0; row < x_rows.size(); ++row) {
            const float * x = x_rows[row];
            const float * y = y_rows[row];
            double input_norm_sq = 0.0;
            for (int64_t j = 0; j < target->n_in; ++j) {
                if (!std::isfinite(x[j])) {
                    error = format("non-finite calibration input for '%s'", target->name.c_str());
                    return false;
                }
                input_norm_sq += double(x[j])*double(x[j]);
            }
            input_norms[row] = std::sqrt(input_norm_sq);
            double output_norm_sq = 0.0;
            for (int64_t j = 0; j < target->n_out; ++j) {
                if (!std::isfinite(y[j])) {
                    error = format("non-finite teacher output for '%s'", target->name.c_str());
                    return false;
                }
                const double square = double(y[j])*double(y[j]);
                output_squares[j] += square;
                output_norm_sq += square;
            }
            output_row_norms.push_back(float(std::sqrt(output_norm_sq)));
        }

        std::vector<double> sorted_input_norms = input_norms;
        const size_t kth_largest = std::max<size_t>(
                1, size_t(double(sorted_input_norms.size())*(1.0 - 0.999)));
        const auto clip_it =
                sorted_input_norms.begin() + sorted_input_norms.size() - kth_largest;
        std::nth_element(sorted_input_norms.begin(), clip_it, sorted_input_norms.end());
        const double batch_clip = *clip_it;
        if (input_clip == 0.0) {
            input_clip = batch_clip;
        } else if (batch_clip > input_clip) {
            const double correction = batch_clip*batch_clip/(input_clip*input_clip);
            for (double & value : input_squares) {
                value *= correction;
            }
            input_clip = batch_clip;
        }

        for (size_t row = 0; row < x_rows.size(); ++row) {
            if (expected_rows >= 0 && observed_rows >= expected_rows) {
                error = format("projection '%s' was evaluated more than once per calibration token", target->name.c_str());
                return false;
            }
            const float * x = x_rows[row];
            const float * y = y_rows[row];
            const double input_scale =
                    input_norms[row] > input_clip && input_norms[row] > 0.0 ?
                    input_clip/input_norms[row] : 1.0;
            for (int64_t j = 0; j < target->n_in; ++j) {
                const double value = double(x[j])*input_scale;
                input_squares[j] += value*value;
            }

            size_t selected = capacity;
            if (!routed()) {
                if (selection_cursor < selected_rows.size() &&
                    observed_rows == selected_rows[selection_cursor]) {
                    selected = selection_cursor++;
                }
            } else if (size_t(observed_rows) < capacity) {
                selected = size_t(observed_rows);
            } else {
                const size_t candidate =
                        size_t(mix64(reservoir_seed ^ uint64_t(observed_rows)) %
                               uint64_t(observed_rows + 1));
                if (candidate < capacity) {
                    selected = candidate;
                }
            }
            if (selected < capacity) {
                if (selected*size_t(target->n_in) == inputs.size()) {
                    inputs.insert(inputs.end(), x, x + target->n_in);
                    outputs.insert(outputs.end(), y, y + target->n_out);
                } else {
                    std::copy_n(x, target->n_in,
                            inputs.begin() + selected*size_t(target->n_in));
                    std::copy_n(y, target->n_out,
                            outputs.begin() + selected*size_t(target->n_out));
                }
            }
            ++observed_rows;
        }
        return true;
    }

    calibration_data finish() {
        std::lock_guard<std::mutex> lock(mutex);
        if (!error.empty()) {
            throw std::runtime_error("NanoQuant: " + error);
        }
        if (observed_rows == 0) {
            if (!routed()) {
                throw std::runtime_error(format(
                        "NanoQuant: projection '%s' received no calibration rows",
                        target->name.c_str()));
            }
            LLAMA_LOG_WARN(
                    "NanoQuant: projection '%s' expert %" PRId64
                    " was not routed by calibration data; using uniform factorization weights\n",
                    target->name.c_str(), expert);
            calibration_data result;
            result.repeated_use = true;
            result.input_norm.assign(size_t(target->n_in), 1.0f);
            result.output_norm.assign(size_t(target->n_out), 1.0f);
            return result;
        }
        if (expected_rows >= 0 && observed_rows != expected_rows) {
            throw std::runtime_error(format(
                    "NanoQuant: projection '%s' produced %" PRId64 " rows, expected %" PRId64,
                    target->name.c_str(), observed_rows, expected_rows));
        }
        if (!routed() && selection_cursor != selected_rows.size()) {
            throw std::runtime_error(format(
                    "NanoQuant: incomplete deterministic projection sample for '%s'",
                    target->name.c_str()));
        }

        calibration_data result;
        result.rows = int64_t(inputs.size()/size_t(target->n_in));
        if (result.rows <= 0 ||
            inputs.size() != size_t(result.rows)*size_t(target->n_in) ||
            outputs.size() != size_t(result.rows)*size_t(target->n_out)) {
            throw std::runtime_error(format(
                    "NanoQuant: invalid routed calibration sample for '%s'", target->name.c_str()));
        }
        result.repeated_use = repeated_use;
        result.inputs = std::move(inputs);
        result.teacher_outputs = std::move(outputs);
        result.nonfactor_outputs = result.teacher_outputs;
        result.input_norm.resize(target->n_in);
        result.output_norm.resize(target->n_out);
        for (int64_t i = 0; i < target->n_in; ++i) {
            result.input_norm[i] = float(input_squares[i]/double(observed_rows));
        }
        for (int64_t i = 0; i < target->n_out; ++i) {
            result.output_norm[i] = float(output_squares[i]/double(observed_rows));
        }
        auto shrink = [](std::vector<float> & values) {
            const double mean = std::accumulate(values.begin(), values.end(), 0.0)/double(values.size());
            for (float & value : values) {
                value = std::max(NUMERIC_EPSILON,
                        (1.0f - CALIBRATION_SHRINKAGE)*value +
                        CALIBRATION_SHRINKAGE*float(mean));
            }
        };
        shrink(result.input_norm);
        shrink(result.output_norm);

        const double position = 0.999*double(output_row_norms.size() - 1);
        const size_t lower = size_t(position);
        const size_t upper = std::min(lower + 1, output_row_norms.size() - 1);
        auto lower_it = output_row_norms.begin() + lower;
        std::nth_element(output_row_norms.begin(), lower_it, output_row_norms.end());
        const float lower_value = *lower_it;
        const float upper_value = upper == lower ? lower_value :
                *std::min_element(lower_it + 1, output_row_norms.end());
        const float tau = lower_value +
                float(position - double(lower))*(upper_value - lower_value);
        for (int64_t row = 0; row < result.rows; ++row) {
            float * y = result.nonfactor_outputs.data() +
                    size_t(row)*size_t(target->n_out);
            double norm_sq = 0.0;
            for (int64_t j = 0; j < target->n_out; ++j) {
                norm_sq += double(y[j])*double(y[j]);
            }
            const float norm = float(std::sqrt(norm_sq));
            if (norm > tau && norm > 0.0f) {
                const float scale = tau/norm;
                for (int64_t j = 0; j < target->n_out; ++j) {
                    y[j] *= scale;
                }
            }
        }
        return result;
    }
};

struct calibration_collector_set {
    std::mutex mutex;
    std::vector<calibration_collector *> active;
    std::vector<uint8_t> input_staging;
    std::vector<uint8_t> output_staging;
    std::vector<uint8_t> ids_staging;

    bool collect(ggml_tensor * tensor, bool ask) {
        std::vector<calibration_collector *> matches;
        for (calibration_collector * collector : active) {
            if (collector->matches(tensor)) {
                matches.push_back(collector);
            }
        }
        if (matches.empty()) {
            return false;
        }
        if (ask) {
            return true;
        }

        std::lock_guard<std::mutex> lock(mutex);
        const uint8_t * input_data =
                calibration_collector::tensor_bytes(tensor->src[1], input_staging);
        const uint8_t * output_data =
                calibration_collector::tensor_bytes(tensor, output_staging);
        const uint8_t * ids_data = tensor->op == GGML_OP_MUL_MAT_ID ?
                calibration_collector::tensor_bytes(tensor->src[2], ids_staging) : nullptr;
        for (calibration_collector * collector : matches) {
            collector->collect_data(tensor, input_data, output_data, ids_data);
        }
        return true;
    }
};

static bool calibration_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    return static_cast<calibration_collector_set *>(user_data)->collect(tensor, ask);
}

struct block_output_collector {
    std::mutex mutex;
    int block = -1;
    std::vector<float> values;
    std::vector<uint8_t> staging;
    bool captured = false;
    std::string target_name;
    std::string error;

    void reset(int block_index) {
        std::lock_guard<std::mutex> lock(mutex);
        block = block_index;
        values.clear();
        error.clear();
        captured = false;
        target_name = format("l_out-%d", block);
    }

    bool collect(ggml_tensor * tensor, bool ask) {
        if (tensor == nullptr || target_name != ggml_get_name(tensor)) {
            return false;
        }
        if (ask) {
            return true;
        }
        std::lock_guard<std::mutex> lock(mutex);
        if (captured) {
            error = "block boundary was evaluated more than once";
            return false;
        }
        captured = true;
        if (tensor->type != GGML_TYPE_F32 || tensor->nb[0] != sizeof(float)) {
            error = "block boundary is not contiguous F32";
            return false;
        }
        const uint8_t * data = calibration_collector::tensor_bytes(tensor, staging);
        const int64_t rows = tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
        values.reserve(size_t(rows) * size_t(tensor->ne[0]));
        for (int64_t row = 0; row < rows; ++row) {
            const int64_t i1 = row % tensor->ne[1];
            const int64_t i2 = (row / tensor->ne[1]) % tensor->ne[2];
            const int64_t i3 = row / (tensor->ne[1] * tensor->ne[2]);
            const float * source = reinterpret_cast<const float *>(
                    data + i1 * tensor->nb[1] + i2 * tensor->nb[2] + i3 * tensor->nb[3]);
            values.insert(values.end(), source, source + tensor->ne[0]);
        }
        return true;
    }

    std::vector<float> finish() {
        std::lock_guard<std::mutex> lock(mutex);
        if (!error.empty()) {
            throw std::runtime_error("NanoQuant: " + error);
        }
        if (values.empty()) {
            throw std::runtime_error(format(
                    "NanoQuant: block boundary l_out-%d was not observed", block));
        }
        return values;
    }
};

static bool block_output_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    return static_cast<block_output_collector *>(user_data)->collect(tensor, ask);
}

static size_t calibration_sample_stride(const llama_model_quantize_params * params) {
    return size_t(params->nanoquant_sequence_length) + 1;
}

static const llama_token * calibration_sample(
        const std::vector<llama_token> & samples,
        int32_t sample,
        const llama_model_quantize_params * params) {
    return samples.data() + size_t(sample)*calibration_sample_stride(params);
}


static std::string load_calibration_text(
        const llama_model_quantize_params * params) {
    const std::filesystem::path path(params->nanoquant_calibration_dataset);
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
            [](unsigned char value) { return char(std::tolower(value)); });
    if (extension == ".parquet") {
        return llama_parquet_load_text(
                params->nanoquant_calibration_dataset,
                params->nanoquant_calibration_column);
    }

    std::ifstream input(params->nanoquant_calibration_dataset, std::ios::binary);
    if (!input) {
        throw std::runtime_error(format("NanoQuant: cannot open calibration dataset '%s'",
                params->nanoquant_calibration_dataset));
    }
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    if (size <= 0 || size > INT32_MAX) {
        throw std::runtime_error(
                "NanoQuant: calibration dataset must contain 1..INT32_MAX bytes of text");
    }
    input.seekg(0, std::ios::beg);
    std::string text(size_t(size), '\0');
    input.read(text.data(), text.size());
    if (!input) {
        throw std::runtime_error(format(
                "NanoQuant: failed to read calibration dataset '%s'",
                params->nanoquant_calibration_dataset));
    }
    return text;
}

static std::vector<llama_token> make_calibration_samples(
        const llama_model * model,
        const llama_model_quantize_params * params) {
    std::string text = load_calibration_text(params);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    int32_t count = llama_tokenize(vocab, text.data(), int32_t(text.size()), nullptr, 0, true, false);
    if (count == INT32_MIN || count == 0) {
        throw std::runtime_error("NanoQuant: calibration dataset tokenization failed");
    }
    std::vector<llama_token> tokens(size_t(count < 0 ? -count : count));
    count = llama_tokenize(vocab, text.data(), int32_t(text.size()), tokens.data(), tokens.size(), true, false);
    if (count <= 0) {
        throw std::runtime_error("NanoQuant: calibration dataset tokenization failed");
    }
    tokens.resize(count);

    const int64_t sequence_length = params->nanoquant_sequence_length;
    const size_t sample_stride = calibration_sample_stride(params);
    if (int64_t(tokens.size()) < sequence_length + 1) {
        throw std::runtime_error(format(
                "NanoQuant: calibration dataset has %zu tokens, fewer than required %d",
                tokens.size(), params->nanoquant_sequence_length + 1));
    }
    const size_t sample_tokens = size_t(params->nanoquant_sample_count)*sample_stride;
    if (sample_tokens/sample_stride != size_t(params->nanoquant_sample_count)) {
        throw std::runtime_error("NanoQuant: calibration sample size overflow");
    }
    std::vector<llama_token> samples(sample_tokens);
    deterministic_rng rng(params->nanoquant_seed);
    const uint64_t n_starts = tokens.size() - sample_stride + 1;
    for (int32_t sample = 0; sample < params->nanoquant_sample_count; ++sample) {
        const size_t start = size_t(rng.next_u64() % n_starts);
        std::copy_n(tokens.data() + start, sample_stride,
                samples.data() + size_t(sample)*sample_stride);
    }
    return samples;
}

static double evaluate_block_loss(
        llama_context * teacher,
        llama_context * student,
        block_output_collector & teacher_collector,
        block_output_collector & student_collector,
        int block,
        const std::vector<llama_token> & samples,
        const llama_model_quantize_params * params) {
    struct batch_owner {
        llama_batch value;
        explicit batch_owner(int32_t n_tokens) :
                value(llama_batch_init(n_tokens, 0, 1)) {}
        ~batch_owner() {
            llama_batch_free(value);
        }
    } owned(params->nanoquant_sequence_length);
    llama_batch & batch = owned.value;
    size_t width = 0;
    int64_t row_count = 0;
    std::vector<double> target_sq;
    std::vector<double> error_sq;
    for (int32_t sample = 0; sample < params->nanoquant_sample_count; ++sample) {
        batch.n_tokens = params->nanoquant_sequence_length;
        const llama_token * sample_tokens =
                calibration_sample(samples, sample, params);
        for (int32_t token = 0; token < batch.n_tokens; ++token) {
            batch.token[token] = sample_tokens[token];
            batch.pos[token] = token;
            batch.n_seq_id[token] = 1;
            batch.seq_id[token][0] = 0;
            batch.logits[token] = 1;
        }
        teacher_collector.reset(block);
        student_collector.reset(block);
        llama_memory_clear(llama_get_memory(teacher), true);
        llama_memory_clear(llama_get_memory(student), true);
        const int teacher_result = llama_decode(teacher, batch);
        const int student_result = llama_decode(student, batch);
        if (teacher_result != 0 || student_result != 0) {
            throw std::runtime_error(format(
                    "NanoQuant: block %d evaluation failed at sample %d (teacher=%d, student=%d)",
                    block, sample, teacher_result, student_result));
        }
        llama_synchronize(teacher);
        llama_synchronize(student);
        const std::vector<float> target = teacher_collector.finish();
        const std::vector<float> prediction = student_collector.finish();
        if (target.size() != prediction.size() ||
            target.size() % size_t(params->nanoquant_sequence_length) != 0) {
            throw std::runtime_error(format(
                    "NanoQuant: block %d boundary shape mismatch", block));
        }
        const size_t sample_width =
                target.size() / size_t(params->nanoquant_sequence_length);
        if (width == 0) {
            width = sample_width;
            target_sq.assign(width, 0.0);
            error_sq.assign(width, 0.0);
        } else if (sample_width != width) {
            throw std::runtime_error(format(
                    "NanoQuant: block %d boundary width changed between samples", block));
        }
        for (size_t i = 0; i < target.size(); ++i) {
            const size_t channel = i % width;
            const double target_value = target[i];
            const double difference = double(prediction[i]) - target_value;
            target_sq[channel] += target_value * target_value;
            error_sq[channel] += difference * difference;
        }
        row_count += params->nanoquant_sequence_length;
    }
    if (width == 0 || row_count == 0) {
        throw std::runtime_error("NanoQuant: block reconstruction observed no outputs");
    }
    std::vector<double> output_norm(width);
    double norm_mean = 0.0;
    for (size_t channel = 0; channel < width; ++channel) {
        output_norm[channel] = target_sq[channel] / double(row_count);
        norm_mean += output_norm[channel];
    }
    norm_mean /= double(width);
    const double denominator = std::max(norm_mean, double(NUMERIC_EPSILON));
    double weighted_error = 0.0;
    for (size_t channel = 0; channel < width; ++channel) {
        const double shrunk_norm =
                (1.0 - CALIBRATION_SHRINKAGE) * output_norm[channel] +
                CALIBRATION_SHRINKAGE * norm_mean;
        weighted_error += (shrunk_norm / denominator) * error_sq[channel];
    }
    const double result = weighted_error / (double(row_count) * double(width));
    if (!std::isfinite(result)) {
        throw std::runtime_error("NanoQuant: block reconstruction loss is non-finite");
    }
    return result;
}

static std::vector<std::vector<float>> collect_output_importance(
        llama_context * context,
        const std::vector<group> & groups,
        const std::vector<size_t> & target_indices,
        std::vector<bool> & reachable,
        const std::vector<llama_token> & samples,
        size_t gradient_memory_budget,
        const llama_model_quantize_params * params,
        llama_batch & batch) {
    LLAMA_LOG_INFO(
            "NanoQuant: task-gradient batching budget = %zu bytes\n",
            gradient_memory_budget);
    std::vector<std::vector<float>> result(groups.size());
    size_t cursor = 0;
    while (cursor < target_indices.size()) {
        const size_t batch_begin = cursor;
        size_t gradient_bytes = 0;
        while (cursor < target_indices.size()) {
            const group & item = groups[target_indices[cursor]];
            const size_t item_bytes =
                    size_t(params->nanoquant_sequence_length) *
                    size_t(item.n_out) * size_t(item.n_expert_used) * sizeof(float);
            if (cursor != batch_begin &&
                gradient_bytes + item_bytes > gradient_memory_budget) {
                break;
            }
            gradient_bytes += item_bytes;
            ++cursor;
        }

        std::vector<std::string> targets;
        targets.reserve(cursor - batch_begin);
        for (size_t i = batch_begin; i < cursor; ++i) {
            targets.push_back(groups[target_indices[i]].name);
        }
        std::unique_ptr<llama_nanoquant_optimizer, std::function<void(llama_nanoquant_optimizer *)>>
                optimizer(
                        context->nanoquant_optimizer_init(
                                llama_nanoquant_opt_loss::CROSS_ENTROPY, -1,
                                {}, 0, targets),
                        [context](llama_nanoquant_optimizer * value) {
                            context->nanoquant_optimizer_free(value);
                        });
        const auto start = std::chrono::steady_clock::now();
        for (int32_t sample = 0; sample < params->nanoquant_sample_count; ++sample) {
            const llama_token * tokens = calibration_sample(samples, sample, params);
            context->nanoquant_optimizer_step(
                    optimizer.get(), batch, tokens,
                    params->nanoquant_sequence_length,
                    nullptr, 0, tokens + 1,
                    params->nanoquant_sequence_length,
                    nullptr, 0, 1.0f);
        }
        std::vector<std::vector<float>> batch_result =
                context->nanoquant_optimizer_output_importance(optimizer.get());
        if (batch_result.size() != cursor - batch_begin) {
            throw std::runtime_error("NanoQuant: output-gradient target count changed");
        }
        for (size_t i = batch_begin; i < cursor; ++i) {
            const size_t index = target_indices[i];
            std::vector<float> & importance = batch_result[i - batch_begin];
            if (importance.empty()) {
                result[index].assign(size_t(groups[index].n_out), 1.0f);
            } else {
                reachable[index] = true;
                result[index] = std::move(importance);
            }
        }
        LLAMA_LOG_INFO(
                "NanoQuant profile: task-gradient calibration=%.3fs targets=%zu cache=%zu bytes\n",
                std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - start).count(),
                cursor - batch_begin, gradient_bytes);
    }
    return result;
}

static std::vector<std::vector<calibration_data>> collect_projections(
        llama_context * context,
        calibration_collector_set & collector_set,
        const std::vector<group> & groups,
        const std::vector<checkpoint_state> & states,
        const std::vector<bool> & reachable,
        size_t block_begin,
        size_t block_end,
        const std::vector<llama_token> & samples,
        const std::vector<std::vector<float>> & output_importance,
        const llama_model_quantize_params * params,
        llama_batch & batch) {
    const int32_t sequence_length = params->nanoquant_sequence_length;
    const int64_t expected_rows =
            int64_t(params->nanoquant_sample_count)*int64_t(sequence_length);
    std::vector<std::vector<std::unique_ptr<calibration_collector>>> collectors(
            block_end - block_begin);
    collector_set.active.clear();
    size_t active_count = 0;
    for (size_t index = block_begin; index < block_end; ++index) {
        if (states[index].stage < checkpoint_stage::GROUP_DONE &&
            reachable[index]) {
            active_count += size_t(groups[index].n_expert - states[index].expert);
        }
    }
    collector_set.active.reserve(active_count);
    const size_t collector_budget =
            PROJECTION_MEMORY_BUDGET/std::max<size_t>(active_count, 1);
    for (size_t index = block_begin; index < block_end; ++index) {
        if (states[index].stage >= checkpoint_stage::GROUP_DONE ||
            !reachable[index]) {
            continue;
        }
        std::vector<std::unique_ptr<calibration_collector>> & group_collectors =
                collectors[index - block_begin];
        group_collectors.resize(groups[index].n_expert);
        for (int64_t expert = states[index].expert;
             expert < groups[index].n_expert;
             ++expert) {
            group_collectors[expert] = std::make_unique<calibration_collector>();
            group_collectors[expert]->reset(
                    groups[index], expert, expected_rows, collector_budget);
            collector_set.active.push_back(group_collectors[expert].get());
        }
    }

    for (int32_t sample = 0; sample < params->nanoquant_sample_count; ++sample) {
        const llama_token * tokens = calibration_sample(samples, sample, params);
        batch.n_tokens = sequence_length;
        for (int32_t token = 0; token < sequence_length; ++token) {
            batch.token[token] = tokens[token];
            batch.pos[token] = token;
            batch.n_seq_id[token] = 1;
            batch.seq_id[token][0] = 0;
            batch.logits[token] = true;
        }
        for (calibration_collector * collector : collector_set.active) {
            collector->begin_evaluation();
        }
        llama_memory_clear(llama_get_memory(context), true);
        const int decode_result = llama_decode(context, batch);
        if (decode_result != 0) {
            throw std::runtime_error(format(
                    "NanoQuant: projection evaluation failed at sample %d (code %d)",
                    sample, decode_result));
        }
    }

    collector_set.active.clear();
    std::vector<std::vector<calibration_data>> results(collectors.size());
    for (size_t index = 0; index < collectors.size(); ++index) {
        results[index].resize(collectors[index].size());
        for (size_t expert = 0; expert < collectors[index].size(); ++expert) {
            if (!collectors[index][expert]) {
                continue;
            }
            calibration_data result = collectors[index][expert]->finish();
            result.output_norm = output_importance[block_begin + index];
            if (result.output_norm.empty()) {
                throw std::runtime_error("NanoQuant: output-gradient calibration is incomplete");
            }
            const double mean = std::accumulate(
                    result.output_norm.begin(), result.output_norm.end(), 0.0)/
                    double(result.output_norm.size());
            for (float & value : result.output_norm) {
                value = std::max(NUMERIC_EPSILON,
                        (1.0f - CALIBRATION_SHRINKAGE)*value +
                        CALIBRATION_SHRINKAGE*float(mean));
            }
            results[index][expert] = std::move(result);
        }
    }
    return results;
}

struct compute_backend {
    enum class graph_op {
        MUL_MAT,
        OUT_PROD,
        SOLVE_TRI,
        SVID,
        SIGN_OUT_PROD,
    };

    struct cached_graph {
        graph_op op;
        int64_t ne_a0;
        int64_t ne_a1;
        int64_t ne_b0;
        int64_t ne_b1;
        ggml_context * ctx = nullptr;
        ggml_backend_t backend = nullptr;
        ggml_tensor * a = nullptr;
        ggml_tensor * b = nullptr;
        ggml_tensor * output = nullptr;
        ggml_cgraph * graph = nullptr;

        ~cached_graph() {
            ggml_free(ctx);
        }
    };

    ggml_backend_t backend = nullptr;
    std::vector<ggml_backend_t> backends;
    std::vector<std::unique_ptr<cached_graph>> graphs;
    std::vector<ggml_gallocr_t> allocators;
    std::vector<cached_graph *> active_graphs;
    ggml_backend_solve_spd_t solve_spd_fn = nullptr;

    explicit compute_backend(const char * device, bool report = true) {
        backend = device != nullptr && device[0] != '\0' ?
                ggml_backend_init_by_name(device, nullptr) :
                ggml_backend_init_best();
        if (backend == nullptr && device != nullptr && device[0] != '\0') {
            throw std::runtime_error(format(
                    "NanoQuant: cannot initialize training device '%s'", device));
        }
        if (backend == nullptr) {
            throw std::runtime_error("NanoQuant: no GGML compute backend is available");
        }
        backends.push_back(backend);
        if (ggml_backend_dev_type(ggml_backend_get_device(backend)) !=
            GGML_BACKEND_DEVICE_TYPE_CPU) {
            ggml_backend_t cpu =
                    ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
            if (cpu != nullptr) {
                backends.push_back(cpu);
            }
        }
        allocators.resize(backends.size(), nullptr);
        active_graphs.resize(backends.size(), nullptr);
        ggml_backend_reg_t backend_reg = ggml_backend_dev_backend_reg(
                ggml_backend_get_device(backend));
        solve_spd_fn = (ggml_backend_solve_spd_t)
                ggml_backend_reg_get_proc_address(
                        backend_reg, "ggml_backend_solve_spd");
        if (report) {
            LLAMA_LOG_INFO("NanoQuant: primary training backend = %s%s\n",
                    ggml_backend_name(backend),
                    backends.size() > 1 ? " (CPU fallback enabled)" : "");
            if (solve_spd_fn != nullptr) {
                LLAMA_LOG_INFO("NanoQuant: accelerated SPD solver enabled\n");
            }
        }
    }

    ~compute_backend() {
        reset_cache();
        for (ggml_backend_t owned : backends) {
            ggml_backend_free(owned);
        }
    }

    compute_backend(const compute_backend &) = delete;
    compute_backend & operator=(const compute_backend &) = delete;

    void reset_cache() {
        std::fill(active_graphs.begin(), active_graphs.end(), nullptr);
        graphs.clear();
        for (ggml_gallocr_t & allocator : allocators) {
            ggml_gallocr_free(allocator);
            allocator = nullptr;
        }
    }
    ggml_backend_buffer_type_t training_buffer_type() const {
        return ggml_backend_get_default_buffer_type(backend);
    }

    size_t gradient_memory_budget() const {
        static constexpr size_t GIB = size_t(1024u)*1024u*1024u;
        ggml_backend_dev_t device = ggml_backend_get_device(backend);
        if (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            return GRADIENT_MEMORY_BUDGET;
        }
        size_t free = 0;
        size_t total = 0;
        ggml_backend_dev_memory(device, &free, &total);
        const size_t reserve = std::max<size_t>(2u*GIB, total/8);
        const size_t available = free > reserve ? free - reserve : 0;
        return std::clamp(
                available/2, GRADIENT_MEMORY_BUDGET, 4u*GIB);
    }

    ggml_backend_buffer_type_t optimizer_buffer_type(
            size_t parameter_bytes,
            const char * name) const {
        ggml_backend_dev_t device = ggml_backend_get_device(backend);
        if (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            return training_buffer_type();
        }

        size_t free = 0;
        size_t total = 0;
        ggml_backend_dev_memory(device, &free, &total);
        static constexpr size_t MIB = 1024u*1024u;
        const size_t reserve = std::max<size_t>(512u*MIB, total/16);
        const size_t available = free > reserve ? free - reserve : 0;
        if (parameter_bytes <= available/5) {
            return training_buffer_type();
        }

        ggml_backend_buffer_type_t host = ggml_backend_dev_host_buffer_type(device);
        if (host == nullptr) {
            for (ggml_backend_t candidate : backends) {
                if (ggml_backend_dev_type(ggml_backend_get_device(candidate)) ==
                    GGML_BACKEND_DEVICE_TYPE_CPU) {
                    host = ggml_backend_get_default_buffer_type(candidate);
                    break;
                }
            }
        }
        if (host == nullptr) {
            throw std::runtime_error(format(
                    "NanoQuant: insufficient device memory for '%s' and no host fallback is available",
                    name));
        }
        LLAMA_LOG_INFO(
                "NanoQuant: %s optimizer uses host-backed parameters "
                "(parameters=%.2f MiB free=%.2f MiB reserve=%.2f MiB)\n",
                name, parameter_bytes/double(MIB), free/double(MIB), reserve/double(MIB));
        return host;
    }

    cached_graph & get_graph(
            graph_op op,
            int64_t ne_a0,
            int64_t ne_a1,
            int64_t ne_b0,
            int64_t ne_b1) {
        for (const auto & graph : graphs) {
            if (graph->op == op &&
                graph->ne_a0 == ne_a0 && graph->ne_a1 == ne_a1 &&
                graph->ne_b0 == ne_b0 && graph->ne_b1 == ne_b1) {
                return *graph;
            }
        }

        auto result = std::make_unique<cached_graph>();
        result->op = op;
        result->ne_a0 = ne_a0;
        result->ne_a1 = ne_a1;
        result->ne_b0 = ne_b0;
        result->ne_b1 = ne_b1;
        const size_t graph_size = op == graph_op::SVID ?
                32 + 16*size_t(ne_b0) : 16;
        ggml_init_params init = {
            graph_size*ggml_tensor_overhead() +
                    ggml_graph_overhead_custom(graph_size, false),
            nullptr,
            true,
        };
        result->ctx = ggml_init(init);
        if (result->ctx == nullptr) {
            throw std::runtime_error("NanoQuant: failed to allocate backend graph metadata");
        }
        result->a = ggml_new_tensor_2d(result->ctx, GGML_TYPE_F32, ne_a0, ne_a1);
        if (op == graph_op::SVID) {
            result->b = ggml_new_tensor_1d(result->ctx, GGML_TYPE_F32, ne_a0);
            auto normalize = [ctx = result->ctx](ggml_tensor * tensor) {
                ggml_tensor * norm =
                        ggml_sqrt(ctx, ggml_sum(ctx, ggml_sqr(ctx, tensor)));
                norm = ggml_clamp(
                        ctx, norm, NUMERIC_EPSILON,
                        std::numeric_limits<float>::infinity());
                return ggml_div(ctx, tensor, norm);
            };
            ggml_tensor * absolute = ggml_abs(result->ctx, result->a);
            ggml_tensor * right = normalize(result->b);
            ggml_tensor * left = nullptr;
            for (int64_t iteration = 0; iteration < ne_b0; ++iteration) {
                left = normalize(ggml_mul_mat(result->ctx, absolute, right));
                right = normalize(ggml_out_prod(
                        result->ctx, absolute,
                        ggml_reshape_2d(result->ctx, left, 1, ne_a1)));
            }
            ggml_tensor * projected =
                    ggml_mul_mat(result->ctx, absolute, right);
            ggml_tensor * sigma =
                    ggml_sum(result->ctx, ggml_mul(result->ctx, left, projected));
            ggml_tensor * rank_one =
                    ggml_mul(result->ctx,
                            ggml_out_prod(result->ctx, right, left), sigma);
            ggml_tensor * sign = ggml_scale_bias(
                    result->ctx,
                    ggml_step(result->ctx,
                            ggml_scale(result->ctx, result->a, -1.0f)),
                    -2.0f, 1.0f);
            result->output = ggml_mul(result->ctx, rank_one, sign);
        } else if (op == graph_op::SIGN_OUT_PROD) {
            result->b = ggml_new_tensor_2d(
                    result->ctx, GGML_TYPE_F32, ne_b0, ne_b1);
            ggml_tensor * sign_a = ggml_scale_bias(
                    result->ctx,
                    ggml_step(result->ctx,
                            ggml_scale(result->ctx, result->a, -1.0f)),
                    -2.0f, 1.0f);
            ggml_tensor * sign_b = ggml_scale_bias(
                    result->ctx,
                    ggml_step(result->ctx,
                            ggml_scale(result->ctx, result->b, -1.0f)),
                    -2.0f, 1.0f);
            result->output = ggml_out_prod(
                    result->ctx, sign_b,
                    ggml_transpose(result->ctx, sign_a));
        } else {
            result->b = ggml_new_tensor_2d(
                    result->ctx, GGML_TYPE_F32, ne_b0, ne_b1);
            switch (op) {
                case graph_op::MUL_MAT:
                    result->output = ggml_mul_mat(result->ctx, result->a, result->b);
                    break;
                case graph_op::OUT_PROD:
                    result->output = ggml_out_prod(result->ctx, result->a, result->b);
                    break;
                case graph_op::SOLVE_TRI:
                    result->output =
                            ggml_solve_tri(result->ctx, result->a, result->b, true, true, false);
                    break;
                case graph_op::SVID:
                    GGML_ABORT("unreachable");
                case graph_op::SIGN_OUT_PROD:
                    GGML_ABORT("unreachable");
            }
        }
        ggml_set_input(result->a);
        ggml_set_input(result->b);
        ggml_set_output(result->output);
        result->graph = ggml_new_graph_custom(result->ctx, graph_size, false);
        ggml_build_forward_expand(result->graph, result->output);
        for (ggml_backend_t candidate : backends) {
            bool supported = true;
            for (int i = 0; i < ggml_graph_n_nodes(result->graph); ++i) {
                supported &= ggml_backend_supports_op(
                        candidate, ggml_graph_node(result->graph, i));
            }
            if (supported) {
                result->backend = candidate;
                break;
            }
        }
        if (result->backend == nullptr) {
            throw std::runtime_error("NanoQuant: no backend supports factorization graph");
        }
        graphs.push_back(std::move(result));
        return *graphs.back();
    }
    static void clear_tensor_allocations(cached_graph & graph) {
        for (ggml_tensor * tensor = ggml_get_first_tensor(graph.ctx);
             tensor != nullptr;
             tensor = ggml_get_next_tensor(graph.ctx, tensor)) {
            tensor->data = nullptr;
            tensor->buffer = nullptr;
        }
    }

    void allocate_graph(cached_graph & graph) {
        const auto current = std::find(backends.begin(), backends.end(), graph.backend);
        const size_t first = current == backends.end() ?
                0 : size_t(std::distance(backends.begin(), current));
        for (size_t index = first; index < backends.size(); ++index) {
            ggml_backend_t candidate = backends[index];
            bool supported = true;
            for (int node = 0; node < ggml_graph_n_nodes(graph.graph); ++node) {
                supported &= ggml_backend_supports_op(
                        candidate, ggml_graph_node(graph.graph, node));
            }
            if (!supported) {
                continue;
            }
            if (active_graphs[index] == &graph) {
                graph.backend = candidate;
                return;
            }

            clear_tensor_allocations(graph);
            if (allocators[index] == nullptr) {
                allocators[index] = ggml_gallocr_new(
                        ggml_backend_get_default_buffer_type(candidate));
            }
            if (ggml_gallocr_reserve(allocators[index], graph.graph) &&
                ggml_gallocr_alloc_graph(allocators[index], graph.graph)) {
                active_graphs[index] = &graph;
                graph.backend = candidate;
                return;
            }

            active_graphs[index] = nullptr;
            ggml_gallocr_free(allocators[index]);
            allocators[index] = nullptr;
        }
        throw std::runtime_error(
                "NanoQuant: failed to allocate factorization graph on any supported backend");
    }

    void reserve_admm(
            int64_t n_in,
            int64_t n_out,
            int64_t rank,
            int32_t inner_iterations,
            bool update_u,
            bool update_v) {
        std::vector<cached_graph *> required;
        required.reserve(6);
        if (update_u) {
            required.push_back(&get_graph(
                    graph_op::SVID, rank, n_out, inner_iterations, 0));
            required.push_back(&get_graph(
                    graph_op::MUL_MAT, n_in, rank, n_in, rank));
            required.push_back(&get_graph(
                    graph_op::MUL_MAT, n_in, rank, n_in, n_out));
        }
        if (update_v) {
            required.push_back(&get_graph(
                    graph_op::SVID, n_in, rank, inner_iterations, 0));
            required.push_back(&get_graph(
                    graph_op::OUT_PROD, rank, n_out, rank, n_out));
            required.push_back(&get_graph(
                    graph_op::OUT_PROD, rank, n_out, n_in, n_out));
        }
        for (cached_graph * graph : required) {
            allocate_graph(*graph);
        }
        if (solve_spd_fn == nullptr) {
            if (update_u) {
                allocate_graph(get_graph(
                        graph_op::SOLVE_TRI, rank, rank, n_out, rank));
            }
            if (update_v) {
                allocate_graph(get_graph(
                        graph_op::SOLVE_TRI, rank, rank, n_in, rank));
            }
        }
        for (const auto & graph : graphs) {
            clear_tensor_allocations(*graph);
        }
        std::fill(active_graphs.begin(), active_graphs.end(), nullptr);
    }


    std::vector<float> execute(
            cached_graph & graph,
            const std::vector<float> & a,
            const std::vector<float> & b,
            const char * name) {
        allocate_graph(graph);
        ggml_backend_tensor_set(graph.a, a.data(), 0, a.size()*sizeof(float));
        ggml_backend_tensor_set(graph.b, b.data(), 0, b.size()*sizeof(float));
        const ggml_status status = ggml_backend_graph_compute(graph.backend, graph.graph);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(format(
                    "NanoQuant: backend %s failed: %s",
                    name, ggml_status_to_string(status)));
        }
        std::vector<float> result(size_t(ggml_nelements(graph.output)));
        ggml_backend_tensor_get(
                graph.output, result.data(), 0, result.size()*sizeof(float));
        return result;
    }

    // A is [rows_a, shared], B is [rows_b, shared]. Returns B*A^T.
    std::vector<float> mul_mat_transposed(
            const std::vector<float> & a,
            int64_t rows_a,
            const std::vector<float> & b,
            int64_t rows_b,
            int64_t shared) {
        if (a.size() != size_t(rows_a)*size_t(shared) ||
            b.size() != size_t(rows_b)*size_t(shared)) {
            throw std::runtime_error("NanoQuant: internal MUL_MAT shape mismatch");
        }
        cached_graph & graph =
                get_graph(graph_op::MUL_MAT, shared, rows_a, shared, rows_b);
        return execute(graph, a, b, "MUL_MAT");
    }

    // A and B share their row count. Returns B^T*A.
    std::vector<float> out_product(
            const std::vector<float> & a,
            int64_t shared_rows,
            int64_t cols_a,
            const std::vector<float> & b,
            int64_t cols_b) {
        if (a.size() != size_t(shared_rows)*size_t(cols_a) ||
            b.size() != size_t(shared_rows)*size_t(cols_b)) {
            throw std::runtime_error("NanoQuant: internal OUT_PROD shape mismatch");
        }
        cached_graph & graph =
                get_graph(graph_op::OUT_PROD, cols_a, shared_rows, cols_b, shared_rows);
        return execute(graph, a, b, "OUT_PROD");
    }

    // L is [n,n], RHS is [n,n_rhs]. Returns the lower-triangular solve.
    std::vector<float> solve_lower(
            const std::vector<float> & lower,
            const std::vector<float> & rhs,
            int64_t n,
            int64_t n_rhs) {
        if (lower.size() != size_t(n)*size_t(n) ||
            rhs.size() != size_t(n)*size_t(n_rhs)) {
            throw std::runtime_error("NanoQuant: internal SOLVE_TRI shape mismatch");
        }
        cached_graph & graph =
                get_graph(graph_op::SOLVE_TRI, n, n, n_rhs, n);
        return execute(graph, lower, rhs, "SOLVE_TRI");
    }

    bool solve_spd_accelerated(
            const std::vector<float> & system,
            const std::vector<float> & rhs,
            int64_t n,
            int64_t n_rhs,
            std::vector<float> & solution) const {
        if (solve_spd_fn == nullptr) {
            return false;
        }
        solution.resize(rhs.size());
        return solve_spd_fn(
                backend, system.data(), rhs.data(), solution.data(), n, n_rhs);
    }

    std::vector<float> svid_rank1(
            const std::vector<float> & matrix,
            int64_t rows,
            int64_t columns,
            int32_t inner_iterations,
            const std::vector<float> & right) {
        if (matrix.size() != size_t(rows)*size_t(columns) ||
            right.size() != size_t(columns) ||
            inner_iterations <= 0) {
            throw std::runtime_error("NanoQuant: internal SVID shape mismatch");
        }
        cached_graph & graph = get_graph(
                graph_op::SVID, columns, rows, inner_iterations, 0);
        return execute(graph, matrix, right, "SVID");
    }

    std::vector<float> materialize_sign_product(
            const std::vector<float> & u,
            const std::vector<float> & v,
            int64_t n_out,
            int64_t n_in,
            int64_t rank) {
        if (u.size() != size_t(n_out)*size_t(rank) ||
            v.size() != size_t(rank)*size_t(n_in)) {
            throw std::runtime_error("NanoQuant: internal factor shape mismatch");
        }
        cached_graph & graph = get_graph(
                graph_op::SIGN_OUT_PROD, rank, n_out, n_in, rank);
        std::vector<float> result = execute(graph, u, v, "SIGN_OUT_PROD");
        reset_cache();
        return result;
    }
};

static std::vector<float> transpose_matrix(
        const std::vector<float> & input,
        int64_t rows,
        int64_t columns) {
    if (input.size() != size_t(rows) * size_t(columns)) {
        throw std::runtime_error("NanoQuant: internal transpose shape mismatch");
    }
    std::vector<float> result(input.size());
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t column = 0; column < columns; ++column) {
            result[size_t(column) * size_t(rows) + size_t(row)] =
                    input[size_t(row) * size_t(columns) + size_t(column)];
        }
    }
    return result;
}

static std::vector<float> solve_spd(
        compute_backend & backend,
        std::vector<float> system,
        const std::vector<float> & rhs_rows,
        int64_t rank,
        int64_t n_rhs) {
    if (system.size() != size_t(rank) * size_t(rank) ||
        rhs_rows.size() != size_t(n_rhs) * size_t(rank)) {
        throw std::runtime_error("NanoQuant: internal SPD solve shape mismatch");
    }
    if (backend.solve_spd_fn != nullptr) {
        std::vector<float> solution;
        float jitter = NUMERIC_EPSILON;
        float applied_jitter = 0.0f;
        for (int attempt = 0; attempt < 6; ++attempt) {
            const float increment = jitter - applied_jitter;
            for (int64_t i = 0; i < rank; ++i) {
                system[size_t(i)*size_t(rank) + size_t(i)] += increment;
            }
            if (backend.solve_spd_accelerated(
                    system, rhs_rows, rank, n_rhs, solution)) {
                return solution;
            }
            applied_jitter = jitter;
            jitter *= 10.0f;
        }
        throw std::runtime_error(
                "NanoQuant: accelerated Cholesky decomposition failed");
    }
    std::vector<float> lower(system.size(), 0.0f);
    float jitter = NUMERIC_EPSILON;
    bool decomposed = false;
    for (int attempt = 0; attempt < 6 && !decomposed; ++attempt) {
        std::fill(lower.begin(), lower.end(), 0.0f);
        decomposed = true;
        for (int64_t i = 0; i < rank && decomposed; ++i) {
            for (int64_t j = 0; j <= i; ++j) {
                double value = system[size_t(i) * size_t(rank) + size_t(j)];
                if (i == j) {
                    value += jitter;
                }
                for (int64_t k = 0; k < j; ++k) {
                    value -= double(lower[size_t(i) * size_t(rank) + size_t(k)]) *
                             double(lower[size_t(j) * size_t(rank) + size_t(k)]);
                }
                if (i == j) {
                    if (!(value > 0.0) || !std::isfinite(value)) {
                        decomposed = false;
                        break;
                    }
                    lower[size_t(i) * size_t(rank) + size_t(j)] = float(std::sqrt(value));
                } else {
                    lower[size_t(i) * size_t(rank) + size_t(j)] =
                            float(value / lower[size_t(j) * size_t(rank) + size_t(j)]);
                }
            }
        }
        jitter *= 10.0f;
    }
    if (!decomposed) {
        throw std::runtime_error("NanoQuant: stabilized Cholesky decomposition failed");
    }

    const std::vector<float> rhs = transpose_matrix(rhs_rows, n_rhs, rank);
    std::vector<float> intermediate = backend.solve_lower(lower, rhs, rank, n_rhs);
    std::vector<float> reverse_lower(lower.size(), 0.0f);
    std::vector<float> reverse_rhs(intermediate.size());
    for (int64_t i = 0; i < rank; ++i) {
        for (int64_t j = 0; j <= i; ++j) {
            reverse_lower[size_t(i) * size_t(rank) + size_t(j)] =
                    lower[size_t(rank - 1 - j) * size_t(rank) + size_t(rank - 1 - i)];
        }
        std::copy_n(intermediate.data() + size_t(rank - 1 - i) * size_t(n_rhs), n_rhs,
                reverse_rhs.data() + size_t(i) * size_t(n_rhs));
    }
    std::vector<float> reversed = backend.solve_lower(reverse_lower, reverse_rhs, rank, n_rhs);
    std::vector<float> solution_t(reversed.size());
    for (int64_t i = 0; i < rank; ++i) {
        std::copy_n(reversed.data() + size_t(rank - 1 - i) * size_t(n_rhs), n_rhs,
                solution_t.data() + size_t(i) * size_t(n_rhs));
    }
    return transpose_matrix(solution_t, rank, n_rhs);
}


static void extract_admm_factors(
        const group & item,
        const calibration_data & calibration,
        checkpoint_state & state) {
    const bool transposed = item.n_out < item.n_in;
    const int64_t n_in = transposed ? item.n_out : item.n_in;
    const int64_t n_out = transposed ? item.n_in : item.n_out;
    const std::vector<float> & input_importance =
            transposed ? calibration.output_norm : calibration.input_norm;
    const std::vector<float> & output_importance =
            transposed ? calibration.input_norm : calibration.output_norm;

    std::vector<float> norm_in(n_in);
    std::vector<float> norm_out(n_out);
    for (int64_t i = 0; i < n_in; ++i) {
        norm_in[i] = std::sqrt(std::max(input_importance[i], NUMERIC_EPSILON));
    }
    for (int64_t i = 0; i < n_out; ++i) {
        norm_out[i] = std::sqrt(std::max(output_importance[i], NUMERIC_EPSILON));
    }

    release_vector(state.u);
    release_vector(state.v);
    release_vector(state.dual_u);
    release_vector(state.dual_v);
    std::vector<float> proxy_u(state.z_u.size());
    std::vector<float> proxy_v(state.z_v.size());
    for (int64_t row = 0; row < n_out; ++row) {
        for (int64_t k = 0; k < item.rank; ++k) {
            const size_t index = size_t(row)*size_t(item.rank) + size_t(k);
            proxy_u[index] = state.z_u[index]/norm_out[row];
        }
    }
    for (int64_t k = 0; k < item.rank; ++k) {
        for (int64_t column = 0; column < n_in; ++column) {
            const size_t index = size_t(k)*size_t(n_in) + size_t(column);
            proxy_v[index] = state.z_v[index]/norm_in[column];
        }
    }

    double norm_u_sq = 0.0;
    double norm_v_sq = 0.0;
    for (float value : proxy_u) {
        norm_u_sq += double(value)*double(value);
    }
    for (float value : proxy_v) {
        norm_v_sq += double(value)*double(value);
    }
    const float balance = std::sqrt(
            std::max(float(std::sqrt(norm_v_sq)), NUMERIC_EPSILON)/
            std::max(float(std::sqrt(norm_u_sq)), NUMERIC_EPSILON));
    release_vector(state.weight);

    std::vector<float> column_scale(item.rank, 0.0f);
    for (int64_t row = 0; row < n_out; ++row) {
        for (int64_t k = 0; k < item.rank; ++k) {
            const float value = state.z_u[size_t(row)*size_t(item.rank) + size_t(k)];
            column_scale[k] += value*value;
        }
    }
    for (float & value : column_scale) {
        value = 1.0f/std::max(std::sqrt(value), NUMERIC_EPSILON);
    }

    for (int64_t row = 0; row < n_out; ++row) {
        for (int64_t k = 0; k < item.rank; ++k) {
            const size_t index = size_t(row)*size_t(item.rank) + size_t(k);
            proxy_u[index] *= balance*column_scale[k];
        }
    }
    for (float & value : proxy_v) {
        value /= balance;
    }

    std::vector<float> inner_scale_pre(n_in, 0.0f);
    std::vector<float> inner_scale_post(n_out, 0.0f);
    for (int64_t row = 0; row < n_out; ++row) {
        double sum = 0.0;
        for (int64_t k = 0; k < item.rank; ++k) {
            sum += std::abs(proxy_u[size_t(row)*size_t(item.rank) + size_t(k)]);
        }
        inner_scale_post[row] = std::max(float(sum/item.rank), NUMERIC_EPSILON);
    }
    for (int64_t column = 0; column < n_in; ++column) {
        double sum = 0.0;
        for (int64_t k = 0; k < item.rank; ++k) {
            sum += std::abs(proxy_v[size_t(k)*size_t(n_in) + size_t(column)]);
        }
        inner_scale_pre[column] = std::max(float(sum/item.rank), NUMERIC_EPSILON);
    }
    if (transposed) {
        state.u = transpose_matrix(proxy_v, item.rank, n_in);
        state.v = transpose_matrix(proxy_u, n_out, item.rank);
        state.scale_pre = std::move(inner_scale_post);
        state.scale_post = std::move(inner_scale_pre);
    } else {
        state.u = std::move(proxy_u);
        state.v = std::move(proxy_v);
        state.scale_pre = std::move(inner_scale_pre);
        state.scale_post = std::move(inner_scale_post);
    }
    release_vector(state.z_u);
    release_vector(state.z_v);
}

static void run_admm(
        compute_backend & backend,
        const group & item,
        const calibration_data & calibration,
        const llama_model_quantize_params * params,
        checkpoint_state & state) {
    if (state.stage >= checkpoint_stage::FACTOR) {
        return;
    }
    if (state.stage != checkpoint_stage::NONFACTOR &&
        state.stage != checkpoint_stage::ADMM) {
        throw std::runtime_error(format(
                "NanoQuant: group '%s' entered ADMM before nonfactor reconstruction completed",
                item.name.c_str()));
    }
    backend.reset_cache();
    std::unique_ptr<compute_backend> auxiliary;
    if (backend.solve_spd_fn != nullptr) {
        auxiliary = std::make_unique<compute_backend>(
                params->nanoquant_device, false);
        if (auxiliary->solve_spd_fn == nullptr) {
            auxiliary.reset();
        }
    }
    const bool transposed = item.n_out < item.n_in;
    const int64_t n_in = transposed ? item.n_out : item.n_in;
    const int64_t n_out = transposed ? item.n_in : item.n_out;
    const int64_t rank = item.rank;
    if (auxiliary) {
        backend.reserve_admm(
                n_in, n_out, rank, params->nanoquant_admm_inner_iterations,
                true, false);
        auxiliary->reserve_admm(
                n_in, n_out, rank, params->nanoquant_admm_inner_iterations,
                false, true);
    } else {
        backend.reserve_admm(
                n_in, n_out, rank, params->nanoquant_admm_inner_iterations,
                true, true);
    }
    const std::vector<float> & input_importance =
            transposed ? calibration.output_norm : calibration.input_norm;
    const std::vector<float> & output_importance =
            transposed ? calibration.input_norm : calibration.output_norm;
    std::vector<float> norm_in(n_in);
    std::vector<float> norm_out(n_out);
    for (int64_t i = 0; i < n_in; ++i) {
        norm_in[i] = std::sqrt(std::max(input_importance[i], NUMERIC_EPSILON));
    }
    for (int64_t i = 0; i < n_out; ++i) {
        norm_out[i] = std::sqrt(std::max(output_importance[i], NUMERIC_EPSILON));
    }
    std::vector<float> weighted_weight(state.weight.size());
    for (int64_t row = 0; row < n_out; ++row) {
        for (int64_t column = 0; column < n_in; ++column) {
            const size_t index = size_t(row)*size_t(n_in) + size_t(column);
            const size_t weight_index = transposed ?
                    size_t(column)*size_t(item.n_in) + size_t(row) : index;
            weighted_weight[index] =
                    state.weight[weight_index]*norm_out[row]*norm_in[column];
        }
    }
    release_vector(state.weight);

    const uint64_t group_seed = item.block >= 0 ?
            uint64_t(item.block + 1) : uint64_t(item.weight->idx + 1);
    deterministic_rng rng(state.rng_state ? state.rng_state :
            (params->nanoquant_seed ^ group_seed*UINT64_C(0x9e3779b97f4a7c15)));
    auto make_svid_right = [&rng](int64_t columns) {
        std::vector<float> values(static_cast<size_t>(columns));
        for (float & value : values) {
            value = rng.normal();
        }
        return values;
    };
    if (state.stage != checkpoint_stage::ADMM) {
        state.u.resize(size_t(n_out)*size_t(rank));
        state.v.resize(size_t(rank)*size_t(n_in));
        for (float & value : state.u) {
            value = rng.normal();
        }
        for (float & value : state.v) {
            value = rng.normal();
        }
        const std::vector<float> right_u = make_svid_right(rank);
        const std::vector<float> right_v = make_svid_right(n_in);
        if (auxiliary) {
            auto z_v_future = std::async(std::launch::async, [&]() {
                return auxiliary->svid_rank1(
                        state.v, rank, n_in,
                        params->nanoquant_admm_inner_iterations, right_v);
            });
            state.z_u = backend.svid_rank1(
                    state.u, n_out, rank,
                    params->nanoquant_admm_inner_iterations, right_u);
            state.z_v = z_v_future.get();
        } else {
            state.z_u = backend.svid_rank1(
                    state.u, n_out, rank,
                    params->nanoquant_admm_inner_iterations, right_u);
            state.z_v = backend.svid_rank1(
                    state.v, rank, n_in,
                    params->nanoquant_admm_inner_iterations, right_v);
        }
        state.dual_u.assign(state.u.size(), 0.0f);
        state.dual_v.assign(state.v.size(), 0.0f);
        state.stage = checkpoint_stage::ADMM;
        state.progress = 0;
        state.rng_state = rng.state;
    }

    struct admm_branch_result {
        std::vector<float> factor;
        std::vector<float> projected;
    };

    for (int32_t iteration = state.progress;
         iteration < params->nanoquant_admm_outer_iterations;
         ++iteration) {
        const float fraction =
                float(iteration)/float(params->nanoquant_admm_outer_iterations);
        const float rho = fraction;
        const std::vector<float> right_u = make_svid_right(rank);
        const std::vector<float> right_v = make_svid_right(n_in);
        release_vector(state.u);
        release_vector(state.v);

        auto update_u = [&]() -> admm_branch_result {
            std::vector<float> normalized_v = state.z_v;
            for (int64_t k = 0; k < rank; ++k) {
                double norm_sq = 0.0;
                for (int64_t column = 0; column < n_in; ++column) {
                    const float value =
                            normalized_v[size_t(k)*size_t(n_in) + size_t(column)];
                    norm_sq += double(value)*double(value);
                }
                const float inverse =
                        1.0f/std::max(float(std::sqrt(norm_sq)), NUMERIC_EPSILON);
                for (int64_t column = 0; column < n_in; ++column) {
                    normalized_v[size_t(k)*size_t(n_in) + size_t(column)] *= inverse;
                }
            }
            std::vector<float> system = backend.mul_mat_transposed(
                    normalized_v, rank, normalized_v, rank, n_in);
            double diagonal_mean = 0.0;
            for (int64_t k = 0; k < rank; ++k) {
                diagonal_mean +=
                        std::abs(system[size_t(k)*size_t(rank) + size_t(k)]);
            }
            const float stabilizer = std::max(
                    rho*float(diagonal_mean/rank) + ADMM_REGULARIZATION,
                    NUMERIC_EPSILON);
            for (int64_t k = 0; k < rank; ++k) {
                system[size_t(k)*size_t(rank) + size_t(k)] += stabilizer;
            }
            std::vector<float> rhs = backend.mul_mat_transposed(
                    normalized_v, rank, weighted_weight, n_out, n_in);
            for (size_t i = 0; i < rhs.size(); ++i) {
                rhs[i] += rho*(state.z_u[i] - state.dual_u[i]);
            }
            std::vector<float> factor =
                    solve_spd(backend, std::move(system), rhs, rank, n_out);
            for (size_t i = 0; i < factor.size(); ++i) {
                factor[i] += state.dual_u[i];
            }
            std::vector<float> projected = backend.svid_rank1(
                    factor, n_out, rank,
                    params->nanoquant_admm_inner_iterations, right_u);
            for (size_t i = 0; i < factor.size(); ++i) {
                factor[i] -= state.dual_u[i];
                state.dual_u[i] += factor[i] - projected[i];
            }
            return {std::move(factor), std::move(projected)};
        };

        compute_backend & v_backend = auxiliary ? *auxiliary : backend;
        auto update_v = [&]() -> admm_branch_result {
            std::vector<float> normalized_u = state.z_u;
            std::vector<float> column_norm(rank, 0.0f);
            for (int64_t row = 0; row < n_out; ++row) {
                for (int64_t k = 0; k < rank; ++k) {
                    const float value =
                            normalized_u[size_t(row)*size_t(rank) + size_t(k)];
                    column_norm[k] += value*value;
                }
            }
            for (float & value : column_norm) {
                value = 1.0f/std::max(std::sqrt(value), NUMERIC_EPSILON);
            }
            for (int64_t row = 0; row < n_out; ++row) {
                for (int64_t k = 0; k < rank; ++k) {
                    normalized_u[size_t(row)*size_t(rank) + size_t(k)] *=
                            column_norm[k];
                }
            }
            std::vector<float> system = v_backend.out_product(
                    normalized_u, n_out, rank, normalized_u, rank);
            double diagonal_mean = 0.0;
            for (int64_t k = 0; k < rank; ++k) {
                diagonal_mean +=
                        std::abs(system[size_t(k)*size_t(rank) + size_t(k)]);
            }
            const float stabilizer = std::max(
                    rho*float(diagonal_mean/rank) + ADMM_REGULARIZATION,
                    NUMERIC_EPSILON);
            for (int64_t k = 0; k < rank; ++k) {
                system[size_t(k)*size_t(rank) + size_t(k)] += stabilizer;
            }
            std::vector<float> rhs_rows = v_backend.out_product(
                    normalized_u, n_out, rank, weighted_weight, n_in);
            for (int64_t column = 0; column < n_in; ++column) {
                for (int64_t k = 0; k < rank; ++k) {
                    const size_t source =
                            size_t(k)*size_t(n_in) + size_t(column);
                    const size_t target =
                            size_t(column)*size_t(rank) + size_t(k);
                    rhs_rows[target] +=
                            rho*(state.z_v[source] - state.dual_v[source]);
                }
            }
            std::vector<float> solved_rows =
                    solve_spd(v_backend, std::move(system), rhs_rows, rank, n_in);
            std::vector<float> factor =
                    transpose_matrix(solved_rows, n_in, rank);
            for (size_t i = 0; i < factor.size(); ++i) {
                factor[i] += state.dual_v[i];
            }
            std::vector<float> projected = v_backend.svid_rank1(
                    factor, rank, n_in,
                    params->nanoquant_admm_inner_iterations, right_v);
            for (size_t i = 0; i < factor.size(); ++i) {
                factor[i] -= state.dual_v[i];
                state.dual_v[i] += factor[i] - projected[i];
            }
            return {std::move(factor), std::move(projected)};
        };

        admm_branch_result result_u;
        admm_branch_result result_v;
        if (auxiliary) {
            auto v_future = std::async(std::launch::async, update_v);
            result_u = update_u();
            result_v = v_future.get();
        } else {
            result_u = update_u();
            result_v = update_v();
        }
        state.u = std::move(result_u.factor);
        state.z_u = std::move(result_u.projected);
        state.v = std::move(result_v.factor);
        state.z_v = std::move(result_v.projected);
        state.progress = uint32_t(iteration + 1);
        state.rng_state = rng.state;
        const bool log_iteration =
                (iteration + 1) % ADMM_LOG_INTERVAL == 0 ||
                iteration + 1 == params->nanoquant_admm_outer_iterations;
        if (iteration == 0 || log_iteration) {
            LLAMA_LOG_INFO("NanoQuant: %s ADMM %d/%d rho=%.6f\n",
                    item.name.c_str(), iteration + 1,
                    params->nanoquant_admm_outer_iterations, rho);
        }
    }
    extract_admm_factors(item, calibration, state);
    state.stage = checkpoint_stage::FACTOR;
    state.optimizer_step = 0;
    release_vector(state.u_first_moment);
    release_vector(state.u_second_moment);
    release_vector(state.v_first_moment);
    release_vector(state.v_second_moment);
    release_vector(state.scale_pre_first_moment);
    release_vector(state.scale_pre_second_moment);
    release_vector(state.scale_post_first_moment);
    release_vector(state.scale_post_second_moment);
    state.progress = 0;
    state.rng_state = rng.state;
    backend.reset_cache();
}







struct training_tensor_storage {
    ggml_context_ptr context;
    ggml_backend_buffer_ptr buffer;

    explicit training_tensor_storage(size_t n_tensors) {
        ggml_init_params params = {
            /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        context.reset(ggml_init(params));
        if (!context) {
            throw std::runtime_error("NanoQuant: failed to allocate training tensor metadata");
        }
    }

    ggml_tensor * new_1d(ggml_type type, int64_t ne0, const char * name) {
        ggml_tensor * tensor = ggml_new_tensor_1d(context.get(), type, ne0);
        ggml_set_name(tensor, name);
        return tensor;
    }

    ggml_tensor * new_1d(int64_t ne0, const char * name) {
        return new_1d(GGML_TYPE_F32, ne0, name);
    }

    ggml_tensor * new_2d(ggml_type type, int64_t ne0, int64_t ne1, const char * name) {
        ggml_tensor * tensor = ggml_new_tensor_2d(context.get(), type, ne0, ne1);
        ggml_set_name(tensor, name);
        return tensor;
    }

    ggml_tensor * new_2d(int64_t ne0, int64_t ne1, const char * name) {
        return new_2d(GGML_TYPE_F32, ne0, ne1, name);
    }

    void allocate(ggml_backend_buffer_type_t buft) {
        buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(context.get(), buft));
        if (!buffer) {
            throw std::runtime_error("NanoQuant: failed to allocate training tensors");
        }
    }
};

struct owned_batch {
    llama_batch value;

    explicit owned_batch(int32_t n_tokens) : value(llama_batch_init(n_tokens, 0, 1)) {}
    ~owned_batch() {
        llama_batch_free(value);
    }
};

static ggml_backend_buffer_type_t training_buft(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->buffer == nullptr) {
        throw std::runtime_error("NanoQuant: training source tensor has no backend buffer");
    }
    return ggml_backend_buffer_get_type(tensor->buffer);
}

static void set_training_tensor(ggml_tensor * tensor, const std::vector<float> & values) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_F32 ||
        ggml_nelements(tensor) != int64_t(values.size())) {
        throw std::runtime_error("NanoQuant: training tensor shape mismatch");
    }
    ggml_backend_tensor_set(tensor, values.data(), 0, values.size()*sizeof(float));
}


static void get_training_tensor(const ggml_tensor * tensor, std::vector<float> & values) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(tensor)) {
        throw std::runtime_error("NanoQuant: training tensor shape mismatch");
    }
    values.resize(size_t(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size()*sizeof(float));
}

static std::vector<float> collect_block_target(
        llama_context * teacher,
        block_output_collector & collector,
        int block,
        const llama_token * tokens,
        int32_t n_tokens,
        llama_batch & batch) {
    batch.n_tokens = n_tokens;
    for (int32_t token = 0; token < n_tokens; ++token) {
        batch.token[token] = tokens[token];
        batch.pos[token] = token;
        batch.n_seq_id[token] = 1;
        batch.seq_id[token][0] = 0;
        batch.logits[token] = true;
    }
    collector.reset(block);
    llama_memory_clear(llama_get_memory(teacher), true);
    const int result = llama_decode(teacher, batch);
    if (result != 0) {
        throw std::runtime_error(format(
                "NanoQuant: teacher block evaluation failed (block %d, code %d)",
                block, result));
    }
    llama_synchronize(teacher);
    return collector.finish();
}

struct block_training_data {
    std::vector<std::vector<float>> targets;
    std::vector<float> loss_weights;
};

static block_training_data collect_block_training_data(
        llama_context * teacher,
        block_output_collector & collector,
        int block,
        const std::vector<llama_token> & samples,
        const llama_model_quantize_params * params) {
    block_training_data result;
    result.targets.reserve(params->nanoquant_sample_count);
    std::vector<double> squares;
    int64_t rows = 0;
    owned_batch batch(params->nanoquant_sequence_length);
    for (int32_t sample = 0; sample < params->nanoquant_sample_count; ++sample) {
        const llama_token * tokens = calibration_sample(samples, sample, params);
        std::vector<float> target = collect_block_target(
                teacher, collector, block, tokens,
                params->nanoquant_sequence_length, batch.value);
        if (target.size() % size_t(params->nanoquant_sequence_length) != 0) {
            throw std::runtime_error("NanoQuant: block target shape mismatch");
        }
        const size_t width = target.size()/size_t(params->nanoquant_sequence_length);
        if (squares.empty()) {
            squares.assign(width, 0.0);
        } else if (squares.size() != width) {
            throw std::runtime_error("NanoQuant: block target width changed");
        }
        for (size_t i = 0; i < target.size(); ++i) {
            squares[i % width] += double(target[i])*double(target[i]);
        }
        result.targets.push_back(std::move(target));
        rows += params->nanoquant_sequence_length;
    }
    if (squares.empty() || rows == 0) {
        throw std::runtime_error("NanoQuant: block loss observed no teacher outputs");
    }
    double mean = 0.0;
    for (double & value : squares) {
        value /= double(rows);
        mean += value;
    }
    mean /= double(squares.size());
    const double denominator = std::max(mean, double(NUMERIC_EPSILON));
    result.loss_weights.resize(squares.size());
    for (size_t i = 0; i < squares.size(); ++i) {
        const double shrunk =
                (1.0 - CALIBRATION_SHRINKAGE)*squares[i] +
                CALIBRATION_SHRINKAGE*mean;
        result.loss_weights[i] = float(std::sqrt(shrunk/denominator));
    }
    return result;
}

static std::vector<int32_t> shuffled_samples(
        int32_t count,
        deterministic_rng & rng) {
    std::vector<int32_t> result(count);
    std::iota(result.begin(), result.end(), 0);
    for (int32_t i = count; i > 1; --i) {
        const int32_t j = int32_t(rng.next_u64() % uint64_t(i));
        std::swap(result[size_t(i - 1)], result[size_t(j)]);
    }
    return result;
}

static float cosine_learning_rate_scale(uint32_t epoch, int32_t epochs) {
    if (epochs <= 1) {
        return 1.0f;
    }
    const double phase = double(epoch)/double(epochs);
    return float(0.5*(1.0 + std::cos(3.14159265358979323846*phase)));
}


static void run_nonfactor_reconstruction(
        compute_backend & backend,
        const group & item,
        const std::vector<llama_token> & samples,
        const block_training_data & training_data,
        const llama_model_quantize_params * params,
        llama_model * student_model,
        llama_context * student_context,
        block_output_collector & student_collector,
        checkpoint_state & state) {
    if (state.stage != checkpoint_stage::NONE &&
        state.stage != checkpoint_stage::NONFACTOR) {
        return;
    }
    if (state.weight.size() != size_t(item.n_in)*size_t(item.n_out)) {
        throw std::runtime_error(format(
                "NanoQuant: missing dense reconstruction state for '%s'",
                item.name.c_str()));
    }
    deterministic_rng rng(state.rng_state ? state.rng_state :
            (params->nanoquant_seed ^ uint64_t(item.block + 1)*UINT64_C(0x94d049bb133111eb)));
    if (state.stage == checkpoint_stage::NONE) {
        state.stage = checkpoint_stage::NONFACTOR;
        state.progress = 0;
        state.optimizer_step = 0;
        state.rng_state = rng.state;
        release_vector(state.weight_first_moment);
        release_vector(state.weight_second_moment);
    }
    if (state.progress > uint32_t(params->nanoquant_nonfactor_epochs)) {
        throw std::runtime_error(format(
                "NanoQuant: invalid nonfactor epoch for '%s'", item.name.c_str()));
    }

    ggml_tensor * source =
            const_cast<ggml_tensor *>(student_model->get_tensor(item.name.c_str()));
    if (source == nullptr) {
        throw std::runtime_error(format(
                "NanoQuant: missing student tensor '%s'", item.name.c_str()));
    }
    training_tensor_storage storage(1);
    ggml_tensor * weight = storage.new_2d(
            item.n_in, item.n_out, "nanoquant_training_weight");
    storage.allocate(backend.optimizer_buffer_type(ggml_nbytes(weight), item.name.c_str()));
    set_training_tensor(weight, state.weight);
    release_vector(state.weight);

    llama_nanoquant_weight & override = student_model->nanoquant_weights[source];
    if (override.training_weight != nullptr || override.training_factorized()) {
        throw std::runtime_error("NanoQuant: overlapping student training override");
    }
    override.training_weight = weight;
    owned_batch batch(params->nanoquant_sequence_length);
    const std::vector<llama_nanoquant_opt_param> opt_params = {{
        weight,
        params->nanoquant_nonfactor_learning_rate,
        -std::numeric_limits<float>::infinity(),
        &state.weight_first_moment,
        &state.weight_second_moment,
    }};
    std::unique_ptr<llama_nanoquant_optimizer, std::function<void(llama_nanoquant_optimizer *)>> optimizer(
            student_context->nanoquant_optimizer_init(
                    llama_nanoquant_opt_loss::MSE, item.block,
                    opt_params, state.optimizer_step),
            [student_context](llama_nanoquant_optimizer * value) {
                student_context->nanoquant_optimizer_free(value);
            });

    try {
        for (uint32_t epoch = state.progress;
             epoch < uint32_t(params->nanoquant_nonfactor_epochs);
             ++epoch) {
            const std::vector<int32_t> order =
                    shuffled_samples(params->nanoquant_sample_count, rng);
            double loss_sum = 0.0;
            for (int32_t sample : order) {
                const llama_token * tokens = calibration_sample(samples, sample, params);
                const std::vector<float> & target =
                        training_data.targets.at(size_t(sample));
                student_collector.reset(item.block);
                loss_sum += student_context->nanoquant_optimizer_step(
                        optimizer.get(), batch.value, tokens,
                        params->nanoquant_sequence_length,
                        target.data(), target.size(),
                        nullptr, 0,
                        training_data.loss_weights.data(),
                        training_data.loss_weights.size(),
                        cosine_learning_rate_scale(
                                epoch, params->nanoquant_nonfactor_epochs));
                ++state.optimizer_step;
            }
            state.progress = epoch + 1;
            state.rng_state = rng.state;
            LLAMA_LOG_INFO(
                    "NanoQuant: block %d %s nonfactor epoch %u/%d MSE=%.9g\n",
                    item.block, item.name.c_str(), epoch + 1,
                    params->nanoquant_nonfactor_epochs,
                    loss_sum/std::max<int32_t>(params->nanoquant_sample_count, 1));
        }
        get_training_tensor(weight, state.weight);
    } catch (...) {
        override.training_weight = nullptr;
        throw;
    }
    override.training_weight = nullptr;
    release_vector(state.weight_first_moment);
    release_vector(state.weight_second_moment);
}

static void harden_latent(std::vector<float> & values) {
    for (float & value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("NanoQuant: latent optimization produced a non-finite value");
        }
        if (std::abs(value) < NUMERIC_EPSILON) {
            value = value < 0.0f ? -NUMERIC_EPSILON : NUMERIC_EPSILON;
        }
    }
}

static void pack_factors(const group & item, checkpoint_state & state) {
    const size_t words_in = size_t((item.n_in + 31) / 32);
    const size_t words_rank = size_t((item.rank + 31) / 32);
    state.packed_v.assign(words_in * size_t(item.rank), 0);
    state.packed_u.assign(words_rank * size_t(item.n_out), 0);

    // Official Samsung convention: +1 is bit 0, -1 is bit 1, LSB first.
    // Zero-initialized padding therefore decodes to +1.
    for (int64_t k = 0; k < item.rank; ++k) {
        for (int64_t input = 0; input < item.n_in; ++input) {
            if (state.v[size_t(k) * size_t(item.n_in) + size_t(input)] < 0.0f) {
                state.packed_v[size_t(k) * words_in + size_t(input / 32)] |=
                        UINT32_C(1) << uint32_t(input % 32);
            }
        }
    }
    for (int64_t output = 0; output < item.n_out; ++output) {
        for (int64_t k = 0; k < item.rank; ++k) {
            if (state.u[size_t(output) * size_t(item.rank) + size_t(k)] < 0.0f) {
                state.packed_u[size_t(output) * words_rank + size_t(k / 32)] |=
                        UINT32_C(1) << uint32_t(k % 32);
            }
        }
    }
}

static void run_factor_reconstruction(
        compute_backend & backend,
        const group & item,
        const std::vector<llama_token> & samples,
        const block_training_data & training_data,
        const llama_model_quantize_params * params,
        const std::filesystem::path & checkpoint_directory,
        const hash256 & source_hash,
        const hash256 & config_hash,
        llama_model * student_model,
        llama_context * student_context,
        block_output_collector & student_collector,
        bool skip_factor_tuning,
        checkpoint_state & state) {
    if (state.stage != checkpoint_stage::FACTOR) {
        return;
    }
    if (state.progress > uint32_t(params->nanoquant_factor_epochs)) {
        throw std::runtime_error(format(
                "NanoQuant: invalid factor epoch for '%s'", item.name.c_str()));
    }
    deterministic_rng rng(state.rng_state ? state.rng_state :
            (params->nanoquant_seed ^ uint64_t(item.block + 1) * UINT64_C(0x2545f4914f6cdd1d)));
    if (skip_factor_tuning) {
        LLAMA_LOG_INFO(
                "NanoQuant: block %d %s expert %u/%" PRId64
                " keeps ADMM factors without factor tuning\n",
                item.block, item.name.c_str(), state.expert + 1, item.n_expert);
        harden_latent(state.u);
        harden_latent(state.v);
        pack_factors(item, state);
        state.stage = checkpoint_stage::EXPERT_DONE;
        state.progress = 0;
        release_completed_state(state);
        save_checkpoint(
                checkpoint_directory, source_hash, config_hash, item, state);
        return;
    }

    ggml_tensor * source =
            const_cast<ggml_tensor *>(student_model->get_tensor(item.name.c_str()));
    if (source == nullptr) {
        throw std::runtime_error(format(
                "NanoQuant: missing student tensor '%s'", item.name.c_str()));
    }
    training_tensor_storage storage(4);
    ggml_tensor * v = storage.new_2d(item.n_in, item.rank, "nanoquant_training_v");
    ggml_tensor * u = storage.new_2d(item.rank, item.n_out, "nanoquant_training_u");
    ggml_tensor * scale_pre = storage.new_1d(item.n_in, "nanoquant_training_scale_pre");
    ggml_tensor * scale_post = storage.new_1d(item.n_out, "nanoquant_training_scale_post");
    const size_t parameter_bytes =
            ggml_nbytes(v) + ggml_nbytes(u) + ggml_nbytes(scale_pre) + ggml_nbytes(scale_post);
    storage.allocate(backend.optimizer_buffer_type(parameter_bytes, item.name.c_str()));
    set_training_tensor(v, state.v);
    set_training_tensor(u, state.u);
    set_training_tensor(scale_pre, state.scale_pre);
    set_training_tensor(scale_post, state.scale_post);
    release_vector(state.v);
    release_vector(state.u);
    release_vector(state.scale_pre);
    release_vector(state.scale_post);

    llama_nanoquant_weight & override = student_model->nanoquant_weights[source];
    if (override.training_weight != nullptr || override.training_factorized() ||
        override.training_scale_pre != nullptr || override.training_scale_post != nullptr) {
        throw std::runtime_error("NanoQuant: overlapping student training override");
    }
    override.training_v = v;
    override.training_u = u;
    override.training_scale_pre = scale_pre;
    override.training_scale_post = scale_post;

    owned_batch batch(params->nanoquant_sequence_length);
    const float learning_rate = params->nanoquant_factor_learning_rate;
    const std::vector<llama_nanoquant_opt_param> opt_params = {
        { v, learning_rate, -std::numeric_limits<float>::infinity(),
          &state.v_first_moment, &state.v_second_moment },
        { u, learning_rate, -std::numeric_limits<float>::infinity(),
          &state.u_first_moment, &state.u_second_moment },
        { scale_pre, learning_rate, NUMERIC_EPSILON,
          &state.scale_pre_first_moment, &state.scale_pre_second_moment },
        { scale_post, learning_rate, NUMERIC_EPSILON,
          &state.scale_post_first_moment, &state.scale_post_second_moment },
    };
    std::unique_ptr<llama_nanoquant_optimizer, std::function<void(llama_nanoquant_optimizer *)>> optimizer(
            student_context->nanoquant_optimizer_init(
                    llama_nanoquant_opt_loss::MSE, item.block, opt_params, state.optimizer_step),
            [student_context](llama_nanoquant_optimizer * value) {
                student_context->nanoquant_optimizer_free(value);
            });

    try {
        for (uint32_t epoch = state.progress;
             epoch < uint32_t(params->nanoquant_factor_epochs);
             ++epoch) {
            const std::vector<int32_t> order =
                    shuffled_samples(params->nanoquant_sample_count, rng);
            double loss_sum = 0.0;
            for (int32_t sample : order) {
                const llama_token * tokens = calibration_sample(samples, sample, params);
                const std::vector<float> & target =
                        training_data.targets.at(size_t(sample));
                student_collector.reset(item.block);
                loss_sum += student_context->nanoquant_optimizer_step(
                        optimizer.get(), batch.value, tokens,
                        params->nanoquant_sequence_length,
                        target.data(), target.size(),
                        nullptr, 0,
                        training_data.loss_weights.data(),
                        training_data.loss_weights.size(),
                        cosine_learning_rate_scale(epoch, params->nanoquant_factor_epochs));
                ++state.optimizer_step;
            }
            state.progress = epoch + 1;
            state.rng_state = rng.state;
            LLAMA_LOG_INFO(
                    "NanoQuant: block %d %s factor epoch %u/%d MSE=%.9g\n",
                    item.block, item.name.c_str(), epoch + 1,
                    params->nanoquant_factor_epochs,
                    loss_sum/std::max<int32_t>(params->nanoquant_sample_count, 1));
        }
        get_training_tensor(v, state.v);
        get_training_tensor(u, state.u);
        get_training_tensor(scale_pre, state.scale_pre);
        get_training_tensor(scale_post, state.scale_post);
    } catch (...) {
        override.training_v = nullptr;
        override.training_u = nullptr;
        override.training_scale_pre = nullptr;
        override.training_scale_post = nullptr;
        throw;
    }
    override.training_v = nullptr;
    override.training_u = nullptr;
    override.training_scale_pre = nullptr;
    override.training_scale_post = nullptr;

    harden_latent(state.u);
    harden_latent(state.v);
    pack_factors(item, state);

    state.stage = checkpoint_stage::EXPERT_DONE;
    state.progress = 0;
    state.rng_state = rng.state;
    release_completed_state(state);
    save_checkpoint(checkpoint_directory, source_hash, config_hash, item, state);
}

static void complete_expert(
        const std::filesystem::path & checkpoint_directory,
        const hash256 & source_hash,
        const hash256 & config_hash,
        const group & item,
        checkpoint_state & state) {
    if (state.stage != checkpoint_stage::EXPERT_DONE ||
        state.expert >= uint32_t(item.n_expert)) {
        throw std::runtime_error(format(
                "NanoQuant: invalid completed expert state for '%s'", item.name.c_str()));
    }
    validate_state_shapes(item, state);
    if (item.n_expert > 1) {
        write_expert_checkpoint(
                checkpoint_path(checkpoint_directory, item.name),
                source_hash, config_hash, item, state.expert,
                state.scale_pre.data(), state.scale_post.data(),
                state.packed_u.data(), state.packed_v.data());
    }
    if (item.n_expert == 1) {
        state.completed_scale_pre.insert(
                state.completed_scale_pre.end(), state.scale_pre.begin(), state.scale_pre.end());
        state.completed_scale_post.insert(
                state.completed_scale_post.end(), state.scale_post.begin(), state.scale_post.end());
        state.completed_packed_u.insert(
                state.completed_packed_u.end(), state.packed_u.begin(), state.packed_u.end());
        state.completed_packed_v.insert(
                state.completed_packed_v.end(), state.packed_v.begin(), state.packed_v.end());
    }
    ++state.expert;

    release_completed_state(state);
    release_vector(state.scale_pre);
    release_vector(state.scale_post);
    release_attached_state(state);
    state.progress = 0;
    state.optimizer_step = 0;
    if (state.expert < uint32_t(item.n_expert)) {
        state.stage = checkpoint_stage::NONE;
        return;
    }

    if (item.n_expert > 1) {
        state.stage = checkpoint_stage::NONE;
        restore_completed_experts(
                checkpoint_path(checkpoint_directory, item.name),
                source_hash, config_hash, item, state, true);
    }
    state.scale_pre = std::move(state.completed_scale_pre);
    state.scale_post = std::move(state.completed_scale_post);
    state.packed_u = std::move(state.completed_packed_u);
    state.packed_v = std::move(state.completed_packed_v);
    state.stage = checkpoint_stage::GROUP_DONE;
}




static void set_student_scales(
        llama_model * student,
        const group & item,
        const checkpoint_state & state) {
    const ggml_tensor * virtual_weight = student->get_tensor(item.name.c_str());
    const llama_nanoquant_weight * nanoquant_weight =
            virtual_weight == nullptr ? nullptr : student->get_nanoquant_weight(virtual_weight);
    if (nanoquant_weight == nullptr || !nanoquant_weight->enabled()) {
        throw std::runtime_error(format(
                "NanoQuant: student model did not load sidecar group for '%s'", item.name.c_str()));
    }
    if (nanoquant_weight->scale_pre->type != GGML_TYPE_F16 ||
        nanoquant_weight->scale_post->type != GGML_TYPE_F16 ||
        nanoquant_weight->scale_pre->ne[0] != item.n_in ||
        nanoquant_weight->scale_post->ne[0] != item.n_out ||
        nanoquant_weight->scale_pre->ne[1] != item.n_expert ||
        nanoquant_weight->scale_post->ne[1] != item.n_expert) {
        throw std::runtime_error(format(
                "NanoQuant: student model scale contract mismatch for '%s'", item.name.c_str()));
    }
    const size_t n_scale_pre = size_t(item.n_in)*size_t(item.n_expert);
    const size_t n_scale_post = size_t(item.n_out)*size_t(item.n_expert);
    std::vector<ggml_fp16_t> scale_pre(n_scale_pre);
    std::vector<ggml_fp16_t> scale_post(n_scale_post);
    ggml_fp32_to_fp16_row(state.scale_pre.data(), scale_pre.data(), n_scale_pre);
    ggml_fp32_to_fp16_row(state.scale_post.data(), scale_post.data(), n_scale_post);
    ggml_backend_tensor_set(
            nanoquant_weight->scale_pre, scale_pre.data(), 0, scale_pre.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(
            nanoquant_weight->scale_post, scale_post.data(), 0, scale_post.size() * sizeof(ggml_fp16_t));
}



static std::vector<float> collect_teacher_probabilities(
        llama_context * teacher,
        const llama_token * tokens,
        int32_t n_tokens,
        llama_batch & batch);

struct teacher_probability_cache {
    std::filesystem::path path;
    std::ifstream input;
    size_t sample_values = 0;
    int32_t sample_count = 0;
    int32_t sequence_length = 0;
    int32_t vocab_size = 0;

    teacher_probability_cache(
            const std::filesystem::path & path,
            llama_context * teacher,
            const std::vector<llama_token> & samples,
            const llama_model_quantize_params * params);
    ~teacher_probability_cache();

    std::vector<float> load(int32_t sample);
};

static double full_model_kl(
        llama_context * student,
        teacher_probability_cache & teacher_cache,
        const std::vector<llama_token> & samples,
        const llama_model_quantize_params * params,
        owned_batch & batch) {
    const int32_t sequence_length = params->nanoquant_sequence_length;
    const int32_t vocab_size = llama_vocab_n_tokens(
            llama_model_get_vocab(llama_get_model(student)));
    if (teacher_cache.vocab_size != vocab_size) {
        throw std::runtime_error("NanoQuant: teacher/student KL shape mismatch");
    }
    double total_kl = 0.0;
    int64_t total_tokens = 0;
    for (int32_t sample = 0; sample < params->nanoquant_sample_count; ++sample) {
        const llama_token * sample_tokens =
                calibration_sample(samples, sample, params);
        std::vector<float> teacher_probabilities = teacher_cache.load(sample);

        batch.value.n_tokens = sequence_length;
        for (int32_t token = 0; token < sequence_length; ++token) {
            batch.value.token[token] = sample_tokens[token];
            batch.value.pos[token] = token;
            batch.value.n_seq_id[token] = 1;
            batch.value.seq_id[token][0] = 0;
            batch.value.logits[token] = 1;
        }
        llama_memory_clear(llama_get_memory(student), true);
        const int decode_result = llama_decode(student, batch.value);
        if (decode_result != 0) {
            throw std::runtime_error(format(
                    "NanoQuant: full-model student KL evaluation failed at sample %d (code %d)",
                    sample, decode_result));
        }
        llama_synchronize(student);
        for (int32_t token = 0; token < sequence_length; ++token) {
            const float * student_logits = llama_get_logits_ith(student, token);
            if (student_logits == nullptr) {
                throw std::runtime_error("NanoQuant: full-model student logits were not retained");
            }
            float student_max = -std::numeric_limits<float>::infinity();
            for (int32_t vocabulary = 0; vocabulary < vocab_size; ++vocabulary) {
                const float logit = student_logits[vocabulary];
                if (std::isnan(logit) ||
                    logit == std::numeric_limits<float>::infinity()) {
                    throw std::runtime_error("NanoQuant: full-model KL encountered invalid logits");
                }
                student_max = std::max(student_max, logit);
            }
            if (!std::isfinite(student_max)) {
                throw std::runtime_error("NanoQuant: full-model student KL has no finite logits");
            }
            double student_sum = 0.0;
            for (int32_t vocabulary = 0; vocabulary < vocab_size; ++vocabulary) {
                if (std::isfinite(student_logits[vocabulary])) {
                    student_sum += std::exp(
                            double(student_logits[vocabulary]) - double(student_max));
                }
            }
            const double student_log_sum = std::log(student_sum);
            const float * probabilities =
                    teacher_probabilities.data() + size_t(token)*size_t(vocab_size);
            for (int32_t vocabulary = 0; vocabulary < vocab_size; ++vocabulary) {
                const double probability = probabilities[vocabulary];
                if (!std::isfinite(probability) || probability < 0.0) {
                    throw std::runtime_error("NanoQuant: teacher probability is invalid");
                }
                if (probability == 0.0) {
                    continue;
                }
                if (!std::isfinite(student_logits[vocabulary])) {
                    throw std::runtime_error(
                            "NanoQuant: student assigns zero probability to a teacher-supported token");
                }
                const double student_log_probability =
                        double(student_logits[vocabulary]) -
                        double(student_max) - student_log_sum;
                total_kl += probability *
                        (std::log(probability) - student_log_probability);
            }
            ++total_tokens;
        }
    }
    const double result = total_kl/std::max<int64_t>(total_tokens, 1);
    if (!std::isfinite(result)) {
        throw std::runtime_error("NanoQuant: full-model teacher/student KL is non-finite");
    }
    return std::max(result, 0.0);
}

static std::filesystem::path model_checkpoint_path(
        const std::filesystem::path & checkpoint_directory) {
    return checkpoint_directory / "model-tuning.nqckpt";
}

static void save_model_checkpoint(
        const std::filesystem::path & checkpoint_directory,
        const hash256 & source_hash,
        const hash256 & config_hash,
        const std::vector<group> & groups,
        const std::vector<checkpoint_state> & states,
        uint32_t progress,
        uint64_t rng_state,
        uint64_t optimizer_step) {
    atomic_replace(model_checkpoint_path(checkpoint_directory), [&](std::ostream & output) {
        static const char magic[8] = { 'N', 'Q', 'M', 'O', 'D', '1', '\0', '\0' };
        output.write(magic, sizeof(magic));
        write_pod(output, CHECKPOINT_VERSION);
        write_pod(output, UINT32_C(0x01020304));
        output.write(reinterpret_cast<const char *>(source_hash.data()), sizeof(source_hash));
        output.write(reinterpret_cast<const char *>(config_hash.data()), sizeof(config_hash));
        write_pod(output, progress);
        write_pod(output, rng_state);
        write_pod(output, optimizer_step);
        write_pod(output, uint64_t(groups.size()));
        for (size_t i = 0; i < groups.size(); ++i) {
            write_string(output, groups[i].name);
            write_vector(output, states[i].scale_pre);
            write_vector(output, states[i].scale_post);
            write_vector(output, states[i].scale_pre_first_moment);
            write_vector(output, states[i].scale_pre_second_moment);
            write_vector(output, states[i].scale_post_first_moment);
            write_vector(output, states[i].scale_post_second_moment);
        }
    });
}

static bool load_model_checkpoint(
        const std::filesystem::path & checkpoint_directory,
        const hash256 & source_hash,
        const hash256 & config_hash,
        const std::vector<group> & groups,
        const std::vector<bool> & reachable,
        std::vector<checkpoint_state> & states,
        uint32_t & progress,
        uint64_t & rng_state,
        uint64_t & optimizer_step) {
    std::filesystem::path path = model_checkpoint_path(checkpoint_directory);
    if (!std::filesystem::exists(path)) {
        path = path.string() + ".bak";
        if (!std::filesystem::exists(path)) {
            return false;
        }
    }
    std::ifstream input(path, std::ios::binary);
    input.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    char magic[8];
    input.read(magic, sizeof(magic));
    static const char expected_magic[8] = { 'N', 'Q', 'M', 'O', 'D', '1', '\0', '\0' };
    if (std::memcmp(magic, expected_magic, sizeof(magic)) != 0 ||
        read_pod<uint32_t>(input) != CHECKPOINT_VERSION ||
        read_pod<uint32_t>(input) != UINT32_C(0x01020304)) {
        throw std::runtime_error(format(
                "NanoQuant: invalid model-tuning checkpoint '%s'", path.string().c_str()));
    }
    hash256 stored_source;
    hash256 stored_config;
    input.read(reinterpret_cast<char *>(stored_source.data()), sizeof(stored_source));
    input.read(reinterpret_cast<char *>(stored_config.data()), sizeof(stored_config));
    if (stored_source != source_hash || stored_config != config_hash) {
        throw std::runtime_error("NanoQuant: model-tuning checkpoint identity mismatch");
    }
    progress = read_pod<uint32_t>(input);
    rng_state = read_pod<uint64_t>(input);
    optimizer_step = read_pod<uint64_t>(input);
    const uint64_t count = read_pod<uint64_t>(input);
    if (count != groups.size() || reachable.size() != groups.size() ||
        states.size() != groups.size()) {
        throw std::runtime_error("NanoQuant: model-tuning checkpoint group count mismatch");
    }
    for (size_t i = 0; i < groups.size(); ++i) {
        if (read_string(input) != groups[i].name) {
            throw std::runtime_error("NanoQuant: model-tuning checkpoint group order mismatch");
        }
        states[i].scale_pre = read_vector<float>(input);
        states[i].scale_post = read_vector<float>(input);
        states[i].scale_pre_first_moment = read_vector<float>(input);
        states[i].scale_pre_second_moment = read_vector<float>(input);
        states[i].scale_post_first_moment = read_vector<float>(input);
        states[i].scale_post_second_moment = read_vector<float>(input);
        const size_t n_scale_pre =
                size_t(groups[i].n_in)*size_t(groups[i].n_expert);
        const size_t n_scale_post =
                size_t(groups[i].n_out)*size_t(groups[i].n_expert);
        if (states[i].scale_pre.size() != n_scale_pre ||
            states[i].scale_post.size() != n_scale_post) {
            throw std::runtime_error(format(
                    "NanoQuant: model-tuning scale shape mismatch for '%s'",
                    groups[i].name.c_str()));
        }
        if (progress > 0 && reachable[i] &&
            (states[i].scale_pre_first_moment.size() != n_scale_pre ||
             states[i].scale_pre_second_moment.size() != n_scale_pre ||
             states[i].scale_post_first_moment.size() != n_scale_post ||
             states[i].scale_post_second_moment.size() != n_scale_post)) {
            throw std::runtime_error(format(
                    "NanoQuant: model-tuning moment shape mismatch for '%s'",
                    groups[i].name.c_str()));
        }
    }
    if (input.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("NanoQuant: trailing data in model-tuning checkpoint");
    }
    return true;
}

static std::vector<float> collect_teacher_probabilities(
        llama_context * teacher,
        const llama_token * tokens,
        int32_t n_tokens,
        llama_batch & batch) {
    batch.n_tokens = n_tokens;
    for (int32_t token = 0; token < n_tokens; ++token) {
        batch.token[token] = tokens[token];
        batch.pos[token] = token;
        batch.n_seq_id[token] = 1;
        batch.seq_id[token][0] = 0;
        batch.logits[token] = true;
    }
    llama_memory_clear(llama_get_memory(teacher), true);
    const int decode_result = llama_decode(teacher, batch);
    if (decode_result != 0) {
        throw std::runtime_error(format(
                "NanoQuant: teacher KL evaluation failed (code %d)", decode_result));
    }
    llama_synchronize(teacher);

    const int32_t vocab_size = llama_vocab_n_tokens(
            llama_model_get_vocab(llama_get_model(teacher)));
    std::vector<float> result(size_t(vocab_size)*size_t(n_tokens), 0.0f);
    for (int32_t token = 0; token < n_tokens; ++token) {
        const float * logits = llama_get_logits_ith(teacher, token);
        if (logits == nullptr) {
            throw std::runtime_error("NanoQuant: teacher logits were not retained");
        }
        float maximum = -std::numeric_limits<float>::infinity();
        for (int32_t vocabulary = 0; vocabulary < vocab_size; ++vocabulary) {
            if (std::isnan(logits[vocabulary]) ||
                logits[vocabulary] == std::numeric_limits<float>::infinity()) {
                throw std::runtime_error("NanoQuant: teacher logits are invalid");
            }
            maximum = std::max(maximum, logits[vocabulary]);
        }
        if (!std::isfinite(maximum)) {
            throw std::runtime_error("NanoQuant: teacher logits have no finite value");
        }
        double sum = 0.0;
        for (int32_t vocabulary = 0; vocabulary < vocab_size; ++vocabulary) {
            if (std::isfinite(logits[vocabulary])) {
                sum += std::exp(double(logits[vocabulary]) - double(maximum));
            }
        }
        for (int32_t vocabulary = 0; vocabulary < vocab_size; ++vocabulary) {
            if (std::isfinite(logits[vocabulary])) {
                result[size_t(token)*size_t(vocab_size) + size_t(vocabulary)] =
                        float(std::exp(double(logits[vocabulary]) - double(maximum))/sum);
            }
        }
    }
    return result;
}

teacher_probability_cache::teacher_probability_cache(
        const std::filesystem::path & cache_path,
        llama_context * teacher,
        const std::vector<llama_token> & samples,
        const llama_model_quantize_params * params) :
        path(cache_path),
        sample_count(params->nanoquant_sample_count),
        sequence_length(params->nanoquant_sequence_length),
        vocab_size(llama_vocab_n_tokens(
                llama_model_get_vocab(llama_get_model(teacher)))) {
    if (sample_count <= 0 || sequence_length <= 0 || vocab_size <= 0 ||
        size_t(sequence_length) > std::numeric_limits<size_t>::max()/size_t(vocab_size)) {
        throw std::runtime_error("NanoQuant: teacher probability cache shape is invalid");
    }
    sample_values = size_t(sequence_length)*size_t(vocab_size);
    if (sample_values > std::numeric_limits<size_t>::max()/sizeof(float) ||
        size_t(sample_count) > std::numeric_limits<size_t>::max()/
                (sample_values*sizeof(float))) {
        throw std::runtime_error("NanoQuant: teacher probability cache size overflow");
    }

    const auto start = std::chrono::steady_clock::now();
    std::error_code ec;
    std::filesystem::remove(path, ec);
    try {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        owned_batch batch(sequence_length);
        for (int32_t sample = 0; sample < sample_count; ++sample) {
            const llama_token * tokens =
                    calibration_sample(samples, sample, params);
            const std::vector<float> probabilities =
                    collect_teacher_probabilities(
                            teacher, tokens, sequence_length, batch.value);
            if (probabilities.size() != sample_values) {
                throw std::runtime_error(
                        "NanoQuant: teacher probability cache shape changed");
            }
            output.write(
                    reinterpret_cast<const char *>(probabilities.data()),
                    probabilities.size()*sizeof(float));
        }
        output.close();
        const size_t expected =
                size_t(sample_count)*sample_values*sizeof(float);
        if (std::filesystem::file_size(path) != expected) {
            throw std::runtime_error(
                    "NanoQuant: teacher probability cache size mismatch");
        }
        input.open(path, std::ios::binary);
        if (!input) {
            throw std::runtime_error(
                    "NanoQuant: failed to reopen teacher probability cache");
        }
        LLAMA_LOG_INFO(
                "NanoQuant profile: cached teacher probabilities=%.3fs size=%zu bytes\n",
                std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - start).count(),
                expected);
    } catch (...) {
        input.close();
        std::filesystem::remove(path, ec);
        throw;
    }
}

teacher_probability_cache::~teacher_probability_cache() {
    input.close();
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

std::vector<float> teacher_probability_cache::load(int32_t sample) {
    if (sample < 0 || sample >= sample_count) {
        throw std::runtime_error(
                "NanoQuant: teacher probability cache sample is out of range");
    }
    const size_t sample_bytes = sample_values*sizeof(float);
    const size_t byte_offset = size_t(sample)*sample_bytes;
    if (sample_bytes > size_t(std::numeric_limits<std::streamsize>::max()) ||
        byte_offset > size_t(std::numeric_limits<std::streamoff>::max())) {
        throw std::runtime_error(
                "NanoQuant: teacher probability cache offset is out of range");
    }
    std::vector<float> result(sample_values);
    input.clear();
    input.seekg(std::streamoff(byte_offset));
    input.read(reinterpret_cast<char *>(result.data()), std::streamsize(sample_bytes));
    if (!input || size_t(input.gcount()) != sample_bytes) {
        throw std::runtime_error(
                "NanoQuant: failed to read teacher probability cache");
    }
    return result;
}



static void run_model_scale_kl(
        teacher_probability_cache & teacher_cache,
        llama_model * student_model,
        llama_context * student_context,
        owned_batch & batch,
        const std::vector<llama_token> & samples,
        const std::vector<group> & groups,
        const std::vector<bool> & reachable,
        const llama_model_quantize_params * params,
        const std::filesystem::path & checkpoint_directory,
        const hash256 & source_hash,
        const hash256 & config_hash,
        std::vector<checkpoint_state> & states) {
    if (reachable.size() != groups.size()) {
        throw std::runtime_error("NanoQuant: global scale reachability shape mismatch");
    }
    if (teacher_cache.sample_count != params->nanoquant_sample_count ||
        teacher_cache.sequence_length != params->nanoquant_sequence_length) {
        throw std::runtime_error("NanoQuant: teacher probability cache contract mismatch");
    }
    uint32_t progress = 0;
    deterministic_rng initial_rng(
            params->nanoquant_seed ^ UINT64_C(0xd1b54a32d192ed03));
    uint64_t rng_state = initial_rng.state;
    uint64_t optimizer_step = 0;
    const bool resumed = load_model_checkpoint(
            checkpoint_directory, source_hash, config_hash, groups, reachable,
            states, progress, rng_state, optimizer_step);
    if (!resumed) {
        for (size_t i = 0; i < groups.size(); ++i) {
            if (states[i].stage != checkpoint_stage::GROUP_DONE) {
                throw std::runtime_error(format(
                        "NanoQuant: group '%s' entered global KL before block reconstruction completed",
                        groups[i].name.c_str()));
            }
            states[i].scale_pre_first_moment.clear();
            states[i].scale_pre_second_moment.clear();
            states[i].scale_post_first_moment.clear();
            states[i].scale_post_second_moment.clear();
        }
        save_model_checkpoint(
                checkpoint_directory, source_hash, config_hash, groups,
                states, progress, rng_state, optimizer_step);
    }
    if (progress > uint32_t(params->nanoquant_model_epochs)) {
        throw std::runtime_error("NanoQuant: invalid global model epoch in checkpoint");
    }
    deterministic_rng rng(rng_state);

    std::vector<std::unique_ptr<training_tensor_storage>> storages;
    std::vector<ggml_tensor *> scale_pre_tensors(groups.size());
    std::vector<ggml_tensor *> scale_post_tensors(groups.size());
    std::vector<llama_nanoquant_weight *> overrides(groups.size());
    std::vector<llama_nanoquant_opt_param> opt_params;
    storages.reserve(groups.size());
    opt_params.reserve(2*groups.size());
    for (size_t i = 0; i < groups.size(); ++i) {
        set_student_scales(student_model, groups[i], states[i]);
        if (!reachable[i]) {
            continue;
        }
        ggml_tensor * virtual_weight =
                const_cast<ggml_tensor *>(student_model->get_tensor(groups[i].name.c_str()));
        if (virtual_weight == nullptr) {
            throw std::runtime_error("NanoQuant: global scale tensor is missing");
        }
        llama_nanoquant_weight & override = student_model->nanoquant_weights[virtual_weight];
        if (!override.enabled() || override.training_weight != nullptr ||
            override.training_v != nullptr || override.training_u != nullptr ||
            override.training_scale_pre != nullptr || override.training_scale_post != nullptr) {
            throw std::runtime_error("NanoQuant: invalid global scale training override");
        }

        auto storage = std::make_unique<training_tensor_storage>(2);
        scale_pre_tensors[i] = storage->new_2d(
                groups[i].n_in, groups[i].n_expert, "nanoquant_model_scale_pre");
        scale_post_tensors[i] = storage->new_2d(
                groups[i].n_out, groups[i].n_expert, "nanoquant_model_scale_post");
        storage->allocate(training_buft(override.scale_pre));
        set_training_tensor(scale_pre_tensors[i], states[i].scale_pre);
        set_training_tensor(scale_post_tensors[i], states[i].scale_post);
        override.training_scale_pre = scale_pre_tensors[i];
        override.training_scale_post = scale_post_tensors[i];
        overrides[i] = &override;
        storages.push_back(std::move(storage));

        opt_params.push_back({
            scale_pre_tensors[i],
            params->nanoquant_model_learning_rate,
            NUMERIC_EPSILON,
            &states[i].scale_pre_first_moment,
            &states[i].scale_pre_second_moment,
        });
        opt_params.push_back({
            scale_post_tensors[i],
            params->nanoquant_model_learning_rate,
            NUMERIC_EPSILON,
            &states[i].scale_post_first_moment,
            &states[i].scale_post_second_moment,
        });
    }

    auto clear_overrides = [&]() {
        for (llama_nanoquant_weight * override : overrides) {
            if (override != nullptr) {
                override->training_v = nullptr;
                override->training_u = nullptr;
                override->training_scale_pre = nullptr;
                override->training_scale_post = nullptr;
            }
        }
    };
    std::unique_ptr<llama_nanoquant_optimizer, std::function<void(llama_nanoquant_optimizer *)>> optimizer(
            student_context->nanoquant_optimizer_init(
                    llama_nanoquant_opt_loss::CROSS_ENTROPY, -1,
                    opt_params, optimizer_step),
            [student_context](llama_nanoquant_optimizer * value) {
                student_context->nanoquant_optimizer_free(value);
            });

    try {
        for (uint32_t epoch = progress;
             epoch < uint32_t(params->nanoquant_model_epochs);
             ++epoch) {
            const std::vector<int32_t> order =
                    shuffled_samples(params->nanoquant_sample_count, rng);
            double loss_sum = 0.0;
            for (int32_t sample : order) {
                const llama_token * tokens =
                        calibration_sample(samples, sample, params);
                std::vector<float> probabilities = teacher_cache.load(sample);
                loss_sum += student_context->nanoquant_optimizer_step(
                        optimizer.get(), batch.value, tokens,
                        params->nanoquant_sequence_length,
                        probabilities.data(), probabilities.size(),
                        nullptr, 0,
                        nullptr, 0,
                        cosine_learning_rate_scale(epoch, params->nanoquant_model_epochs));
                ++optimizer_step;
            }
            student_context->nanoquant_optimizer_export(optimizer.get());
            for (size_t i = 0; i < groups.size(); ++i) {
                if (reachable[i]) {
                    get_training_tensor(scale_pre_tensors[i], states[i].scale_pre);
                    get_training_tensor(scale_post_tensors[i], states[i].scale_post);
                }
            }
            progress = epoch + 1;
            rng_state = rng.state;
            save_model_checkpoint(
                    checkpoint_directory, source_hash, config_hash, groups,
                    states, progress, rng_state, optimizer_step);
            for (checkpoint_state & state : states) {
                release_completed_state(state);
            }
            LLAMA_LOG_INFO(
                    "NanoQuant: global full-model KL epoch %u/%d cross-entropy=%.9g\n",
                    epoch + 1, params->nanoquant_model_epochs,
                    loss_sum/std::max<int32_t>(params->nanoquant_sample_count, 1));
        }
    } catch (...) {
        clear_overrides();
        throw;
    }
    clear_overrides();

    for (size_t i = 0; i < groups.size(); ++i) {
        set_student_scales(student_model, groups[i], states[i]);
        checkpoint_state output_state;
        if (!load_checkpoint_file(
                    checkpoint_path(checkpoint_directory, groups[i].name),
                    source_hash, config_hash, groups[i], output_state)) {
            throw std::runtime_error(format(
                    "NanoQuant: missing completed checkpoint for '%s'",
                    groups[i].name.c_str()));
        }
        validate_state_shapes(groups[i], output_state);
        output_state.scale_pre = std::move(states[i].scale_pre);
        output_state.scale_post = std::move(states[i].scale_post);
        output_state.stage = checkpoint_stage::MODEL_DONE;
        output_state.progress = 0;
        output_state.rng_state = rng_state;
        output_state.optimizer_step = optimizer_step;
        release_completed_state(output_state);
        save_checkpoint(
                checkpoint_directory, source_hash, config_hash, groups[i], output_state);
        states[i].stage = checkpoint_stage::MODEL_DONE;
    }
}

static void validate_options(const llama_model_quantize_params * params) {
    if (params->keep_split) {
        throw std::runtime_error("NanoQuant: --keep-split is not supported; output is one strict sidecar GGUF");
    }
    if (params->only_copy || params->pure ||
        params->imatrix != nullptr || params->tt_overrides != nullptr ||
        params->prune_layers != nullptr ||
        params->output_tensor_type != GGML_TYPE_COUNT ||
        params->token_embedding_type != GGML_TYPE_COUNT) {
        throw std::runtime_error(
                "NanoQuant: copy/pure/imatrix/tensor-type/pruning/type-override options cannot be combined with NANOQUANT");
    }
    if (!(params->nanoquant_target_bits > 0.0f) ||
        !std::isfinite(params->nanoquant_target_bits) ||
        params->nanoquant_target_bits > 16.0f) {
        throw std::runtime_error("NanoQuant: --nanoquant-target-bits must be finite and in (0, 16]");
    }
    if (params->dry_run) {
        return;
    }
    if (params->nanoquant_calibration_dataset == nullptr ||
        params->nanoquant_calibration_dataset[0] == '\0') {
        throw std::runtime_error(
                "NanoQuant: actual NANOQUANT conversion requires --nanoquant-calibration-dataset PATH");
    }
    if (params->nanoquant_sequence_length <= 0 ||
        params->nanoquant_sample_count <= 0) {
        throw std::runtime_error("NanoQuant: calibration sequence length and sample count must be positive");
    }
    if (params->nanoquant_admm_outer_iterations <= 0 ||
        params->nanoquant_admm_inner_iterations <= 0 ||
        params->nanoquant_nonfactor_epochs <= 0 ||
        params->nanoquant_factor_epochs <= 0 ||
        params->nanoquant_model_epochs <= 0) {
        throw std::runtime_error(
                "NanoQuant: ADMM and all reconstruction phase iteration/epoch counts must be positive");
    }
    if (!(params->nanoquant_nonfactor_learning_rate > 0.0f) ||
        !(params->nanoquant_factor_learning_rate > 0.0f) ||
        !(params->nanoquant_model_learning_rate > 0.0f) ||
        !std::isfinite(params->nanoquant_nonfactor_learning_rate) ||
        !std::isfinite(params->nanoquant_factor_learning_rate) ||
        !std::isfinite(params->nanoquant_model_learning_rate)) {
        throw std::runtime_error("NanoQuant: all reconstruction learning rates must be positive and finite");
    }
    const uint32_t endian_probe = 1;
    if (*reinterpret_cast<const uint8_t *>(&endian_probe) != 1) {
        throw std::runtime_error(
                "NanoQuant: raw official packed sign encoding is only supported on little-endian hosts");
    }
}

static bool is_supported_projection(const std::string & name, int n_dims) {
    static constexpr const char * suffix = ".weight";
    const size_t suffix_size = std::strlen(suffix);
    if (name.size() <= suffix_size ||
        name.compare(name.size() - suffix_size, suffix_size, suffix) != 0) {
        return false;
    }
    llm_tensor_info info;
    if (!llm_tensor_info_for_name(name.substr(0, name.size() - suffix_size), info)) {
        return false;
    }
    return (n_dims == 2 && info.op == GGML_OP_MUL_MAT) ||
           (n_dims == 3 && info.op == GGML_OP_MUL_MAT_ID);
}

static std::vector<group> find_groups(
        llama_model_loader & loader,
        bool allow_requantize) {
    uint32_t model_n_expert_used = 1;
    loader.get_key(LLM_KV_EXPERT_USED_COUNT, model_n_expert_used, false);
    model_n_expert_used = std::max<uint32_t>(model_n_expert_used, 1);
    std::unordered_set<std::string> source_names;
    source_names.reserve(loader.weights_map.size());
    for (const auto & entry : loader.weights_map) {
        source_names.insert(entry.first);
    }

    std::vector<group> groups;
    for (const auto & entry : loader.weights_map) {
        ggml_tensor * tensor = entry.second.tensor;
        const std::string name = ggml_get_name(tensor);
        const int block = decoder_block(name);
        const int n_dims = ggml_n_dims(tensor);
        if ((n_dims != 2 && n_dims != 3) ||
            block < 0 ||
            !is_supported_projection(name, n_dims)) {
            continue;
        }
        if (ggml_is_quantized(tensor->type) && !allow_requantize) {
            throw std::runtime_error(format(
                    "NanoQuant: requantizing selected tensor '%s' from type %s is disabled; use --allow-requantize",
                    name.c_str(), ggml_type_name(tensor->type)));
        }
        if (!source_type_supports_f32(tensor->type)) {
            throw std::runtime_error(format(
                    "NanoQuant: selected tensor '%s' has unsupported source type %s",
                    name.c_str(), ggml_type_name(tensor->type)));
        }
        group item;
        item.weight = &entry.second;
        item.name = name;
        item.name_v = sidecar_name(name, ".nq_v");
        item.name_u = sidecar_name(name, ".nq_u");
        item.name_scale_pre = sidecar_name(name, ".nq_scale_pre");
        item.name_scale_post = sidecar_name(name, ".nq_scale_post");
        item.block = block;
        item.n_in = tensor->ne[0];
        item.n_out = tensor->ne[1];
        item.n_expert = n_dims == 3 ? tensor->ne[2] : 1;
        item.n_expert_used = n_dims == 3 ? model_n_expert_used : 1;
        item.rank = 1;
        const std::string names[] = {
            item.name_v, item.name_u, item.name_scale_pre, item.name_scale_post,
        };
        for (const std::string & sidecar : names) {
            if (source_names.count(sidecar) != 0) {
                throw std::runtime_error(format(
                        "NanoQuant: source already contains reserved sidecar '%s'", sidecar.c_str()));
            }
        }
        groups.push_back(std::move(item));
    }
    if (groups.empty()) {
        throw std::runtime_error(format(
                "NanoQuant: architecture '%s' has no eligible decoder projections",
                loader.get_arch_name().c_str()));
    }
    std::sort(groups.begin(), groups.end(), [](const group & left, const group & right) {
        if (left.block != right.block) {
            return left.block < right.block;
        }
        if (left.weight->idx != right.weight->idx) {
            return left.weight->idx < right.weight->idx;
        }
        return left.weight->offs < right.weight->offs;
    });
    std::unordered_set<std::string> sidecars;
    for (const group & item : groups) {
        const std::string names[] = {
            item.name_v, item.name_u, item.name_scale_pre, item.name_scale_post,
        };
        for (const std::string & name : names) {
            if (!sidecars.insert(name).second) {
                throw std::runtime_error(format("NanoQuant: duplicate sidecar group member '%s'", name.c_str()));
            }
        }
    }
    return groups;
}

static std::vector<const llama_model_loader::llama_tensor_weight *> ordered_weights(
        llama_model_loader & loader) {
    std::vector<const llama_model_loader::llama_tensor_weight *> result;
    result.reserve(loader.weights_map.size());
    for (const auto & entry : loader.weights_map) {
        result.push_back(&entry.second);
    }
    std::sort(result.begin(), result.end(), [](const auto * left, const auto * right) {
        if (left->idx != right->idx) {
            return left->idx < right->idx;
        }
        return left->offs < right->offs;
    });
    return result;
}

static void apply_kv_overrides(
        gguf_context * output,
        const llama_model_quantize_params * params) {
    if (params->kv_overrides == nullptr) {
        return;
    }
    for (const llama_model_kv_override * override = params->kv_overrides;
         override->key[0] != '\0';
         ++override) {
        switch (override->tag) {
            case LLAMA_KV_OVERRIDE_TYPE_FLOAT:
                gguf_set_val_f32(output, override->key, override->val_f64);
                break;
            case LLAMA_KV_OVERRIDE_TYPE_INT:
                gguf_set_val_u32(output, override->key, uint32_t(std::abs(override->val_i64)));
                break;
            case LLAMA_KV_OVERRIDE_TYPE_BOOL:
                gguf_set_val_bool(output, override->key, override->val_bool);
                break;
            case LLAMA_KV_OVERRIDE_TYPE_STR:
                gguf_set_val_str(output, override->key, override->val_str);
                break;
            default:
                throw std::runtime_error(format(
                        "NanoQuant: unknown KV override type for '%s'", override->key));
        }
    }
}

static std::vector<float> load_weight(
        const llama_model * model,
        const group & item,
        int64_t expert,
        std::vector<no_init<uint8_t>> & read_data,
        std::vector<std::thread> & workers,
        int nthread) {
    const ggml_tensor * tensor = model->get_tensor(item.name.c_str());
    if (tensor == nullptr ||
        tensor->ne[0] != item.n_in || tensor->ne[1] != item.n_out ||
        tensor->ne[2] != item.n_expert || tensor->ne[3] != 1 ||
        expert < 0 || expert >= item.n_expert) {
        throw std::runtime_error(format(
                "NanoQuant: resident source tensor '%s' is missing or has the wrong shape",
                item.name.c_str()));
    }
    const int64_t elements = item.n_in*item.n_out;
    const size_t slice_bytes = ggml_row_size(tensor->type, item.n_in)*size_t(item.n_out);
    const size_t offset = size_t(expert)*slice_bytes;
    std::vector<float> result(elements);
    if (tensor->type == GGML_TYPE_F32) {
        GGML_ASSERT(slice_bytes == result.size()*sizeof(float));
        ggml_backend_tensor_get(
                tensor, result.data(), offset, result.size()*sizeof(float));
    } else {
        read_data.resize(slice_bytes);
        ggml_backend_tensor_get(tensor, read_data.data(), offset, read_data.size());
        ggml_tensor staging = *tensor;
        staging.ne[2] = 1;
        staging.buffer = nullptr;
        staging.data = read_data.data();
        llama_tensor_dequantize_to_f32(
                &staging, result.data(), workers, elements, nthread);
    }
    for (float value : result) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(format(
                    "NanoQuant: source tensor '%s' expert %" PRId64
                    " contains a non-finite value",
                    item.name.c_str(), expert));
        }
    }
    return result;
}

static void write_block_checkpoint(
        const std::filesystem::path & directory,
        int block,
        const hash256 & source_hash,
        const hash256 & config_hash) {
    char filename[64];
    std::snprintf(filename, sizeof(filename), "block-%05d.done", block);
    atomic_replace(directory / filename, [&](std::ostream & output) {
        static const char magic[8] = { 'N', 'Q', 'B', 'L', 'K', '1', '\0', '\0' };
        output.write(magic, sizeof(magic));
        output.write(reinterpret_cast<const char *>(source_hash.data()), sizeof(source_hash));
        output.write(reinterpret_cast<const char *>(config_hash.data()), sizeof(config_hash));
        write_pod(output, block);
    });
}

static void install_output_file(
        const std::filesystem::path & temporary,
        const std::filesystem::path & destination) {
    const std::filesystem::path backup = destination.string() + ".bak";
    std::error_code ec;
    std::filesystem::remove(backup, ec);
    ec.clear();
    if (std::filesystem::exists(destination)) {
        std::filesystem::rename(destination, backup, ec);
        if (ec) {
            throw std::runtime_error(format("NanoQuant: cannot rotate output '%s': %s",
                    destination.string().c_str(), ec.message().c_str()));
        }
    }
    std::filesystem::rename(temporary, destination, ec);
    if (ec) {
        if (std::filesystem::exists(backup)) {
            std::error_code restore_ec;
            std::filesystem::rename(backup, destination, restore_ec);
        }
        throw std::runtime_error(format("NanoQuant: cannot install output '%s': %s",
                destination.string().c_str(), ec.message().c_str()));
    }
    std::filesystem::remove(backup, ec);
}

using auxiliary_type_map = std::unordered_map<std::string, ggml_type>;

static bool is_auxiliary_weight(const std::string & name) {
    return name == "token_embd.weight" || name == "output.weight";
}

static bool can_quantize_auxiliary(const ggml_tensor * tensor, ggml_type type) {
    return ggml_n_dims(tensor) == 2 &&
           source_type_supports_f32(tensor->type) &&
           ggml_is_quantized(type) &&
           tensor->ne[0] % ggml_blck_size(type) == 0;
}

static size_t tensor_size_as(const ggml_tensor * tensor, ggml_type type) {
    if (type == tensor->type) {
        return ggml_nbytes(tensor);
    }
    if (!can_quantize_auxiliary(tensor, type)) {
        throw std::runtime_error(format(
                "NanoQuant: tensor '%s' cannot be converted to %s",
                ggml_get_name(tensor), ggml_type_name(type)));
    }
    const size_t rows = size_t(ggml_nrows(tensor));
    const size_t row_size = ggml_row_size(type, tensor->ne[0]);
    if (row_size != 0 && rows > std::numeric_limits<size_t>::max()/row_size) {
        throw std::runtime_error("NanoQuant: auxiliary tensor size overflow");
    }
    return rows*row_size;
}

static size_t checked_size_sum(size_t left, size_t right) {
    if (right > std::numeric_limits<size_t>::max() - left) {
        throw std::runtime_error("NanoQuant: physical size overflow");
    }
    return left + right;
}

static ggml_type auxiliary_type_at_level(const ggml_tensor * tensor, size_t level) {
    static constexpr ggml_type levels[][2] = {
        { GGML_TYPE_COUNT, GGML_TYPE_COUNT },
        { GGML_TYPE_Q8_0,  GGML_TYPE_COUNT },
        { GGML_TYPE_Q6_K,  GGML_TYPE_Q5_0  },
        { GGML_TYPE_Q5_K,  GGML_TYPE_Q5_0  },
        { GGML_TYPE_Q4_K,  GGML_TYPE_Q4_0  },
        { GGML_TYPE_Q3_K,  GGML_TYPE_Q4_0  },
        { GGML_TYPE_Q2_K,  GGML_TYPE_Q1_0  },
        { GGML_TYPE_Q1_0,  GGML_TYPE_COUNT },
    };
    if (level == 0) {
        return tensor->type;
    }
    for (ggml_type type : levels[level]) {
        if (type != GGML_TYPE_COUNT && can_quantize_auxiliary(tensor, type)) {
            return tensor_size_as(tensor, type) < ggml_nbytes(tensor) ? type : tensor->type;
        }
    }
    return GGML_TYPE_COUNT;
}

static auxiliary_type_map select_auxiliary_types(
        const std::vector<const llama_model_loader::llama_tensor_weight *> & weights,
        const std::unordered_map<std::string, size_t> & group_by_name,
        size_t alignment,
        size_t metadata_size,
        size_t target_file_size,
        size_t minimum_projection_size,
        size_t maximum_projection_size,
        uint64_t original_elements,
        float target_bits,
        bool allow_requantize,
        size_t & fixed_physical_data) {
    std::vector<ggml_tensor *> auxiliaries;
    size_t immutable_physical_data = 0;
    for (const auto * weight : weights) {
        ggml_tensor * tensor = weight->tensor;
        const std::string name = ggml_get_name(tensor);
        if (group_by_name.count(name) != 0) {
            continue;
        }
        if (is_auxiliary_weight(name) &&
            source_type_supports_f32(tensor->type) &&
            (!ggml_is_quantized(tensor->type) || allow_requantize) &&
            ggml_n_dims(tensor) == 2) {
            auxiliaries.push_back(tensor);
        } else {
            immutable_physical_data = checked_size_sum(
                    immutable_physical_data, GGML_PAD(ggml_nbytes(tensor), alignment));
        }
    }

    auxiliary_type_map selected;
    size_t selected_fixed_physical_data = 0;
    bool has_selected = false;
    size_t minimum_file_size = std::numeric_limits<size_t>::max();
    for (size_t level = 0; level < 8; ++level) {
        auxiliary_type_map candidate;
        size_t auxiliary_physical_data = 0;
        bool supported = true;
        for (ggml_tensor * tensor : auxiliaries) {
            const ggml_type type = auxiliary_type_at_level(tensor, level);
            if (type == GGML_TYPE_COUNT) {
                supported = false;
                break;
            }
            const size_t size = tensor_size_as(tensor, type);
            auxiliary_physical_data = checked_size_sum(
                    auxiliary_physical_data, GGML_PAD(size, alignment));
            if (type != tensor->type) {
                candidate.emplace(ggml_get_name(tensor), type);
            }
        }
        if (!supported) {
            continue;
        }
        const size_t fixed = checked_size_sum(immutable_physical_data, auxiliary_physical_data);
        const size_t minimum = checked_size_sum(
                checked_size_sum(metadata_size, fixed), minimum_projection_size);
        minimum_file_size = std::min(minimum_file_size, minimum);
        if (minimum <= target_file_size) {
            const size_t projection_budget =
                    target_file_size - metadata_size - fixed;
            selected = std::move(candidate);
            selected_fixed_physical_data = fixed;
            has_selected = true;
            if (projection_budget >= maximum_projection_size) {
                fixed_physical_data = selected_fixed_physical_data;
                return selected;
            }
        }
    }

    if (has_selected) {
        fixed_physical_data = selected_fixed_physical_data;
        return selected;
    }
    if (minimum_file_size == std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("NanoQuant: no supported auxiliary tensor layout");
    }
    throw std::runtime_error(format(
            "NanoQuant: target %.6f BPW is below the %.6f BPW minimum whole-file layout",
            target_bits, minimum_file_size*8.0/double(original_elements)));
}

static gguf_context_ptr make_output_metadata(
        llama_model_loader & loader,
        const std::vector<const llama_model_loader::llama_tensor_weight *> & weights,
        const std::vector<group> & groups,
        const std::unordered_map<std::string, size_t> & group_by_name,
        const auxiliary_type_map & auxiliary_types,
        const llama_model_quantize_params * params,
        uint64_t projection_elements,
        float projection_physical_bits,
        float tensor_data_bits,
        float file_bits) {
    gguf_context_ptr output { gguf_init_empty() };
    gguf_set_kv(output.get(), loader.metadata);
    apply_kv_overrides(output.get(), params);
    gguf_set_val_u32(output.get(),
            loader.llm_kv(LLM_KV_GENERAL_QUANTIZATION_VERSION).c_str(), GGML_QNT_VERSION);
    gguf_set_val_u32(output.get(),
            loader.llm_kv(LLM_KV_GENERAL_FILE_TYPE).c_str(), LLAMA_FTYPE_MOSTLY_NANOQUANT);
    gguf_set_val_u64(output.get(), "quantize.nanoquant.original_parameter_count", loader.n_elements);
    gguf_set_val_u64(output.get(), "quantize.nanoquant.projection_parameter_count", projection_elements);
    gguf_set_val_f32(output.get(), "quantize.nanoquant.target_bits", params->nanoquant_target_bits);
    gguf_set_val_f32(output.get(), "quantize.nanoquant.projection_physical_bits", projection_physical_bits);
    gguf_set_val_f32(output.get(), "quantize.nanoquant.tensor_data_bits", tensor_data_bits);
    gguf_set_val_f32(output.get(), "quantize.nanoquant.physical_bits", file_bits);
    gguf_remove_key(output.get(), loader.llm_kv(LLM_KV_SPLIT_NO).c_str());
    gguf_remove_key(output.get(), loader.llm_kv(LLM_KV_SPLIT_COUNT).c_str());
    gguf_remove_key(output.get(), loader.llm_kv(LLM_KV_SPLIT_TENSORS_COUNT).c_str());

    ggml_init_params descriptor_init = {
        std::max<size_t>(1, groups.size() * 4) * ggml_tensor_overhead(),
        nullptr,
        true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> descriptor_owner(
            ggml_init(descriptor_init), ggml_free);
    ggml_context * descriptor_context = descriptor_owner.get();
    if (descriptor_context == nullptr) {
        throw std::runtime_error("NanoQuant: failed to allocate sidecar descriptors");
    }

    for (const auto * weight : weights) {
        ggml_tensor * tensor = weight->tensor;
        const std::string name = ggml_get_name(tensor);
        const auto found = group_by_name.find(name);
        if (found == group_by_name.end()) {
            gguf_add_tensor(output.get(), tensor);
            const auto auxiliary = auxiliary_types.find(name);
            if (auxiliary != auxiliary_types.end()) {
                gguf_set_tensor_type(output.get(), name.c_str(), auxiliary->second);
            }
            continue;
        }
        const group & item = groups[found->second];
        ggml_tensor * v = ggml_new_tensor_3d(
                descriptor_context, GGML_TYPE_I32,
                (item.n_in + 31) / 32, item.rank, item.n_expert);
        ggml_tensor * u = ggml_new_tensor_3d(
                descriptor_context, GGML_TYPE_I32,
                (item.rank + 31) / 32, item.n_out, item.n_expert);
        ggml_tensor * scale_pre = ggml_new_tensor_2d(
                descriptor_context, GGML_TYPE_F16, item.n_in, item.n_expert);
        ggml_tensor * scale_post = ggml_new_tensor_2d(
                descriptor_context, GGML_TYPE_F16, item.n_out, item.n_expert);
        ggml_set_name(v, item.name_v.c_str());
        ggml_set_name(u, item.name_u.c_str());
        ggml_set_name(scale_pre, item.name_scale_pre.c_str());
        ggml_set_name(scale_post, item.name_scale_post.c_str());
        gguf_add_tensor(output.get(), v);
        gguf_add_tensor(output.get(), u);
        gguf_add_tensor(output.get(), scale_pre);
        gguf_add_tensor(output.get(), scale_post);
    }
    return output;
}

static size_t write_quantized_auxiliary(
        std::ofstream & output,
        llama_model_loader & loader,
        ggml_tensor * tensor,
        ggml_type type,
        std::vector<no_init<uint8_t>> & read_data,
        std::vector<no_init<float>> & conversion,
        std::vector<std::thread> & workers,
        int nthread) {
    const size_t source_size = ggml_nbytes(tensor);
    if (loader.use_mmap) {
        tensor->data = nullptr;
    } else {
        read_data.resize(source_size);
        tensor->data = read_data.data();
    }
    loader.load_data_for(tensor);

    static constexpr size_t CONVERSION_MEMORY_BUDGET = 64u * 1024u * 1024u;
    const int64_t n_per_row = tensor->ne[0];
    const int64_t nrows = ggml_nrows(tensor);
    const int64_t rows_per_chunk = std::max<int64_t>(
            1, int64_t(CONVERSION_MEMORY_BUDGET/(sizeof(float)*size_t(n_per_row))));
    const size_t row_size = ggml_row_size(type, n_per_row);
    std::vector<no_init<uint8_t>> quantized;
    size_t written = 0;
    for (int64_t first_row = 0; first_row < nrows; first_row += rows_per_chunk) {
        const int64_t rows = std::min(nrows - first_row, rows_per_chunk);
        const size_t elements = size_t(rows)*size_t(n_per_row);
        const size_t element_offset = size_t(first_row)*size_t(n_per_row);
        const float * f32_data = nullptr;
        if (tensor->type == GGML_TYPE_F32) {
            f32_data = reinterpret_cast<const float *>(tensor->data) + element_offset;
        } else {
            conversion.resize(elements);
            ggml_tensor staging = *tensor;
            staging.buffer = nullptr;
            staging.data = reinterpret_cast<uint8_t *>(tensor->data) +
                    size_t(first_row)*ggml_row_size(tensor->type, n_per_row);
            llama_tensor_dequantize_to_f32(
                    &staging, reinterpret_cast<float *>(conversion.data()),
                    workers, elements, nthread);
            f32_data = reinterpret_cast<const float *>(conversion.data());
        }
        const size_t chunk_size = size_t(rows)*row_size;
        quantized.resize(chunk_size);
        const size_t actual_size = llama_tensor_quantize_impl(
                type, f32_data, quantized.data(), elements,
                rows, n_per_row, nullptr, workers, nthread);
        if (actual_size != chunk_size) {
            throw std::runtime_error("NanoQuant: auxiliary quantization size mismatch");
        }
        output.write(reinterpret_cast<const char *>(quantized.data()), actual_size);
        written = checked_size_sum(written, actual_size);
    }
    return written;
}

static void write_grouped_gguf(
        llama_model_loader & loader,
        gguf_context * metadata_context,
        const std::vector<const llama_model_loader::llama_tensor_weight *> & weights,
        const std::vector<group> & groups,
        const std::unordered_map<std::string, size_t> & group_by_name,
        const auxiliary_type_map & auxiliary_types,
        const std::filesystem::path & checkpoint_directory,
        const hash256 & source_hash,
        const hash256 & config_hash,
        const std::filesystem::path & path,
        checkpoint_stage required_stage,
        size_t projected_payload,
        size_t projected_physical_data,
        size_t projected_file_size,
        size_t metadata_size,
        size_t alignment,
        std::vector<no_init<uint8_t>> & read_data,
        std::vector<no_init<float>> & conversion,
        std::vector<std::thread> & workers,
        int nthread) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    zeros(output, metadata_size);
    size_t written_payload = 0;
    size_t written_physical_data = 0;

    for (const auto * weight : weights) {
        ggml_tensor * tensor = weight->tensor;
        const std::string name = ggml_get_name(tensor);
        const auto found = group_by_name.find(name);
        if (found == group_by_name.end()) {
            const auto auxiliary = auxiliary_types.find(name);
            const size_t size = auxiliary == auxiliary_types.end() ?
                    ggml_nbytes(tensor) : tensor_size_as(tensor, auxiliary->second);
            if (auxiliary == auxiliary_types.end()) {
                if (loader.use_mmap) {
                    tensor->data = nullptr;
                } else {
                    read_data.resize(size);
                    tensor->data = read_data.data();
                }
                loader.load_data_for(tensor);
                output.write(reinterpret_cast<const char *>(tensor->data), size);
            } else {
                const size_t actual_size = write_quantized_auxiliary(
                        output, loader, tensor, auxiliary->second,
                        read_data, conversion, workers, nthread);
                if (actual_size != size) {
                    throw std::runtime_error("NanoQuant: auxiliary tensor size mismatch");
                }
            }
            gguf_set_tensor_data(metadata_context, name.c_str(), tensor->data);
            zeros(output, GGML_PAD(size, alignment) - size);
            written_payload += size;
            written_physical_data += GGML_PAD(size, alignment);
            continue;
        }

        const group & item = groups[found->second];
        checkpoint_state state;
        if (!load_checkpoint_file(
                    checkpoint_path(checkpoint_directory, item.name),
                    source_hash, config_hash, item, state)) {
            throw std::runtime_error(format(
                    "NanoQuant: missing output checkpoint for '%s'", item.name.c_str()));
        }
        validate_state_shapes(item, state);
        if (state.stage < required_stage) {
            throw std::runtime_error(format(
                    "NanoQuant: refusing partial output; '%s' has stage %u, required %u",
                    item.name.c_str(), uint32_t(state.stage), uint32_t(required_stage)));
        }
        const size_t n_scale_pre = size_t(item.n_in)*size_t(item.n_expert);
        const size_t n_scale_post = size_t(item.n_out)*size_t(item.n_expert);
        std::vector<ggml_fp16_t> scale_pre(n_scale_pre);
        std::vector<ggml_fp16_t> scale_post(n_scale_post);
        ggml_fp32_to_fp16_row(state.scale_pre.data(), scale_pre.data(), n_scale_pre);
        ggml_fp32_to_fp16_row(state.scale_post.data(), scale_post.data(), n_scale_post);

        auto write_sidecar = [&](const std::string & sidecar, const void * data, size_t size) {
            const int64_t tensor_index = gguf_find_tensor(metadata_context, sidecar.c_str());
            if (tensor_index < 0 ||
                gguf_get_tensor_size(metadata_context, tensor_index) != size) {
                throw std::runtime_error(format(
                        "NanoQuant: GGUF sidecar size validation failed for '%s'", sidecar.c_str()));
            }
            gguf_set_tensor_data(metadata_context, sidecar.c_str(), data);
            output.write(reinterpret_cast<const char *>(data), size);
            zeros(output, GGML_PAD(size, alignment) - size);
            written_payload += size;
            written_physical_data += GGML_PAD(size, alignment);
        };
        write_sidecar(item.name_v, state.packed_v.data(), item.v_size());
        write_sidecar(item.name_u, state.packed_u.data(), item.u_size());
        write_sidecar(item.name_scale_pre, scale_pre.data(), item.scale_pre_size());
        write_sidecar(item.name_scale_post, scale_post.data(), item.scale_post_size());
    }
    if (written_payload != projected_payload ||
        written_physical_data != projected_physical_data) {
        throw std::runtime_error(
                "NanoQuant: GGUF accounting differs from the validated dry-layout projection");
    }
    output.seekp(0);
    std::vector<uint8_t> metadata(metadata_size);
    gguf_get_meta_data(metadata_context, metadata.data());
    output.write(reinterpret_cast<const char *>(metadata.data()), metadata.size());
    output.close();
    if (std::filesystem::file_size(path) != projected_file_size) {
        throw std::runtime_error(
                "NanoQuant: GGUF physical size differs from the dry-layout projection");
    }
}

struct model_fit_config {
    ggml_backend_dev_t training_device = nullptr;
    size_t n_model_devices = 0;
    size_t n_contexts = 1;
};

struct model_fit_probe {
    int32_t max_gpu_layers = 0;
    size_t n_gpu_devices = 0;
    bool fits = true;
};

struct model_fit_result {
    int32_t n_gpu_layers = 0;
    int32_t max_gpu_layers = 0;
};

// the device that also holds the training tensors needs headroom beyond the source model itself
static size_t model_device_reserve(
        ggml_backend_dev_t device,
        const model_fit_config & config) {
    const bool is_meta = ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_META;
    const size_t reserve_count = is_meta ? std::max<size_t>(config.n_model_devices, 1) : 1;
    if (reserve_count > std::numeric_limits<size_t>::max()/MODEL_DEVICE_RESERVE) {
        throw std::runtime_error("NanoQuant: device reserve size overflow");
    }
    size_t reserve = MODEL_DEVICE_RESERVE*reserve_count;
    if (device == config.training_device || is_meta) {
        size_t free = 0;
        size_t total = 0;
        ggml_backend_dev_memory(device, &free, &total);
        reserve += std::min(MODEL_DEVICE_RESERVE, total/4);
    }
    return reserve;
}

static model_fit_probe probe_model_fit(
        const std::string & path,
        llama_model_params model_params,
        llama_context_params context_params,
        const model_fit_config & config) {
    model_params.no_alloc = true;
    model_params.load_mode = LLAMA_LOAD_MODE_NONE;
    model_params.check_tensors = false;
    const enum llama_flash_attn_type flash_attn_type = context_params.flash_attn_type;
    // the projection context runs without flash attention, which needs the largest compute buffers
    context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    context_params.cb_eval = nullptr;
    context_params.cb_eval_user_data = nullptr;

    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(path.c_str(), model_params), llama_model_free);
    if (!model) {
        throw std::runtime_error("NanoQuant: failed to probe source model memory");
    }
    std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_init_from_model(model.get(), context_params), llama_free);
    if (!context) {
        // architectures that require flash attention only support the caller's setting
        context_params.flash_attn_type = flash_attn_type;
        context.reset(llama_init_from_model(model.get(), context_params));
    }
    if (!context) {
        throw std::runtime_error("NanoQuant: failed to probe source context memory");
    }

    model_fit_probe result;
    result.max_gpu_layers =
            llama_model_n_layer(model.get()) + llama_model_n_layer_nextn(model.get()) + 1;
    // the reconstruction phase keeps several contexts on the same model alive at once
    const size_t n_contexts = std::max<size_t>(config.n_contexts, 1);
    std::unordered_map<ggml_backend_dev_t, size_t> device_usage;
    for (const auto & [buft, memory] : llama_get_memory_breakdown(context.get())) {
        if (ggml_backend_buft_is_host(buft)) {
            continue;
        }
        ggml_backend_dev_t device = ggml_backend_buft_get_device(buft);
        if (device == nullptr) {
            continue;
        }
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(device);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU &&
            type != GGML_BACKEND_DEVICE_TYPE_IGPU &&
            type != GGML_BACKEND_DEVICE_TYPE_META) {
            continue;
        }
        const size_t per_context = memory.context + memory.compute;
        if (per_context > (std::numeric_limits<size_t>::max() - memory.model)/n_contexts) {
            throw std::runtime_error("NanoQuant: source model memory size overflow");
        }
        const size_t bytes = memory.model + per_context*n_contexts;
        size_t & used = device_usage[device];
        if (bytes > std::numeric_limits<size_t>::max() - used) {
            throw std::runtime_error("NanoQuant: source model memory size overflow");
        }
        used += bytes;
    }

    result.n_gpu_devices = device_usage.size();
    for (const auto & [device, used] : device_usage) {
        size_t free = 0;
        size_t total = 0;
        ggml_backend_dev_memory(device, &free, &total);
        const size_t reserve = model_device_reserve(device, config);
        const size_t available = free > reserve ? free - reserve : 0;
        if (used > available) {
            result.fits = false;
        }
    }
    return result;
}

static model_fit_result fit_model_gpu_layers(
        const std::string & path,
        llama_model_params model_params,
        const llama_context_params & context_params,
        const model_fit_config & config) {
    model_params.n_gpu_layers = -1;
    model_fit_probe probe = probe_model_fit(path, model_params, context_params, config);

    model_fit_result result;
    result.max_gpu_layers = probe.max_gpu_layers;
    if (probe.max_gpu_layers <= 0 || probe.n_gpu_devices == 0) {
        return result;
    }
    if (probe.fits) {
        result.n_gpu_layers = probe.max_gpu_layers;
        return result;
    }

    LLAMA_LOG_INFO(
            "NanoQuant: model exceeds available device memory; finding the maximum GPU layer count\n");
    int32_t first = 1;
    int32_t last = probe.max_gpu_layers - 1;
    while (first <= last) {
        const int32_t candidate = first + (last - first)/2;
        model_params.n_gpu_layers = candidate;
        probe = probe_model_fit(path, model_params, context_params, config);
        if (probe.fits && probe.n_gpu_devices > 0) {
            result.n_gpu_layers = candidate;
            first = candidate + 1;
        } else {
            last = candidate - 1;
        }
    }
    LLAMA_LOG_INFO(
            "NanoQuant: model uses %d/%d GPU layers; %d remain host-resident\n",
            result.n_gpu_layers, result.max_gpu_layers,
            result.max_gpu_layers - result.n_gpu_layers);
    return result;
}

// an over-optimistic memory estimate must degrade towards the CPU instead of aborting the run
static bool reduce_model_gpu_layers(llama_model_params & model_params, const char * what) {
    if (model_params.n_gpu_layers <= 0) {
        return false;
    }
    const int32_t next = model_params.n_gpu_layers/2;
    LLAMA_LOG_WARN(
            "NanoQuant: %s does not fit with %d GPU layers; retrying with %d\n",
            what, model_params.n_gpu_layers, next);
    model_params.n_gpu_layers = next;
    return true;
}


static void quantize(
        const std::string & input_path,
        const std::string & output_path,
        const llama_model_quantize_params * params) {
    validate_options(params);
    int nthread = params->nthread;
    if (nthread <= 0) {
        nthread = std::max(1u, std::thread::hardware_concurrency());
    }

#if defined(__linux__) || defined(_WIN32)
    constexpr llama_load_mode load_mode = LLAMA_LOAD_MODE_MMAP;
#else
    constexpr llama_load_mode load_mode = LLAMA_LOAD_MODE_NONE;
#endif
    std::vector<std::string> splits;
    llama_model_loader loader(
            nullptr, nullptr, nullptr, input_path, splits, nullptr,
            load_mode, true, false, params->kv_overrides, nullptr);
    std::vector<group> groups = find_groups(loader, params->allow_requantize);
    const std::vector<const llama_model_loader::llama_tensor_weight *> weights =
            ordered_weights(loader);
    std::unordered_map<std::string, size_t> group_by_name;
    group_by_name.reserve(groups.size());
    for (size_t i = 0; i < groups.size(); ++i) {
        group_by_name.emplace(groups[i].name, i);
    }

    size_t original_payload = 0;
    uint64_t projection_elements = 0;
    for (const auto * weight : weights) {
        original_payload = checked_size_sum(original_payload, ggml_nbytes(weight->tensor));
    }
    for (const group & item : groups) {
        const uint64_t matrix_elements = uint64_t(item.n_in)*uint64_t(item.n_out);
        if (matrix_elements > std::numeric_limits<uint64_t>::max()/uint64_t(item.n_expert)) {
            throw std::runtime_error("NanoQuant: projection element count overflow");
        }
        const uint64_t elements = matrix_elements*uint64_t(item.n_expert);
        if (elements > std::numeric_limits<uint64_t>::max() - projection_elements) {
            throw std::runtime_error("NanoQuant: projection element count overflow");
        }
        projection_elements += elements;
    }

    for (group & item : groups) {
        item.rank = std::min<int64_t>(2, std::min(item.n_in, item.n_out));
    }
    const auxiliary_type_map no_auxiliary_types;
    gguf_context_ptr trial_metadata = make_output_metadata(
            loader, weights, groups, group_by_name, no_auxiliary_types, params,
            projection_elements, 0.0f, 0.0f, 0.0f);
    const size_t alignment = gguf_get_alignment(trial_metadata.get());
    size_t metadata_size = gguf_get_meta_size(trial_metadata.get());
    trial_metadata.reset();

    size_t minimum_projection_size = 0;
    size_t maximum_projection_size = 0;
    for (group & item : groups) {
        item.rank = 1;
        minimum_projection_size = checked_size_sum(
                minimum_projection_size, item.physical_size(alignment));
        item.rank = std::min(item.n_in, item.n_out);
        maximum_projection_size = checked_size_sum(
                maximum_projection_size, item.physical_size(alignment));
        item.rank = 1;
    }
    const long double requested_file_bytes = std::floor(
            (long double) loader.n_elements*(long double) params->nanoquant_target_bits/8.0L);
    if (requested_file_bytes > (long double) std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("NanoQuant: target file byte budget is out of range");
    }
    const size_t target_file_size = size_t(requested_file_bytes);
    size_t fixed_physical_data = 0;
    const auxiliary_type_map auxiliary_types = select_auxiliary_types(
            weights, group_by_name, alignment, metadata_size, target_file_size,
            minimum_projection_size, maximum_projection_size,
            loader.n_elements, params->nanoquant_target_bits,
            params->allow_requantize, fixed_physical_data);
    for (const auto & auxiliary : auxiliary_types) {
        ggml_tensor * tensor = loader.get_tensor_meta(auxiliary.first.c_str());
        LLAMA_LOG_INFO(
                "NanoQuant: auxiliary %-36s %s -> %s\n",
                auxiliary.first.c_str(), ggml_type_name(tensor->type),
                ggml_type_name(auxiliary.second));
    }
    const size_t projection_budget = target_file_size - metadata_size - fixed_physical_data;
    const size_t projection_physical_data =
            allocate_ranks(groups, alignment, projection_budget);
    const size_t projected_physical_data =
            checked_size_sum(fixed_physical_data, projection_physical_data);
    size_t projected_file_size =
            checked_size_sum(metadata_size, projected_physical_data);
    if (projected_file_size > target_file_size) {
        throw std::runtime_error("NanoQuant: whole-file layout exceeded its target budget");
    }

    size_t projected_payload = 0;
    for (const auto * weight : weights) {
        ggml_tensor * tensor = weight->tensor;
        const std::string name = ggml_get_name(tensor);
        const auto found = group_by_name.find(name);
        if (found != group_by_name.end()) {
            projected_payload = checked_size_sum(
                    projected_payload, groups[found->second].payload_size());
            continue;
        }
        const auto auxiliary = auxiliary_types.find(name);
        const size_t size = auxiliary == auxiliary_types.end() ?
                ggml_nbytes(tensor) : tensor_size_as(tensor, auxiliary->second);
        projected_payload = checked_size_sum(projected_payload, size);
    }

    gguf_context_ptr output_metadata = make_output_metadata(
            loader, weights, groups, group_by_name, auxiliary_types, params,
            projection_elements,
            float(projection_physical_data*8.0/projection_elements),
            float(projected_physical_data*8.0/loader.n_elements),
            float(projected_file_size*8.0/loader.n_elements));
    const size_t actual_metadata_size = gguf_get_meta_size(output_metadata.get());
    if (actual_metadata_size > metadata_size) {
        throw std::runtime_error("NanoQuant: GGUF metadata exceeded its conservative budget");
    }
    if (actual_metadata_size != metadata_size) {
        metadata_size = actual_metadata_size;
        projected_file_size = checked_size_sum(metadata_size, projected_physical_data);
        output_metadata = make_output_metadata(
                loader, weights, groups, group_by_name, auxiliary_types, params,
                projection_elements,
                float(projection_physical_data*8.0/projection_elements),
                float(projected_physical_data*8.0/loader.n_elements),
                float(projected_file_size*8.0/loader.n_elements));
        if (gguf_get_meta_size(output_metadata.get()) != metadata_size) {
            throw std::runtime_error("NanoQuant: GGUF metadata size is unstable");
        }
    }

    for (const group & item : groups) {
        LLAMA_LOG_INFO(
                "NanoQuant dry-layout: %-36s [%6" PRId64 ", %6" PRId64 ", %4" PRId64 "] rank=%5" PRId64
                " payload=%zu bytes physical=%zu bytes\n",
                item.name.c_str(), item.n_in, item.n_out, item.n_expert, item.rank,
                item.payload_size(), item.physical_size(alignment));
    }
    LLAMA_LOG_INFO(
            "NanoQuant: selected projection data = %zu bytes (%.6f bits/selected parameter)\n",
            projection_physical_data, projection_physical_data * 8.0 / projection_elements);
    LLAMA_LOG_INFO(
            "NanoQuant: projected tensor payload = %zu bytes; aligned tensor data = %zu bytes\n",
            projected_payload, projected_physical_data);
    LLAMA_LOG_INFO(
            "NanoQuant: exact projected GGUF size = %zu bytes (%zu metadata + %zu aligned tensor data); "
            "target=%.6f BPW achieved=%.6f BPW slack=%.6f BPW\n",
            projected_file_size, metadata_size, projected_physical_data,
            params->nanoquant_target_bits,
            projected_file_size*8.0/loader.n_elements,
            (target_file_size - projected_file_size)*8.0/loader.n_elements);
    if (params->dry_run) {
        return;
    }

    if (output_path.empty()) {
        throw std::runtime_error("NanoQuant: output path is empty");
    }
    if (std::error_code ec; std::filesystem::equivalent(input_path, output_path, ec)) {
        throw std::runtime_error("NanoQuant: input and output files must differ");
    }
    LLAMA_LOG_INFO(
            "NanoQuant: fingerprinting %zu source model file(s) for checkpoint identity "
            "(at most 4 MiB read per file)\n",
            splits.empty() ? size_t(1) : splits.size());
    const hash256 source_hash = hash_model_files(input_path, splits);
    const hash256 dataset_hash = hash_file(params->nanoquant_calibration_dataset);
    const hash256 config_hash =
            make_config_hash(params, nthread, dataset_hash, groups);
    const std::filesystem::path checkpoint_directory =
            params->nanoquant_checkpoint_directory != nullptr &&
            params->nanoquant_checkpoint_directory[0] != '\0'
            ? std::filesystem::path(params->nanoquant_checkpoint_directory)
            : std::filesystem::path(output_path + ".nq-checkpoint");

    if (params->nanoquant_resume) {
        validate_manifest(
                checkpoint_directory, source_hash, dataset_hash, config_hash, groups);
    } else {
        if (std::filesystem::exists(checkpoint_directory)) {
            if (!std::filesystem::is_directory(checkpoint_directory) ||
                std::filesystem::directory_iterator(checkpoint_directory) !=
                        std::filesystem::directory_iterator()) {
                throw std::runtime_error(format(
                        "NanoQuant: checkpoint directory '%s' is not empty; use --nanoquant-resume or another directory",
                        checkpoint_directory.string().c_str()));
            }
        } else {
            std::filesystem::create_directories(checkpoint_directory);
        }
        write_manifest(
                checkpoint_directory, source_hash, dataset_hash, config_hash, groups);
    }
    LLAMA_LOG_INFO("NanoQuant: source hash=%s dataset hash=%s config hash=%s\n",
            hash_hex(source_hash).c_str(), hash_hex(dataset_hash).c_str(), hash_hex(config_hash).c_str());

    calibration_collector_set collector_set;
    compute_backend backend(params->nanoquant_device);
    llama_model_params model_params = llama_model_default_params();
    std::vector<ggml_backend_dev_t> model_devices;
    std::vector<float> model_tensor_split(llama_max_devices(), 0.0f);
    model_params.load_mode = load_mode;
    ggml_backend_dev_t training_device = ggml_backend_get_device(backend.backend);
    const bool cpu_training =
            ggml_backend_dev_type(training_device) == GGML_BACKEND_DEVICE_TYPE_CPU;
    if (cpu_training) {
        model_devices.push_back(training_device);
    } else {
        ggml_backend_reg_t training_reg = ggml_backend_dev_backend_reg(training_device);
        for (size_t i = 0; i < ggml_backend_reg_dev_count(training_reg); ++i) {
            ggml_backend_dev_t device = ggml_backend_reg_dev_get(training_reg, i);
            const enum ggml_backend_dev_type type = ggml_backend_dev_type(device);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU ||
                type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                model_devices.push_back(device);
            }
        }
        model_params.split_mode = LLAMA_SPLIT_MODE_LAYER;
        if (model_devices.empty()) {
            LLAMA_LOG_WARN(
                    "NanoQuant: backend %s exposes no model devices; the source model stays host-resident\n",
                    ggml_backend_reg_name(training_reg));
        } else {
            LLAMA_LOG_INFO(
                    "NanoQuant: source layers are distributed across %zu devices from backend %s\n",
                    model_devices.size(), ggml_backend_reg_name(training_reg));
        }
    }
    model_fit_config fit_config;
    fit_config.training_device = cpu_training ? nullptr : training_device;
    fit_config.n_model_devices = model_devices.size();
    // the projection and block reconstruction contexts are alive at the same time
    fit_config.n_contexts = 2;
    if (!cpu_training && !model_devices.empty()) {
        GGML_ASSERT(model_devices.size() <= model_tensor_split.size());
        double usable_gib = 0.0;
        for (size_t i = 0; i < model_devices.size(); ++i) {
            size_t free = 0;
            size_t total = 0;
            ggml_backend_dev_memory(model_devices[i], &free, &total);
            const size_t reserve = model_device_reserve(model_devices[i], fit_config);
            const size_t usable = free > reserve ? free - reserve : 0;
            model_tensor_split[i] = float(usable/double(1024u*1024u*1024u));
            usable_gib += model_tensor_split[i];
            LLAMA_LOG_INFO(
                    "NanoQuant: source device %s has %.2f GiB available for model layers\n",
                    ggml_backend_dev_name(model_devices[i]), model_tensor_split[i]);
        }
        if (usable_gib > 0.0) {
            model_params.tensor_split = model_tensor_split.data();
        } else {
            LLAMA_LOG_WARN(
                    "NanoQuant: no device has memory left for model layers; the source model stays host-resident\n");
        }
    }
    model_devices.push_back(nullptr);
    model_params.devices = model_devices.data();
    const int32_t configured_model_gpu_layers =
            cpu_training ? 0 : params->nanoquant_n_gpu_layers;
    model_params.n_gpu_layers = configured_model_gpu_layers;
    model_params.check_tensors = true;
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = params->nanoquant_sequence_length;
    context_params.n_batch = params->nanoquant_sequence_length;
    context_params.n_ubatch = params->nanoquant_sequence_length;
    context_params.n_seq_max = 1;
    context_params.n_threads = nthread;
    context_params.n_threads_batch = nthread;
    context_params.no_perf = true;
    if (model_params.n_gpu_layers < 0) {
        model_params.n_gpu_layers =
                fit_model_gpu_layers(input_path, model_params, context_params, fit_config).n_gpu_layers;
    }
    std::vector<bool> projection_reachable(groups.size(), false);
    projection_reachability reachability { group_by_name, projection_reachable };
    context_params.cb_eval = projection_reachability_callback;
    context_params.cb_eval_user_data = &reachability;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> teacher(nullptr, llama_model_free);
    std::unique_ptr<llama_context, decltype(&llama_free)> teacher_context(nullptr, llama_free);
    while (true) {
        teacher.reset(llama_model_load_from_file(input_path.c_str(), model_params));
        if (teacher) {
            teacher_context.reset(llama_init_from_model(teacher.get(), context_params));
            if (teacher_context) {
                break;
            }
            teacher.reset();
        }
        if (!reduce_model_gpu_layers(model_params, "the source teacher model")) {
            throw std::runtime_error("NanoQuant: failed to load the source teacher model");
        }
    }
    const std::vector<llama_token> samples = make_calibration_samples(teacher.get(), params);
    LLAMA_LOG_INFO(
            "NanoQuant phase 1/3: calibrated %d deterministic samples x %d tokens; projection cache <= %zu bytes/block\n",
            params->nanoquant_sample_count, params->nanoquant_sequence_length,
            PROJECTION_MEMORY_BUDGET);

    std::vector<no_init<uint8_t>> read_data;
    std::vector<no_init<float>> conversion;
    std::vector<std::thread> workers;
    workers.reserve(nthread);

    LLAMA_LOG_INFO("NanoQuant phase 2/3: independent transformer-block reconstruction\n");
    std::vector<checkpoint_state> model_states(groups.size());
    std::vector<size_t> projection_targets;
    projection_targets.reserve(groups.size());
    for (size_t index = 0; index < groups.size(); ++index) {
        const group & item = groups[index];
        checkpoint_state & state = model_states[index];
        if (params->nanoquant_resume &&
            load_checkpoint_file(
                    checkpoint_path(checkpoint_directory, item.name),
                    source_hash, config_hash, item, state)) {
            validate_state_shapes(item, state);
        }
        if (state.stage == checkpoint_stage::EXPERT_DONE) {
            complete_expert(
                    checkpoint_directory, source_hash, config_hash, item, state);
            save_checkpoint(checkpoint_directory, source_hash, config_hash, item, state);
            validate_state_shapes(item, state);
        }
        if (state.stage >= checkpoint_stage::GROUP_DONE) {
            release_completed_state(state);
            release_attached_state(state);
        }
        if (state.stage < checkpoint_stage::GROUP_DONE) {
            projection_targets.push_back(index);
        }
    }
    const size_t first_block = 0;

    owned_batch projection_batch(params->nanoquant_sequence_length);
    const std::vector<std::vector<float>> output_importance =
            collect_output_importance(
                    teacher_context.get(), groups, projection_targets,
                    projection_reachable,
                    samples, backend.gradient_memory_budget(),
                    params, projection_batch.value);
    if (projection_targets.empty()) {
        llama_batch & reachability_batch = projection_batch.value;
        reachability_batch.n_tokens = params->nanoquant_sequence_length;
        const llama_token * tokens = calibration_sample(samples, 0, params);
        for (int32_t token = 0; token < reachability_batch.n_tokens; ++token) {
            reachability_batch.token[token] = tokens[token];
            reachability_batch.pos[token] = token;
            reachability_batch.n_seq_id[token] = 1;
            reachability_batch.seq_id[token][0] = 0;
            reachability_batch.logits[token] = token + 1 == reachability_batch.n_tokens;
        }
        llama_memory_clear(llama_get_memory(teacher_context.get()), true);
        if (const int result = llama_decode(teacher_context.get(), reachability_batch); result != 0) {
            throw std::runtime_error(format(
                    "NanoQuant: projection reachability probe failed (code %d)", result));
        }
    }
    teacher_context.reset();
    context_params.cb_eval = nullptr;
    context_params.cb_eval_user_data = nullptr;

    llama_context_params projection_context_params = context_params;
    projection_context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    projection_context_params.cb_eval = calibration_callback;
    projection_context_params.cb_eval_user_data = &collector_set;
    std::unique_ptr<llama_context, decltype(&llama_free)> projection_context(
            llama_init_from_model(teacher.get(), projection_context_params), llama_free);
    if (!projection_context) {
        throw std::runtime_error("NanoQuant: failed to create the teacher projection context");
    }
    block_output_collector block_collector;
    llama_context_params block_context_params = context_params;
    block_context_params.cb_eval = block_output_callback;
    block_context_params.cb_eval_user_data = &block_collector;
    std::unique_ptr<llama_context, decltype(&llama_free)> block_context(
            llama_init_from_model(teacher.get(), block_context_params), llama_free);
    if (!block_context) {
        throw std::runtime_error("NanoQuant: failed to create the block reconstruction context");
    }
    LLAMA_LOG_INFO(
            "NanoQuant: source and reconstructed weights use disjoint residency phases; "
            "optimizer parameters are allocated one group at a time\n");

    for (size_t block_begin = first_block; block_begin < groups.size();) {
        size_t block_end = block_begin + 1;
        while (block_end < groups.size() &&
               groups[block_end].block == groups[block_begin].block) {
            ++block_end;
        }

        bool reconstruct_block = false;
        for (size_t index = block_begin; index < block_end; ++index) {
            if (model_states[index].stage < checkpoint_stage::GROUP_DONE &&
                projection_reachable[index]) {
                reconstruct_block = true;
                break;
            }
        }
        block_training_data training_data;
        std::vector<std::vector<calibration_data>> calibrations;
        if (reconstruct_block) {
            training_data = collect_block_training_data(
                    block_context.get(), block_collector,
                    groups[block_begin].block, samples, params);
            const auto projection_start = std::chrono::steady_clock::now();
            calibrations = collect_projections(
                    projection_context.get(), collector_set, groups, model_states,
                    projection_reachable,
                    block_begin, block_end, samples, output_importance,
                    params, projection_batch.value);
            const auto projection_end = std::chrono::steady_clock::now();
            LLAMA_LOG_INFO(
                    "NanoQuant profile: block %d activation collection=%.3fs targets=%zu\n",
                    groups[block_begin].block,
                    std::chrono::duration<double>(
                            projection_end - projection_start).count(),
                    block_end - block_begin);

        }
        for (size_t index = block_begin; index < block_end; ++index) {
            const group & item = groups[index];
            checkpoint_state & state = model_states[index];
            const bool reachable = projection_reachable[index];
            while (state.stage < checkpoint_stage::GROUP_DONE) {
                if (state.stage == checkpoint_stage::EXPERT_DONE) {
                    complete_expert(
                            checkpoint_directory, source_hash, config_hash, item, state);
                    save_checkpoint(
                            checkpoint_directory, source_hash, config_hash, item, state);
                    continue;
                }
                const uint32_t expert = state.expert;
                if (state.stage < checkpoint_stage::FACTOR && state.weight.empty()) {
                    state.weight = load_weight(
                            teacher.get(), item, expert, read_data, workers, nthread);
                    release_vector(read_data);
                }

                calibration_data fallback_calibration;
                const calibration_data * calibration = nullptr;
                if (reachable) {
                    calibration = &calibrations.at(index - block_begin).at(expert);
                    if (item.n_expert > 1) {
                        LLAMA_LOG_INFO(
                                "NanoQuant: block %d group %s expert %u/%" PRId64
                                " captured %" PRId64 " routed projection rows\n",
                                item.block, item.name.c_str(), expert + 1,
                                item.n_expert, calibration->rows);
                    } else {
                        LLAMA_LOG_INFO(
                                "NanoQuant: block %d group %s captured %" PRId64
                                " of %" PRId64 " projection rows\n",
                                item.block, item.name.c_str(), calibration->rows,
                                int64_t(params->nanoquant_sample_count)*
                                        params->nanoquant_sequence_length);
                    }
                } else {
                    fallback_calibration.input_norm.assign(size_t(item.n_in), 1.0f);
                    fallback_calibration.output_norm.assign(size_t(item.n_out), 1.0f);
                    fallback_calibration.repeated_use = true;
                    calibration = &fallback_calibration;
                }

                const auto nonfactor_start = std::chrono::steady_clock::now();
                if (reachable && item.n_expert == 1) {
                    run_nonfactor_reconstruction(
                            backend, item, samples, training_data, params, teacher.get(),
                            block_context.get(), block_collector, state);
                } else if (state.stage < checkpoint_stage::ADMM) {
                    state.stage = checkpoint_stage::NONFACTOR;
                    state.progress = uint32_t(params->nanoquant_nonfactor_epochs);
                    state.optimizer_step = 0;
                    release_vector(state.weight_first_moment);
                    release_vector(state.weight_second_moment);
                    if (item.n_expert > 1) {
                        LLAMA_LOG_INFO(
                                "NanoQuant: block %d group %s expert %u/%" PRId64
                                " uses the source expert weight for factorization\n",
                                item.block, item.name.c_str(), expert + 1, item.n_expert);
                    } else {
                        LLAMA_LOG_INFO(
                                "NanoQuant: block %d group %s is absent from the primary graph; "
                                "using uniform factorization weights\n",
                                item.block, item.name.c_str());
                    }
                }
                const auto admm_start = std::chrono::steady_clock::now();
                run_admm(backend, item, *calibration, params, state);
                const auto factor_start = std::chrono::steady_clock::now();
                run_factor_reconstruction(
                        backend, item, samples, training_data, params,
                        checkpoint_directory, source_hash, config_hash,
                        teacher.get(), block_context.get(), block_collector,
                        item.n_expert > 1 || !reachable || calibration->repeated_use, state);
                const auto factor_end = std::chrono::steady_clock::now();
                const auto seconds = [](auto begin, auto end) {
                    return std::chrono::duration<double>(end - begin).count();
                };
                LLAMA_LOG_INFO(
                        "NanoQuant profile: %s expert %u/%" PRId64
                        " nonfactor=%.3fs ADMM=%.3fs factor=%.3fs\n",
                        item.name.c_str(), expert + 1, item.n_expert,
                        seconds(nonfactor_start, admm_start),
                        seconds(admm_start, factor_start),
                        seconds(factor_start, factor_end));
                if (state.stage != checkpoint_stage::EXPERT_DONE) {
                    throw std::runtime_error(format(
                            "NanoQuant: group '%s' expert %u did not complete reconstruction",
                            item.name.c_str(), expert));
                }
                complete_expert(
                        checkpoint_directory, source_hash, config_hash, item, state);
                save_checkpoint(
                        checkpoint_directory, source_hash, config_hash, item, state);
                validate_state_shapes(item, state);
            }
            release_attached_state(state);
        }
        write_block_checkpoint(
                checkpoint_directory, groups[block_begin].block, source_hash, config_hash);
        LLAMA_LOG_INFO(
                "NanoQuant: completed block %d checkpoint\n", groups[block_begin].block);
        block_begin = block_end;
    }

    projection_context.reset();
    block_context.reset();
    collector_set.active.clear();
    for (size_t i = 0; i < groups.size(); ++i) {
        validate_state_shapes(groups[i], model_states[i], false);
        if (model_states[i].stage < checkpoint_stage::GROUP_DONE) {
            throw std::runtime_error(format(
                    "NanoQuant: incomplete block reconstruction for '%s'",
                    groups[i].name.c_str()));
        }
    }

    teacher_context.reset(llama_init_from_model(teacher.get(), context_params));
    if (!teacher_context) {
        throw std::runtime_error(
                "NanoQuant: failed to create the teacher probability context");
    }
    const std::filesystem::path teacher_cache_path =
            output_path + ".nanoquant.teacher.tmp";
    auto teacher_cache = std::make_unique<teacher_probability_cache>(
            teacher_cache_path, teacher_context.get(), samples, params);
    teacher_context.reset();
    teacher.reset();
    backend.reset_cache();
    loader.init_mappings(false);

    const std::filesystem::path intermediate_student =
            output_path + ".nanoquant.student.tmp";
    std::error_code intermediate_remove_error;
    std::filesystem::remove(intermediate_student, intermediate_remove_error);
    try {
        write_grouped_gguf(
                loader, output_metadata.get(), weights, groups, group_by_name, auxiliary_types,
                checkpoint_directory, source_hash, config_hash, intermediate_student,
                checkpoint_stage::GROUP_DONE, projected_payload, projected_physical_data,
                projected_file_size, metadata_size, alignment,
                read_data, conversion, workers, nthread);
        release_vector(read_data);
        release_vector(conversion);

        llama_context_params student_context_params = context_params;
        student_context_params.cb_eval = nullptr;
        student_context_params.cb_eval_user_data = nullptr;
        model_fit_config student_fit_config = fit_config;
        student_fit_config.n_contexts = 1;
        model_params.n_gpu_layers = configured_model_gpu_layers;
        if (model_params.n_gpu_layers < 0) {
            model_params.n_gpu_layers = fit_model_gpu_layers(
                    intermediate_student.string(), model_params,
                    student_context_params, student_fit_config).n_gpu_layers;
        }
        std::unique_ptr<llama_model, decltype(&llama_model_free)> student(nullptr, llama_model_free);
        std::unique_ptr<llama_context, decltype(&llama_free)> student_context(nullptr, llama_free);
        while (true) {
            student.reset(llama_model_load_from_file(
                    intermediate_student.string().c_str(), model_params));
            if (student) {
                student_context.reset(
                        llama_init_from_model(student.get(), student_context_params));
                if (student_context) {
                    break;
                }
                student.reset();
            }
            if (!reduce_model_gpu_layers(model_params, "the reconstructed student model")) {
                throw std::runtime_error(
                        "NanoQuant: failed to load the reconstructed student model");
            }
        }

        std::vector<bool> scale_tuning_reachable = projection_reachable;
        size_t fixed_expert_groups = 0;
        for (size_t i = 0; i < groups.size(); ++i) {
            if (groups[i].n_expert > 1 && scale_tuning_reachable[i]) {
                scale_tuning_reachable[i] = false;
                ++fixed_expert_groups;
            }
        }
        if (fixed_expert_groups > 0) {
            LLAMA_LOG_INFO(
                    "NanoQuant: keeping %zu routed expert scale groups fixed during global KL tuning\n",
                    fixed_expert_groups);
        }
        owned_batch kl_batch(params->nanoquant_sequence_length);
        LLAMA_LOG_INFO(
                "NanoQuant phase 3/3: packed-student full-model KL scale tuning "
                "(%d tokens/evaluation; source teacher released)\n",
                params->nanoquant_sample_count * params->nanoquant_sequence_length);
        GGML_ASSERT(scale_tuning_reachable.size() == groups.size());
        run_model_scale_kl(
                *teacher_cache, student.get(), student_context.get(),
                kl_batch, samples, groups, scale_tuning_reachable, params,
                checkpoint_directory, source_hash, config_hash, model_states);
        const double final_model_kl = full_model_kl(
                student_context.get(), *teacher_cache, samples, params, kl_batch);
        LLAMA_LOG_INFO(
                "NanoQuant: final full-model teacher/student KL = %.9g\n",
                final_model_kl);
        student_context.reset();
        student.reset();
    } catch (...) {
        std::filesystem::remove(intermediate_student, intermediate_remove_error);
        throw;
    }
    std::filesystem::remove(intermediate_student, intermediate_remove_error);
    teacher_cache.reset();

    const std::filesystem::path temporary_output = output_path + ".nanoquant.tmp";
    std::error_code remove_error;
    std::filesystem::remove(temporary_output, remove_error);
    write_grouped_gguf(
            loader, output_metadata.get(), weights, groups, group_by_name, auxiliary_types,
            checkpoint_directory, source_hash, config_hash, temporary_output,
            checkpoint_stage::MODEL_DONE, projected_payload, projected_physical_data,
            projected_file_size, metadata_size, alignment, read_data, conversion, workers, nthread);
    install_output_file(temporary_output, output_path);

    LLAMA_LOG_INFO(
            "NanoQuant: model payload %zu -> %zu bytes; exact output %zu bytes; %.6f whole-file BPW\n",
            original_payload, projected_payload, projected_file_size,
            projected_file_size * 8.0 / loader.n_elements);
}

} // namespace nanoquant

//
// main quantization driver
//

static void llama_model_quantize_impl(const std::string & fname_inp, const std::string & fname_out, const llama_model_quantize_params * params) {
    llama_ftype ftype = params->ftype;
    if (ftype == LLAMA_FTYPE_MOSTLY_NANOQUANT) {
        nanoquant::quantize(fname_inp, fname_out, params);
        return;
    }

    int nthread = params->nthread;

    if (nthread <= 0) {
        nthread = std::thread::hardware_concurrency();
    }

    ggml_type default_type = llama_ftype_get_default_type(ftype);
    if (default_type == GGML_TYPE_COUNT) {
        throw std::runtime_error(format("invalid output file type %d\n", ftype));
    }

    // mmap consistently increases speed on Linux, and also increases speed on Windows with
    // hot cache. It may cause a slowdown on macOS, possibly related to free memory.
#if defined(__linux__) || defined(_WIN32)
    constexpr llama_load_mode load_mode = LLAMA_LOAD_MODE_MMAP;
#else
    constexpr llama_load_mode load_mode = LLAMA_LOAD_MODE_NONE;
#endif

    const llama_model_kv_override * kv_overrides = params->kv_overrides;
    std::vector<std::string> splits = {};
    llama_model_loader ml(/*metadata*/ nullptr, /*set_tensor_data*/ nullptr, /*set_tensor_data_ud*/ nullptr,
        fname_inp, splits, /*file*/ nullptr, /*load_mode*/ load_mode, /*check_tensors*/ true, /*no_alloc*/ false, kv_overrides, nullptr);
    ml.init_mappings(false); // no prefetching

    auto mparams = llama_model_default_params();
    std::unique_ptr<llama_model> model_ptr(llama_model_create(ml, mparams));

    auto * model = dynamic_cast<llama_model_base *>(model_ptr.get());
    if (model == nullptr) {
        GGML_ABORT("fatal error: model does not implement llama_model_base");
    }

    model->load_hparams(ml);
    model->load_stats  (ml);

    quantize_state_impl qs(*model, params);

    if (params->only_copy) {
        ftype = ml.ftype;
    }
    std::unordered_map<std::string, std::vector<float>> i_data;
    const std::unordered_map<std::string, std::vector<float>> * imatrix_data = nullptr;
    if (params->imatrix) {
        for (const llama_model_imatrix_data * p = params->imatrix; p->name != nullptr; p++) {
            i_data.emplace(p->name, std::vector<float>(p->data, p->data + p->size));
        }
        imatrix_data = & i_data;
        if (imatrix_data) {
            LLAMA_LOG_INFO("\n%s: have importance matrix data with %d entries\n",
                           __func__, (int)imatrix_data->size());
            qs.has_imatrix = true;
            // check imatrix for nans or infs
            for (const auto & kv : *imatrix_data) {
                for (float f : kv.second) {
                    if (!std::isfinite(f)) {
                        throw std::runtime_error(format("imatrix contains non-finite value %f\n", f));
                    }
                }
            }
        }
    }

    const size_t align = GGUF_DEFAULT_ALIGNMENT;
    gguf_context_ptr ctx_out { gguf_init_empty() };

    std::vector<int> prune_list = {};
    if (params->prune_layers) {
        for (const int32_t * p = params->prune_layers; * p != -1; p++) {
            prune_list.push_back(* p);
        }
    }

    // copy the KV pairs from the input file
    gguf_set_kv     (ctx_out.get(), ml.metadata);
    gguf_set_val_u32(ctx_out.get(), ml.llm_kv(LLM_KV_GENERAL_QUANTIZATION_VERSION).c_str(), GGML_QNT_VERSION);
    gguf_set_val_u32(ctx_out.get(), ml.llm_kv(LLM_KV_GENERAL_FILE_TYPE).c_str(), ftype);

    // Remove split metadata
    gguf_remove_key(ctx_out.get(), ml.llm_kv(LLM_KV_SPLIT_NO).c_str());
    gguf_remove_key(ctx_out.get(), ml.llm_kv(LLM_KV_SPLIT_COUNT).c_str());
    gguf_remove_key(ctx_out.get(), ml.llm_kv(LLM_KV_SPLIT_TENSORS_COUNT).c_str());

    if (params->kv_overrides) {
        for (const llama_model_kv_override * o = params->kv_overrides; o->key[0] != 0; ++o) {
            if (o->tag == LLAMA_KV_OVERRIDE_TYPE_FLOAT) {
                gguf_set_val_f32(ctx_out.get(), o->key, o->val_f64);
            } else if (o->tag == LLAMA_KV_OVERRIDE_TYPE_INT) {
                // Setting type to UINT32. See https://github.com/ggml-org/llama.cpp/pull/14182 for context
                gguf_set_val_u32(ctx_out.get(), o->key, (uint32_t)std::abs(o->val_i64));
            } else if (o->tag == LLAMA_KV_OVERRIDE_TYPE_BOOL) {
                gguf_set_val_bool(ctx_out.get(), o->key, o->val_bool);
            } else if (o->tag == LLAMA_KV_OVERRIDE_TYPE_STR) {
                gguf_set_val_str(ctx_out.get(), o->key, o->val_str);
            } else {
                LLAMA_LOG_WARN("%s: unknown KV override type for key %s\n", __func__, o->key);
            }
        }
    }

    std::map<int, std::string> mapped;
    int blk_id = 0;

    // make a list of weights
    std::vector<const llama_model_loader::llama_tensor_weight *> tensors;
    tensors.reserve(ml.weights_map.size());
    for (const auto & it : ml.weights_map) {
        const std::string remapped_name(remap_layer(it.first, prune_list, mapped, blk_id));
        if (remapped_name.empty()) {
            LLAMA_LOG_DEBUG("%s: pruning tensor %s\n", __func__, it.first.c_str());
            continue;
        }

        if (remapped_name != it.first) {
            ggml_set_name(it.second.tensor, remapped_name.c_str());
            LLAMA_LOG_DEBUG("%s: tensor %s remapped to %s\n", __func__, it.first.c_str(), ggml_get_name(it.second.tensor));
        }
        tensors.push_back(&it.second);
    }
    if (!prune_list.empty()) {
        gguf_set_val_u32(ctx_out.get(), ml.llm_kv(LLM_KV_BLOCK_COUNT).c_str(), blk_id);
    }

    // keep_split requires that the weights are sorted by split index
    if (params->keep_split) {
        std::sort(tensors.begin(), tensors.end(), [](const llama_model_loader::llama_tensor_weight * a, const llama_model_loader::llama_tensor_weight * b) {
            if (a->idx == b->idx) {
                return a->offs < b->offs;
            }
            return a->idx < b->idx;
        });
    }

    // compute tensor metadata once and cache it
    std::vector<tensor_metadata> metadata(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
        metadata[i].name = ggml_get_name(tensors[i]->tensor);
    }

    // initialize quantization state counters and metadata categories
    init_quantize_state_counters(qs, metadata);

    int idx = 0;
    uint16_t n_split = 1;

    // Assume split index is continuous
    if (params->keep_split) {
        for (const auto * it : tensors) {
            n_split = std::max(uint16_t(it->idx + 1), n_split);
        }
    }
    std::vector<gguf_context_ptr> ctx_outs(n_split);
    ctx_outs[0] = std::move(ctx_out);

    // flag for --dry-run
    bool will_require_imatrix = false;

    //
    // preliminary iteration over all weights
    //

    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto * it = tensors[i];
        const struct ggml_tensor * tensor = it->tensor;

        uint16_t i_split = params->keep_split ? it->idx : 0;
        if (!ctx_outs[i_split]) {
            ctx_outs[i_split].reset(gguf_init_empty());
        }
        gguf_add_tensor(ctx_outs[i_split].get(), tensor);

        metadata[i].allows_quantization = tensor_allows_quantization(params, model->arch, tensor);

        if (metadata[i].allows_quantization) {
            metadata[i].target_type = llama_tensor_get_type(qs, params, tensor, default_type, metadata[i]);
        } else {
            metadata[i].target_type = tensor->type;
        }

        metadata[i].requires_imatrix = tensor_requires_imatrix(tensor->name, metadata[i].target_type, ftype);

        if (params->imatrix) {
            metadata[i].remapped_imatrix_name = remap_imatrix(tensor->name, mapped);
        } else if (metadata[i].allows_quantization && metadata[i].requires_imatrix) {
            if (params->dry_run) {
                will_require_imatrix = true;
            } else {
                LLAMA_LOG_ERROR("\n============================================================================\n"
                                " ERROR: this quantization requires an importance matrix!\n"
                                "        - offending tensor: %s\n"
                                "        - target type: %s\n"
                                "============================================================================\n\n",
                                metadata[i].name.c_str(), ggml_type_name(metadata[i].target_type));
                throw std::runtime_error("this quantization requires an imatrix!");
            }
        }
    }

    // Set split info if needed
    if (n_split > 1) {
        for (size_t i = 0; i < ctx_outs.size(); ++i) {
            gguf_set_val_u16(ctx_outs[i].get(), ml.llm_kv(LLM_KV_SPLIT_NO).c_str(), i);
            gguf_set_val_u16(ctx_outs[i].get(), ml.llm_kv(LLM_KV_SPLIT_COUNT).c_str(), n_split);
            gguf_set_val_i32(ctx_outs[i].get(), ml.llm_kv(LLM_KV_SPLIT_TENSORS_COUNT).c_str(), (int32_t)tensors.size());
        }
    }

    size_t total_size_org = 0;
    size_t total_size_new = 0;

    std::vector<std::thread> workers;
    workers.reserve(nthread);

    std::vector<no_init<uint8_t>> read_data;
    std::vector<no_init<uint8_t>> work;
    std::vector<no_init<float>> f32_conv_buf;

    int cur_split = -1;
    std::ofstream fout;
    auto close_ofstream = [&]() {
        // Write metadata and close file handler
        if (fout.is_open()) {
            fout.seekp(0);
            std::vector<uint8_t> data(gguf_get_meta_size(ctx_outs[cur_split].get()));
            gguf_get_meta_data(ctx_outs[cur_split].get(), data.data());
            fout.write((const char *) data.data(), data.size());
            fout.close();
        }
    };
    auto new_ofstream = [&](int index) {
        cur_split = index;
        GGML_ASSERT(ctx_outs[cur_split] && "Find uninitialized gguf_context");
        std::string fname = fname_out;
        if (params->keep_split) {
            std::vector<char> split_path(llama_path_max(), 0);
            llama_split_path(split_path.data(), split_path.size(), fname_out.c_str(), cur_split, n_split);
            fname = std::string(split_path.data());
        }

        fout = std::ofstream(fname, std::ios::binary);
        fout.exceptions(std::ofstream::failbit); // fail fast on write errors
        const size_t meta_size = gguf_get_meta_size(ctx_outs[cur_split].get());
        // placeholder for the meta data
        ::zeros(fout, meta_size);
    };

    // no output file for --dry-run
    if (!params->dry_run) {
        new_ofstream(0);
    }

    //
    // main loop: iterate over all weights
    //

    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto & weight = *tensors[i];
        const auto & tm = metadata[i];
        ggml_tensor * tensor = weight.tensor;

        if (!params->dry_run && (weight.idx != cur_split && params->keep_split)) {
            close_ofstream();
            new_ofstream(weight.idx);
        }

        const size_t tensor_size = ggml_nbytes(tensor);

        if (!params->dry_run) {
            if (!ml.use_mmap) {
                if (read_data.size() < tensor_size) {
                    read_data.resize(tensor_size);
                }
                tensor->data = read_data.data();
            }
            ml.load_data_for(tensor);
        }

        LLAMA_LOG_INFO("[%4d/%4d] %-36s - [%s], type = %6s, ",
               ++idx, ml.n_tensors,
               ggml_get_name(tensor),
               llama_format_tensor_shape(tensor).c_str(),
               ggml_type_name(tensor->type));

        const ggml_type cur_type = tensor->type;
        const ggml_type new_type = tm.target_type;

        // If we've decided to quantize to the same type the tensor is already
        // in then there's nothing to do.
        bool quantize = cur_type != new_type;

        void * new_data;
        size_t new_size;

        if (params->dry_run) {
            // the --dry-run option calculates the final quantization size without quantizing
            if (quantize) {
                new_size = ggml_nrows(tensor) * ggml_row_size(new_type, tensor->ne[0]);
                LLAMA_LOG_INFO("size = %8.2f MiB -> %8.2f MiB (%s)\n",
                               tensor_size/1024.0/1024.0,
                               new_size/1024.0/1024.0,
                               ggml_type_name(new_type));
                if (!will_require_imatrix && tm.requires_imatrix) {
                    will_require_imatrix = true;
                }
            } else {
                new_size = tensor_size;
                LLAMA_LOG_INFO("size = %8.3f MiB\n", new_size/1024.0/1024.0);
            }
            total_size_org += tensor_size;
            total_size_new += new_size;
            continue;
        } else {
            // no --dry-run, perform quantization
            if (!quantize) {
                new_data = tensor->data;
                new_size = tensor_size;
                LLAMA_LOG_INFO("size = %8.3f MiB\n", tensor_size/1024.0/1024.0);
            } else {
                const int64_t nelements = ggml_nelements(tensor);

                const float * imatrix = nullptr;
                if (imatrix_data) {
                    auto it = imatrix_data->find(tm.remapped_imatrix_name);
                    if (it == imatrix_data->end()) {
                        LLAMA_LOG_INFO("\n====== %s: did not find weights for %s\n", __func__, tensor->name);
                    } else {
                        if (it->second.size() == (size_t)tensor->ne[0]*tensor->ne[2]) {
                            imatrix = it->second.data();
                        } else {
                            LLAMA_LOG_INFO("\n====== %s: imatrix size %d is different from tensor size %d for %s\n", __func__,
                                    int(it->second.size()), int(tensor->ne[0]*tensor->ne[2]), tensor->name);

                            // this can happen when quantizing an old mixtral model with split tensors with a new incompatible imatrix
                            // this is a significant error and it may be good idea to abort the process if this happens,
                            // since many people will miss the error and not realize that most of the model is being quantized without an imatrix
                            // tok_embd should be ignored in this case, since it always causes this warning
                            if (!tensor_name_match_token_embd(tensor->name)) {
                                throw std::runtime_error(format("imatrix size %d is different from tensor size %d for %s",
                                        int(it->second.size()), int(tensor->ne[0]*tensor->ne[2]), tensor->name));
                            }
                        }
                    }
                }
                if (!imatrix && tm.requires_imatrix) {
                    LLAMA_LOG_ERROR("\n\n============================================================\n");
                    LLAMA_LOG_ERROR("Missing importance matrix for tensor %s in a very low-bit quantization\n", tensor->name);
                    LLAMA_LOG_ERROR("The result will be garbage, so bailing out\n");
                    LLAMA_LOG_ERROR("============================================================\n\n");
                    throw std::runtime_error(format("Missing importance matrix for tensor %s in a very low-bit quantization", tensor->name));
                }

                float * f32_data;

                if (tensor->type == GGML_TYPE_F32) {
                    f32_data = (float *) tensor->data;
                } else if (ggml_is_quantized(tensor->type) && !params->allow_requantize) {
                    throw std::runtime_error(format("requantizing from type %s is disabled", ggml_type_name(tensor->type)));
                } else {
                    llama_tensor_dequantize_impl(tensor, f32_conv_buf, workers, nelements, nthread);
                    f32_data = (float *) f32_conv_buf.data();
                }

                LLAMA_LOG_INFO("converting to %s .. ", ggml_type_name(new_type));
                fflush(stdout);

                if (work.size() < (size_t)nelements * 4) {
                    work.resize(nelements * 4); // upper bound on size
                }
                new_data = work.data();

                const int64_t n_per_row = tensor->ne[0];
                const int64_t nrows = tensor->ne[1];

                static const int64_t min_chunk_size = 32 * 512;
                const int64_t chunk_size = (n_per_row >= min_chunk_size ? n_per_row : n_per_row * ((min_chunk_size + n_per_row - 1)/n_per_row));

                const int64_t nelements_matrix = tensor->ne[0] * tensor->ne[1];
                const int64_t nchunk = (nelements_matrix + chunk_size - 1)/chunk_size;
                const int64_t nthread_use = nthread > 1 ? std::max((int64_t)1, std::min((int64_t)nthread, nchunk)) : 1;

                // quantize each expert separately since they have different importance matrices
                new_size = 0;
                for (int64_t i03 = 0; i03 < tensor->ne[2]; ++i03) {
                    const float * f32_data_03 = f32_data + i03 * nelements_matrix;
                    void * new_data_03 = (char *)new_data + ggml_row_size(new_type, n_per_row) * i03 * nrows;
                    const float * imatrix_03 = imatrix ? imatrix + i03 * n_per_row : nullptr;

                    new_size += llama_tensor_quantize_impl(new_type, f32_data_03, new_data_03, chunk_size, nrows, n_per_row, imatrix_03, workers, nthread_use);
                }
                LLAMA_LOG_INFO("size = %8.2f MiB -> %8.2f MiB\n", tensor_size/1024.0/1024.0, new_size/1024.0/1024.0);
            }
            total_size_org += tensor_size;
            total_size_new += new_size;

            // update the gguf meta data as we go
            gguf_set_tensor_type(ctx_outs[cur_split].get(), metadata[i].name.c_str(), new_type);
            GGML_ASSERT(gguf_get_tensor_size(ctx_outs[cur_split].get(), gguf_find_tensor(ctx_outs[cur_split].get(), metadata[i].name.c_str())) == new_size);
            gguf_set_tensor_data(ctx_outs[cur_split].get(), metadata[i].name.c_str(), new_data);

            // write tensor data + padding
            fout.write((const char *) new_data, new_size);
            zeros(fout, GGML_PAD(new_size, align) - new_size);
        } // no --dry-run
    } // main loop

    if (!params->dry_run) {
        close_ofstream();
    }

    LLAMA_LOG_INFO("%s: model size  = %8.2f MiB (%.2f BPW)\n", __func__, total_size_org/1024.0/1024.0, total_size_org*8.0/ml.n_elements);
    LLAMA_LOG_INFO("%s: quant size  = %8.2f MiB (%.2f BPW)\n", __func__, total_size_new/1024.0/1024.0, total_size_new*8.0/ml.n_elements);

    if (!params->imatrix && params->dry_run && will_require_imatrix) {
        LLAMA_LOG_WARN("%s: WARNING: dry run completed successfully, but actually completing this quantization will require an imatrix!\n",
                       __func__
        );
    }

    if (qs.n_fallback > 0) {
        LLAMA_LOG_WARN("%s: WARNING: %d of %d tensor(s) required fallback quantization\n",
                __func__, qs.n_fallback, ml.n_tensors);
    }
}

//
// interface implementation
//

llama_model_quantize_params llama_model_quantize_default_params() {
    llama_model_quantize_params result = {
        /*.nthread                     =*/ 0,
        /*.ftype                       =*/ LLAMA_FTYPE_MOSTLY_Q8_0,
        /*.output_tensor_type          =*/ GGML_TYPE_COUNT,
        /*.token_embedding_type        =*/ GGML_TYPE_COUNT,
        /*.allow_requantize            =*/ false,
        /*.quantize_output_tensor      =*/ true,
        /*.only_copy                   =*/ false,
        /*.pure                        =*/ false,
        /*.keep_split                  =*/ false,
        /*.dry_run                     =*/ false,
        /*.imatrix                     =*/ nullptr,
        /*.kv_overrides                =*/ nullptr,
        /*.tt_overrides                =*/ nullptr,
        /*.prune_layers                =*/ nullptr,
        /*.nanoquant_calibration_dataset =*/ nullptr,
        /*.nanoquant_checkpoint_directory =*/ nullptr,
        /*.nanoquant_calibration_column =*/ nullptr,
        /*.nanoquant_device            =*/ nullptr,
        /*.nanoquant_sequence_length   =*/ 2048,
        /*.nanoquant_sample_count      =*/ 128,
        /*.nanoquant_n_gpu_layers      =*/ -1,
        /*.nanoquant_target_bits       =*/ 1.0f,
        /*.nanoquant_admm_outer_iterations =*/ 400,
        /*.nanoquant_admm_inner_iterations =*/ 5,
        /*.nanoquant_nonfactor_epochs  =*/ 8,
        /*.nanoquant_factor_epochs     =*/ 8,
        /*.nanoquant_model_epochs      =*/ 8,
        /*.nanoquant_nonfactor_learning_rate =*/ 1.0e-4f,
        /*.nanoquant_factor_learning_rate =*/ 1.0e-5f,
        /*.nanoquant_model_learning_rate =*/ 1.0e-5f,
        /*.nanoquant_seed              =*/ 0,
        /*.nanoquant_resume            =*/ false
    };

    return result;
}

uint32_t llama_model_quantize(
        const char * fname_inp,
        const char * fname_out,
        const llama_model_quantize_params * params) {
    try {
        llama_model_quantize_impl(fname_inp, fname_out, params);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to quantize: %s\n", __func__, err.what());
        return 1;
    }

    return 0;
}

//
// Helper functions for external tools exposed in llama-ext.h
//

quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params) {
    return new quantize_state_impl(*model, params);
}

void llama_quant_free(quantize_state_impl * qs) {
    delete qs;
}

llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc) {
    struct llama_model_params mparams = llama_model_default_params();
    auto arch = llm_arch_from_string(desc->architecture);
    auto * model = llama_model_create(arch, mparams);
    model->arch = arch;

    // infer llm_type: only LLM_TYPE_70B matters for quantization logic
    if (model->arch == LLM_ARCH_LLAMA && desc->n_layer == 80 && desc->n_head != desc->n_head_kv) {
        model->type = LLM_TYPE_70B;
    }

    model->hparams.n_embd             = desc->n_embd;
    model->hparams.n_embd_head_k_full = desc->n_embd_head_k;
    model->hparams.n_embd_head_v_full = desc->n_embd_head_v;
    model->hparams.n_layer_all        = desc->n_layer;
    GGML_ASSERT(desc->n_layer > 0 && desc->n_layer <= LLAMA_MAX_LAYERS);
    model->hparams.n_expert           = desc->n_expert;

    for (uint32_t i = 0; i < desc->n_layer; i++) {
        model->hparams.n_head_arr[i]    = desc->n_head;
        model->hparams.n_head_kv_arr[i] = desc->n_head_kv;
        model->hparams.n_ff_arr[i]      = desc->n_ff;
    }

    return model;
}

bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor) {
    return tensor_allows_quantization(qs->params, qs->model.arch, tensor);
}

void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors) {
    // reset per-computation state
    qs->n_attention_wv      = 0;
    qs->n_ffn_down          = 0;
    qs->n_ffn_gate          = 0;
    qs->n_ffn_up            = 0;
    qs->i_attention_wv      = 0;
    qs->i_ffn_down          = 0;
    qs->i_ffn_gate          = 0;
    qs->i_ffn_up            = 0;
    qs->n_fallback          = 0;
    qs->has_imatrix         = false;
    qs->has_tied_embeddings = true;

    // build metadata from tensor names
    std::vector<tensor_metadata> metadata(n_tensors);
    for (size_t i = 0; i < n_tensors; i++) {
        metadata[i].name = ggml_get_name(tensors[i]);
    }

    // initialize counters and categories
    init_quantize_state_counters(*qs, metadata);

    // use a local copy of params with the requested ftype
    llama_model_quantize_params local_params = *qs->params;
    local_params.ftype = ftype;

    ggml_type default_type = llama_ftype_get_default_type(ftype);

    // compute types
    for (size_t i = 0; i < n_tensors; i++) {
        result_types[i] = llama_tensor_get_type(*qs, &local_params, tensors[i], default_type, metadata[i]);
    }
}
