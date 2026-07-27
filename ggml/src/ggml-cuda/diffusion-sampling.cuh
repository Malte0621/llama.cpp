#pragma once

#include "common.cuh"

// Dense device sampler for DiffusionGemma. Reads per-position canvas logits directly from a
// CUDA-family device tensor [n_vocab, n_tokens] and returns the small per-position arrays.
//   logits   : device tensor, F32, contiguous, ne[0] = n_vocab, nrows >= n_tokens
//   u_host   : host [n_tokens] pre-drawn uniforms
//   *_host   : host outputs [n_tokens] (argmax, entropy, sampled)
// Returns false for an invalid or unsupported tensor so the caller can use the host path.
bool ggml_cuda_diffusion_sample(
        struct ggml_tensor * logits,
        const float        * u_host,
        int                * argmax_host,
        float              * entropy_host,
        int                * sampled_host,
        int                  n_tokens,
        float                inv_temp);
