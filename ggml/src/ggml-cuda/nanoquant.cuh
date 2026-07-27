#pragma once

#include "common.cuh"

void ggml_cuda_nanoquant_linear(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
