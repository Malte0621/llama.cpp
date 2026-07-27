#include "nanoquant.cuh"

#include <cstdint>
#include <climits>

template <typename T>
static __device__ __forceinline__ float nanoquant_to_float(T value) {
    return static_cast<float>(value);
}

template <int block_size, typename scale_t>
static __global__ void nanoquant_stage1(
        const float * GGML_CUDA_RESTRICT x,
        const uint32_t * GGML_CUDA_RESTRICT v_bits,
        const scale_t * GGML_CUDA_RESTRICT scale_pre,
        float * GGML_CUDA_RESTRICT tmp,
        int64_t n_in,
        int64_t n_rank,
        int64_t n_tasks) {
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    for (int64_t task = blockIdx.x; task < n_tasks; task += gridDim.x) {
        const int64_t ir = task % n_rank;
        const int64_t iv = task / n_rank;
        const int64_t n_words = (n_in + 31)/32;
        const float * xv = x + iv*n_in;
        const uint32_t * bits = v_bits + ir*n_words;

        float sum = 0.0f;
        for (int64_t i = threadIdx.x; i < n_in; i += block_size) {
            const float value = xv[i] * nanoquant_to_float(scale_pre[i]);
            sum += (bits[i/32] & (uint32_t(1) << (i % 32))) ? -value : value;
        }

        __shared__ float sums[block_size/WARP_SIZE];
        sum = block_reduce<block_reduce_method::SUM, block_size>(sum, sums);
        if (threadIdx.x == 0) {
            tmp[task] = sum;
        }
        if (task + gridDim.x < n_tasks) {
            __syncthreads();
        }
    }
}

template <int block_size, typename scale_t>
static __global__ void nanoquant_stage2(
        const float * GGML_CUDA_RESTRICT tmp,
        const uint32_t * GGML_CUDA_RESTRICT u_bits,
        const scale_t * GGML_CUDA_RESTRICT scale_post,
        float * GGML_CUDA_RESTRICT dst,
        int64_t n_rank,
        int64_t n_out,
        int64_t n_tasks) {
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    for (int64_t task = blockIdx.x; task < n_tasks; task += gridDim.x) {
        const int64_t io = task % n_out;
        const int64_t iv = task / n_out;
        const int64_t n_words = (n_rank + 31)/32;
        const float * tv = tmp + iv*n_rank;
        const uint32_t * bits = u_bits + io*n_words;

        float sum = 0.0f;
        for (int64_t i = threadIdx.x; i < n_rank; i += block_size) {
            const float value = tv[i];
            sum += (bits[i/32] & (uint32_t(1) << (i % 32))) ? -value : value;
        }

        __shared__ float sums[block_size/WARP_SIZE];
        sum = block_reduce<block_reduce_method::SUM, block_size>(sum, sums);
        if (threadIdx.x == 0) {
            dst[task] = sum * nanoquant_to_float(scale_post[io]);
        }
        if (task + gridDim.x < n_tasks) {
            __syncthreads();
        }
    }
}

template <int block_size, typename scale_t>
static __global__ void nanoquant_stage1_back(
        const float * GGML_CUDA_RESTRICT grad,
        const uint32_t * GGML_CUDA_RESTRICT u_bits,
        const scale_t * GGML_CUDA_RESTRICT scale_post,
        float * GGML_CUDA_RESTRICT tmp,
        int64_t n_rank,
        int64_t n_out,
        int64_t n_tasks) {
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    for (int64_t task = blockIdx.x; task < n_tasks; task += gridDim.x) {
        const int64_t ir = task % n_rank;
        const int64_t iv = task / n_rank;
        const int64_t n_words = (n_rank + 31)/32;
        const float * gv = grad + iv*n_out;

        float sum = 0.0f;
        for (int64_t io = threadIdx.x; io < n_out; io += block_size) {
            const uint32_t * bits = u_bits + io*n_words;
            const float value = gv[io] * nanoquant_to_float(scale_post[io]);
            sum += (bits[ir/32] & (uint32_t(1) << (ir % 32))) ? -value : value;
        }

        __shared__ float sums[block_size/WARP_SIZE];
        sum = block_reduce<block_reduce_method::SUM, block_size>(sum, sums);
        if (threadIdx.x == 0) {
            tmp[task] = sum;
        }
        if (task + gridDim.x < n_tasks) {
            __syncthreads();
        }
    }
}

template <int block_size, typename scale_t>
static __global__ void nanoquant_stage2_back(
        const float * GGML_CUDA_RESTRICT tmp,
        const uint32_t * GGML_CUDA_RESTRICT v_bits,
        const scale_t * GGML_CUDA_RESTRICT scale_pre,
        float * GGML_CUDA_RESTRICT dst,
        int64_t n_in,
        int64_t n_rank,
        int64_t n_tasks) {
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    for (int64_t task = blockIdx.x; task < n_tasks; task += gridDim.x) {
        const int64_t ii = task % n_in;
        const int64_t iv = task / n_in;
        const int64_t n_words = (n_in + 31)/32;
        const float * tv = tmp + iv*n_rank;

        float sum = 0.0f;
        for (int64_t ir = threadIdx.x; ir < n_rank; ir += block_size) {
            const uint32_t * bits = v_bits + ir*n_words;
            const float value = tv[ir];
            sum += (bits[ii/32] & (uint32_t(1) << (ii % 32))) ? -value : value;
        }

        __shared__ float sums[block_size/WARP_SIZE];
        sum = block_reduce<block_reduce_method::SUM, block_size>(sum, sums);
        if (threadIdx.x == 0) {
            dst[task] = sum * nanoquant_to_float(scale_pre[ii]);
        }
        if (task + gridDim.x < n_tasks) {
            __syncthreads();
        }
    }
}

template <typename scale_pre_t, typename scale_post_t>
static void nanoquant_launch(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * x,
        const ggml_tensor * v_bits,
        const ggml_tensor * u_bits,
        const ggml_tensor * scale_pre,
        const ggml_tensor * scale_post,
        ggml_tensor * dst) {
    const int64_t n_in = x->ne[0];
    const int64_t n_rank = v_bits->ne[1];
    const int64_t n_out = u_bits->ne[1];
    const int64_t n_vectors = ggml_nrows(x);

    ggml_cuda_pool_alloc<float> tmp(ctx.pool(), n_vectors*n_rank);
    constexpr int block_size = 256;
    cudaStream_t stream = ctx.stream();

    const int64_t n_tasks_stage1 = n_vectors*n_rank;
    const int64_t n_tasks_stage2 = n_vectors*n_out;
    const ggml_cuda_kernel_launch_params stage1_params(
        dim3((unsigned int) MIN(n_tasks_stage1, int64_t(INT_MAX)), 1, 1), block_size, 0, stream);
    ggml_cuda_kernel_launch(nanoquant_stage1<block_size, scale_pre_t>, stage1_params,
        (const float *) x->data,
        (const uint32_t *) v_bits->data,
        (const scale_pre_t *) scale_pre->data,
        tmp.get(), n_in, n_rank, n_tasks_stage1);

    const ggml_cuda_kernel_launch_params stage2_params(
        dim3((unsigned int) MIN(n_tasks_stage2, int64_t(INT_MAX)), 1, 1), block_size, 0, stream);
    ggml_cuda_kernel_launch(nanoquant_stage2<block_size, scale_post_t>, stage2_params,
        tmp.get(),
        (const uint32_t *) u_bits->data,
        (const scale_post_t *) scale_post->data,
        (float *) dst->data, n_rank, n_out, n_tasks_stage2);
}

template <typename scale_pre_t, typename scale_post_t>
static void nanoquant_launch_back(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * grad,
        const ggml_tensor * v_bits,
        const ggml_tensor * u_bits,
        const ggml_tensor * scale_pre,
        const ggml_tensor * scale_post,
        ggml_tensor * dst) {
    const int64_t n_in = scale_pre->ne[0];
    const int64_t n_rank = v_bits->ne[1];
    const int64_t n_out = scale_post->ne[0];
    const int64_t n_vectors = ggml_nrows(grad);

    ggml_cuda_pool_alloc<float> tmp(ctx.pool(), n_vectors*n_rank);
    constexpr int block_size = 256;
    cudaStream_t stream = ctx.stream();

    const int64_t n_tasks_stage1 = n_vectors*n_rank;
    const int64_t n_tasks_stage2 = n_vectors*n_in;
    const ggml_cuda_kernel_launch_params stage1_params(
        dim3((unsigned int) MIN(n_tasks_stage1, int64_t(INT_MAX)), 1, 1), block_size, 0, stream);
    ggml_cuda_kernel_launch(nanoquant_stage1_back<block_size, scale_post_t>, stage1_params,
        (const float *) grad->data,
        (const uint32_t *) u_bits->data,
        (const scale_post_t *) scale_post->data,
        tmp.get(), n_rank, n_out, n_tasks_stage1);

    const ggml_cuda_kernel_launch_params stage2_params(
        dim3((unsigned int) MIN(n_tasks_stage2, int64_t(INT_MAX)), 1, 1), block_size, 0, stream);
    ggml_cuda_kernel_launch(nanoquant_stage2_back<block_size, scale_pre_t>, stage2_params,
        tmp.get(),
        (const uint32_t *) v_bits->data,
        (const scale_pre_t *) scale_pre->data,
        (float *) dst->data, n_in, n_rank, n_tasks_stage2);
}

template <typename scale_pre_t>
static void nanoquant_launch_post(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * x,
        const ggml_tensor * v_bits,
        const ggml_tensor * u_bits,
        const ggml_tensor * scale_pre,
        const ggml_tensor * scale_post,
        ggml_tensor * dst,
        bool backward) {
    switch (scale_post->type) {
        case GGML_TYPE_F32:
            if (backward) {
                nanoquant_launch_back<scale_pre_t, float>(ctx, x, v_bits, u_bits, scale_pre, scale_post, dst);
            } else {
                nanoquant_launch<scale_pre_t, float>(ctx, x, v_bits, u_bits, scale_pre, scale_post, dst);
            }
            break;
        case GGML_TYPE_F16:
            if (backward) {
                nanoquant_launch_back<scale_pre_t, half>(ctx, x, v_bits, u_bits, scale_pre, scale_post, dst);
            } else {
                nanoquant_launch<scale_pre_t, half>(ctx, x, v_bits, u_bits, scale_pre, scale_post, dst);
            }
            break;
        case GGML_TYPE_BF16:
            if (backward) {
                nanoquant_launch_back<scale_pre_t, nv_bfloat16>(ctx, x, v_bits, u_bits, scale_pre, scale_post, dst);
            } else {
                nanoquant_launch<scale_pre_t, nv_bfloat16>(ctx, x, v_bits, u_bits, scale_pre, scale_post, dst);
            }
            break;
        default:
            GGML_ABORT("unsupported NanoQuant post-scale type");
    }
}

void ggml_cuda_nanoquant_linear(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0];
    const ggml_tensor * v_bits = dst->src[1];
    const ggml_tensor * u_bits = dst->src[2];
    const ggml_tensor * scale_pre = dst->src[3];
    const ggml_tensor * scale_post = dst->src[4];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(v_bits->type == GGML_TYPE_I32);
    GGML_ASSERT(u_bits->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(v_bits));
    GGML_ASSERT(ggml_is_contiguous(u_bits));
    GGML_ASSERT(ggml_is_contiguous(scale_pre));
    GGML_ASSERT(ggml_is_contiguous(scale_post));

    const bool backward = ggml_get_op_params_i32(dst, 0) != 0;
    switch (scale_pre->type) {
        case GGML_TYPE_F32:
            nanoquant_launch_post<float>(ctx, x, v_bits, u_bits, scale_pre, scale_post, dst, backward);
            break;
        case GGML_TYPE_F16:
            nanoquant_launch_post<half>(ctx, x, v_bits, u_bits, scale_pre, scale_post, dst, backward);
            break;
        case GGML_TYPE_BF16:
            nanoquant_launch_post<nv_bfloat16>(ctx, x, v_bits, u_bits, scale_pre, scale_post, dst, backward);
            break;
        default:
            GGML_ABORT("unsupported NanoQuant pre-scale type");
    }
}
