#include "diffusion-sampling.cuh"

#include <climits>
#include <map>
#include <mutex>

// One block per canvas position. Parallel max/argmax, Z, and T (T = sum d*e), followed by a
// vocabulary-order multinomial draw. The max reduction compares (value, -index), preserving the
// host sampler's first-maximum tie behavior.
static __global__ void diffusion_dense_sample_kernel(
        const float * __restrict__ logits,
        const float * __restrict__ u,
        int   * __restrict__ argmax,
        float * __restrict__ entropy,
        int   * __restrict__ sampled,
        const int   n_vocab,
        const float inv_temp) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ float s_val[256];
    __shared__ float s_sum[256];
    __shared__ int   s_idx[256];

    const float * row_logits = logits + (size_t) row * n_vocab;

    float local_max = -INFINITY;
    int   local_idx = 0;
    for (int v = tid; v < n_vocab; v += blockDim.x) {
        const float x = row_logits[v] * inv_temp;
        if (x > local_max) {
            local_max = x;
            local_idx = v;
        }
    }
    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other_val = s_val[tid + stride];
            const int   other_idx = s_idx[tid + stride];
            if (other_val > s_val[tid] || (other_val == s_val[tid] && other_idx < s_idx[tid])) {
                s_val[tid] = other_val;
                s_idx[tid] = other_idx;
            }
        }
        __syncthreads();
    }
    const float max_l = s_val[0];
    const int   amax  = s_idx[0];

    float local_sum = 0.0f;
    float local_t   = 0.0f;
    for (int v = tid; v < n_vocab; v += blockDim.x) {
        const float d = row_logits[v] * inv_temp - max_l;
        const float e = expf(d);
        local_sum += e;
        local_t   += d * e;
    }
    s_sum[tid] = local_sum;
    s_val[tid] = local_t;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_val[tid] += s_val[tid + stride];
        }
        __syncthreads();
    }
    const float z = s_sum[0];
    const float t = s_val[0];
    if (tid == 0) {
        argmax[row]  = amax;
        entropy[row] = logf(z) - t / z;
    }
    __syncthreads();

    // Split the vocabulary into contiguous slices. Thread 0 locates the slice containing the CDF
    // crossing, then only that slice is scanned serially, preserving vocabulary order.
    const float r     = u[row] * z;
    const int   chunk = (n_vocab - 1) / blockDim.x + 1;
    const int   beg   = tid * chunk;
    const int   end   = beg < n_vocab ? beg + min(chunk, n_vocab - beg) : n_vocab;

    float slice_sum = 0.0f;
    for (int v = beg; v < end; ++v) {
        slice_sum += expf(row_logits[v] * inv_temp - max_l);
    }
    s_sum[tid] = slice_sum;
    __syncthreads();

    __shared__ int s_tok;
    if (tid == 0) {
        s_tok    = n_vocab - 1;
        s_idx[0] = -1;
        float pref = 0.0f;
        for (int i = 0; i < blockDim.x; ++i) {
            const float next = pref + s_sum[i];
            if (next >= r) {
                s_idx[0] = i;
                s_val[0] = pref;
                break;
            }
            pref = next;
        }
    }
    __syncthreads();

    if (tid == s_idx[0]) {
        float cum = s_val[0];
        for (int v = beg; v < end; ++v) {
            cum += expf(row_logits[v] * inv_temp - max_l);
            if (cum >= r) {
                s_tok = v;
                break;
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        sampled[row] = s_tok;
    }
}

// Per-physical-device scratch. It grows with the largest request and is reused thereafter so the
// steady-state path performs no device allocations.
struct dg_devsample_scratch {
    float * u       = nullptr;
    int   * argmax  = nullptr;
    float * entropy = nullptr;
    int   * sampled = nullptr;
    int     cap     = 0;
};

static std::mutex g_dg_devsample_mutex;
static std::map<int, dg_devsample_scratch> g_dg_devsample;

static void dg_devsample_reserve(dg_devsample_scratch & s, int n) {
    if (s.cap >= n) {
        return;
    }
    if (s.u) {
        CUDA_CHECK(cudaFree(s.u));
    }
    if (s.argmax) {
        CUDA_CHECK(cudaFree(s.argmax));
    }
    if (s.entropy) {
        CUDA_CHECK(cudaFree(s.entropy));
    }
    if (s.sampled) {
        CUDA_CHECK(cudaFree(s.sampled));
    }
    CUDA_CHECK(cudaMalloc((void **) &s.u,       (size_t) n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &s.argmax,  (size_t) n * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **) &s.entropy, (size_t) n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &s.sampled, (size_t) n * sizeof(int)));
    s.cap = n;
}

// Returns the logical CUDA-family device that owns this tensor, or -1 for buffer types that this
// dense kernel cannot dereference. Exact buffer-type matching also rejects CUDA host and split buffers.
static int dg_tensor_cuda_device(const ggml_tensor * logits) {
    const ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(logits->buffer);
    if (buft == nullptr || ggml_backend_buft_is_host(buft)) {
        return -1;
    }

    const ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (dev == nullptr) {
        return -1;
    }

    const ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr || reg != ggml_backend_cuda_reg()) {
        return -1;
    }

    const size_t n_devices = ggml_backend_reg_dev_count(reg);
    for (size_t i = 0; i < n_devices; ++i) {
        if (ggml_backend_reg_dev_get(reg, i) == dev) {
            if (i > (size_t) INT_MAX || buft != ggml_backend_cuda_buffer_type((int) i)) {
                return -1;
            }
            return (int) i;
        }
    }
    return -1;
}

bool ggml_cuda_diffusion_sample(
        struct ggml_tensor * logits,
        const float        * u_host,
        int                * argmax_host,
        float              * entropy_host,
        int                * sampled_host,
        int                  n_tokens,
        float                inv_temp) {
    if (logits == nullptr || u_host == nullptr || argmax_host == nullptr || entropy_host == nullptr ||
            sampled_host == nullptr || n_tokens <= 0) {
        return false;
    }
    if (logits->type != GGML_TYPE_F32 || !ggml_is_contiguous(logits) || logits->data == nullptr ||
            logits->buffer == nullptr) {
        return false;
    }
    if (logits->ne[0] <= 0 || logits->ne[0] > INT_MAX || ggml_nrows(logits) < n_tokens) {
        return false;
    }

    const int logical_device = dg_tensor_cuda_device(logits);
    if (logical_device < 0) {
        return false;
    }
    ggml_cuda_set_device(logical_device);
    const int device = ggml_cuda_get_device();

    const int n_vocab = (int) logits->ne[0];
    const float * logits_d = (const float *) logits->data;

    std::lock_guard<std::mutex> lock(g_dg_devsample_mutex);
    dg_devsample_scratch & s = g_dg_devsample[device];
    dg_devsample_reserve(s, n_tokens);

    CUDA_CHECK(cudaMemcpyAsync(s.u, u_host, (size_t) n_tokens * sizeof(float), cudaMemcpyHostToDevice, 0));
    diffusion_dense_sample_kernel<<<n_tokens, 256, 0, 0>>>(
            logits_d, s.u, s.argmax, s.entropy, s.sampled, n_vocab, inv_temp);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(
            argmax_host, s.argmax, (size_t) n_tokens * sizeof(int), cudaMemcpyDeviceToHost, 0));
    CUDA_CHECK(cudaMemcpyAsync(
            entropy_host, s.entropy, (size_t) n_tokens * sizeof(float), cudaMemcpyDeviceToHost, 0));
    CUDA_CHECK(cudaMemcpyAsync(
            sampled_host, s.sampled, (size_t) n_tokens * sizeof(int), cudaMemcpyDeviceToHost, 0));
    CUDA_CHECK(cudaStreamSynchronize(0));
    return true;
}
