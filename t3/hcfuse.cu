// SM80 HC consumer: BF16 SiLU(lora / 4) -> BF16 up projection -> gated stream mean.
// Fixed checkpoint layout: HC=4, hidden=2560, rank=320; W is [10240,320] row-major.
// Down/injection GEMM and normalization stay outside this kernel. lora can be a
// [:,:320] view of the 336-wide merged down/injection output. All strides are elements.
//
// Default V4: copy-ordered weight prefetch + ldmatrix, eight warps/CTA, M<=8.
// Host production dispatch should keep the three-op cuBLAS path for M>8.
// The library retains V1 above eight rows for numerical checks, not performance.
// -DHC_V1 / -DHC_V2 / -DHC_V3 select the preceding kernels unchanged.
// Built by nix/runtime.nix; no host-specific toolchain paths are required.
// nvcc -O3 -arch=sm_80 -std=c++17 -shared -Xcompiler -fPIC hcfuse.cu -o libhcfuse.so
// nvcc -O3 -arch=sm_80 -std=c++17 -DHC_MAIN hcfuse.cu -lcublas -o hc-bench
// Run: hc-bench [--weights hc-up.bf16] [--sets 16] [--iters 100] [--stride 336]
// The standalone baseline uses CUDA translations of the vLLM glue kernels plus
// cuBLAS's default BF16 GEMM (including split-K if selected). For literal Triton
// glue timings and numerical checks, use hc_check.py --bench.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace hc {
using bf16 = __nv_bfloat16;
constexpr int H = 2560, HC = 4, D = HC * H, K = 320;
#if (defined(HC_V1) + defined(HC_V2) + defined(HC_V3)) > 1
#error "Select at most one of HC_V1, HC_V2 and HC_V3"
#endif
#ifdef HC_V2
constexpr int THREADS = 64, D_TILE = 8;
#else
constexpr int THREADS = 128, D_TILE = 16;
#endif

__device__ __forceinline__ float sigmoid(float x) {
  return __fdividef(1.f, 1.f + __expf(-x));
}

__device__ __forceinline__ void mma(float (&c)[4], uint32_t a0, uint32_t a1,
                                   uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

#ifndef HC_V2
template <int GROUPS>
__global__ void __launch_bounds__(THREADS) up_mix(
    const bf16* __restrict__ lora, const bf16* __restrict__ weight,
    const bf16* __restrict__ xn, bf16* __restrict__ out,
    int rows, int lora_stride, int xn_stride, int out_stride) {
  constexpr int ROWS = GROUPS * 8;
  // Padding makes the MMA activation loads and gate stores bank-conflict-free.
  __shared__ __align__(16) bf16 latent[ROWS][K + 8];
  __shared__ float gates[HC][ROWS][20];
  const int lane = threadIdx.x & 31, stream = threadIdx.x >> 5;
  const int g = lane >> 2, c = (lane & 3) * 2;
  const int row_base = blockIdx.y * ROWS, d_base = blockIdx.x * 16;
  for (int i = threadIdx.x; i < ROWS * K; i += THREADS) {
    const int r = i / K, k = i % K;
    float y = 0.f;
    if (row_base + r < rows) {
      const float v = __bfloat162float(lora[size_t(row_base + r) * lora_stride + k]) * .25f;
      y = v * sigmoid(v);
    }
    latent[r][k] = __float2bfloat16_rn(y);  // hc_silu's BF16 store boundary
  }
  __syncthreads();

  float acc[GROUPS][4] = {};
  const int wrow = stream * H + d_base + g;
#pragma unroll
  for (int k = 0; k < K; k += 16) {
    const uint32_t* a0 = reinterpret_cast<const uint32_t*>(weight + wrow * K + k + c);
    const uint32_t* a1 = reinterpret_cast<const uint32_t*>(weight + (wrow + 8) * K + k + c);
    const uint32_t w0 = __ldg(a0), w1 = __ldg(a1), w2 = __ldg(a0 + 4), w3 = __ldg(a1 + 4);
#pragma unroll
    for (int q = 0; q < GROUPS; ++q) {
      const uint32_t* b = reinterpret_cast<const uint32_t*>(&latent[q * 8 + g][k + c]);
      mma(acc[q], w0, w1, w2, w3, b[0], b[4]);
    }
  }
#pragma unroll
  for (int q = 0; q < GROUPS; ++q) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int d = g + (j >= 2 ? 8 : 0), r = q * 8 + c + (j & 1);
      const float gate = __bfloat162float(__float2bfloat16_rn(acc[q][j]));
      gates[stream][r][d] = sigmoid(gate);  // up GEMM's BF16 output boundary
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < ROWS * 16; i += THREADS) {
    const int r = i / 16, d = i % 16, row = row_base + r;
    if (row < rows) {
      float mixed = 0.f;
#pragma unroll
      for (int s = 0; s < HC; ++s)
        mixed = fmaf(gates[s][r][d], __bfloat162float(xn[size_t(row) * xn_stride + s * H + d_base + d]), mixed);
      out[size_t(row) * out_stride + d_base + d] = __float2bfloat16_rn(mixed * .25f);
    }
  }
}
#else
template <int GROUPS>
__global__ void __launch_bounds__(THREADS) up_mix(
    const bf16* __restrict__ lora, const bf16* __restrict__ weight,
    const bf16* __restrict__ xn, bf16* __restrict__ out,
    int rows, int lora_stride, int xn_stride, int out_stride) {
  constexpr int ROWS = GROUPS * 8, W_ROWS = HC * D_TILE;
  constexpr int CHUNKS = W_ROWS * K / 8, COPIES_PER_GROUP = 10;
  static_assert(CHUNKS % (THREADS * COPIES_PER_GROUP) == 0);
  // 328 BF16 elements per row: 16-byte aligned copies, conflict-free MMA loads.
  __shared__ __align__(16) bf16 staged[W_ROWS][K + 8];
  __shared__ __align__(16) bf16 latent[ROWS][K + 8];
  __shared__ float gates[HC][ROWS][12];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, c = (lane & 3) * 2;
  const int row_base = blockIdx.y * ROWS, d_base = blockIdx.x * D_TILE;

  // Two groups of ten 16-byte copies per thread: the entire 20-K-step weight
  // tile is requested before any MMA, without keeping it in 80 registers/lane.
  // Copies retain ordinary [stream, hidden, K] order; no K-slot permutation.
#pragma unroll
  for (int start = 0; start < CHUNKS; start += THREADS * COPIES_PER_GROUP) {
#pragma unroll
    for (int i = 0; i < COPIES_PER_GROUP; ++i) {
      const int chunk = start + i * THREADS + threadIdx.x;
      const int wr = chunk / (K / 8), k = (chunk % (K / 8)) * 8;
      const int wrow = (wr / D_TILE) * H + d_base + wr % D_TILE;
      const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(&staged[wr][k]));
      const bf16* src = weight + wrow * K + k;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
  // Overlap the weight fetch with the original SiLU staging.
  for (int i = threadIdx.x; i < ROWS * K; i += THREADS) {
    const int r = i / K, k = i % K;
    float y = 0.f;
    if (row_base + r < rows) {
      const float v = __bfloat162float(lora[size_t(row_base + r) * lora_stride + k]) * .25f;
      y = v * sigmoid(v);
    }
    latent[r][k] = __float2bfloat16_rn(y);
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();

  // Each warp's lower/upper eight MMA rows now belong to adjacent HC streams.
  // Each output still uses the identical 20 ordered m16n8k16 accumulations.
  float acc[GROUPS][4] = {};
  const int wr = warp * 2 * D_TILE + g;
#pragma unroll
  for (int k = 0; k < K; k += 16) {
    const uint32_t* a0 = reinterpret_cast<const uint32_t*>(&staged[wr][k + c]);
    const uint32_t* a1 = reinterpret_cast<const uint32_t*>(&staged[wr + D_TILE][k + c]);
    const uint32_t w0 = a0[0], w1 = a1[0], w2 = a0[4], w3 = a1[4];
#pragma unroll
    for (int q = 0; q < GROUPS; ++q) {
      const uint32_t* b = reinterpret_cast<const uint32_t*>(&latent[q * 8 + g][k + c]);
      mma(acc[q], w0, w1, w2, w3, b[0], b[4]);
    }
  }
#pragma unroll
  for (int q = 0; q < GROUPS; ++q) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int stream = warp * 2 + (j >> 1), r = q * 8 + c + (j & 1);
      const float gate = __bfloat162float(__float2bfloat16_rn(acc[q][j]));
      gates[stream][r][g] = sigmoid(gate);
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < ROWS * D_TILE; i += THREADS) {
    const int r = i / D_TILE, d = i % D_TILE, row = row_base + r;
    if (row < rows) {
      float mixed = 0.f;
#pragma unroll
      for (int s = 0; s < HC; ++s)
        mixed = fmaf(gates[s][r][d], __bfloat162float(xn[size_t(row) * xn_stride + s * H + d_base + d]), mixed);
      out[size_t(row) * out_stride + d_base + d] = __float2bfloat16_rn(mixed * .25f);
    }
  }
}
#endif

#ifdef HC_V3
constexpr int V3_WARPS = 16, V3_THREADS = V3_WARPS * 32;

template <int ROW_CAP>
__global__ void __launch_bounds__(V3_THREADS, 2) up_mix_v3(
    const bf16* __restrict__ lora, const bf16* __restrict__ weight,
    const bf16* __restrict__ xn, bf16* __restrict__ out,
    int rows, int lora_stride, int xn_stride, int out_stride) {
  __shared__ __align__(16) bf16 staged[V3_WARPS][HC][K + 8];
  __shared__ __align__(16) bf16 latent[ROW_CAP][K + 8];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3, c = t * 2;
  const int d = blockIdx.x * V3_WARPS + warp;

  // Each warp requests its four complete weight rows in five coalesced copies
  // per lane. More warp-level copy queues, with no duplicated global weights.
#pragma unroll
  for (int i = 0; i < HC * K / (8 * 32); ++i) {
    const int chunk = i * 32 + lane;
    const int s = chunk / (K / 8), k = (chunk % (K / 8)) * 8;
    const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(&staged[warp][s][k]));
    const bf16* src = weight + (s * H + d) * K + k;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
  }
  asm volatile("cp.async.commit_group;" ::: "memory");

  // Only live rows are staged: exactly 320 values at M=1, not an eight-row tile.
  for (int i = threadIdx.x; i < rows * K; i += V3_THREADS) {
    const int r = i / K, k = i % K;
    const float v = __bfloat162float(lora[size_t(r) * lora_stride + k]) * .25f;
    latent[r][k] = __float2bfloat16_rn(v * sigmoid(v));
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();

  // Four live MMA rows = the four streams of one hidden column; other rows
  // are zero. Extra MMA work buys four times as many warps without splitting K.
  float acc[4] = {};
#pragma unroll 1
  for (int k = 0; k < K; k += 16) {
    uint32_t a0 = 0, a2 = 0, b0 = 0, b1 = 0;
    if (g < HC) {
      const uint32_t* a = reinterpret_cast<const uint32_t*>(&staged[warp][g][k + c]);
      a0 = a[0]; a2 = a[4];
    }
    if (g < rows) {
      const uint32_t* b = reinterpret_cast<const uint32_t*>(&latent[g][k + c]);
      b0 = b[0]; b1 = b[4];
    }
    // Same twenty ascending-K FP32 accumulations and BF16 rounding as V1/V2.
    mma(acc, a0, 0, a2, 0, b0, b1);
  }
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int r = c + j;
    float gate = 0.f;
    if (g < HC && r < rows)
      gate = sigmoid(__bfloat162float(__float2bfloat16_rn(acc[j])));
    float mixed = 0.f;
#pragma unroll
    for (int s = 0; s < HC; ++s) {
      // All lanes participate; source lane s*4+t owns stream s, row c+j.
      const float factor = __shfl_sync(0xffffffffu, gate, s * 4 + t);
      if (g == 0 && r < rows)
        mixed = fmaf(factor, __bfloat162float(xn[size_t(r) * xn_stride + s * H + d]), mixed);
    }
    if (g == 0 && r < rows)
      out[size_t(r) * out_stride + d] = __float2bfloat16_rn(mixed * .25f);
  }
}
#endif

#if !defined(HC_V1) && !defined(HC_V2) && !defined(HC_V3)
constexpr int V4_WARPS = 8, V4_THREADS = V4_WARPS * 32;

__device__ __forceinline__ void ldmat2(uint32_t (&v)[2], uint32_t address) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
               : "=r"(v[0]), "=r"(v[1]) : "r"(address) : "memory");
}

template <int ROW_CAP>
__global__ void __launch_bounds__(V4_THREADS, 5) up_mix_v4(
    const bf16* __restrict__ lora, const bf16* __restrict__ weight,
    const bf16* __restrict__ xn, bf16* __restrict__ out,
    int rows, int lora_stride, int xn_stride, int out_stride) {
  __shared__ __align__(16) bf16 staged[V4_WARPS][HC][K + 8];
  __shared__ __align__(16) bf16 latent[ROW_CAP][K + 8];
  __shared__ __align__(16) bf16 zeros[16];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int d_base = blockIdx.x * V4_WARPS, d = d_base + warp;
  constexpr int COPIES = V4_WARPS * HC * K / (8 * V4_THREADS);
  static_assert(COPIES == 5);

  // Global copy order is [stream, hidden, K], not [hidden, stream, K].
  // Eight complete 640-byte rows per stream form a contiguous 5120-byte run:
  // every warp issues a fully contiguous, aligned 512-byte global request.
  // Permute only the shared destination, retaining its conflict-free MMA rows.
  // Two in-flight copy groups (3 + 2 loads/lane) overlap the live SiLU staging.
#pragma unroll
  for (int i = 0; i < COPIES; ++i) {
    const int chunk = i * V4_THREADS + threadIdx.x;
    const int wr = chunk / (K / 8), k = (chunk % (K / 8)) * 8;
    const int s = wr / V4_WARPS, hidden = wr % V4_WARPS;
    const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(&staged[hidden][s][k]));
    const bf16* src = weight + (s * H + d_base + hidden) * K + k;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
    if (i == 2 || i == COPIES - 1)
      asm volatile("cp.async.commit_group;" ::: "memory");
  }
  if (threadIdx.x < 16) zeros[threadIdx.x] = __float2bfloat16_rn(0.f);
  for (int i = threadIdx.x; i < rows * K; i += V4_THREADS) {
    const int r = i / K, k = i % K;
    const float v = __bfloat162float(lora[size_t(r) * lora_stride + k]) * .25f;
    latent[r][k] = __float2bfloat16_rn(v * sigmoid(v));
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();

  // ldmatrix.x2's first/second eight address lanes load K[0:8]/K[8:16].
  // It yields precisely the old a0/a2 and b0/b1 pairs (no transpose or K
  // permutation). Dummy stream/activation rows use the small shared zero row.
  const int matrix_row = lane & 7, half = (lane & 8) ? 8 : 0;
  uint32_t a_address = static_cast<uint32_t>(__cvta_generic_to_shared(
      matrix_row < HC ? &staged[warp][matrix_row][half] : &zeros[half]));
  uint32_t b_address = static_cast<uint32_t>(__cvta_generic_to_shared(
      matrix_row < rows ? &latent[matrix_row][half] : &zeros[half]));
  const uint32_t a_step = matrix_row < HC ? 32 : 0, b_step = matrix_row < rows ? 32 : 0;
  float acc[4] = {};
#pragma unroll 4
  for (int k = 0; k < K; k += 16) {
    uint32_t a[2], b[2];
    ldmat2(a, a_address);
    ldmat2(b, b_address);
    mma(acc, a[0], 0, a[1], 0, b[0], b[1]);
    a_address += a_step;
    b_address += b_step;
  }

  const int g = lane >> 2, t = lane & 3, c = t * 2;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int r = c + j;
    float gate = 0.f;
    if (g < HC && r < rows)
      gate = sigmoid(__bfloat162float(__float2bfloat16_rn(acc[j])));
    float mixed = 0.f;
#pragma unroll
    for (int s = 0; s < HC; ++s) {
      const float factor = __shfl_sync(0xffffffffu, gate, s * 4 + t);
      if (g == 0 && r < rows)
        mixed = fmaf(factor, __bfloat162float(xn[size_t(r) * xn_stride + s * H + d]), mixed);
    }
    if (g == 0 && r < rows)
      out[size_t(r) * out_stride + d] = __float2bfloat16_rn(mixed * .25f);
  }
}
#endif
}  // namespace hc

// Returns cudaError_t as int; does not synchronize or own scratch storage.
// Inputs/output must not overlap. Weight must be contiguous BF16 [10240,320].
extern "C" int hc_fuse_forward(void* stream, const void* lora, const void* weight,
                               const void* xn, void* out, int rows, int lora_stride,
                               int xn_stride, int out_stride) {
  using namespace hc;
  if (rows < 0 || lora_stride < K || xn_stride < D || out_stride < H)
    return int(cudaErrorInvalidValue);
  if (rows == 0) return int(cudaSuccess);
  if (!lora || !weight || !xn || !out) return int(cudaErrorInvalidValue);
  auto st = reinterpret_cast<cudaStream_t>(stream);
#ifdef HC_V3
  if (rows <= 8) {
    if (rows == 1)
      up_mix_v3<1><<<H / V3_WARPS, V3_THREADS, 0, st>>>(
          static_cast<const bf16*>(lora), static_cast<const bf16*>(weight),
          static_cast<const bf16*>(xn), static_cast<bf16*>(out), rows, lora_stride, xn_stride, out_stride);
    else if (rows <= 4)
      up_mix_v3<4><<<H / V3_WARPS, V3_THREADS, 0, st>>>(
          static_cast<const bf16*>(lora), static_cast<const bf16*>(weight),
          static_cast<const bf16*>(xn), static_cast<bf16*>(out), rows, lora_stride, xn_stride, out_stride);
    else
      up_mix_v3<8><<<H / V3_WARPS, V3_THREADS, 0, st>>>(
          static_cast<const bf16*>(lora), static_cast<const bf16*>(weight),
          static_cast<const bf16*>(xn), static_cast<bf16*>(out), rows, lora_stride, xn_stride, out_stride);
    return int(cudaGetLastError());
  }
#elif !defined(HC_V1) && !defined(HC_V2)
  if (rows <= 8) {
    if (rows == 1)
      up_mix_v4<1><<<H / V4_WARPS, V4_THREADS, 0, st>>>(
          static_cast<const bf16*>(lora), static_cast<const bf16*>(weight),
          static_cast<const bf16*>(xn), static_cast<bf16*>(out), rows, lora_stride, xn_stride, out_stride);
    else if (rows <= 4)
      up_mix_v4<4><<<H / V4_WARPS, V4_THREADS, 0, st>>>(
          static_cast<const bf16*>(lora), static_cast<const bf16*>(weight),
          static_cast<const bf16*>(xn), static_cast<bf16*>(out), rows, lora_stride, xn_stride, out_stride);
    else
      up_mix_v4<8><<<H / V4_WARPS, V4_THREADS, 0, st>>>(
          static_cast<const bf16*>(lora), static_cast<const bf16*>(weight),
          static_cast<const bf16*>(xn), static_cast<bf16*>(out), rows, lora_stride, xn_stride, out_stride);
    return int(cudaGetLastError());
  }
#endif
  if (rows <= 8)
    up_mix<1><<<dim3(H / D_TILE, 1), THREADS, 0, st>>>(
        static_cast<const bf16*>(lora), static_cast<const bf16*>(weight),
        static_cast<const bf16*>(xn), static_cast<bf16*>(out), rows, lora_stride, xn_stride, out_stride);
  else
    up_mix<2><<<dim3(H / D_TILE, (rows + 15) / 16), THREADS, 0, st>>>(
        static_cast<const bf16*>(lora), static_cast<const bf16*>(weight),
        static_cast<const bf16*>(xn), static_cast<bf16*>(out), rows, lora_stride, xn_stride, out_stride);
  return int(cudaGetLastError());
}

#ifdef HC_MAIN
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#define CUDA_OK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1); } } while (0)
#define BLAS_OK(call) do { cublasStatus_t e = (call); if (e != CUBLAS_STATUS_SUCCESS) { \
  std::fprintf(stderr, "%s:%d: cuBLAS status %d\n", __FILE__, __LINE__, int(e)); std::exit(1); } } while (0)

namespace hc {
__global__ void silu_reference(const bf16* lora, bf16* activated, int stride) {
  const int k = threadIdx.x, row = blockIdx.x;
  if (k < K) {
    const float v = __bfloat162float(lora[size_t(row) * stride + k]) * .25f;
    activated[row * K + k] = __float2bfloat16_rn(v * sigmoid(v));
  }
}
__global__ void mix_reference(const bf16* xn, const bf16* gate, bf16* out) {
  const int row = blockIdx.x, d = blockIdx.y * 512 + threadIdx.x;
  if (d < H) {
    float acc = 0.f;
#pragma unroll
    for (int s = 0; s < HC; ++s) {
      const int offset = row * D + s * H + d;
      acc = fmaf(sigmoid(__bfloat162float(gate[offset])), __bfloat162float(xn[offset]), acc);
    }
    out[row * H + d] = __float2bfloat16_rn(acc * .25f);
  }
}

struct Graph {
  cudaGraph_t graph{};
  cudaGraphExec_t exec{};
  ~Graph() { if (exec) cudaGraphExecDestroy(exec); if (graph) cudaGraphDestroy(graph); }
};
template <class F>
void capture(Graph& graph, cudaStream_t stream, F launch) {
  for (int i = 0; i < 3; ++i) launch();
  CUDA_OK(cudaStreamSynchronize(stream));
  CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  launch();
  CUDA_OK(cudaStreamEndCapture(stream, &graph.graph));
  CUDA_OK(cudaGraphInstantiate(&graph.exec, graph.graph, nullptr, nullptr, 0));
}
float time_graph(const Graph& graph, cudaStream_t stream, cudaEvent_t start,
                 cudaEvent_t stop, int iters, int sets) {
  CUDA_OK(cudaEventRecord(start, stream));
  for (int i = 0; i < iters; ++i) CUDA_OK(cudaGraphLaunch(graph.exec, stream));
  CUDA_OK(cudaEventRecord(stop, stream));
  CUDA_OK(cudaEventSynchronize(stop));
  float ms = 0.f;
  CUDA_OK(cudaEventElapsedTime(&ms, start, stop));
  return ms * 1000.f / (iters * sets);
}
}  // namespace hc

int main(int argc, char** argv) {
  using namespace hc;
  int sets = 16, iters = 100, stride = 336;
  std::string weight_file;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (i + 1 == argc) { std::fprintf(stderr, "missing value for %s\n", argv[i]); return 2; }
    if (arg == "--sets") sets = std::stoi(argv[++i]);
    else if (arg == "--iters") iters = std::stoi(argv[++i]);
    else if (arg == "--stride") stride = std::stoi(argv[++i]);
    else if (arg == "--weights") weight_file = argv[++i];
    else { std::fprintf(stderr, "unknown option: %s\n", argv[i]); return 2; }
  }
  if (sets < 1 || iters < 1 || (stride != K && stride != 336)) return 2;
  constexpr size_t weight_elements = size_t(D) * K;
  constexpr size_t weight_bytes = weight_elements * sizeof(bf16);
  std::ifstream file;
  if (!weight_file.empty()) {
    file.open(weight_file, std::ios::binary | std::ios::ate);
    const auto bytes = file.tellg();
    if (!file || bytes <= 0 || size_t(bytes) % weight_bytes) {
      std::fprintf(stderr, "weights must be raw BF16 [sets,10240,320], no header\n"); return 2;
    }
    sets = std::min(size_t(sets), size_t(bytes) / weight_bytes);
    file.seekg(0);
  }
  std::mt19937 rng(3821);
  std::normal_distribution<float> normal(0.f, 1.f);
  std::vector<bf16> hw(size_t(sets) * weight_elements);
  if (file.is_open()) {
    if (!file.read(reinterpret_cast<char*>(hw.data()), hw.size() * sizeof(bf16))) return 2;
  } else {
    for (auto& x : hw) x = __float2bfloat16_rn(normal(rng) / std::sqrt(float(K)));
  }
  cudaStream_t stream;
  CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cublasHandle_t handle;
  BLAS_OK(cublasCreate(&handle));
  BLAS_OK(cublasSetStream(handle, stream));
  BLAS_OK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  void* workspace;
  CUDA_OK(cudaMalloc(&workspace, 8 << 20));
  BLAS_OK(cublasSetWorkspace(handle, workspace, 8 << 20));
  bf16* weights;
  CUDA_OK(cudaMalloc(&weights, hw.size() * sizeof(bf16)));
  CUDA_OK(cudaMemcpy(weights, hw.data(), hw.size() * sizeof(bf16), cudaMemcpyHostToDevice));
  cudaEvent_t start, stop;
  CUDA_OK(cudaEventCreate(&start)); CUDA_OK(cudaEventCreate(&stop));
  std::printf("CUDA glue + default cuBLAS baseline; rotating %d %s matrices (%.1f MiB), lora stride %d\n",
              sets, weight_file.empty() ? "synthetic" : "checkpoint", double(hw.size() * sizeof(bf16)) / (1 << 20), stride);
#ifdef HC_V3
  std::printf("V3 for M<=8; M>8 fused_us is the V1 numerical fallback. Keep host cuBLAS for M>8.\n");
#elif !defined(HC_V1) && !defined(HC_V2)
  std::printf("V4 for M<=8; M>8 fused_us is the V1 numerical fallback. Keep host cuBLAS for M>8.\n");
#endif
  std::printf("M,three_op_us,fused_us,speedup,max_abs,relative_l2\n");
  for (int rows : {1, 4, 16}) {
    std::vector<bf16> hl(size_t(sets) * rows * stride, __float2bfloat16_rn(NAN));
    std::vector<bf16> hx(size_t(sets) * rows * D);
    for (int s = 0; s < sets; ++s)
      for (int r = 0; r < rows; ++r)
        for (int k = 0; k < K; ++k) hl[(size_t(s) * rows + r) * stride + k] = __float2bfloat16_rn(4.f * normal(rng));
    for (auto& x : hx) x = __float2bfloat16_rn(normal(rng));
    bf16 *lora, *xn, *activated, *gate, *ref, *out;
    const size_t output_bytes = size_t(sets) * rows * H * sizeof(bf16);
    CUDA_OK(cudaMalloc(&lora, hl.size() * sizeof(bf16)));
    CUDA_OK(cudaMalloc(&xn, hx.size() * sizeof(bf16)));
    CUDA_OK(cudaMalloc(&activated, size_t(rows) * K * sizeof(bf16)));
    CUDA_OK(cudaMalloc(&gate, size_t(rows) * D * sizeof(bf16)));
    CUDA_OK(cudaMalloc(&ref, output_bytes)); CUDA_OK(cudaMalloc(&out, output_bytes));
    CUDA_OK(cudaMemcpy(lora, hl.data(), hl.size() * sizeof(bf16), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(xn, hx.data(), hx.size() * sizeof(bf16), cudaMemcpyHostToDevice));
    auto baseline = [&] {
      const float alpha = 1.f, beta = 0.f;
      for (int s = 0; s < sets; ++s) {
        silu_reference<<<rows, 512, 0, stream>>>(lora + size_t(s) * rows * stride, activated, stride);
        // Row-major A[M,K] * W[D,K]^T, viewed as column-major W^T * A.
        BLAS_OK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, D, rows, K,
            &alpha, weights + size_t(s) * weight_elements, CUDA_R_16BF, K,
            activated, CUDA_R_16BF, K, &beta, gate, CUDA_R_16BF, D,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        mix_reference<<<dim3(rows, H / 512), 512, 0, stream>>>(
            xn + size_t(s) * rows * D, gate, ref + size_t(s) * rows * H);
      }
      CUDA_OK(cudaGetLastError());
    };
    auto fused = [&] {
      for (int s = 0; s < sets; ++s)
        CUDA_OK(static_cast<cudaError_t>(hc_fuse_forward(stream,
            lora + size_t(s) * rows * stride, weights + size_t(s) * weight_elements,
            xn + size_t(s) * rows * D, out + size_t(s) * rows * H, rows, stride, D, H)));
    };
    {
      Graph bg, fg;
      capture(bg, stream, baseline); capture(fg, stream, fused);
      for (int i = 0; i < 5; ++i) {
        CUDA_OK(cudaGraphLaunch(bg.exec, stream)); CUDA_OK(cudaGraphLaunch(fg.exec, stream));
      }
      CUDA_OK(cudaStreamSynchronize(stream));
      std::vector<bf16> hr(output_bytes / sizeof(bf16)), ho(hr.size());
      CUDA_OK(cudaMemcpy(hr.data(), ref, output_bytes, cudaMemcpyDeviceToHost));
      CUDA_OK(cudaMemcpy(ho.data(), out, output_bytes, cudaMemcpyDeviceToHost));
      double error2 = 0., norm2 = 0.; float max_abs = 0.f;
      bool match = true;
      for (size_t i = 0; i < hr.size(); ++i) {
        const float r = __bfloat162float(hr[i]), o = __bfloat162float(ho[i]), e = std::abs(o - r);
        match &= std::isfinite(o) && std::isfinite(r) && e <= 1.f / 512 + std::abs(r) / 128;
        max_abs = std::max(max_abs, e); error2 += double(e) * e; norm2 += double(r) * r;
      }
      if (!match) { std::fprintf(stderr, "M=%d failed numerical check; max_abs=%g\n", rows, max_abs); return 1; }
      std::vector<float> bt, ft;
      for (int trial = 0; trial < 5; ++trial) {
        if (trial & 1) {
          ft.push_back(time_graph(fg, stream, start, stop, iters, sets));
          bt.push_back(time_graph(bg, stream, start, stop, iters, sets));
        } else {
          bt.push_back(time_graph(bg, stream, start, stop, iters, sets));
          ft.push_back(time_graph(fg, stream, start, stop, iters, sets));
        }
      }
      std::sort(bt.begin(), bt.end()); std::sort(ft.begin(), ft.end());
      std::printf("%d,%.3f,%.3f,%.3f,%.8g,%.8g\n", rows, bt[2], ft[2], bt[2] / ft[2],
                  max_abs, std::sqrt(error2 / std::max(norm2, 1e-30)));
    }
    CUDA_OK(cudaFree(lora)); CUDA_OK(cudaFree(xn)); CUDA_OK(cudaFree(activated));
    CUDA_OK(cudaFree(gate)); CUDA_OK(cudaFree(ref)); CUDA_OK(cudaFree(out));
  }
  CUDA_OK(cudaEventDestroy(start)); CUDA_OK(cudaEventDestroy(stop));
  BLAS_OK(cublasDestroy(handle)); CUDA_OK(cudaFree(workspace)); CUDA_OK(cudaFree(weights));
  CUDA_OK(cudaStreamDestroy(stream));
}
#endif
