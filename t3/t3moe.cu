// t3moe.cu - row-compacted MoE kernels for the ring-coded trellis formats T2/T3/T4 on SM80.
//
// An expert's routed rows are grouped into blocks of 8. Each block computes
//   h = silu(x Wgate^T) * (x Wup^T)    per row tile, with 8 activation rows in LDS
//   out[row] += w * (h Wdown^T)        fp32 atomic scatter
// so an expert only pays for the rows routed to it, whatever the batch size.
//
// Packet = 16 rows x 128 k. Words per lane: T2 4, T3 6, T4 8. Pair P = F*i + j (F = 8, 4) in word i, field j;
// in T3, P = 10*(i/2) + 2*j + (i&1), so consecutive words of a ring hold rows g and g+8 at the same k.
// Trellis: y = v ^ (w[i-2] & U), z = w[i-1], value = 4*y + 2*(z bit) - (2^(R+1) - 1); w[-1], w[-2] come from the
// previous packet in the unit's ring: 4 packets along k in w13, the sibling row tile in w2.
// T3 pairs 30/31 are scalar: 4q - 14 from the spare bits 15/31.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

namespace t3 {

constexpr int H = 2560, I = 640, E = 512, TOPK = 10;
constexpr int W13_RT = 80, W13_KP = 20, W2_RT = 160, W2_KP = 5;
constexpr int WARPS = 5, THREADS = WARPS * 32;
constexpr int W13_PPW = 4, W2_RTU = 2, W2_UPE = W2_RT / W2_RTU;
constexpr int XS_ROW_U4 = H / 8 + 4, HB_ROW_U4 = I / 8;
#ifndef T3_BLOCK_ROWS
#define T3_BLOCK_ROWS 8
#endif
constexpr int BLOCK_ROWS = T3_BLOCK_ROWS;
// Opt-in tile ownership: four independent output tiles per CTA, with K reduced in registers.
// W13 retains activation staging; W2 owns both sibling row tiles and needs no shared memory.
#ifdef T3_W13_TILE
constexpr int W13_TILE_WARPS = 4, W13_THREADS = W13_TILE_WARPS * 32;
constexpr int RED13 = 0, SMEM13 = BLOCK_ROWS * XS_ROW_U4 * 16;
#else
constexpr int W13_THREADS = THREADS;
constexpr int SMEM13 = 8 * XS_ROW_U4 * 16 + 2 * WARPS * 32 * 4 * 4;
constexpr int RED13 = 2 * WARPS * 32 * 4 * 4;
#endif
#ifdef T3_W2_TILE
constexpr int W2_TILE_WARPS = 4, W2_THREADS = W2_TILE_WARPS * 32, SMEM2 = 0;
#else
constexpr int W2_THREADS = THREADS;
constexpr int SMEM2 = 2 * WARPS * 32 * W2_RTU * 4 * 4;
#endif
// Blocks per SM the compiler budgets registers for. ncu on w13: issue slots 45% busy, memory 18%, occupancy 23%, so more
// active warps should help. Registers and shared memory were both limiting at 3.
#ifndef T3_W13_BLOCKS
#define T3_W13_BLOCKS 3
#endif
#ifndef T3_W2_BLOCKS
#define T3_W2_BLOCKS 4
#endif
template <int R> struct Fmt;
template <> struct Fmt<2> { static constexpr int WORDS = 4, PKT = 544, SOFF = 512; };
template <> struct Fmt<3> { static constexpr int WORDS = 6, PKT = 800, SOFF = 768; };
template <> struct Fmt<4> { static constexpr int WORDS = 8, PKT = 1056, SOFF = 1024; };

// ---------------------------------------------------------------- device helpers
__device__ __forceinline__ uint4 ldg_stream_u4(const void* p) {
  uint4 r;
  asm("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
      : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ uint2 ldg_stream_u2(const void* p) {
  uint2 r;
  asm("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}
__device__ __forceinline__ uint32_t ldg_stream_u32(const void* p) {
  uint32_t r;
  asm("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(r) : "l"(p));
  return r;
}
__device__ __forceinline__ uint4 ldg_u4(const void* p) {
  uint4 r;
  asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ uint32_t and_or(uint32_t a, uint32_t m, uint32_t c) {
  uint32_t d;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(d) : "r"(a), "r"(m), "r"(c));
  return d;
}
__device__ __forceinline__ uint32_t bf2_fma(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}
__device__ __forceinline__ uint32_t xa(uint32_t v, uint32_t b, uint32_t m) {   // v ^ (b & m)
  uint32_t d; asm("lop3.b32 %0, %1, %2, %3, 0x78;" : "=r"(d) : "r"(v), "r"(b), "r"(m)); return d;
}
__device__ __forceinline__ uint32_t sel(uint32_t a, uint32_t b, uint32_t m) {  // (a&m)|(b&~m)
  uint32_t d; asm("lop3.b32 %0, %1, %2, %3, 0xCA;" : "=r"(d) : "r"(m), "r"(a), "r"(b)); return d;
}
__device__ __forceinline__ void mma16816(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                         uint32_t a3, uint32_t b0, uint32_t b1, const float (&c)[4]) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%10,%11,%12,%13};"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}
__device__ __forceinline__ float bf_lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf_hi(uint32_t v) { return __uint_as_float(v & 0xFFFF0000u); }
__device__ __forceinline__ float silu(float v) { return v / (1.f + __expf(-v)); }

__host__ __device__ inline int frag_off(int k) {
  const int i = k >> 4, kk = k & 15;
  const int pair = i >> 1, par = i & 1, t = (kk & 7) >> 1, sub = ((kk >> 3) << 1) | (kk & 1);
  return ((pair * 4 + t) * 2 + par) * 4 + sub;
}

// ---------------------------------------------------------------- dequant
constexpr uint32_t A0 = 0x42004200u, FOUR = 0x40804080u;

// Non-uniform levels: L = (v*v + B) * v, in two FMAs. The group scale absorbs the factor B.
// B = 250 (T2), 1000 (T3), 4000 (T4) gives v(1 + v*v/B), wider steps toward the tails.
#ifdef T3_C2
template <uint32_t BB>
__device__ __forceinline__ uint32_t shape(uint32_t v) { return bf2_fma(bf2_fma(v, v, BB), v, 0u); }
#else
template <uint32_t BB>
__device__ __forceinline__ uint32_t shape(uint32_t v) { return v; }
#endif
constexpr uint32_t BB2 = 0x437A437Au, BB3 = 0x447A447Au, BB4 = 0x457A457Au;

__device__ __forceinline__ void dequant_t2(const uint4& a, const uint4& pa, uint32_t (&f)[32]) {
  constexpr uint32_t U = 0x55555555u, N = 0xC307C307u, M = 0x000E000Eu, C = 0xCCCCCCCCu;
  const uint32_t w[6] = {pa.z, pa.w, a.x, a.y, a.z, a.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t v = w[i + 2], y = xa(v, w[i], U), z = w[i + 1];
    const uint32_t R0 = sel(y << 2, z << 1, C), R1 = sel(y, z >> 1, C);
    f[8 * i + 0] = shape<BB2>(bf2_fma(and_or(R0,       M, A0), FOUR, N));
    f[8 * i + 1] = shape<BB2>(bf2_fma(and_or(R1,       M, A0), FOUR, N));
    f[8 * i + 2] = shape<BB2>(bf2_fma(and_or(R0 >> 4,  M, A0), FOUR, N));
    f[8 * i + 3] = shape<BB2>(bf2_fma(and_or(R1 >> 4,  M, A0), FOUR, N));
    f[8 * i + 4] = shape<BB2>(bf2_fma(and_or(R0 >> 8,  M, A0), FOUR, N));
    f[8 * i + 5] = shape<BB2>(bf2_fma(and_or(R1 >> 8,  M, A0), FOUR, N));
    f[8 * i + 6] = shape<BB2>(bf2_fma(and_or(R0 >> 12, M, A0), FOUR, N));
    f[8 * i + 7] = shape<BB2>(bf2_fma(and_or(R1 >> 12, M, A0), FOUR, N));
  }
}

__device__ __forceinline__ void dequant_t3(const uint4& a, const uint4& b, const uint4& pb, uint32_t (&f)[32]) {
  constexpr uint32_t U = 0x12491249u, N143 = 0xC30FC30Fu, M = 0x001E001Eu;
  const uint32_t w[8] = {pb.x, pb.y, a.x, a.y, a.z, a.w, b.x, b.y};
#pragma unroll
  for (int i = 0; i < 6; ++i) {
    const uint32_t v = w[i + 2];
    const uint32_t y = xa(v, w[i], U), z = w[i + 1];
    const uint32_t R0 = sel(y << 1, z, 0xE38EE38Eu);
    const uint32_t R1 = sel(y, z >> 1, 0x0E380E38u);
    // Words i, i+1 hold rows g, g+8 at the same k: pair = 10*(i/2) + 2*j + (i&1).
    const int P0 = 10 * (i >> 1) + (i & 1);
    f[P0 + 0] = shape<BB3>(bf2_fma(and_or(R0 << 1, M, A0), FOUR, N143));
    f[P0 + 2] = shape<BB3>(bf2_fma(and_or(R1 >> 1, M, A0), FOUR, N143));
    f[P0 + 4] = shape<BB3>(bf2_fma(and_or(R0 >> 5, M, A0), FOUR, N143));
    f[P0 + 6] = shape<BB3>(bf2_fma(and_or(R1 >> 7, M, A0), FOUR, N143));
    f[P0 + 8] = shape<BB3>(bf2_fma(and_or(R0 >> 11, M, A0), FOUR, N143));
  }
  uint32_t s;
  constexpr uint32_t B1 = 0xC30EC30Eu;
  s = and_or(w[2] >> 13, 0x00040004u, A0);
  s = and_or(w[3] >> 12, 0x00080008u, s);
  s = and_or(w[4] >> 11, 0x00100010u, s);
  f[30] = shape<BB3>(bf2_fma(s, FOUR, B1));
  s = and_or(w[5] >> 13, 0x00040004u, A0);
  s = and_or(w[6] >> 12, 0x00080008u, s);
  s = and_or(w[7] >> 11, 0x00100010u, s);
  f[31] = shape<BB3>(bf2_fma(s, FOUR, B1));
}

__device__ __forceinline__ void dequant_t4(const uint4& a, const uint4& b, const uint4& pb, uint32_t (&f)[32]) {
  constexpr uint32_t U = 0x11111111u, N = 0xC31FC31Fu, M = 0x003E003Eu;
  const uint32_t w[10] = {pb.z, pb.w, a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint32_t v = w[i + 2], y = xa(v, w[i], U), z = w[i + 1];
    const uint32_t R0 = sel(y << 1, z, 0x1E1E1E1Eu), R1 = sel(y, z >> 1, 0xF0F0F0F0u);
    f[4 * i + 0] = shape<BB4>(bf2_fma(and_or(R0 << 1, M, A0), FOUR, N));
    f[4 * i + 1] = shape<BB4>(bf2_fma(and_or(R1 >> 2, M, A0), FOUR, N));
    f[4 * i + 2] = shape<BB4>(bf2_fma(and_or(R0 >> 7, M, A0), FOUR, N));
    f[4 * i + 3] = shape<BB4>(bf2_fma(and_or(R1 >> 10, M, A0), FOUR, N));
  }
}

template <int R>
__device__ __forceinline__ void dequant(const uint4& a, const uint4& b, const uint4& pa, const uint4& pb, uint32_t (&f)[32]) {
  if constexpr (R == 2) dequant_t2(a, pa, f);
  else if constexpr (R == 3) dequant_t3(a, b, pb, f);
  else dequant_t4(a, b, pb, f);
}

template <int R>
__device__ __forceinline__ void load_packet(const uint8_t* p, int lane, uint4& a, uint4& b, uint32_t& s) {
  a = ldg_stream_u4(p + lane * 16);
  if constexpr (R == 3) {
    const uint2 v = ldg_stream_u2(p + 512 + lane * 8);
    b.x = v.x; b.y = v.y; b.z = 0; b.w = 0;
  } else if constexpr (R == 4) {
    b = ldg_stream_u4(p + 512 + lane * 16);
  } else {
    b = make_uint4(0, 0, 0, 0);
  }
  s = ldg_stream_u32(p + Fmt<R>::SOFF + (lane >> 2) * 4);
}

// ---------------------------------------------------------------- per-layer tables
struct Tables {             // device pointers
  const uint8_t* w13; const uint8_t* w2;
  const uint64_t* off13; const uint64_t* off2;
  const uint8_t* fmt13; const uint8_t* fmt2;
};
struct Work {               // device pointers, one set per process
  int* cnt; int* nxt; int* base; int* blk; int* n_blocks; int* rid; float* rw;
  __nv_bfloat16* hB; float* acc;
};

// ---------------------------------------------------------------- routing, small batches: one CTA
__global__ void __launch_bounds__(512) route_small(const int* __restrict__ ids, const float* __restrict__ tw, int M, Work w, int br) {
  __shared__ int cnt[E], scan[E], base[E];
  const int tid = threadIdx.x;
  cnt[tid] = 0;
  __syncthreads();
  for (int i = tid; i < M * TOPK; i += 512) { const int e = ids[i]; if (e >= 0) atomicAdd(&cnt[e], 1); }
  __syncthreads();
  const int blocks = (cnt[tid] + br - 1) / br;
  scan[tid] = blocks;
  __syncthreads();
  for (int off = 1; off < E; off <<= 1) {
    const int add = tid >= off ? scan[tid - off] : 0;
    __syncthreads();
    scan[tid] += add;
    __syncthreads();
  }
  const int total = scan[E - 1];
  const int first = scan[tid] - blocks;
  base[tid] = first;
  for (int b = 0; b < blocks; ++b) w.blk[first + b] = tid;
  cnt[tid] = 0;
  if (tid == 0) *w.n_blocks = total;
  __syncthreads();
  for (int i = tid; i < total * br; i += 512) w.rid[i] = -1;
  __syncthreads();
  for (int i = tid; i < M * TOPK; i += 512) {
    const int e = ids[i];
    if (e < 0) continue;
    const int pos = base[e] * br + atomicAdd(&cnt[e], 1);
    w.rid[pos] = i / TOPK;
    w.rw[pos] = tw[i];
  }
}

// ---------------------------------------------------------------- routing, large batches: three launches
__global__ void __launch_bounds__(512) route_zero(Work w) { w.cnt[threadIdx.x] = 0; w.nxt[threadIdx.x] = 0; }
__global__ void __launch_bounds__(512) route_count(const int* __restrict__ ids, int M, Work w) {
  for (int i = blockIdx.x * 512 + threadIdx.x; i < M * TOPK; i += gridDim.x * 512) {
    const int e = ids[i];
    if (e >= 0) atomicAdd(&w.cnt[e], 1);
  }
}
__global__ void __launch_bounds__(512) route_scan(Work w, int br) {
  __shared__ int scan[E];
  const int tid = threadIdx.x;
  const int blocks = (w.cnt[tid] + br - 1) / br;
  scan[tid] = blocks;
  __syncthreads();
  for (int off = 1; off < E; off <<= 1) {
    const int add = tid >= off ? scan[tid - off] : 0;
    __syncthreads();
    scan[tid] += add;
    __syncthreads();
  }
  const int first = scan[tid] - blocks;
  w.base[tid] = first;
  for (int b = 0; b < blocks; ++b) w.blk[first + b] = tid;
  if (tid == 0) *w.n_blocks = scan[E - 1];
  __syncthreads();
  for (int i = tid; i < scan[E - 1] * br; i += 512) w.rid[i] = -1;
}
__global__ void __launch_bounds__(512) route_place(const int* __restrict__ ids, const float* __restrict__ tw, int M, Work w, int br) {
  for (int i = blockIdx.x * 512 + threadIdx.x; i < M * TOPK; i += gridDim.x * 512) {
    const int e = ids[i];
    if (e < 0) continue;
    const int pos = w.base[e] * br + atomicAdd(&w.nxt[e], 1);
    w.rid[pos] = i / TOPK;
    w.rw[pos] = tw[i];
  }
}

// ---------------------------------------------------------------- w13 + SwiGLU
struct W13Buf { uint4 a[W13_PPW]; uint4 b[W13_PPW]; uint32_t s[W13_PPW]; };

template <int R>
__device__ __forceinline__ void w13_load(W13Buf& B, const uint8_t* base, int rt, int warp, int lane) {
  const uint8_t* p = base + (size_t(rt) * W13_KP + warp * W13_PPW) * Fmt<R>::PKT;
#pragma unroll
  for (int i = 0; i < W13_PPW; ++i) load_packet<R>(p + i * Fmt<R>::PKT, lane, B.a[i], B.b[i], B.s[i]);
}

#ifdef T3_W13_TILE
template <int R>
__device__ __forceinline__ void w13_step(const W13Buf& B, int warp, int lane, int xg,
                                         const uint4* __restrict__ xs, float (&sum)[4]) {
  const int t = lane & 3;
#else
template <int R>
__device__ __forceinline__ void w13_step(const W13Buf& B, int block, int rt, int par, int warp, int lane, int xg, int rid0, int rid1,
                                         const uint4* __restrict__ xs, float* __restrict__ red, __nv_bfloat16* __restrict__ hB) {
  const int g = lane >> 2, t = lane & 3;
#endif
  const float zero[4] = {0.f, 0.f, 0.f, 0.f};
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int i = 0; i < W13_PPW; ++i) {
    const int pv = (i + W13_PPW - 1) % W13_PPW;
    const int kg = warp * W13_PPW + i;
    float cg[4];
#pragma unroll
    for (int tp = 0; tp < 4; ++tp) {
      // Only this MMA group's eight outputs survive inlining and DCE.
      uint32_t f[32];
      dequant<R>(B.a[i], B.b[i], B.a[pv], B.b[pv], f);
      const int pair = kg * 4 + tp;
      const uint4 xv = xs[xg * XS_ROW_U4 + pair * 4 + t];
      if (tp == 0) mma16816(cg, f[0], f[1], f[2], f[3], xv.x, xv.y, zero);
      else         mma16816(cg, f[8 * tp + 0], f[8 * tp + 1], f[8 * tp + 2], f[8 * tp + 3], xv.x, xv.y, cg);
      mma16816(cg, f[8 * tp + 4], f[8 * tp + 5], f[8 * tp + 6], f[8 * tp + 7], xv.z, xv.w, cg);
    }
    const float s0 = bf_lo(B.s[i]), s1 = bf_hi(B.s[i]);
    acc[0] = fmaf(s0, cg[0], acc[0]);
    acc[1] = fmaf(s0, cg[1], acc[1]);
    acc[2] = fmaf(s1, cg[2], acc[2]);
    acc[3] = fmaf(s1, cg[3], acc[3]);
  }
#ifdef T3_W13_TILE
  // Keep the four-packet FMA chain above separate from the ordered five-group sum.
#pragma unroll
  for (int i = 0; i < 4; ++i) sum[i] = __fadd_rn(sum[i], acc[i]);
#else
  float* r = red + size_t(par) * WARPS * 32 * 4;
  *reinterpret_cast<float4*>(r + (warp * 32 + lane) * 4) = make_float4(acc[0], acc[1], acc[2], acc[3]);
  __syncthreads();
  if (warp == rt % WARPS) {
    const int ch = rt * 8 + g;
    const int off = frag_off(ch);
    float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int w = 0; w < WARPS; ++w) {
      const float4 p = *reinterpret_cast<const float4*>(r + (w * 32 + lane) * 4);
      s.x += p.x; s.y += p.y; s.z += p.z; s.w += p.w;
    }
    __nv_bfloat16* row0 = hB + (size_t(block) * BLOCK_ROWS + 2 * t) * I;
    if (rid0 >= 0) row0[off]     = __float2bfloat16(silu(s.x) * s.z);
    if (rid1 >= 0) row0[I + off] = __float2bfloat16(silu(s.y) * s.w);
  }
#endif
}

#ifdef T3_W13_TILE
template <int R>
__device__ __forceinline__ void w13_tile(const uint8_t* base, int block, int rt, int lane, int xg, int rid0, int rid1,
                                         const uint4* __restrict__ xs, __nv_bfloat16* __restrict__ hB) {
  float sum[4] = {0.f, 0.f, 0.f, 0.f};
  W13Buf b;
#pragma unroll 1
  for (int kg = 0; kg < W13_KP / W13_PPW; ++kg) {
    // Load a complete four-packet ring; its last packet precedes its first, not the previous K group.
    w13_load<R>(b, base, rt, kg, lane);
    w13_step<R>(b, kg, lane, xg, xs, sum);
  }
  const int off = frag_off(rt * 8 + (lane >> 2));
  __nv_bfloat16* row0 = hB + (size_t(block) * BLOCK_ROWS + 2 * (lane & 3)) * I;
  if (rid0 >= 0) row0[off]     = __float2bfloat16(silu(sum[0]) * sum[2]);
  if (rid1 >= 0) row0[I + off] = __float2bfloat16(silu(sum[1]) * sum[3]);
}
#endif

// One unit = one block and `span` row tiles: stage its activation rows, then run the tiles.
__global__ void __launch_bounds__(W13_THREADS, T3_W13_BLOCKS)
w13_swiglu(Tables L, Work W, const __nv_bfloat16* __restrict__ x, int xs_rows) {
  extern __shared__ __align__(16) uint8_t smem[];
  uint4* xs = reinterpret_cast<uint4*>(smem);
#ifndef T3_W13_TILE
  float* red = reinterpret_cast<float*>(smem + xs_rows * XS_ROW_U4 * 16);
#endif
  // Lanes past the staged rows feed activation columns that are never written, so any staged row will do.
  __nv_bfloat16* xsb = reinterpret_cast<__nv_bfloat16*>(xs);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  // Lanes past the staged rows feed activation columns that are never written, so any staged row will do.
  const int xg = min(lane >> 2, xs_rows - 1);
  const int n_blocks = *W.n_blocks;
  // Spread the blocks over all CTAs: with few blocks each CTA takes a run of row tiles of one block.
#ifdef T3_W13_TILE
  constexpr int tile_groups = (W13_RT + W13_TILE_WARPS - 1) / W13_TILE_WARPS;
  const int split = n_blocks ? max(1, min(tile_groups, gridDim.x / n_blocks)) : 1;
#else
  const int split = n_blocks ? max(1, min(W13_RT, gridDim.x / n_blocks)) : 1;
#endif
  const int total = n_blocks * split;
#ifndef T3_W13_TILE
  int par = 0;
#endif
  for (int unit = blockIdx.x; unit < total; unit += gridDim.x) {
    const int block = unit / split, part = unit - block * split;
    const int expert = W.blk[block];
    const int* rows = W.rid + size_t(block) * BLOCK_ROWS;
    const int rid0 = 2 * (lane & 3) < BLOCK_ROWS ? rows[2 * (lane & 3)] : -1, rid1 = 2 * (lane & 3) < BLOCK_ROWS ? rows[2 * (lane & 3) + 1] : -1;
    const uint8_t* base = L.w13 + L.off13[expert];
    const int fmt = L.fmt13[expert];
    // Filled rows sit at the start of the block. Columns of unfilled rows hold stale data, and their outputs are never written.
#pragma unroll 1
    for (int n = 0; n < BLOCK_ROWS; ++n) {
      const int row = rows[n];
      if (row < 0) break;
      const uint4* src = reinterpret_cast<const uint4*>(x + size_t(row) * H);
      for (int i = threadIdx.x; i < H / 8; i += W13_THREADS) {
        const uint4 v = src[i];
        const int k = i * 8;
        const uint32_t e[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int j = 0; j < 4; ++j)
          *reinterpret_cast<uint32_t*>(xsb + n * XS_ROW_U4 * 8 + frag_off(k + 2 * j)) = e[j];
      }
    }
    __syncthreads();
    const int first = part * W13_RT / split, last = (part + 1) * W13_RT / split;
#ifdef T3_W13_TILE
    if (fmt == 3) {
      for (int rt = first + warp; rt < last; rt += W13_TILE_WARPS)
        w13_tile<3>(base, block, rt, lane, xg, rid0, rid1, xs, W.hB);
    } else {
      for (int rt = first + warp; rt < last; rt += W13_TILE_WARPS)
        w13_tile<2>(base, block, rt, lane, xg, rid0, rid1, xs, W.hB);
    }
    // Tile warps can finish at different times. Do not overwrite xs for the next unit early.
    __syncthreads();
#else
#ifdef T3_NO_AHEAD
    // One packet set in registers: warps wait on their loads, and more of them fit on the SM.
    W13Buf b0;
    if (fmt == 3) {
      for (int rt = first; rt < last; ++rt) {
        w13_load<3>(b0, base, rt, warp, lane);
        w13_step<3>(b0, block, rt, par, warp, lane, xg, rid0, rid1, xs, red, W.hB);
        par ^= 1;
      }
    } else {
      for (int rt = first; rt < last; ++rt) {
        w13_load<2>(b0, base, rt, warp, lane);
        w13_step<2>(b0, block, rt, par, warp, lane, xg, rid0, rid1, xs, red, W.hB);
        par ^= 1;
      }
    }
#else
    W13Buf b0, b1;
    if (fmt == 3) {
      w13_load<3>(b0, base, first, warp, lane);
      for (int rt = first; rt < last; ) {
        if (rt + 1 < last) w13_load<3>(b1, base, rt + 1, warp, lane);
        w13_step<3>(b0, block, rt, par, warp, lane, xg, rid0, rid1, xs, red, W.hB);
        ++rt; par ^= 1;
        if (rt >= last) break;
        if (rt + 1 < last) w13_load<3>(b0, base, rt + 1, warp, lane);
        w13_step<3>(b1, block, rt, par, warp, lane, xg, rid0, rid1, xs, red, W.hB);
        ++rt; par ^= 1;
      }
    } else {
      w13_load<2>(b0, base, first, warp, lane);
      for (int rt = first; rt < last; ) {
        if (rt + 1 < last) w13_load<2>(b1, base, rt + 1, warp, lane);
        w13_step<2>(b0, block, rt, par, warp, lane, xg, rid0, rid1, xs, red, W.hB);
        ++rt; par ^= 1;
        if (rt >= last) break;
        if (rt + 1 < last) w13_load<2>(b0, base, rt + 1, warp, lane);
        w13_step<2>(b1, block, rt, par, warp, lane, xg, rid0, rid1, xs, red, W.hB);
        ++rt; par ^= 1;
      }
    }
#endif
#endif
  }
}


// ---------------------------------------------------------------- w2 + weighted scatter-add
struct W2Buf { uint4 a[W2_RTU]; uint4 b[W2_RTU]; uint32_t s[W2_RTU]; uint4 h[4]; };

template <int R>
__device__ __forceinline__ void w2_load(W2Buf& B, const uint8_t* base, const __nv_bfloat16* __restrict__ hB,
                                        int block, int up, int warp, int lane) {
#pragma unroll
  for (int j = 0; j < W2_RTU; ++j) {
    const uint8_t* p = base + (size_t(up * W2_RTU + j) * W2_KP + warp) * Fmt<R>::PKT;
    load_packet<R>(p, lane, B.a[j], B.b[j], B.s[j]);
  }
  const uint4* h4 = reinterpret_cast<const uint4*>(hB);
#ifdef T3_W2_TILE
  // Unused MMA columns must still read inside this routed block.
  const int g = min(lane >> 2, BLOCK_ROWS - 1), t = lane & 3;
#else
  const int g = lane >> 2, t = lane & 3;
#endif
#pragma unroll
  for (int tp = 0; tp < 4; ++tp)
    B.h[tp] = ldg_u4(h4 + (size_t(block) * BLOCK_ROWS + g) * HB_ROW_U4 + (warp * 4 + tp) * 4 + t);
}

#ifdef T3_W2_TILE
template <int R>
__device__ __forceinline__ void w2_step(const W2Buf& B, float (&sum)[W2_RTU][4]) {
#else
template <int R>
__device__ __forceinline__ void w2_step(const W2Buf& B, int up, int par, int warp, int lane, int rid0, int rid1, float w0, float w1,
                                        float* __restrict__ red, float* __restrict__ out) {
  const int g = lane >> 2;
#endif
  const float zero[4] = {0.f, 0.f, 0.f, 0.f};
  float acc[W2_RTU][4];
#pragma unroll
  for (int j = 0; j < W2_RTU; ++j) {
#pragma unroll
    for (int tp = 0; tp < 4; ++tp) {
      uint32_t f[32];
      dequant<R>(B.a[j], B.b[j], B.a[j ^ 1], B.b[j ^ 1], f);
      const uint4 hv = B.h[tp];
      if (tp == 0) mma16816(acc[j], f[0], f[1], f[2], f[3], hv.x, hv.y, zero);
      else         mma16816(acc[j], f[8 * tp + 0], f[8 * tp + 1], f[8 * tp + 2], f[8 * tp + 3], hv.x, hv.y, acc[j]);
      mma16816(acc[j], f[8 * tp + 4], f[8 * tp + 5], f[8 * tp + 6], f[8 * tp + 7], hv.z, hv.w, acc[j]);
    }
    const float s0 = bf_lo(B.s[j]), s1 = bf_hi(B.s[j]);
#ifdef T3_W2_TILE
    // Match the old separately rounded packet scale, then warp-0..4 addition; never fuse these.
    sum[j][0] = __fadd_rn(sum[j][0], __fmul_rn(acc[j][0], s0));
    sum[j][1] = __fadd_rn(sum[j][1], __fmul_rn(acc[j][1], s0));
    sum[j][2] = __fadd_rn(sum[j][2], __fmul_rn(acc[j][2], s1));
    sum[j][3] = __fadd_rn(sum[j][3], __fmul_rn(acc[j][3], s1));
#else
    acc[j][0] *= s0; acc[j][1] *= s0; acc[j][2] *= s1; acc[j][3] *= s1;
#endif
  }
#ifndef T3_W2_TILE
  float* r = red + size_t(par) * WARPS * 32 * W2_RTU * 4;
#pragma unroll
  for (int j = 0; j < W2_RTU; ++j)
    *reinterpret_cast<float4*>(r + (warp * 32 + lane) * W2_RTU * 4 + j * 4) =
        make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);
  __syncthreads();
  if (warp == up % WARPS) {
#pragma unroll
    for (int j = 0; j < W2_RTU; ++j) {
      float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int w = 0; w < WARPS; ++w) {
        const float4 p = *reinterpret_cast<const float4*>(r + (w * 32 + lane) * W2_RTU * 4 + j * 4);
        s.x += p.x; s.y += p.y; s.z += p.z; s.w += p.w;
      }
      const int row = (up * W2_RTU + j) * 16 + g;
      if (rid0 >= 0) { atomicAdd(out + size_t(rid0) * H + row, w0 * s.x);
                       atomicAdd(out + size_t(rid0) * H + row + 8, w0 * s.z); }
      if (rid1 >= 0) { atomicAdd(out + size_t(rid1) * H + row, w1 * s.y);
                       atomicAdd(out + size_t(rid1) * H + row + 8, w1 * s.w); }
    }
  }
#endif
}

#ifdef T3_W2_TILE
template <int R>
__device__ __forceinline__ void w2_tile(const uint8_t* base, Work W, int block, int up, int lane,
                                        int rid0, int rid1, float w0, float w1) {
  float sum[W2_RTU][4] = {};
  W2Buf b;
#pragma unroll 1
  for (int kp = 0; kp < W2_KP; ++kp) {
    // Both sibling packets belong to this K position; never carry a trellis predecessor across K.
    w2_load<R>(b, base, W.hB, block, up, kp, lane);
    w2_step<R>(b, sum);
  }
#pragma unroll
  for (int j = 0; j < W2_RTU; ++j) {
    const int row = (up * W2_RTU + j) * 16 + (lane >> 2);
    if (rid0 >= 0) { atomicAdd(W.acc + size_t(rid0) * H + row, w0 * sum[j][0]);
                     atomicAdd(W.acc + size_t(rid0) * H + row + 8, w0 * sum[j][2]); }
    if (rid1 >= 0) { atomicAdd(W.acc + size_t(rid1) * H + row, w1 * sum[j][1]);
                     atomicAdd(W.acc + size_t(rid1) * H + row + 8, w1 * sum[j][3]); }
  }
}

__global__ void __launch_bounds__(W2_THREADS, T3_W2_BLOCKS)
w2_scatter(Tables L, Work W) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, t = lane & 3;
  const int total = *W.n_blocks * W2_UPE;
  for (int u = blockIdx.x * W2_TILE_WARPS + warp; u < total; u += gridDim.x * W2_TILE_WARPS) {
    const int block = u / W2_UPE, up = u - block * W2_UPE;
    const int* rows = W.rid + size_t(block) * BLOCK_ROWS;
    const int rid0 = 2 * t < BLOCK_ROWS ? rows[2 * t] : -1;
    const int rid1 = 2 * t + 1 < BLOCK_ROWS ? rows[2 * t + 1] : -1;
    const float w0 = rid0 >= 0 ? W.rw[size_t(block) * BLOCK_ROWS + 2 * t] : 0.f;
    const float w1 = rid1 >= 0 ? W.rw[size_t(block) * BLOCK_ROWS + 2 * t + 1] : 0.f;
    const int expert = W.blk[block], fmt = L.fmt2[expert];
    const uint8_t* base = L.w2 + L.off2[expert];
    if (fmt == 3) w2_tile<3>(base, W, block, up, lane, rid0, rid1, w0, w1);
    else if (fmt == 4) w2_tile<4>(base, W, block, up, lane, rid0, rid1, w0, w1);
    else w2_tile<2>(base, W, block, up, lane, rid0, rid1, w0, w1);
  }
}
#else
__global__ void __launch_bounds__(THREADS, T3_W2_BLOCKS)
w2_scatter(Tables L, Work W) {
  extern __shared__ __align__(16) uint8_t smem[];
  float* red = reinterpret_cast<float*>(smem);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, t = lane & 3;
  const int total = *W.n_blocks * W2_UPE;
  W2Buf b0;
  int par = 0;
  int u = blockIdx.x;
#define W2_ROUTE(BLOCK) \
    const int block = (BLOCK); \
    const int* rows = W.rid + size_t(block) * BLOCK_ROWS; \
    const int rid0 = 2 * t < BLOCK_ROWS ? rows[2 * t] : -1, rid1 = 2 * t < BLOCK_ROWS ? rows[2 * t + 1] : -1; \
    const float w0 = W.rw[size_t(block) * BLOCK_ROWS + 2 * t], w1 = W.rw[size_t(block) * BLOCK_ROWS + 2 * t + 1];
  while (u < total) {
    W2_ROUTE(u / W2_UPE)
    const int up = u - block * W2_UPE;
    const int expert = W.blk[block];
    const uint8_t* base = L.w2 + L.off2[expert];
    const int fmt = L.fmt2[expert];
    if (fmt == 3) {
      w2_load<3>(b0, base, W.hB, block, up, warp, lane);
      w2_step<3>(b0, up, par, warp, lane, rid0, rid1, w0, w1, red, W.acc);
    } else if (fmt == 4) {
      w2_load<4>(b0, base, W.hB, block, up, warp, lane);
      w2_step<4>(b0, up, par, warp, lane, rid0, rid1, w0, w1, red, W.acc);
    } else {
      w2_load<2>(b0, base, W.hB, block, up, warp, lane);
      w2_step<2>(b0, up, par, warp, lane, rid0, rid1, w0, w1, red, W.acc);
    }
    par ^= 1;
    u += gridDim.x;
  }
}
#endif

// ---------------------------------------------------------------- large batches: 16 rows per decode
// One CTA per SM holds 16 activation rows in LDS, and three groups of 5 warps each run a row tile.
constexpr int WIDE_ROWS = 16, WIDE_THREADS = 3 * THREADS;
constexpr int SMEM13W = 16 * XS_ROW_U4 * 16 + 2 * 3 * WARPS * 32 * 2 * 4 * 4;
constexpr int SMEM2W = 2 * WARPS * 32 * W2_RTU * 2 * 4 * 4;

template <int R>
__device__ __forceinline__ void w13w_step(const W13Buf& B, int block, int rt, int par, int group, int warp, int lane, const int (&rid)[4],
                                          const uint4* __restrict__ xs, float* __restrict__ red, __nv_bfloat16* __restrict__ hB) {
  const int g = lane >> 2, t = lane & 3;
  const float zero[4] = {0.f, 0.f, 0.f, 0.f};
  float acc[2][4];
#pragma unroll
  for (int n = 0; n < 2; ++n)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[n][j] = 0.f;
#pragma unroll
  for (int i = 0; i < W13_PPW; ++i) {
    uint32_t f[32];
    const int pv = (i + W13_PPW - 1) % W13_PPW;
    dequant<R>(B.a[i], B.b[i], B.a[pv], B.b[pv], f);
    const int kg = warp * W13_PPW + i;
    float cg[2][4];
#pragma unroll
    for (int tp = 0; tp < 4; ++tp) {
      const int pair = kg * 4 + tp;
#pragma unroll
      for (int n = 0; n < 2; ++n) {
        const uint4 xv = xs[(n * 8 + g) * XS_ROW_U4 + pair * 4 + t];
        if (tp == 0) mma16816(cg[n], f[0], f[1], f[2], f[3], xv.x, xv.y, zero);
        else         mma16816(cg[n], f[8 * tp + 0], f[8 * tp + 1], f[8 * tp + 2], f[8 * tp + 3], xv.x, xv.y, cg[n]);
        mma16816(cg[n], f[8 * tp + 4], f[8 * tp + 5], f[8 * tp + 6], f[8 * tp + 7], xv.z, xv.w, cg[n]);
      }
    }
    const float s0 = bf_lo(B.s[i]), s1 = bf_hi(B.s[i]);
#pragma unroll
    for (int n = 0; n < 2; ++n) {
      acc[n][0] = fmaf(s0, cg[n][0], acc[n][0]);
      acc[n][1] = fmaf(s0, cg[n][1], acc[n][1]);
      acc[n][2] = fmaf(s1, cg[n][2], acc[n][2]);
      acc[n][3] = fmaf(s1, cg[n][3], acc[n][3]);
    }
  }
  float* r = red + size_t(par) * 3 * WARPS * 32 * 2 * 4;
  constexpr int LANE_FLOATS = 2 * 4;
#pragma unroll
  for (int n = 0; n < 2; ++n)
    *reinterpret_cast<float4*>(r + ((group * WARPS + warp) * 32 + lane) * LANE_FLOATS + n * 4) = make_float4(acc[n][0], acc[n][1], acc[n][2], acc[n][3]);
  __syncthreads();
  if (warp == rt % WARPS) {
    const int off = frag_off(rt * 8 + g);
#pragma unroll
    for (int n = 0; n < 2; ++n) {
      float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int w = 0; w < WARPS; ++w) {
        const float4 p = *reinterpret_cast<const float4*>(r + ((group * WARPS + w) * 32 + lane) * LANE_FLOATS + n * 4);
        s.x += p.x; s.y += p.y; s.z += p.z; s.w += p.w;
      }
      __nv_bfloat16* row0 = hB + (size_t(block) * WIDE_ROWS + n * 8 + 2 * t) * I;
      if (rid[n * 2] >= 0)     row0[off]     = __float2bfloat16(silu(s.x) * s.z);
      if (rid[n * 2 + 1] >= 0) row0[I + off] = __float2bfloat16(silu(s.y) * s.w);
    }
  }
}

__global__ void __launch_bounds__(WIDE_THREADS, 1)
w13_wide(Tables L, Work W, const __nv_bfloat16* __restrict__ x) {
  extern __shared__ __align__(16) uint8_t smem[];
  uint4* xs = reinterpret_cast<uint4*>(smem);
  float* red = reinterpret_cast<float*>(smem + 16 * XS_ROW_U4 * 16);
  __nv_bfloat16* xsb = reinterpret_cast<__nv_bfloat16*>(xs);
  const int tid = threadIdx.x, lane = tid & 31, group = (tid >> 5) / WARPS, warp = (tid >> 5) % WARPS;
  const int total = *W.n_blocks;
  int par = 0;
  for (int block = blockIdx.x; block < total; block += gridDim.x) {
    const int expert = W.blk[block];
    const int* rows = W.rid + size_t(block) * WIDE_ROWS;
    const int t = lane & 3;
    const int rid[4] = {rows[2 * t], rows[2 * t + 1], rows[8 + 2 * t], rows[8 + 2 * t + 1]};
    const uint8_t* base = L.w13 + L.off13[expert];
    const int fmt = L.fmt13[expert];
#pragma unroll 1
    for (int n = 0; n < WIDE_ROWS; ++n) {
      const int row = rows[n];
      if (row < 0) break;
      if (tid < H / 8) {
        const uint4 v = reinterpret_cast<const uint4*>(x + size_t(row) * H)[tid];
        const int k = tid * 8;
        const uint32_t e[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int j = 0; j < 4; ++j)
          *reinterpret_cast<uint32_t*>(xsb + n * XS_ROW_U4 * 8 + frag_off(k + 2 * j)) = e[j];
      }
    }
    __syncthreads();
    W13Buf b0;
    if (fmt == 3) {
      for (int rt = group; rt < W13_RT; rt += 3) {
        w13_load<3>(b0, base, rt, warp, lane);
        w13w_step<3>(b0, block, rt, par, group, warp, lane, rid, xs, red, W.hB);
        par ^= 1;
      }
    } else {
      for (int rt = group; rt < W13_RT; rt += 3) {
        w13_load<2>(b0, base, rt, warp, lane);
        w13w_step<2>(b0, block, rt, par, group, warp, lane, rid, xs, red, W.hB);
        par ^= 1;
      }
    }
    // Groups with fewer tiles still meet the others' barriers.
    const int rounds = (W13_RT + 2) / 3, mine = (W13_RT - group + 2) / 3;
    for (int i = mine; i < rounds; ++i) { __syncthreads(); par ^= 1; }
    __syncthreads();
  }
}

struct W2WBuf { uint4 a[W2_RTU]; uint4 b[W2_RTU]; uint32_t s[W2_RTU]; uint4 h[2][4]; };

template <int R>
__device__ __forceinline__ void w2w_load(W2WBuf& B, const uint8_t* base, const __nv_bfloat16* __restrict__ hB, int block, int up, int warp, int lane) {
#pragma unroll
  for (int j = 0; j < W2_RTU; ++j) {
    const uint8_t* p = base + (size_t(up * W2_RTU + j) * W2_KP + warp) * Fmt<R>::PKT;
    load_packet<R>(p, lane, B.a[j], B.b[j], B.s[j]);
  }
  const uint4* h4 = reinterpret_cast<const uint4*>(hB);
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int n = 0; n < 2; ++n)
#pragma unroll
    for (int tp = 0; tp < 4; ++tp)
      B.h[n][tp] = ldg_u4(h4 + (size_t(block) * WIDE_ROWS + n * 8 + g) * HB_ROW_U4 + (warp * 4 + tp) * 4 + t);
}

template <int R>
__device__ __forceinline__ void w2w_step(const W2WBuf& B, int up, int par, int warp, int lane, const int (&rid)[4], const float (&wt)[4],
                                         float* __restrict__ red, float* __restrict__ out) {
  const int g = lane >> 2;
  const float zero[4] = {0.f, 0.f, 0.f, 0.f};
  float acc[W2_RTU][2][4];
#pragma unroll
  for (int j = 0; j < W2_RTU; ++j) {
    uint32_t f[32];
    dequant<R>(B.a[j], B.b[j], B.a[j ^ 1], B.b[j ^ 1], f);
    const float s0 = bf_lo(B.s[j]), s1 = bf_hi(B.s[j]);
#pragma unroll
    for (int n = 0; n < 2; ++n) {
#pragma unroll
      for (int tp = 0; tp < 4; ++tp) {
        const uint4 hv = B.h[n][tp];
        if (tp == 0) mma16816(acc[j][n], f[0], f[1], f[2], f[3], hv.x, hv.y, zero);
        else         mma16816(acc[j][n], f[8 * tp + 0], f[8 * tp + 1], f[8 * tp + 2], f[8 * tp + 3], hv.x, hv.y, acc[j][n]);
        mma16816(acc[j][n], f[8 * tp + 4], f[8 * tp + 5], f[8 * tp + 6], f[8 * tp + 7], hv.z, hv.w, acc[j][n]);
      }
      acc[j][n][0] *= s0; acc[j][n][1] *= s0; acc[j][n][2] *= s1; acc[j][n][3] *= s1;
    }
  }
  constexpr int PER = W2_RTU * 2 * 4;
  float* r = red + size_t(par) * WARPS * 32 * PER;
#pragma unroll
  for (int j = 0; j < W2_RTU; ++j)
#pragma unroll
    for (int n = 0; n < 2; ++n)
      *reinterpret_cast<float4*>(r + (warp * 32 + lane) * PER + (j * 2 + n) * 4) = make_float4(acc[j][n][0], acc[j][n][1], acc[j][n][2], acc[j][n][3]);
  __syncthreads();
  if (warp == up % WARPS) {
#pragma unroll
    for (int n = 0; n < 2; ++n)
#pragma unroll
      for (int j = 0; j < W2_RTU; ++j) {
        float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int w = 0; w < WARPS; ++w) {
          const float4 p = *reinterpret_cast<const float4*>(r + (w * 32 + lane) * PER + (j * 2 + n) * 4);
          s.x += p.x; s.y += p.y; s.z += p.z; s.w += p.w;
        }
        const int row = (up * W2_RTU + j) * 16 + g;
        const int r0 = rid[n * 2], r1 = rid[n * 2 + 1];
        const float w0 = wt[n * 2], w1 = wt[n * 2 + 1];
        if (r0 >= 0) { atomicAdd(out + size_t(r0) * H + row, w0 * s.x); atomicAdd(out + size_t(r0) * H + row + 8, w0 * s.z); }
        if (r1 >= 0) { atomicAdd(out + size_t(r1) * H + row, w1 * s.y); atomicAdd(out + size_t(r1) * H + row + 8, w1 * s.w); }
      }
  }
}

__global__ void __launch_bounds__(THREADS, 4)
w2_wide(Tables L, Work W) {
  extern __shared__ __align__(16) uint8_t smem[];
  float* red = reinterpret_cast<float*>(smem);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, t = lane & 3;
  const int total = *W.n_blocks * W2_UPE;
  W2WBuf b0;
  int par = 0;
  for (int u = blockIdx.x; u < total; u += gridDim.x) {
    const int block = u / W2_UPE, up = u - block * W2_UPE;
    const int* rows = W.rid + size_t(block) * WIDE_ROWS;
    const float* wts = W.rw + size_t(block) * WIDE_ROWS;
    const int rid[4] = {rows[2 * t], rows[2 * t + 1], rows[8 + 2 * t], rows[8 + 2 * t + 1]};
    const float wt[4] = {wts[2 * t], wts[2 * t + 1], wts[8 + 2 * t], wts[8 + 2 * t + 1]};
    const int expert = W.blk[block];
    const uint8_t* base = L.w2 + L.off2[expert];
    const int fmt = L.fmt2[expert];
    if (fmt == 3) { w2w_load<3>(b0, base, W.hB, block, up, warp, lane); w2w_step<3>(b0, up, par, warp, lane, rid, wt, red, W.acc); }
    else if (fmt == 4) { w2w_load<4>(b0, base, W.hB, block, up, warp, lane); w2w_step<4>(b0, up, par, warp, lane, rid, wt, red, W.acc); }
    else { w2w_load<2>(b0, base, W.hB, block, up, warp, lane); w2w_step<2>(b0, up, par, warp, lane, rid, wt, red, W.acc); }
    par ^= 1;
  }
}

// fp32 accumulator to bf16 output, and clear the accumulator for the next layer.
// ---------------------------------------------------------------- skinny BF16 multiply
// y[m][n] = sum_k x[m][k] W[n][k]. Replaces cuBLAS on SM80 for the wide skinny projections, in one launch at any batch size.
// W is stored only in this layout: [row tile][packet][lane][P], 16 rows x 128 k per packet, one uint32 per P holding two BF16:
//   row = tile*16 + (lane>>2) + 8*(P&1), k = packet*128 + 16*(P>>2) + 2*(lane&3) + 8*((P>>1)&1) + half
// which is the T3 packet's order, so each lane loads its 32 words in 8 ldg128s. Up to 8 x rows sit in LDS per chunk.
constexpr int GEMM_RED = 2 * WARPS * 32 * 4 * 4;
__global__ void __launch_bounds__(THREADS)
gemm_bf16(const uint32_t* __restrict__ w, const __nv_bfloat16* __restrict__ x, long sx, __nv_bfloat16* __restrict__ y, long sy,
          int rts, int kps, int K, int N, int M, int chunk_rows, int split, int xs_row) {
  extern __shared__ __align__(16) uint8_t smem[];
  uint4* xs = reinterpret_cast<uint4*>(smem);
  __nv_bfloat16* xsb = reinterpret_cast<__nv_bfloat16*>(xs);
  float* red = reinterpret_cast<float*>(smem + size_t(chunk_rows) * xs_row * 16);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int chunks = (M + chunk_rows - 1) / chunk_rows, total = chunks * split;
  int staged = -1, par = 0;
  for (int unit = blockIdx.x; unit < total; unit += gridDim.x) {
    const int chunk = unit / split, part = unit - chunk * split;
    const int base = chunk * chunk_rows, rows = min(chunk_rows, M - base);
    if (chunk != staged) {
      for (int n = 0; n < rows; ++n) {
        const uint4* src = reinterpret_cast<const uint4*>(x + size_t(base + n) * sx);
        for (int i = threadIdx.x; i < (K + 7) / 8; i += THREADS) {
          const uint4 v = src[i];
          const uint32_t e[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int j = 0; j < 4; ++j)
            *reinterpret_cast<uint32_t*>(xsb + size_t(n) * xs_row * 8 + frag_off(i * 8 + 2 * j)) = e[j];
        }
        // The packed weights are zero past K, but 0 * stale NaN is NaN.
        for (int i = (K + 7) / 8 + threadIdx.x; i < kps * 16; i += THREADS) {
#pragma unroll
          for (int j = 0; j < 4; ++j)
            *reinterpret_cast<uint32_t*>(xsb + size_t(n) * xs_row * 8 + frag_off(i * 8 + 2 * j)) = 0u;
        }
      }
      __syncthreads();
      staged = chunk;
    }
    const int first = part * rts / split, last = (part + 1) * rts / split;
    for (int rt = first; rt < last; ++rt) {
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      for (int kp = warp; kp < kps; kp += WARPS) {
        const uint4* p = reinterpret_cast<const uint4*>(w + (size_t(rt) * kps + kp) * 1024 + lane * 32);
        uint32_t f[32];
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const uint4 v = ldg_u4(p + q);
          f[q * 4] = v.x; f[q * 4 + 1] = v.y; f[q * 4 + 2] = v.z; f[q * 4 + 3] = v.w;
        }
#pragma unroll
        for (int tp = 0; tp < 4; ++tp) {
          // Lanes past the staged rows feed columns that are never stored, so any valid row will do.
          const uint4 xv = xs[size_t(min(g, chunk_rows - 1)) * xs_row + (kp * 4 + tp) * 4 + t];
          mma16816(acc, f[8 * tp + 0], f[8 * tp + 1], f[8 * tp + 2], f[8 * tp + 3], xv.x, xv.y, acc);
          mma16816(acc, f[8 * tp + 4], f[8 * tp + 5], f[8 * tp + 6], f[8 * tp + 7], xv.z, xv.w, acc);
        }
      }
      float* r = red + size_t(par) * WARPS * 32 * 4;
      *reinterpret_cast<float4*>(r + (warp * 32 + lane) * 4) = make_float4(acc[0], acc[1], acc[2], acc[3]);
      __syncthreads();
      if (warp == rt % WARPS) {
        float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int v = 0; v < WARPS; ++v) {
          const float4 q = *reinterpret_cast<const float4*>(r + (v * 32 + lane) * 4);
          s.x += q.x; s.y += q.y; s.z += q.z; s.w += q.w;
        }
        const int c0 = rt * 16 + g, c1 = c0 + 8, n0 = 2 * t, n1 = n0 + 1;
        if (n0 < rows) {
          __nv_bfloat16* out = y + size_t(base + n0) * sy;
          if (c0 < N) out[c0] = __float2bfloat16(s.x);
          if (c1 < N) out[c1] = __float2bfloat16(s.z);
        }
        if (n1 < rows) {
          __nv_bfloat16* out = y + size_t(base + n1) * sy;
          if (c0 < N) out[c0] = __float2bfloat16(s.y);
          if (c1 < N) out[c1] = __float2bfloat16(s.w);
        }
      }
      par ^= 1;
    }
  }
}

__global__ void __launch_bounds__(256)
finish(float* __restrict__ acc, __nv_bfloat16* __restrict__ out, long stride, int M) {
  const size_t total = size_t(M) * H;
  for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < total; i += gridDim.x * 256ull) {
    const size_t row = i / H, col = i - row * H;
    out[row * stride + col] = __float2bfloat16(acc[i]);
    acc[i] = 0.f;
  }
}

}  // namespace t3

// ======================================================================== host API (ctypes)
using namespace t3;

extern "C" {

struct T3Grid { int grid13, grid2, grid13w, grid2w; };
int g_mask = 7;                // which kernels run, for timing
int g_wide_from = 48;          // rows at which the 16-row path takes over
static T3Grid g_grid = {0, 0};
static int g_gemm_max_smem = 0, g_sms = 0;

void t3_init() {
  if (g_grid.grid13) return;
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  CK(cudaFuncSetAttribute(w13_swiglu, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM13));
#ifndef T3_W2_TILE
  CK(cudaFuncSetAttribute(w2_scatter, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM2));
#endif
  int o13 = 0, o2 = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o13, w13_swiglu, W13_THREADS, SMEM13));
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o2, w2_scatter, W2_THREADS, SMEM2));
  CK(cudaFuncSetAttribute(w13_wide, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM13W));
  CK(cudaFuncSetAttribute(w2_wide, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM2W));
  int o13w = 0, o2w = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o13w, w13_wide, WIDE_THREADS, SMEM13W));
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o2w, w2_wide, THREADS, SMEM2W));
  g_grid.grid13w = prop.multiProcessorCount * o13w;
  g_grid.grid2w = prop.multiProcessorCount * o2w;
  fprintf(stderr, "t3moe wide: w13 %d CTA/SM, w2 %d CTA/SM\n", o13w, o2w);
  g_grid.grid13 = prop.multiProcessorCount * o13;
  g_grid.grid2 = prop.multiProcessorCount * o2;
  cudaFuncAttributes a13, a2;
  CK(cudaFuncGetAttributes(&a13, w13_swiglu));
  CK(cudaFuncGetAttributes(&a2, w2_scatter));
  g_gemm_max_smem = int(prop.sharedMemPerBlockOptin);
  g_sms = prop.multiProcessorCount;
  CK(cudaFuncSetAttribute(gemm_bf16, cudaFuncAttributeMaxDynamicSharedMemorySize, g_gemm_max_smem));
  fprintf(stderr, "t3moe: %s, %d SMs | w13 %d regs, %d CTA/SM | w2 %d regs, %d CTA/SM\n", prop.name, prop.multiProcessorCount, a13.numRegs, o13, a2.numRegs, o2);
}

// All pointers are device addresses. Output is bf16 with row stride `stride`, in elements.
void t3_forward(void* stream_, const void* w13, const void* w2, const void* off13, const void* off2, const void* fmt13, const void* fmt2,
                void* cnt, void* nxt, void* base, void* blk, void* n_blocks, void* rid, void* rw, void* hB, void* acc,
                const void* x, const void* ids, const void* tw, int M, void* out, long stride) {
  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_);
  Tables L{(const uint8_t*)w13, (const uint8_t*)w2, (const uint64_t*)off13, (const uint64_t*)off2, (const uint8_t*)fmt13, (const uint8_t*)fmt2};
  Work W{(int*)cnt, (int*)nxt, (int*)base, (int*)blk, (int*)n_blocks, (int*)rid, (float*)rw, (__nv_bfloat16*)hB, (float*)acc};
  const int* ids_p = (const int*)ids;
  const float* tw_p = (const float*)tw;
  const bool wide = M >= g_wide_from;
  const int br = wide ? WIDE_ROWS : BLOCK_ROWS;
  if (M * TOPK <= 2048) {
    route_small<<<1, 512, 0, st>>>(ids_p, tw_p, M, W, br);
  } else {
    const int g = std::min(64, (M * TOPK + 511) / 512);
    route_zero<<<1, 512, 0, st>>>(W);
    route_count<<<g, 512, 0, st>>>(ids_p, M, W);
    route_scan<<<1, 512, 0, st>>>(W, br);
    route_place<<<g, 512, 0, st>>>(ids_p, tw_p, M, W, br);
  }
  if (wide) {
    w13_wide<<<g_grid.grid13w, WIDE_THREADS, SMEM13W, st>>>(L, W, (const __nv_bfloat16*)x);
    w2_wide<<<g_grid.grid2w, THREADS, SMEM2W, st>>>(L, W);
  } else {
  if (g_mask & 1) {
    // Only as many activation rows in LDS as a block can have: more CTAs fit, and warps are what this kernel is short of.
    const int xs_rows = std::min(BLOCK_ROWS, M);
    const int smem = xs_rows * XS_ROW_U4 * 16 + RED13;
    int occupancy = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, w13_swiglu, W13_THREADS, smem));
#ifdef T3_W13_TILE
    const int grid13 = std::max(1, std::min(g_sms * occupancy, M * TOPK * ((W13_RT + W13_TILE_WARPS - 1) / W13_TILE_WARPS)));
#else
    const int grid13 = g_sms * occupancy;
#endif
    w13_swiglu<<<grid13, W13_THREADS, smem, st>>>(L, W, (const __nv_bfloat16*)x, xs_rows);
  }
#ifdef T3_W2_TILE
  const int grid2 = std::max(1, std::min(g_grid.grid2, (M * TOPK * W2_UPE + W2_TILE_WARPS - 1) / W2_TILE_WARPS));
#else
  const int grid2 = g_grid.grid2;
#endif
  if (g_mask & 2) w2_scatter<<<grid2, W2_THREADS, SMEM2, st>>>(L, W);
  }
  if (!(g_mask & 4)) return;
  const int gf = std::min(2 * g_sms, std::max(1, int((size_t(M) * H + 255) / 256)));
  finish<<<gf, 256, 0, st>>>((float*)acc, (__nv_bfloat16*)out, stride, M);
}

// y[M][N] = x[M][K] W^T, W in gemm_bf16's packet layout with rts row tiles and kps packets. Strides are in elements.
void t3_gemm(void* stream_, const void* w, const void* x, long sx, void* y, long sy, int rts, int kps, int K, int N, int M) {
  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_);
  const int xs_row = kps * 128 / 8 + 4;
  const int cap = std::max(1, std::min(8, (g_gemm_max_smem - GEMM_RED) / (xs_row * 16)));
  const int chunk_rows = std::min(M, cap), chunks = (M + chunk_rows - 1) / chunk_rows;
  const int smem = chunk_rows * xs_row * 16 + GEMM_RED;
  int occupancy = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, gemm_bf16, THREADS, smem));
  const int grid = std::max(1, g_sms * occupancy);
  const int split = std::max(1, std::min(rts, grid / chunks));
  const int total = chunks * split;
  gemm_bf16<<<std::min(grid, total), THREADS, smem, st>>>((const uint32_t*)w, (const __nv_bfloat16*)x, sx, (__nv_bfloat16*)y, sy,
                                                          rts, kps, K, N, M, chunk_rows, split, xs_row);
}

}  // extern "C"

#ifdef T3_MAIN
// Speed bench on synthetic weights, graph replay, same routing model as w3t.cu.
static void gen_routing(std::mt19937& rng, int M, double overlap, std::vector<int>& ids, std::vector<float>& tw) {
  ids.assign(size_t(M) * TOPK, 0); tw.assign(size_t(M) * TOPK, 0.f);
  std::uniform_real_distribution<double> U(0, 1);
  std::uniform_int_distribution<int> EX(0, E - 1);
  for (int m = 0; m < M; ++m) {
    std::vector<int> sel;
    if (m > 0 && (m % 4) != 0)
      for (int k = 0; k < TOPK; ++k) if (U(rng) < overlap) sel.push_back(ids[(m - 1) * TOPK + k]);
    while ((int)sel.size() < TOPK) { int e = EX(rng); if (std::find(sel.begin(), sel.end(), e) == sel.end()) sel.push_back(e); }
    double z = 0; std::vector<double> l(TOPK);
    for (int k = 0; k < TOPK; ++k) { l[k] = std::exp(U(rng) * 2.0); z += l[k]; }
    for (int k = 0; k < TOPK; ++k) { ids[m * TOPK + k] = sel[k]; tw[m * TOPK + k] = float(l[k] / z); }
  }
}
__global__ void init_rand(uint8_t* p, size_t bytes, uint32_t seed) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < bytes / 4; i += size_t(gridDim.x) * blockDim.x) {
    uint32_t h = uint32_t(i) * 0x9E3779B1u ^ seed;
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
    reinterpret_cast<uint32_t*>(p)[i] = h;
  }
}
__global__ void fix_scales(uint8_t* p, size_t n, int pkt, int soff) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n * 8; i += size_t(gridDim.x) * blockDim.x)
    reinterpret_cast<uint32_t*>(p + (i / 8) * pkt + soff)[i % 8] = 0x3C233C23u;
}
int main(int argc, char** argv) {
  int M = argc > 1 ? atoi(argv[1]) : 4, fmt_w13 = argc > 2 ? atoi(argv[2]) : 3, fmt_w2 = argc > 3 ? atoi(argv[3]) : 3, L = 48, R = 8, iters = 50;
  t3_init();
  if (argc > 4) g_mask = atoi(argv[4]);
  if (argc > 5) g_wide_from = atoi(argv[5]);
  const int pk13 = fmt_w13 == 3 ? 800 : 544, pk2 = fmt_w2 == 3 ? 800 : fmt_w2 == 4 ? 1056 : 544;
  const int sf13 = fmt_w13 == 3 ? 768 : 512, sf2 = fmt_w2 == 3 ? 768 : fmt_w2 == 4 ? 1024 : 512;
  const size_t eb13 = size_t(W13_RT) * W13_KP * pk13, eb2 = size_t(W2_RT) * W2_KP * pk2;
  uint8_t *w13, *w2;
  CK(cudaMalloc(&w13, R * E * eb13)); CK(cudaMalloc(&w2, R * E * eb2));
  init_rand<<<4096, 256>>>(w13, R * E * eb13, 1);
  init_rand<<<4096, 256>>>(w2, R * E * eb2, 2);
  fix_scales<<<2048, 256>>>(w13, R * E * W13_RT * W13_KP, pk13, sf13);
  fix_scales<<<2048, 256>>>(w2, R * E * W2_RT * W2_KP, pk2, sf2);
  std::vector<uint64_t> off13(E), off2(E);
  std::vector<uint8_t> f13(E, fmt_w13), f2(E, fmt_w2);
  for (int e = 0; e < E; ++e) { off13[e] = e * eb13; off2[e] = e * eb2; }
  uint64_t *doff13, *doff2; uint8_t *df13, *df2;
  CK(cudaMalloc(&doff13, E * 8)); CK(cudaMalloc(&doff2, E * 8)); CK(cudaMalloc(&df13, E)); CK(cudaMalloc(&df2, E));
  CK(cudaMemcpy(doff13, off13.data(), E * 8, cudaMemcpyHostToDevice)); CK(cudaMemcpy(doff2, off2.data(), E * 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(df13, f13.data(), E, cudaMemcpyHostToDevice)); CK(cudaMemcpy(df2, f2.data(), E, cudaMemcpyHostToDevice));
  const int maxrows = M * TOPK + E * WIDE_ROWS;
  int *cnt, *nxt, *base, *blk, *nb, *rid; float *rw, *acc; __nv_bfloat16 *hB, *x, *out;
  CK(cudaMalloc(&cnt, E * 4)); CK(cudaMalloc(&nxt, E * 4)); CK(cudaMalloc(&base, E * 4)); CK(cudaMalloc(&blk, maxrows * 4)); CK(cudaMalloc(&nb, 4));
  CK(cudaMalloc(&rid, maxrows * 4)); CK(cudaMalloc(&rw, maxrows * 4)); CK(cudaMalloc(&hB, size_t(maxrows) * I * 2));
  CK(cudaMalloc(&acc, size_t(M) * H * 4)); CK(cudaMalloc(&x, size_t(M) * H * 2)); CK(cudaMalloc(&out, size_t(M) * H * 2));
  CK(cudaMemset(acc, 0, size_t(M) * H * 4));
  std::mt19937 rng(1234);
  std::vector<__nv_bfloat16> xh(size_t(M) * H);
  std::normal_distribution<float> N01(0, 1);
  for (auto& v : xh) v = __float2bfloat16(N01(rng));
  CK(cudaMemcpy(x, xh.data(), size_t(M) * H * 2, cudaMemcpyHostToDevice));
  std::vector<int*> dids(L); std::vector<float*> dtw(L);
  size_t touched = 0;
  for (int l = 0; l < L; ++l) {
    std::vector<int> ids; std::vector<float> tw;
    gen_routing(rng, M, 0.6, ids, tw);
    CK(cudaMalloc(&dids[l], ids.size() * 4)); CK(cudaMalloc(&dtw[l], tw.size() * 4));
    CK(cudaMemcpy(dids[l], ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dtw[l], tw.data(), tw.size() * 4, cudaMemcpyHostToDevice));
    std::vector<int> u(ids); std::sort(u.begin(), u.end()); u.erase(std::unique(u.begin(), u.end()), u.end());
    touched += u.size();
  }
  cudaStream_t st; CK(cudaStreamCreate(&st));
  cudaGraph_t graph; cudaGraphExec_t ex;
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
  for (int l = 0; l < L; ++l)
    t3_forward(st, w13 + (l % R) * E * eb13, w2 + (l % R) * E * eb2, doff13, doff2, df13, df2, cnt, nxt, base, blk, nb, rid, rw, hB, acc, x, dids[l], dtw[l], M, out, H);
  CK(cudaStreamEndCapture(st, &graph));
  CK(cudaGraphInstantiate(&ex, graph, 0));
  std::vector<double> times;
  for (int i = 0; i < iters + 5; ++i) {
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a, st)); CK(cudaGraphLaunch(ex, st)); CK(cudaEventRecord(b, st));
    CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    if (i >= 5) times.push_back(ms);
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
  }
  std::sort(times.begin(), times.end());
  const double bytes = double(touched) * (eb13 + eb2);
  printf("M=%d fmt %d/%d: median %.0f us, min %.0f us; distinct experts/layer %.1f; kernel %.0f GB/s\n", M, fmt_w13, fmt_w2, times[times.size() / 2] * 1e3, times[0] * 1e3,
         double(touched) / L, bytes / (times[times.size() / 2] * 1e-3) / 1e9);
  return 0;
}
#endif
