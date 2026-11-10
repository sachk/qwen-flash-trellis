// Exact-output acceleration of the PINNED vLLM top-k/top-p mask; SM80.
// nvcc -O3 -arch=sm_80 -std=c++17 -shared -Xcompiler -fPIC topkmask.cu -o libtopkmask.so
// Add -Xptxas=-v for register/spill reporting. Do NOT use --use_fast_math.
//
// IMPORTANT: this is a CUDA/Triton pipeline, NOT a standalone sort replacement.
// topk_mask.py owns the launch sequence; apply.py extracts the pinned reference's
// scalar/pivot arithmetic into topk_mask_reference.py with asserted anchors.
// CUDA parallelizes gather/compaction and final masking; the original Triton
// search retains its operation order, 8192/4096 tiles and eight warps. Replacing
// that search by sorted cumulative softmax is NOT bitwise compatible.
//
// Semantics found in vllm/v1/sample/ops/topk_topp_triton.py (supplied task15 copy):
// * FP32 [rows,V], -inf mask, in-place for vocab stride=1 (row stride may differ).
//   Non-unit vocab strides are copied to contiguous by the reference wrapper.
//   Kept values retain their original bits. No temperature/softmax output here.
// * K/P are optional per-row tensors. K=None disables top-k; K>=V skips it.
//   Request top_k<=0 is normalized to V by SamplingStates.add_request. RAW K=0
//   passed directly to Triton is NOT a disabled sentinel: its bounded search
//   executes. We preserve it by running the original search, not normalizing it.
// * Top-k uses sampled Gaussian truncation then at most 18 ternary iterations,
//   strict logit>pivot, and abs(value-boundary)<1e-9 duplicate matching. When
//   converged it keeps the earliest vocabulary indices from the kth-value tie,
//   NOT all ties. The cap/range-collapse exit can differ from exact sorted top-k.
//   If <=k finite logits exist, top-k is a no-op (top-p can still run).
// * Top-p after top-k uses exp(logit-max)/sum_exp of the top-k survivors, FP32
//   block reductions in ORIGINAL INDEX order, not sorted order. It searches
//   probability pivots for mass>=p and mass-min_boundary*count<p; boundary
//   budget=count-uint32((mass-p)/min_boundary), earliest indices first. Final
//   pivot/boundary are mapped back with log; final comparison is strict >.
//   This is inclusive mass>=p in the convergence test, not an exclusive cumsum.
//   Duplicate matching after log/exp round-trip can miss exact original ties.
//   The fast outlier path retains its original candidate buffer when computing
//   normalized probabilities (including candidates removed by top-k): preserve
//   this reference behavior rather than substituting a mathematically nicer one.
// * P=None or P>=1 skips top-p. K-disabled/P=1 rows get NO stores. Uniform rows,
//   a NaN pivot, or pivot>=max may also produce no masking; no promise of k or
//   nucleus cardinality in these cases. All -inf rows are untouched. NaN/+inf
//   inputs are not sanitized: original comparisons/reductions/search determine
//   whether NaNs survive or are replaced by -inf. See the checker cases.
// * CUDA rows<=64 with no active top-k (K=None or K>=V) use vLLM's separate
//   split top-p: online-softmax partials, 5 rounds of 8 geometric pivots,
//   tightest mass>=p fallback, near-uniform rows untouched, probability-space
//   equality for ties. Its duplicate prefix includes preceding slices, but the
//   pinned mask does NOT advance that prefix between tiles within a slice.
//   We call this existing pipeline unchanged, including all its edge behavior.
// * Active top-k with <=k finite values uses the monolithic standalone top-p
//   (sample mean+10*std normalization), not the split pipeline.
//
// Architecture: seed (original Triton sample arithmetic), CUDA gather [rows,31]
// for V=248320, CUDA stable compact, original pivot search with precomputed
// first pass, CUDA boundary counts, CUDA parallel mask, reference split top-p
// if P is supplied. NaN/+inf/-0, invalid/nonpositive K, failed truncation and
// dense outlier sets use the unmodified reference first pass, selected entirely
// on GPU. No sort fallback: retaining reference searches is necessary for exact
// compatibility even on non-ties. No host synchronization, atomics or counters.
// Scratch is per invocation (graph-pool allocations in Python), so overlapping
// streams/graphs do not share candidate buffers. Deterministic index order.
//
// Bounds: 1..64 rows, 1..262144 vocabulary; FP32 logits, int32 K, FP32 P.
// Gather uses 256 threads, 32 values/thread and CUB transpose + integer scan;
// about 32 KiB shared and ~48-80 registers/thread expected, NOT measured.
// Common k20 finite Gaussian path: 6 launches plus 7 trivial split launches
// when P supplied. UNMEASURED estimate M4 40-90 us, M16 70-140 us; original
// pivot search and launch overhead may dominate. No 20-us claim without timing.
// Fallback-heavy rows can be slower than baseline. Keep disabled until checked.
// Validation at handoff: host-only Python/code-generation/entrypoint smoke and
// integer tie-prefix review. CUDA compilation, bitwise numerical comparison,
// sanitizer/graph checks and timings are deliberately left to the build host.
//
// ABI: topkmask_gather(stream,logits,k,seed,local,tiles,rows,vocab,row_stride)
//      topkmask_compact(stream,k,local,tiles,buffer,rowmeta,rows,vocab)
//      topkmask_counts(stream,logits,decisions,counts,rows,vocab,row_stride)
//      topkmask_mask(stream,logits,decisions,counts,rows,vocab,row_stride)
// Float seed [rows]; local [rows,ceil(V/8192),8192]; buffer [rows,V].
// Tiles int32 [rows,ceil(V/8192),5], rowmeta int32 [rows,5],
// decisions int32 [rows,5], counts int32 [rows,ceil(V/8192)].
// Rowmeta={fast,min_bits,max_bits,finite_count,outlier_count}.
// Decisions={apply,pivot_bits,duplicate_bits,num_duplicate,num_keep}.
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cstdint>
#include <cstddef>

// CUDA 13's CCCL dropped cub::Min/cub::Max; these reproduce their old
// definitions exactly (CUB_MIN/CUB_MAX comparisons), so reductions are unchanged.
struct CubMin { template <class T> __device__ __forceinline__ T operator()(const T& a, const T& b) const { return (b < a) ? b : a; } };
struct CubMax { template <class T> __device__ __forceinline__ T operator()(const T& a, const T& b) const { return (a < b) ? b : a; } };

namespace topkmask {
constexpr int TILE = 8192, THREADS = 256, ITEMS = TILE / THREADS;
struct Tile { float lo, hi; uint32_t finite, count, exceptional; };
struct Row { uint32_t fast; float lo, hi; uint32_t finite, count; };
struct Decision { uint32_t apply; float pivot, duplicate; uint32_t duplicates, keep; };
static_assert(sizeof(Tile) == 20 && sizeof(Row) == 20 && sizeof(Decision) == 20);

inline bool valid(int rows, int vocab, int64_t stride) {
  return rows >= 1 && rows <= 64 && vocab >= 1 && vocab <= 262144 && stride >= vocab;
}
__device__ __forceinline__ bool active(const int32_t* k, int row, int vocab) {
  return k && k[row] > 0 && k[row] < vocab;
}

__global__ void gather(const float* logits, const int32_t* k, const float* seed,
                       float* local, Tile* tiles, int vocab, int64_t stride) {
  using Load = cub::BlockLoad<float, THREADS, ITEMS, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using ReduceF = cub::BlockReduce<float, THREADS>;
  using ReduceU = cub::BlockReduce<uint32_t, THREADS>;
  using Scan = cub::BlockScan<uint32_t, THREADS>;
  __shared__ union Temp {
    typename Load::TempStorage load;
    typename ReduceF::TempStorage rf;
    typename ReduceU::TempStorage ru;
    typename Scan::TempStorage scan;
  } temp;
  __shared__ Tile result;
  const int row = blockIdx.y, tile = blockIdx.x, nt = gridDim.x;
  const int start = tile * TILE;
  if (!active(k, row, vocab)) {
    if (threadIdx.x == 0) tiles[row * nt + tile] = {CUDART_INF_F, -CUDART_INF_F, 0, 0, 0};
    return;
  }
  float values[ITEMS];
  Load(temp.load).Load(logits + int64_t(row) * stride + start, values,
                       min(TILE, vocab - start), -CUDART_INF_F);
  __syncthreads();
  const float pivot = seed[row];
  float lo = CUDART_INF_F, hi = -CUDART_INF_F;
  uint32_t finite = 0, selected = 0, exceptional = 0;
#pragma unroll
  for (int j = 0; j < ITEMS; ++j) {
    const float v = values[j];
    lo = fminf(lo, v > -CUDART_INF_F ? v : CUDART_INF_F);
    hi = fmaxf(hi, v);
    finite += v > -CUDART_INF_F;
    selected += v > pivot;
    const uint32_t bits = __float_as_uint(v);
    exceptional |= (bits == 0x80000000u || v == CUDART_INF_F || isnan(v));
  }
  const float min_value = ReduceF(temp.rf).Reduce(lo, CubMin());
  if (threadIdx.x == 0) result.lo = min_value;
  __syncthreads();
  const float max_value = ReduceF(temp.rf).Reduce(hi, CubMax());
  if (threadIdx.x == 0) result.hi = max_value;
  __syncthreads();
  const uint32_t finite_count = ReduceU(temp.ru).Sum(finite);
  if (threadIdx.x == 0) result.finite = finite_count;
  __syncthreads();
  const uint32_t any_exception = ReduceU(temp.ru).Sum(exceptional);
  if (threadIdx.x == 0) result.exceptional = any_exception;
  __syncthreads();
  uint32_t offset, total;
  Scan(temp.scan).ExclusiveSum(selected, offset, total);
  if (threadIdx.x == 0) result.count = total;
  // BlockLoad supplies blocked-order registers, so thread prefix + register
  // order is vocabulary order, independent of execution/warp arrival order.
  float* dst = local + (size_t(row) * nt + tile) * TILE;
#pragma unroll
  for (int j = 0; j < ITEMS; ++j)
    if (values[j] > pivot) dst[offset++] = values[j];
  __syncthreads();
  if (threadIdx.x == 0) tiles[row * nt + tile] = result;
}

__global__ void compact(const int32_t* k, const float* local, const Tile* tiles,
                        float* buffer, Row* rows, int vocab) {
  const int row = blockIdx.y, tile = blockIdx.x, nt = gridDim.x;
  __shared__ uint32_t prefix, count, fast;
  if (threadIdx.x == 0) {
    Row r{0, CUDART_INF_F, -CUDART_INF_F, 0, 0};
    uint32_t exceptions = 0, before = 0;
    for (int t = 0; t < nt; ++t) {
      const Tile x = tiles[row * nt + t];
      r.lo = fminf(r.lo, x.lo);
      r.hi = fmaxf(r.hi, x.hi);
      r.finite += x.finite;
      r.count += x.count;
      exceptions |= x.exceptional;
      if (t < tile) before += x.count;
    }
    r.lo = fminf(r.lo, r.hi);
    // Do not bypass the reference first pass unless its truncated-buffer
    // branch is selected. Cap dense candidate sets to avoid wasteful copies.
    r.fast = active(k, row, vocab) && exceptions == 0 && r.count > uint32_t(k[row])
             && r.count <= uint32_t(vocab / 4);
    if (tile == 0) rows[row] = r;
    prefix = before;
    count = tiles[row * nt + tile].count;
    fast = r.fast;
  }
  __syncthreads();
  if (!fast) return;
  const float* src = local + (size_t(row) * nt + tile) * TILE;
  for (uint32_t i = threadIdx.x; i < count; i += blockDim.x)
    buffer[size_t(row) * vocab + prefix + i] = src[i];
}

__global__ void count_ties(const float* logits, const Decision* decisions,
                           uint32_t* counts, int vocab, int64_t stride) {
  using Reduce = cub::BlockReduce<uint32_t, THREADS>;
  __shared__ typename Reduce::TempStorage temp;
  const int row = blockIdx.y, tile = blockIdx.x;
  const Decision d = decisions[row];
  uint32_t count = 0;
  if (d.apply && d.keep < d.duplicates) {
    for (int i = tile * TILE + threadIdx.x; i < min((tile + 1) * TILE, vocab); i += THREADS)
      count += fabsf(logits[int64_t(row) * stride + i] - d.duplicate) < 1e-9f;
  }
  const uint32_t total = Reduce(temp).Sum(count);
  if (threadIdx.x == 0) counts[row * gridDim.x + tile] = total;
}

__global__ void mask(float* logits, const Decision* decisions, const uint32_t* counts,
                     int vocab, int64_t stride) {
  using Scan = cub::BlockScan<uint32_t, THREADS>;
  __shared__ typename Scan::TempStorage temp;
  __shared__ uint32_t before;
  const int row = blockIdx.y, tile = blockIdx.x;
  const Decision d = decisions[row];
  if (!d.apply) return;  // Inactive rows are genuinely untouched, including NaN payloads.
  const bool ties = d.keep < d.duplicates;
  if (threadIdx.x == 0) {
    uint32_t sum = 0;
    if (ties)
      for (int t = 0; t < tile; ++t) sum += counts[row * gridDim.x + t];
    before = sum;
  }
  __syncthreads();
  uint32_t preceding = before;
  for (int base = tile * TILE; base < min((tile + 1) * TILE, vocab); base += THREADS) {
    const int i = base + threadIdx.x;
    const bool valid_i = i < vocab;
    float v = valid_i ? logits[int64_t(row) * stride + i] : -CUDART_INF_F;
    bool keep = v > d.pivot;
    if (ties) {
      const uint32_t duplicate = valid_i && fabsf(v - d.duplicate) < 1e-9f;
      uint32_t rank, total;
      Scan(temp).InclusiveSum(duplicate, rank, total);
      keep &= !duplicate || rank + preceding <= d.keep;
      preceding += total;
      __syncthreads();
    }
    if (valid_i && !keep) logits[int64_t(row) * stride + i] = -CUDART_INF_F;
  }
}
}  // namespace topkmask

extern "C" cudaError_t topkmask_gather(cudaStream_t stream, const float* logits,
    const int32_t* k, const float* seed, float* local, void* tiles,
    int rows, int vocab, int64_t stride) {
  if (!topkmask::valid(rows, vocab, stride) || !logits || !seed || !local || !tiles)
    return cudaErrorInvalidValue;
  const dim3 grid((vocab + topkmask::TILE - 1) / topkmask::TILE, rows);
  topkmask::gather<<<grid, topkmask::THREADS, 0, stream>>>(
      logits, k, seed, local, static_cast<topkmask::Tile*>(tiles), vocab, stride);
  return cudaGetLastError();
}
extern "C" cudaError_t topkmask_compact(cudaStream_t stream, const int32_t* k,
    const float* local, const void* tiles, float* buffer, void* rowmeta,
    int rows, int vocab) {
  if (!topkmask::valid(rows, vocab, vocab) || !local || !tiles || !buffer || !rowmeta)
    return cudaErrorInvalidValue;
  const dim3 grid((vocab + topkmask::TILE - 1) / topkmask::TILE, rows);
  topkmask::compact<<<grid, topkmask::THREADS, 0, stream>>>(
      k, local, static_cast<const topkmask::Tile*>(tiles), buffer,
      static_cast<topkmask::Row*>(rowmeta), vocab);
  return cudaGetLastError();
}
extern "C" cudaError_t topkmask_counts(cudaStream_t stream, const float* logits,
    const void* decisions, void* counts, int rows, int vocab, int64_t stride) {
  if (!topkmask::valid(rows, vocab, stride) || !logits || !decisions || !counts)
    return cudaErrorInvalidValue;
  const dim3 grid((vocab + topkmask::TILE - 1) / topkmask::TILE, rows);
  topkmask::count_ties<<<grid, topkmask::THREADS, 0, stream>>>(logits,
      static_cast<const topkmask::Decision*>(decisions), static_cast<uint32_t*>(counts), vocab, stride);
  return cudaGetLastError();
}
extern "C" cudaError_t topkmask_mask(cudaStream_t stream, float* logits,
    const void* decisions, const void* counts, int rows, int vocab, int64_t stride) {
  if (!topkmask::valid(rows, vocab, stride) || !logits || !decisions || !counts)
    return cudaErrorInvalidValue;
  const dim3 grid((vocab + topkmask::TILE - 1) / topkmask::TILE, rows);
  topkmask::mask<<<grid, topkmask::THREADS, 0, stream>>>(logits,
      static_cast<const topkmask::Decision*>(decisions), static_cast<const uint32_t*>(counts), vocab, stride);
  return cudaGetLastError();
}
