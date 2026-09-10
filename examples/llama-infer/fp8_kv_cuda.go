//go:build !darwin && !tpu

package main

// FP8 KV cache: cudakv.FP8Cache storage behind LLAMA_INFER_FP8_KV=1.
//
// UNQUALIFIED. No GPU has run this path. It is reachable only when the engine
// is compiled with LLAMA_INFER_FP8_KV=1; the default path does not touch it.
// The tests that would qualify it need a CUDA device and have never been
// executed by their author:
//
//   - TestFP8KVKernelMatchesHost, fp8_kv_engine_linux_test.go: the store and
//     dequantize kernels reproduce the host codec in fp8_kv.go bit for bit.
//   - TestFP8KVEngineTracksDense, fp8_kv_engine_linux_test.go: an FP8 engine
//     and a dense engine built from the same weights agree within the bound
//     the test states. E4M3 keeps three mantissa bits, so the two cannot
//     agree exactly; how far they may drift on a real model is unmeasured.
//
// What it is. Every K and V element is divided by one scale, rounded to OCP
// FP8 E4M3 and stored as a byte in a cudakv.FP8Cache: a quarter of the dense
// cache's memory. cudaattn has no attention kernel that reads E4M3, so before
// each layer's attention the sequence's K and V rows are dequantized into a
// float32 scratch of one layer and the dense CausalFlashAttentionDecode runs
// on that. This is a storage saving, not a bandwidth saving: the dequantize
// pass reads one byte and writes four per element, and attention then reads
// the four back, so per token the path moves more than twice the bytes the
// dense path does. The saving that motivates FP8 caches elsewhere needs an
// attention kernel that consumes E4M3 directly, which is not written.
//
// Known limits, each a choice made to keep the change small:
//
//   - The scale is one number for every head and layer, from
//     LLAMA_INFER_FP8_KV_SCALE (default 1). cudakv.FP8LayerCache has a slot
//     per head; nothing here fills them differently.
//   - Exclusive with LLAMA_INFER_PAGED_KV=1 (the block pool holds float32)
//     and with LLAMA_INFER_CUDA_GRAPH=1 (the step is not graph-captured).
//   - Not safe for concurrent forward calls on one engine: the dequantize
//     scratch is one buffer per engine.
//   - inferbench's KV observation reads only the dense cache and skips this
//     one (observed_output_linux.go).

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaattn"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// fp8KVKernelsCUDA holds the store and dequantize kernels. The codec is the
// one in fp8_kv.go, operation for operation on the same bits, so the host
// reference and the device agree exactly; keep the two in step.
const fp8KVKernelsCUDA = `
__device__ __forceinline__ unsigned char fp8_e4m3(float x) {
    unsigned int b = __float_as_uint(x);
    unsigned char sign = (unsigned char)((b >> 31) << 7);
    b &= 0x7fffffffu;
    if (b > 0x7f800000u) return sign | 0x7f;
    if (b >= 0x43e00000u) return sign | 0x7e;
    unsigned int exp = b >> 23, mant = b & 0x7fffffu;
    if (exp >= 121u) {
        unsigned int code = ((exp - 120u) << 3) | (mant >> 20);
        unsigned int rem = mant & 0xfffffu;
        if (rem > 0x80000u || (rem == 0x80000u && (code & 1u))) code++;
        return sign | (unsigned char)code;
    }
    unsigned int shift = 141u - exp;
    if (shift >= 25u) return sign;
    unsigned int full = mant | 0x800000u;
    unsigned int m = full >> shift;
    unsigned int rem = full & ((1u << shift) - 1u);
    unsigned int half = 1u << (shift - 1u);
    if (rem > half || (rem == half && (m & 1u))) m++;
    return sign | (unsigned char)m;
}

__device__ __forceinline__ float fp8_e4m3_value(unsigned char c) {
    int e = (c >> 3) & 0xf, m = c & 7;
    float v;
    if (e == 0) v = (float)m * (1.0f / 512.0f);
    else if (e == 15 && m == 7) v = __int_as_float(0x7fc00000);
    else v = (float)(8 + m) * exp2f((float)(e - 10));
    return (c & 0x80) ? -v : v;
}

// FP8StoreKV writes row pos of one layer: k and v are one token's
// [kv_heads, head_dim], the caches are [max_seq_len, kv_heads, head_dim].
extern "C" __global__ void FP8StoreKV(
    const float* __restrict__ k, const float* __restrict__ v,
    unsigned char* __restrict__ k_cache, unsigned char* __restrict__ v_cache,
    const float* __restrict__ k_scale, const float* __restrict__ v_scale,
    int pos, int kv_heads, int head_dim)
{
    int kv_dim = kv_heads * head_dim;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kv_dim) return;
    int h = i / head_dim;
    k_cache[pos * kv_dim + i] = fp8_e4m3(k[i] / k_scale[h]);
    v_cache[pos * kv_dim + i] = fp8_e4m3(v[i] / v_scale[h]);
}

// FP8DequantKV expands rows 0..seq_len-1 of one layer into float32 buffers of
// the same layout.
extern "C" __global__ void FP8DequantKV(
    const unsigned char* __restrict__ k_cache, const unsigned char* __restrict__ v_cache,
    const float* __restrict__ k_scale, const float* __restrict__ v_scale,
    float* __restrict__ k_out, float* __restrict__ v_out,
    int seq_len, int kv_heads, int head_dim)
{
    int kv_dim = kv_heads * head_dim;
    int total = seq_len * kv_dim;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
        int h = (i % kv_dim) / head_dim;
        k_out[i] = fp8_e4m3_value(k_cache[i]) * k_scale[h];
        v_out[i] = fp8_e4m3_value(v_cache[i]) * v_scale[h];
    }
}
`

const (
	fp8StoreThreads   = 128
	fp8DequantThreads = 256
	fp8DequantBlocks  = 1024
)

// fp8KVRuntime owns the FP8 kernels and the one-layer float32 scratch the
// attention kernel reads. It borrows the engine's RoPE and attention kernels
// rather than compiling its own.
type fp8KVRuntime struct {
	dev     *cuda.Device
	tracker *cudamem.Tracker
	mod     *cuda.Module
	store   *cuda.Kernel
	dequant *cuda.Kernel
	rope    *cuda.Kernel
	attn    *boundedKernel

	// kScratch and vScratch are [maxSeqLen, kvHeads, headDim], the dense
	// layout the attention kernel expects.
	kScratch, vScratch cudamem.Slice[float32]

	layers, qHeads, kvHeads, headDim, maxSeqLen int
	scale                                       float32
}

func newFP8KVRuntime(dev *cuda.Device, tracker *cudamem.Tracker, rope *cuda.Kernel, attn *boundedKernel,
	layers, qHeads, kvHeads, headDim, maxSeqLen int, scale float32) (*fp8KVRuntime, error) {
	if dev == nil || rope == nil || attn == nil {
		return nil, errors.New("fp8 kv: nil device or kernel")
	}
	if layers <= 0 || qHeads <= 0 || kvHeads <= 0 || headDim <= 0 || maxSeqLen <= 0 || !(scale > 0) {
		return nil, fmt.Errorf("fp8 kv: invalid shape (layers=%d, q_heads=%d, kv_heads=%d, head_dim=%d, max_seq=%d, scale=%g)",
			layers, qHeads, kvHeads, headDim, maxSeqLen, scale)
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		return nil, fmt.Errorf("fp8 kv: compute capability: %w", err)
	}
	compiled, err := nvrtcx.Compile(fp8KVKernelsCUDA, nvrtcx.WithName("fp8_kv.cu"), nvrtcx.WithArch(major, minor))
	if err != nil {
		return nil, fmt.Errorf("fp8 kv: compile kernels: %w", err)
	}
	mod, err := loadCUDAProgram(dev, compiled)
	if err != nil {
		return nil, fmt.Errorf("fp8 kv: load module: %w", err)
	}
	r := &fp8KVRuntime{
		dev: dev, tracker: tracker, mod: mod, rope: rope, attn: attn,
		layers: layers, qHeads: qHeads, kvHeads: kvHeads, headDim: headDim, maxSeqLen: maxSeqLen, scale: scale,
	}
	fail := func(err error) (*fp8KVRuntime, error) {
		return nil, errors.Join(err, r.Close())
	}
	if r.store, err = mod.Function("FP8StoreKV"); err != nil {
		return fail(fmt.Errorf("fp8 kv: %w", err))
	}
	if r.dequant, err = mod.Function("FP8DequantKV"); err != nil {
		return fail(fmt.Errorf("fp8 kv: %w", err))
	}
	n := maxSeqLen * kvHeads * headDim
	if r.kScratch, err = cudamem.AllocTracked[float32](dev, tracker, n); err != nil {
		return fail(fmt.Errorf("fp8 kv: alloc K scratch: %w", err))
	}
	if r.vScratch, err = cudamem.AllocTracked[float32](dev, tracker, n); err != nil {
		return fail(fmt.Errorf("fp8 kv: alloc V scratch: %w", err))
	}
	return r, nil
}

// newCache allocates one sequence's FP8 cache with the runtime's scale written
// into every layer's per-head slots.
func (r *fp8KVRuntime) newCache() (*cudakv.FP8Cache, error) {
	c, err := cudakv.NewFP8Tracked(r.dev, r.tracker, r.layers, r.kvHeads, r.headDim, r.maxSeqLen)
	if err != nil {
		return nil, fmt.Errorf("fp8 kv: %w", err)
	}
	scales := make([]float32, r.kvHeads)
	for i := range scales {
		scales[i] = r.scale
	}
	for layer := 0; layer < r.layers; layer++ {
		lc, err := c.Layer(layer)
		if err == nil {
			err = lc.KScale.CopyFromHost(scales)
		}
		if err == nil {
			err = lc.VScale.CopyFromHost(scales)
		}
		if err != nil {
			_ = c.Close()
			return nil, fmt.Errorf("fp8 kv: write layer %d scales: %w", layer, err)
		}
	}
	return c, nil
}

// storeRow quantizes one token's k and v into row pos of cache's layer.
func (r *fp8KVRuntime) storeRow(stream *cuda.Stream, counter *cudalaunch.Counter, cache *cudakv.FP8Cache, layer int, k, v cudamem.Slice[float32], pos int) error {
	lc, err := cache.Layer(layer)
	if err != nil {
		return err
	}
	kvDim := r.kvHeads * r.headDim
	if k.Len() < kvDim || v.Len() < kvDim {
		return fmt.Errorf("fp8 kv: layer %d: k and v hold %d and %d elements, want %d", layer, k.Len(), v.Len(), kvDim)
	}
	if pos < 0 || pos >= r.maxSeqLen {
		return fmt.Errorf("fp8 kv: position %d outside the %d-token context window", pos, r.maxSeqLen)
	}
	grid := uint32((kvDim + fp8StoreThreads - 1) / fp8StoreThreads)
	return cudalaunch.OwnedKernel(r.store).Grid(grid).Block(fp8StoreThreads).On(stream).Count(counter).
		Launch(k, v, lc.K, lc.V, lc.KScale, lc.VScale, int32(pos), int32(r.kvHeads), int32(r.headDim))
}

// dequantize expands rows 0..seqLen-1 of cache's layer into the scratch.
func (r *fp8KVRuntime) dequantize(stream *cuda.Stream, counter *cudalaunch.Counter, cache *cudakv.FP8Cache, layer, seqLen int) error {
	lc, err := cache.Layer(layer)
	if err != nil {
		return err
	}
	if seqLen <= 0 || seqLen > r.maxSeqLen {
		return fmt.Errorf("fp8 kv: sequence length %d outside 1..%d", seqLen, r.maxSeqLen)
	}
	total := seqLen * r.kvHeads * r.headDim
	grid := uint32(min((total+fp8DequantThreads-1)/fp8DequantThreads, fp8DequantBlocks))
	return cudalaunch.OwnedKernel(r.dequant).Grid(grid).Block(fp8DequantThreads).On(stream).Count(counter).
		Launch(lc.K, lc.V, lc.KScale, lc.VScale, r.kScratch, r.vScratch, int32(seqLen), int32(r.kvHeads), int32(r.headDim))
}

// attend is one row's attention for one layer: RoPE on q and k, the FP8
// store of k and v at pos, the dequantize of positions 0..pos, and the dense
// attention kernel over the scratch into out.
func (r *fp8KVRuntime) attend(stream *cuda.Stream, counter *cudalaunch.Counter, cache *cudakv.FP8Cache, layer int,
	q, k, v, out cudamem.Slice[float32], pos int, ropeTheta float32) error {
	if cache == nil {
		return errors.New("fp8 kv: nil cache")
	}
	launcher := func(fn *cuda.Kernel) *cudalaunch.Launcher {
		return cudalaunch.OwnedKernel(fn).On(stream).Count(counter)
	}
	if err := launcher(r.rope).Grid(uint32(r.qHeads)).Block(uint32(r.headDim/2)).
		Launch(q, k, int32(pos), int32(r.qHeads), int32(r.kvHeads), int32(r.headDim), ropeTheta); err != nil {
		return fmt.Errorf("fp8 kv: rope at position %d: %w", pos, err)
	}
	if err := r.storeRow(stream, counter, cache, layer, k, v, pos); err != nil {
		return err
	}
	if err := r.dequantize(stream, counter, cache, layer, pos+1); err != nil {
		return err
	}
	shared, err := r.attn.sharedMem((pos + 1) * 4)
	if err != nil {
		return fmt.Errorf("fp8 kv: attention at position %d: %w", pos, err)
	}
	attnCfg := cudaattn.DefaultConfig(r.qHeads, r.kvHeads, r.headDim)
	return launcher(r.attn.fn).Grid(uint32(r.qHeads)).Block(1).SharedMem(shared).
		Launch(q, r.kScratch, r.vScratch, out, int32(pos), int32(r.maxSeqLen), int32(r.qHeads), int32(r.kvHeads), int32(r.headDim), attnCfg.Scale)
}

// Close unloads the module and frees the scratch. Caches allocated with
// newCache are the caller's to close.
func (r *fp8KVRuntime) Close() error {
	if r == nil {
		return nil
	}
	var errs []error
	if r.kScratch.Len() != 0 {
		errs = append(errs, r.kScratch.Free())
	}
	if r.vScratch.Len() != 0 {
		errs = append(errs, r.vScratch.Free())
	}
	if r.mod != nil {
		errs = append(errs, r.mod.Close())
		r.mod = nil
	}
	return errors.Join(errs...)
}

// fp8Step is the attention of one forward call on FP8 caches: the engine
// binds it with the rows' caches and positions before the call, and the
// forward body calls attend once per layer in place of the dense RoPE,
// StoreKV and attention launches, exactly as pagedStep does.
type fp8Step struct {
	rt        *fp8KVRuntime
	caches    []*cudakv.FP8Cache
	positions []int

	// armed is set by bind and cleared by the first layer of the next call,
	// so a forward without a fresh bind is an error rather than a replay.
	armed bool
}

func (s *fp8Step) bind(caches []*cudakv.FP8Cache, positions []int) error {
	if len(caches) == 0 || len(caches) != len(positions) {
		return fmt.Errorf("fp8 kv: %d caches for %d positions", len(caches), len(positions))
	}
	for i, c := range caches {
		if c == nil {
			return fmt.Errorf("fp8 kv: batch row %d has no cache", i)
		}
		if positions[i] < 0 || positions[i] >= s.rt.maxSeqLen {
			return fmt.Errorf("fp8 kv: batch row %d at position %d outside the %d-token context window", i, positions[i], s.rt.maxSeqLen)
		}
	}
	s.caches = append(s.caches[:0], caches...)
	s.positions = append(s.positions[:0], positions...)
	s.armed = true
	return nil
}

// attend runs layer for every bound row. q and out hold rows of
// qHeads*headDim, k and v rows of kvHeads*headDim.
func (s *fp8Step) attend(stream *cuda.Stream, counter *cudalaunch.Counter, layer int,
	q, k, v, out cudamem.Slice[float32], rows int, ropeTheta float32) error {
	if rows != len(s.caches) {
		return fmt.Errorf("fp8 kv: forward has %d rows, step bound %d", rows, len(s.caches))
	}
	if layer == 0 {
		if !s.armed {
			return errors.New("fp8 kv: forward call without a bound step")
		}
		s.armed = false
	}
	qDim := s.rt.qHeads * s.rt.headDim
	kvDim := s.rt.kvHeads * s.rt.headDim
	if q.Len() < rows*qDim || out.Len() < rows*qDim || k.Len() < rows*kvDim || v.Len() < rows*kvDim {
		return fmt.Errorf("fp8 kv: activations hold fewer than %d rows", rows)
	}
	for row := range rows {
		if err := s.rt.attend(stream, counter, s.caches[row], layer,
			q.Slice(row*qDim, (row+1)*qDim), k.Slice(row*kvDim, (row+1)*kvDim), v.Slice(row*kvDim, (row+1)*kvDim), out.Slice(row*qDim, (row+1)*qDim),
			s.positions[row], ropeTheta); err != nil {
			return fmt.Errorf("row %d: %w", row, err)
		}
	}
	return nil
}

// fp8CacheFor is the cache a forward call on skv reads: the sequence's own,
// or the engine's single-stream cache when skv is nil.
func (eng *ModelEngine) fp8CacheFor(skv *SequenceKVCache) *cudakv.FP8Cache {
	if skv != nil && skv.fp8 != nil {
		return skv.fp8
	}
	return eng.fp8Main
}

// bindFP8Single stages the engine scratchpad for one token of skv at pos. It
// does nothing unless the engine was compiled with LLAMA_INFER_FP8_KV=1.
func (eng *ModelEngine) bindFP8Single(skv *SequenceKVCache, pos int) error {
	if eng.fp8 == nil {
		return nil
	}
	if eng.scratch.fp8 == nil {
		eng.scratch.fp8 = &fp8Step{rt: eng.fp8}
	}
	return eng.scratch.fp8.bind([]*cudakv.FP8Cache{eng.fp8CacheFor(skv)}, []int{pos})
}

// bindFP8Batch stages sp for one row per cache at positions. It does nothing
// unless the engine was compiled with LLAMA_INFER_FP8_KV=1.
func (eng *ModelEngine) bindFP8Batch(sp *GPUBatchScratchpad, caches []*SequenceKVCache, positions []int) error {
	if eng.fp8 == nil {
		return nil
	}
	if sp == nil {
		return errors.New("fp8 kv: nil batch scratchpad")
	}
	fp8 := make([]*cudakv.FP8Cache, len(caches))
	for i, c := range caches {
		if c == nil || c.fp8 == nil {
			return fmt.Errorf("fp8 kv: batch row %d has no fp8 cache", i)
		}
		fp8[i] = c.fp8
	}
	if sp.fp8 == nil {
		sp.fp8 = &fp8Step{rt: eng.fp8}
	}
	return sp.fp8.bind(fp8, positions)
}
