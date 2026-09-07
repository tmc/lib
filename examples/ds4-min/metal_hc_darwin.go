//go:build darwin && arm64

package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/apple/objectivec"
	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
	"github.com/tmc/lib/apple/x/metal/metalx"
)

// The pinned 24-row HC arithmetic uses a 1024-virtual-thread RMS and an
// eight-simdgroup F16 projection, then a 1024-thread weighted collapse/norm.
// See dsv4_hc.metal:1113-1214,462-608 and dense.metal:73-112 at b0a147a7.
// The four-row output head has a separate contract and does not use this path.
type metalHC struct {
	dev     *libmetal.Device
	library apple.MTLLibrary
	kernels struct {
		Mix   apple.MTLComputePipelineState `metal:"hc_parallel_mix"`
		Pre   apple.MTLComputePipelineState `metal:"hc_parallel_pre"`
		Split apple.MTLComputePipelineState `metal:"hc_parallel_split"`
		Post  apple.MTLComputePipelineState `metal:"hc_parallel_post"`
	}
	arena      *metalmem.Arena
	mix, split metalmem.Slice[float32]
	key        metalHCKey
	pending    *metalBatch
}

// Buffers and weights are immutable between a layer's pre/post operations.
// Cached splits are reused only within their producing command buffer. A
// cross-batch post recomputes its split in the same buffer as its expansion,
// so another submitted batch cannot overwrite an already-encoded dependency.
type metalHCKey struct {
	State, Fn, Scale, Base             uintptr
	StateOff, FnOff, ScaleOff, BaseOff uint
}

func metalHCIdentity(state metalmem.Slice[float32], fn, scale, base boundMetalTensor) metalHCKey {
	return metalHCKey{uintptr(state.Buffer().GetID()), uintptr(fn.data.Buffer().GetID()), uintptr(scale.data.Buffer().GetID()), uintptr(base.data.Buffer().GetID()), state.Offset(), fn.data.Offset(), scale.data.Offset(), base.data.Offset()}
}
func newMetalHC(dev *libmetal.Device) (h *metalHC, err error) {
	h = &metalHC{dev: dev}
	owned := h
	defer func() {
		if err != nil {
			owned.Close()
		}
	}()
	h.library, err = dev.CompileLibrary(metalHCMSL)
	if err != nil {
		return nil, err
	}
	if err = metalx.Bind(dev, h.library, &h.kernels); err != nil {
		return nil, err
	}
	h.arena = metalmem.NewArena(dev)
	h.mix = metalmem.AllocIn[float32](h.arena, "hc-mix", 24)
	h.split = metalmem.AllocIn[float32](h.arena, "hc-split", 24)
	if err = h.arena.Err(); err != nil {
		return nil, err
	}
	return h, nil
}
func (h *metalHC) Close() {
	if h == nil {
		return
	}
	h.pending = nil
	if h.arena != nil {
		h.arena.Free()
		h.arena = nil
	}
	for _, p := range []apple.MTLComputePipelineState{h.kernels.Mix, h.kernels.Pre, h.kernels.Split, h.kernels.Post} {
		if p != nil && p.GetID() != 0 {
			objectivec.ObjectFromID(p.GetID()).Release()
		}
	}
	if h.library != nil && h.library.GetID() != 0 {
		objectivec.ObjectFromID(h.library.GetID()).Release()
	}
	h.library = nil
	h.kernels.Mix, h.kernels.Pre, h.kernels.Split, h.kernels.Post = nil, nil, nil, nil
}
func (h *metalHC) complete(b *metalBatch) {
	if h.pending == b {
		h.pending = nil
	}
}
func metalHCProfile(state metalmem.Slice[float32], fn, scale, base boundMetalTensor) bool {
	return state.Len() == 16384 && fn.typeID == 1 && len(fn.dimensions) == 2 && fn.dimensions[0] == 16384 && fn.dimensions[1] == 24 && fn.data.Len() == 786432 && scale.typeID == 0 && len(scale.dimensions) == 1 && scale.dimensions[0] == 3 && scale.data.Len() == 12 && base.typeID == 0 && len(base.dimensions) == 1 && base.dimensions[0] == 24 && base.data.Len() == 96
}
func (h *metalHC) encodeMix(b *metalBatch, state metalmem.Slice[float32], fn boundMetalTensor) {
	enc := b.encoder()
	enc.SetComputePipelineState(h.kernels.Mix)
	args := make([]byte, 12)
	binary.LittleEndian.PutUint32(args, 16384)
	binary.LittleEndian.PutUint32(args[4:], 24)
	binary.LittleEndian.PutUint32(args[8:], math.Float32bits(1e-6))
	enc.SetBytesLengthAtIndex(args, 0)
	enc.SetBufferWithOffsetAtIndex(state.Buffer(), state.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(fn.data.Buffer(), fn.data.Offset(), 2)
	enc.SetBufferWithOffsetAtIndex(h.mix.Buffer(), h.mix.Offset(), 3)
	enc.SetThreadgroupMemoryLengthAtIndex(96*4, 0)
	enc.DispatchThreadgroupsThreadsPerThreadgroup(libmetal.Threads(12), libmetal.Threads(256))
}
func (h *metalHC) encodePre(b *metalBatch, state metalmem.Slice[float32], fn, scale, base, norm boundMetalTensor, out metalmem.Slice[float32]) error {
	h.pending = nil
	if !metalHCProfile(state, fn, scale, base) || out.Len() != 8192 || norm.typeID != 0 || len(norm.dimensions) != 1 || norm.dimensions[0] != 4096 || norm.data.Len() != 16384 {
		return fmt.Errorf("HC pre: invalid tensor profile")
	}
	h.encodeMix(b, state, fn)
	enc := b.encoder()
	enc.SetComputePipelineState(h.kernels.Pre)
	enc.SetBufferWithOffsetAtIndex(h.mix.Buffer(), h.mix.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(scale.data.Buffer(), scale.data.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(base.data.Buffer(), base.data.Offset(), 2)
	enc.SetBufferWithOffsetAtIndex(state.Buffer(), state.Offset(), 3)
	enc.SetBufferWithOffsetAtIndex(h.split.Buffer(), h.split.Offset(), 4)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 5)
	enc.SetBufferWithOffsetAtIndex(norm.data.Buffer(), norm.data.Offset(), 6)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset()+4096*4, 7)
	enc.SetThreadgroupMemoryLengthAtIndex((4096+4+32)*4, 0)
	enc.DispatchThreadgroupsThreadsPerThreadgroup(libmetal.Threads(1), libmetal.Threads(1024))
	h.key = metalHCIdentity(state, fn, scale, base)
	h.pending = b
	return nil
}
func (h *metalHC) encodePost(b *metalBatch, state, block metalmem.Slice[float32], fn, scale, base boundMetalTensor, out metalmem.Slice[float32]) error {
	if !metalHCProfile(state, fn, scale, base) || block.Len() != 4096 || out.Len() != 16384 {
		h.pending = nil
		return fmt.Errorf("HC post: invalid tensor profile")
	}
	hit := h.key == metalHCIdentity(state, fn, scale, base) && h.pending == b
	h.pending = nil
	if !hit {
		// Standalone callers get the same parallel projection/split arithmetic;
		// a cached split belonging to another residual or weight is never reused.
		h.encodeMix(b, state, fn)
		enc := b.encoder()
		enc.SetComputePipelineState(h.kernels.Split)
		enc.SetBufferWithOffsetAtIndex(h.mix.Buffer(), h.mix.Offset(), 0)
		enc.SetBufferWithOffsetAtIndex(scale.data.Buffer(), scale.data.Offset(), 1)
		enc.SetBufferWithOffsetAtIndex(base.data.Buffer(), base.data.Offset(), 2)
		enc.SetBufferWithOffsetAtIndex(h.split.Buffer(), h.split.Offset(), 3)
		enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(1), libmetal.Threads(1))
	}
	enc := b.encoder()
	enc.SetComputePipelineState(h.kernels.Post)
	for i, v := range []metalmem.Slice[float32]{state, block, h.split, out} {
		enc.SetBufferWithOffsetAtIndex(v.Buffer(), v.Offset(), uint(i))
	}
	enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(4096), libmetal.Threads(256))
	return nil
}

func (o *metalPinnedOps) HCPre(state []float32, fn, scale, base, norm boundMetalTensor) ([]float32, []float32, error) {
	in, err := metalmem.AllocFromSlice(o.dev, state)
	if err != nil {
		return nil, nil, err
	}
	defer in.Free()
	out, err := metalmem.Alloc[float32](o.dev, 8192)
	if err != nil {
		return nil, nil, err
	}
	defer out.Free()
	b, err := newMetalBatch(o)
	if err != nil {
		return nil, nil, err
	}
	if err = b.EncodeHCPre(in, fn, scale, base, norm, out); err != nil {
		return nil, nil, err
	}
	if err = b.Wait(context.Background()); err != nil {
		return nil, nil, err
	}
	// The producing batch completed, so a later post recomputes its split.
	v := out.HostSlice()
	return append([]float32(nil), v[:4096]...), append([]float32(nil), v[4096:]...), nil
}
func (o *metalPinnedOps) HCPost(state, block []float32, fn, scale, base boundMetalTensor) ([]float32, error) {
	in, err := metalmem.AllocFromSlice(o.dev, state)
	if err != nil {
		return nil, err
	}
	defer in.Free()
	blockIn, err := metalmem.AllocFromSlice(o.dev, block)
	if err != nil {
		return nil, err
	}
	defer blockIn.Free()
	out, err := metalmem.Alloc[float32](o.dev, 16384)
	if err != nil {
		return nil, err
	}
	defer out.Free()
	o.hc.pending = nil
	b, err := newMetalBatch(o)
	if err != nil {
		return nil, err
	}
	if err = b.EncodeHCPost(in, blockIn, fn, scale, base, out); err != nil {
		return nil, err
	}
	if err = b.Wait(context.Background()); err != nil {
		return nil, err
	}
	return append([]float32(nil), out.HostSlice()...), nil
}

const metalHCMSL = `
#include <metal_stdlib>
using namespace metal;
#define N_SIMDWIDTH 32
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
template<short NR0>
static inline void helper_mv_reduce_and_write(
        device float * dst_f32,
        float sumf[NR0],
        const int r0,
        const int ne01,
        ushort tiisg,
        ushort sgitg,
        threadgroup char * shmem) {
    constexpr short NW = N_SIMDWIDTH;

    threadgroup float * shmem_f32[NR0];

    for (short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *) shmem + NW*row;

        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }

        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0 && r0 + row < ne01; ++row) {
        float tot = simd_sum(shmem_f32[row][tiisg]);

        if (tiisg == 0 && sgitg == 0) {
            dst_f32[r0 + row] = tot;
        }
    }
}

struct ds4_metal_args_hc_norm_mix {
    int32_t n;
    int32_t out_dim;
    float   eps;
};

// Fused unweighted RMSNorm + F16 HC-mix projection for DS4 decode HC-pre.
// The standalone decode path runs kernel_rms_norm_f32_4 over the flattened
// 4*embd HC row (1024 threads, one threadgroup) and then
// kernel_mul_mv_f16_f32_4 (nsg=8, nr0=2) over the normalized row.  Both
// stages are reproduced bit-exactly in one dispatch: every threadgroup
// redundantly recomputes the norm partials with the original 1024-thread
// mapping (each real lane covers one virtual thread of each 256-thread
// slice, preserving every simd_sum tree), and the matvec keeps the original
// per-row accumulation order with y = x*scale computed on the fly, which
// rounds identically to the materialized normalized row.  The host wrapper
// gates this to n == 16384 && out_dim == 24, where the virtual-thread count
// is exactly 1024 and the mv tail loop is empty.
kernel void hc_parallel_mix(
        constant ds4_metal_args_hc_norm_mix & args [[buffer(0)]],
        device const char  * x [[buffer(1)]],
        device const char  * weight [[buffer(2)]],
        device       char  * dst [[buffer(3)]],
        threadgroup  char  * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NSG = 8;   // ds4_gpu_make_plain_mv_dispatch(16384)
    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NR0 = 2;   // plain mv nr0
    constexpr short NB  = 32;
    constexpr short NF  = 16;
    constexpr short NF4 = NF/4;
    constexpr uint  VTHREADS = 1024u;                 // rms norm threads at n == 16384
    constexpr short VSLICES  = VTHREADS/(NSG*NW);     // virtual 256-thread slices

    const uint n  = (uint)args.n;
    const uint n4 = n >> 2;

    device const float4 *x4 = (device const float4 *)x;

    threadgroup float *norm_shmem = (threadgroup float *)shmem;        // NW slots
    threadgroup float *mv_shmem   = (threadgroup float *)shmem + NW;   // NW*NR0 slots

    // Phase A: exact replica of kernel_rms_norm_f32_4's reduction tree with
    // the 1024 virtual threads folded onto this threadgroup's 8 simdgroups.
    for (short v = 0; v < VSLICES; ++v) {
        const uint vt = (uint)(sgitg + NSG*v)*NW + tiisg;
        float sumf = 0.0f;
        for (uint i00 = vt; i00 < n4; i00 += VTHREADS) {
            sumf += dot(x4[i00], x4[i00]);
        }
        sumf = simd_sum(sumf);
        if (tiisg == 0) {
            norm_shmem[sgitg + NSG*v] = sumf;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = norm_shmem[tiisg];
    total = simd_sum(total);
    const float mean  = total/(float)args.n;
    const float scale = 1.0f/sqrt(mean + args.eps);

    // Phase B: exact replica of kernel_mul_mv_f16_f32_4 (nsg=8, nr0=2) with
    // the normalized operand recomputed as x*scale instead of reloaded.
    const int nb = args.n/NB;
    const int r0 = tgpig.x*NR0;

    device const half4 * ax4[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        ax4[row] = (device const half4 *)
            (weight + (uint64_t)(r0 + row)*(uint64_t)n*sizeof(half));
    }

    float sumf_mv[NR0] = { 0.f };

    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);
    const int ib0 = sgitg*NF + ix;

    for (int ib = ib0; ib < nb; ib += NSG*NF) {
        float4 yl4[NF4];
        FOR_UNROLL (short i = 0; i < NF4; ++i) {
            yl4[i] = x4[(ib*NB + il*NF)/4 + i]*scale;
        }

        FOR_UNROLL (short row = 0; row < NR0; row++) {
            device const half4 * xb4 = ax4[row] + (ib*NB + il*NF)/4;

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NF4; ++i) {
                sumq += dot(float4(xb4[i]), yl4[i]);
            }

            sumf_mv[row] += sumq;
        }
    }

    // n == 16384 makes the scalar tail loop of the original empty.
    device float * dst_f32 = (device float *) dst;
    helper_mv_reduce_and_write<NR0>(dst_f32, sumf_mv, r0, args.out_dim,
                                    tiisg, sgitg, (threadgroup char *)mv_shmem);
}

kernel void hc_parallel_pre(
        device  const char  * mixes [[buffer(0)]],
        device  const float * scale [[buffer(1)]],
        device  const float * base [[buffer(2)]],
        device  const char  * x [[buffer(3)]],
        device        char  * split [[buffer(4)]],
        device        char  * dst [[buffer(5)]],
        device  const char  * norm_weight [[buffer(6)]],
        device        char  * norm_dst [[buffer(7)]],
        threadgroup   float * shared [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort ntg [[threads_per_threadgroup]]) {
    if ((int64_t)row >= 1 || 4 != 4 || (4096 & 3) != 0) {
        return;
    }

    const uint n_embd = uint(4096);
    const uint n4 = n_embd >> 2;
    threadgroup float4 *row_shmem = (threadgroup float4 *)shared;
    threadgroup float *pre_shmem = shared + n_embd;
    threadgroup float *sum_shmem = pre_shmem + 4;

    device const float *mix = (device const float *)(mixes + (uint64_t)row * 96);
    device float *out = (device float *)(split + (uint64_t)row * 96);

    if (sgitg == 0) {
        sum_shmem[tiisg] = 0.0f;
    }

    if (tid == 0) {
        const float epsv = 1.0e-6f;
        const float pre_scale = scale[0];
        const float post_scale = scale[1];
        const float comb_scale = scale[2];

        const float4 pre_z =
            *((device const float4 *)mix) * pre_scale +
            *((device const float4 *)base);
        const float4 pre = 1.0f / (1.0f + exp(-pre_z)) + epsv;
        *((device float4 *)out) = pre;
        pre_shmem[0] = pre.x;
        pre_shmem[1] = pre.y;
        pre_shmem[2] = pre.z;
        pre_shmem[3] = pre.w;

        const float4 post_z =
            *((device const float4 *)(mix + 4)) * post_scale +
            *((device const float4 *)(base + 4));
        *((device float4 *)(out + 4)) = 2.0f / (1.0f + exp(-post_z));

        float4 r0 =
            *((device const float4 *)(mix + 8)) * comb_scale +
            *((device const float4 *)(base + 8));
        float4 r1 =
            *((device const float4 *)(mix + 12)) * comb_scale +
            *((device const float4 *)(base + 12));
        float4 r2 =
            *((device const float4 *)(mix + 16)) * comb_scale +
            *((device const float4 *)(base + 16));
        float4 r3 =
            *((device const float4 *)(mix + 20)) * comb_scale +
            *((device const float4 *)(base + 20));

        const float m0 = max(max(r0.x, r0.y), max(r0.z, r0.w));
        const float m1 = max(max(r1.x, r1.y), max(r1.z, r1.w));
        const float m2 = max(max(r2.x, r2.y), max(r2.z, r2.w));
        const float m3 = max(max(r3.x, r3.y), max(r3.z, r3.w));

        r0 = exp(r0 - m0);
        r1 = exp(r1 - m1);
        r2 = exp(r2 - m2);
        r3 = exp(r3 - m3);

        r0 = r0 * (1.0f / (r0.x + r0.y + r0.z + r0.w)) + epsv;
        r1 = r1 * (1.0f / (r1.x + r1.y + r1.z + r1.w)) + epsv;
        r2 = r2 * (1.0f / (r2.x + r2.y + r2.z + r2.w)) + epsv;
        r3 = r3 * (1.0f / (r3.x + r3.y + r3.z + r3.w)) + epsv;

        float4 col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
        r0 *= col_inv;
        r1 *= col_inv;
        r2 *= col_inv;
        r3 *= col_inv;

        for (int iter = 1; iter < 20; ++iter) {
            r0 *= 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);
            r1 *= 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);
            r2 *= 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);
            r3 *= 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);

            col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
            r0 *= col_inv;
            r1 *= col_inv;
            r2 *= col_inv;
            r3 *= col_inv;
        }

        *((device float4 *)(out + 8)) = r0;
        *((device float4 *)(out + 12)) = r1;
        *((device float4 *)(out + 16)) = r2;
        *((device float4 *)(out + 20)) = r3;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sumf = 0.0f;
    for (uint i = tid; i < n4; i += ntg) {
        device const float4 *x0 = (device const float4 *)(x + 0 * 16384 + (uint64_t)row * 65536);
        device const float4 *x1 = (device const float4 *)(x + 1 * 16384 + (uint64_t)row * 65536);
        device const float4 *x2 = (device const float4 *)(x + 2 * 16384 + (uint64_t)row * 65536);
        device const float4 *x3 = (device const float4 *)(x + 3 * 16384 + (uint64_t)row * 65536);
        // Preserve the standalone HC collapse's explicit accumulation order.
        float4 v = 0.0f;
        v += x0[i] * pre_shmem[0];
        v += x1[i] * pre_shmem[1];
        v += x2[i] * pre_shmem[2];
        v += x3[i] * pre_shmem[3];
        row_shmem[i] = v;
        sumf += dot(v, v);
    }

    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        sum_shmem[sgitg] = sumf;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = sum_shmem[tiisg];
    sumf = simd_sum(sumf);
    // Batched prefill must match kernel_rms_norm_fuse_impl so enabling this
    // fusion does not change its scale by an ULP. Keep the established
    // single-row decode result unchanged; that path historically used rsqrt.
    const float norm_arg = sumf / float(n_embd) + 1.0e-6f;
    const float norm_scale = 1 > 1 ? 1.0f / sqrt(norm_arg) : rsqrt(norm_arg);

    device float4 *dst4 = (device float4 *)(dst + (uint64_t)row * 16384);
    device const float4 *w4 = (device const float4 *)norm_weight;
    device float4 *norm4 = (device float4 *)(norm_dst + (uint64_t)row * 16384);
    for (uint i = tid; i < n4; i += ntg) {
        const float4 v = row_shmem[i];
        dst4[i] = v;
        norm4[i] = (v * norm_scale) * w4[i];
    }
}

// Expands an embedding-sized block back into HC channels after attention/FFN.
// The post gate scales the current block, while the Sinkhorn combination matrix
// mixes residual HC channels from the previous state.
kernel void hc_parallel_post(device const float *state [[buffer(0)]], device const float *block [[buffer(1)]], device const float *split [[buffer(2)]], device float *out [[buffer(3)]], uint d [[thread_position_in_grid]]) {
 if(d>=4096)return;
 float r0=state[d],r1=state[4096+d],r2=state[8192+d],r3=state[12288+d];
 for(uint dst=0;dst<4;dst++) {
 float acc=block[d]*split[4+dst];
 acc+=split[8+dst]*r0;acc+=split[12+dst]*r1;acc+=split[16+dst]*r2;acc+=split[20+dst]*r3;
 out[dst*4096+d]=acc;
 }
}

kernel void hc_parallel_split(device const float *mix [[buffer(0)]], device const float *scale [[buffer(1)]], device const float *base [[buffer(2)]], device float *out [[buffer(3)]]) {
thread float pre_shmem[4];

        const float epsv = 1.0e-6f;
        const float pre_scale = scale[0];
        const float post_scale = scale[1];
        const float comb_scale = scale[2];

        const float4 pre_z =
            *((device const float4 *)mix) * pre_scale +
            *((device const float4 *)base);
        const float4 pre = 1.0f / (1.0f + exp(-pre_z)) + epsv;
        *((device float4 *)out) = pre;
        pre_shmem[0] = pre.x;
        pre_shmem[1] = pre.y;
        pre_shmem[2] = pre.z;
        pre_shmem[3] = pre.w;

        const float4 post_z =
            *((device const float4 *)(mix + 4)) * post_scale +
            *((device const float4 *)(base + 4));
        *((device float4 *)(out + 4)) = 2.0f / (1.0f + exp(-post_z));

        float4 r0 =
            *((device const float4 *)(mix + 8)) * comb_scale +
            *((device const float4 *)(base + 8));
        float4 r1 =
            *((device const float4 *)(mix + 12)) * comb_scale +
            *((device const float4 *)(base + 12));
        float4 r2 =
            *((device const float4 *)(mix + 16)) * comb_scale +
            *((device const float4 *)(base + 16));
        float4 r3 =
            *((device const float4 *)(mix + 20)) * comb_scale +
            *((device const float4 *)(base + 20));

        const float m0 = max(max(r0.x, r0.y), max(r0.z, r0.w));
        const float m1 = max(max(r1.x, r1.y), max(r1.z, r1.w));
        const float m2 = max(max(r2.x, r2.y), max(r2.z, r2.w));
        const float m3 = max(max(r3.x, r3.y), max(r3.z, r3.w));

        r0 = exp(r0 - m0);
        r1 = exp(r1 - m1);
        r2 = exp(r2 - m2);
        r3 = exp(r3 - m3);

        r0 = r0 * (1.0f / (r0.x + r0.y + r0.z + r0.w)) + epsv;
        r1 = r1 * (1.0f / (r1.x + r1.y + r1.z + r1.w)) + epsv;
        r2 = r2 * (1.0f / (r2.x + r2.y + r2.z + r2.w)) + epsv;
        r3 = r3 * (1.0f / (r3.x + r3.y + r3.z + r3.w)) + epsv;

        float4 col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
        r0 *= col_inv;
        r1 *= col_inv;
        r2 *= col_inv;
        r3 *= col_inv;

        for (int iter = 1; iter < 20; ++iter) {
            r0 *= 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);
            r1 *= 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);
            r2 *= 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);
            r3 *= 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);

            col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
            r0 *= col_inv;
            r1 *= col_inv;
            r2 *= col_inv;
            r3 *= col_inv;
        }

        *((device float4 *)(out + 8)) = r0;
        *((device float4 *)(out + 12)) = r1;
        *((device float4 *)(out + 16)) = r2;
        *((device float4 *)(out + 20)) = r3;
    
}
`
