//go:build darwin && arm64

package main

import (
	"context"
	"fmt"
	"unsafe"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/apple/objectivec"
	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// metalHeadProjection preserves the standalone output-head F16 reduction at
// antirez/ds4 b0a147a: ds4_metal.m:5311-5332,5393-5402 and metal/dense.metal.
// Four rows use two threadgroups, each with eight SIMD groups and two rows.
// The normalized activation remains float32; only weights are binary16.
type metalHeadProjection struct {
	dev      *libmetal.Device
	library  apple.MTLLibrary
	pipeline apple.MTLComputePipelineState
}

func newMetalHeadProjection(dev *libmetal.Device) (_ *metalHeadProjection, err error) {
	h := &metalHeadProjection{dev: dev}
	defer func() {
		if err != nil {
			h.Close()
		}
	}()
	h.library, err = dev.CompileLibrary(metalHeadProjectionMSL)
	if err != nil {
		return nil, fmt.Errorf("output head library: %w", err)
	}
	constants := apple.NewMTLFunctionConstantValues()
	defer objectivec.ObjectFromID(constants.GetID()).Release()
	nsg := int16(8)
	constants.SetConstantValueTypeAtIndex(unsafe.Pointer(&nsg), apple.MTLDataTypeShort, 600)
	fn, err := h.library.NewFunctionWithNameConstantValuesError("kernel_mul_mv_f16_f32_4", constants)
	if err != nil {
		return nil, fmt.Errorf("output head function: %w", err)
	}
	defer objectivec.ObjectFromID(fn.GetID()).Release()
	h.pipeline, err = dev.Raw().NewComputePipelineStateWithFunctionError(fn)
	if err != nil {
		return nil, fmt.Errorf("output head pipeline: %w", err)
	}
	return h, nil
}

func (h *metalHeadProjection) Close() {
	if h == nil {
		return
	}
	if h.pipeline != nil && h.pipeline.GetID() != 0 {
		objectivec.ObjectFromID(h.pipeline.GetID()).Release()
	}
	if h.library != nil && h.library.GetID() != 0 {
		objectivec.ObjectFromID(h.library.GetID()).Release()
	}
	h.pipeline = nil
	h.library = nil
}

// Mirrors ds4_metal_args_mul_mv: 112 bytes, with nb00 at16, nb10 at64,
// ne0 at96 and nr0 at104. These offsets are checked against the pinned C ABI
// by the independently compiled output-head differential fixture.
type metalHeadProjectionArgs struct {
	NE00, NE01, NE02       int32
	NB00, NB01, NB02, NB03 uint64
	NE10, NE11, NE12       int32
	NB10, NB11, NB12, NB13 uint64
	NE0, NE1, NR0          int32
	R2, R3                 int16
}

func (h *metalHeadProjection) matvec(t boundMetalTensor, x []float32) ([]float32, error) {
	if h == nil || h.pipeline == nil {
		return nil, fmt.Errorf("output head projection: closed")
	}
	if t.typeID != 1 || len(t.dimensions) != 2 || t.dimensions[0] != 16384 || t.dimensions[1] != 4 || len(x) != 16384 || t.data.Len() < 4*16384*2 {
		return nil, fmt.Errorf("output head projection %q: invalid shape or weight extent", t.name)
	}
	in, err := metalmem.AllocFromSlice(h.dev, x)
	if err != nil {
		return nil, err
	}
	defer in.Free()
	out, err := metalmem.Alloc[float32](h.dev, 4)
	if err != nil {
		return nil, err
	}
	defer out.Free()
	args := metalHeadProjectionArgs{NE00: 16384, NE01: 4, NE02: 1, NB00: 2, NB01: 32768, NB02: 131072, NB03: 131072, NE10: 16384, NE11: 1, NE12: 1, NB10: 4, NB11: 65536, NB12: 65536, NB13: 65536, NE0: 4, NE1: 1, NR0: 2, R2: 1, R3: 1}
	cb := h.dev.Queue().CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(h.pipeline)
	enc.SetBytesLengthAtIndex(unsafe.Slice((*byte)(unsafe.Pointer(&args)), unsafe.Sizeof(args)), 0)
	enc.SetBufferWithOffsetAtIndex(t.data.Buffer(), t.data.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(in.Buffer(), in.Offset(), 2)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 3)
	enc.SetThreadgroupMemoryLengthAtIndex(256, 0)
	enc.DispatchThreadgroupsThreadsPerThreadgroup(libmetal.Threads(2), libmetal.Threads(256))
	enc.EndEncoding()
	cb.Commit()
	if err := libmetal.WaitCommandBuffer(context.Background(), cb); err != nil {
		return nil, fmt.Errorf("output head projection %q: %w", t.name, err)
	}
	return append([]float32(nil), out.HostSlice()...), nil
}

// Pinned vector-matvec source, including its original SIMD reduction helper.
const metalHeadProjectionMSL = `#include <metal_stdlib>
using namespace metal;
#define N_SIMDWIDTH 32
#define FC_MUL_MV 600
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
constant short FC_mul_mv_nsg   [[function_constant(FC_MUL_MV + 0)]];
constant short FC_mul_mv_nxpsg [[function_constant(FC_MUL_MV + 1)]];

struct ds4_metal_args_mul_mv {
    int ne00;
    int ne01;
    int ne02;
    ulong nb00;
    ulong nb01;
    ulong nb02;
    ulong nb03;
    int ne10;
    int ne11;
    int ne12;
    ulong nb10;
    ulong nb11;
    ulong nb12;
    ulong nb13;
    int ne0;
    int ne1;
    int nr0;
    short r2;
    short r3;
};

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

template<typename T0, typename T04, typename T1, typename T14, short NR0, typename args_t>
void kernel_mul_mv_t_t_4_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NB  = 32;
    constexpr short NF  = 16;
    constexpr short NF4 = NF/4;

    const int nb = args.ne00/NB;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    device const T1  * y  = (device const T1  *) (src1 + offset1);
    device const T14 * y4 = (device const T14 *) (src1 + offset1);

    device const T0  * ax [NR0];
    device const T04 * ax4[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

        ax [row] = (device const T0  *) ((device char *) src0 + offset0);
        ax4[row] = (device const T04 *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);

    const int ib0 = sgitg*NF + ix;

    T14 yl4[NF4];

    device const T14 * yb4 = y4 + (ib0*NB + il*NF)/4;

    for (int ib = ib0; ib < nb; ib += NSG*NF) {
        for (short i = 0; i < NF4; ++i) {
            yl4[i] = yb4[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const T04 * xb4 = ax4[row] + (ib*NB + il*NF)/4;

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NF4; ++i) {
                sumq += dot(float4(xb4[i]), float4(yl4[i]));
            }

            sumf[row] += sumq;
        }

        yb4 += NSG*NF*NW/4;
    }

    for (int i = nb*NB + sgitg*NW + tiisg; i < args.ne00; i += NW*NSG) {
        for (short row = 0; row < NR0; row++) {
            sumf[row] += ax[row][i] * y[i];
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

template<typename T0, typename T04, typename T1, typename T14, typename args_t>
void kernel_mul_mv_t_t_4_disp(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    switch (args.nr0) {
        case 2: kernel_mul_mv_t_t_4_impl<T0, T04, T1, T14, 2, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
        case 4: kernel_mul_mv_t_t_4_impl<T0, T04, T1, T14, 4, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
    };
}

// Vectorized dense matvec using float4/half4 loads. DS4 uses this where the
// inner dimension and alignment make vector loads cheaper than scalar lanes.
template<typename T0, typename T04, typename T1, typename T14>
kernel void kernel_mul_mv_t_t_4(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_t_t_4_disp<T0, T04, T1, T14, constant ds4_metal_args_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

typedef decltype(kernel_mul_mv_t_t_4<half, half4, half, half4>) mul_mv_t_t_4;

// Host-visible vectorized dense matvec variants for F32 and F16 weights.
template [[host_name("kernel_mul_mv_f32_f32_4")]] kernel mul_mv_t_t_4 kernel_mul_mv_t_t_4<float, float4, float, float4>;
template [[host_name("kernel_mul_mv_f16_f32_4")]] kernel mul_mv_t_t_4 kernel_mul_mv_t_t_4<half,  half4,  float, float4>;

`
