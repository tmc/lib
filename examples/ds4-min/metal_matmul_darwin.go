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

// metalMatmulPreservedF16 performs batched matrix multiplication of F16 weights
// by F32 activations: Weight [K x N] F16, Input [K x T] F32 -> Output [N x T] F32.
//
// Crucially, it preserves the exact inner-loop order and SIMD reduction tree of
// kernel_mul_mv_t_t_4 across the K dimension, guaranteeing bit-exact identity
// with sequential single-token matvec calls.
type metalMatmulPreservedF16 struct {
	dev      *libmetal.Device
	library  apple.MTLLibrary
	pipeline apple.MTLComputePipelineState

	// ops centralizes command buffer creation and span recording, so this
	// kernel's buffers reach cbufs/tok like every other. Constructing from
	// the ops rather than the device is what keeps that from being optional.
	ops *metalPinnedOps
}

func newMetalMatmulPreservedF16(ops *metalPinnedOps) (_ *metalMatmulPreservedF16, err error) {
	if ops == nil || ops.dev == nil {
		return nil, fmt.Errorf("metal matmul: missing ops")
	}
	dev := ops.dev
	m := &metalMatmulPreservedF16{dev: dev, ops: ops}
	defer func() {
		if err != nil {
			m.Close()
		}
	}()
	m.library, err = dev.CompileLibrary(metalMatmulPreservedMSL)
	if err != nil {
		return nil, fmt.Errorf("metal matmul library: %w", err)
	}
	constants := apple.NewMTLFunctionConstantValues()
	defer objectivec.ObjectFromID(constants.GetID()).Release()
	nsg := int16(8)
	constants.SetConstantValueTypeAtIndex(unsafe.Pointer(&nsg), apple.MTLDataTypeShort, 600)
	fn, err := m.library.NewFunctionWithNameConstantValuesError("kernel_mul_mm_f16_f32_4", constants)
	if err != nil {
		return nil, fmt.Errorf("metal matmul function: %w", err)
	}
	defer objectivec.ObjectFromID(fn.GetID()).Release()
	m.pipeline, err = dev.Raw().NewComputePipelineStateWithFunctionError(fn)
	if err != nil {
		return nil, fmt.Errorf("metal matmul pipeline: %w", err)
	}
	return m, nil
}

func (m *metalMatmulPreservedF16) Close() {
	if m == nil {
		return
	}
	if m.pipeline != nil && m.pipeline.GetID() != 0 {
		objectivec.ObjectFromID(m.pipeline.GetID()).Release()
	}
	if m.library != nil && m.library.GetID() != 0 {
		objectivec.ObjectFromID(m.library.GetID()).Release()
	}
	m.pipeline = nil
	m.library = nil
}

// metalMatmulArgs mirrors ds4_metal_args_mul_mm.
type metalMatmulArgs struct {
	NE00, NE01, NE02       int32
	NB00, NB01, NB02, NB03 uint64
	NE10, NE11, NE12       int32
	NB10, NB11, NB12, NB13 uint64
	NE0, NE1, NR0          int32
	R2, R3                 int16
}

// Mul performs out [N x T] = weight [K x N] * in [K x T].
// weight: boundMetalTensor of F16 (typeID 1), dimensions [K, N]
// in: slice of float32, length K * T (column-major: token t is in[t*K : (t+1)*K])
// tokens: T
func (m *metalMatmulPreservedF16) Mul(t boundMetalTensor, inSlice []float32, tokens int) ([]float32, error) {
	if m == nil || m.pipeline == nil || m.ops == nil {
		return nil, fmt.Errorf("metal matmul: closed")
	}
	if t.typeID != 1 || len(t.dimensions) != 2 {
		return nil, fmt.Errorf("metal matmul %q: invalid weight tensor (must be 2D F16)", t.name)
	}
	k := int(t.dimensions[0])
	n := int(t.dimensions[1])
	if tokens <= 0 {
		return nil, fmt.Errorf("metal matmul: non-positive tokens %d", tokens)
	}
	if len(inSlice) != k*tokens {
		return nil, fmt.Errorf("metal matmul: input slice length %d != K(%d)*T(%d)", len(inSlice), k, tokens)
	}
	if t.data.Len() < k*n*2 {
		return nil, fmt.Errorf("metal matmul %q: weight buffer too small", t.name)
	}

	inBuf, err := metalmem.AllocFromSlice(m.dev, inSlice)
	if err != nil {
		return nil, err
	}
	defer inBuf.Free()

	outBuf, err := metalmem.Alloc[float32](m.dev, n*tokens)
	if err != nil {
		return nil, err
	}
	defer outBuf.Free()

	const nr0 = 2
	tgX := (n + nr0 - 1) / nr0
	tgY := tokens

	args := metalMatmulArgs{
		NE00: int32(k),
		NE01: int32(n),
		NE02: 1,
		NB00: 2,
		NB01: uint64(k * 2),
		NB02: uint64(k * n * 2),
		NB03: uint64(k * n * 2),

		NE10: int32(k),
		NE11: int32(tokens),
		NE12: 1,
		NB10: 4,
		NB11: uint64(k * 4),
		NB12: uint64(k * tokens * 4),
		NB13: uint64(k * tokens * 4),

		NE0: int32(n),
		NE1: int32(tokens),
		NR0: nr0,
		R2:  1,
		R3:  1,
	}

	cb := m.ops.commandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(m.pipeline)
	enc.SetBytesLengthAtIndex(unsafe.Slice((*byte)(unsafe.Pointer(&args)), unsafe.Sizeof(args)), 0)
	enc.SetBufferWithOffsetAtIndex(t.data.Buffer(), t.data.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(inBuf.Buffer(), inBuf.Offset(), 2)
	enc.SetBufferWithOffsetAtIndex(outBuf.Buffer(), outBuf.Offset(), 3)
	enc.SetThreadgroupMemoryLengthAtIndex(256, 0)
	enc.DispatchThreadgroupsThreadsPerThreadgroup(
		apple.MTLSize{Width: uint(tgX), Height: uint(tgY), Depth: 1},
		apple.MTLSize{Width: 256, Height: 1, Depth: 1},
	)
	enc.EndEncoding()
	cb.Commit()

	if err := libmetal.WaitCommandBuffer(context.Background(), cb); err != nil {
		return nil, fmt.Errorf("metal matmul %q: %w", t.name, err)
	}
	m.ops.record("batched-matmul", -1, 0, cb)
	return append([]float32(nil), outBuf.HostSlice()...), nil
}

// metalMatmulPreservedMSL preserves the exact reduction loop and order of kernel_mul_mv_t_t_4.
const metalMatmulPreservedMSL = `#include <metal_stdlib>
using namespace metal;
#define N_SIMDWIDTH 32
#define FC_MUL_MV 600
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
constant short FC_mul_mv_nsg   [[function_constant(FC_MUL_MV + 0)]];
constant short FC_mul_mv_nxpsg [[function_constant(FC_MUL_MV + 1)]];

struct ds4_metal_args_mul_mm {
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
static inline void helper_mm_reduce_and_write(
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
void kernel_mul_mm_t_t_4_impl(
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

    if (r1 >= args.ne11) {
        return;
    }

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

    helper_mm_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

template<typename T0, typename T04, typename T1, typename T14, typename args_t>
void kernel_mul_mm_t_t_4_disp(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    switch (args.nr0) {
        case 2: kernel_mul_mm_t_t_4_impl<T0, T04, T1, T14, 2, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
        case 4: kernel_mul_mm_t_t_4_impl<T0, T04, T1, T14, 4, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
    };
}

template<typename T0, typename T04, typename T1, typename T14>
kernel void kernel_mul_mm_t_t_4(
        constant ds4_metal_args_mul_mm & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mm_t_t_4_disp<T0, T04, T1, T14, constant ds4_metal_args_mul_mm &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

typedef decltype(kernel_mul_mm_t_t_4<half, half4, float, float4>) mul_mm_t_t_4;

template [[host_name("kernel_mul_mm_f16_f32_4")]] kernel mul_mm_t_t_4 kernel_mul_mm_t_t_4<half, half4, float, float4>;
`
