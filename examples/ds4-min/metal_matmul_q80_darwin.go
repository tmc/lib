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

// metalMatmulPreservedQ80 performs batched matrix multiplication of Q8_0 weights
// by F32 activations: Weight [K x N] Q8_0, Input [K x T] F32 -> Output [N x T] F32.
//
// Crucially, it preserves the exact accumulation order over blocks and simd reductions
// of matvec_q8_0 across the K dimension, guaranteeing bit-exact identity
// with sequential single-token MatvecQ80 calls.
type metalMatmulPreservedQ80 struct {
	dev       *libmetal.Device
	library   apple.MTLLibrary
	matmulPip apple.MTLComputePipelineState

	// reversedPip runs the same kernel with the block accumulation reversed.
	// Nothing in the runtime uses it; it is the order-sensitivity guard's
	// twin, so the guard costs a pipeline rather than a source mutation.
	reversedPip apple.MTLComputePipelineState

	ops *metalPinnedOps
}

// specialize builds the matmul pipeline with kReverseBlocks bound.
func (m *metalMatmulPreservedQ80) specialize(reverse bool) (apple.MTLComputePipelineState, error) {
	constants := apple.NewMTLFunctionConstantValues()
	defer objectivec.ObjectFromID(constants.GetID()).Release()
	constants.SetConstantValueTypeAtIndex(unsafe.Pointer(&reverse), apple.MTLDataTypeBool, 0)
	fn, err := m.library.NewFunctionWithNameConstantValuesError("matmul_q8_0", constants)
	if err != nil {
		return nil, err
	}
	defer objectivec.ObjectFromID(fn.GetID()).Release()
	return m.dev.Raw().NewComputePipelineStateWithFunctionError(fn)
}

func newMetalMatmulPreservedQ80(ops *metalPinnedOps) (_ *metalMatmulPreservedQ80, err error) {
	if ops == nil || ops.dev == nil {
		return nil, fmt.Errorf("metal matmul q8_0: missing ops")
	}
	dev := ops.dev
	m := &metalMatmulPreservedQ80{dev: dev, ops: ops}
	defer func() {
		if err != nil {
			m.Close()
		}
	}()
	m.library, err = dev.CompileLibrary(metalMatmulQ80PreservedMSL)
	if err != nil {
		return nil, fmt.Errorf("metal matmul q8_0 library: %w", err)
	}
	m.matmulPip, err = m.specialize(false)
	if err != nil {
		return nil, fmt.Errorf("metal matmul q8_0 matmul pipeline: %w", err)
	}
	m.reversedPip, err = m.specialize(true)
	if err != nil {
		return nil, fmt.Errorf("metal matmul q8_0 reversed pipeline: %w", err)
	}
	return m, nil
}

func (m *metalMatmulPreservedQ80) Close() {
	if m == nil {
		return
	}
	if m.matmulPip != nil && m.matmulPip.GetID() != 0 {
		objectivec.ObjectFromID(m.matmulPip.GetID()).Release()
	}
	if m.reversedPip != nil && m.reversedPip.GetID() != 0 {
		objectivec.ObjectFromID(m.reversedPip.GetID()).Release()
	}
	if m.library != nil && m.library.GetID() != 0 {
		objectivec.ObjectFromID(m.library.GetID()).Release()
	}
	m.matmulPip = nil
	m.reversedPip = nil
	m.library = nil
}

// Mul performs out [N x T] = weight [K x N] * in [K x T].
// weight: boundMetalTensor of Q8_0 (typeID 8), dimensions [K, N]
// inSlice: slice of float32, length K * T (column-major: token t is inSlice[t*K : (t+1)*K])
// tokens: T
func (m *metalMatmulPreservedQ80) Mul(t boundMetalTensor, inSlice []float32, tokens int) ([]float32, error) {
	return m.mul(t, inSlice, tokens, m.matmulPip)
}

// mulReversed runs the reassociated twin. Tests only.
func (m *metalMatmulPreservedQ80) mulReversed(t boundMetalTensor, inSlice []float32, tokens int) ([]float32, error) {
	return m.mul(t, inSlice, tokens, m.reversedPip)
}

func (m *metalMatmulPreservedQ80) mul(t boundMetalTensor, inSlice []float32, tokens int, pipeline apple.MTLComputePipelineState) ([]float32, error) {
	if m == nil || m.matmulPip == nil || m.ops == nil {
		return nil, fmt.Errorf("metal matmul q8_0: closed")
	}
	if t.typeID != 8 || len(t.dimensions) != 2 || t.dimensions[0]%32 != 0 {
		return nil, fmt.Errorf("metal matmul q8_0 %q: invalid weight tensor (must be 2D Q8_0 with K multiple of 32)", t.name)
	}
	k := int(t.dimensions[0])
	n := int(t.dimensions[1])
	if tokens <= 0 {
		return nil, fmt.Errorf("metal matmul q8_0: non-positive tokens %d", tokens)
	}
	if len(inSlice) != k*tokens {
		return nil, fmt.Errorf("metal matmul q8_0: input slice length %d != K(%d)*T(%d)", len(inSlice), k, tokens)
	}
	blocks := k / 32
	if t.data.Len() < n*blocks*34 {
		return nil, fmt.Errorf("metal matmul q8_0 %q: weight buffer too small", t.name)
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

	cb := m.ops.commandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(pipeline)
	enc.SetBufferWithOffsetAtIndex(t.data.Buffer(), t.data.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(inBuf.Buffer(), inBuf.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(outBuf.Buffer(), outBuf.Offset(), 2)
	libmetal.SetScalar(enc, uint32(k), 3)
	libmetal.SetScalar(enc, uint32(n), 4)
	libmetal.SetScalar(enc, uint32(tokens), 5)
	enc.SetThreadgroupMemoryLengthAtIndex(2*32*4, 0)
	enc.DispatchThreadgroupsThreadsPerThreadgroup(
		apple.MTLSize{Width: uint((n + 1) / 2), Height: uint(tokens), Depth: 1},
		apple.MTLSize{Width: 32, Height: 4, Depth: 1},
	)
	enc.EndEncoding()
	cb.Commit()

	if err := libmetal.WaitCommandBuffer(context.Background(), cb); err != nil {
		return nil, fmt.Errorf("metal matmul q8_0 %q: %w", t.name, err)
	}
	m.ops.record("batched-matmul-q80", -1, 0, cb)
	return append([]float32(nil), outBuf.HostSlice()...), nil
}

const metalMatmulQ80PreservedMSL = `#include <metal_stdlib>
using namespace metal;

// kReverseBlocks exists only so a test can build a deliberately reassociated
// twin of this kernel and prove its own inputs can tell the two apart. It is
// false in every pipeline the runtime creates; specialization removes the
// select.
constant bool kReverseBlocks [[function_constant(0)]];

kernel void matmul_q8_0(
    device const uchar *weights [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &cols [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &tokens [[buffer(5)]],
    threadgroup float *scratch [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
  uint tok = group.y;
  if (tok >= tokens) return;

  constexpr uint rows_per_group = 2;
  constexpr uint simdgroups = 4;
  const uint blocks = cols / 32;
  const uint first_row = group.x * rows_per_group;
  const ushort block_part = lane / 4;
  const ushort element_part = lane % 4;
  const uint first_block = simdgroup * 8 + block_part;
  device const float *tok_x = x + uint64_t(tok) * cols;
  device float *tok_out = out + uint64_t(tok) * rows;

  float sums[rows_per_group] = {0.0f};
  for (uint bi = first_block; bi < blocks; bi += simdgroups * 8) {
    uint b = kReverseBlocks ? (blocks - 1 - bi) : bi;
    float xv[8];
    for (ushort i = 0; i < 8; i++) xv[i] = tok_x[b * 32 + element_part * 8 + i];
    for (ushort r = 0; r < rows_per_group; r++) {
      if (first_row + r >= rows) continue;
      const device uchar *p = weights + uint64_t(first_row + r) * blocks * 34 + uint64_t(b) * 34;
      float dot = 0.0f;
      for (ushort i = 0; i < 8; i++) dot += float(as_type<char>(p[2 + element_part * 8 + i])) * xv[i];
      sums[r] += dot * float(*reinterpret_cast<device const half *>(p));
    }
  }
  for (ushort r = 0; r < rows_per_group; r++) {
    if (simdgroup == 0) scratch[r * 32 + lane] = 0.0f;
    sums[r] = simd_sum(sums[r]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (ushort r = 0; r < rows_per_group; r++) if (lane == 0) scratch[r * 32 + simdgroup] = sums[r];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (ushort r = 0; r < rows_per_group; r++) {
    float total = simd_sum(scratch[r * 32 + lane]);
    if (lane == 0 && simdgroup == 0 && first_row + r < rows) tok_out[first_row + r] = total;
  }
}
`
