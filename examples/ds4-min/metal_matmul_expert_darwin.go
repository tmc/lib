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

// metalMatmulPreservedExpert performs batched matrix multiplication for routed experts:
// 1. IQ2_XXS paired gate & up projections:
//    Weight [K x N x E] IQ2_XXS, Input [K x T] F32, ExpertIDs [S x T] -> GateOut [N x S x T], UpOut [N x S x T]
// 2. Q2_K down projection:
//    Weight [K x N x E] Q2_K, Input [K x S x T] F32, ExpertIDs [S x T] -> Out [N x S x T]
//
// Crucially, it preserves the exact accumulation order over blocks/parts and simd reductions
// of matvec_iq2_xxs_pair_selected_f32 and matvec_q2_k_selected_f32 across the K dimension,
// guaranteeing bit-exact identity with sequential single-token reference calls.
type metalMatmulPreservedExpert struct {
	dev         *libmetal.Device
	library     apple.MTLLibrary
	iq2Pip      apple.MTLComputePipelineState
	iq2Reversed apple.MTLComputePipelineState
	q2kPip      apple.MTLComputePipelineState
	q2kReversed apple.MTLComputePipelineState

	ops *metalPinnedOps
}

// specialize builds a pipeline with kReverseBlocks bound.
func (m *metalMatmulPreservedExpert) specialize(name string, reverse bool) (apple.MTLComputePipelineState, error) {
	constants := apple.NewMTLFunctionConstantValues()
	defer objectivec.ObjectFromID(constants.GetID()).Release()
	constants.SetConstantValueTypeAtIndex(unsafe.Pointer(&reverse), apple.MTLDataTypeBool, 0)
	fn, err := m.library.NewFunctionWithNameConstantValuesError(name, constants)
	if err != nil {
		return nil, err
	}
	defer objectivec.ObjectFromID(fn.GetID()).Release()
	return m.dev.Raw().NewComputePipelineStateWithFunctionError(fn)
}

func newMetalMatmulPreservedExpert(ops *metalPinnedOps) (_ *metalMatmulPreservedExpert, err error) {
	if ops == nil || ops.dev == nil {
		return nil, fmt.Errorf("metal matmul expert: missing ops")
	}
	dev := ops.dev
	m := &metalMatmulPreservedExpert{dev: dev, ops: ops}
	defer func() {
		if err != nil {
			m.Close()
		}
	}()
	m.library, err = dev.CompileLibrary(metalMatmulExpertPreservedMSL)
	if err != nil {
		return nil, fmt.Errorf("metal matmul expert library: %w", err)
	}
	m.iq2Pip, err = m.specialize("matmul_iq2_xxs_pair_selected_f32", false)
	if err != nil {
		return nil, fmt.Errorf("metal matmul expert iq2 pipeline: %w", err)
	}
	m.iq2Reversed, err = m.specialize("matmul_iq2_xxs_pair_selected_f32", true)
	if err != nil {
		return nil, fmt.Errorf("metal matmul expert iq2 reversed pipeline: %w", err)
	}
	m.q2kPip, err = m.specialize("matmul_q2_k_selected_f32", false)
	if err != nil {
		return nil, fmt.Errorf("metal matmul expert q2k pipeline: %w", err)
	}
	m.q2kReversed, err = m.specialize("matmul_q2_k_selected_f32", true)
	if err != nil {
		return nil, fmt.Errorf("metal matmul expert q2k reversed pipeline: %w", err)
	}
	return m, nil
}

func (m *metalMatmulPreservedExpert) Close() {
	if m == nil {
		return
	}
	if m.iq2Pip != nil && m.iq2Pip.GetID() != 0 {
		objectivec.ObjectFromID(m.iq2Pip.GetID()).Release()
	}
	if m.iq2Reversed != nil && m.iq2Reversed.GetID() != 0 {
		objectivec.ObjectFromID(m.iq2Reversed.GetID()).Release()
	}
	if m.q2kPip != nil && m.q2kPip.GetID() != 0 {
		objectivec.ObjectFromID(m.q2kPip.GetID()).Release()
	}
	if m.q2kReversed != nil && m.q2kReversed.GetID() != 0 {
		objectivec.ObjectFromID(m.q2kReversed.GetID()).Release()
	}
	if m.library != nil && m.library.GetID() != 0 {
		objectivec.ObjectFromID(m.library.GetID()).Release()
	}
	m.iq2Pip = nil
	m.iq2Reversed = nil
	m.q2kPip = nil
	m.q2kReversed = nil
	m.library = nil
}

// MulIQ2 performs batched selected IQ2_XXS paired gate & up projections:
// gate, up: 3D boundMetalTensor of IQ2_XXS (typeID 16), dimensions [K, N, E]
// expertIDs: slice of int32, length tokens * slots (token t, slot s at t*slots + s)
// inSlice: slice of float32, length K * tokens (token t at t*K)
// tokens: T
// slots: S (usually 6)
// Returns gateOut, upOut each of length tokens * slots * N (token t, slot s at (t*slots + s)*N).
func (m *metalMatmulPreservedExpert) MulIQ2(gate, up boundMetalTensor, expertIDs []int32, inSlice []float32, tokens, slots int) ([]float32, []float32, error) {
	return m.mulIQ2(gate, up, expertIDs, inSlice, tokens, slots, m.iq2Pip)
}

// MulIQ2Reversed runs the reassociated twin of MulIQ2. Tests only.
func (m *metalMatmulPreservedExpert) MulIQ2Reversed(gate, up boundMetalTensor, expertIDs []int32, inSlice []float32, tokens, slots int) ([]float32, []float32, error) {
	return m.mulIQ2(gate, up, expertIDs, inSlice, tokens, slots, m.iq2Reversed)
}

func (m *metalMatmulPreservedExpert) mulIQ2(gate, up boundMetalTensor, expertIDs []int32, inSlice []float32, tokens, slots int, pipeline apple.MTLComputePipelineState) ([]float32, []float32, error) {
	if m == nil || m.iq2Pip == nil || m.ops == nil {
		return nil, nil, fmt.Errorf("metal matmul expert: closed")
	}
	if gate.typeID != 16 || up.typeID != 16 || len(gate.dimensions) != 3 || len(up.dimensions) != 3 ||
		gate.dimensions[0] != up.dimensions[0] || gate.dimensions[1] != up.dimensions[1] || gate.dimensions[2] != up.dimensions[2] ||
		gate.dimensions[0]%256 != 0 {
		return nil, nil, fmt.Errorf("metal matmul expert IQ2_XXS: invalid tensor profile")
	}
	cols := int(gate.dimensions[0])
	rows := int(gate.dimensions[1])
	experts := int(gate.dimensions[2])
	if tokens <= 0 || slots <= 0 {
		return nil, nil, fmt.Errorf("metal matmul expert IQ2_XXS: non-positive tokens (%d) or slots (%d)", tokens, slots)
	}
	if len(expertIDs) != tokens*slots {
		return nil, nil, fmt.Errorf("metal matmul expert IQ2_XXS: expertIDs length %d != tokens(%d)*slots(%d)", len(expertIDs), tokens, slots)
	}
	if len(inSlice) != cols*tokens {
		return nil, nil, fmt.Errorf("metal matmul expert IQ2_XXS: input slice length %d != cols(%d)*tokens(%d)", len(inSlice), cols, tokens)
	}
	blocks := cols / 256
	expectedBytes := rows * blocks * 66 * experts
	if gate.data.Len() < expectedBytes || up.data.Len() < expectedBytes {
		return nil, nil, fmt.Errorf("metal matmul expert IQ2_XXS: weight buffer too small")
	}

	idsBuf, err := metalmem.AllocFromSlice(m.dev, expertIDs)
	if err != nil {
		return nil, nil, err
	}
	defer idsBuf.Free()

	inBuf, err := metalmem.AllocFromSlice(m.dev, inSlice)
	if err != nil {
		return nil, nil, err
	}
	defer inBuf.Free()

	outCount := rows * slots * tokens
	gateOutBuf, err := metalmem.Alloc[float32](m.dev, outCount)
	if err != nil {
		return nil, nil, err
	}
	defer gateOutBuf.Free()

	upOutBuf, err := metalmem.Alloc[float32](m.dev, outCount)
	if err != nil {
		return nil, nil, err
	}
	defer upOutBuf.Free()

	cb := m.ops.commandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(pipeline)
	enc.SetBufferWithOffsetAtIndex(gate.data.Buffer(), gate.data.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(up.data.Buffer(), up.data.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(idsBuf.Buffer(), idsBuf.Offset(), 2)
	enc.SetBufferWithOffsetAtIndex(inBuf.Buffer(), inBuf.Offset(), 3)
	enc.SetBufferWithOffsetAtIndex(gateOutBuf.Buffer(), gateOutBuf.Offset(), 4)
	enc.SetBufferWithOffsetAtIndex(upOutBuf.Buffer(), upOutBuf.Offset(), 5)
	libmetal.SetScalar(enc, uint32(cols), 6)
	libmetal.SetScalar(enc, uint32(rows), 7)
	libmetal.SetScalar(enc, uint32(experts), 8)
	libmetal.SetScalar(enc, uint32(slots), 9)
	libmetal.SetScalar(enc, uint32(tokens), 10)
	enc.DispatchThreadgroupsThreadsPerThreadgroup(
		apple.MTLSize{Width: uint(rows), Height: uint(slots), Depth: uint(tokens)},
		libmetal.Threads(32),
	)
	enc.EndEncoding()
	cb.Commit()

	if err := libmetal.WaitCommandBuffer(context.Background(), cb); err != nil {
		return nil, nil, fmt.Errorf("metal matmul expert IQ2_XXS: %w", err)
	}
	m.ops.record("batched-matmul-iq2-selected", -1, 0, cb)
	return append([]float32(nil), gateOutBuf.HostSlice()...), append([]float32(nil), upOutBuf.HostSlice()...), nil
}

// MulQ2K performs batched selected Q2_K down projections:
// down: 3D boundMetalTensor of Q2_K (typeID 10), dimensions [K, N, E]
// expertIDs: slice of int32, length tokens * slots (token t, slot s at t*slots + s)
// inSlice: slice of float32, length K * tokens * slots (token t, slot s at (t*slots + s)*K)
// tokens: T
// slots: S (usually 6)
// Returns out of length tokens * slots * N (token t, slot s at (t*slots + s)*N).
func (m *metalMatmulPreservedExpert) MulQ2K(down boundMetalTensor, expertIDs []int32, inSlice []float32, tokens, slots int) ([]float32, error) {
	return m.mulQ2K(down, expertIDs, inSlice, tokens, slots, m.q2kPip)
}

// MulQ2KReversed runs the reassociated twin of MulQ2K. Tests only.
func (m *metalMatmulPreservedExpert) MulQ2KReversed(down boundMetalTensor, expertIDs []int32, inSlice []float32, tokens, slots int) ([]float32, error) {
	return m.mulQ2K(down, expertIDs, inSlice, tokens, slots, m.q2kReversed)
}

func (m *metalMatmulPreservedExpert) mulQ2K(down boundMetalTensor, expertIDs []int32, inSlice []float32, tokens, slots int, pipeline apple.MTLComputePipelineState) ([]float32, error) {
	if m == nil || m.q2kPip == nil || m.ops == nil {
		return nil, fmt.Errorf("metal matmul expert: closed")
	}
	if down.typeID != 10 || len(down.dimensions) != 3 || down.dimensions[0]%256 != 0 {
		return nil, fmt.Errorf("metal matmul expert Q2_K: invalid tensor profile")
	}
	cols := int(down.dimensions[0])
	rows := int(down.dimensions[1])
	experts := int(down.dimensions[2])
	if tokens <= 0 || slots <= 0 {
		return nil, fmt.Errorf("metal matmul expert Q2_K: non-positive tokens (%d) or slots (%d)", tokens, slots)
	}
	if len(expertIDs) != tokens*slots {
		return nil, fmt.Errorf("metal matmul expert Q2_K: expertIDs length %d != tokens(%d)*slots(%d)", len(expertIDs), tokens, slots)
	}
	if len(inSlice) != cols*tokens*slots {
		return nil, fmt.Errorf("metal matmul expert Q2_K: input slice length %d != cols(%d)*tokens(%d)*slots(%d)", len(inSlice), cols, tokens, slots)
	}
	blocks := cols / 256
	expectedBytes := rows * blocks * 84 * experts
	if down.data.Len() < expectedBytes {
		return nil, fmt.Errorf("metal matmul expert Q2_K: weight buffer too small")
	}

	idsBuf, err := metalmem.AllocFromSlice(m.dev, expertIDs)
	if err != nil {
		return nil, err
	}
	defer idsBuf.Free()

	inBuf, err := metalmem.AllocFromSlice(m.dev, inSlice)
	if err != nil {
		return nil, err
	}
	defer inBuf.Free()

	outCount := rows * slots * tokens
	outBuf, err := metalmem.Alloc[float32](m.dev, outCount)
	if err != nil {
		return nil, err
	}
	defer outBuf.Free()

	cb := m.ops.commandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(pipeline)
	enc.SetBufferWithOffsetAtIndex(down.data.Buffer(), down.data.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(idsBuf.Buffer(), idsBuf.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(inBuf.Buffer(), inBuf.Offset(), 2)
	enc.SetBufferWithOffsetAtIndex(outBuf.Buffer(), outBuf.Offset(), 3)
	libmetal.SetScalar(enc, uint32(cols), 4)
	libmetal.SetScalar(enc, uint32(rows), 5)
	libmetal.SetScalar(enc, uint32(experts), 6)
	libmetal.SetScalar(enc, uint32(slots), 7)
	libmetal.SetScalar(enc, uint32(tokens), 8)
	enc.DispatchThreadsThreadsPerThreadgroup(
		apple.MTLSize{Width: uint(rows), Height: uint(slots), Depth: uint(tokens)},
		apple.MTLSize{Width: uint(min(rows, 256)), Height: 1, Depth: 1},
	)
	enc.EndEncoding()
	cb.Commit()

	if err := libmetal.WaitCommandBuffer(context.Background(), cb); err != nil {
		return nil, fmt.Errorf("metal matmul expert Q2_K: %w", err)
	}
	m.ops.record("batched-matmul-q2k-selected", -1, 0, cb)
	return append([]float32(nil), outBuf.HostSlice()...), nil
}

const metalMatmulExpertPreservedMSL = `#include <metal_stdlib>
using namespace metal;

constant bool kReverseBlocks [[function_constant(0)]];

// IQ2_XXS codebooks copied from clean ds4 b0a147a metal/moe.metal.
static constant uchar ds4_metal_ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

static constant ulong ds4_metal_iq2xxs_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

kernel void matmul_iq2_xxs_pair_selected_f32(
    device const uchar *gate [[buffer(0)]], device const uchar *up [[buffer(1)]],
    device const int *expert_ids [[buffer(2)]], device const float *x [[buffer(3)]],
    device float *gate_out [[buffer(4)]], device float *up_out [[buffer(5)]],
    constant uint &cols [[buffer(6)]], constant uint &rows [[buffer(7)]],
    constant uint &experts [[buffer(8)]], constant uint &slots [[buffer(9)]],
    constant uint &tokens [[buffer(10)]],
    uint3 group [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
  uint row = group.x, slot = group.y, tok = group.z;
  if (row >= rows || slot >= slots || tok >= tokens) return;
  uint global_slot = tok * slots + slot;
  int expert = expert_ids[global_slot];
  if (expert < 0 || uint(expert) >= experts) return;

  const uint blocks = cols / 256, parts = cols / 32;
  const uint64_t expert_stride = uint64_t(rows) * blocks * 66;
  const device uchar *selected_gate = gate + uint64_t(expert) * expert_stride;
  const device uchar *selected_up = up + uint64_t(expert) * expert_stride;
  device const float *tok_x = x + uint64_t(tok) * cols;

  float gate_sum = 0.0f, up_sum = 0.0f;
  for (uint parti = lane; parti < parts; parti += 32) {
    uint part = kReverseBlocks ? (parts - 1 - parti) : parti;
    float xv[32];
    for (ushort i = 0; i < 32; i++) xv[i] = tok_x[part * 32 + i];
    uint block = part / 8, sub = part % 8;
    const device uchar *pg = selected_gate + uint64_t(row) * blocks * 66 + uint64_t(block) * 66;
    const device uchar *pu = selected_up + uint64_t(row) * blocks * 66 + uint64_t(block) * 66;
    const device ushort *qg = reinterpret_cast<device const ushort *>(pg + 2) + 4 * sub;
    const device ushort *qu = reinterpret_cast<device const ushort *>(pu + 2) + 4 * sub;
    device const uchar *ag = reinterpret_cast<device const uchar *>(qg);
    device const uchar *au = reinterpret_cast<device const uchar *>(qu);
    uint sg = uint(qg[2]) | (uint(qg[3]) << 16), su = uint(qu[2]) | (uint(qu[3]) << 16);
    float vg = 0.0f, vu = 0.0f;
    for (ushort l = 0; l < 4; l++) {
      constant const uchar *gg = reinterpret_cast<constant const uchar *>(ds4_metal_iq2xxs_grid + ag[l]);
      constant const uchar *gu = reinterpret_cast<constant const uchar *>(ds4_metal_iq2xxs_grid + au[l]);
      uchar signg = ds4_metal_ksigns_iq2xs[(sg >> (7 * l)) & 127], signu = ds4_metal_ksigns_iq2xs[(su >> (7 * l)) & 127];
      for (ushort j = 0; j < 8; j++) {
        float v = xv[8 * l + j];
        uchar mask = uchar(1u << j);
        vg += v * float(gg[j]) * ((signg & mask) ? -1.0f : 1.0f);
        vu += v * float(gu[j]) * ((signu & mask) ? -1.0f : 1.0f);
      }
    }
    gate_sum += float(*reinterpret_cast<device const half *>(pg)) * (0.5f + float(sg >> 28)) * vg;
    up_sum += float(*reinterpret_cast<device const half *>(pu)) * (0.5f + float(su >> 28)) * vu;
  }
  gate_sum = simd_sum(gate_sum) * 0.25f;
  up_sum = simd_sum(up_sum) * 0.25f;
  if (lane == 0) {
    uint64_t i = uint64_t(global_slot) * rows + row;
    gate_out[i] = gate_sum;
    up_out[i] = up_sum;
  }
}

kernel void matmul_q2_k_selected_f32(
    device const uchar *weights [[buffer(0)]], device const int *expert_ids [[buffer(1)]],
    device const float *x [[buffer(2)]], device float *out [[buffer(3)]],
    constant uint &cols [[buffer(4)]], constant uint &rows [[buffer(5)]],
    constant uint &experts [[buffer(6)]], constant uint &slots [[buffer(7)]],
    constant uint &tokens [[buffer(8)]],
    uint3 pos [[thread_position_in_grid]]) {
  uint row = pos.x, slot = pos.y, tok = pos.z;
  if (row >= rows || slot >= slots || tok >= tokens) return;
  uint global_slot = tok * slots + slot;
  int expert = expert_ids[global_slot];
  if (expert < 0 || uint(expert) >= experts) return;

  uint nb = cols / 256;
  uint64_t expert_stride = uint64_t(rows) * nb * 84;
  device const uchar *selected = weights + uint64_t(expert) * expert_stride;
  device const float *selected_x = x + uint64_t(global_slot) * cols;
  float total = 0.0f;
  for (uint bi = 0; bi < nb; bi++) {
    uint block = kReverseBlocks ? (nb - 1 - bi) : bi;
    const device uchar *p = selected + uint64_t(row) * nb * 84 + uint64_t(block) * 84;
    float d = float(*reinterpret_cast<device const half *>(p + 80));
    float dm = float(*reinterpret_cast<device const half *>(p + 82));
    for (uint i = 0; i < 256; i++) {
      uint group = i / 16, l = i - group * 16;
      uint qbase = 32 * (group / 8) + 16 * (group & 1);
      uint shift = ((group / 2) & 3) * 2;
      uchar sm = p[group];
      float w = d * float(sm & 15) * float((p[16 + qbase + l] >> shift) & 3) - dm * float(sm >> 4);
      total += w * selected_x[block * 256 + i];
    }
  }
  out[uint64_t(global_slot) * rows + row] = total;
}
`
