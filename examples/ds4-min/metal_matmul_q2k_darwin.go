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

// metalMatmulPreservedQ2K performs batched matrix multiplication of Q2_K weights
// by F32 activations (quantized to Q8_K per token column):
// Weight [K x N] Q2_K, Input [K x T] F32 -> Output [N x T] F32.
//
// Crucially, it preserves the exact accumulation order over blocks and groups
// of matvec_q2_k across the K dimension, guaranteeing bit-exact identity
// with sequential single-token MatvecQ2K calls.
type metalMatmulPreservedQ2K struct {
	dev       *libmetal.Device
	library   apple.MTLLibrary
	quantPip  apple.MTLComputePipelineState
	matmulPip apple.MTLComputePipelineState

	// reversedPip runs the same kernel with the block accumulation reversed.
	// Nothing in the runtime uses it; it is the order-sensitivity guard's
	// twin, so the guard costs a pipeline rather than a source mutation.
	reversedPip apple.MTLComputePipelineState

	ops *metalPinnedOps
}

// specialize builds the matmul pipeline with kReverseBlocks bound.
func (m *metalMatmulPreservedQ2K) specialize(reverse bool) (apple.MTLComputePipelineState, error) {
	constants := apple.NewMTLFunctionConstantValues()
	defer objectivec.ObjectFromID(constants.GetID()).Release()
	constants.SetConstantValueTypeAtIndex(unsafe.Pointer(&reverse), apple.MTLDataTypeBool, 0)
	fn, err := m.library.NewFunctionWithNameConstantValuesError("matmul_q2_k", constants)
	if err != nil {
		return nil, err
	}
	defer objectivec.ObjectFromID(fn.GetID()).Release()
	return m.dev.Raw().NewComputePipelineStateWithFunctionError(fn)
}

func newMetalMatmulPreservedQ2K(ops *metalPinnedOps) (_ *metalMatmulPreservedQ2K, err error) {
	if ops == nil || ops.dev == nil {
		return nil, fmt.Errorf("metal matmul q2_k: missing ops")
	}
	if ops.matmulQ2K != nil {
		return ops.matmulQ2K, nil
	}
	dev := ops.dev
	m := &metalMatmulPreservedQ2K{dev: dev, ops: ops}
	defer func() {
		if err != nil {
			m.Close()
		}
	}()
	m.library, err = dev.CompileLibrary(metalMatmulQ2KPreservedMSL)
	if err != nil {
		return nil, fmt.Errorf("metal matmul q2_k library: %w", err)
	}
	m.quantPip, err = dev.Raw().NewComputePipelineStateWithFunctionError(
		m.library.NewFunctionWithName("quant_q8_k_batched"),
	)
	if err != nil {
		return nil, fmt.Errorf("metal matmul q2_k quant pipeline: %w", err)
	}
	m.matmulPip, err = m.specialize(false)
	if err != nil {
		return nil, fmt.Errorf("metal matmul q2_k matmul pipeline: %w", err)
	}
	m.reversedPip, err = m.specialize(true)
	if err != nil {
		return nil, fmt.Errorf("metal matmul q2_k reversed pipeline: %w", err)
	}
	ops.matmulQ2K = m
	return m, nil
}

func (m *metalMatmulPreservedQ2K) Close() {
	if m == nil {
		return
	}
	if m.ops != nil && m.ops.matmulQ2K == m {
		m.ops.matmulQ2K = nil
	}
	if m.quantPip != nil && m.quantPip.GetID() != 0 {
		objectivec.ObjectFromID(m.quantPip.GetID()).Release()
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
	m.quantPip = nil
	m.matmulPip = nil
	m.reversedPip = nil
	m.library = nil
}

// EncodeMatmulQ2K appends a Q2_K batched matrix multiplication using caller-owned storage.
// It quantizes `in` to Q8_K scratch storage and computes out [N x T] = weight [K x N] * q8Scratch.
func (b *metalBatch) EncodeMatmulQ2K(t boundMetalTensor, in, out metalmem.Slice[float32], q8Scratch metalmem.Slice[byte], tokens int) error {
	if err := b.ready(); err != nil {
		return err
	}
	mm, err := b.ops.ensureMatmulQ2K()
	if err != nil {
		return err
	}
	return b.encodeMatmulQ2K(t, in, out, q8Scratch, tokens, mm.matmulPip)
}

func checkMatmulQ2KProfile(t boundMetalTensor, inLen, tokens int) (k, n, blocks int, err error) {
	if t.typeID != 10 || len(t.dimensions) != 2 || t.dimensions[0]%256 != 0 {
		return 0, 0, 0, fmt.Errorf("metal matmul q2_k %q: invalid weight tensor (must be 2D Q2_K with K multiple of 256)", t.name)
	}
	k = int(t.dimensions[0])
	n = int(t.dimensions[1])
	if tokens <= 0 {
		return 0, 0, 0, fmt.Errorf("metal matmul q2_k: non-positive tokens %d", tokens)
	}
	if inLen != k*tokens {
		return 0, 0, 0, fmt.Errorf("metal matmul q2_k: input slice length %d != K(%d)*T(%d)", inLen, k, tokens)
	}
	blocks = k / 256
	if t.data.Len() < n*blocks*84 {
		return 0, 0, 0, fmt.Errorf("metal matmul q2_k %q: weight buffer too small", t.name)
	}
	return k, n, blocks, nil
}

func (b *metalBatch) encodeMatmulQ2K(t boundMetalTensor, in, out metalmem.Slice[float32], q8Scratch metalmem.Slice[byte], tokens int, pipeline apple.MTLComputePipelineState) error {
	if b.ops == nil || b.ops.matmulQ2K == nil || b.ops.matmulQ2K.quantPip == nil || pipeline == nil {
		return fmt.Errorf("metal matmul q2_k: missing pipeline")
	}
	k, n, blocks, err := checkMatmulQ2KProfile(t, in.Len(), tokens)
	if err != nil {
		return err
	}
	if out.Len() != n*tokens {
		return fmt.Errorf("metal matmul q2_k: output slice length %d != N(%d)*T(%d)", out.Len(), n, tokens)
	}
	wantScratch := tokens * blocks * 292
	if q8Scratch.Len() < wantScratch {
		return fmt.Errorf("metal matmul q2_k: scratch buffer length %d < required %d", q8Scratch.Len(), wantScratch)
	}

	// 1. Quantize all tokens to Q8_K
	enc := b.encoder()
	enc.SetComputePipelineState(b.ops.matmulQ2K.quantPip)
	enc.SetBufferWithOffsetAtIndex(in.Buffer(), in.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(q8Scratch.Buffer(), q8Scratch.Offset(), 1)
	libmetal.SetScalar(enc, uint32(k), 2)
	libmetal.SetScalar(enc, uint32(tokens), 3)
	enc.DispatchThreadsThreadsPerThreadgroup(
		apple.MTLSize{Width: uint(blocks), Height: uint(tokens), Depth: 1},
		apple.MTLSize{Width: 1, Height: 1, Depth: 1},
	)

	// 2. Batched Q2_K Matmul
	enc = b.encoder()
	enc.SetComputePipelineState(pipeline)
	enc.SetBufferWithOffsetAtIndex(t.data.Buffer(), t.data.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(q8Scratch.Buffer(), q8Scratch.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 2)
	libmetal.SetScalar(enc, uint32(k), 3)
	libmetal.SetScalar(enc, uint32(n), 4)
	libmetal.SetScalar(enc, uint32(tokens), 5)
	threadsPerTG := min(n, 256)
	if threadsPerTG == 0 {
		threadsPerTG = 1
	}
	enc.DispatchThreadsThreadsPerThreadgroup(
		apple.MTLSize{Width: uint(n), Height: uint(tokens), Depth: 1},
		apple.MTLSize{Width: uint(threadsPerTG), Height: 1, Depth: 1},
	)
	b.encoded++
	return nil
}

// Mul performs out [N x T] = weight [K x N] * in [K x T].
// weight: boundMetalTensor of Q2_K (typeID 10), dimensions [K, N]
// in: slice of float32, length K * T (column-major: token t is in[t*K : (t+1)*K])
// tokens: T
func (m *metalMatmulPreservedQ2K) Mul(t boundMetalTensor, inSlice []float32, tokens int) ([]float32, error) {
	return m.mul(t, inSlice, tokens, m.matmulPip)
}

// mulReversed runs the reassociated twin. Tests only.
func (m *metalMatmulPreservedQ2K) mulReversed(t boundMetalTensor, inSlice []float32, tokens int) ([]float32, error) {
	return m.mul(t, inSlice, tokens, m.reversedPip)
}

func (m *metalMatmulPreservedQ2K) mul(t boundMetalTensor, inSlice []float32, tokens int, pipeline apple.MTLComputePipelineState) ([]float32, error) {
	if m == nil || m.ops == nil {
		return nil, fmt.Errorf("metal matmul q2_k: closed")
	}
	_, n, blocks, err := checkMatmulQ2KProfile(t, len(inSlice), tokens)
	if err != nil {
		return nil, err
	}

	inBuf, err := metalmem.AllocFromSlice(m.dev, inSlice)
	if err != nil {
		return nil, err
	}
	defer inBuf.Free()

	// Q8_K buffer for all T tokens: T * blocks * 292 bytes
	q8Buf, err := metalmem.Alloc[byte](m.dev, tokens*blocks*292)
	if err != nil {
		return nil, err
	}
	defer q8Buf.Free()

	outBuf, err := metalmem.Alloc[float32](m.dev, n*tokens)
	if err != nil {
		return nil, err
	}
	defer outBuf.Free()

	batch, err := newMetalBatch(m.ops)
	if err != nil {
		return nil, err
	}
	if err := batch.encodeMatmulQ2K(t, inBuf, outBuf, q8Buf, tokens, pipeline); err != nil {
		return nil, err
	}
	if err := batch.Wait(context.Background()); err != nil {
		return nil, fmt.Errorf("metal matmul q2_k %q: %w", t.name, err)
	}
	return append([]float32(nil), outBuf.HostSlice()...), nil
}

const metalMatmulQ2KPreservedMSL = `#include <metal_stdlib>
using namespace metal;

// kReverseBlocks exists only so a test can build a deliberately reassociated
// twin of this kernel and prove its own inputs can tell the two apart. It is
// false in every pipeline the runtime creates; specialization removes the
// select. The inner group and element loops accumulate into int, which is
// associative, so this outer float accumulation is the only order that can
// change a result.
constant bool kReverseBlocks [[function_constant(0)]];

kernel void quant_q8_k_batched(
    device const float *x [[buffer(0)]],
    device uchar *out [[buffer(1)]],
    constant uint &cols [[buffer(2)]],
    constant uint &tokens [[buffer(3)]],
    uint2 pos [[thread_position_in_grid]]) {
  uint block = pos.x;
  uint tok = pos.y;
  uint nb = cols / 256;
  if (block >= nb || tok >= tokens) return;

  uint64_t x_offset = uint64_t(tok) * cols + uint64_t(block) * 256;
  device const float *bx = x + x_offset;

  uint64_t out_offset = (uint64_t(tok) * nb + uint64_t(block)) * 292;
  device uchar *p = out + out_offset;
  device char *q = reinterpret_cast<device char *>(p + 4);

  float signed_max = 0.0f, amax = 0.0f;
  for (uint i = 0; i < 256; i++) {
    float v = bx[i], a = abs(v);
    if (a > amax) {
      amax = a;
      signed_max = v;
    }
  }
  if (amax == 0.0f) {
    *reinterpret_cast<device float *>(p) = 0.0f;
    for (uint i = 0; i < 288; i++) p[4 + i] = 0;
    return;
  }
  float iscale = -127.0f / signed_max;
  for (uint i = 0; i < 256; i++) {
    int v = int(rint(iscale * bx[i]));
    v = clamp(v, -128, 127);
    q[i] = char(v);
  }
  device short *s = reinterpret_cast<device short *>(p + 260);
  for (uint j = 0; j < 16; j++) {
    int v = 0;
    for (uint i = 0; i < 16; i++) v += int(q[j * 16 + i]);
    s[j] = short(v);
  }
  *reinterpret_cast<device float *>(p) = 1.0f / iscale;
}

kernel void matmul_q2_k(
    device const uchar *weights [[buffer(0)]],
    device const uchar *q8 [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &cols [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &tokens [[buffer(5)]],
    uint2 pos [[thread_position_in_grid]]) {
  uint row = pos.x;
  uint tok = pos.y;
  if (row >= rows || tok >= tokens) return;

  uint nb = cols / 256;
  float total = 0.0f;
  uint64_t tok_q8_offset = uint64_t(tok) * nb * 292;
  device const uchar *tok_q8 = q8 + tok_q8_offset;

  for (uint bi = 0; bi < nb; bi++) {
    uint block = kReverseBlocks ? (nb - 1 - bi) : bi;
    const device uchar *p = weights + uint64_t(row) * nb * 84 + uint64_t(block) * 84;
    const device uchar *qy = tok_q8 + uint64_t(block) * 292;
    const device char *v = reinterpret_cast<device const char *>(qy + 4);
    const device short *bs = reinterpret_cast<device const short *>(qy + 260);
    int isum = 0, summs = 0;
    for (uint group = 0; group < 16; group++) {
      uchar sm = p[group];
      uint qbase = 32 * (group / 8) + 16 * (group & 1);
      uint shift = ((group / 2) & 3) * 2;
      int dotv = 0;
      for (uint i = 0; i < 16; i++) {
        dotv += int((p[16 + qbase + i] >> shift) & 3) * int(v[group * 16 + i]);
      }
      isum += int(sm & 15) * dotv;
      summs += int(bs[group]) * int(sm >> 4);
    }
    float yd = *reinterpret_cast<device const float *>(qy);
    total += yd * float(*reinterpret_cast<device const half *>(p + 80)) * float(isum) -
             yd * float(*reinterpret_cast<device const half *>(p + 82)) * float(summs);
  }
  out[uint64_t(tok) * rows + row] = total;
}
`
