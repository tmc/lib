//go:build darwin && arm64

package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"testing"

	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// hcTokenFixture is the weight set every token in a batch shares. Only the
// per-token state and block rows vary, which is what the token dimension has
// to keep apart.
type hcTokenFixture struct {
	fn, scale, base, norm boundMetalTensor
}

func newHCTokenFixture(t *testing.T, arena *metalmem.Arena, next func() float32) hcTokenFixture {
	t.Helper()
	fn := metalmem.AllocIn[byte](arena, "hc-tokens-fn", 2*16384*24)
	for i := 0; i < 16384*24; i++ {
		binary.LittleEndian.PutUint16(fn.HostSlice()[2*i:], floatToF16Bits(next()))
	}
	scale := metalmem.AllocIn[byte](arena, "hc-tokens-scale", 12)
	for i := 0; i < 3; i++ {
		binary.LittleEndian.PutUint32(scale.HostSlice()[4*i:], math.Float32bits(0.1))
	}
	base := metalmem.AllocIn[byte](arena, "hc-tokens-base", 96)
	for i := 0; i < 24; i++ {
		binary.LittleEndian.PutUint32(base.HostSlice()[4*i:], math.Float32bits(next()*8))
	}
	norm := metalmem.AllocIn[byte](arena, "hc-tokens-norm", 16384)
	for i := 0; i < 4096; i++ {
		binary.LittleEndian.PutUint32(norm.HostSlice()[4*i:], math.Float32bits(1+next()))
	}
	if err := arena.Err(); err != nil {
		t.Fatal(err)
	}
	return hcTokenFixture{
		fn:    boundMetalTensor{name: "hc_fn.weight", typeID: 1, dimensions: []uint64{16384, 24}, data: fn},
		scale: boundMetalTensor{name: "hc_scale.weight", typeID: 0, dimensions: []uint64{3}, data: scale},
		base:  boundMetalTensor{name: "hc_base.weight", typeID: 0, dimensions: []uint64{24}, data: base},
		norm:  boundMetalTensor{name: "attn_norm.weight", typeID: 0, dimensions: []uint64{4096}, data: norm},
	}
}

// TestMetalHCTokensBitIdentity asserts that HC pre and post over N tokens in
// one dispatch produce exactly the float32 bits that N separate single-token
// calls produce.
//
// Both arms run the same encode bodies, so this does not police the arithmetic
// those bodies share -- a change that moves every token equally moves both arms
// equally and stays green. What it does police is the token dimension itself:
// a wrong stride, a grid index read from the wrong axis, a token reading
// another token's mix or split scratch, or a threadgroup whose reduction tree
// silently changes shape once more than one of them runs. Those are exactly the
// defects this batching introduces, and each of them moves one arm only.
func TestMetalHCTokensBitIdentity(t *testing.T) {
	dev, err := libmetal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	ops, err := newMetalPinnedOps(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer ops.Close()
	arena := metalmem.NewArena(dev)
	defer arena.Free()

	rng := uint32(0x2545f491)
	next := func() float32 {
		rng = rng*1664525 + 1013904223
		return (float32(rng>>8)/float32(1<<24) - 0.5) * 0.1
	}
	f := newHCTokenFixture(t, arena, next)

	// 1 is the pinned shape; 2 is the smallest batch; 3 and 5 are odd counts
	// that no power-of-two stride hides; 8 grows the scratch past every
	// earlier count.
	for _, tokens := range []int{1, 2, 3, 5, 8} {
		t.Run(fmt.Sprintf("tokens=%d", tokens), func(t *testing.T) {
			state := make([]float32, 16384*tokens)
			for i := range state {
				state[i] = next()
			}
			block := make([]float32, 4096*tokens)
			for i := range block {
				block[i] = next()
			}

			// Reference: one call per token, each over its own row.
			wantCur := make([]float32, 4096*tokens)
			wantNorm := make([]float32, 4096*tokens)
			wantPost := make([]float32, 16384*tokens)
			for tok := 0; tok < tokens; tok++ {
				row := state[tok*16384 : (tok+1)*16384]
				cur, normalized, err := ops.HCPre(row, f.fn, f.scale, f.base, f.norm)
				if err != nil {
					t.Fatalf("HCPre token %d: %v", tok, err)
				}
				copy(wantCur[tok*4096:], cur)
				copy(wantNorm[tok*4096:], normalized)
				post, err := ops.HCPost(row, block[tok*4096:(tok+1)*4096], f.fn, f.scale, f.base)
				if err != nil {
					t.Fatalf("HCPost token %d: %v", tok, err)
				}
				copy(wantPost[tok*16384:], post)
			}
			requireTokensDistinct(t, tokens, wantCur, wantPost)

			// Batched: all tokens in one command buffer per operation.
			beforePre := ops.submissions
			gotCur, gotNorm, err := ops.HCPreTokens(state, f.fn, f.scale, f.base, f.norm, tokens)
			if err != nil {
				t.Fatalf("HCPreTokens(%d): %v", tokens, err)
			}
			if got := ops.submissions.sub(beforePre).Buffers; got != 1 {
				t.Errorf("HCPreTokens over %d tokens used %d command buffers, want 1", tokens, got)
			}
			beforePost := ops.submissions
			gotPost, err := ops.HCPostTokens(state, block, f.fn, f.scale, f.base, tokens)
			if err != nil {
				t.Fatalf("HCPostTokens(%d): %v", tokens, err)
			}
			if got := ops.submissions.sub(beforePost).Buffers; got != 1 {
				t.Errorf("HCPostTokens over %d tokens used %d command buffers, want 1", tokens, got)
			}
			requireBitIdentical(t, "pre projection", gotCur, wantCur)
			requireBitIdentical(t, "pre normalization", gotNorm, wantNorm)
			requireBitIdentical(t, "post", gotPost, wantPost)

			// The paired pre-then-post path reuses the split its own pre
			// cached, which is a second encoding of post over the same
			// tokens. It has to land on the same bits as the standalone post
			// that recomputes the split.
			paired := runHCPairedTokens(t, ops, dev, state, block, f, tokens)
			requireBitIdentical(t, "paired post (cached split)", paired, wantPost)
		})
	}
}

// runHCPairedTokens encodes pre and post for tokens tokens into one command
// buffer, the shape a layer's HC pair takes, and returns the post output.
func runHCPairedTokens(t *testing.T, ops *metalPinnedOps, dev *libmetal.Device, state, block []float32, f hcTokenFixture, tokens int) []float32 {
	t.Helper()
	stateGPU, err := metalmem.AllocFromSlice(dev, state)
	if err != nil {
		t.Fatal(err)
	}
	defer stateGPU.Free()
	blockGPU, err := metalmem.AllocFromSlice(dev, block)
	if err != nil {
		t.Fatal(err)
	}
	defer blockGPU.Free()
	preOut, err := metalmem.Alloc[float32](dev, 8192*tokens)
	if err != nil {
		t.Fatal(err)
	}
	defer preOut.Free()
	postOut, err := metalmem.Alloc[float32](dev, 16384*tokens)
	if err != nil {
		t.Fatal(err)
	}
	defer postOut.Free()
	b, err := newMetalBatch(ops)
	if err != nil {
		t.Fatal(err)
	}
	if err := ops.hc.encodePreTokens(b, stateGPU, f.fn, f.scale, f.base, f.norm, preOut, tokens); err != nil {
		t.Fatal(err)
	}
	if ops.hc.pending != b {
		t.Fatalf("encodePreTokens(%d) left no reusable split for its own batch", tokens)
	}
	if err := ops.hc.encodePostTokens(b, stateGPU, blockGPU, f.fn, f.scale, f.base, postOut, tokens); err != nil {
		t.Fatal(err)
	}
	if err := b.Wait(t.Context()); err != nil {
		t.Fatal(err)
	}
	return append([]float32(nil), postOut.HostSlice()...)
}

func requireBitIdentical(t *testing.T, what string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: got %d elements, want %d", what, len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s: bit mismatch at index %d: got 0x%08x (%g), want 0x%08x (%g)",
				what, i, math.Float32bits(got[i]), got[i], math.Float32bits(want[i]), want[i])
		}
	}
}

// requireTokensDistinct fails unless every token's reference output differs
// bit-for-bit from token 0's.
//
// Without it the bit-identity assertion above is fail-open on the one property
// it exists to check. A kernel that ignored its token index entirely -- writing
// token 0's answer into every row, or reading every row from token 0's state --
// would still match a reference whose tokens all happened to agree. The guard
// runs over the single-token reference outputs themselves, so it measures the
// actual test data rather than asserting that the inputs were varied.
func requireTokensDistinct(t *testing.T, tokens int, cur, post []float32) {
	t.Helper()
	if tokens == 1 {
		return
	}
	for tok := 1; tok < tokens; tok++ {
		if !differsSomewhere(cur[0:4096], cur[tok*4096:(tok+1)*4096]) {
			t.Fatalf("pre output for token %d is bit-identical to token 0 in all 4096 elements, so the bit-identity assertion cannot detect a kernel that ignores its token index", tok)
		}
		if !differsSomewhere(post[0:16384], post[tok*16384:(tok+1)*16384]) {
			t.Fatalf("post output for token %d is bit-identical to token 0 in all 16384 elements, so the bit-identity assertion cannot detect a kernel that ignores its token index", tok)
		}
	}
}

func differsSomewhere(a, b []float32) bool {
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return true
		}
	}
	return false
}
