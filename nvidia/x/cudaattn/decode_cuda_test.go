package cudaattn_test

import (
	"context"
	"fmt"
	"math"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaattn"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

// decodeReference computes single-sequence causal decode attention on the CPU
// in float64. It is the oracle: the kernel must match it elementwise at every
// block size, because the block size is a launch parameter and not part of the
// function being computed.
func decodeReference(q, kCache, vCache []float32, curPos, qHeads, kvHeads, headDim int, scale float64) []float32 {
	out := make([]float32, qHeads*headDim)
	group := qHeads / kvHeads
	for qh := 0; qh < qHeads; qh++ {
		kh := qh / group
		scores := make([]float64, curPos+1)
		maxScore := math.Inf(-1)
		for t := 0; t <= curPos; t++ {
			kbase := (t*kvHeads + kh) * headDim
			qbase := qh * headDim
			for d := 0; d < headDim; d++ {
				scores[t] += float64(q[qbase+d]) * float64(kCache[kbase+d])
			}
			scores[t] *= scale
			maxScore = math.Max(maxScore, scores[t])
		}
		sum := 0.0
		for t := range scores {
			scores[t] = math.Exp(scores[t] - maxScore)
			sum += scores[t]
		}
		for d := 0; d < headDim; d++ {
			acc := 0.0
			for t, s := range scores {
				vbase := (t*kvHeads + kh) * headDim
				acc += s * float64(vCache[vbase+d])
			}
			out[qh*headDim+d] = float32(acc / sum)
		}
	}
	return out
}

// TestCausalFlashAttentionDecodeMatchesReference compares the single-sequence
// decode kernel against decodeReference elementwise, at several block sizes.
//
// The block size must not change the result. Every call site launches this
// kernel with one thread per block, and at that width the arithmetic is right,
// so a test that only ever launched .Block(1) would pass over a kernel that is
// wrong everywhere else. Raising the block size for performance is the most
// natural change anyone would make to this launch, which is exactly why the
// oracle is checked across widths.
func TestCausalFlashAttentionDecodeMatchesReference(t *testing.T) {
	for _, threads := range []uint32{1, 2, 8, 32, 64, 128} {
		t.Run(fmt.Sprintf("block%d", threads), func(t *testing.T) {
			testDecodeAgainstReference(t, threads, false)
		})
	}
	// Callers that predate the cross-thread reduction size their shared memory
	// for the scores alone. The kernel must stay correct for them, because a
	// silent increase in the shared-memory requirement is an out-of-bounds
	// write at every existing launch site rather than a compile error.
	t.Run("block1LegacySharedMemory", func(t *testing.T) {
		testDecodeAgainstReference(t, 1, true)
	})
}

func testDecodeAgainstReference(t *testing.T, threads uint32, legacyShared bool) {
	dev := cudatest.Require(t)
	defer dev.Close()
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	compiled, err := cudaattn.Compile(major, minor)
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	mod, err := compiled.Load(dev)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer mod.Unload()
	decode, err := mod.GetFunction("CausalFlashAttentionDecode")
	if err != nil {
		t.Fatal(err)
	}
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()

	const (
		qHeads  = 4
		kvHeads = 2
		headDim = 6 // deliberately not a vector-width multiple
		curPos  = 63
		maxSeq  = 64
	)
	// Values are spread widely enough that the softmax weights differ
	// substantially between tokens. If every token received the same weight
	// the result would be the plain mean of V, which is what an unreduced
	// maximum and sum produce, so a flat input would hide the defect.
	q := make([]float32, qHeads*headDim)
	k := make([]float32, maxSeq*kvHeads*headDim)
	v := make([]float32, len(k))
	for i := range q {
		q[i] = float32((i*17)%29-14) / 7
	}
	for i := range k {
		k[i] = float32((i*11)%31-15) / 9
		v[i] = float32((i*7)%23-11) / 13
	}
	scale := 1.0 / math.Sqrt(float64(headDim))

	dq := deviceSlice(t, dev, q)
	dk := deviceSlice(t, dev, k)
	dv := deviceSlice(t, dev, v)
	dout := deviceZeros[float32](t, dev, qHeads*headDim)
	for _, free := range []func() error{dq.Free, dk.Free, dv.Free, dout.Free} {
		defer free()
	}

	sharedBytes, err := cudaattn.DecodeSharedMemoryBytes(curPos+1, int(threads))
	if err != nil {
		t.Fatalf("DecodeSharedMemoryBytes: %v", err)
	}
	if legacyShared {
		sharedBytes = (curPos + 1) * 4
	}
	shared := uint32(sharedBytes)
	if err := cudalaunch.Kernel(decode).Grid(qHeads, 1).Block(threads).SharedMem(shared).On(stream).Launch(
		dq, dk, dv, dout,
		int32(curPos), int32(maxSeq), int32(qHeads), int32(kvHeads), int32(headDim), float32(scale),
	); err != nil {
		t.Fatalf("launch: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := stream.Synchronize(ctx); err != nil {
		t.Fatalf("synchronize: %v", err)
	}
	got := make([]float32, qHeads*headDim)
	if err := dev.WithContext(func() error { return dout.CopyToHost(got) }); err != nil {
		t.Fatal(err)
	}

	want := decodeReference(q, k, v, curPos, qHeads, kvHeads, headDim, scale)
	const tol = 3e-6
	bad := 0
	for i := range want {
		if diff := math.Abs(float64(got[i] - want[i])); diff > tol {
			if bad < 8 {
				t.Errorf("out[%d] = %v, want %v (diff %v)", i, got[i], want[i], diff)
			}
			bad++
		}
	}
	if bad != 0 {
		t.Fatalf("%d of %d elements differ from the reference at blockDim=%d", bad, len(want), threads)
	}
}

// TestCausalFlashAttentionDecodeDependsOnQK guards the specific way this kernel
// fails when its per-thread maximum and sum are not reduced across the block:
// every score exponentiates to one, the normaliser is one, and the output
// collapses to the unweighted sum of the value rows, so Q and K stop affecting
// the result at all. Perturbing Q must change the output.
func TestCausalFlashAttentionDecodeDependsOnQK(t *testing.T) {
	for _, threads := range []uint32{1, 64} {
		t.Run(fmt.Sprintf("block%d", threads), func(t *testing.T) {
			a := runDecode(t, threads, 1.0)
			b := runDecode(t, threads, -1.0)
			same := true
			for i := range a {
				if math.Abs(float64(a[i]-b[i])) > 1e-9 {
					same = false
					break
				}
			}
			if same {
				t.Fatalf("negating Q left the output unchanged at blockDim=%d; "+
					"Q and K are not affecting the result", threads)
			}
		})
	}
}

// runDecode launches the kernel with Q scaled by qSign and returns the output.
func runDecode(t *testing.T, threads uint32, qSign float32) []float32 {
	t.Helper()
	dev := cudatest.Require(t)
	defer dev.Close()
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	compiled, err := cudaattn.Compile(major, minor)
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	mod, err := compiled.Load(dev)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer mod.Unload()
	decode, err := mod.GetFunction("CausalFlashAttentionDecode")
	if err != nil {
		t.Fatal(err)
	}
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()

	const (
		qHeads  = 2
		kvHeads = 1
		headDim = 4
		curPos  = 63
		maxSeq  = 64
	)
	q := make([]float32, qHeads*headDim)
	k := make([]float32, maxSeq*kvHeads*headDim)
	v := make([]float32, len(k))
	for i := range q {
		q[i] = qSign * float32((i*17)%29-14) / 7
	}
	for i := range k {
		k[i] = float32((i*11)%31-15) / 9
		v[i] = float32((i*7)%23-11) / 13
	}
	dq := deviceSlice(t, dev, q)
	dk := deviceSlice(t, dev, k)
	dv := deviceSlice(t, dev, v)
	dout := deviceZeros[float32](t, dev, qHeads*headDim)
	for _, free := range []func() error{dq.Free, dk.Free, dv.Free, dout.Free} {
		defer free()
	}
	sharedBytes, err := cudaattn.DecodeSharedMemoryBytes(curPos+1, int(threads))
	if err != nil {
		t.Fatalf("DecodeSharedMemoryBytes: %v", err)
	}
	shared := uint32(sharedBytes)
	if err := cudalaunch.Kernel(decode).Grid(qHeads, 1).Block(threads).SharedMem(shared).On(stream).Launch(
		dq, dk, dv, dout,
		int32(curPos), int32(maxSeq), int32(qHeads), int32(kvHeads), int32(headDim),
		float32(1.0/math.Sqrt(headDim)),
	); err != nil {
		t.Fatalf("launch: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := stream.Synchronize(ctx); err != nil {
		t.Fatalf("synchronize: %v", err)
	}
	got := make([]float32, qHeads*headDim)
	if err := dev.WithContext(func() error { return dout.CopyToHost(got) }); err != nil {
		t.Fatal(err)
	}
	return got
}
