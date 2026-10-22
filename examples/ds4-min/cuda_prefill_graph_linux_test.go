//go:build linux

package main

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"
	"time"
)

// TestCUDAPrefillCUDAGraph tests instantiated CUDA graph capture and replay for prefill batch execution.
// It verifies that prefill with CUDA graphs produces exact canonical continuation tokens
// (15342, 201), valid finite logits across all 43 layers, preserves layer compressed counts,
// and reproduces the recorded logit digest for whichever compressor projection is selected.
func TestCUDAPrefillCUDAGraph(t *testing.T) {
	gate := os.Getenv("DS4_CUDA_PREFILL_DENSE_D2R_GATE")
	if gate == "" {
		t.Skip("DS4_CUDA_PREFILL_DENSE_D2R_GATE is not set")
	}
	if gate != "1" || os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("DS4_CUDA_PREFILL_DENSE_D2R_GATE=1 and a GPU lock are required")
	}
	model, input := os.Getenv("DS4_MODEL"), os.Getenv("DS4_CUDA_PREFILL_TOKENS")
	if model == "" || input == "" {
		t.Fatal("DS4_MODEL and DS4_CUDA_PREFILL_TOKENS are required")
	}
	data, err := os.ReadFile(input)
	if err != nil {
		t.Fatal(err)
	}
	tokens, err := parsePrefillTokens(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(tokens) != 2048 {
		t.Fatalf("input contains %d tokens, want 2048", len(tokens))
	}
	// Select the arm through the user's door, so this test measures a
	// configuration a user can ask for; see cudaShipDenseD2R.
	t.Setenv("DS4_CUDA_DENSE_D2R", "1")
	t.Setenv("DS4_CUDA_MOE_D2R", "1")
	// The HMMA tokentile path builds its records from a group-final snapshot of
	// the compressor state, so it is the first suspect whenever a change to the
	// prefill group width moves the logits. Disabling it on both arms of a width
	// comparison is the discriminating control: if the two widths then agree
	// with each other, the width dependence lives entirely in this path.
	if os.Getenv("DS4_CUDA_DISABLE_ATTN_HMMA") != "1" {
		t.Setenv("DS4_CUDA_ATTENTION_HMMA", "1")
	}
	r, err := newCUDARuntime(model, 4096)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := r.Close(); err != nil {
			t.Error(err)
		}
	}()

	// The compressor pair runs through cuBLAS by default; the hand-written
	// kernel is the rollback arm and the bit-exact one.
	r.disableLibraryGEMM = os.Getenv("DS4_CUDA_DISABLE_LIBRARY_GEMM") == "1"
	r.disableCompressorBatch = os.Getenv("DS4_CUDA_DISABLE_COMPRESSOR_BATCH") == "1"
	r.disableAttnQKVD2R = os.Getenv("DS4_CUDA_DISABLE_ATTN_QKV_D2R") == "1"
	r.disableRawAttentionBatch = os.Getenv("DS4_CUDA_DISABLE_RAW_ATTENTION_BATCH") == "1"
	r.disableChunkedArena = os.Getenv("DS4_CUDA_DISABLE_CHUNKED_ARENA") == "1"
	if err := r.Reset(); err != nil {
		t.Fatal(err)
	}

	start := time.Now()
	if err := r.EvalBatch(tokens[:2048], 0); err != nil {
		t.Fatalf("CUDA graph prefill 2048: %v", err)
	}
	elapsed := time.Since(start)

	// Verify that CUDA graphs were captured for all layers.
	if r.batch == nil {
		t.Fatal("expected batch resources to be allocated")
	}
	for i := 0; i < cudaLayers; i++ {
		if r.batch.graphs[i].qkv == nil {
			t.Fatalf("layer %d qkv batch graph was not instantiated", i)
		}
		if r.batch.graphs[i].ffn == nil {
			t.Fatalf("layer %d ffn batch graph was not instantiated", i)
		}
	}

	logits := r.Logits()
	if len(logits) != 129280 {
		t.Fatalf("logits length %d, want 129280", len(logits))
	}
	rawDigest, nonFinite := prefillLogitDigest(logits)
	// The continuation tokens are the invariant: they have survived every change
	// to this path, including both attention implementations and every prefill
	// group width. The digest is a deliberate baseline rather than a second
	// invariant -- it is re-recorded whenever the arithmetic is changed on
	// purpose, and it earns its place by catching the changes that were not on
	// purpose. Whoever moves it must say in the commit message why.
	digest := rawDigest
	// Re-baselined twice as projections moved to cuBLAS: those paths sum the
	// products in the library's order rather than the kernel's, and the
	// compressor pair also narrows its activations to half, which the kernel
	// kept in float. Every hand-written and per-token arm, and the digest they
	// produce together, stays reachable under DS4_CUDA_DISABLE_LIBRARY_GEMM=1.
	//	b8645073...  every projection hand-written or per-token (still reachable)
	//	d9bea8ca...  compressor pair through cuBLAS
	//	0439f3e1...  and the router
	//	99c96768...  and the heavy-compute batch, in one GEMM per layer
	//	09076406...  and attnQA/attnQB/attnKV, through the INT8 tensor-core GEMM
	//	             (DS4_CUDA_DISABLE_ATTN_QKV_D2R=1 keeps the matvec arm, at 99c96768)
	//	f0e4094b...  and layers 0 and 1 a group at a time, through the tokentile
	//	             path (DS4_CUDA_DISABLE_RAW_ATTENTION_BATCH=1 is at 09076406)
	const wantDigest = "f0e4094b28390eab266b119a0d7e61a45719c1eac4250e16d9c33c777e9f35fe"

	// Every invariant is observed before any of them stops the run. The digest
	// is the strictest of the three and the first to move, so failing on it
	// immediately hides whether the continuation tokens -- the semantic
	// invariant -- survived, and hides the throughput the change was made for.
	// A low-bit digest drift that keeps [15342, 201] is a different finding
	// from one that does not, and only a run that reports both can tell them
	// apart. All three still gate: the failure is raised at the end.
	// The continuation eval runs on the state the first prefill left, so it has
	// to happen before any repeat resets that state. Its error is recorded
	// rather than raised: a first pass that produced NaN makes this eval fail,
	// and raising here would skip the repeat in exactly the runs the repeat
	// exists to describe.
	tok0, tok1, evalErr := greedyToken(logits), -1, error(nil)
	if evalErr = r.Eval(tok0, 2048); evalErr == nil {
		tok1 = greedyToken(r.Logits())
	}

	// Under DS4_CUDA_PREFILL_REPEAT the same process prefills a second time
	// from a fresh Reset and reports both digests. The pair answers a question
	// one run cannot. warmModel's comment says an unfaulted page gives the
	// first evaluation wrong bytes for whichever tensors it reaches first and
	// leaves every later evaluation correct; a wrong first digest beside a
	// right second one is that bug, and two equally wrong digests are
	// something that persists past the first touch.
	repeat := ""
	if os.Getenv("DS4_CUDA_PREFILL_REPEAT") == "1" {
		if err := r.Reset(); err != nil {
			t.Fatalf("reset for repeat: %v", err)
		}
		if err := r.EvalBatch(tokens[:2048], 0); err != nil {
			t.Fatalf("repeat prefill 2048: %v", err)
		}
		d2, nf2 := prefillLogitDigest(r.Logits())
		repeat = fmt.Sprintf(" repeat=%s repeatNonFinite=%v", d2[:8], nf2)
	}

	var broken []string
	if nonFinite.bad() {
		broken = append(broken, fmt.Sprintf("non-finite logits %v", nonFinite))
	}
	if digest != wantDigest {
		broken = append(broken, fmt.Sprintf("logit digest: got %s, want %s", digest, wantDigest))
	}
	if evalErr != nil {
		broken = append(broken, fmt.Sprintf("eval continuation token 0: %v", evalErr))
	}
	if tok0 != 15342 || tok1 != 201 {
		broken = append(broken, fmt.Sprintf("continuation tokens: got [%d, %d], want [15342, 201]", tok0, tok1))
	}
	for l := 2; l < len(r.state.layers); l++ {
		if nComp := r.state.layers[l].attn.n; nComp <= 0 {
			broken = append(broken, fmt.Sprintf("layer %d compressed count = %d, want > 0", l, nComp))
		}
	}

	toksPerSec := float64(len(tokens)) / elapsed.Seconds()
	t.Logf("Prefill CUDA graph: width=%d elapsed=%v throughput=%.2f tok/s digest=%s tokens=[%d, %d]%s",
		cudaPrefillGroupWidth, elapsed, toksPerSec, digest, tok0, tok1, repeat)
	if len(broken) > 0 {
		t.Fatalf("prefill invariants broken:\n\t%s", strings.Join(broken, "\n\t"))
	}
}

// prefillLogitDigest returns the SHA-256 over the logits' little-endian float
// bits and the index of the first non-finite value, or -1 if there is none. A
// non-finite logit is reported rather than raised so the caller can observe
// every invariant before any of them stops the run: a run that produces NaN is
// still a run whose throughput and continuation tokens say something.
func prefillLogitDigest(logits []float32) (string, logitExtent) {
	raw := make([]byte, len(logits)*4)
	for i, v := range logits {
		binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(v))
	}
	return fmt.Sprintf("%x", sha256.Sum256(raw)), nonFiniteExtent(logits)
}
