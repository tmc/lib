//go:build darwin && arm64

package main

import (
	"crypto/sha256"
	"fmt"
	"os"
	"testing"
)

// The deep regime is the branch at metal_runtime_darwin.go where evalLayer
// commits the batch mid-layer, runs the compressor and attention on the host,
// copies the heads back, and encodes the rest of the layer into a second
// command buffer. TestDeepRegimeBoundary derives its first position
// arithmetically: 2051.
//
// The condition is metal_runtime_darwin.go:635:
//
//	batchedCompressed := i >= 2 && (i%2 != 0 || state.indexer.n < 512 ||
//		state.indexer.n == 512 && !indexerWillEmit)
//
// so an even layer leaves the batched regime when its indexer cache passes 512
// entries. The indexer emits on (position+1)%ratio == 0 at ratio 4, which puts
// the 512th emit at position 2047 and the 513th at 2051; 2048 through 2050 are
// still batched because n == 512 and no emit is due. Hence 2051, and hence the
// windows below: the constant is a consequence of the branch, and it moves if
// the ratio ever does. That is why the assertion reads the runtime's counters
// instead of this number.
//
// The same branch exists on CUDA. cuda_session_linux.go:804 is
// `if state.indexer.n > 512`, checked after the increment at :801, over the
// same emit rule at cuda_compressed_linux.go:36 -- so it turns on at position
// 2051 too, and gates two extra matvecs per indexer layer rather than a
// command-buffer split. One model-level branch, two runtimes, one position.
// It is the only mechanism found so far in this work that does cross backends;
// the constant-cache serialization measured in ea822874 does not.
//
// A 2048-token prefill ends at position 2047, but TestOracleWorkload also
// decodes 128 tokens at positions 2048 through 2175. That decode crosses this
// boundary. Its final-logits digest covers the endpoint of that execution,
// not each intermediate. The n=64 anchor covers only positions 0 through 63.
//
// This anchor brackets command-buffer counts on either side of the boundary
// and requires the deep count to exceed the shallow count. It records the
// final-logits digest; bit-for-bit regression checking remains disabled until
// deepRegimeAnchorLogitsSHA256 is registered.
//
// The first run of this anchor recorded 2.50 command buffers per token below
// the boundary against 232.50 above, and 1.00 GPU wait per token against 42.00. The gap is not the mid-layer
// split itself -- that is one extra buffer for each of the 21 even layers at
// or above 2 -- but the host-side ops the branch then calls, each of which
// opens and commits a command buffer of its own. Those numbers are recorded
// here as an observation, not asserted: the assertion below is only that the
// deep count exceeds the shallow one, which is what proves the regime was
// entered without fitting a bound to a single run.
const (
	deepRegimeAnchorTokens  = 2060
	deepRegimeAnchorContext = 4096

	// The window below the boundary, and the window above it. Both sit
	// clear of position 2051 so neither straddles the transition.
	deepRegimeShallowFrom = 2040
	deepRegimeShallowTo   = 2048
	deepRegimeDeepFrom    = 2052
	deepRegimeDeepTo      = 2060
)

// deepRegimeAnchorTokensSHA256 binds the sequence this anchor is scoped to,
// the same way metalRegionPrefixSHA256 binds the 64-token prefix. It was
// computed from the token file and the wrap rule below -- both pure data --
// before the model was ever loaded, so it is a registered input rather than
// a recorded output.
const deepRegimeAnchorTokensSHA256 = "23c22002cdb61c989960a54692828f46b3fab6ba76f1b0fd9dab761e4e3f9fa4"

// deepRegimeAnchorLogitsSHA256 is unset pending two independent agreeing
// runs. While it is empty, a successful test records a finite output digest
// and checks the submission counts; it does not assert numerical identity.
// Fill it in only from retained results of two independent runs that agree.
const deepRegimeAnchorLogitsSHA256 = ""

// deepRegimeAnchorTokenIDs extends the 2048-token oracle workload to 2060 by
// wrapping back to its own head. Four more tokens would reach position 2051;
// twelve leaves eight positions above the boundary, so the assertion covers
// the regime holding rather than one position touching it.
//
// Wrapping rather than inventing IDs keeps every token one the workload
// already contains, so the sequence stays in the corpus this package is
// scoped to and no token can fall outside the vocabulary.
func deepRegimeAnchorTokenIDs(t *testing.T) []int {
	t.Helper()
	input, err := os.ReadFile(metalRegionTokensPath)
	if err != nil {
		t.Fatal(err)
	}
	base, err := parsePrefillTokens(input)
	if err != nil {
		t.Fatal(err)
	}
	if len(base) != 2048 {
		t.Fatalf("token file holds %d tokens, want the 2048 this extension assumes", len(base))
	}
	tokens := make([]int, 0, deepRegimeAnchorTokens)
	tokens = append(tokens, base...)
	tokens = append(tokens, base[:deepRegimeAnchorTokens-len(base)]...)
	got := fmt.Sprintf("%x", sha256.Sum256(oracleWorkloadTokenBytes(tokens)))
	if got != deepRegimeAnchorTokensSHA256 {
		t.Fatalf("token sequence is %s, want the registered %s: this is not the corpus this anchor is scoped to", got, deepRegimeAnchorTokensSHA256)
	}
	return tokens
}

// TestMetalDeepRegimeAnchor evaluates 2060 positions, proves from the command
// buffer counters that the run crossed into the deep regime, and reports the
// logits digest.
func TestMetalDeepRegimeAnchor(t *testing.T) {
	model := os.Getenv("DS4_MODEL")
	if model == "" {
		t.Skip("DS4_MODEL is not set")
	}
	tokens := deepRegimeAnchorTokenIDs(t)

	// The boundary this anchor exists to cross is derived, not assumed.
	// If the predicate moves, this fails here rather than passing on a
	// window that no longer straddles anything.
	first, ok := deepRegimeFirstPosition(deepRegimeAnchorContext)
	if !ok {
		t.Fatal("no position in this context reaches the deep regime")
	}
	if first <= deepRegimeShallowTo || first >= deepRegimeDeepFrom {
		t.Fatalf("deep regime starts at %d, which does not lie between the shallow window ending at %d and the deep window starting at %d", first, deepRegimeShallowTo, deepRegimeDeepFrom)
	}

	r, err := newMetalRuntime(model, deepRegimeAnchorContext)
	if err != nil {
		t.Fatal(err)
	}
	closed := false
	defer func() {
		if !closed {
			if err := r.Close(); err != nil {
				t.Errorf("close: %v", err)
			}
		}
	}()
	if err := r.Reset(); err != nil {
		t.Fatalf("reset: %v", err)
	}

	eval := func(from, to int) {
		t.Helper()
		if err := r.EvalBatch(tokens[from:to], from); err != nil {
			t.Fatalf("eval positions %d..%d: %v", from, to-1, err)
		}
	}
	bracket := func(from, to int) metalSubmissions {
		t.Helper()
		start := r.Submissions()
		eval(from, to)
		return r.Submissions().sub(start)
	}

	eval(0, deepRegimeShallowFrom)
	shallow := bracket(deepRegimeShallowFrom, deepRegimeShallowTo)
	eval(deepRegimeShallowTo, deepRegimeDeepFrom)
	deep := bracket(deepRegimeDeepFrom, deepRegimeDeepTo)

	shallowBuffers, _, _, _ := shallow.perToken(deepRegimeShallowTo - deepRegimeShallowFrom)
	deepBuffers, _, _, _ := deep.perToken(deepRegimeDeepTo - deepRegimeDeepFrom)
	shallowTokens := deepRegimeShallowTo - deepRegimeShallowFrom
	deepTokens := deepRegimeDeepTo - deepRegimeDeepFrom
	t.Logf("command buffers per token: shallow %.2f, deep %.2f", shallowBuffers, deepBuffers)
	t.Logf("waits per token: shallow %.2f, deep %.2f", float64(shallow.Waits)/float64(shallowTokens), float64(deep.Waits)/float64(deepTokens))

	// Every layer that takes the host round trip commits its batch, waits,
	// and opens another, so crossing the boundary can only raise this
	// count. Equal counts mean the run stayed batched and the digest below
	// would say nothing about the regime this test is named for.
	if deepBuffers <= shallowBuffers {
		t.Fatalf("command buffers per token did not rise across the boundary: shallow %.2f, deep %.2f -- the run never entered the deep regime, so this test proved nothing", shallowBuffers, deepBuffers)
	}

	logits := r.Logits()
	if len(logits) != metalRegionVocab {
		t.Fatalf("produced %d logits, want %d", len(logits), metalRegionVocab)
	}
	hash, err := oracleWorkloadHash(logits)
	if err != nil {
		t.Fatal(err)
	}
	closed = true
	if err := r.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	t.Logf("metal deep regime logits sha256=%s (tokens=%d, context=%d)", hash, deepRegimeAnchorTokens, deepRegimeAnchorContext)
	if deepRegimeAnchorLogitsSHA256 == "" {
		t.Logf("deepRegimeAnchorLogitsSHA256 is unset, so this run records the digest rather than asserting it; register it only after a second run agrees")
		return
	}
	if hash != deepRegimeAnchorLogitsSHA256 {
		t.Fatalf("logits hash = %s, want registered %s; a mismatch here may be the constant rather than the code if it was registered from fewer than two agreeing runs", hash, deepRegimeAnchorLogitsSHA256)
	}
}
