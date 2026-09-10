// Deep-regime boundary.
//
// A layer takes the batched-compressed path when batchedCompressed is true.
// The interesting regime, where it is false, is what the workload benchmarks
// have never reached: both the prefill and the decode baselines report 86.00
// command buffers per token at every depth they sample, including ctx512.
//
// The condition is deterministic integer arithmetic over the position, so the
// first position that reaches the deep regime can be derived rather than
// searched for with an expensive model run. That derivation is the point of
// this file: it costs no GPU, no model load, and no quiet host, and it says
// which depth a measurement must use before the measurement is scheduled.
//
// deepRegimeFirstPosition replicates the predicate rather than calling it, so
// that this test needs nothing exported from the runtime. A replica can drift
// from its original, so TestDeepRegimePredicateMatchesSource pins the source
// line it replicates and fails if that line changes. Read the two together:
// neither is sufficient alone.

package main

import (
	"os"
	"strings"
	"testing"
)

// indexerRatio is the compressor ratio for even layers, from the allocation in
// metal_runtime_darwin.go: "ratio, width := 4, 1024", with odd layers taking
// "ratio, width = 128, 512" and no indexer state at all.
const indexerRatio = 4

// deepRegimeFirstPosition returns the lowest position at which an even layer
// i >= 2 leaves the batched-compressed path, and reports whether any position
// below limit does so.
func deepRegimeFirstPosition(limit int) (int, bool) {
	for position := 0; position < limit; position++ {
		// state.indexer.n counts emits strictly before this position; it is
		// incremented after the predicate is evaluated, not before.
		n := position / indexerRatio
		willEmit := (position+1)%indexerRatio == 0
		batched := n < 512 || (n == 512 && !willEmit)
		if !batched {
			return position, true
		}
	}
	return 0, false
}

func TestDeepRegimeBoundary(t *testing.T) {
	first, ok := deepRegimeFirstPosition(1 << 14)
	if !ok {
		t.Fatal("no position reaches the deep regime")
	}
	if want := 2051; first != want {
		t.Errorf("first deep-regime position = %d, want %d", first, want)
	}
	// Once past the boundary the regime holds, so a measurement that starts
	// above it does not need to hit an exact position.
	for position := 2052; position < 2052+256; position++ {
		n := position / indexerRatio
		willEmit := (position+1)%indexerRatio == 0
		if n < 512 || (n == 512 && !willEmit) {
			t.Fatalf("position %d fell back to the batched path", position)
		}
	}
}

// TestPrefill2048MissesDeepRegime records why the obvious measurement is the
// wrong one. A 2048-token prefill covers positions 0 through 2047, where the
// indexer has emitted 511 times and the predicate is still batched. Scheduling
// ctx2048 to observe the deep regime yields a null result at full cost.
func TestPrefill2048MissesDeepRegime(t *testing.T) {
	const prefill = 2048
	if _, ok := deepRegimeFirstPosition(prefill); ok {
		t.Fatal("a 2048-token prefill reaches the deep regime; the comment above is stale")
	}
	if n := (prefill - 1) / indexerRatio; n != 511 {
		t.Errorf("indexer.n at the last prefilled position = %d, want 511", n)
	}
}

// TestDeepRegimePredicateMatchesSource fails when the predicate this file
// replicates is edited, so the derived boundary cannot silently go stale.
func TestDeepRegimePredicateMatchesSource(t *testing.T) {
	const (
		file = "metal_runtime_darwin.go"
		want = "batchedCompressed := i >= 2 && (i%2 != 0 || state.indexer.n < 512 || state.indexer.n == 512 && !indexerWillEmit)"
		emit = "indexerWillEmit := i >= 2 && i%2 == 0 && (position+1)%state.indexer.ratio == 0"
	)
	src, err := os.ReadFile(file)
	if err != nil {
		t.Fatalf("read %s: %v", file, err)
	}
	for _, line := range []string{want, emit} {
		if !strings.Contains(string(src), line) {
			t.Errorf("%s no longer contains the replicated predicate:\n\t%s\nre-derive deepRegimeFirstPosition against the current source", file, line)
		}
	}
	// A guard that matches anything proves nothing: confirm the search can fail.
	if strings.Contains(string(src), want+" // sentinel that must not exist") {
		t.Fatal("negative control matched; the guard is fail-open")
	}
}
