package cudabench

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
)

func joined(b *B) string { return strings.Join(b.limits(), "\n") }

// TestLimitsNameTheCacheState is the control on the declared caveats: the two
// configurations must not produce the same text. A caveat list that reads the
// same whether or not cache state is controlled tells a reader nothing, which
// is the failure this package's output is supposed to stop making.
func TestLimitsNameTheCacheState(t *testing.T) {
	uncontrolled := &B{bytesPerOp: 1 << 20}
	controlled := &B{bytesPerOp: 1 << 20,
		evict:     func(*cuda.Stream) error { return nil },
		evictName: "L2 overwrite, 2 bytes (2x the 1-byte L2)"}

	if joined(uncontrolled) == joined(controlled) {
		t.Fatal("limits are identical with and without an evictor; they describe nothing")
	}
	if !strings.Contains(joined(uncontrolled), "uncontrolled") {
		t.Errorf("a run with no evictor must say so: %q", joined(uncontrolled))
	}
	if strings.Contains(joined(controlled), "uncontrolled") {
		t.Errorf("a run with an evictor must not claim uncontrolled cache: %q", joined(controlled))
	}
	if !strings.Contains(joined(controlled), "L2 overwrite") {
		t.Errorf("an evicted run must name the evictor: %q", joined(controlled))
	}
}

// TestGBPerSecCaveatFollowsSetBytes checks the declared-bytes caveat appears
// exactly when a bytes-derived rate is reported, and not otherwise.
func TestGBPerSecCaveatFollowsSetBytes(t *testing.T) {
	const caveat = "traffic the workload requires"
	if with := joined(&B{bytesPerOp: 1024}); !strings.Contains(with, caveat) {
		t.Errorf("SetBytes was called, so GB/s is reported and must be qualified: %q", with)
	}
	if without := joined(&B{}); strings.Contains(without, caveat) {
		t.Errorf("no GB/s is reported, so the caveat is noise: %q", without)
	}
}

// TestHostDispatchCaveatAlwaysPresent pins the one limit that holds for every
// configuration, since it is the term the tool is most often used to chase.
func TestHostDispatchCaveatAlwaysPresent(t *testing.T) {
	for _, b := range []*B{
		{},
		{bytesPerOp: 1024},
		{flopsPerOp: 1024},
		{evict: func(*cuda.Stream) error { return nil }, evictName: "x"},
	} {
		if !strings.Contains(joined(b), "host dispatch") {
			t.Errorf("every configuration must state that host dispatch is excluded: %q", joined(b))
		}
	}
}

// TestEvictL2RejectsABadMultiple keeps a zero multiple from installing an
// evictor that allocates nothing and evicts nothing while looking configured.
func TestEvictL2RejectsABadMultiple(t *testing.T) {
	b := &B{}
	if _, err := EvictL2(b, 0); err == nil {
		t.Error("EvictL2 accepted multiple 0")
	}
	if b.evict != nil {
		t.Error("a rejected EvictL2 must not install an evictor")
	}
}

// TestSpanPerIterationIsReachableWithoutAnEvictor is the point of the option:
// before it, the per-iteration-span form could only be had by installing an
// evictor, so the warm arm of a cache comparison was stuck in the other form
// and the two arms were not comparable. Each configuration must state the form
// it used and claim cache control only when it has it.
func TestSpanPerIterationIsReachableWithoutAnEvictor(t *testing.T) {
	const form = "sum of per-iteration spans"

	single := &B{}
	if single.spanPerIteration() {
		t.Fatal("a run with no evictor and no request must use one span around the loop")
	}
	if strings.Contains(joined(single), form) {
		t.Errorf("the single-span form must not claim the per-iteration one: %q", joined(single))
	}

	warm := &B{perIter: true}
	if !warm.spanPerIteration() {
		t.Fatal("SpanPerIteration was requested and not honored")
	}
	if got := joined(warm); !strings.Contains(got, form) {
		t.Errorf("a per-iteration-span run must state the form: %q", got)
	} else if !strings.Contains(got, "uncontrolled") {
		t.Errorf("no evictor is installed, so cache state is still uncontrolled: %q", got)
	}

	evicted := &B{evict: func(*cuda.Stream) error { return nil }, evictName: "L2 overwrite"}
	if got := joined(evicted); !strings.Contains(got, form) {
		t.Errorf("an evictor forces the per-iteration form and must say so: %q", got)
	}
}
