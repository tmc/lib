package cudasanitizer

import (
	"sync"
	"sync/atomic"
	"testing"
)

func TestSanitizerGating_SimplifyTraceDeduplication(t *testing.T) {
	g, _ := NewTraceNovelty(GatingSimplifyTrace, 4, 8)
	for _, step := range []struct {
		raw  []byte
		want bool
	}{
		{[]byte{0, 1, 0, 0}, true}, {[]byte{0, 200, 0, 0}, false}, {[]byte{0, 200, 1, 0}, true},
	} {
		got, err := g.Evaluate(step.raw)
		if err != nil || got != step.want {
			t.Fatal(got, err, step.want)
		}
	}
}

func TestSanitizerGatingUniqueQuantization(t *testing.T) {
	g, _ := NewTraceNovelty(GatingUniqueTrace, 4, 8)
	for _, step := range []struct {
		n    byte
		want bool
	}{{4, true}, {7, false}, {8, true}, {15, false}, {16, true}, {1, true}, {2, true}} {
		got, err := g.Evaluate([]byte{0, step.n, 0, 0})
		if err != nil || got != step.want {
			t.Fatal(step, got, err)
		}
	}
}

func TestSanitizerGating_CoverageIncreaseFiltering(t *testing.T) {
	g, _ := NewTraceNovelty(GatingCoverageIncrease, 4, 1)
	for _, step := range []struct {
		raw  []byte
		want bool
	}{
		{[]byte{0, 1, 0, 0}, true}, {[]byte{0, 1, 0, 0}, false}, {[]byte{0, 2, 0, 0}, true},
		{[]byte{0, 1, 0, 0}, false}, {[]byte{0, 2, 1, 0}, true}, {[]byte{0, 0, 1, 0}, false},
	} {
		got, err := g.Evaluate(step.raw)
		if err != nil || got != step.want {
			t.Fatal(got, err, step.want)
		}
	}
}

func TestControl_ZeroCoverageMaskFailsGating(t *testing.T) {
	for mode := GatingAllTrace; mode <= GatingCoverageIncrease; mode++ {
		g, _ := NewTraceNovelty(mode, 4, 1)
		for _, raw := range [][]byte{nil, {0, 0}, {0, 0, 0, 0}} {
			if ok, err := g.Evaluate(raw); ok || err == nil {
				t.Fatal(mode, ok, err)
			}
		}
		if ok, err := g.Evaluate([]byte{0, 1, 0, 0}); !ok || err != nil {
			t.Fatal("invalid input polluted state", mode, ok, err)
		}
	}
}

func TestTraceNoveltyCapacityAndConcurrency(t *testing.T) {
	g, _ := NewTraceNovelty(GatingUniqueTrace, 4, 1)
	var admitted atomic.Int32
	var wg sync.WaitGroup
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ok, err := g.Evaluate([]byte{0, 1, 0, 0})
			if err != nil {
				t.Error(err)
			}
			if ok {
				admitted.Add(1)
			}
		}()
	}
	wg.Wait()
	if admitted.Load() != 1 {
		t.Fatal("duplicate admitted concurrently", admitted.Load())
	}
	if ok, err := g.Evaluate([]byte{0, 2, 0, 0}); ok || err == nil {
		t.Fatal("capacity failed open")
	}
	if ok, err := g.Evaluate([]byte{0, 1, 0, 0}); ok || err != nil {
		t.Fatal("capacity changed duplicate state")
	}
	all, _ := NewTraceNovelty(GatingAllTrace, 4, 1)
	for i := 0; i < 3; i++ {
		if ok, err := all.Evaluate([]byte{1, 0, 0, 0}); !ok || err != nil {
			t.Fatal(ok, err)
		}
	}
}

func FuzzTraceNovelty(f *testing.F) {
	f.Add([]byte{0, 1, 4, 255})
	f.Fuzz(func(t *testing.T, raw []byte) {
		if len(raw) != 4 {
			return
		}
		for mode := GatingSimplifyTrace; mode <= GatingCoverageIncrease; mode++ {
			g, _ := NewTraceNovelty(mode, 4, 2)
			ok, err := g.Evaluate(raw)
			if err != nil {
				if ok {
					t.Fatal("error admitted")
				}
				continue
			}
			if !ok {
				t.Fatal("first nonzero trace rejected")
			}
			if ok, err := g.Evaluate(raw); ok || err != nil {
				t.Fatal("duplicate", ok, err)
			}
		}
	})
}
