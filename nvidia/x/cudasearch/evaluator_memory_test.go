package cudasearch

import (
	"errors"
	"strings"
	"testing"

	"github.com/tmc/lib/egraphcore"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func TestMockMetadataOwner(t *testing.T) {
	setupHostOnly(t)
	m := setupMock()
	ctx, work, err := searchWorkBudget(t.Context(), SearchBudget{MaxMemoryBytes: 1 << 20}, m)
	if err != nil {
		t.Fatal(err)
	}
	defer work.closeSearch()
	baseline := work.memory.live
	s := setupState()
	c, err := m.Evaluate(ctx, s, s.Roots["root"])
	if err != nil || c == nil || c.metadata == nil || c.metadataLedger != work.memory || work.memory.live <= baseline {
		t.Fatalf("candidate=%+v live=%d baseline=%d error=%v", c, work.memory.live, baseline, err)
	}
	if c.Receipt == nil || len(c.Receipt.Timings) != 2 || c.ID != "cand-input-0" || c.Digest == "" {
		t.Fatalf("metadata changed: %+v", c)
	}
	alias := *c
	c.Close()
	if work.memory.live != baseline || c.Receipt != nil || c.Expr != nil || c.Derivation != nil {
		t.Fatalf("candidate close left ownership: live=%d baseline=%d candidate=%+v", work.memory.live, baseline, c)
	}
	alias.Close()
	if work.memory.live != baseline {
		t.Fatal("copied owner released twice", work.memory.live)
	}
	work.closeSearch()
	if work.memory.live != 0 {
		t.Fatal("work owner leaked", work.memory.live)
	}
}

func TestMockMetadataPreflight(t *testing.T) {
	setupHostOnly(t)
	for _, large := range []bool{false, true} {
		t.Run(map[bool]string{false: "small", true: "large-derivation"}[large], func(t *testing.T) {
			m := setupMock()
			ctx, work, err := searchWorkBudget(t.Context(), SearchBudget{MaxMemoryBytes: 16 << 10}, m)
			if err != nil {
				t.Fatal(err)
			}
			defer work.closeSearch()
			baseline := work.memory.live
			s := setupState()
			if large {
				s.Derivation = make([]egraph.Action, 4096)
			}
			c, err := m.Evaluate(ctx, s, s.Roots["root"])
			if large {
				if c != nil || !errors.Is(err, ErrMemoryBudgetExceeded) || !strings.Contains(err.Error(), "mock metadata") {
					t.Fatalf("metadata allocation not refused: candidate=%+v error=%v", c, err)
				}
			} else {
				if err != nil || c == nil {
					t.Fatalf("small candidate refused: %+v %v", c, err)
				}
				c.Close()
			}
			if work.memory.live != baseline {
				t.Fatalf("metadata refusal/close leaked: live=%d baseline=%d", work.memory.live, baseline)
			}
		})
	}
}

func TestMockMetadataFailedScoring(t *testing.T) {
	setupHostOnly(t)
	m := setupMock()
	m.Scoring.TimingKind = cudatrial.TimingGPUBusy
	ctx, work, err := searchWorkBudget(t.Context(), SearchBudget{MaxMemoryBytes: 1 << 20}, m)
	if err != nil {
		t.Fatal(err)
	}
	defer work.closeSearch()
	baseline := work.memory.live
	s := setupState()
	c, err := m.Evaluate(ctx, s, s.Roots["root"])
	if c != nil || !errors.Is(err, ErrTimingKindMismatch) {
		t.Fatalf("candidate=%+v error=%v", c, err)
	}
	if work.memory.live != baseline {
		t.Fatalf("failed scoring retained receipt/fingerprint/extraction: live=%d baseline=%d", work.memory.live, baseline)
	}
}

type mockOpaqueFormat string

var mockOpaqueCalls int

func (f mockOpaqueFormat) String() string {
	mockOpaqueCalls++
	return string(f)
}

func TestMockFingerprintOpaqueFormatter(t *testing.T) {
	setupHostOnly(t)
	// Extraction already refuses this opaque named type under a cap. Test
	// the fingerprint sizer's independent guard without claiming that the
	// public evaluator reaches it after extraction.
	b := &memoryBudget{limit: 1 << 20}
	scope, err := egraphcore.NewMemoryScope(b)
	if err != nil {
		t.Fatal(err)
	}
	mockOpaqueCalls = 0
	err = reserveMockFingerprint(scope, &egraph.Expr{Op: "opaque", Data: mockOpaqueFormat("opaque")})
	if !errors.Is(err, ErrMemoryAccountingUnavailable) || mockOpaqueCalls != 0 {
		t.Fatalf("opaque formatter invoked without a bound: calls=%d error=%v", mockOpaqueCalls, err)
	}
	scope.Close()
	if b.live != 0 {
		t.Fatal("sizer scope leaked", b.live)
	}
}

func TestMockFingerprintPreflight(t *testing.T) {
	setupHostOnly(t)
	for _, large := range []bool{false, true} {
		t.Run(map[bool]string{false: "small", true: "large-fingerprint"}[large], func(t *testing.T) {
			m := setupMock()
			ctx, work, err := searchWorkBudget(t.Context(), SearchBudget{MaxMemoryBytes: 32 << 10}, m)
			if err != nil {
				t.Fatal(err)
			}
			defer work.closeSearch()
			baseline := work.memory.live
			data := "small"
			if large {
				data = strings.Repeat("x", 4096)
			}
			g := egraph.New()
			id := g.Add(egraph.ENode{Op: "input", Data: data})
			state := egraph.NewSearchState(g, map[string]egraph.Id{"root": id})
			c, err := m.Evaluate(ctx, state, id)
			if large {
				if c != nil || !errors.Is(err, ErrMemoryBudgetExceeded) || !strings.Contains(err.Error(), "mock metadata") {
					t.Fatalf("fingerprint not preflighted: candidate=%+v error=%v", c, err)
				}
			} else {
				if c == nil || err != nil {
					t.Fatalf("small fingerprint refused: %+v %v", c, err)
				}
				c.Close()
			}
			if work.memory.live != baseline {
				t.Fatalf("fingerprint lifetime leak: live=%d baseline=%d", work.memory.live, baseline)
			}
		})
	}
}
