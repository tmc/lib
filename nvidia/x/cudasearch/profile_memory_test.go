package cudasearch

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/egraphcore"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/egraph"
)

const (
	profileSmallLimit = 1 << 20
	profileLargeLimit = 256 << 20
)

// memoryProfile returns actionProfile with pcs sampled PCs in two metrics.
func memoryProfile(pcs int) *ProfileActions {
	p := actionProfile()
	p.Snapshot.MetricUnits["stall"] = "samples"
	p.Snapshot.PCStalls = make(map[uint64]map[string]int64, pcs)
	for i := 0; i < pcs; i++ {
		p.Snapshot.PCStalls[uint64(16*i)] = map[string]int64{"reason": int64(i%5 + 1), "stall": 2}
	}
	return p
}

type profileRun struct {
	result  *SearchResult
	err     error
	calls   int
	ctx     context.Context
	book    *searchBookkeeping
	cancels int
}

// profileSearch runs one public strategy. mode "canceled" cancels the search
// from its second evaluation.
func profileSearch(t *testing.T, strategy, mode string, profile *ProfileActions, limit int64) *profileRun {
	t.Helper()
	state, rules := beamContractState()
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	run := new(profileRun)
	eval := beamEvaluatorFunc(func(ctx context.Context, s *egraph.SearchState, _ egraph.Id) (*CandidatePlan, error) {
		run.calls++
		run.ctx = ctx
		run.book = searchMemoryBudget(ctx).book
		if run.calls > 1 && mode == "canceled" {
			cancel()
			return nil, ctx.Err()
		}
		return &CandidatePlan{Cost: float64(10 - run.calls), Validated: true, Derivation: s.Derivation}, nil
	})
	budget := SearchBudget{MaxMemoryBytes: limit, CallbackMemoryBytes: 4096, MaxStates: 5, MaxDepth: 2, MaxEvaluations: 4}
	if strategy == "beam" {
		run.result, run.err = NewBeamSearch(BeamConfig{BeamWidth: 1, Budget: budget, Profile: profile}).Search(ctx, state, rules, eval)
	} else {
		run.result, run.err = NewMCTSSearch(MCTSConfig{Iterations: 4, RolloutDepth: 1, Seed: 42, Budget: budget, Profile: profile}).Search(ctx, state, rules, eval)
	}
	return run
}

var profileStrategies = []string{"beam", "mcts"}

func TestProfileMemoryPublicBound(t *testing.T) {
	setupHostOnly(t)
	for _, strategy := range profileStrategies {
		t.Run(strategy, func(t *testing.T) {
			t.Run("small", func(t *testing.T) {
				p := memoryProfile(3)
				want, err := p.Snapshot.Guidance(p.Target)
				if err != nil {
					t.Fatal(err)
				}
				unlimited, err := freezeBoundedProfile(p, nil, nil)
				if err != nil {
					t.Fatal(err)
				}
				id := unlimited.decision.ID
				for _, limit := range []int64{profileSmallLimit, 0} {
					run := profileSearch(t, strategy, "", p, limit)
					if run.err != nil || run.result == nil || run.calls == 0 {
						t.Fatalf("limit %d: result=%+v calls=%d err=%v", limit, run.result, run.calls, run.err)
					}
					r := run.result
					if r.ProfileFallback != "" || !reflect.DeepEqual(r.ProfileGuidance, want) || r.ProfileDecision.ID != id || !reflect.DeepEqual(r.ProfileDecision.ActiveRules, []string{"z"}) {
						t.Fatalf("limit %d: profile not preserved: %+v %+v", limit, r.ProfileDecision, r.ProfileGuidance)
					}
					if limit > 0 && (r.AccountedBytes <= 0 || r.AccountedBytes > limit) {
						t.Fatalf("accounted %d outside limit %d", r.AccountedBytes, limit)
					}
					r.Close()
				}
			})
			for _, kind := range []string{"valid", "stale", "mismatch"} {
				t.Run("large_"+kind, func(t *testing.T) {
					p := memoryProfile(4000)
					switch kind {
					case "stale":
						p.Snapshot.Timestamp = p.Time.Add(-time.Hour)
					case "mismatch":
						p.Target.Device = "other"
					}
					run := profileSearch(t, strategy, "", p, profileSmallLimit)
					if run.result != nil || run.calls != 0 || !errors.Is(run.err, ErrMemoryBudgetExceeded) {
						t.Fatalf("not refused before evaluation: result=%+v calls=%d err=%v", run.result, run.calls, run.err)
					}
					run = profileSearch(t, strategy, "", p, profileLargeLimit)
					if run.err != nil || run.result == nil || run.calls == 0 {
						t.Fatalf("large cap refused: result=%+v err=%v", run.result, run.err)
					}
					defer run.result.Close()
					r := run.result
					if r.AccountedBytes <= profileSmallLimit || r.AccountedBytes > profileLargeLimit {
						t.Fatalf("accounted %d does not cover the profile within %d", r.AccountedBytes, profileLargeLimit)
					}
					if kind == "valid" {
						if r.ProfileFallback != "" || len(r.ProfileGuidance.PCStalls) != 4000 || len(r.ProfileGuidance.HotPCs["reason"]) != 10 {
							t.Fatalf("valid profile not admitted: %q %+v", r.ProfileFallback, r.ProfileDecision)
						}
						return
					}
					want := map[string]string{"stale": "expired", "mismatch": "device"}[kind]
					if !strings.Contains(r.ProfileFallback, want) || r.ProfileGuidance != nil || r.ProfileDecision.Fallback != r.ProfileFallback || r.ProfileDecision.SnapshotSHA256 == "" {
						t.Fatalf("fallback=%q decision=%+v", r.ProfileFallback, r.ProfileDecision)
					}
					unlimited, err := freezeBoundedProfile(p, nil, nil)
					if err != nil {
						t.Fatal(err)
					}
					if r.ProfileDecision.ID != unlimited.decision.ID {
						t.Fatal("fallback identity differs from the unlimited decision")
					}
				})
			}
		})
	}
}

func TestProfileMemoryDetaches(t *testing.T) {
	setupHostOnly(t)
	for _, strategy := range profileStrategies {
		t.Run(strategy, func(t *testing.T) {
			p := memoryProfile(40)
			want, err := p.Snapshot.Guidance(p.Target)
			if err != nil {
				t.Fatal(err)
			}
			priors := append([]MetricRulePrior(nil), p.Priors...)
			state, rules := beamContractState()
			eval := beamEvaluatorFunc(func(_ context.Context, s *egraph.SearchState, _ egraph.Id) (*CandidatePlan, error) {
				p.Snapshot.Metrics["reason"] = 0
				p.Snapshot.MetricUnits["reason"] = "cycles"
				for _, counters := range p.Snapshot.PCStalls {
					counters["reason"] = 99
				}
				delete(p.Snapshot.PCStalls, 0)
				p.Priors[0].Rule = "other"
				return &CandidatePlan{Cost: 1, Validated: true, Derivation: s.Derivation}, nil
			})
			budget := SearchBudget{MaxMemoryBytes: profileSmallLimit, CallbackMemoryBytes: 4096, MaxStates: 2, MaxDepth: 2, MaxEvaluations: 2}
			var r *SearchResult
			if strategy == "beam" {
				r, err = NewBeamSearch(BeamConfig{BeamWidth: 1, Budget: budget, Profile: p}).Search(t.Context(), state, rules, eval)
			} else {
				r, err = NewMCTSSearch(MCTSConfig{Iterations: 2, RolloutDepth: 1, Seed: 1, Budget: budget, Profile: p}).Search(t.Context(), state, rules, eval)
			}
			if err != nil {
				t.Fatal(err)
			}
			defer r.Close()
			if !reflect.DeepEqual(r.ProfileGuidance, want) || !reflect.DeepEqual(r.ProfileDecision.Priors, priors) {
				t.Fatalf("source mutation reached the frozen profile: %+v", r.ProfileGuidance)
			}
		})
	}
}

func TestProfileMemoryLifetimes(t *testing.T) {
	setupHostOnly(t)
	for _, strategy := range profileStrategies {
		for _, mode := range []string{"normal", "canceled"} {
			t.Run(strategy+"/"+mode, func(t *testing.T) {
				without := profileSearch(t, strategy, mode, nil, profileSmallLimit)
				if without.result == nil {
					t.Fatal(without.err)
				}
				base := without.result.bookkeeping.Bytes()
				without.result.Close()

				run := profileSearch(t, strategy, mode, memoryProfile(8), profileSmallLimit)
				r := run.result
				if r == nil || r.ProfileGuidance == nil || (mode == "canceled") != errors.Is(run.err, ErrSearchCancelled) || (mode == "normal" && run.err != nil) {
					t.Fatalf("result=%+v err=%v", r, run.err)
				}
				if run.book == nil || run.book.scratch.Bytes() != 0 || searchMemoryBudget(run.ctx).book != nil {
					t.Fatal("profile scratch survived Search return")
				}
				owned := reflect.TypeFor[cudaprof.SearchGuidance]().Size() + reflect.TypeFor[ProfileDecision]().Size() + 8*(16+2*24)
				held := r.bookkeeping.Bytes()
				if held-base < int64(owned) {
					t.Fatalf("result owner holds %d bytes beyond %d; profile storage is not under it", held-base, base)
				}
				if r.ProfileGuidance.StallCounts["reason"] != 20 || len(r.ProfileGuidance.PCStalls) != 8 {
					t.Fatal("guidance unusable before Close")
				}
				for owner := r.clones; owner != nil; owner = owner.next {
					held += owner.memory.Bytes()
				}
				before := cloneAvailable(run.ctx, profileSmallLimit)
				alias := *r
				r.Close()
				after := cloneAvailable(run.ctx, profileSmallLimit)
				if after != before+held {
					t.Fatalf("Close released %d, want %d", after-before, held)
				}
				alias.Close()
				if cloneAvailable(run.ctx, profileSmallLimit) != after {
					t.Fatal("result alias released profile storage twice")
				}
			})
		}
	}
}

// TestProfileMemoryUnwind refuses the same profile at every reservation point.
func TestProfileMemoryUnwind(t *testing.T) {
	for _, kind := range []string{"valid", "stale", "mismatch", "missing"} {
		t.Run(kind, func(t *testing.T) {
			p := memoryProfile(6)
			switch kind {
			case "stale":
				p.Snapshot.Timestamp = p.Time.Add(-time.Hour)
			case "mismatch":
				p.Target.Region = "other"
			case "missing":
				p.Snapshot = nil
			}
			want, err := freezeBoundedProfile(p, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			refused, admitted := 0, 0
			for limit := int64(1); limit <= 1<<18; limit += 61 {
				b := &memoryBudget{limit: limit}
				scratch, err := egraphcore.NewMemoryScope(b)
				if err != nil {
					refused++
					continue
				}
				result, err := egraphcore.NewMemoryScope(b)
				if err != nil {
					scratch.Close()
					refused++
					continue
				}
				got, err := freezeBoundedProfile(p, scratch, result)
				switch {
				case err != nil:
					refused++
					if !errors.Is(err, ErrMemoryBudgetExceeded) || got.decision != nil || got.guidance != nil {
						t.Fatalf("limit %d: %+v %v", limit, got, err)
					}
				case got.decision.ID != want.decision.ID || got.fallback != want.fallback || len(got.ranks) != len(want.ranks):
					t.Fatalf("limit %d: bounded decision differs: %+v want %+v", limit, got.decision, want.decision)
				default:
					admitted++
					if scratch.Bytes() == 0 || result.Bytes() == 0 {
						t.Fatalf("limit %d: admitted with unaccounted storage", limit)
					}
					// The retained guidance copy: header and six PCs of two counters.
					if min := int64(reflect.TypeFor[cudaprof.SearchGuidance]().Size()+reflect.TypeFor[ProfileDecision]().Size()) + 6*(16+2*24); kind == "valid" && result.Bytes() < min {
						t.Fatalf("limit %d: result owner holds %d bytes, want at least %d", limit, result.Bytes(), min)
					}
				}
				scratch.Close()
				scratch.Close()
				result.Close()
				result.Close()
				if b.live != 0 {
					t.Fatalf("limit %d: %d bytes remain after both scopes closed", limit, b.live)
				}
			}
			if refused == 0 || admitted == 0 {
				t.Fatalf("sweep did not cross the bound: refused=%d admitted=%d", refused, admitted)
			}
		})
	}
}

func TestProfileMemoryNilScopes(t *testing.T) {
	if p, err := freezeBoundedProfile(nil, nil, nil); err != nil || p.decision != nil {
		t.Fatalf("nil config: %+v %v", p, err)
	}
	p := memoryProfile(3)
	got, err := freezeBoundedProfile(p, nil, nil)
	if err != nil || got.guidance == nil || got.decision.ID == "" {
		t.Fatalf("unlimited freeze: %+v %v", got, err)
	}
	want, err := p.Snapshot.Guidance(p.Target)
	if err != nil || !reflect.DeepEqual(got.guidance, want) {
		t.Fatalf("guidance differs from Guidance: %v", err)
	}
	if _, err := (*cudaprof.ProfileSnapshot)(nil).Guidance(p.Target); err == nil {
		t.Fatal("nil snapshot accepted")
	}
}

// TestProfileMemoryEncodingBound checks that the closed-form bounds cover the
// encodings they reserve, including strings that escape to six bytes per byte.
func TestProfileMemoryEncodingBound(t *testing.T) {
	hostile := "<&> \x00\xff\"\\" + strings.Repeat("\xfe", 20)
	for _, pcs := range []int{0, 1, 50} {
		p := memoryProfile(pcs)
		p.Snapshot.Region, p.Snapshot.Workload = hostile, hostile
		p.Snapshot.MetricUnits[hostile], p.Snapshot.Metrics[hostile] = hostile, 1
		p.Snapshot.Sampling = &cudaprof.SamplingProvenance{RunID: hostile, KernelName: hostile}
		p.Priors = append(p.Priors, MetricRulePrior{Metric: hostile, Unit: hostile, Rule: hostile})
		data, err := json.Marshal(p.Snapshot)
		if err != nil {
			t.Fatal(err)
		}
		if b := snapshotBound(p.Snapshot); b.err != nil || b.bytes < int64(len(data)) {
			t.Fatalf("snapshot bound %d < encoding %d (%v)", b.bytes, len(data), b.err)
		}
		unlimited, err := freezeBoundedProfile(p, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		d := unlimited.decision // the ID is part of the encoding
		d.Target.Device = hostile
		d.ActiveRules = append(d.ActiveRules, hostile)
		data, _ = json.Marshal(d)
		if b := decisionBound(d); b.err != nil || b.bytes < int64(len(data)) {
			t.Fatalf("decision bound %d < encoding %d (%v)", b.bytes, len(data), b.err)
		}
	}
}

// TestProfileMemoryLargeProfile admits profiles with more PCs than the
// generic policy encoder's visit limit could size.
func TestProfileMemoryLargeProfile(t *testing.T) {
	setupHostOnly(t)
	const pcs, limit = 250000, 4 << 30
	for _, strategy := range profileStrategies {
		for _, kind := range []string{"valid", "stale"} {
			t.Run(strategy+"/"+kind, func(t *testing.T) {
				p := memoryProfile(pcs)
				if kind == "stale" {
					p.Snapshot.Timestamp = p.Time.Add(-time.Hour)
				}
				run := profileSearch(t, strategy, "", p, limit)
				if run.err != nil || run.result == nil {
					t.Fatalf("large profile refused under a sufficient cap: %v", run.err)
				}
				defer run.result.Close()
				r := run.result
				if kind == "valid" && (len(r.ProfileGuidance.PCStalls) != pcs || len(r.ProfileGuidance.HotPCs["reason"]) != 10) || kind == "stale" && r.ProfileFallback == "" {
					t.Fatalf("fallback=%q guidance=%v", r.ProfileFallback, r.ProfileGuidance != nil)
				}
				unlimited, err := freezeBoundedProfile(p, nil, nil)
				if err != nil {
					t.Fatal(err)
				}
				if r.AccountedBytes > limit || r.ProfileDecision.ID != unlimited.decision.ID {
					t.Fatalf("accounted=%d id differs from unlimited", r.AccountedBytes)
				}
			})
		}
	}
}

// TestProfileMemoryConfigPrecedence keeps an invalid search memory error ahead
// of any profile storage refusal.
func TestProfileMemoryConfigPrecedence(t *testing.T) {
	for _, strategy := range profileStrategies {
		t.Run(strategy, func(t *testing.T) {
			state, rules := beamContractState()
			budget := SearchBudget{MaxMemoryBytes: profileSmallLimit, CallbackMemoryBytes: 4096}
			memory := &SearchMemory{TargetArch: "invalid"}
			eval := beamEvaluatorFunc(func(context.Context, *egraph.SearchState, egraph.Id) (*CandidatePlan, error) {
				t.Fatal("evaluated")
				return nil, nil
			})
			var err error
			if strategy == "beam" {
				_, err = NewBeamSearch(BeamConfig{Budget: budget, Profile: memoryProfile(4000), Memory: memory}).Search(t.Context(), state, rules, eval)
			} else {
				_, err = NewMCTSSearch(MCTSConfig{Budget: budget, Profile: memoryProfile(4000), Memory: memory}).Search(t.Context(), state, rules, eval)
			}
			if err == nil || errors.Is(err, ErrMemoryBudgetExceeded) || !strings.Contains(err.Error(), "search memory") {
				t.Fatalf("err=%v", err)
			}
		})
	}
}

// TestProfileMemoryEncodingReservation checks the scratch charge against
// json.Marshal's peak: its pooled buffer plus the returned copy, three times
// an escape-heavy output, and the text of every integer key.
func TestProfileMemoryEncodingReservation(t *testing.T) {
	var p *ProfileActions
	for _, pcs := range []int{2, 300} {
		p = memoryProfile(pcs)
		p.Snapshot.Region = strings.Repeat("\x00", 4096) // six output bytes each
		data, err := json.Marshal(p.Snapshot)
		if err != nil {
			t.Fatal(err)
		}
		scratch, err := egraphcore.NewMemoryScope(new(ledger))
		if err != nil {
			t.Fatal(err)
		}
		before := scratch.Bytes()
		if err := snapshotBound(p.Snapshot).reserve(scratch); err != nil {
			t.Fatal(err)
		}
		if got, want := scratch.Bytes()-before, int64(3*len(data)+maxKeyText*len(p.Snapshot.PCStalls)); got < want {
			t.Fatalf("%d PCs: reserved %d, want at least %d (3x%d output plus key text)", pcs, got, want, len(data))
		}
		scratch.Close()
	}
	unlimited, err := freezeBoundedProfile(p, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	d := unlimited.decision
	noID := *d
	noID.ID = ""
	if with, without := decisionBound(d).bytes, decisionBound(&noID).bytes; with-without < int64(len(d.ID)) {
		t.Fatal("decision bound omits its ID")
	}
}

// ledger is a Memory without a limit.
type ledger struct{ live int64 }

func (l *ledger) Reserve(n int64) error { l.live += n; return nil }
func (l *ledger) Release(n int64)       { l.live -= n }
