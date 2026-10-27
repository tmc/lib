package egraph_test

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacollective"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cudasearch"
	"github.com/tmc/lib/nvidia/x/cudashape"
	"github.com/tmc/lib/nvidia/x/egraph"
)

type collectiveProfileEvaluator func(context.Context, *egraph.SearchState, egraph.Id) (*cudasearch.CandidatePlan, error)

func (f collectiveProfileEvaluator) Evaluate(ctx context.Context, state *egraph.SearchState, root egraph.Id) (*cudasearch.CandidatePlan, error) {
	return f(ctx, state, root)
}

func collectiveProfileWorkload(t *testing.T, inputs [][][]float32) cudasearch.WorkloadBindings {
	t.Helper()
	shape, err := cudashape.New(len(inputs[0][0]))
	if err != nil {
		t.Fatal(err)
	}
	w := cudasearch.WorkloadBindings{
		Name: "collective-f32-replay", Shape: shape, DType: cudair.Float32,
		Layout: "row_major", NumericalPolicy: cudasearch.NumericalPolicyExact,
		InputIdentities: make(map[string]string),
	}
	for rank, data := range inputs[0] {
		name := fmt.Sprintf("rank%d", rank)
		h := sha256.New()
		var word [4]byte
		for _, value := range data {
			binary.LittleEndian.PutUint32(word[:], math.Float32bits(value))
			h.Write(word[:])
		}
		w.InputNames = append(w.InputNames, name)
		w.InputIdentities[name] = hex.EncodeToString(h.Sum(nil))
	}
	if err := w.Validate(); err != nil {
		t.Fatal(err)
	}
	return w
}

// The caller's profile workload includes the actual admitted collective plan.
// Plan.ID includes communicator generation and all calls, excluding estimates.
// CodeDigest remains a separate synthetic policy-test context below.
func collectiveProfileBinding(plan *egraph.CollectivePlan, workload cudasearch.WorkloadBindings) string {
	workload.Name += "/plan:" + plan.ID()
	return workload.Key()
}

func TestCollectiveProfileBinding(t *testing.T) {
	for _, kind := range []string{"fresh", "expired", "generation", "plan", "workload"} {
		t.Run(kind, func(t *testing.T) {
			const generation = 7
			profileCalls := collectiveCalls(1, generation)
			calls := cloneCalls(profileCalls)
			if kind == "generation" {
				for rank := range calls {
					calls[rank][0].Generation++
				}
			}
			if kind == "plan" {
				for rank := range calls {
					calls[rank][0].EffectToken += 100
				}
			}
			inputs := collectiveInputs(1)
			profileWorkload := collectiveProfileWorkload(t, inputs)
			if kind == "workload" {
				inputs[0][1][0] += 13
			}
			workload := collectiveProfileWorkload(t, inputs)
			profilePlan, err := planCalls(t, profileCalls, collectiveComms(t, generation, nil), nil)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := profilePlan.Close(); err != nil {
					t.Error(err)
				}
			})
			comms := collectiveComms(t, cudacollective.Generation(calls[0][0].Generation), nil)
			quarantine := new(egraph.CollectiveQuarantine)
			plan, err := planCalls(t, calls, comms, quarantine)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := plan.Close(); err != nil {
					t.Error(err)
				}
			})
			targetWorkload := collectiveProfileBinding(plan, workload)
			snapshotWorkload := collectiveProfileBinding(profilePlan, profileWorkload)
			now := time.Date(2026, 10, 26, 12, 0, 0, 0, time.UTC)
			stamp := now.Add(-time.Second)
			if kind == "expired" {
				stamp = now.Add(-2 * time.Minute)
			}
			// Synthetic, asserted policy input only. This digest is not Plan.ID
			// or evidence of GPU code, native samples, latency or hardware.
			contextDigest := sha256.Sum256([]byte("collective-profile-policy-test-context/v1"))
			target := cudaprof.ProfileTarget{
				CodeDigest: hex.EncodeToString(contextDigest[:]), Region: "host-collective-policy",
				Workload: targetWorkload, Device: "synthetic-host-policy",
				CollectorMode: "synthetic-policy-test-only",
			}
			snapshot, err := cudaprof.NewProfileSnapshot(cudaprof.ProfileSnapshot{
				Version: 1, Timestamp: stamp, CodeDigest: target.CodeDigest,
				Region: target.Region, Workload: snapshotWorkload, Device: target.Device,
				CollectorMode: target.CollectorMode, Coverage: 1, ClockUncertaintyNs: 0,
				Metrics: map[string]int64{"rank_wait": 20}, MetricUnits: map[string]string{"rank_wait": "samples"},
			})
			if err != nil {
				t.Fatal(err)
			}
			policy := &cudasearch.ProfileActions{
				Snapshot: snapshot, Target: target, Time: now, MaxAge: time.Minute,
				MinCoverage: .5, MaxLosses: 0, MaxUncertaintyNs: 0,
				Priors: []cudasearch.MetricRulePrior{{Metric: "rank_wait", Unit: "samples", Minimum: 10, Rule: "z_profile_estimate"}},
			}
			g := egraph.New()
			root := g.Add(egraph.ENode{Op: "collective_policy_base"})
			state := egraph.NewSearchState(g, map[string]egraph.Id{"root": root})
			// These labels select modeled estimates, not communication rewrites.
			// The actual calls below must pass CheckCollectiveRewrite unchanged.
			rules := []*egraph.Rewrite{
				egraph.MustRewrite("a_static_estimate", "collective_policy_base", "static_estimate"),
				egraph.MustRewrite("z_profile_estimate", "collective_policy_base", "profile_estimate"),
			}
			var evaluated []string
			evaluator := collectiveProfileEvaluator(func(ctx context.Context, s *egraph.SearchState, _ egraph.Id) (*cudasearch.CandidatePlan, error) {
				label := "initial"
				if len(s.Derivation) != 0 {
					label = s.Derivation[len(s.Derivation)-1].RuleName
				}
				if len(evaluated) == 0 {
					// Admission must have been frozen before the first evaluator.
					// Repairing stale input here cannot revive its guidance; breaking
					// fresh input cannot withdraw an already accepted decision.
					policy.Snapshot.Timestamp = now
					policy.Snapshot.Workload = targetWorkload
					if kind == "fresh" {
						policy.Snapshot.Timestamp = now.Add(-time.Hour)
						policy.Snapshot.Workload = "changed-by-evaluator"
					}
				}
				evaluated = append(evaluated, label)
				variantCalls := cloneCalls(calls)
				for rank := range variantCalls {
					variantCalls[rank][0].Estimate = time.Millisecond
					if label == "z_profile_estimate" {
						variantCalls[rank][0].Estimate = 2 * time.Millisecond
					}
					if err := egraph.CheckCollectiveRewrite(calls[rank], variantCalls[rank]); err != nil {
						return nil, err
					}
				}
				variant, err := (egraph.CollectiveSchedule{Communicators: comms, Calls: variantCalls, Quarantine: quarantine}).Plan()
				if err != nil {
					return nil, err
				}
				if variant.ID() != plan.ID() {
					closeErr := variant.Close()
					return nil, fmt.Errorf("estimate changed plan identity: %s != %s; close: %v", variant.ID(), plan.ID(), closeErr)
				}
				receipt, replayErr := variant.Replay(ctx, cloneInputs(inputs))
				closeErr := variant.Close()
				if replayErr != nil || closeErr != nil {
					return nil, fmt.Errorf("collective replay: %v; close: %v", replayErr, closeErr)
				}
				want := sumOracle(inputs[0])
				if len(receipt.Steps) != 1 || len(receipt.Steps[0].Outputs) != len(calls) {
					return nil, fmt.Errorf("incomplete collective receipt: %+v", receipt)
				}
				for rank, output := range receipt.Steps[0].Outputs {
					if !slices.Equal(output, want) || receipt.Steps[0].Completions[rank] != calls[rank][0].Completion {
						return nil, fmt.Errorf("rank %d output %v, want %v", rank, output, want)
					}
				}
				return &cudasearch.CandidatePlan{ID: plan.ID() + "/" + label,
					Derivation: slices.Clone(s.Derivation), Cost: float64(10 - len(s.Derivation)),
					CostSource: cudasearch.CostEstimated, Validated: true}, nil
			})
			result, err := cudasearch.NewBeamSearch(cudasearch.BeamConfig{
				BeamWidth: 2, Profile: policy,
				Budget: cudasearch.SearchBudget{MaxStates: 2, MaxDepth: 1, MaxActions: 2, MaxTotalActions: 2, MaxEvaluations: 2, Timeout: 5 * time.Second},
			}).Search(context.Background(), state, rules, evaluator)
			if result != nil {
				defer result.Close()
			}
			if err != nil {
				t.Fatal(err)
			}
			wantRule := "a_static_estimate"
			if kind == "fresh" {
				wantRule = "z_profile_estimate"
			}
			decision := result.ProfileDecision
			if decision == nil || decision.ID == "" || decision.Target.Workload != targetWorkload || result.PolicyID.Options["profile_decision"] != decision.ID {
				t.Fatalf("missing attributable profile decision: %+v", result)
			}
			if kind == "fresh" {
				if result.ProfileFallback != "" || result.ProfileGuidance == nil || result.ProfileGuidance.Workload != targetWorkload || !result.ProfileGuidance.Timestamp.Equal(stamp) || !slices.Equal(decision.ActiveRules, []string{"z_profile_estimate"}) {
					t.Fatalf("fresh bound profile lost guidance: %+v", result)
				}
			} else {
				reason := "workload"
				if kind == "expired" {
					reason = "expired"
				}
				if !strings.Contains(result.ProfileFallback, reason) || result.ProfileGuidance != nil || decision.Fallback != result.ProfileFallback || len(decision.ActiveRules) != 0 {
					t.Fatalf("stale %s profile guided collective search: %+v", kind, result)
				}
			}
			if !slices.Equal(evaluated, []string{"initial", wantRule}) || result.Evaluations != 2 || result.Winner == nil || len(result.Winner.Derivation) != 1 || result.Winner.Derivation[0].RuleName != wantRule {
				t.Fatalf("evaluated=%v, result=%+v; want initial then %s", evaluated, result, wantRule)
			}
		})
	}
}
