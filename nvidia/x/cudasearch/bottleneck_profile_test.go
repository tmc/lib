package cudasearch

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/cuptibench"
	"github.com/tmc/lib/nvidia/x/cuptirange"
	"github.com/tmc/lib/nvidia/x/egraph"
)

// This is a synthetic integrity fixture. Host controls exercise admission and
// policy wiring, and do not establish collection or live hardware observation.
func profileAdmissionFixture() (BottleneckProfile, *cuptibench.Result) {
	id := strings.Repeat("a", 64)
	p := BottleneckProfile{WorkloadKey: id, InputDigest: id, ArtifactDigest: id, CompilerKey: id}
	r := &cuptibench.Result{Operations: 1, Metadata: map[string]string{"mode": "whole", "capture_id": "fixture", "preparation": "host fixture", "cupti_version": "fixture", "cuda_driver_version": "fixture"}, Quality: cuptirange.CollectionQuality{PassesStarted: 1, PassesStopped: 1, AllPassesSubmitted: true, Decoded: true}}
	r.Ranges = []cuptibench.Range{{Kind: "operation", Inclusive: true}}
	for i, name := range BottleneckMetrics() {
		r.Ranges[0].Metrics = append(r.Ranges[0].Metrics, cuptibench.Metric{Name: name, Unit: "percent", Value: []float64{80, 20, 60}[i]})
	}
	return p, r
}

func TestSearchProfileTypedNilEvaluatorRefused(t *testing.T) {
	var evaluator *ExecutableEvaluator
	memory := &SearchMemory{Profile: new(BottleneckProfile)}
	for _, strategy := range []string{"beam", "mcts"} {
		t.Run(strategy, func(t *testing.T) {
			state, _ := beamContractState()
			var err error
			if strategy == "beam" {
				_, err = NewBeamSearch(BeamConfig{Memory: memory}).Search(context.Background(), state, []*egraph.Rewrite{}, evaluator)
			} else {
				_, err = NewMCTSSearch(MCTSConfig{Memory: memory}).Search(context.Background(), state, []*egraph.Rewrite{}, evaluator)
			}
			if err == nil || !strings.Contains(err.Error(), "bound executable evaluator") {
				t.Fatalf("typed-nil evaluator was not refused at profile seam: %v", err)
			}
		})
	}
}

func TestBottleneckCaptureAdmission(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*BottleneckProfile, *cuptibench.Result)
	}{
		{"positive", func(*BottleneckProfile, *cuptibench.Result) {}},
		{"identity", func(p *BottleneckProfile, _ *cuptibench.Result) { p.CompilerKey = "unknown" }},
		{"loss", func(_ *BottleneckProfile, r *cuptibench.Result) { r.Quality.DroppedRanges = 1 }},
		{"decode", func(_ *BottleneckProfile, r *cuptibench.Result) { r.Quality.Decoded = false }},
		{"pass", func(_ *BottleneckProfile, r *cuptibench.Result) { r.Quality.PassesStopped = 0 }},
		{"ambiguous", func(_ *BottleneckProfile, r *cuptibench.Result) { r.Ranges = append(r.Ranges, r.Ranges[0]) }},
		{"unit", func(_ *BottleneckProfile, r *cuptibench.Result) { r.Ranges[0].Metrics[0].Unit = "bytes" }},
		{"additive", func(_ *BottleneckProfile, r *cuptibench.Result) { r.Ranges[0].Metrics[0].Additive = true }},
		{"nonfinite", func(_ *BottleneckProfile, r *cuptibench.Result) { r.Ranges[0].Metrics[0].Value = math.NaN() }},
		{"counter-order", func(_ *BottleneckProfile, r *cuptibench.Result) {
			r.Ranges[0].Metrics[0].Name = r.Ranges[0].Metrics[1].Name
		}},
		{"unsubmitted", func(_ *BottleneckProfile, r *cuptibench.Result) { r.Quality.AllPassesSubmitted = false }},
		{"missing-tool", func(_ *BottleneckProfile, r *cuptibench.Result) { delete(r.Metadata, "cupti_version") }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, r := profileAdmissionFixture()
			tt.mutate(&p, r)
			m, err := admitBottleneckCapture(p, r)
			if tt.name == "positive" {
				if err != nil || ClassifyBottleneck(m).StateName != cudatrial.MemoryBandwidthLimited {
					t.Fatalf("metrics=%+v err=%v", m, err)
				}
			} else if !errors.Is(err, cudatrial.ErrUnqualifiedOutcome) {
				t.Fatalf("accepted capture: %v", err)
			}
		})
	}
}

func TestBottleneckProfileWithoutHardware(t *testing.T) {
	p, _ := profileAdmissionFixture()
	p, raw, err := CollectBottleneckProfile(context.Background(), nil, p, cuptibench.Workload{})
	if !errors.Is(err, cudatrial.ErrUnqualifiedOutcome) || raw != nil || !p.State.PendingHardware || p.HardwareObserved() {
		t.Fatalf("nil device admitted: %+v %v", p, err)
	}
}

func TestSearchProfileIdentityAndMutation(t *testing.T) {
	p, r := profileAdmissionFixture()
	p.Metrics, _ = admitBottleneckCapture(p, r)
	p.State = ClassifyBottleneck(p.Metrics)
	p.State.PendingHardware = false
	p.TargetArch = "sm_121"
	// White-box policy control only: a host fixture cannot call the constructor
	// to obtain a seal, and this value must never be exported as native evidence.
	p.ID = p.digest()
	p.seal = p.ID
	m := searchMemoryFixture(t)
	m.Profile, m.WorkloadKey, m.InputDigest = &p, p.WorkloadKey, p.InputDigest
	policy := new(profileActions)
	if err := freezeSearchMemory(m, policy); err != nil {
		t.Fatal(err)
	}
	if policy.memory.State.PendingHardware || policy.memory.ProfileID != p.ID {
		t.Fatal("search lost admitted profile identity")
	}
	for _, kind := range []string{"serialized", "mutation", "workload", "input", "target", "family"} {
		t.Run(kind, func(t *testing.T) {
			copy := p
			config := *m
			config.Profile = &copy
			switch kind {
			case "serialized":
				b, _ := json.Marshal(p)
				copy = BottleneckProfile{}
				json.Unmarshal(b, &copy)
			case "mutation":
				copy.CaptureID = "different"
			case "workload":
				config.WorkloadKey = strings.Repeat("b", 64)
			case "input":
				config.InputDigest = strings.Repeat("b", 64)
			case "target":
				config.TargetArch = "sm_100"
			case "family":
				config.Family = cudatrial.ComputeThroughputLimited
			}
			if err := freezeSearchMemory(&config, new(profileActions)); err == nil {
				t.Fatal("accepted incompatible or mutated profile")
			}
		})
	}
}

func TestSearchProfileRequiresBoundEvaluator(t *testing.T) {
	bindings := WorkloadBindings{Name: "profile-fixture", InputNames: []string{"x"}, NumericalPolicy: NumericalPolicyExact}
	m := &SearchMemory{Profile: new(BottleneckProfile), WorkloadKey: bindings.Key()}
	if err := validateMemoryEvaluator(m, &ExecutableEvaluator{Workload: bindings}); err != nil {
		t.Fatal(err)
	}
	if err := validateMemoryEvaluator(m, &ExecutableEvaluator{Workload: WorkloadBindings{Name: "other"}}); err == nil {
		t.Fatal("accepted other workload")
	}
	if err := validateMemoryEvaluator(m, &MockEvaluator{Workload: bindings}); err == nil {
		t.Fatal("accepted mock evaluator")
	}
	var nilEvaluator *ExecutableEvaluator
	if err := validateMemoryEvaluator(m, nilEvaluator); err == nil {
		t.Fatal("accepted typed-nil evaluator")
	}
}
