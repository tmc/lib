package cudatrial

import (
	"errors"
	"strings"
	"testing"
	"time"
)

func TestTrajectoryActualConstructorPreflight(t *testing.T) {
	for _, name := range []string{"valid", "incomplete", "native-error", "synthetic", "legacy-only", "missing-channel", "zero", "negative", "input", "artifact", "compiler", "policy", "baseline-correctness", "candidate-correctness", "baseline-artifact", "oversize-rule", "oversize-workload"} {
		t.Run(name, func(t *testing.T) {
			r := trajectoryFixture("tile", 50, true)
			r.PendingHardware = true
			makeReceipt := func() Receipt {
				return Receipt{Complete: true, Timings: map[TimingKind]time.Duration{TimingSummedKernel: 50 * time.Nanosecond}, Provenance: map[string]string{"timer": "cuda_event", "input_digest": r.InputDigest, "artifact_digest": r.ArtifactDigest, "compiler_key": r.CompilerKey, "policy_id": r.PolicyID, "verified_correct": "true"}}
			}
			base, candidate := makeReceipt(), makeReceipt()
			switch name {
			case "incomplete":
				candidate.Complete = false
			case "native-error":
				candidate.Err = errors.New("launch failed")
			case "synthetic":
				candidate.Provenance["timer"] = "mock"
			case "legacy-only":
				candidate.Timings = nil
				candidate.SummedKernel = time.Nanosecond
			case "missing-channel":
				delete(candidate.Timings, TimingSummedKernel)
				candidate.SummedKernel = time.Nanosecond
			case "zero":
				candidate.Timings[TimingSummedKernel] = 0
			case "negative":
				candidate.Timings[TimingSummedKernel] = -1
			case "input":
				candidate.Provenance["input_digest"] = strings.Repeat("b", 64)
			case "artifact":
				candidate.Provenance["artifact_digest"] = strings.Repeat("b", 64)
			case "compiler":
				candidate.Provenance["compiler_key"] = strings.Repeat("b", 64)
			case "policy":
				base.Provenance["policy_id"] = strings.Repeat("b", 64)
			case "baseline-correctness":
				base.Provenance["verified_correct"] = "false"
			case "candidate-correctness":
				candidate.Provenance["verified_correct"] = "false"
			case "baseline-artifact":
				base.Provenance["artifact_digest"] = "not-a-digest"
			case "oversize-rule":
				r.AppliedRules[0] = strings.Repeat("x", 257)
			case "oversize-workload":
				r.WorkloadKey = strings.Repeat("x", 1<<20)
			}
			a, b, err := admitTrajectoryReceipts(&r, base, candidate)
			if name == "valid" {
				if err != nil || a != 50*time.Nanosecond || b != a || !r.PendingHardware {
					t.Fatalf("preflight: %v %v %+v %v", a, b, r, err)
				}
			} else if !errors.Is(err, ErrUnqualifiedOutcome) {
				t.Fatalf("invalid %s accepted: %v", name, err)
			}
			if !r.PendingHardware {
				t.Fatal("pure preflight promoted hardware")
			}
		})
	}
}
