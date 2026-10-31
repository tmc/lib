package cudadiagdemo

import (
	"fmt"
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

// TestNativeTimingMatchedRoundInputs runs the real executor serially on two
// rounds. Under launch-ready-v1 every arm of a round uploads the same input,
// while rounds differ, and request, attempt and operation identities stay
// distinct. The test uses only declarations its predecessor also has, so it
// runs unchanged on that source, where each arm's input followed its plan
// index and same_round fails.
func TestNativeTimingMatchedRoundInputs(t *testing.T) {
	ctx := testContext(t)
	f := &fakeNative{}
	e := f.executor()
	process := cudaworkload.ProcessGenID{PID: uint32(os.Getpid()), Generation: 1}
	e.annotators = []*cudaworkload.Annotator{cudaworkload.NewAnnotator(process, "gpudiagdemo-native", cudaworkload.WithRunID("native-test"), cudaworkload.WithCorrelationHook(&fakeCorrelation{}))}
	plan := planNativeRequests(2, 2)
	attempts, err := e.run(ctx, 1, plan, &nativeCleanup{})
	if err != nil || len(attempts) != len(plan) || f.opened != len(plan) {
		t.Fatalf("attempts=%d opened=%d err=%v, want %d", len(attempts), f.opened, err, len(plan))
	}
	rounds := map[int][]NativeTrial{}
	ops := map[cudaworkload.OpID]bool{}
	for i, a := range attempts {
		r, trial := plan[i], a.trial
		if !trial.Correct || trial.Correlation != "complete" || trial.Round != r.round || trial.Attempt != r.arm || trial.RequestID != fmt.Sprintf("request-%d", r.round) || trial.KernelLaunches != r.launches {
			t.Fatalf("attempt %d = %+v, want plan %+v", i, trial, r)
		}
		op := a.log.GroundTruth.OpID
		if op == 0 || ops[op] {
			t.Fatalf("attempt %d operation %d is missing or shared", i, op)
		}
		ops[op] = true
		rounds[trial.Round] = append(rounds[trial.Round], trial)
	}
	t.Run("same_round", func(t *testing.T) {
		for round := 1; round <= 2; round++ {
			var baseline NativeTrial
			for _, trial := range rounds[round] {
				if trial.Arm == "baseline" {
					baseline = trial
				}
			}
			for _, trial := range rounds[round] {
				if trial.Arm != "baseline" && trial.InputSHA256 != baseline.InputSHA256 {
					t.Errorf("%s: %s input %s, want baseline's %s", trial.RequestID, trial.Arm, trial.InputSHA256, baseline.InputSHA256)
				}
			}
		}
	})
	t.Run("distinct_rounds", func(t *testing.T) {
		for _, a := range rounds[1] {
			for _, b := range rounds[2] {
				if a.InputSHA256 == b.InputSHA256 || a.OutputSHA256 == b.OutputSHA256 {
					t.Errorf("%s/%s and %s/%s share input %s or output %s", a.RequestID, a.Arm, b.RequestID, b.Arm, a.InputSHA256, a.OutputSHA256)
				}
			}
		}
	})
}
