package cudasearch

import (
	"fmt"
	"sort"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudatrial"
)

// ObserveCandidateTrajectory converts an executable evaluator's admitted ABBA
// trials and actual rewrite derivation into live replay memory evidence. Each
// arm uses its upper median event duration and retains the corresponding whole
// receipt. r supplies Domain, TargetArch and BottleneckState; execution identities
// and rules come from candidate and its receipts. The raw trials, outputs and
// artifacts must be retained for independent review. No promotion is implied.
func ObserveCandidateTrajectory(device *cuda.Device, r cudatrial.TrajectoryReceipt, candidate *CandidatePlan) (cudatrial.TrajectoryReceipt, error) {
	r = cudatrial.TrajectoryReceipt{Domain: r.Domain, TargetArch: r.TargetArch, BottleneckState: r.BottleneckState, PendingHardware: true}
	if candidate == nil || !candidate.Validated || candidate.CostSource != CostMeasured || len(candidate.Derivation) == 0 || len(candidate.Trials) < 4 || len(candidate.Trials) > 2048 || len(candidate.Trials)%4 != 0 {
		return r, cudatrial.ErrUnqualifiedOutcome
	}
	baseline, err := trajectoryArmMedian(candidate.Trials, "baseline")
	if err != nil {
		return r, err
	}
	observed, err := trajectoryArmMedian(candidate.Trials, "candidate")
	if err != nil {
		return r, err
	}
	if candidate.Cost != float64(observed.Timings[cudatrial.TimingSummedKernel]) {
		return r, cudatrial.ErrUnqualifiedOutcome
	}
	r.WorkloadKey = candidate.WorkloadKey
	r.InputDigest = observed.Provenance["input_digest"]
	r.ArtifactDigest = observed.Provenance["artifact_digest"]
	r.CompilerKey = observed.Provenance["compiler_key"]
	r.PolicyID = observed.Provenance["policy_id"]
	r.VerifiedCorrect = true
	r.AppliedRules = nil
	for _, action := range candidate.Derivation {
		r.AppliedRules = append(r.AppliedRules, action.RuleName)
	}
	return cudatrial.ObserveTrajectory(device, r, baseline, observed)
}

func trajectoryArmMedian(trials []cudatrial.Receipt, arm string) (cudatrial.Receipt, error) {
	var rows []cudatrial.Receipt
	for i, trial := range trials {
		want := "candidate"
		if i%4 == 0 || i%4 == 3 {
			want = "baseline"
		}
		if trial.Name != want {
			return cudatrial.Receipt{}, fmt.Errorf("%w: trajectory order is not ABBA", cudatrial.ErrUnqualifiedOutcome)
		}
		if trial.Name != arm {
			continue
		}
		if !trial.Complete || trial.Err != nil || trial.TimingKind != cudatrial.TimingSummedKernel || trial.Timings == nil || trial.Provenance["timer"] != "cuda_event" || trial.Provenance["verified_correct"] != "true" || trial.Timings[cudatrial.TimingSummedKernel] <= 0 {
			return cudatrial.Receipt{}, cudatrial.ErrUnqualifiedOutcome
		}
		if len(rows) > 0 {
			for _, key := range []string{"input_digest", "artifact_digest", "compiler_key", "policy_id", "device_uuid"} {
				if trial.Provenance[key] != rows[0].Provenance[key] {
					return cudatrial.Receipt{}, fmt.Errorf("%w: mixed trajectory provenance", cudatrial.ErrUnqualifiedOutcome)
				}
			}
		}
		rows = append(rows, trial)
	}
	if len(rows) < 2 || len(rows)*2 != len(trials) {
		return cudatrial.Receipt{}, fmt.Errorf("%w: unbalanced trajectory arms", cudatrial.ErrUnqualifiedOutcome)
	}
	sort.SliceStable(rows, func(i, j int) bool {
		return rows[i].Timings[cudatrial.TimingSummedKernel] < rows[j].Timings[cudatrial.TimingSummedKernel]
	})
	return rows[len(rows)/2], nil
}
