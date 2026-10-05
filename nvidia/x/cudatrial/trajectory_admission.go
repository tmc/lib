package cudatrial

import (
	"fmt"
	"strconv"
	"time"
)

// This is the actual constructor's bounded, non-driver preflight. Passing it
// never establishes a device or changes PendingHardware.
func admitTrajectoryReceipts(r *TrajectoryReceipt, baseline, candidate Receipt) (time.Duration, time.Duration, error) {
	if !baseline.Complete || !candidate.Complete || baseline.Err != nil || candidate.Err != nil || !baseline.HasTiming(TimingSummedKernel) || !candidate.HasTiming(TimingSummedKernel) || baseline.Provenance["timer"] != "cuda_event" || candidate.Provenance["timer"] != "cuda_event" {
		return 0, 0, fmt.Errorf("%w: incomplete or non-event receipts", ErrUnqualifiedOutcome)
	}
	if baseline.Timings == nil || candidate.Timings == nil {
		return 0, 0, fmt.Errorf("%w: missing explicit event channels", ErrUnqualifiedOutcome)
	}
	if baseline.Provenance["input_digest"] != r.InputDigest || candidate.Provenance["input_digest"] != r.InputDigest || candidate.Provenance["artifact_digest"] != r.ArtifactDigest || candidate.Provenance["compiler_key"] != r.CompilerKey || baseline.Provenance["policy_id"] != r.PolicyID || candidate.Provenance["policy_id"] != r.PolicyID || baseline.Provenance["verified_correct"] != "true" || candidate.Provenance["verified_correct"] != strconv.FormatBool(r.VerifiedCorrect) {
		return 0, 0, fmt.Errorf("%w: execution provenance mismatch", ErrUnqualifiedOutcome)
	}
	r.BaselineArtifactDigest = baseline.Provenance["artifact_digest"]
	r.BaselineCompilerKey = baseline.Provenance["compiler_key"]
	if err := r.validateIdentity(); err != nil {
		return 0, 0, fmt.Errorf("%w: %w", ErrUnqualifiedOutcome, err)
	}
	a, err := baseline.DurationFor(TimingSummedKernel)
	if err != nil {
		return 0, 0, err
	}
	b, err := candidate.DurationFor(TimingSummedKernel)
	if err != nil {
		return 0, 0, err
	}
	if a <= 0 || b <= 0 {
		return 0, 0, fmt.Errorf("%w: nonpositive event duration", ErrUnqualifiedOutcome)
	}
	return a, b, nil
}
