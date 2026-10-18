package cudacache

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/tmc/lib/nvidia/x/cudatrial"
)

// ReceiptSchemaVersion is the persisted JSON schema version for cached search receipts.
const ReceiptSchemaVersion = 1

type receiptAlias SearchReceipt

type persistedReceipt struct {
	SchemaVersion int `json:"schema_version"`
	receiptAlias
}

// SearchReceipt records the evaluation outcome of an arm or candidate under a specific search policy.
// It explicitly references the binary artifact's digest and retains search provenance,
// but remains distinct from the artifact storage itself.
type SearchReceipt struct {
	PolicyID     SearchPolicyID       `json:"policy_id"`
	WorkloadKey  string               `json:"workload_key"`
	CandidateID  string               `json:"candidate_id"`
	ArtifactKey  string               `json:"artifact_key"` // References CompilerCacheKey or Artifact.Digest
	Complete     bool                 `json:"complete"`
	Started      time.Time            `json:"started"`
	Finished     time.Time            `json:"finished"`
	TimingKind   cudatrial.TimingKind `json:"timing_kind"`
	Timings      map[string]int64     `json:"timings"`    // In nanoseconds
	RankScore    float64              `json:"rank_score"` // Lower is better or higher is better per scoring epoch
	Epoch        string               `json:"epoch"`      // Scoring epoch when evaluated
	ErrorMessage string               `json:"error_message,omitempty"`
}

// Validate verifies the internal consistency of a SearchReceipt.
func (r *SearchReceipt) Validate() error {
	if r == nil {
		return errors.New("cudacache: nil receipt")
	}
	if r.PolicyID.Strategy == "" {
		return errors.New("cudacache: receipt missing policy strategy")
	}
	if r.WorkloadKey == "" {
		return errors.New("cudacache: receipt missing workload key")
	}
	if r.CandidateID == "" {
		return errors.New("cudacache: receipt missing candidate id")
	}
	if r.ArtifactKey == "" {
		return errors.New("cudacache: receipt missing artifact key")
	}
	if math.IsNaN(r.RankScore) || math.IsInf(r.RankScore, 0) {
		return errors.New("cudacache: nonfinite receipt rank")
	}
	for kind, ns := range r.Timings {
		if ns < 0 {
			return fmt.Errorf("cudacache: negative timing for %q", kind)
		}
	}
	if !r.Started.IsZero() && !r.Finished.IsZero() && r.Finished.Before(r.Started) {
		return errors.New("cudacache: receipt finishes before it starts")
	}
	return nil
}

// DurationFor returns the duration for the specified timing kind.
func (r *SearchReceipt) DurationFor(kind cudatrial.TimingKind) (time.Duration, error) {
	if r == nil {
		return 0, errors.New("cudacache: nil receipt")
	}
	if r.Timings == nil {
		return 0, fmt.Errorf("cudacache: no timings in receipt")
	}
	ns, ok := r.Timings[string(kind)]
	if !ok {
		return 0, fmt.Errorf("cudacache: unmeasured timing kind %q", kind)
	}
	return time.Duration(ns), nil
}

// IsStale reports whether this receipt was evaluated under a different scoring epoch
// than currentEpoch.
func (r *SearchReceipt) IsStale(currentEpoch string) bool {
	if r == nil {
		return true
	}
	return r.Epoch != currentEpoch
}

// MarshalJSON serializes the SearchReceipt with ReceiptSchemaVersion.
func (r SearchReceipt) MarshalJSON() ([]byte, error) {
	return json.Marshal(persistedReceipt{
		SchemaVersion: ReceiptSchemaVersion,
		receiptAlias:  receiptAlias(r),
	})
}
