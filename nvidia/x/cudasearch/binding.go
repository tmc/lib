package cudasearch

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudashape"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

var (
	// ErrNoAdmittedPlan is returned when no legal or qualified candidate plan was found within budget.
	ErrNoAdmittedPlan = errors.New("cudasearch: no admitted candidate plan within search budget")
	// ErrBudgetExhausted is returned when search budgets are exceeded before finding an admitted candidate.
	ErrBudgetExhausted = errors.New("cudasearch: search budget exhausted")
	// ErrMissingBinding is returned when a required workload binding parameter is missing or invalid.
	ErrMissingBinding = errors.New("cudasearch: missing required workload binding")
	// ErrVerificationFailed is returned when a candidate fails correctness verification or output digests diverge.
	ErrVerificationFailed = errors.New("cudasearch: candidate failed numerical correctness verification")
	// ErrStaleGeneration is returned when attempting to promote a candidate from an older generation/epoch.
	ErrStaleGeneration = errors.New("cudasearch: candidate generation is stale")
	// ErrTimingKindMismatch indicates candidate comparison attempted across incompatible timing kinds.
	ErrTimingKindMismatch = errors.New("cudasearch: timing kind mismatch between evaluation receipts")
	// ErrSearchCancelled is returned when search is interrupted via context cancellation.
	ErrSearchCancelled = errors.New("cudasearch: search cancelled")
	// ErrMemoryBudgetExceeded is returned when the accounted search state memory budget is exceeded.
	ErrMemoryBudgetExceeded = errors.New("cudasearch: search state memory budget exceeded")
)

// NumericalPolicy identifies the numerical tolerance and precision rules for candidate evaluation.
type NumericalPolicy string

const (
	NumericalPolicyExact     NumericalPolicy = "exact"
	NumericalPolicyFP32      NumericalPolicy = "fp32"
	NumericalPolicyBF16      NumericalPolicy = "bf16"
	NumericalPolicyFP16      NumericalPolicy = "fp16"
	NumericalPolicyRMSNormQ8 NumericalPolicy = "rmsnorm-q8"
	NumericalPolicySwiGLUQ8  NumericalPolicy = "swiglu-q8_1"
)

// Objective defines the optimization goal for candidate ranking.
type Objective string

const (
	ObjectiveMinLatency    Objective = "min_latency"
	ObjectiveMaxThroughput Objective = "max_throughput"
	ObjectiveProfileCost   Objective = "profile_cost"
)

// WorkloadBindings specifies concrete tensor shapes, layouts, precision, and input/weight identities.
type WorkloadBindings struct {
	Name            string            // Identifying workload name
	Shape           cudashape.Shape   // Concrete activation/input shape
	DType           cudair.DataType   // Activation tensor primitive data type
	Layout          string            // Memory layout ("row_major", "col_major", "strided")
	InputNames      []string          // Expected input identities
	WeightIdentities map[string]string // Mapping of weight name to content digest/version
	NumericalPolicy NumericalPolicy   // Precision/numerical requirements
	Objective       Objective         // Target optimization objective
	ExpectedDigests map[int]string    // Expected SHA-256 logit digests per sequence length N (e.g. 2..2048)
	ResetState      func() error      // Callback to reset mutable state (KV cache, scratch buffers) between runs
}

// Validate checks that all required fields are present and valid.
func (b WorkloadBindings) Validate() error {
	if b.Name == "" {
		return fmt.Errorf("%w: name must not be empty", ErrMissingBinding)
	}
	if b.Shape.Rank() == 0 && len(b.InputNames) == 0 {
		// A rank 0 shape without input names is missing dimensions
		return fmt.Errorf("%w: shape dimensions or input names required", ErrMissingBinding)
	}
	if b.NumericalPolicy == "" {
		return fmt.Errorf("%w: numerical policy must not be specified", ErrMissingBinding)
	}
	return nil
}

// WorkloadIdentity converts bindings to the cudacache.WorkloadIdentity representation.
func (b WorkloadBindings) WorkloadIdentity() cudacache.WorkloadIdentity {
	dims := b.Shape.Dims()
	return cudacache.WorkloadIdentity{
		ProgramID:     b.Name,
		ShapeDims:     dims,
		NumericalMode: string(b.NumericalPolicy),
	}
}

// Key returns a canonical SHA-256 digest string for the workload bindings.
func (b WorkloadBindings) Key() string {
	h := sha256.New()
	write := func(s string) {
		h.Write([]byte(s))
		h.Write([]byte{0})
	}
	write(b.Name)
	write(fmt.Sprintf("%v", b.Shape.Dims()))
	write(b.DType.String())
	write(b.Layout)
	write(string(b.NumericalPolicy))
	write(string(b.Objective))

	keys := make([]string, 0, len(b.WeightIdentities))
	for k := range b.WeightIdentities {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		write(k)
		write(b.WeightIdentities[k])
	}
	return hex.EncodeToString(h.Sum(nil))
}

// CandidatePlan represents an admitted candidate with its derivation history and score.
type CandidatePlan struct {
	ID          string            // Unique identifier for the candidate
	Expr        *egraph.Expr      // Extracted AST expression
	Derivation  []egraph.Action   // Replayable sequence of actions from initial state
	Cost        float64           // Heuristic or evaluated cost score (lower is better)
	Receipt     *cudatrial.Receipt// Timing and qualification receipt from evaluation (if evaluated)
	Generation  int               // Generation or epoch sequence number
	Validated   bool              // True if validated against reference or held-out tests
	Digest      string            // Numerical output digest (e.g. SHA-256)
	ProfileCost float64           // Attributed stall/profile penalty cost
}

// Fingerprint returns a stable structural identifier for the candidate's AST and derivation.
func (c CandidatePlan) Fingerprint() string {
	if c.Expr != nil {
		return c.Expr.Fingerprint()
	}
	var sb strings.Builder
	for _, a := range c.Derivation {
		sb.WriteString(a.Key())
		sb.WriteString(";")
	}
	return sb.String()
}

// PromotionProposal encapsulates a qualification receipt and executable generation
// for an admitted candidate, proposed to execution owners for boundary switching.
type PromotionProposal struct {
	CandidateID string
	Generation  int
	Receipt     cudatrial.Receipt
	WorkloadKey string
	PolicyID    cudacache.SearchPolicyID
	Digest      string
	Promoted    bool
	Reason      string
}
