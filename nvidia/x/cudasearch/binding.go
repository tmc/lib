package cudasearch

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/tmc/lib/egraphcore"
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
	// ErrVerificationFailed is returned when a candidate fails numerical correctness verification.
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

// Objective defines an evaluator's optimization goal. ExecutableEvaluator
// supports ObjectiveMinLatency and treats an empty objective as minimum latency.
// Bindings accept only the objectives declared below. Each evaluator must
// reject an objective it does not implement before doing executable work.
type Objective string

const (
	ObjectiveMinLatency    Objective = "min_latency"
	ObjectiveMaxThroughput Objective = "max_throughput"
	ObjectiveProfileCost   Objective = "profile_cost"
)

// WorkloadBindings specifies concrete tensor shapes, layouts, precision, and input/weight identities.
type WorkloadBindings struct {
	Name             string            // Identifying workload name
	Shape            cudashape.Shape   // Concrete activation/input shape
	DType            cudair.DataType   // Activation tensor primitive data type
	Layout           string            // Memory layout ("row_major", "col_major", "strided"); empty leaves it unspecified
	InputNames       []string          // Required input/weight names, each covered by a nonempty identity below
	InputIdentities  map[string]string // Input content digests or immutable versions
	WeightIdentities map[string]string // Mapping of weight name to content digest/version
	NumericalPolicy  NumericalPolicy   // Precision/numerical requirements
	Objective        Objective         // Optimization goal interpreted by the evaluator
	ResetState       func() error      // Callback to reset mutable state (KV cache, scratch buffers) between runs
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
		return fmt.Errorf("%w: numerical policy must be specified", ErrMissingBinding)
	}
	switch b.DType {
	case cudair.Float32, cudair.Float16, cudair.BFloat16, cudair.Int32, cudair.Int64, cudair.Uint8, cudair.Int8:
	default:
		return fmt.Errorf("%w: unsupported dtype %d", ErrMissingBinding, b.DType)
	}
	switch b.Layout {
	case "", "row_major", "col_major", "strided":
	default:
		return fmt.Errorf("%w: unsupported layout %q", ErrMissingBinding, b.Layout)
	}
	switch b.Objective {
	case "", ObjectiveMinLatency, ObjectiveMaxThroughput, ObjectiveProfileCost:
	default:
		return fmt.Errorf("%w: unsupported objective %q", ErrMissingBinding, b.Objective)
	}
	for _, identities := range []map[string]string{b.InputIdentities, b.WeightIdentities} {
		keys := make([]string, 0, len(identities))
		for name := range identities {
			keys = append(keys, name)
		}
		sort.Strings(keys)
		for _, name := range keys {
			if name == "" || identities[name] == "" {
				return fmt.Errorf("%w: empty input or weight identity for %q", ErrMissingBinding, name)
			}
		}
	}
	seen := make(map[string]bool, len(b.InputNames))
	for _, name := range b.InputNames {
		if name == "" || seen[name] {
			return fmt.Errorf("%w: empty or duplicate required input name %q", ErrMissingBinding, name)
		}
		seen[name] = true
		input, weight := b.InputIdentities[name], b.WeightIdentities[name]
		if input == "" && weight == "" {
			return fmt.Errorf("%w: no identity for required input %q", ErrMissingBinding, name)
		}
		if input != "" && weight != "" && input != weight {
			return fmt.Errorf("%w: conflicting identities for required input %q", ErrMissingBinding, name)
		}
	}
	return nil
}

// WorkloadIdentity converts bindings to the cudacache.WorkloadIdentity representation.
func (b WorkloadBindings) WorkloadIdentity() cudacache.WorkloadIdentity {
	dims := b.Shape.Dims()
	return cudacache.WorkloadIdentity{
		BindingsKey:   b.Key(),
		ProgramID:     b.Name,
		ShapeDims:     dims,
		NumericalMode: string(b.NumericalPolicy),
	}
}

// Key returns a canonical SHA-256 digest string for the workload bindings.
// Dtypes are encoded numerically, including invalid values that Validate rejects.
// Keys from the earlier string-based dtype encoding are not reused.
func (b WorkloadBindings) Key() string {
	h := sha256.New()
	write := func(s string) {
		var size [8]byte
		binary.LittleEndian.PutUint64(size[:], uint64(len(s)))
		h.Write(size[:])
		h.Write([]byte(s))
	}
	write("cudasearch-workload-bindings-v2")
	write(b.Name)
	write(fmt.Sprintf("%v", b.Shape.Dims()))
	write("dtype")
	write(fmt.Sprint(int(b.DType)))
	write(b.Layout)
	write(string(b.NumericalPolicy))
	write(string(b.Objective))
	write(fmt.Sprintf("input-names:%d", len(b.InputNames)))
	for _, name := range b.InputNames {
		write(name)
	}
	inputKeys := make([]string, 0, len(b.InputIdentities))
	for name := range b.InputIdentities {
		inputKeys = append(inputKeys, name)
	}
	sort.Strings(inputKeys)
	write(fmt.Sprintf("input-identities:%d", len(inputKeys)))
	for _, name := range inputKeys {
		write(name)
		write(b.InputIdentities[name])
	}

	keys := make([]string, 0, len(b.WeightIdentities))
	for k := range b.WeightIdentities {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	write(fmt.Sprintf("weight-identities:%d", len(keys)))
	for _, k := range keys {
		write(k)
		write(b.WeightIdentities[k])
	}
	return hex.EncodeToString(h.Sum(nil))
}

// CostSource distinguishes host estimates from measured executable costs.
// The zero value is unspecified and cannot be compared with either known source.
type CostSource string

const (
	CostEstimated CostSource = "estimated"
	CostMeasured  CostSource = "measured"
)

// CandidatePlan represents an admitted candidate with its derivation history and score.
type CandidatePlan struct {
	WorkloadKey        string          // Bound workload identity recorded by ExecutableEvaluator.
	InputKey           string          // Declared input content identities, or empty when unavailable.
	ConfirmationOf     string          // Screening executable ID when produced by Confirm.
	VerificationPolicy OutputPolicy    // Numerical policy actually used by ExecutableEvaluator.
	ID                 string          // Unique identifier for the candidate
	Expr               *egraph.Expr    // Extracted AST expression
	Derivation         []egraph.Action // Replayable sequence of actions from initial state
	Cost               float64         // Heuristic or evaluated cost score (lower is better)
	CostSource         CostSource
	Receipt            *cudatrial.Receipt          // Timing and qualification receipt from evaluation (if evaluated)
	Trials             []cudatrial.Receipt         // All interleaved baseline/candidate trials, detached from the evaluator.
	Generation         int                         // Generation or epoch sequence number
	ScoringEpoch       string                      // Scoring model version; empty when no profile scoring model is used.
	Validated          bool                        // True if validated against reference or held-out tests
	Digest             string                      // Optional structural plan digest; never a numerical acceptance criterion.
	Profile            *ProfileEvidence            // Detached accepted observations; never interpreted as latency.
	Comparison         map[string]OutputComparison // Observed numerical deviations, not output hashes
	Preparation        []PreparationPhase          // Host preparation, separate from device Cost
	extraction         *egraph.Extraction
	extractionLedger   *memoryBudget
	metadata           *egraphcore.MemoryScope
	metadataLedger     *memoryBudget
}

// Close releases this candidate's owned extraction and metadata reservations once and
// invalidates the owned expression, synthetic receipt, copied metadata and their
// aliases. Caller-supplied expressions
// without an extraction owner are borrowed and left unchanged. Close does not
// close an executable, evaluator or external resource. A candidate retained in
// SearchResult remains owned until its candidate or result is closed.
// Close must not race with use or another Close.
func (c *CandidatePlan) Close() {
	if c == nil {
		return
	}
	if c.extraction != nil {
		c.extraction.Close()
		c.extraction = nil
		c.extractionLedger = nil
		c.Expr = nil
	}
	if c.metadata != nil {
		c.metadata.Close()
		c.metadata = nil
		c.metadataLedger = nil
		c.Derivation = nil
		c.Receipt = nil
		c.Profile = nil
		c.ID = ""
		c.Digest = ""
	}
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
	CandidateID        string
	Generation         int
	Receipt            cudatrial.Receipt
	WorkloadKey        string
	PolicyID           cudacache.SearchPolicyID
	ConfirmationID     string
	HeldOutWorkloadKey string
	HeldOutReceipt     cudatrial.Receipt
	ScreeningTrials    []cudatrial.Receipt
	HeldOutTrials      []cudatrial.Receipt
	Comparison         map[string]OutputComparison
	HeldOutComparison  map[string]OutputComparison
	VerificationPolicy OutputPolicy
}
