package cudaregress

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/cudatrial"
)

// DefaultStoppingRules returns standard conservative regression evaluation thresholds.
func DefaultStoppingRules() StoppingRules {
	return StoppingRules{
		MaxTrials:              10,
		RelativeThreshold:      0.10, // 10% relative regression
		NoiseNanos:             0,
		MinSamples:             3,
		MinEffectSize:          0.05,
		OracleTolerance:        1e-6,
		MaxConsecutiveFailures: 3,
		HaltOnDrift:            true,
	}
}

// ComputeFileDigest returns the hex-encoded SHA-256 digest of the specified file.
func ComputeFileDigest(path string) (string, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return "", 0, fmt.Errorf("%w: %s", ErrArtifactMissing, path)
		}
		return "", 0, err
	}
	defer f.Close()

	h := sha256.New()
	n, err := io.Copy(h, f)
	if err != nil {
		return "", 0, err
	}
	return hex.EncodeToString(h.Sum(nil)), n, nil
}

// NewPlanFromIncident derives a bounded, replayable ExperimentPlan from an admitted incident.
// If the incident was not admitted by its evidence contract, plan generation is visibly refused.
func NewPlanFromIncident(inc Incident, opts PlanOptions) (*ExperimentPlan, *PlanReceipt, error) {
	if !inc.Admitted {
		reason := inc.WithheldReason
		if reason == "" {
			reason = "incident evidence not admitted"
		}
		return nil, nil, fmt.Errorf("%w: %s", ErrIncidentNotAdmitted, reason)
	}

	if inc.Diagnosis.Mechanism == "withheld" || (!inc.Diagnosis.Admitted && inc.Diagnosis.Mechanism != "") {
		reason := inc.Diagnosis.WithheldReason
		if reason == "" {
			reason = "diagnosis withheld"
		}
		return nil, nil, fmt.Errorf("%w: diagnosis not admitted (%s)", ErrIncidentNotAdmitted, reason)
	}

	if inc.PrimaryFinding.Admission.Status == "withheld" || (!inc.PrimaryFinding.Admission.Admitted && inc.PrimaryFinding.Rule != "") {
		reason := inc.PrimaryFinding.Admission.Reason
		if reason == "" {
			reason = "primary finding withheld"
		}
		return nil, nil, fmt.Errorf("%w: primary finding not admitted (%s)", ErrIncidentNotAdmitted, reason)
	}

	incBytes, _ := json.Marshal(inc)
	incHash := sha256.Sum256(incBytes)
	incDigest := hex.EncodeToString(incHash[:])

	order := opts.Order
	if order == "" {
		order = "ABBA"
	}
	rounds := opts.Rounds
	if rounds <= 0 {
		rounds = 5
	}
	warmup := opts.WarmupRuns
	if warmup < 0 {
		warmup = 2
	}

	scope := opts.ExecutionScope
	if scope.Channel == "" {
		scope.Channel = cudatrial.TimingWallElapsed
	}

	devCfg := DeviceConfig{
		DeviceIndex:         inc.ObservedEnvironment.DeviceIndex,
		DeviceUUID:          inc.ObservedEnvironment.DeviceUUID,
		Model:               inc.ObservedEnvironment.Model,
		ComputeCapability:   inc.ObservedEnvironment.ComputeCapability,
		DriverVersion:       inc.ObservedEnvironment.Driver,
		CUDAVersion:         inc.ObservedEnvironment.CUDA,
		RequireHardwareLock: false,
	}
	if opts.DeviceOverrides != nil {
		devCfg = *opts.DeviceOverrides
	}

	nullControls := opts.NullControls
	if !nullControls.Enabled {
		nullControls = NullControlConfig{
			Enabled:         true,
			BasketRuns:      3,
			MaxNoiseFloorNs: 50 * time.Microsecond,
		}
	}

	stopping := opts.StoppingRules
	if stopping.MaxTrials <= 0 {
		stopping = DefaultStoppingRules()
	}

	plan := &ExperimentPlan{
		Version:           Version,
		PlanID:            fmt.Sprintf("plan-%s-%d", inc.ID, time.Now().UnixNano()),
		IncidentID:        inc.ID,
		IncidentDigest:    incDigest,
		Workload:          inc.Workload,
		SourceArtifacts:   opts.Sources,
		BinaryArtifacts:   opts.Binaries,
		LibraryClosure:    opts.Libraries,
		Inputs:            opts.Inputs,
		CorrectnessOracle: opts.Oracle,
		TimingScope:       scope,
		DeviceConfig:      devCfg,
		CaptureConfig: CaptureConfig{
			Enabled:         false,
			MaxCaptureBytes: 16 << 20, // 16 MiB bound
			TraceLossBudget: 0,
		},
		ExecutionOrder: order,
		WarmupRuns:     warmup,
		Rounds:         rounds,
		NullControls:   nullControls,
		StoppingRules:  stopping,
	}

	// Compute plan digest
	pDigest, err := ComputePlanDigest(plan)
	if err != nil {
		return nil, nil, err
	}
	plan.PlanDigest = pDigest

	// Validate artifact closure on disk
	receipt := &PlanReceipt{
		PlanID:               plan.PlanID,
		PlanDigest:           plan.PlanDigest,
		IncidentID:           plan.IncidentID,
		CreatedAt:            time.Now(),
		ArtifactClosureValid: true,
	}

	if inc.CaptureReceiptRef.Path != "" {
		receipt.ImmutableRefs = append(receipt.ImmutableRefs, inc.CaptureReceiptRef)
	}
	receipt.ImmutableRefs = append(receipt.ImmutableRefs, plan.SourceArtifacts...)
	receipt.ImmutableRefs = append(receipt.ImmutableRefs, plan.BinaryArtifacts...)
	for _, in := range plan.Inputs {
		receipt.ImmutableRefs = append(receipt.ImmutableRefs, ArtifactRef{
			Name:      in.Name,
			Path:      in.Path,
			Digest:    in.Digest,
			Role:      "input",
			SizeBytes: 0,
		})
	}

	if err := ValidatePlan(plan); err != nil {
		return nil, nil, err
	}
	receipt.ArtifactsVerified = len(receipt.ImmutableRefs)

	return plan, receipt, nil
}

// ComputePlanDigest calculates a deterministic SHA-256 digest of the ExperimentPlan.
func ComputePlanDigest(plan *ExperimentPlan) (string, error) {
	copyPlan := *plan
	copyPlan.PlanDigest = ""
	data, err := json.Marshal(copyPlan)
	if err != nil {
		return "", fmt.Errorf("cudaregress: failed to marshal plan for digest: %w", err)
	}
	h := sha256.Sum256(data)
	return hex.EncodeToString(h[:]), nil
}

// ValidatePlan verifies structural completeness and authenticates all declared artifacts on disk.
//
// For a derived plan, ValidatePlan checks only the shape of the lineage
// fields; the lineage itself is a claim that only [VerifyDerivation]
// authenticates against the parent plan.
func ValidatePlan(plan *ExperimentPlan) error {
	if plan == nil {
		return errors.New("cudaregress: nil experiment plan")
	}
	if plan.Version != Version {
		return fmt.Errorf("cudaregress: unsupported plan version %d", plan.Version)
	}
	if plan.PlanID == "" {
		return errors.New("cudaregress: missing plan ID")
	}
	if plan.IncidentID == "" {
		return errors.New("cudaregress: missing incident ID")
	}
	if plan.Rounds <= 0 {
		return fmt.Errorf("cudaregress: rounds must be positive: %d", plan.Rounds)
	}
	if plan.WarmupRuns < 0 {
		return fmt.Errorf("cudaregress: warmup runs cannot be negative: %d", plan.WarmupRuns)
	}

	if err := validateLineage(plan); err != nil {
		return err
	}

	switch plan.ExecutionOrder {
	case "ABBA", "interleaved", "sequential":
	default:
		return fmt.Errorf("cudaregress: unsupported execution order %q", plan.ExecutionOrder)
	}

	// Verify immutable artifacts on disk.
	verifyArtifact := func(a ArtifactRef) error {
		if a.Path == "" || len(a.Digest) != 64 {
			return fmt.Errorf("cudaregress: artifact %q requires path and SHA256 digest", a.Name)
		}
		cleanPath := filepath.Clean(a.Path)
		info, err := os.Stat(cleanPath)
		if err != nil {
			if os.IsNotExist(err) {
				return fmt.Errorf("%w: %s (%s)", ErrArtifactMissing, a.Name, cleanPath)
			}
			return fmt.Errorf("cudaregress: cannot access artifact %s: %w", cleanPath, err)
		}
		if !info.Mode().IsRegular() {
			return fmt.Errorf("cudaregress: artifact %s is not a regular file", cleanPath)
		}
		if a.Digest != "" {
			actualDigest, _, err := ComputeFileDigest(cleanPath)
			if err != nil {
				return err
			}
			if actualDigest != a.Digest {
				return fmt.Errorf("%w: %s (path %s, got sha256 %s, want %s)",
					ErrArtifactCorrupted, a.Name, cleanPath, actualDigest, a.Digest)
			}
		}
		return nil
	}

	for _, a := range plan.SourceArtifacts {
		if err := verifyArtifact(a); err != nil {
			return err
		}
	}
	for _, a := range plan.BinaryArtifacts {
		if err := verifyArtifact(a); err != nil {
			return err
		}
	}
	for _, in := range plan.Inputs {
		if in.Path != "" {
			a := ArtifactRef{Name: in.Name, Path: in.Path, Digest: in.Digest}
			if err := verifyArtifact(a); err != nil {
				return err
			}
		}
	}

	for _, lib := range plan.LibraryClosure {
		if err := verifyArtifact(ArtifactRef{Name: lib.Name, Path: lib.Path, Digest: lib.Digest}); err != nil {
			return err
		}
	}
	if plan.DeviceConfig.RequireHardwareLock && (len(plan.SourceArtifacts) == 0 || len(plan.BinaryArtifacts) == 0 || len(plan.Inputs) == 0 || len(plan.LibraryClosure) == 0) {
		return errors.New("cudaregress: hardware execution requires complete artifact closure")
	}

	if plan.CorrectnessOracle.ReferencePath != "" {
		a := ArtifactRef{
			Name:   "oracle_reference",
			Path:   plan.CorrectnessOracle.ReferencePath,
			Digest: plan.CorrectnessOracle.ReferenceDigest,
		}
		if err := verifyArtifact(a); err != nil {
			return err
		}
	}

	actualDigest, err := ComputePlanDigest(plan)
	if err != nil {
		return err
	}
	if plan.PlanDigest != actualDigest {
		return fmt.Errorf("%w: plan digest mismatch", ErrArtifactCorrupted)
	}
	if plan.StoppingRules.MaxTrials <= 0 || plan.Rounds > plan.StoppingRules.MaxTrials/2 {
		return errors.New("cudaregress: comparative arm invocations exceed max trials")
	}

	return nil
}

// ValidateEnvironment checks that the current host and GPU match the plan's requirements.
// If drift is detected, a descriptive ErrEnvironmentDrift is returned.
func ValidateEnvironment(plan *ExperimentPlan, env Environment) error {
	if plan == nil {
		return errors.New("cudaregress: nil experiment plan")
	}

	req := plan.DeviceConfig
	if req.Model != "" && req.Model != env.Model {
		return fmt.Errorf("%w: device model drift (required %q, current %q)",
			ErrEnvironmentDrift, req.Model, env.Model)
	}
	if req.ComputeCapability != "" && req.ComputeCapability != env.ComputeCapability {
		return fmt.Errorf("%w: compute capability drift (required %q, current %q)",
			ErrEnvironmentDrift, req.ComputeCapability, env.ComputeCapability)
	}
	if req.DriverVersion != "" && req.DriverVersion != env.Driver {
		return fmt.Errorf("%w: driver version drift (required %q, current %q)",
			ErrEnvironmentDrift, req.DriverVersion, env.Driver)
	}
	if req.CUDAVersion != "" && req.CUDAVersion != env.CUDA {
		return fmt.Errorf("%w: CUDA toolkit drift (required %q, current %q)",
			ErrEnvironmentDrift, req.CUDAVersion, env.CUDA)
	}
	if req.DeviceUUID != "" && req.DeviceUUID != env.DeviceUUID {
		return fmt.Errorf("%w: device UUID missing or changed", ErrEnvironmentDrift)
	}
	if req.DeviceIndex != env.DeviceIndex {
		return fmt.Errorf("%w: device index changed", ErrEnvironmentDrift)
	}

	return nil
}

// validateLineage checks that a plan is either a root plan, with no parent
// digest and no rationale, or a derived plan with both.
func validateLineage(plan *ExperimentPlan) error {
	if plan.ParentPlanDigest == "" && plan.DerivationRationale == "" {
		return nil
	}
	if !isSHA256Hex(plan.ParentPlanDigest) {
		return fmt.Errorf("cudaregress: derived plan requires sha256 parent plan digest, have %q", plan.ParentPlanDigest)
	}
	if strings.TrimSpace(plan.DerivationRationale) == "" {
		return errors.New("cudaregress: derived plan requires derivation rationale")
	}
	return nil
}

func isSHA256Hex(s string) bool {
	if len(s) != 2*sha256.Size {
		return false
	}
	for _, c := range []byte(s) {
		if !('0' <= c && c <= '9' || 'a' <= c && c <= 'f') {
			return false
		}
	}
	return true
}

// DerivePlan returns a new plan that differs from parent only in the device
// fields taken from newEnv and in its lineage: the derived plan records
// parent's PlanDigest as ParentPlanDigest and rationale as DerivationRationale.
// It is used to rerun an experiment under an intentional environment change,
// such as a different GPU model.
//
// DerivePlan refuses a parent that fails [ValidatePlan] and an empty rationale.
// The derived plan shares no memory with parent. Its PlanID is computed from
// parent's PlanID and PlanDigest, the rationale, and the new device fields, so
// deriving twice from the same inputs yields the same plan.
func DerivePlan(parent *ExperimentPlan, newEnv Environment, rationale string) (*ExperimentPlan, error) {
	if parent == nil {
		return nil, errors.New("cudaregress: nil parent plan")
	}
	if strings.TrimSpace(rationale) == "" {
		return nil, errors.New("cudaregress: rationale required to derive experiment plan")
	}
	if err := ValidatePlan(parent); err != nil {
		return nil, fmt.Errorf("cudaregress: invalid parent plan: %w", err)
	}

	derived := clonePlan(parent)
	derived.PlanDigest = ""
	derived.ParentPlanDigest = parent.PlanDigest
	derived.DerivationRationale = rationale
	derived.DeviceConfig.Model = newEnv.Model
	derived.DeviceConfig.ComputeCapability = newEnv.ComputeCapability
	derived.DeviceConfig.DriverVersion = newEnv.Driver
	derived.DeviceConfig.CUDAVersion = newEnv.CUDA
	derived.DeviceConfig.DeviceIndex = newEnv.DeviceIndex
	derived.DeviceConfig.DeviceUUID = newEnv.DeviceUUID

	id, err := json.Marshal(struct {
		ParentPlanID     string       `json:"parent_plan_id"`
		ParentPlanDigest string       `json:"parent_plan_digest"`
		Rationale        string       `json:"rationale"`
		DeviceConfig     DeviceConfig `json:"device_config"`
	}{parent.PlanID, parent.PlanDigest, rationale, derived.DeviceConfig})
	if err != nil {
		return nil, fmt.Errorf("cudaregress: marshal derivation identity: %w", err)
	}
	h := sha256.Sum256(id)
	derived.PlanID = parent.PlanID + "-derived-" + hex.EncodeToString(h[:8])

	digest, err := ComputePlanDigest(derived)
	if err != nil {
		return nil, err
	}
	derived.PlanDigest = digest
	if err := ValidatePlan(derived); err != nil {
		return nil, fmt.Errorf("cudaregress: invalid derived plan: %w", err)
	}
	return derived, nil
}

// VerifyDerivation checks that derived is exactly the plan that
// [DerivePlan] produces from parent with derived's device fields and
// rationale. It returns an error if either plan fails [ValidatePlan],
// if derived is a root plan, if derived names a different parent, or if
// derived differs from parent in anything other than its device fields
// and lineage.
func VerifyDerivation(parent, derived *ExperimentPlan) error {
	if parent == nil || derived == nil {
		return errors.New("cudaregress: nil experiment plan")
	}
	if err := ValidatePlan(derived); err != nil {
		return fmt.Errorf("cudaregress: invalid derived plan: %w", err)
	}
	if derived.ParentPlanDigest == "" {
		return errors.New("cudaregress: plan is not derived")
	}
	if derived.ParentPlanDigest != parent.PlanDigest {
		return fmt.Errorf("cudaregress: derived plan names parent %s, not %s", derived.ParentPlanDigest, parent.PlanDigest)
	}
	dc := derived.DeviceConfig
	want, err := DerivePlan(parent, Environment{
		DeviceIndex:       dc.DeviceIndex,
		DeviceUUID:        dc.DeviceUUID,
		Model:             dc.Model,
		ComputeCapability: dc.ComputeCapability,
		Driver:            dc.DriverVersion,
		CUDA:              dc.CUDAVersion,
	}, derived.DerivationRationale)
	if err != nil {
		return err
	}
	if want.PlanDigest != derived.PlanDigest {
		return fmt.Errorf("cudaregress: derived plan differs from derivation of parent (digest %s, want %s)", derived.PlanDigest, want.PlanDigest)
	}
	return nil
}

// clonePlan returns a copy of p that shares no slices or maps with p.
func clonePlan(p *ExperimentPlan) *ExperimentPlan {
	c := *p
	c.Workload = p.Workload.Clone()
	c.SourceArtifacts = slices.Clone(p.SourceArtifacts)
	c.BinaryArtifacts = slices.Clone(p.BinaryArtifacts)
	c.LibraryClosure = slices.Clone(p.LibraryClosure)
	c.Inputs = slices.Clone(p.Inputs)
	for i := range c.Inputs {
		c.Inputs[i].ShapeDims = slices.Clone(c.Inputs[i].ShapeDims)
	}
	c.TimingScope.Phases = slices.Clone(p.TimingScope.Phases)
	return &c
}
