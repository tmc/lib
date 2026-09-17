package cudacapture

import (
	"errors"
	"fmt"
)

// ErrIneligibleComparison indicates that two captures or identities cannot be compared
// due to missing version, mismatched phase/workload/tokens/device, or incomplete runs.
var ErrIneligibleComparison = errors.New("cudacapture: ineligible comparison")

// ComparisonIdentity provides a versioned identity schema for captures used in
// comparative performance evaluations.
type ComparisonIdentity struct {
	Version      string `json:"version"`                 // Identity schema version (e.g. "v1")
	Phase        string `json:"phase,omitempty"`         // Workload phase (e.g. "prefill", "decode")
	WorkloadName string `json:"workload_name,omitempty"` // Workload model/name
	WorkloadHash string `json:"workload_hash,omitempty"` // Digest of workload/model
	TokensHash   string `json:"tokens_hash,omitempty"`   // Digest of input prompt / tokens
	Revision     string `json:"revision,omitempty"`      // Source code or build revision
	Device       string `json:"device,omitempty"`        // Device model name
	DeviceIndex  int    `json:"device_index,omitempty"`  // GPU index
	ComputeCap   string `json:"compute_cap,omitempty"`   // Compute capability (e.g. "sm_90a")
	Toolchain    string `json:"toolchain,omitempty"`     // Compiler and toolchain version
	CompileOpts  string `json:"compile_opts,omitempty"`  // Relevant compilation flags
	SamplingCfg  string `json:"sampling_cfg,omitempty"`  // Sampling configuration
	Complete     bool   `json:"complete"`                // True if execution and collection completed normally
}

// Valid returns whether the identity has a recognized schema version and completed normally.
func (id ComparisonIdentity) Valid() error {
	if id.Version == "" {
		return fmt.Errorf("%w: missing version", ErrIneligibleComparison)
	}
	if !id.Complete {
		return fmt.Errorf("%w: run or collection was incomplete", ErrIneligibleComparison)
	}
	if id.Phase == "" {
		return fmt.Errorf("%w: missing phase", ErrIneligibleComparison)
	}
	return nil
}

// Compatible verifies that two comparison identities are compatible for comparative analysis.
//
// In an A/B trial, Phase, WorkloadHash, TokensHash, Device, ComputeCap, and SamplingCfg
// must match. The Revision, Toolchain, and CompileOpts fields are permitted to differ as they
// represent the intended experimental difference.
func (id ComparisonIdentity) Compatible(other ComparisonIdentity) error {
	if err := id.Valid(); err != nil {
		return fmt.Errorf("%w: candidate invalid: %v", ErrIneligibleComparison, err)
	}
	if err := other.Valid(); err != nil {
		return fmt.Errorf("%w: baseline invalid: %v", ErrIneligibleComparison, err)
	}
	if id.Version != other.Version {
		return fmt.Errorf("%w: version mismatch: %q != %q", ErrIneligibleComparison, id.Version, other.Version)
	}
	if id.Phase != other.Phase {
		return fmt.Errorf("%w: phase mismatch: %q != %q", ErrIneligibleComparison, id.Phase, other.Phase)
	}
	if id.WorkloadHash != "" && other.WorkloadHash != "" && id.WorkloadHash != other.WorkloadHash {
		return fmt.Errorf("%w: workload hash mismatch: %q != %q", ErrIneligibleComparison, id.WorkloadHash, other.WorkloadHash)
	}
	if id.TokensHash != "" && other.TokensHash != "" && id.TokensHash != other.TokensHash {
		return fmt.Errorf("%w: tokens hash mismatch: %q != %q", ErrIneligibleComparison, id.TokensHash, other.TokensHash)
	}
	if id.Device != "" && other.Device != "" && id.Device != other.Device {
		return fmt.Errorf("%w: device mismatch: %q != %q", ErrIneligibleComparison, id.Device, other.Device)
	}
	if id.ComputeCap != "" && other.ComputeCap != "" && id.ComputeCap != other.ComputeCap {
		return fmt.Errorf("%w: compute capability mismatch: %q != %q", ErrIneligibleComparison, id.ComputeCap, other.ComputeCap)
	}
	if id.SamplingCfg != "" && other.SamplingCfg != "" && id.SamplingCfg != other.SamplingCfg {
		return fmt.Errorf("%w: sampling config mismatch: %q != %q", ErrIneligibleComparison, id.SamplingCfg, other.SamplingCfg)
	}
	return nil
}
