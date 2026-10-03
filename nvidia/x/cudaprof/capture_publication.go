package cudaprof

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

// CapturePublication carries one target's provider result and its producer
// artifact/workload/compiler receipt. It is not a driver attestation. The CLI
// must separately check the image recorded by the loaded target module and
// enforce observed pass wall time in its final manifest. Native timestamp
// boundaries alone do not establish controller wall-clock elapsed time.
type CapturePublication struct {
	Version     int                        `json:"version"`
	Result      CaptureProviderResult      `json:"result"`
	Artifact    *cudakernel.Artifact       `json:"artifact"`
	Workload    cudacache.WorkloadIdentity `json:"workload"`
	CompilerKey string                     `json:"compiler_key"`
}

// WriteCapturePublication validates and bounds one result before writing any
// bytes. It neither executes providers nor closes caller-owned sessions.
func WriteCapturePublication(w io.Writer, plan CapturePlan, p CapturePublication) error {
	if err := plan.Validate(); err != nil {
		return err
	}
	if err := publicationPreflight(plan, p); err != nil {
		return err
	}
	if err := validatePublication(plan, p); err != nil {
		return err
	}
	data, err := json.Marshal(p)
	if err != nil {
		return err
	}
	if len(data) > plan.Budget.MaxBytes {
		return fmt.Errorf("capture publication: byte budget exceeded")
	}
	n, err := w.Write(data)
	if err == nil && n != len(data) {
		err = io.ErrShortWrite
	}
	return err
}

// ReadCapturePublication reads strict bounded JSON and admits a single provider
// against plan. Original JSON bytes, including whitespace, count toward budget.
func ReadCapturePublication(r io.Reader, plan CapturePlan, maxBytes int) (CapturePublication, error) {
	if err := plan.Validate(); err != nil {
		return CapturePublication{}, err
	}
	if maxBytes <= 0 || maxBytes > 64<<20 {
		return CapturePublication{}, fmt.Errorf("capture publication: invalid read byte limit")
	}
	// Avoid Artifact.UnmarshalJSON's permissive unknown-field behavior while
	// retaining the existing artifact representation and validation contract.
	type artifactFields cudakernel.Artifact
	var wire struct {
		Version     int                        `json:"version"`
		Result      CaptureProviderResult      `json:"result"`
		Artifact    *artifactFields            `json:"artifact"`
		Workload    cudacache.WorkloadIdentity `json:"workload"`
		CompilerKey string                     `json:"compiler_key"`
	}
	var encoded int
	if err := readCaptureJSON(r, min(maxBytes, plan.Budget.MaxBytes), &wire, &encoded); err != nil {
		return CapturePublication{}, err
	}
	p := CapturePublication{Version: wire.Version, Result: wire.Result, Artifact: (*cudakernel.Artifact)(wire.Artifact), Workload: wire.Workload, CompilerKey: wire.CompilerKey}
	if err := publicationPreflight(plan, p); err != nil {
		return CapturePublication{}, err
	}
	if err := validatePublication(plan, p); err != nil {
		return CapturePublication{}, err
	}
	if encoded > plan.Budget.MaxBytes {
		return CapturePublication{}, fmt.Errorf("capture publication: original byte budget exceeded")
	}
	return p, nil
}

func validatePublication(plan CapturePlan, p CapturePublication) error {
	if p.CompilerKey != "" {
		if err := captureDigest(p.CompilerKey); err != nil {
			return err
		}
	}
	if p.Version != CapturePlanVersion {
		return fmt.Errorf("capture publication: unsupported version")
	}
	if p.Artifact == nil {
		if p.Result.Status == "complete" {
			return fmt.Errorf("capture publication: complete result has no artifact")
		}
	} else {
		id, err := BindEvidence(p.Artifact, p.Workload, p.CompilerKey)
		if err != nil {
			return fmt.Errorf("capture publication: %w", err)
		}
		if id != p.Result.Identity {
			return fmt.Errorf("capture publication: result disagrees with artifact receipt")
		}
	}
	var selected *CapturePass
	for _, pass := range plan.Passes {
		if pass.ID != p.Result.PassID {
			continue
		}
		for _, provider := range pass.Providers {
			if provider == p.Result.Provider {
				copyPass := pass
				copyPass.Providers = []CaptureProvider{provider}
				copyPass.Configs = map[CaptureProvider]CaptureProviderConfig{provider: pass.Configs[provider]}
				selected = &copyPass
			}
		}
	}
	if selected == nil {
		return fmt.Errorf("capture publication: provider not requested")
	}
	filtered := plan
	filtered.Passes = []CapturePass{*selected}
	return (CaptureManifest{Version: CapturePlanVersion, Plan: filtered, Results: []CaptureProviderResult{p.Result}}).Validate()
}

// Bound variable-sized artifact receipt fields before hashing or JSON encoding.
// The manifest preflight separately bounds result and request fields.
func publicationPreflight(plan CapturePlan, p CapturePublication) error {
	if err := captureEncodingPreflight(CaptureManifest{Plan: plan, Results: []CaptureProviderResult{p.Result}}, plan.Budget.MaxBytes); err != nil {
		return err
	}
	remaining := plan.Budget.MaxBytes
	take := func(n int) bool {
		if n > remaining {
			return false
		}
		remaining -= n
		return true
	}
	fields := []string{p.CompilerKey, p.Workload.ProgramID, p.Workload.NumericalMode, p.Workload.PrepackID}
	for _, s := range fields {
		if !take(len(s)) {
			return fmt.Errorf("capture publication: receipt byte budget exceeded")
		}
	}
	if len(p.Workload.ShapeDims) > remaining/8 || !take(len(p.Workload.ShapeDims)*8) {
		return fmt.Errorf("capture publication: workload byte budget exceeded")
	}
	a := p.Artifact
	if a == nil {
		return captureEncodingPreflight(CaptureManifest{Plan: plan, Results: []CaptureProviderResult{p.Result}}, remaining)
	}
	for _, s := range []string{a.Digest, a.EntryPoint, a.Target, a.Toolchain, a.Resources.ExecutionGroup} {
		if !take(len(s)) {
			return fmt.Errorf("capture publication: artifact byte budget exceeded")
		}
	}
	if len(a.Raw) > remaining/4*3 || !take((len(a.Raw)+2)/3*4) || !take(len(a.CompilerOpts)) || !take(len(a.Dependencies)) || !take(len(a.Schema.Params)) {
		return fmt.Errorf("capture publication: artifact byte budget exceeded")
	}
	for _, s := range a.CompilerOpts {
		if !take(len(s)) {
			return fmt.Errorf("capture publication: option byte budget exceeded")
		}
	}
	for _, d := range a.Dependencies {
		if !take(len(d.Path)) || !take(len(d.Digest)) || !take(len(d.ErrorMsg)) {
			return fmt.Errorf("capture publication: dependency byte budget exceeded")
		}
	}
	for _, param := range a.Schema.Params {
		if !take(len(param.Name)) || !take(len(param.Type)) {
			return fmt.Errorf("capture publication: schema byte budget exceeded")
		}
	}
	return captureEncodingPreflight(CaptureManifest{Plan: plan, Results: []CaptureProviderResult{p.Result}}, remaining)
}
