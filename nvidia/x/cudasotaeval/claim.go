package cudasotaeval

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

var (
	// ErrIncompleteClaim indicates required claim fields are missing.
	ErrIncompleteClaim = errors.New("cudasotaeval: claim missing required bounded fields")
)

// FormulateClaim derives a strictly bounded comparative claim from an EvaluationReport.
// It verifies that all four properties passed and names the exact task, alternatives,
// hardware closure, receipts, and explicit operational limits.
func FormulateClaim(report *EvaluationReport) (*Claim, error) {
	if report == nil {
		return nil, errors.New("cudasotaeval: nil evaluation report")
	}

	if report.Synthetic || !report.Passed || report.Protocol.Phase != PhaseQualification || !report.Protocol.Frozen {
		return nil, errors.New("cudasotaeval: qualification evidence incomplete")
	}
	for _, gate := range report.Gates {
		if gate.Disposition != "passed" {
			return nil, errors.New("cudasotaeval: qualification gate not passed")
		}
	}
	if len(report.Reproducibility.ArtifactManifestSHA256) != 64 {
		return nil, errors.New("cudasotaeval: missing artifact manifest digest")
	}

	task := fmt.Sprintf("Continuous Go GPU workload submission delay diagnosis under concurrency (%s)",
		report.Protocol.Name)

	advantage := fmt.Sprintf("Zero incorrect exact joins (%d evaluated, %d exact) with admitted causal diagnosis and continuous flight recorder throughput degradation <= %.2f%%",
		report.Attribution.TotalRequests,
		report.Attribution.ExactJoins,
		report.Cost.ThroughputDegradationPct)

	var comparedWith []string
	for _, b := range report.Baselines {
		comparedWith = append(comparedWith, fmt.Sprintf("%s (%s, %s)", b.Name, b.Version, b.ComparisonScope))
	}

	qualifiedOn := fmt.Sprintf("%s (%s, sm_count=%d), driver %s, CUDA %s, %s/%s",
		report.Protocol.Hardware.GPUModel,
		report.Protocol.Hardware.Architecture,
		report.Protocol.Hardware.SMCount,
		report.Protocol.Hardware.DriverVersion,
		report.Protocol.Hardware.CUDAVersion,
		report.Protocol.Hardware.HostOS,
		report.Protocol.Hardware.HostArch)

	noFaultPct := 0.0
	if report.Diagnostic.NoFaultTotal > 0 {
		noFaultPct = float64(report.Diagnostic.NoFaultPassed) / float64(report.Diagnostic.NoFaultTotal) * 100.0
	}

	evidence := fmt.Sprintf("attribution precision=%.2f%% (zero incorrect joins); held-out diagnosis accuracy=%.1f%% (no-fault rejection=%.1f%%); collector throughput degradation=%.2f%% (RSS=%.1fMB); replay match=%t (manifest=%s)",
		report.Attribution.JoinPrecision*100,
		report.Diagnostic.AccuracyPct,
		noFaultPct,
		report.Cost.ThroughputDegradationPct,
		report.Cost.MemoryGrowthMB,
		report.Reproducibility.ReplayMatches,
		report.Reproducibility.ArtifactManifestSHA256[:16])

	limits := []string{
		"Restricted to Go CUDA Driver API launch paths; does not monitor unannotated closed-source external binaries",
		"Kernel-level instruction stalls and warp divergence defer to NVIDIA Nsight Compute (ncu)",
		"Full unified host-device graphical timelines defer to NVIDIA Nsight Systems (nsys)",
		"Physical hardware qualification must be conducted separately for each GPU architecture/driver combination",
		"Related telemetry tools (Beyla, OBI) share common tracing foundations and are not independent baselines",
		"Claim is strictly bounded to evidence-admitted host submission delay diagnosis; does not assert superiority in kernel optimization, scheduling, or fleet management",
	}

	// Include any failed or withheld gate dispositions directly into limits
	for _, g := range report.Gates {
		if g.Disposition == "failed" || g.Disposition == "withheld" || g.Disposition == "unavailable" {
			limits = append(limits, fmt.Sprintf("Gate %s [%s]: %s (%s)", g.GateName, g.Disposition, g.Reason, g.MeasuredValue))
		}
	}

	claim := &Claim{
		Task:              task,
		SpecificAdvantage: advantage,
		ComparedWith:      comparedWith,
		QualifiedOn:       qualifiedOn,
		Evidence:          evidence,
		Limits:            limits,
	}

	if err := claim.Validate(); err != nil {
		return nil, fmt.Errorf("formulate claim: %w", err)
	}

	return claim, nil
}

// Validate verifies that the claim conforms to the bounded five-part format
// required by sota-evaluation.md.
func (c *Claim) Validate() error {
	if c.Task == "" {
		return fmt.Errorf("%w: missing task", ErrIncompleteClaim)
	}
	if c.SpecificAdvantage == "" {
		return fmt.Errorf("%w: missing specific advantage", ErrIncompleteClaim)
	}
	if len(c.ComparedWith) == 0 {
		return fmt.Errorf("%w: missing compared_with alternatives", ErrIncompleteClaim)
	}
	if c.QualifiedOn == "" {
		return fmt.Errorf("%w: missing qualified_on hardware specification", ErrIncompleteClaim)
	}
	if c.Evidence == "" {
		return fmt.Errorf("%w: missing evidence receipts", ErrIncompleteClaim)
	}
	if len(c.Limits) == 0 {
		return fmt.Errorf("%w: missing limits and boundaries", ErrIncompleteClaim)
	}
	return nil
}

// Format renders the claim in the exact 5-part format specified in sota-evaluation.md.
func (c *Claim) Format() string {
	var sb strings.Builder

	sb.WriteString(fmt.Sprintf("Claim: %s for %s.\n", c.SpecificAdvantage, c.Task))

	sb.WriteString("Compared with:\n")
	for _, comp := range c.ComparedWith {
		sb.WriteString(fmt.Sprintf("  - %s\n", comp))
	}

	sb.WriteString(fmt.Sprintf("Qualified on: %s.\n", c.QualifiedOn))
	sb.WriteString(fmt.Sprintf("Evidence: %s.\n", c.Evidence))

	sb.WriteString("Limits:\n")
	for _, lim := range c.Limits {
		sb.WriteString(fmt.Sprintf("  - %s\n", lim))
	}

	return sb.String()
}

// JSON returns indented JSON representation of the Claim.
func (c *Claim) JSON() ([]byte, error) {
	return json.MarshalIndent(c, "", "  ")
}
