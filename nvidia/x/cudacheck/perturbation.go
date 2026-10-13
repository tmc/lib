package cudacheck

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// ErrPerturbationArtifactMismatch indicates that an artifact returned by compilation
// or cache lookup does not reflect the requested perturbation parameters.
var ErrPerturbationArtifactMismatch = errors.New("cudalmvet: compiled artifact does not match requested perturbation options")

// Perturbation represents a specific combination of scheduling-affecting compiler options.
type Perturbation struct {
	Name                 string
	ExtraDeviceVectorize bool
	JITOptLevel          int // 1 to 4; zero means unset in cudakernel.JITOptions.
}

// DefaultPerturbations returns the bounded set of scheduling-affecting perturbation configurations:
//   - --extra-device-vectorization on / off at NVRTC phase
//   - CU_JIT_OPT_LEVEL 1 through 4 at driver JIT phase; zero is unsupported because
//     cudakernel uses zero to mean unset.
func DefaultPerturbations() []Perturbation {
	return []Perturbation{
		{
			Name:                 "baseline",
			ExtraDeviceVectorize: false,
			JITOptLevel:          4,
		},
		{
			Name:                 "vectorize_on",
			ExtraDeviceVectorize: true,
			JITOptLevel:          4,
		},
		{
			Name:                 "jit_opt1",
			ExtraDeviceVectorize: false,
			JITOptLevel:          1,
		},
		{
			Name:                 "jit_opt2",
			ExtraDeviceVectorize: false,
			JITOptLevel:          2,
		},
		{
			Name:                 "jit_opt3",
			ExtraDeviceVectorize: false,
			JITOptLevel:          3,
		},
		{
			Name:                 "vectorize_opt3",
			ExtraDeviceVectorize: true,
			JITOptLevel:          3,
		},
	}
}

// ValidatePerturbationArtifact verifies that a compiled artifact faithfully embodies
// the requested perturbation options, preventing silent cache reuse of unperturbed artifacts.
func ValidatePerturbationArtifact(pert Perturbation, art *cudakernel.Artifact) error {
	if pert.JITOptLevel < 1 || pert.JITOptLevel > 4 {
		return fmt.Errorf("cudalmvet: unsupported JIT optimization level %d (want 1 through 4)", pert.JITOptLevel)
	}
	if art == nil {
		return fmt.Errorf("%w: nil artifact", ErrPerturbationArtifactMismatch)
	}

	hasVec := false
	for _, opt := range art.CompilerOpts {
		if opt == "--extra-device-vectorization" {
			hasVec = true
			break
		}
	}

	if pert.ExtraDeviceVectorize && !hasVec {
		return fmt.Errorf("%w: expected --extra-device-vectorization in compiler options", ErrPerturbationArtifactMismatch)
	}
	if !pert.ExtraDeviceVectorize && hasVec {
		return fmt.Errorf("%w: unexpected --extra-device-vectorization in compiler options", ErrPerturbationArtifactMismatch)
	}

	if art.JITOptions.OptLevel != uint32(pert.JITOptLevel) {
		return fmt.Errorf("%w: JIT OptLevel mismatch (got %d, want %d)",
			ErrPerturbationArtifactMismatch, art.JITOptions.OptLevel, pert.JITOptLevel)
	}

	return nil
}

// PerturbationResult captures the outcome of evaluating a single perturbation.
type PerturbationResult struct {
	Perturbation Perturbation
	Status       ComparisonStatus
	Stage        FailureStage
	Hash         uint64
	Triage       *TriageReport
	ArtifactErr  error
	PolicyErr    error
	CompileErr   error
	RunErr       error
}

// PerturbationReport collects the outcomes across all tested perturbations.
type PerturbationReport struct {
	ReferenceHash      uint64
	BaselineDeviceHash uint64
	Results            []PerturbationResult
}

// AllAgree reports whether all perturbations compiled, ran, and matched the reference hash.
func (r *PerturbationReport) AllAgree() bool {
	if len(r.Results) == 0 {
		return false
	}
	for _, res := range r.Results {
		if res.ArtifactErr != nil || res.PolicyErr != nil || res.CompileErr != nil || res.RunErr != nil || res.Status != StatusAgreed || res.Hash != r.ReferenceHash {
			return false
		}
	}
	return true
}

// HasDisagreements reports whether any perturbation ran successfully but diverged in trace hash.
func (r *PerturbationReport) HasDisagreements() bool {
	for _, res := range r.Results {
		if res.CompileErr == nil && res.RunErr == nil && res.Hash != r.ReferenceHash && res.Status == StatusDisagreed {
			return true
		}
	}
	return false
}

// String formats the perturbation outcomes into a clear aligned table.
func (r *PerturbationReport) String() string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Schedule Perturbation Report (Reference Hash: 0x%016x)\n", r.ReferenceHash))
	sb.WriteString(fmt.Sprintf("%-18s %-10s %-8s %-12s %-20s %s\n",
		"Perturbation", "Vectorize", "JIT_Opt", "Status", "Trace Hash", "Notes"))
	sb.WriteString(strings.Repeat("-", 80) + "\n")

	for _, res := range r.Results {
		vecStr := "off"
		if res.Perturbation.ExtraDeviceVectorize {
			vecStr = "on"
		}
		hashStr := "-"
		if res.Hash != 0 {
			hashStr = fmt.Sprintf("0x%016x", res.Hash)
		}
		notes := "-"
		if res.CompileErr != nil {
			notes = fmt.Sprintf("compile error: %v", res.CompileErr)
		} else if res.PolicyErr != nil {
			notes = fmt.Sprintf("unsupported device policy: %v", res.PolicyErr)
		} else if res.ArtifactErr != nil {
			notes = fmt.Sprintf("artifact mismatch: %v", res.ArtifactErr)
		} else if res.RunErr != nil {
			notes = fmt.Sprintf("run error: %v", res.RunErr)
		} else {
			if res.Status != StatusUnmeasured && r.BaselineDeviceHash != 0 && res.Hash != r.BaselineDeviceHash {
				notes = "HASH MOVED vs baseline device hash"
			}
			if res.Triage != nil {
				if notes == "-" {
					notes = res.Triage.String()
				} else {
					notes += "; " + res.Triage.String()
				}
			}
		}

		sb.WriteString(fmt.Sprintf("%-18s %-10s %-8d %-12s %-20s %s\n",
			res.Perturbation.Name,
			vecStr,
			res.Perturbation.JITOptLevel,
			res.Status.String(),
			hashStr,
			notes,
		))
	}

	return sb.String()
}

// RunPerturbations tests a computation graph under each specified perturbation option,
// producing a comprehensive report.
func RunPerturbations(ctx context.Context, g *cudair.Graph, perts []Perturbation, opts ...CompareOption) (*PerturbationReport, error) {
	if len(perts) == 0 {
		perts = DefaultPerturbations()
	}

	cfg := &CompareConfig{
		Policy:           PolicyRawBits,
		TriageOnMismatch: true,
	}
	for _, opt := range opts {
		opt(cfg)
	}

	// Compute CPU reference hash
	accCPU, err := EvaluateCPUGraph(g, cfg.Inputs, cfg.Policy, false)
	if err != nil {
		return nil, fmt.Errorf("cudalmvet.RunPerturbations: CPU reference failed: %w", err)
	}
	refHash := accCPU.Hash()

	report := &PerturbationReport{
		ReferenceHash: refHash,
		Results:       make([]PerturbationResult, 0, len(perts)),
	}
	if accCPU.hasNaN {
		for _, p := range perts {
			report.Results = append(report.Results, PerturbationResult{Perturbation: p, Status: StatusUnmeasured, Stage: StagePolicy, PolicyErr: ErrNaNTraceUnsupported})
		}
		return report, nil
	}

	for _, p := range perts {
		res := PerturbationResult{
			Perturbation: p,
			Status:       StatusUnmeasured,
		}

		// Configure compiler options for this perturbation
		var compilerOpts []nvrtcx.Option
		compilerOpts = append(compilerOpts, cfg.CompilerOpts...)
		if p.ExtraDeviceVectorize {
			compilerOpts = append(compilerOpts, nvrtcx.WithExtraDeviceVectorization())
		}

		if err := ValidatePerturbationArtifact(p, &cudakernel.Artifact{JITOptions: cudakernel.JITOptions{OptLevel: uint32(p.JITOptLevel)}, CompilerOpts: perturbationCompilerFlags(p)}); err != nil {
			res.CompileErr = err
			report.Results = append(report.Results, res)
			continue
		}
		jitOpts := cfg.JITOptions
		jitOpts.OptLevel = uint32(p.JITOptLevel)

		pertCompareOpts := append([]CompareOption{}, opts...)
		pertCompareOpts = append(pertCompareOpts,
			WithCompilerOptions(compilerOpts...),
			WithJITOpts(jitOpts),
			withPerturbation(p),
		)

		cmpResult, cmpErr := Compare(ctx, g, pertCompareOpts...)
		res = perturbationOutcome(p, cmpResult, cmpErr)

		report.Results = append(report.Results, res)
		if p.Name == "baseline" && res.Status != StatusUnmeasured && res.CompileErr == nil && res.RunErr == nil {
			report.BaselineDeviceHash = res.Hash
		}
	}

	return report, nil
}

func perturbationOutcome(p Perturbation, result *ComparisonResult, err error) PerturbationResult {
	row := PerturbationResult{Perturbation: p, Status: StatusUnmeasured}
	if err != nil {
		row.RunErr = err
		return row
	}
	if result == nil {
		row.RunErr = errors.New("cudalmvet: compare returned no result")
		return row
	}
	row.Status, row.Stage, row.Hash, row.Triage = result.Status, result.Stage, result.DeviceHash, result.Triage
	if result.Err != nil && result.Status == StatusUnmeasured {
		switch result.Stage {
		case StageCompile:
			row.CompileErr = result.Err
		case StageArtifactValidation:
			row.ArtifactErr = result.Err
		case StagePolicy:
			row.PolicyErr = result.Err
		case StageLoad, StageLaunch, StageReadback:
			row.RunErr = result.Err
		}
	}
	return row
}

func perturbationCompilerFlags(p Perturbation) []string {
	if p.ExtraDeviceVectorize {
		return []string{"--extra-device-vectorization"}
	}
	return nil
}

func validateCompiledPerturbation(p Perturbation, compilerOpts []string, module *nvrtcx.CompiledModule) error {
	if module == nil {
		return fmt.Errorf("%w: compiler returned nil module", ErrPerturbationArtifactMismatch)
	}
	artifact := &cudakernel.Artifact{CompilerOpts: append([]string(nil), compilerOpts...), JITOptions: module.JITOptions()}
	return ValidatePerturbationArtifact(p, artifact)
}
