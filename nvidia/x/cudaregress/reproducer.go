package cudaregress

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

// Manifest describes the artifact closure and verification status for reproduction.
type Manifest struct {
	PlanID             string        `json:"plan_id"`
	PlanDigest         string        `json:"plan_digest"`
	IncidentID         string        `json:"incident_id"`
	TargetArchitecture string        `json:"target_architecture"`
	Artifacts          []ArtifactRef `json:"artifacts"`
	Reconstructable    bool          `json:"reconstructable"`
	Blockers           []string      `json:"blockers,omitempty"`
}

// CheckReconstruction evaluates whether an experiment plan can be reconstructed
// on the current host. If unavailable originals or environment mismatches exist,
// it returns reconstructable=false and a list of specific blockers.
func CheckReconstruction(plan *ExperimentPlan, currentEnv Environment) (bool, []string) {
	if plan == nil {
		return false, []string{"nil experiment plan"}
	}

	var blockers []string

	// Check environment drift
	if err := ValidateEnvironment(plan, currentEnv); err != nil {
		blockers = append(blockers, err.Error())
	}

	// Check artifact closure
	for _, a := range plan.SourceArtifacts {
		if a.Path != "" {
			if _, err := os.Stat(a.Path); os.IsNotExist(err) {
				blockers = append(blockers, fmt.Sprintf("missing source artifact %q at %s", a.Name, a.Path))
			}
		}
	}
	for _, a := range plan.BinaryArtifacts {
		if a.Path != "" {
			if _, err := os.Stat(a.Path); os.IsNotExist(err) {
				blockers = append(blockers, fmt.Sprintf("missing binary artifact %q at %s", a.Name, a.Path))
			}
		}
	}
	for _, in := range plan.Inputs {
		if in.Path != "" {
			if _, err := os.Stat(in.Path); os.IsNotExist(err) {
				blockers = append(blockers, fmt.Sprintf("missing input artifact %q at %s", in.Name, in.Path))
			}
		}
	}

	if len(blockers) > 0 {
		return false, blockers
	}
	return true, nil
}

// ExportMinimalReproducer exports a minimal, bounded reproducer bundle to targetDir.
// It serializes the plan, generates an artifact manifest with digests, and writes
// reproduction documentation without copying unbounded raw capture archives.
func ExportMinimalReproducer(plan *ExperimentPlan, targetDir string) error {
	if plan == nil {
		return errors.New("cudaregress: nil plan")
	}
	if targetDir == "" {
		return errors.New("cudaregress: target directory cannot be empty")
	}

	if err := os.MkdirAll(targetDir, 0755); err != nil {
		return fmt.Errorf("cudaregress: create reproducer dir: %w", err)
	}

	// 1. Write plan.json
	planBytes, err := json.MarshalIndent(plan, "", "  ")
	if err != nil {
		return fmt.Errorf("cudaregress: marshal plan: %w", err)
	}
	if err := os.WriteFile(filepath.Join(targetDir, "plan.json"), planBytes, 0644); err != nil {
		return err
	}

	// 2. Build manifest
	manifest := Manifest{
		PlanID:             plan.PlanID,
		PlanDigest:         plan.PlanDigest,
		IncidentID:         plan.IncidentID,
		TargetArchitecture: plan.DeviceConfig.ComputeCapability,
	}
	manifest.Artifacts = append(manifest.Artifacts, plan.SourceArtifacts...)
	manifest.Artifacts = append(manifest.Artifacts, plan.BinaryArtifacts...)
	for _, in := range plan.Inputs {
		manifest.Artifacts = append(manifest.Artifacts, ArtifactRef{
			Name:   in.Name,
			Path:   in.Path,
			Digest: in.Digest,
			Role:   "input",
		})
	}

	reconstructable, blockers := CheckReconstruction(plan, Environment{
		ComputeCapability: plan.DeviceConfig.ComputeCapability,
		Model:             plan.DeviceConfig.Model,
		Driver:            plan.DeviceConfig.DriverVersion,
		CUDA:              plan.DeviceConfig.CUDAVersion,
	})
	manifest.Reconstructable = reconstructable
	manifest.Blockers = blockers

	manifestBytes, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return fmt.Errorf("cudaregress: marshal manifest: %w", err)
	}
	if err := os.WriteFile(filepath.Join(targetDir, "manifest.json"), manifestBytes, 0644); err != nil {
		return err
	}

	// 3. Write README.md reproduction instructions
	readme := fmt.Sprintf("# Experiment Reproducer: %s\n\n"+
		"Incident ID: `%s`\n"+
		"Plan Digest: `%s`\n"+
		"Target Architecture: `%s`\n"+
		"Timing Channel: `%s`\n\n"+
		"## Execution Instructions\n\n"+
		"To replay this regression trial deterministically:\n\n"+
		"```sh\n"+
		"cudaprof regress run -plan plan.json\n"+
		"```\n\n"+
		"Reference immutable artifacts are declared in `manifest.json`.\n",
		plan.PlanID, plan.IncidentID, plan.PlanDigest,
		plan.DeviceConfig.ComputeCapability, plan.TimingScope.Channel)

	return os.WriteFile(filepath.Join(targetDir, "README.md"), []byte(readme), 0644)
}
