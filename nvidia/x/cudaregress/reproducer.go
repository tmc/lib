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
	PlanID               string        `json:"plan_id"`
	PlanDigest           string        `json:"plan_digest"`
	IncidentID           string        `json:"incident_id"`
	TargetArchitecture   string        `json:"target_architecture"`
	Artifacts            []ArtifactRef `json:"artifacts"`
	ArtifactClosureValid bool          `json:"artifact_closure_valid"`
	Reconstructable      bool          `json:"reconstructable"`
	Blockers             []string      `json:"blockers,omitempty"`
}

// CheckReconstruction evaluates whether an experiment plan can be reconstructed
// against the caller-supplied environment. It authenticates the plan and declared
// files with [ValidatePlan], then checks [ValidateEnvironment]. It does not
// observe the environment, execute the plan, or authenticate claimed lineage.
func CheckReconstruction(plan *ExperimentPlan, currentEnv Environment) (bool, []string) {
	if plan == nil {
		return false, []string{"nil experiment plan"}
	}
	var blockers []string
	if err := ValidatePlan(plan); err != nil {
		blockers = append(blockers, err.Error())
	}
	if err := ValidateEnvironment(plan, currentEnv); err != nil {
		blockers = append(blockers, err.Error())
	}

	if len(blockers) > 0 {
		return false, blockers
	}
	return true, nil
}

// ExportMinimalReproducer exports a minimal, bounded reproducer bundle to targetDir.
// It serializes the plan, generates an artifact manifest with digests, and writes
// reproduction documentation without copying unbounded raw capture archives.
// It calls [ValidatePlan] before creating output files. The manifest records a
// verified artifact closure, but leaves reconstruction blocked until a caller
// supplies a separately observed environment to [CheckReconstruction]. No plan
// execution or environment observation is performed.
func ExportMinimalReproducer(plan *ExperimentPlan, targetDir string) error {
	if plan == nil {
		return errors.New("cudaregress: nil plan")
	}
	if targetDir == "" {
		return errors.New("cudaregress: target directory cannot be empty")
	}
	if err := ValidatePlan(plan); err != nil {
		return fmt.Errorf("cudaregress: validate reproducer: %w", err)
	}

	// Serialize the validated plan and manifest before creating output files.
	planBytes, err := json.MarshalIndent(plan, "", "  ")
	if err != nil {
		return fmt.Errorf("cudaregress: marshal plan: %w", err)
	}
	manifest := Manifest{
		PlanID:               plan.PlanID,
		PlanDigest:           plan.PlanDigest,
		IncidentID:           plan.IncidentID,
		TargetArchitecture:   plan.DeviceConfig.ComputeCapability,
		ArtifactClosureValid: true,
		Blockers:             []string{"reconstruction requires a separately observed environment"},
	}
	manifest.Artifacts = append(manifest.Artifacts, plan.SourceArtifacts...)
	manifest.Artifacts = append(manifest.Artifacts, plan.BinaryArtifacts...)
	for _, in := range plan.Inputs {
		if in.Path == "" {
			continue
		}
		manifest.Artifacts = append(manifest.Artifacts, ArtifactRef{
			Name:   in.Name,
			Path:   in.Path,
			Digest: in.Digest,
			Role:   "input",
		})
	}
	for _, lib := range plan.LibraryClosure {
		manifest.Artifacts = append(manifest.Artifacts, ArtifactRef{
			Name: lib.Name, Path: lib.Path, Digest: lib.Digest, Role: "library",
		})
	}
	if oracle := plan.CorrectnessOracle; oracle.ReferencePath != "" {
		manifest.Artifacts = append(manifest.Artifacts, ArtifactRef{
			Name: "oracle_reference", Path: oracle.ReferencePath, Digest: oracle.ReferenceDigest, Role: "reference",
		})
	}

	manifestBytes, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return fmt.Errorf("cudaregress: marshal manifest: %w", err)
	}

	readme := fmt.Sprintf("# Experiment Reproducer: %s\n\n"+
		"Incident ID: `%s`\n"+
		"Plan Digest: `%s`\n"+
		"Target Architecture: `%s`\n"+
		"Timing Channel: `%s`\n\n"+
		"## Execution Instructions\n\n"+
		"To replay this regression trial, supply a declared execution spec and a separately observed environment:\n\n"+
		"```sh\n"+
		"cudaprof regress replay -plan plan.json -spec execution-spec.json -environment observed-environment.json -o receipts\n"+
		"```\n\n"+
		"Artifact paths and verified digests are declared in `manifest.json`; the files are not copied.\n"+
		"Reconstruction remains blocked until a separately observed environment is checked.\n"+
		"Export does not execute the plan or authenticate claimed parent lineage.\n",
		plan.PlanID, plan.IncidentID, plan.PlanDigest,
		plan.DeviceConfig.ComputeCapability, plan.TimingScope.Channel)

	if err := os.MkdirAll(targetDir, 0755); err != nil {
		return fmt.Errorf("cudaregress: create reproducer dir: %w", err)
	}
	for _, file := range []struct {
		name string
		data []byte
	}{
		{"plan.json", planBytes},
		{"manifest.json", manifestBytes},
		{"README.md", []byte(readme)},
	} {
		if err := os.WriteFile(filepath.Join(targetDir, file.name), file.data, 0644); err != nil {
			return fmt.Errorf("cudaregress: write %s: %w", file.name, err)
		}
	}
	return nil
}
