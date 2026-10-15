package cudaregress_test

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaregress"
)

func reconstructionPlan(t *testing.T) (*cudaregress.ExperimentPlan, cudaregress.Environment) {
	t.Helper()
	dir := t.TempDir()
	source := makeTestArtifact(t, dir, "source", "supplied source bytes")
	source.Role = "source"
	binary := makeTestArtifact(t, dir, "binary", "supplied binary bytes, never executed")
	binary.Role = "binary"
	library := makeTestArtifact(t, dir, "library", "supplied library bytes, never loaded")
	input := makeTestArtifact(t, dir, "input", "supplied input bytes")
	reference := makeTestArtifact(t, dir, "reference", "supplied reference bytes")
	inc := validAdmittedIncident()
	inc.ObservedEnvironment.DeviceIndex = 2
	inc.ObservedEnvironment.DeviceUUID = "supplied-device-uuid"
	plan, _, err := cudaregress.NewPlanFromIncident(inc, cudaregress.PlanOptions{
		Sources:  []cudaregress.ArtifactRef{source},
		Binaries: []cudaregress.ArtifactRef{binary},
		Libraries: []cudaregress.LibraryRef{{
			Name: library.Name, Path: library.Path, Digest: library.Digest, Version: "supplied-version",
		}},
		Inputs: []cudaregress.InputRef{{
			Name: input.Name, Path: input.Path, Digest: input.Digest, Seed: 17,
			ShapeDims: []int64{2, 3}, NumericalMode: "fp32",
		}},
		Oracle: cudaregress.CorrectnessOracle{
			Method: "digest-exact", ExpectedDigest: reference.Digest,
			ReferencePath: reference.Path, ReferenceDigest: reference.Digest,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return plan, inc.ObservedEnvironment
}

func sealReconstructionPlan(t *testing.T, plan *cudaregress.ExperimentPlan) {
	t.Helper()
	var err error
	plan.PlanDigest, err = cudaregress.ComputePlanDigest(plan)
	if err != nil {
		t.Fatal(err)
	}
}

func TestReproducerIntegritySuppliedClosure(t *testing.T) {
	plan, env := reconstructionPlan(t)
	if ok, blockers := cudaregress.CheckReconstruction(plan, env); !ok || len(blockers) != 0 {
		t.Fatalf("supplied valid closure refused: %v", blockers)
	}
	dir := filepath.Join(t.TempDir(), "export")
	if err := cudaregress.ExportMinimalReproducer(plan, dir); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	var manifest cudaregress.Manifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatal(err)
	}
	if !manifest.ArtifactClosureValid || manifest.Reconstructable || len(manifest.Blockers) != 1 || !strings.Contains(manifest.Blockers[0], "separately observed environment") {
		t.Fatalf("export invented observed readiness or lost verified closure: %+v", manifest)
	}
	if manifest.PlanID != plan.PlanID || manifest.PlanDigest != plan.PlanDigest || manifest.IncidentID != plan.IncidentID || manifest.TargetArchitecture != plan.DeviceConfig.ComputeCapability {
		t.Fatalf("manifest identity changed: %+v", manifest)
	}
	want := append([]cudaregress.ArtifactRef{}, plan.SourceArtifacts...)
	want = append(want, plan.BinaryArtifacts...)
	want = append(want,
		cudaregress.ArtifactRef{Name: plan.Inputs[0].Name, Path: plan.Inputs[0].Path, Digest: plan.Inputs[0].Digest, Role: "input"},
		cudaregress.ArtifactRef{Name: plan.LibraryClosure[0].Name, Path: plan.LibraryClosure[0].Path, Digest: plan.LibraryClosure[0].Digest, Role: "library"},
		cudaregress.ArtifactRef{Name: "oracle_reference", Path: plan.CorrectnessOracle.ReferencePath, Digest: plan.CorrectnessOracle.ReferenceDigest, Role: "reference"},
	)
	if !reflect.DeepEqual(manifest.Artifacts, want) {
		t.Fatalf("manifest changed or omitted supplied references: got %+v, want %+v", manifest.Artifacts, want)
	}
	data, err = os.ReadFile(filepath.Join(dir, "plan.json"))
	if err != nil {
		t.Fatal(err)
	}
	var exported cudaregress.ExperimentPlan
	if err := json.Unmarshal(data, &exported); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(&exported, plan) {
		t.Fatal("export changed plan metadata")
	}
	if ok, blockers := cudaregress.CheckReconstruction(&exported, env); !ok {
		t.Fatalf("exported plan failed supplied environment check: %v", blockers)
	}
	entries, err := os.ReadDir(dir)
	if err != nil || len(entries) != 3 {
		t.Fatalf("export copied artifacts or omitted documents: %v, %v", entries, err)
	}
}

func TestReproducerIntegrityRefusesMutationsBeforeWrites(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*testing.T, *cudaregress.ExperimentPlan)
		reseal bool
		want   error
	}{
		{"source bytes", func(t *testing.T, p *cudaregress.ExperimentPlan) { rewriteArtifact(t, p.SourceArtifacts[0].Path) }, false, cudaregress.ErrArtifactCorrupted},
		{"binary bytes", func(t *testing.T, p *cudaregress.ExperimentPlan) { rewriteArtifact(t, p.BinaryArtifacts[0].Path) }, false, cudaregress.ErrArtifactCorrupted},
		{"input bytes", func(t *testing.T, p *cudaregress.ExperimentPlan) { rewriteArtifact(t, p.Inputs[0].Path) }, false, cudaregress.ErrArtifactCorrupted},
		{"library bytes", func(t *testing.T, p *cudaregress.ExperimentPlan) { rewriteArtifact(t, p.LibraryClosure[0].Path) }, false, cudaregress.ErrArtifactCorrupted},
		{"oracle bytes", func(t *testing.T, p *cudaregress.ExperimentPlan) {
			rewriteArtifact(t, p.CorrectnessOracle.ReferencePath)
		}, false, cudaregress.ErrArtifactCorrupted},
		{"source digest", func(t *testing.T, p *cudaregress.ExperimentPlan) {
			p.SourceArtifacts[0].Digest = strings.Repeat("0", 64)
		}, true, cudaregress.ErrArtifactCorrupted},
		{"library digest", func(t *testing.T, p *cudaregress.ExperimentPlan) {
			p.LibraryClosure[0].Digest = strings.Repeat("0", 64)
		}, true, cudaregress.ErrArtifactCorrupted},
		{"oracle digest", func(t *testing.T, p *cudaregress.ExperimentPlan) {
			p.CorrectnessOracle.ReferenceDigest = strings.Repeat("0", 64)
		}, true, cudaregress.ErrArtifactCorrupted},
		{"input digest", func(t *testing.T, p *cudaregress.ExperimentPlan) { p.Inputs[0].Digest = strings.Repeat("0", 64) }, true, cudaregress.ErrArtifactCorrupted},
		{"missing library", func(t *testing.T, p *cudaregress.ExperimentPlan) {
			if err := os.Remove(p.LibraryClosure[0].Path); err != nil {
				t.Fatal(err)
			}
		}, false, cudaregress.ErrArtifactMissing},
		{"missing reference", func(t *testing.T, p *cudaregress.ExperimentPlan) {
			if err := os.Remove(p.CorrectnessOracle.ReferencePath); err != nil {
				t.Fatal(err)
			}
		}, false, cudaregress.ErrArtifactMissing},
		{"directory input", func(t *testing.T, p *cudaregress.ExperimentPlan) { p.Inputs[0].Path = t.TempDir() }, true, nil},
		{"inaccessible path", func(t *testing.T, p *cudaregress.ExperimentPlan) { p.SourceArtifacts[0].Path += "/child" }, true, nil},
		{"input without path", func(t *testing.T, p *cudaregress.ExperimentPlan) { p.Inputs[0].Path = "" }, true, nil},
		{"input without digest", func(t *testing.T, p *cudaregress.ExperimentPlan) { p.Inputs[0].Digest = "" }, true, nil},
		{"reference without path", func(t *testing.T, p *cudaregress.ExperimentPlan) { p.CorrectnessOracle.ReferencePath = "" }, true, nil},
		{"reference without digest", func(t *testing.T, p *cudaregress.ExperimentPlan) { p.CorrectnessOracle.ReferenceDigest = "" }, true, nil},
		{"plan digest", func(t *testing.T, p *cudaregress.ExperimentPlan) { p.PlanDigest = strings.Repeat("0", 64) }, false, cudaregress.ErrArtifactCorrupted},
		{"input seed", func(t *testing.T, p *cudaregress.ExperimentPlan) { p.Inputs[0].Seed++ }, false, cudaregress.ErrArtifactCorrupted},
		{"input shape", func(t *testing.T, p *cudaregress.ExperimentPlan) { p.Inputs[0].ShapeDims[0]++ }, false, cudaregress.ErrArtifactCorrupted},
		{"input numerical mode", func(t *testing.T, p *cudaregress.ExperimentPlan) { p.Inputs[0].NumericalMode = "fp64" }, false, cudaregress.ErrArtifactCorrupted},
		{"library version", func(t *testing.T, p *cudaregress.ExperimentPlan) { p.LibraryClosure[0].Version = "changed" }, false, cudaregress.ErrArtifactCorrupted},
		{"oracle expected output", func(t *testing.T, p *cudaregress.ExperimentPlan) { p.CorrectnessOracle.ExpectedDigest = "changed" }, false, cudaregress.ErrArtifactCorrupted},
		{"schema", func(t *testing.T, p *cudaregress.ExperimentPlan) { p.Version++ }, true, nil},
		{"identity", func(t *testing.T, p *cudaregress.ExperimentPlan) { p.PlanID = "" }, true, nil},
		{"round count", func(t *testing.T, p *cudaregress.ExperimentPlan) { p.Rounds = 0 }, true, nil},
		{"order", func(t *testing.T, p *cudaregress.ExperimentPlan) { p.ExecutionOrder = "unknown" }, true, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			plan, env := reconstructionPlan(t)
			tt.mutate(t, plan)
			if tt.reseal {
				sealReconstructionPlan(t, plan)
			}
			if err := cudaregress.ValidatePlan(plan); err == nil {
				t.Fatal("validation accepted mutation")
			}
			if ok, blockers := cudaregress.CheckReconstruction(plan, env); ok || len(blockers) == 0 {
				t.Fatalf("reconstruction accepted mutation: %v", blockers)
			}
			parent := t.TempDir()
			target := filepath.Join(parent, "must-not-exist", "export")
			err := cudaregress.ExportMinimalReproducer(plan, target)
			if err == nil || tt.want != nil && !errors.Is(err, tt.want) {
				t.Fatalf("export error = %v, want refusal wrapping %v", err, tt.want)
			}
			if _, err := os.Stat(filepath.Join(parent, "must-not-exist")); !os.IsNotExist(err) {
				t.Fatalf("invalid export created output directories: %v", err)
			}
			existing := t.TempDir()
			sentinel := filepath.Join(existing, "plan.json")
			if err := os.WriteFile(sentinel, []byte("preserve"), 0600); err != nil {
				t.Fatal(err)
			}
			if err := cudaregress.ExportMinimalReproducer(plan, existing); err == nil {
				t.Fatal("existing output accepted mutation")
			}
			data, err := os.ReadFile(sentinel)
			if err != nil || string(data) != "preserve" {
				t.Fatalf("invalid export changed existing output: %q, %v", data, err)
			}
			entries, err := os.ReadDir(existing)
			if err != nil || len(entries) != 1 {
				t.Fatalf("invalid export added output: %v, %v", entries, err)
			}
		})
	}
}

func rewriteArtifact(t *testing.T, path string) {
	t.Helper()
	if err := os.WriteFile(path, []byte("different bytes"), 0600); err != nil {
		t.Fatal(err)
	}
}

func TestReproducerIntegrityEnvironmentAndSeedOnlyInput(t *testing.T) {
	plan, env := reconstructionPlan(t)
	for _, change := range []func(*cudaregress.Environment){
		func(e *cudaregress.Environment) { e.DeviceIndex++ },
		func(e *cudaregress.Environment) { e.DeviceUUID = "" },
		func(e *cudaregress.Environment) { e.Model = "" },
		func(e *cudaregress.Environment) { e.ComputeCapability = "" },
		func(e *cudaregress.Environment) { e.Driver = "" },
		func(e *cudaregress.Environment) { e.CUDA = "" },
	} {
		drifted := env
		change(&drifted)
		if ok, blockers := cudaregress.CheckReconstruction(plan, drifted); ok || len(blockers) != 1 || !strings.Contains(blockers[0], cudaregress.ErrEnvironmentDrift.Error()) {
			t.Fatalf("environment mismatch lost its distinct blocker: %v", blockers)
		}
	}
	// A seed-only declaration is retained as plan metadata, not a file reference.
	plan.Inputs[0].Path, plan.Inputs[0].Digest = "", ""
	sealReconstructionPlan(t, plan)
	if ok, blockers := cudaregress.CheckReconstruction(plan, env); !ok {
		t.Fatalf("seed-only declaration refused: %v", blockers)
	}
	dir := t.TempDir()
	if err := cudaregress.ExportMinimalReproducer(plan, dir); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	var manifest cudaregress.Manifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatal(err)
	}
	if len(manifest.Artifacts) != 4 || !manifest.ArtifactClosureValid || manifest.Reconstructable {
		t.Fatalf("seed invented a verified file or readiness: %+v", manifest)
	}
}

func TestReproducerIntegrityArgumentAndOutputErrors(t *testing.T) {
	if ok, blockers := cudaregress.CheckReconstruction(nil, cudaregress.Environment{}); ok || len(blockers) == 0 {
		t.Fatal("nil plan accepted")
	}
	target := filepath.Join(t.TempDir(), "export")
	if err := cudaregress.ExportMinimalReproducer(nil, target); err == nil {
		t.Fatal("nil plan exported")
	}
	if _, err := os.Stat(target); !os.IsNotExist(err) {
		t.Fatalf("nil plan created output: %v", err)
	}
	plan, _ := reconstructionPlan(t)
	if err := cudaregress.ExportMinimalReproducer(plan, ""); err == nil {
		t.Fatal("empty output accepted")
	}
	file := filepath.Join(t.TempDir(), "file")
	if err := os.WriteFile(file, []byte("preserve"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := cudaregress.ExportMinimalReproducer(plan, file); err == nil {
		t.Fatal("output filesystem failure hidden")
	}
}
