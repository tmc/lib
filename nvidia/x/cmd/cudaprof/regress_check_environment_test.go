package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudaregress"
)

func regressCheckFixture(t *testing.T) (string, string, *cudaregress.ExperimentPlan, observedEnvironmentFile) {
	t.Helper()
	if driver.Loaded() {
		t.Fatal("check fixture unexpectedly loaded driver")
	}
	t.Cleanup(func() {
		if driver.Loaded() {
			t.Error("check command loaded driver")
		}
	})
	dir := t.TempDir()
	artifact := func(name string) cudaregress.ArtifactRef {
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte("supplied "+name+" bytes, never executed"), 0600); err != nil {
			t.Fatal(err)
		}
		digest, size, err := cudaregress.ComputeFileDigest(path)
		if err != nil {
			t.Fatal(err)
		}
		return cudaregress.ArtifactRef{Name: name, Path: path, Digest: digest, SizeBytes: size, Role: name}
	}
	source, binary, library, input, reference := artifact("source"), artifact("binary"), artifact("library"), artifact("input"), artifact("reference")
	observed := observedEnvironmentFile{
		Version: 1, Provenance: "independently supplied fixture observation",
		Environment: cudaregress.Environment{DeviceIndex: 2, DeviceUUID: "supplied-device", Model: "supplied-model", ComputeCapability: "sm_90", Driver: "supplied-driver", CUDA: "supplied-toolkit"},
	}
	plan, _, err := cudaregress.NewPlanFromIncident(cudaregress.Incident{
		ID: "supplied-incident", Admitted: true, ObservedEnvironment: observed.Environment,
	}, cudaregress.PlanOptions{
		Sources: []cudaregress.ArtifactRef{source}, Binaries: []cudaregress.ArtifactRef{binary},
		Libraries: []cudaregress.LibraryRef{{Name: library.Name, Path: library.Path, Digest: library.Digest}},
		Inputs:    []cudaregress.InputRef{{Name: input.Name, Path: input.Path, Digest: input.Digest, Seed: 17, ShapeDims: []int64{2, 3}}},
		Oracle:    cudaregress.CorrectnessOracle{Method: "digest-exact", ExpectedDigest: reference.Digest, ReferencePath: reference.Path, ReferenceDigest: reference.Digest},
	})
	if err != nil {
		t.Fatal(err)
	}
	planPath, envPath := filepath.Join(dir, "plan.json"), filepath.Join(dir, "environment.json")
	writeJSONFile(t, planPath, plan)
	writeJSONFile(t, envPath, observed)
	return planPath, envPath, plan, observed
}

func TestRegressCheckEnvironmentPositive(t *testing.T) {
	planPath, envPath, plan, observed := regressCheckFixture(t)
	var output bytes.Buffer
	if err := runRegressCommand([]string{"check", "-plan", planPath, "-environment", envPath}, &output); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(output.String(), "Plan "+plan.PlanID+": reconstructable (") || !strings.Contains(output.String(), "caller-supplied environment") || !strings.Contains(output.String(), observed.Provenance) {
		t.Fatalf("valid explicit environment was not accepted with provenance: %s", output.String())
	}
}

func TestRegressCheckEnvironmentRequired(t *testing.T) {
	planPath, _, plan, _ := regressCheckFixture(t)
	// The old CLI fabricated exactly these requirements as its observation.
	plan.DeviceConfig.DeviceIndex, plan.DeviceConfig.DeviceUUID = 0, ""
	var err error
	plan.PlanDigest, err = cudaregress.ComputePlanDigest(plan)
	if err != nil {
		t.Fatal(err)
	}
	writeJSONFile(t, planPath, plan)
	var output bytes.Buffer
	err = runRegressCommand([]string{"check", "-plan", planPath}, &output)
	if err == nil || !strings.Contains(err.Error(), "-environment") || output.Len() != 0 {
		t.Fatalf("missing observation manufactured readiness: output=%q error=%v", output.String(), err)
	}
}

func TestRegressCheckEnvironmentInputRefusals(t *testing.T) {
	for _, test := range []struct{ name, raw string }{
		{"malformed", "{"},
		{"null", "null"},
		{"unknown field", `{"version":1,"provenance":"fixture","environment":{},"unknown":true}`},
		{"duplicate version", `{"version":1,"version":1,"provenance":"fixture","environment":{}}`},
		{"case alias", `{"version":1,"Version":1,"provenance":"fixture","environment":{}}`},
		{"trailing JSON", `{"version":1,"provenance":"fixture","environment":{}} {}`},
		{"unsupported version", `{"version":2,"provenance":"fixture","environment":{}}`},
		{"missing version", `{"provenance":"fixture","environment":{}}`},
		{"missing provenance", `{"version":1,"environment":{}}`},
		{"blank provenance", `{"version":1,"provenance":" \n\t ","environment":{}}`},
		{"missing environment", `{"version":1,"provenance":"fixture"}`},
		{"null environment", `{"version":1,"provenance":"fixture","environment":null}`},
		{"empty environment", `{"version":1,"provenance":"fixture","environment":{}}`},
		{"wrong type", `{"version":1,"provenance":"fixture","environment":{"device_index":"2"}}`},
		{"fractional index", `{"version":1,"provenance":"fixture","environment":{"device_index":2.5}}`},
		{"nested duplicate", `{"version":1,"provenance":"fixture","environment":{"model":"a","model":"b"}}`},
	} {
		t.Run(test.name, func(t *testing.T) {
			planPath, envPath, _, _ := regressCheckFixture(t)
			if err := os.WriteFile(envPath, []byte(test.raw), 0600); err != nil {
				t.Fatal(err)
			}
			var output bytes.Buffer
			if err := runRegressCheckCommand([]string{"-plan", planPath, "-environment", envPath}, &output); err == nil || output.Len() != 0 {
				t.Fatalf("invalid observation manufactured readiness: output=%q error=%v", output.String(), err)
			}
		})
	}
	planPath, envPath, _, _ := regressCheckFixture(t)
	if err := os.Remove(envPath); err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	if err := runRegressCheckCommand([]string{"-plan", planPath, "-environment", envPath}, &output); !errors.Is(err, os.ErrNotExist) || output.Len() != 0 {
		t.Fatalf("unavailable observation lost filesystem error: output=%q error=%v", output.String(), err)
	}
}

func TestRegressCheckEnvironmentDrift(t *testing.T) {
	for _, test := range []struct {
		name   string
		mutate func(*cudaregress.Environment)
	}{
		{"index", func(e *cudaregress.Environment) { e.DeviceIndex++ }},
		{"uuid", func(e *cudaregress.Environment) { e.DeviceUUID = "" }},
		{"model", func(e *cudaregress.Environment) { e.Model = "different" }},
		{"architecture", func(e *cudaregress.Environment) { e.ComputeCapability = "" }},
		{"driver", func(e *cudaregress.Environment) { e.Driver = "different" }},
		{"toolkit", func(e *cudaregress.Environment) { e.CUDA = "" }},
	} {
		t.Run(test.name, func(t *testing.T) {
			planPath, envPath, _, observed := regressCheckFixture(t)
			test.mutate(&observed.Environment)
			writeJSONFile(t, envPath, observed)
			var output bytes.Buffer
			if err := runRegressCheckCommand([]string{"-plan", planPath, "-environment", envPath}, &output); !errors.Is(err, cudaregress.ErrEnvironmentDrift) || output.Len() != 0 {
				t.Fatalf("incompatible observation lost drift refusal: output=%q error=%v", output.String(), err)
			}
		})
	}
}

func TestRegressCheckEnvironmentUnavailableAndHostOnly(t *testing.T) {
	planPath, envPath, plan, observed := regressCheckFixture(t)
	plan.DeviceConfig = cudaregress.DeviceConfig{}
	var err error
	plan.PlanDigest, err = cudaregress.ComputePlanDigest(plan)
	if err != nil {
		t.Fatal(err)
	}
	writeJSONFile(t, planPath, plan)
	observed.Environment = cudaregress.Environment{}
	writeJSONFile(t, envPath, observed)
	var output bytes.Buffer
	if err := runRegressCheckCommand([]string{"-plan", planPath, "-environment", envPath}, &output); err == nil || !strings.Contains(err.Error(), "unavailable") || output.Len() != 0 {
		t.Fatalf("absent requirements invented an observation: output=%q error=%v", output.String(), err)
	}
	observed.Environment.HostArch = "supplied-host-architecture"
	writeJSONFile(t, envPath, observed)
	if err := runRegressCheckCommand([]string{"-plan", planPath, "-environment", envPath}, &output); err != nil || !strings.Contains(output.String(), ": reconstructable (") {
		t.Fatalf("valid caller-supplied host observation refused: output=%q error=%v", output.String(), err)
	}
}

func TestRegressCheckEnvironmentAuthenticatesClosure(t *testing.T) {
	for _, role := range []string{"source", "binary", "input", "library", "reference", "missing", "plan digest", "schema"} {
		t.Run(role, func(t *testing.T) {
			planPath, envPath, plan, _ := regressCheckFixture(t)
			paths := map[string]string{"source": plan.SourceArtifacts[0].Path, "binary": plan.BinaryArtifacts[0].Path, "input": plan.Inputs[0].Path, "library": plan.LibraryClosure[0].Path, "reference": plan.CorrectnessOracle.ReferencePath}
			want := cudaregress.ErrArtifactCorrupted
			switch role {
			case "missing":
				if err := os.Remove(paths["binary"]); err != nil {
					t.Fatal(err)
				}
				want = cudaregress.ErrArtifactMissing
			case "plan digest":
				plan.PlanDigest = strings.Repeat("0", 64)
				writeJSONFile(t, planPath, plan)
			case "schema":
				plan.Version++
				writeJSONFile(t, planPath, plan)
				want = nil
			default:
				if err := os.WriteFile(paths[role], []byte("changed bytes"), 0600); err != nil {
					t.Fatal(err)
				}
			}
			var output bytes.Buffer
			if _, err := readRegressPlanFile(planPath); err == nil || want != nil && !errors.Is(err, want) {
				t.Fatalf("shared plan reader accepted invalid closure or lost cause: %v", err)
			}
			err := runRegressCheckCommand([]string{"-plan", planPath, "-environment", envPath}, &output)
			if err == nil || want != nil && !errors.Is(err, want) || output.Len() != 0 {
				t.Fatalf("invalid closure manufactured readiness or lost cause: output=%q error=%v", output.String(), err)
			}
		})
	}
}

func TestRegressPlanFileBundleRefusals(t *testing.T) {
	for _, name := range []string{"missing plan", "missing receipt", "null plan", "null receipt", "plan id", "plan digest", "incident id", "plan content", "plan schema", "unknown", "ambiguous", "case alias", "duplicate", "receipt unknown"} {
		t.Run(name, func(t *testing.T) {
			planPath, envPath, plan, _ := regressCheckFixture(t)
			receipt := &cudaregress.PlanReceipt{PlanID: plan.PlanID, PlanDigest: plan.PlanDigest, IncidentID: plan.IncidentID, ArtifactClosureValid: true}
			bundle := map[string]any{"plan": plan, "receipt": receipt}
			switch name {
			case "missing plan":
				delete(bundle, "plan")
			case "missing receipt":
				delete(bundle, "receipt")
			case "null plan":
				bundle["plan"] = nil
			case "null receipt":
				bundle["receipt"] = nil
			case "plan id":
				receipt.PlanID = "different"
			case "plan digest":
				receipt.PlanDigest = strings.Repeat("0", 64)
			case "incident id":
				receipt.IncidentID = "different"
			case "plan content":
				plan.Rounds++
			case "plan schema":
				plan.Version++
			case "unknown":
				bundle["unknown"] = true
			case "ambiguous":
				bundle["version"] = 1
			case "case alias":
				bundle["Plan"] = plan
				delete(bundle, "plan")
			case "receipt unknown":
				bundle["receipt"] = map[string]any{"plan_id": plan.PlanID, "plan_digest": plan.PlanDigest, "incident_id": plan.IncidentID, "unknown": true}
			}
			data, err := json.Marshal(bundle)
			if err != nil {
				t.Fatal(err)
			}
			if name == "duplicate" {
				data = append(data[:len(data)-1], []byte(`,"plan":null}`)...)
			}
			if err := os.WriteFile(planPath, data, 0600); err != nil {
				t.Fatal(err)
			}
			dir := t.TempDir()
			for _, args := range [][]string{
				{"check", "-plan", planPath, "-environment", envPath},
				{"export", "-plan", planPath, "-o", filepath.Join(dir, "export")},
				{"replay", "-plan", planPath, "-environment", envPath, "-spec", filepath.Join(dir, "missing-spec"), "-o", filepath.Join(dir, "replay")},
			} {
				var output bytes.Buffer
				err := runRegressCommand(args, &output)
				if err == nil || !strings.Contains(err.Error(), "read plan") || output.Len() != 0 {
					t.Fatalf("invalid bundle reached consumer: args=%v output=%q error=%v", args, output.String(), err)
				}
			}
			entries, err := os.ReadDir(dir)
			if err != nil || len(entries) != 0 {
				t.Fatalf("invalid bundle wrote output: %v %v", entries, err)
			}
		})
	}
}

func TestRegressPlanFileStandaloneConsumers(t *testing.T) {
	planPath, envPath, plan, _ := regressCheckFixture(t)
	dir := t.TempDir()
	var output bytes.Buffer
	if err := runRegressCommand([]string{"export", "-plan", planPath, "-o", filepath.Join(dir, "export")}, &output); err != nil {
		t.Fatal(err)
	}
	exported, err := readRegressPlanFile(filepath.Join(dir, "export", "plan.json"))
	if err != nil || exported.PlanID != plan.PlanID || exported.PlanDigest != plan.PlanDigest {
		t.Fatalf("standalone export lost identity: %v %+v", err, exported)
	}
	output.Reset()
	err = runRegressCommand([]string{"replay", "-plan", planPath, "-environment", envPath, "-spec", filepath.Join(dir, "missing-spec"), "-o", filepath.Join(dir, "replay")}, &output)
	if !errors.Is(err, os.ErrNotExist) || !strings.Contains(err.Error(), "open execution spec") || output.Len() != 0 {
		t.Fatalf("standalone replay read reached wrong boundary: %v %q", err, output.String())
	}
	if _, err := os.Stat(filepath.Join(dir, "replay")); !os.IsNotExist(err) {
		t.Fatalf("unexecuted replay wrote output: %v", err)
	}
}

func TestRegressCheckEnvironmentFileBounds(t *testing.T) {
	planPath, envPath, _, _ := regressCheckFixture(t)
	large := filepath.Join(t.TempDir(), "oversized.json")
	f, err := os.Create(large)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Truncate(maxReplayInputBytes + 1); err != nil {
		f.Close()
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	for _, args := range [][]string{
		{"-plan", planPath, "-environment", large},
		{"-plan", large, "-environment", envPath},
	} {
		var output bytes.Buffer
		if err := runRegressCheckCommand(args, &output); err == nil || !strings.Contains(err.Error(), "input byte limit exceeded") || output.Len() != 0 {
			t.Fatalf("file bound not enforced: output=%q error=%v", output.String(), err)
		}
	}
}
