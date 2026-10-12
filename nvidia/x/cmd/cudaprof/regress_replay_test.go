package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaevidence"
	"github.com/tmc/lib/nvidia/x/cudaregress"
	"github.com/tmc/lib/nvidia/x/cudatrial"
)

func TestRegressReplayCommandWritesBoundReceipts(t *testing.T) {
	dir := t.TempDir()
	values := []float64{2}
	valueJSON, _ := json.Marshal(values)
	valueDigest := sha256.Sum256(valueJSON)
	outputDigest := hex.EncodeToString(valueDigest[:])
	currentBinary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	binary := filepath.Join(dir, "runner.test")
	source, err := os.Open(currentBinary)
	if err != nil {
		t.Fatal(err)
	}
	dest, err := os.OpenFile(binary, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0700)
	if err != nil {
		source.Close()
		t.Fatal(err)
	}
	_, copyErr := io.Copy(dest, source)
	closeDestErr := dest.Close()
	closeSourceErr := source.Close()
	if err := errors.Join(copyErr, closeDestErr, closeSourceErr); err != nil {
		t.Fatalf("copy task-owned helper executable: %v", err)
	}
	if err := os.Chmod(binary, 0700); err != nil {
		t.Fatal(err)
	}
	binaryDigest, binarySize, err := cudaregress.ComputeFileDigest(binary)
	if err != nil {
		t.Fatal(err)
	}
	inputPath := filepath.Join(dir, "input.dat")
	if err := os.WriteFile(inputPath, []byte("input fixture"), 0600); err != nil {
		t.Fatal(err)
	}
	inputDigest, _, err := cudaregress.ComputeFileDigest(inputPath)
	if err != nil {
		t.Fatal(err)
	}
	libraryPath := filepath.Join(dir, "library.dat")
	if err := os.WriteFile(libraryPath, []byte("library fixture"), 0600); err != nil {
		t.Fatal(err)
	}
	libraryDigest, _, err := cudaregress.ComputeFileDigest(libraryPath)
	if err != nil {
		t.Fatal(err)
	}
	plan := cudaregress.ExperimentPlan{Version: cudaregress.Version, PlanID: "plan-cli", IncidentID: "incident-cli", Rounds: 1, ExecutionOrder: "ABBA", BinaryArtifacts: []cudaregress.ArtifactRef{{Name: "runner", Path: binary, Digest: binaryDigest, SizeBytes: binarySize, Role: "binary"}}, Inputs: []cudaregress.InputRef{{Name: "input", Path: inputPath, Digest: inputDigest}}, LibraryClosure: []cudaregress.LibraryRef{{Name: "library", Path: libraryPath, Digest: libraryDigest}}, CorrectnessOracle: cudaregress.CorrectnessOracle{Method: "digest-exact", ExpectedDigest: outputDigest}, TimingScope: cudaregress.TimingScope{Channel: cudatrial.TimingWallElapsed}, StoppingRules: cudaregress.StoppingRules{MaxTrials: 2, RelativeThreshold: 0.1, MinSamples: 1}}
	plan.PlanDigest, err = cudaregress.ComputePlanDigest(&plan)
	if err != nil {
		t.Fatal(err)
	}
	planPath := filepath.Join(dir, "plan.json")
	writeJSONFile(t, planPath, plan)
	args := []string{"command-replay-child-cli"}
	spec := cudaregress.ExecutionSpec{Version: 1, PlanID: plan.PlanID, PlanDigest: plan.PlanDigest, Mode: "host_fixture", WorkingDirectory: dir, Environment: map[string]string{}, TimeoutNanos: int64(10 * time.Second), MaxOutputBytes: 4096, Baseline: cudaregress.CommandArm{Path: binary, SHA256: binaryDigest, Args: args}, Candidate: cudaregress.CommandArm{Path: binary, SHA256: binaryDigest, Args: args}, Inputs: []cudaregress.CommandArtifact{{Name: "input", Path: inputPath, SHA256: inputDigest}}, Libraries: []cudaregress.CommandArtifact{{Name: "library", Path: libraryPath, SHA256: libraryDigest}}}
	spec.SpecDigest, err = cudaregress.ComputeExecutionSpecDigest(spec)
	if err != nil {
		t.Fatal(err)
	}
	specPath := filepath.Join(dir, "spec.json")
	writeJSONFile(t, specPath, spec)
	envPath := filepath.Join(dir, "observed.json")
	writeJSONFile(t, envPath, observedEnvironmentFile{Version: 1, Environment: cudaregress.Environment{DeviceIndex: 0}, Provenance: "test-fixture"})
	outDir := filepath.Join(dir, "receipts")
	var stdout bytes.Buffer
	err = runRegressCommand([]string{"replay", "-plan", planPath, "-spec", specPath, "-environment", envPath, "-o", outDir}, &stdout)
	if err != nil {
		t.Fatal(err)
	}
	var command replayCommandReceipt
	if err := readStrictFile(filepath.Join(outDir, "command.json"), &command); err != nil {
		t.Fatal(err)
	}
	if command.SpecDigest != spec.SpecDigest || command.Execution == nil || command.Comparison == nil || len(command.Attempts) != 2 || command.ReplayError != "" {
		t.Fatalf("unexpected replay receipt: %+v", command)
	}
	if len(command.ArtifactClosure) != 3 {
		t.Fatalf("command receipt lacks complete frozen artifact closure: %+v", command.ArtifactClosure)
	}
	wantClosure := map[string]string{"binary:runner:" + binary: binaryDigest, "input:input:" + inputPath: inputDigest, "library:library:" + libraryPath: libraryDigest}
	checkClosure := func(name string, closure []replayArtifactReceipt) {
		t.Helper()
		remaining := make(map[string]string, len(wantClosure))
		for key, digest := range wantClosure {
			remaining[key] = digest
		}
		if len(closure) != len(remaining) {
			t.Fatalf("%s closure length = %d, want %d: %+v", name, len(closure), len(remaining), closure)
		}
		for _, artifact := range closure {
			key := artifact.Role + ":" + artifact.Name + ":" + artifact.Path
			if remaining[key] != artifact.SHA256 {
				t.Fatalf("%s closure entry not from frozen plan: %+v", name, artifact)
			}
			delete(remaining, key)
		}
		if len(remaining) != 0 {
			t.Fatalf("%s closure omitted plan entries: %+v", name, remaining)
		}
	}
	checkClosure("command", command.ArtifactClosure)
	for _, name := range []string{"execution.json", "comparison.json"} {
		var bound replayBoundReceipt
		if err := readStrictFile(filepath.Join(outDir, name), &bound); err != nil {
			t.Fatal(err)
		}
		if bound.PlanDigest != plan.PlanDigest || bound.SpecDigest != spec.SpecDigest {
			t.Fatalf("unbound receipt %s: %+v", name, bound)
		}
		if len(bound.ArtifactClosure) != 3 {
			t.Fatalf("standalone receipt %s lacks complete frozen closure: %+v", name, bound.ArtifactClosure)
		}
		checkClosure(name, bound.ArtifactClosure)
	}
	if stdout.Len() == 0 {
		t.Fatal("missing command result")
	}
	secondOutDir := filepath.Join(dir, "receipts-rerun")
	var secondStdout bytes.Buffer
	err = runRegressCommand([]string{"replay", "-plan", planPath, "-spec", specPath, "-environment", envPath, "-o", secondOutDir}, &secondStdout)
	if err != nil {
		t.Fatal(err)
	}
	var rerun replayCommandReceipt
	if err := readStrictFile(filepath.Join(secondOutDir, "command.json"), &rerun); err != nil {
		t.Fatal(err)
	}
	if rerun.PlanDigest != command.PlanDigest || rerun.SpecDigest != command.SpecDigest || rerun.Execution == nil || command.Execution.OutputDigests["candidate"] != rerun.Execution.OutputDigests["candidate"] || rerun.Comparison == nil || command.Comparison.Verdict != rerun.Comparison.Verdict {
		t.Fatalf("independent rerun does not reconstruct the declared case: first=%+v second=%+v", command, rerun)
	}
	spec.Baseline.Args = []string{"command-replay-child-cli-fail"}
	spec.Candidate.Args = []string{"command-replay-child-cli-fail"}
	spec.SpecDigest, err = cudaregress.ComputeExecutionSpecDigest(spec)
	if err != nil {
		t.Fatal(err)
	}
	writeJSONFile(t, specPath, spec)
	partialDir := filepath.Join(dir, "receipts-partial")
	var partialStdout bytes.Buffer
	historyOut := filepath.Join(dir, "partial-history.ndjson")
	err = runRegressCommand([]string{"replay", "-plan", planPath, "-spec", specPath, "-environment", envPath, "-o", partialDir, "-history-template", filepath.Join(dir, "unused-template.json"), "-history-binding", filepath.Join(dir, "unused-binding.json"), "-history-out", historyOut}, &partialStdout)
	if err == nil {
		t.Fatal("partial child run returned success")
	}
	var partial replayCommandReceipt
	if readErr := readStrictFile(filepath.Join(partialDir, "command.json"), &partial); readErr != nil {
		t.Fatal(readErr)
	}
	if partial.Execution == nil || !partial.Execution.Partial || partial.Comparison == nil || partial.Comparison.Verdict != "unknown" || partial.ReplayError == "" || partial.HistoryWithheldReason == "" || len(partial.Attempts) != 2 || partial.Attempts[0].ExitCode != 9 || partial.Attempts[1].ExitCode != 9 || partial.Attempts[0].StdoutSHA256 == "" || partial.Attempts[0].StderrSHA256 == "" || partialStdout.String() != "" {
		t.Fatalf("partial execution did not retain evidence or report failure: err=%v receipt=%+v stdout=%q", err, partial, partialStdout.String())
	}
	checkClosure("partial command", partial.ArtifactClosure)
	if _, statErr := os.Stat(filepath.Join(partialDir, "execution.json")); statErr != nil {
		t.Fatalf("partial execution receipt missing: %v", statErr)
	}
	if _, statErr := os.Stat(filepath.Join(partialDir, "comparison.json")); statErr != nil {
		t.Fatalf("partial comparison receipt missing: %v", statErr)
	}
	for _, name := range []string{"execution.json", "comparison.json"} {
		var bound replayBoundReceipt
		if readErr := readStrictFile(filepath.Join(partialDir, name), &bound); readErr != nil {
			t.Fatal(readErr)
		}
		checkClosure("partial "+name, bound.ArtifactClosure)
	}
	if _, statErr := os.Stat(historyOut); !os.IsNotExist(statErr) {
		t.Fatalf("partial execution wrote a history snapshot: stat err=%v", statErr)
	}
}

func TestReadReferenceValuesDistinguishesNullFromZero(t *testing.T) {
	path := filepath.Join(t.TempDir(), "reference.json")
	if err := os.WriteFile(path, []byte(`[0,2.5]`), 0600); err != nil {
		t.Fatal(err)
	}
	values, err := readReferenceValues(path)
	if err != nil || len(values) != 2 || values[0] != 0 || values[1] != 2.5 {
		t.Fatalf("numeric zero reference changed: %v %v", values, err)
	}
	for _, bad := range []string{`[null,2.5]`, `["0",2.5]`, `[true,2.5]`} {
		if err := os.WriteFile(path, []byte(bad), 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := readReferenceValues(path); err == nil {
			t.Fatalf("invalid reference accepted: %s", bad)
		}
	}
}

func TestReadStrictFileDelegatesCustomEvidenceSchema(t *testing.T) {
	type evidenceProbe struct {
		Fact struct {
			Value cudaevidence.Value `json:"value"`
		} `json:"fact"`
	}
	identity := `"identity":{"artifact_digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","entry":"kernel","target":"sm_90","compiler_key":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","workload_key":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"}`
	valid := `{"fact":{"value":{"class":"measured","number":0,"unit":"ns","basis":"fixture","source":{` + identity + `,"producer":"test","locator":"busy"},"loss":{"total":1}}}}`
	path := filepath.Join(t.TempDir(), "value.json")
	if err := os.WriteFile(path, []byte(valid), 0600); err != nil {
		t.Fatal(err)
	}
	var got evidenceProbe
	if err := readStrictFile(path, &got); err != nil {
		t.Fatalf("owner-validated custom Value schema rejected: %v", err)
	}
	if value := got.Fact.Value.Record(); value.Class != cudaevidence.Measured || value.Number == nil || *value.Number != 0 {
		t.Fatalf("custom value changed: %+v", value)
	}
	for _, bad := range []string{
		strings.Replace(valid, `"class":"measured"`, `"class":"future"`, 1),
		strings.Replace(valid, `"class":"measured"`, `"Class":"measured"`, 1),
		strings.Replace(valid, `"class":"measured"`, `"class":"measured","class":"measured"`, 1),
	} {
		if err := os.WriteFile(path, []byte(bad), 0600); err != nil {
			t.Fatal(err)
		}
		if err := readStrictFile(path, &got); err == nil {
			t.Fatalf("invalid custom Value accepted: %s", bad)
		}
	}
}

func init() {
	if len(os.Args) == 0 || (os.Args[len(os.Args)-1] != "command-replay-child-cli" && os.Args[len(os.Args)-1] != "command-replay-child-cli-fail") {
		return
	}
	if os.Args[len(os.Args)-1] == "command-replay-child-cli-fail" {
		fmt.Fprintln(os.Stderr, "partial child fixture")
	}
	digest, err := cudaregress.OutputDigest([]float64{2})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	fmt.Fprintf(os.Stdout, "{\"version\":1,\"output_values\":[2],\"output_sha256\":%q,\"timing\":{\"kind\":\"wall_elapsed\",\"status\":\"measured\",\"unit\":\"ns\",\"count\":42}}\n", digest)
	if os.Args[len(os.Args)-1] == "command-replay-child-cli-fail" {
		os.Exit(9)
	}
	os.Exit(0)
}

func writeJSONFile(t *testing.T, path string, value any) {
	t.Helper()
	b, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0600); err != nil {
		t.Fatal(err)
	}
}
