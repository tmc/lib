package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaevidence"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaprofilehistory"
	"github.com/tmc/lib/nvidia/x/cudaregress"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

type composedHostDelayFixture struct {
	reconstructFixture
	expectedOutput       []float64
	expectedOutputDigest string
	wrongOutputDigest    string
	ptxArtifact          *cudakernel.Artifact
	compilerInputKey     string
	compiler             cudacache.CompilerIdentity
	historyWorkload      cudacache.WorkloadIdentity
}

func newComposedHostDelayFixture(t *testing.T) composedHostDelayFixture {
	t.Helper()
	dir := t.TempDir()
	f := cudadiag.BuildSyntheticFixture(cudadiag.InjectionHostDelay)
	traceBytes := []byte("synthetic host-delay independent trace receipt input")
	rawBytes := []byte("synthetic host-delay source observation bytes")
	transformBytes := []byte("synthetic host-delay identity transform bytes")
	traceHash := reconstructTestHash(traceBytes)
	rawHash := reconstructTestHash(rawBytes)
	transformHash := reconstructTestHash(transformBytes)
	f.Input.Receipt.InputSHA256 = traceHash
	f.Input.Receipt.CaptureBounds = &tracecontract.CaptureBounds{Start: 1000, End: 200000, Clock: "monotonic"}

	assignment := nvtelemetry.EntityAssignment{
		Process: nvtelemetry.EntityProcess{
			Process:           f.Workload.Process,
			BootID:            "synthetic-boot",
			NodeID:            "synthetic-node",
			PIDNamespace:      "synthetic-pidns",
			HasExecGeneration: true,
			ExecGeneration:    1,
		},
		Device: nvtelemetry.DeviceMetadata{
			UUID:          "GPU-synthetic-host-delay",
			Model:         "synthetic device",
			DriverVersion: "synthetic driver",
			Index:         0,
			MIGScope:      nvtelemetry.MIGScopeDisabled,
		},
		Owner:          nvtelemetry.EntityOwner{AllocationID: "synthetic-allocation"},
		Source:         "synthetic host-delay fixture",
		SourceRevision: "1",
		State:          "assigned",
		Authoritative:  true,
		ClockDomain:    "monotonic",
		ClockEpoch:     "reconstruct-epoch",
		StartNanos:     0,
		EndNanos:       1_000_000,
		Sharing:        "exclusive",
	}
	observer := nvtelemetry.Observer{
		ID:         "synthetic-observer",
		Version:    "1",
		Mode:       "activity",
		Pass:       "single",
		Population: "all synthetic records",
		Filter:     "none",
		Replay:     "none",
	}

	var readings []nvtelemetry.Reading
	add := func(name, family, kind string, span tracecontract.Interval) *nvtelemetry.Observation {
		uncertainty := int64(10)
		o := &nvtelemetry.Observation{
			Family:          family,
			SourceClass:     "synthetic",
			Producer:        "synthetic-producer",
			ProducerVersion: "1",
			ProducerSchema:  "synthetic.v1",
			RawSHA256:       rawHash,
			Transform:       "identity",
			TransformSHA256: transformHash,
			Assignment:      &assignment,
			Workload:        &f.Workload,
			OpID:            f.OpID,
			Interval:        &span,
			ClockEpoch:      "reconstruct-epoch",
			UncertaintyNS:   &uncertainty,
			Observer:        observer,
			Kind:            kind,
			TID:             f.Input.Sched.TID,
		}
		zero := float64(0)
		readings = append(readings, nvtelemetry.Reading{
			Name:        name,
			Unit:        "ns",
			DeviceUUID:  assignment.Device.UUID,
			Backend:     "synthetic-producer",
			Evidence:    nvtelemetry.EvidenceMeasured,
			Number:      &nvtelemetry.Number{Float: &zero},
			Observation: o,
		})
		return o
	}

	host := add("cudaLaunchKernel", "host", "host_call", f.Input.HostCalls[0].Span)
	host.HasReturn = true
	contextID := cudaworkload.ContextStreamID{Context: 1, Stream: 1}
	host.Host = &cudaworkload.HostSubmission{
		Process:             f.Workload.Process,
		OpID:                f.OpID,
		Workload:            f.Workload,
		NativeCorrelationID: 7,
		Provenance:          cudaworkload.ProvenanceSynthetic,
		Context:             contextID,
		TID:                 f.Input.Sched.TID,
		TimestampNs:         uint64(host.Interval.Start),
		TimestampClock:      "monotonic",
		TimestampEpoch:      "reconstruct-epoch",
	}
	device := add("gemm_kernel", "activity", "kernel", f.Input.Kernels[0].Span)
	device.Device = &cudaworkload.DeviceActivity{
		Process:               f.Workload.Process,
		NativeCorrelationID:   7,
		ExternalCorrelationID: uint64(f.OpID),
		Context:               contextID,
		StartNs:               uint64(device.Interval.Start),
		EndNs:                 uint64(device.Interval.End),
		Kind:                  "kernel",
		Name:                  "gemm_kernel",
	}
	for _, interval := range f.Input.Sched.Intervals {
		add("scheduler", "sched", "sched_"+string(interval.State), interval.Span)
	}
	for _, family := range []string{"nvml", "dcgm", "managed", "topology", "workload", "nccl"} {
		add(family+"_zero", family, "gauge", f.Input.WallInterval)
	}

	clock := &reconstructTestClock{now: 1000}
	sampler := &reconstructTestSampler{readings: readings, clock: clock, device: assignment.Device}
	var stream bytes.Buffer
	config := nvtelemetry.CollectorConfig{SchemaVersion: nvtelemetry.SchemaVersionV3, Receipt: &f.Input.Receipt}
	collector := nvtelemetry.NewCollector(f.Workload.RunID, sampler, clock, config, nvtelemetry.NewWriter(&stream, config))
	if err := collector.InitStream(); err != nil {
		t.Fatal(err)
	}
	if err := collector.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := collector.Stop(); err != nil {
		t.Fatal(err)
	}
	streamBytes := stream.Bytes()
	expected := *host

	expectedOutput := []float64{1, 2, 3, 4}
	expectedOutputDigest, err := cudaregress.OutputDigest(expectedOutput)
	if err != nil {
		t.Fatal(err)
	}
	wrongOutputDigest, err := cudaregress.OutputDigest([]float64{1, 2, 3, 999})
	if err != nil {
		t.Fatal(err)
	}
	oracleBytes, err := json.Marshal(expectedOutput)
	if err != nil {
		t.Fatal(err)
	}

	runnerScript := fmt.Sprintf(`#!/bin/sh
mode="$1"
case "$mode" in
  host-delay-on)
    printf '{"version":1,"output_values":[1,2,3,4],"output_sha256":"%s","timing":{"kind":"wall_elapsed","status":"measured","unit":"ns","count":70000}}\n'
    ;;
  host-delay-off)
    printf '{"version":1,"output_values":[1,2,3,4],"output_sha256":"%s","timing":{"kind":"wall_elapsed","status":"measured","unit":"ns","count":15000}}\n'
    ;;
  host-delay-noisy-null)
    count=70000
    if [ ! -f null-step-1 ]; then
      : > null-step-1
    elif [ ! -f null-step-2 ]; then
      : > null-step-2
      count=71000
    fi
    printf '{"version":1,"output_values":[1,2,3,4],"output_sha256":"%s","timing":{"kind":"wall_elapsed","status":"measured","unit":"ns","count":%%d}}\n' "$count"
    ;;
  wrong-output)
    printf '{"version":1,"output_values":[1,2,3,999],"output_sha256":"%s","timing":{"kind":"wall_elapsed","status":"measured","unit":"ns","count":15000}}\n'
    ;;
  *)
    echo "unknown mode: $mode" >&2
    exit 2
    ;;
esac
`, expectedOutputDigest, expectedOutputDigest, expectedOutputDigest, wrongOutputDigest)

	ptxBytes := []byte(".version 8.0\n.target sm_90\n.address_size 64\n.visible .entry gemm_kernel() {\n\tret;\n}\n")
	ptxArtifact, err := cudakernel.NewArtifact(cudakernel.KindPTX, ptxBytes, "gemm_kernel", cudakernel.Schema{}, "sm_90")
	if err != nil {
		t.Fatal(err)
	}

	observedEnvironment := cudaregress.Environment{
		DeviceIndex:       0,
		DeviceUUID:        assignment.Device.UUID,
		Model:             assignment.Device.Model,
		ComputeCapability: "sm_90",
		Driver:            assignment.Device.DriverVersion,
		CUDA:              "12.8",
		Clock:             "monotonic",
		Domain:            "reconstruct-epoch",
	}

	files := map[string][]byte{
		"telemetry.jsonl": streamBytes,
		"trace.bin":       traceBytes,
		"raw.bin":         rawBytes,
		"transform.bin":   transformBytes,
		"source.cu":       []byte("__global__ void gemm_kernel() {}\n"),
		"runner.bin":      []byte(runnerScript),
		"kernel.ptx":      ptxBytes,
		"module.ptx":      []byte(".version 8.0\n.target sm_90\n.address_size 64\n// module closure\n.visible .entry gemm_kernel() { ret; }\n"),
		"library.so":      []byte("synthetic library closure"),
		"tool.bin":        []byte("synthetic producer tool"),
		"input.bin":       []byte("synthetic input tensor bytes"),
		"oracle.json":     oracleBytes,
		"tree.txt":        []byte("synthetic candidate tree"),
		"capture.json":    []byte("synthetic capture configuration"),
	}
	roles := map[string]string{
		"telemetry.jsonl": "telemetry_stream",
		"trace.bin":       "trace",
		"raw.bin":         "raw",
		"transform.bin":   "transform",
		"source.cu":       "source",
		"runner.bin":      "binary",
		"kernel.ptx":      "binary",
		"module.ptx":      "module",
		"library.so":      "library",
		"tool.bin":        "tool",
		"input.bin":       "input",
		"oracle.json":     "oracle",
		"tree.txt":        "candidate_tree",
		"capture.json":    "capture_config",
	}
	refs := make([]cudaregress.ArtifactRef, 0, len(files))
	for name, data := range files {
		refs = append(refs, cudaregress.ArtifactRef{
			Name:      name,
			Path:      name,
			Digest:    reconstructTestHash(data),
			SizeBytes: int64(len(data)),
			Role:      roles[name],
		})
	}
	ref := func(name string) cudaregress.ArtifactRef {
		for _, r := range refs {
			if r.Name == name {
				return r
			}
		}
		t.Fatalf("missing ref %s", name)
		return cudaregress.ArtifactRef{}
	}

	planOpts := cudaregress.PlanOptions{
		WarmupRuns: 1,
		Rounds:     3,
		Order:      "ABBA",
		Sources:    []cudaregress.ArtifactRef{ref("source.cu")},
		Binaries:   []cudaregress.ArtifactRef{ref("runner.bin"), ref("kernel.ptx")},
		Libraries: []cudaregress.LibraryRef{{
			Name:    "synthetic library",
			Path:    "library.so",
			Version: "1",
			Digest:  reconstructTestHash(files["library.so"]),
		}},
		Inputs: []cudaregress.InputRef{{
			Name:      "synthetic input",
			Path:      "input.bin",
			Digest:    reconstructTestHash(files["input.bin"]),
			Seed:      42,
			ShapeDims: []int64{4},
		}},
		Oracle: cudaregress.CorrectnessOracle{
			Method:          "digest-exact",
			ExpectedDigest:  expectedOutputDigest,
			ReferencePath:   "oracle.json",
			ReferenceDigest: reconstructTestHash(files["oracle.json"]),
			Tolerance:       1e-6,
		},
		ExecutionScope: cudaregress.TimingScope{Channel: cudatrial.TimingWallElapsed},
		NullControls: cudaregress.NullControlConfig{
			Enabled:         true,
			BasketRuns:      2,
			MaxNoiseFloorNs: 10 * time.Microsecond,
		},
		StoppingRules: cudaregress.StoppingRules{
			MaxTrials:              6,
			RelativeThreshold:      0.10,
			MinSamples:             3,
			MinEffectSize:          0.05,
			OracleTolerance:        1e-6,
			MaxConsecutiveFailures: 3,
			HaltOnDrift:            true,
		},
	}

	inventory := reconstructInventory{
		Version: 1,
		Schema:  reconstructInventorySchema,
		Producer: reconstructProducer{
			Name:                "synthetic-producer",
			Version:             "1",
			Schema:              "synthetic.v1",
			Tool:                "synthetic-test",
			ToolVersion:         "1",
			ToolSHA256:          reconstructTestHash(files["tool.bin"]),
			ModuleSHA256s:       []string{reconstructTestHash(files["module.ptx"])},
			CandidateTreeSHA256: reconstructTestHash(files["tree.txt"]),
			CaptureConfigSHA256: reconstructTestHash(files["capture.json"]),
		},
		Artifacts:        refs,
		TelemetryStream:  "telemetry.jsonl",
		RawArtifactPaths: map[string]string{traceHash: "trace.bin", rawHash: "raw.bin", transformHash: "transform.bin"},
		Telemetry: reconstructTelemetryOptions{
			MaxBytes:        1 << 20,
			Workload:        f.Workload,
			OpID:            f.OpID,
			Assignment:      assignment,
			ClockEpoch:      "reconstruct-epoch",
			Observer:        observer,
			Receipt:         &f.Input.Receipt,
			ExpectedSources: map[string]nvtelemetry.Observation{rawHash: expected},
			WallInterval:    f.Input.WallInterval,
			Thresholds:      f.Input.Thresholds,
		},
		Observed: observedEnvironment,
		Plan:     planOpts,
	}

	rf := reconstructFixture{
		dir:       dir,
		manifest:  filepath.Join(dir, "inventory.json"),
		inventory: inventory,
		files:     files,
	}
	rf.write(t)

	compiler := cudacache.CompilerIdentity{
		NVRTCVersion: "12.8",
		DriverSONAME: "libcuda.so.1",
		TargetArch:   "sm_90",
		OutputKind:   nvrtcx.OutputPTX,
	}
	historyWorkload := cudacache.WorkloadIdentity{
		ProgramID:     f.Workload.WorkloadName,
		NumericalMode: "fp32",
		ShapeDims:     []int{4},
	}

	return composedHostDelayFixture{
		reconstructFixture:   rf,
		expectedOutput:       expectedOutput,
		expectedOutputDigest: expectedOutputDigest,
		wrongOutputDigest:    wrongOutputDigest,
		ptxArtifact:          ptxArtifact,
		compilerInputKey:     "gemm_kernel_source_v1",
		compiler:             compiler,
		historyWorkload:      historyWorkload,
	}
}

func (f composedHostDelayFixture) writeExecutionSpec(t *testing.T, dir string, plan cudaregress.ExperimentPlan, baselineArg, candidateArg string) (cudaregress.ExecutionSpec, string) {
	t.Helper()
	var runnerArtifact cudaregress.ArtifactRef
	for _, b := range plan.BinaryArtifacts {
		if b.Name == "runner.bin" {
			runnerArtifact = b
			break
		}
	}
	if runnerArtifact.Path == "" {
		t.Fatal("reconstructed plan missing runner.bin binary artifact")
	}
	if err := os.Chmod(runnerArtifact.Path, 0700); err != nil {
		t.Fatal(err)
	}
	inputs := make([]cudaregress.CommandArtifact, len(plan.Inputs))
	for i, in := range plan.Inputs {
		inputs[i] = cudaregress.CommandArtifact{Name: in.Name, Path: in.Path, SHA256: in.Digest}
	}
	libraries := make([]cudaregress.CommandArtifact, len(plan.LibraryClosure))
	for i, lib := range plan.LibraryClosure {
		libraries[i] = cudaregress.CommandArtifact{Name: lib.Name, Path: lib.Path, SHA256: lib.Digest}
	}
	workDir := filepath.Join(dir, "workdir")
	if err := os.MkdirAll(workDir, 0700); err != nil {
		t.Fatal(err)
	}
	spec := cudaregress.ExecutionSpec{
		Version:          1,
		PlanID:           plan.PlanID,
		PlanDigest:       plan.PlanDigest,
		Mode:             "host_fixture",
		WorkingDirectory: workDir,
		Environment:      map[string]string{},
		TimeoutNanos:     int64(10 * time.Second),
		MaxOutputBytes:   4096,
		Baseline: cudaregress.CommandArm{
			Path:   runnerArtifact.Path,
			SHA256: runnerArtifact.Digest,
			Args:   []string{baselineArg},
		},
		Candidate: cudaregress.CommandArm{
			Path:   runnerArtifact.Path,
			SHA256: runnerArtifact.Digest,
			Args:   []string{candidateArg},
		},
		Inputs:    inputs,
		Libraries: libraries,
	}
	var err error
	spec.SpecDigest, err = cudaregress.ComputeExecutionSpecDigest(spec)
	if err != nil {
		t.Fatal(err)
	}
	specPath := filepath.Join(dir, "spec.json")
	writeJSONFile(t, specPath, spec)
	return spec, specPath
}

func (f composedHostDelayFixture) writeHistoryFiles(t *testing.T, dir string, plan cudaregress.ExperimentPlan) (string, string, cudaprofilehistory.Record, historyBindingFile) {
	t.Helper()
	compilerKey := cudacache.CompilerCacheKey(f.compilerInputKey, f.compiler)
	workloadKey := f.historyWorkload.Key()
	identity := cudaevidence.Identity{
		ArtifactDigest: f.ptxArtifact.Digest,
		Entry:          "gemm_kernel",
		Target:         "sm_90",
		CompilerKey:    compilerKey,
		WorkloadKey:    workloadKey,
	}
	seedReceipt := []byte(`{"seed":"template-receipt"}`)
	template := cudaprofilehistory.Record{
		Version:          cudaprofilehistory.Version,
		SourceDigest:     plan.SourceArtifacts[0].Digest,
		InputDigest:      plan.Inputs[0].Digest,
		CompilerInputKey: f.compilerInputKey,
		Compiler:         f.compiler,
		CompilerKey:      compilerKey,
		Workload:         f.historyWorkload,
		WorkloadKey:      workloadKey,
		NumericalMode:    f.historyWorkload.NumericalMode,
		Entry:            "gemm_kernel",
		Target:           "sm_90",
		PlanDigest:       plan.PlanDigest,
		Receipt:          seedReceipt,
		ReceiptDigest:    reconstructTestHash(seedReceipt),
		Artifact:         f.ptxArtifact,
		Identity:         identity,
		Environment: cudaprofilehistory.Environment{
			Device: f.inventory.Observed.Model,
			Driver: f.inventory.Observed.Driver,
			CUDA:   f.inventory.Observed.CUDA,
			Clock:  f.inventory.Observed.Clock,
			Domain: f.inventory.Observed.Domain,
		},
		Status:      "complete",
		Correctness: cudaprofilehistory.Correctness{Verdict: cudaprofilehistory.Unknown},
	}
	if err := template.Validate(); err != nil {
		t.Fatalf("invalid history template fixture: %v", err)
	}
	templatePath := filepath.Join(dir, "history-template.json")
	writeJSONFile(t, templatePath, template)
	templateDigest, _, err := cudaregress.ComputeFileDigest(templatePath)
	if err != nil {
		t.Fatal(err)
	}
	binding := historyBindingFile{
		Version:               1,
		PlanFrozenDigest:      plan.PlanDigest,
		HistoryTemplateDigest: templateDigest,
		PlanWorkload:          plan.Workload,
		HistoryWorkload:       f.historyWorkload,
		Provenance:            "composed CLI host-delay trial binding",
	}
	bindingPath := filepath.Join(dir, "history-binding.json")
	writeJSONFile(t, bindingPath, binding)
	return templatePath, bindingPath, template, binding
}

func TestRegressComposedReconstructCheckReplayHistory(t *testing.T) {
	if !inDriverFreeProcess(t) {
		return
	}
	fixture := newComposedHostDelayFixture(t)
	beforeInventoryTree := reconstructTestTreeDigest(t, fixture.dir)

	// 1. Reconstruct from the pinned host-delay incident inventory through the CLI entry point.
	reconstructDir := filepath.Join(t.TempDir(), "reconstructed")
	var reconstructOut bytes.Buffer
	if err := runRegressCommand([]string{
		"reconstruct",
		"-inventory", fixture.manifest,
		"-inventory-sha256", fixture.digest,
		"-o", reconstructDir,
	}, &reconstructOut); err != nil {
		t.Fatalf("regress reconstruct failed: %v", err)
	}
	if got := reconstructTestTreeDigest(t, fixture.dir); got != beforeInventoryTree {
		t.Fatal("regress reconstruct mutated input inventory")
	}

	var reconReceipt reconstructReceipt
	if err := readStrictFile(filepath.Join(reconstructDir, "reconstruction.json"), &reconReceipt); err != nil {
		t.Fatalf("read reconstruction receipt: %v", err)
	}
	if !reconReceipt.Incident.Admitted || !reconReceipt.Diagnosis.Diagnosis.Admitted || reconReceipt.Diagnosis.Diagnosis.Mechanism != cudadiag.MechanismHostSubmissionDelay || reconReceipt.PlanReceipt == nil || reconReceipt.PlanError != "" {
		t.Fatalf("expected admitted host_submission_delay reconstruction with plan receipt, got %+v", reconReceipt)
	}

	planPath := filepath.Join(reconstructDir, "plan.json")
	plan, err := readRegressPlanFile(planPath)
	if err != nil {
		t.Fatalf("read reconstructed plan bundle: %v", err)
	}
	if plan.IncidentID != reconReceipt.Incident.ID || plan.PlanDigest != reconReceipt.PlanReceipt.PlanDigest {
		t.Fatalf("reconstructed plan identity mismatch: plan=%+v receipt=%+v", plan, reconReceipt.PlanReceipt)
	}

	// 2. Check the reconstructed plan and observed environment through the CLI entry point.
	envFile := observedEnvironmentFile{
		Version:     1,
		Environment: fixture.inventory.Observed,
		Provenance:  "composed CLI host-delay fixture observation",
	}
	envPath := filepath.Join(t.TempDir(), "observed-environment.json")
	writeJSONFile(t, envPath, envFile)

	var checkOut bytes.Buffer
	if err := runRegressCommand([]string{"check", "-plan", planPath, "-environment", envPath}, &checkOut); err != nil {
		t.Fatalf("regress check failed on reconstructed plan: %v", err)
	}
	wantCheck := fmt.Sprintf("Plan %s: reconstructable (validated artifacts; caller-supplied environment, provenance %q)\n", plan.PlanID, envFile.Provenance)
	if checkOut.String() != wantCheck {
		t.Fatalf("regress check output = %q, want %q", checkOut.String(), wantCheck)
	}

	// 3a. Positive trial 1: incident regression reproduction (clean baseline 15us vs host-delay candidate 70us) with history link.
	workRoot := t.TempDir()
	templatePath, bindingPath, _, _ := fixture.writeHistoryFiles(t, workRoot, plan)
	historyOutPath := filepath.Join(workRoot, "history.ndjson")

	reproDir := filepath.Join(workRoot, "repro-trial")
	reproSpec, reproSpecPath := fixture.writeExecutionSpec(t, reproDir, plan, "host-delay-off", "host-delay-on")
	reproReceiptsDir := filepath.Join(reproDir, "receipts")
	var reproOut bytes.Buffer
	if err := runRegressCommand([]string{
		"replay",
		"-plan", planPath,
		"-spec", reproSpecPath,
		"-environment", envPath,
		"-o", reproReceiptsDir,
		"-history-template", templatePath,
		"-history-binding", bindingPath,
		"-history-out", historyOutPath,
	}, &reproOut); err != nil {
		t.Fatalf("regress replay (host-delay reproduction) failed: %v", err)
	}
	if !strings.Contains(reproOut.String(), "Replayed plan "+plan.PlanID) {
		t.Fatalf("unexpected replay stdout: %q", reproOut.String())
	}

	var reproCmdReceipt replayCommandReceipt
	if err := readStrictFile(filepath.Join(reproReceiptsDir, "command.json"), &reproCmdReceipt); err != nil {
		t.Fatal(err)
	}
	if reproCmdReceipt.PlanDigest != plan.PlanDigest || reproCmdReceipt.SpecDigest != reproSpec.SpecDigest || reproCmdReceipt.Execution == nil || reproCmdReceipt.Comparison == nil || reproCmdReceipt.ReplayError != "" {
		t.Fatalf("unexpected reproduction command receipt: %+v", reproCmdReceipt)
	}
	if !reproCmdReceipt.Comparison.RegressionDetected || reproCmdReceipt.Comparison.Verdict != cudaprofilehistory.Fail || !reproCmdReceipt.Comparison.ExceedsNoiseFloor || reproCmdReceipt.Comparison.BaselineDuration != 15*time.Microsecond || reproCmdReceipt.Comparison.CandidateDuration != 70*time.Microsecond || reproCmdReceipt.Comparison.Delta != 55*time.Microsecond {
		t.Fatalf("expected detected 55us host-delay regression, got %+v", reproCmdReceipt.Comparison)
	}
	if reproCmdReceipt.HistoryRecordKey == "" || reproCmdReceipt.HistoryWithheldReason != "" || reproCmdReceipt.Comparison.HistoryRecordKey != reproCmdReceipt.HistoryRecordKey {
		t.Fatalf("expected linked profile history key on reproduction trial, got %+v", reproCmdReceipt)
	}

	// Verify ABBA interleaved arm ordering and null-control basket execution.
	if reproCmdReceipt.Execution.WarmupRunsCompleted != 1 || len(reproCmdReceipt.Execution.NullControlReceipts) != 2 || reproCmdReceipt.Execution.RoundsCompleted != 3 || len(reproCmdReceipt.Execution.ArmReceipts) != 6 {
		t.Fatalf("unexpected execution counts: %+v", reproCmdReceipt.Execution)
	}
	wantOrder := []string{"baseline", "candidate", "candidate", "baseline", "baseline", "candidate"}
	for i, arm := range reproCmdReceipt.Execution.ArmReceipts {
		if arm.Name != wantOrder[i] {
			t.Fatalf("arm[%d] = %q, want ABBA order %q", i, arm.Name, wantOrder[i])
		}
	}

	// 3b. Positive trial 2: removable host intervention removes the 55us runnable delay (delayed baseline 70us vs intervention candidate 15us) and appends to history.
	interveneDir := filepath.Join(workRoot, "intervene-trial")
	_, interveneSpecPath := fixture.writeExecutionSpec(t, interveneDir, plan, "host-delay-on", "host-delay-off")
	interveneReceiptsDir := filepath.Join(interveneDir, "receipts")
	var interveneOut bytes.Buffer
	if err := runRegressCommand([]string{
		"replay",
		"-plan", planPath,
		"-spec", interveneSpecPath,
		"-environment", envPath,
		"-o", interveneReceiptsDir,
		"-history-template", templatePath,
		"-history-binding", bindingPath,
		"-history-out", historyOutPath,
	}, &interveneOut); err != nil {
		t.Fatalf("regress replay (removable host intervention) failed: %v", err)
	}

	var interveneCmdReceipt replayCommandReceipt
	if err := readStrictFile(filepath.Join(interveneReceiptsDir, "command.json"), &interveneCmdReceipt); err != nil {
		t.Fatal(err)
	}
	if interveneCmdReceipt.Comparison == nil || interveneCmdReceipt.Comparison.RegressionDetected || interveneCmdReceipt.Comparison.Verdict != cudaprofilehistory.Pass || !interveneCmdReceipt.Comparison.ExceedsNoiseFloor || interveneCmdReceipt.Comparison.BaselineDuration != 70*time.Microsecond || interveneCmdReceipt.Comparison.CandidateDuration != 15*time.Microsecond || interveneCmdReceipt.Comparison.Delta != -55*time.Microsecond {
		t.Fatalf("expected removable host intervention to eliminate 55us delay and pass, got %+v", interveneCmdReceipt.Comparison)
	}
	if interveneCmdReceipt.HistoryRecordKey == "" || interveneCmdReceipt.HistoryWithheldReason != "" || interveneCmdReceipt.HistoryRecordKey == reproCmdReceipt.HistoryRecordKey {
		t.Fatalf("expected distinct linked profile history key on intervention trial, got %+v", interveneCmdReceipt)
	}

	// Verify the persisted profile history snapshot via the CLI `history` consumer.
	var historyCLIOut bytes.Buffer
	if err := runHistoryCommand([]string{historyOutPath}, &historyCLIOut); err != nil {
		t.Fatalf("cudaprof history read failed: %v", err)
	}
	var historyRecords []cudaprofilehistory.Record
	if err := json.Unmarshal(historyCLIOut.Bytes(), &historyRecords); err != nil {
		t.Fatalf("unmarshal history CLI output: %v", err)
	}
	if len(historyRecords) != 2 {
		t.Fatalf("history record count = %d, want 2", len(historyRecords))
	}
	for i, rec := range historyRecords {
		if err := rec.Validate(); err != nil {
			t.Fatalf("history record %d failed validation: %v", i, err)
		}
		if rec.PlanDigest != plan.PlanDigest || rec.Artifact == nil || rec.Artifact.Digest != fixture.ptxArtifact.Digest || rec.Correctness.Verdict != cudaprofilehistory.Pass || rec.Correctness.ExpectedDigest != fixture.expectedOutputDigest || len(rec.Facts) != 1 || rec.Facts[0].Qualified {
			t.Fatalf("unexpected history record %d: %+v", i, rec)
		}
	}

	// 3c. Baseline-vs-baseline null arm controls.
	t.Run("null_arm_matched_no_regression", func(t *testing.T) {
		nullDir := filepath.Join(t.TempDir(), "null-matched")
		_, nullSpecPath := fixture.writeExecutionSpec(t, nullDir, plan, "host-delay-on", "host-delay-on")
		nullReceiptsDir := filepath.Join(nullDir, "receipts")
		nullHistoryOut := filepath.Join(nullDir, "history.ndjson")
		var nullOut bytes.Buffer
		if err := runRegressCommand([]string{
			"replay",
			"-plan", planPath,
			"-spec", nullSpecPath,
			"-environment", envPath,
			"-o", nullReceiptsDir,
			"-history-template", templatePath,
			"-history-binding", bindingPath,
			"-history-out", nullHistoryOut,
		}, &nullOut); err != nil {
			t.Fatalf("matched null-arm replay failed: %v", err)
		}
		var nullReceipt replayCommandReceipt
		if err := readStrictFile(filepath.Join(nullReceiptsDir, "command.json"), &nullReceipt); err != nil {
			t.Fatal(err)
		}
		if nullReceipt.Comparison == nil || nullReceipt.Comparison.RegressionDetected || nullReceipt.Comparison.Delta != 0 || nullReceipt.Comparison.Ratio != 1.0 || nullReceipt.Comparison.ExceedsNoiseFloor || nullReceipt.Comparison.Verdict != cudaprofilehistory.Pass {
			t.Fatalf("matched null arm misreported delta or regression: %+v", nullReceipt.Comparison)
		}
	})

	t.Run("null_arm_within_noise_floor_withholds", func(t *testing.T) {
		noisyDir := filepath.Join(t.TempDir(), "null-noisy")
		_, noisySpecPath := fixture.writeExecutionSpec(t, noisyDir, plan, "host-delay-noisy-null", "host-delay-on")
		noisyReceiptsDir := filepath.Join(noisyDir, "receipts")
		noisyHistoryOut := filepath.Join(noisyDir, "history.ndjson")
		var noisyOut bytes.Buffer
		err := runRegressCommand([]string{
			"replay",
			"-plan", planPath,
			"-spec", noisySpecPath,
			"-environment", envPath,
			"-o", noisyReceiptsDir,
			"-history-template", templatePath,
			"-history-binding", bindingPath,
			"-history-out", noisyHistoryOut,
		}, &noisyOut)
		if err == nil || noisyOut.Len() != 0 {
			t.Fatalf("expected indeterminate within-noise-floor replay to refuse completion, got err=%v out=%q", err, noisyOut.String())
		}
		var noisyReceipt replayCommandReceipt
		if err := readStrictFile(filepath.Join(noisyReceiptsDir, "command.json"), &noisyReceipt); err != nil {
			t.Fatal(err)
		}
		if noisyReceipt.Execution == nil || noisyReceipt.Execution.NullDrift != time.Microsecond || noisyReceipt.Comparison == nil || noisyReceipt.Comparison.ExceedsNoiseFloor || noisyReceipt.Comparison.RegressionDetected || noisyReceipt.Comparison.Verdict != cudaprofilehistory.Unknown || !strings.Contains(noisyReceipt.Comparison.Reason, "within null control noise floor") || noisyReceipt.HistoryRecordKey != "" || noisyReceipt.HistoryWithheldReason == "" {
			t.Fatalf("within-noise-floor null arm did not withhold verdict and history: %+v", noisyReceipt)
		}
		if _, statErr := os.Stat(noisyHistoryOut); !os.IsNotExist(statErr) {
			t.Fatalf("indeterminate null arm wrote history snapshot: %v", statErr)
		}
	})
}

func TestRegressComposedChainMutants(t *testing.T) {
	if !inDriverFreeProcess(t) {
		return
	}

	newReconstructedChain := func(t *testing.T) (composedHostDelayFixture, string, string, cudaregress.ExperimentPlan, string) {
		t.Helper()
		fixture := newComposedHostDelayFixture(t)
		reconstructDir := filepath.Join(t.TempDir(), "reconstructed")
		var out bytes.Buffer
		if err := runRegressCommand([]string{
			"reconstruct",
			"-inventory", fixture.manifest,
			"-inventory-sha256", fixture.digest,
			"-o", reconstructDir,
		}, &out); err != nil {
			t.Fatalf("reconstruct setup failed: %v", err)
		}
		planPath := filepath.Join(reconstructDir, "plan.json")
		plan, err := readRegressPlanFile(planPath)
		if err != nil {
			t.Fatalf("read plan setup: %v", err)
		}
		envPath := filepath.Join(t.TempDir(), "observed-environment.json")
		writeJSONFile(t, envPath, observedEnvironmentFile{
			Version:     1,
			Environment: fixture.inventory.Observed,
			Provenance:  "composed mutant control observation",
		})
		return fixture, reconstructDir, planPath, plan, envPath
	}

	t.Run("tamper_reconstructed_plan_between_reconstruct_and_check_replay", func(t *testing.T) {
		fixture, _, planPath, plan, envPath := newReconstructedChain(t)
		_, specPath := fixture.writeExecutionSpec(t, t.TempDir(), plan, "host-delay-on", "host-delay-off")

		rawBytes, err := os.ReadFile(planPath)
		if err != nil {
			t.Fatal(err)
		}
		tampered := strings.Replace(string(rawBytes), `"rounds": 3`, `"rounds": 4`, 1)
		if tampered == string(rawBytes) {
			t.Fatal("failed to locate rounds field in plan bundle")
		}
		if err := os.WriteFile(planPath, []byte(tampered), 0600); err != nil {
			t.Fatal(err)
		}

		var out bytes.Buffer
		if err := runRegressCommand([]string{"check", "-plan", planPath, "-environment", envPath}, &out); !errors.Is(err, cudaregress.ErrArtifactCorrupted) || out.Len() != 0 {
			t.Fatalf("tampered plan accepted by check: err=%v out=%q", err, out.String())
		}
		if err := runRegressCommand([]string{"replay", "-plan", planPath, "-spec", specPath, "-environment", envPath, "-o", filepath.Join(t.TempDir(), "receipts")}, &out); !errors.Is(err, cudaregress.ErrArtifactCorrupted) || out.Len() != 0 {
			t.Fatalf("tampered plan accepted by replay: err=%v out=%q", err, out.String())
		}
	})

	t.Run("tamper_snapshot_artifact_after_reconstruct", func(t *testing.T) {
		fixture, reconstructDir, planPath, plan, envPath := newReconstructedChain(t)
		_, specPath := fixture.writeExecutionSpec(t, t.TempDir(), plan, "host-delay-on", "host-delay-off")

		if err := os.WriteFile(filepath.Join(reconstructDir, "snapshot", "input.bin"), []byte("corrupted input payload"), 0600); err != nil {
			t.Fatal(err)
		}

		var out bytes.Buffer
		if err := runRegressCommand([]string{"check", "-plan", planPath, "-environment", envPath}, &out); !errors.Is(err, cudaregress.ErrArtifactCorrupted) || out.Len() != 0 {
			t.Fatalf("corrupted snapshot artifact accepted by check: err=%v out=%q", err, out.String())
		}
		if err := runRegressCommand([]string{"replay", "-plan", planPath, "-spec", specPath, "-environment", envPath, "-o", filepath.Join(t.TempDir(), "receipts")}, &out); !errors.Is(err, cudaregress.ErrArtifactCorrupted) || out.Len() != 0 {
			t.Fatalf("corrupted snapshot artifact accepted by replay: err=%v out=%q", err, out.String())
		}
	})

	t.Run("environment_drift_between_reconstruct_and_check_replay", func(t *testing.T) {
		fixture, _, planPath, plan, envPath := newReconstructedChain(t)
		_, specPath := fixture.writeExecutionSpec(t, t.TempDir(), plan, "host-delay-on", "host-delay-off")

		drifted := fixture.inventory.Observed
		drifted.Driver = "drifted-driver-version"
		writeJSONFile(t, envPath, observedEnvironmentFile{
			Version:     1,
			Environment: drifted,
			Provenance:  "drifted environment observation",
		})

		var out bytes.Buffer
		if err := runRegressCommand([]string{"check", "-plan", planPath, "-environment", envPath}, &out); !errors.Is(err, cudaregress.ErrEnvironmentDrift) || out.Len() != 0 {
			t.Fatalf("drifted environment accepted by check: err=%v out=%q", err, out.String())
		}
		if err := runRegressCommand([]string{"replay", "-plan", planPath, "-spec", specPath, "-environment", envPath, "-o", filepath.Join(t.TempDir(), "receipts")}, &out); !errors.Is(err, cudaregress.ErrEnvironmentDrift) || out.Len() != 0 {
			t.Fatalf("drifted environment accepted by replay: err=%v out=%q", err, out.String())
		}
	})

	t.Run("oracle_mismatch_halts_replay_and_withholds_history", func(t *testing.T) {
		fixture, _, planPath, plan, envPath := newReconstructedChain(t)
		workDir := t.TempDir()
		templatePath, bindingPath, _, _ := fixture.writeHistoryFiles(t, workDir, plan)
		historyOut := filepath.Join(workDir, "history.ndjson")
		_, specPath := fixture.writeExecutionSpec(t, workDir, plan, "host-delay-on", "wrong-output")
		receiptsDir := filepath.Join(workDir, "receipts")

		var out bytes.Buffer
		err := runRegressCommand([]string{
			"replay",
			"-plan", planPath,
			"-spec", specPath,
			"-environment", envPath,
			"-o", receiptsDir,
			"-history-template", templatePath,
			"-history-binding", bindingPath,
			"-history-out", historyOut,
		}, &out)
		if !errors.Is(err, cudaregress.ErrExecutionHalted) || !strings.Contains(err.Error(), cudaregress.ErrCorrectnessOracleFailed.Error()) || out.Len() != 0 {
			t.Fatalf("oracle mismatch did not fail with halted oracle error: err=%v out=%q", err, out.String())
		}
		var cmdReceipt replayCommandReceipt
		if err := readStrictFile(filepath.Join(receiptsDir, "command.json"), &cmdReceipt); err != nil {
			t.Fatal(err)
		}
		if cmdReceipt.Execution == nil || !cmdReceipt.Execution.Partial || cmdReceipt.Execution.OutputsVerified || cmdReceipt.Comparison == nil || cmdReceipt.Comparison.Verdict != cudaprofilehistory.Unknown || cmdReceipt.HistoryRecordKey != "" || cmdReceipt.HistoryWithheldReason == "" {
			t.Fatalf("oracle mismatch did not withhold comparison and history: %+v", cmdReceipt)
		}
		if _, statErr := os.Stat(historyOut); !os.IsNotExist(statErr) {
			t.Fatalf("oracle mismatch wrote history snapshot: %v", statErr)
		}
	})

	t.Run("history_binding_and_template_mutants_withhold_history", func(t *testing.T) {
		for _, tc := range []struct {
			name   string
			mutate func(*testing.T, *cudaprofilehistory.Record, *historyBindingFile)
			want   string
		}{
			{
				name: "stale_plan_digest_in_binding",
				mutate: func(_ *testing.T, _ *cudaprofilehistory.Record, b *historyBindingFile) {
					b.PlanFrozenDigest = strings.Repeat("0", 64)
				},
				want: "history binding does not match",
			},
			{
				name: "stale_template_digest_in_binding",
				mutate: func(_ *testing.T, _ *cudaprofilehistory.Record, b *historyBindingFile) {
					b.HistoryTemplateDigest = strings.Repeat("0", 64)
				},
				want: "history binding does not match",
			},
			{
				name: "workload_mismatch_in_binding",
				mutate: func(_ *testing.T, _ *cudaprofilehistory.Record, b *historyBindingFile) {
					b.PlanWorkload.WorkloadName = "other_workload"
				},
				want: "history binding does not match",
			},
			{
				name: "environment_clock_mismatch_in_template",
				mutate: func(_ *testing.T, rec *cudaprofilehistory.Record, _ *historyBindingFile) {
					rec.Environment.Clock = "different-clock"
				},
				want: "history environment does not match observed environment",
			},
			{
				name: "unclosed_artifact_in_template",
				mutate: func(t *testing.T, rec *cudaprofilehistory.Record, _ *historyBindingFile) {
					otherArt, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(".version 8.0\n.target sm_90\n.address_size 64\n// unclosed\n.visible .entry gemm_kernel() { ret; }\n"), "gemm_kernel", cudakernel.Schema{}, "sm_90")
					if err != nil {
						t.Fatal(err)
					}
					rec.Artifact = otherArt
					rec.Identity.ArtifactDigest = otherArt.Digest
				},
				want: "history artifact digest is not in plan binary closure",
			},
		} {
			t.Run(tc.name, func(t *testing.T) {
				fixture, _, planPath, plan, envPath := newReconstructedChain(t)
				workDir := t.TempDir()
				templatePath, bindingPath, template, binding := fixture.writeHistoryFiles(t, workDir, plan)
				tc.mutate(t, &template, &binding)
				writeJSONFile(t, templatePath, template)
				if tc.name != "stale_template_digest_in_binding" {
					var err error
					binding.HistoryTemplateDigest, _, err = cudaregress.ComputeFileDigest(templatePath)
					if err != nil {
						t.Fatal(err)
					}
				}
				writeJSONFile(t, bindingPath, binding)

				_, specPath := fixture.writeExecutionSpec(t, workDir, plan, "host-delay-on", "host-delay-off")
				receiptsDir := filepath.Join(workDir, "receipts")
				historyOut := filepath.Join(workDir, "history.ndjson")
				var out bytes.Buffer
				if err := runRegressCommand([]string{
					"replay",
					"-plan", planPath,
					"-spec", specPath,
					"-environment", envPath,
					"-o", receiptsDir,
					"-history-template", templatePath,
					"-history-binding", bindingPath,
					"-history-out", historyOut,
				}, &out); err != nil {
					t.Fatalf("replay failed unexpectedly: %v", err)
				}
				var cmdReceipt replayCommandReceipt
				if err := readStrictFile(filepath.Join(receiptsDir, "command.json"), &cmdReceipt); err != nil {
					t.Fatal(err)
				}
				if cmdReceipt.HistoryRecordKey != "" || !strings.Contains(cmdReceipt.HistoryWithheldReason, tc.want) {
					t.Fatalf("expected history withheld containing %q, got key=%q reason=%q", tc.want, cmdReceipt.HistoryRecordKey, cmdReceipt.HistoryWithheldReason)
				}
				if _, statErr := os.Stat(historyOut); !os.IsNotExist(statErr) {
					t.Fatalf("corrupted history link wrote snapshot: %v", statErr)
				}
			})
		}
	})

	t.Run("incomplete_history_flags_refused", func(t *testing.T) {
		fixture, _, planPath, plan, envPath := newReconstructedChain(t)
		workDir := t.TempDir()
		templatePath, _, _, _ := fixture.writeHistoryFiles(t, workDir, plan)
		_, specPath := fixture.writeExecutionSpec(t, workDir, plan, "host-delay-on", "host-delay-off")
		var out bytes.Buffer
		err := runRegressCommand([]string{
			"replay",
			"-plan", planPath,
			"-spec", specPath,
			"-environment", envPath,
			"-o", filepath.Join(workDir, "receipts"),
			"-history-template", templatePath,
			"-history-out", filepath.Join(workDir, "history.ndjson"),
		}, &out)
		if err == nil || !strings.Contains(err.Error(), "history-template, history-binding, and history-out must be supplied together") || out.Len() != 0 {
			t.Fatalf("incomplete history flags not refused: err=%v out=%q", err, out.String())
		}
	})
}
