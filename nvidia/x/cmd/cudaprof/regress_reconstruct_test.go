package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaregress"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

type reconstructTestClock struct{ now int64 }

func (c *reconstructTestClock) NowMonotonic() int64    { return c.now }
func (c *reconstructTestClock) NowUnix() int64         { return c.now }
func (*reconstructTestClock) MonotonicDomain() string  { return "monotonic" }
func (*reconstructTestClock) MonotonicEpochID() string { return "reconstruct-epoch" }
func (*reconstructTestClock) WallDomain() string       { return "unix" }
func (*reconstructTestClock) Provenance() string       { return "synthetic reconstruct test" }

type reconstructTestSampler struct {
	readings []nvtelemetry.Reading
	clock    *reconstructTestClock
	device   nvtelemetry.DeviceMetadata
}

func (s *reconstructTestSampler) Devices() []nvtelemetry.DeviceMetadata {
	return []nvtelemetry.DeviceMetadata{s.device}
}

func (s *reconstructTestSampler) Sample(context.Context) ([]nvtelemetry.Reading, error) {
	s.clock.now = 200000
	return s.readings, nil
}

func TestRegressReconstructAdmissionAndPlan(t *testing.T) {
	fixture := newReconstructFixture(t)
	before := reconstructTestTreeDigest(t, fixture.dir)
	out := filepath.Join(t.TempDir(), "reconstruction")
	if err := runRegressReconstructCommand(reconstructArgs(fixture, out), io.Discard); err != nil {
		t.Fatal(err)
	}
	if got := reconstructTestTreeDigest(t, fixture.dir); got != before {
		t.Fatal("reconstruction changed an input byte")
	}
	var receipt reconstructReceipt
	receiptBytes, err := os.ReadFile(filepath.Join(out, "reconstruction.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := decodeStrictJSON(receiptBytes, &receipt); err != nil {
		t.Fatal(err)
	}
	if !receipt.Incident.Admitted || !receipt.Diagnosis.Diagnosis.Admitted || receipt.PlanReceipt == nil {
		t.Fatalf("valid synthetic packet not reconstructed: %+v", receipt)
	}
	plan, err := readRegressPlanFile(filepath.Join(out, "plan.json"))
	if err != nil {
		t.Fatalf("standard plan bundle is not consumable: %v", err)
	}
	if plan.IncidentID != receipt.Incident.ID || plan.IncidentDigest == "" {
		t.Fatalf("plan is detached from reconstructed incident: %+v", plan)
	}
}

func TestRegressReconstructRefusesBeforeTelemetryDecode(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*testing.T, reconstructFixture)
		want   string
	}{
		{"wrong root pin", func(t *testing.T, f reconstructFixture) {}, "inventory digest mismatch"},
		{"corrupt member", func(t *testing.T, f reconstructFixture) {
			path := filepath.Join(f.dir, "telemetry.jsonl")
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			data[len(data)-2] ^= 1
			if err := os.WriteFile(path, data, 0600); err != nil {
				t.Fatal(err)
			}
		}, "size or digest mismatch"},
		{"missing member", func(t *testing.T, f reconstructFixture) {
			if err := os.Remove(filepath.Join(f.dir, "module.ptx")); err != nil {
				t.Fatal(err)
			}
		}, "missing or undeclared files"},
		{"undeclared member", func(t *testing.T, f reconstructFixture) {
			if err := os.WriteFile(filepath.Join(f.dir, "extra"), []byte("extra"), 0600); err != nil {
				t.Fatal(err)
			}
		}, "missing or undeclared files"},
		{"directory enumeration bound", func(t *testing.T, f reconstructFixture) {
			for i := 0; i < reconstructMaxFiles; i++ {
				path := filepath.Join(f.dir, fmt.Sprintf("extra-%03d", i))
				if err := os.WriteFile(path, nil, 0600); err != nil {
					t.Fatal(err)
				}
			}
		}, "inventory directory file count limit"},
		{"symlink", func(t *testing.T, f reconstructFixture) {
			if err := os.Remove(filepath.Join(f.dir, "tool.bin")); err != nil {
				t.Fatal(err)
			}
			if err := os.Symlink("source.go", filepath.Join(f.dir, "tool.bin")); err != nil {
				t.Skipf("symlink unavailable: %v", err)
			}
		}, "not a regular file"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fixture := newReconstructFixture(t)
			if test.name == "wrong root pin" {
				fixture.digest = strings.Repeat("0", 64)
			} else {
				test.mutate(t, fixture)
			}
			out := filepath.Join(t.TempDir(), "must-not-exist")
			err := runRegressReconstructCommand(reconstructArgs(fixture, out), io.Discard)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("got %v, want error containing %q", err, test.want)
			}
			if _, statErr := os.Lstat(out); !os.IsNotExist(statErr) {
				t.Fatalf("refusal created output: %v", statErr)
			}
		})
	}
}

func TestRegressReconstructRetainsWithheldDiagnosis(t *testing.T) {
	fixture := newReconstructFixture(t)
	fixture.inventory.Telemetry.WallInterval.Start += 500000
	fixture.write(t)
	out := filepath.Join(t.TempDir(), "withheld")
	if err := runRegressReconstructCommand(reconstructArgs(fixture, out), io.Discard); err != nil {
		t.Fatal(err)
	}
	var receipt reconstructReceipt
	data, err := os.ReadFile(filepath.Join(out, "reconstruction.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := decodeStrictJSON(data, &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.Incident.Admitted || receipt.Diagnosis.Diagnosis.Admitted || receipt.PlanReceipt != nil || receipt.PlanError == "" {
		t.Fatalf("withheld diagnosis was not preserved or plan was emitted: %+v", receipt)
	}
	if _, err := os.Stat(filepath.Join(out, "plan.json")); !os.IsNotExist(err) {
		t.Fatalf("withheld diagnosis wrote plan: %v", err)
	}
}

func TestRegressReconstructWithholdsInvalidEvidence(t *testing.T) {
	tests := []struct {
		name      string
		mutate    func(*testing.T, *reconstructFixture)
		wantError bool
	}{
		{"loss", mutateReconstructLoss, false},
		{"trace receipt mismatch", mutateReconstructTraceReceipt, false},
		{"source transform mutation", mutateReconstructTransform, false},
		{"unsupported telemetry schema", mutateReconstructSchema, true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fixture := newReconstructFixture(t)
			test.mutate(t, &fixture)
			fixture.write(t)
			out := filepath.Join(t.TempDir(), "withheld")
			err := runRegressReconstructCommand(reconstructArgs(fixture, out), io.Discard)
			if test.wantError {
				if err == nil || !strings.Contains(err.Error(), "unsupported telemetry diagnosis schema") {
					t.Fatalf("unsupported schema error = %v", err)
				}
				if _, statErr := os.Lstat(out); !os.IsNotExist(statErr) {
					t.Fatalf("unsupported schema created output: %v", statErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			var receipt reconstructReceipt
			data, err := os.ReadFile(filepath.Join(out, "reconstruction.json"))
			if err != nil {
				t.Fatal(err)
			}
			if err := decodeStrictJSON(data, &receipt); err != nil {
				t.Fatal(err)
			}
			if receipt.Incident.Admitted || receipt.Diagnosis.Diagnosis.Admitted || receipt.PlanReceipt != nil || receipt.PlanError == "" {
				t.Fatalf("invalid evidence was admitted or omitted: %+v", receipt)
			}
			if _, err := os.Stat(filepath.Join(out, "plan.json")); !os.IsNotExist(err) {
				t.Fatalf("withheld diagnosis wrote plan: %v", err)
			}
		})
	}
}

func TestRegressReconstructFreshProcessHelper(t *testing.T) {
	if os.Getenv("LIB_RECONSTRUCT_CHILD") != "1" {
		return
	}
	args := []string{"-inventory", os.Getenv("LIB_RECONSTRUCT_INVENTORY"), "-inventory-sha256", os.Getenv("LIB_RECONSTRUCT_DIGEST"), "-o", os.Getenv("LIB_RECONSTRUCT_OUTPUT")}
	if err := runRegressReconstructCommand(args, io.Discard); err != nil {
		t.Fatal(err)
	}
}

func TestRegressReconstructFreshProcess(t *testing.T) {
	fixture := newReconstructFixture(t)
	out := filepath.Join(t.TempDir(), "fresh-process")
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestRegressReconstructFreshProcessHelper$")
	cmd.Env = []string{
		"LIB_RECONSTRUCT_CHILD=1",
		"LIB_RECONSTRUCT_INVENTORY=" + fixture.manifest,
		"LIB_RECONSTRUCT_DIGEST=" + fixture.digest,
		"LIB_RECONSTRUCT_OUTPUT=" + out,
	}
	var output reconstructLimitedBuffer
	cmd.Stdout, cmd.Stderr = &output, &output
	if err := cmd.Run(); err != nil {
		t.Fatalf("fresh process failed: %v: %s", err, output.String())
	}
	var receipt reconstructReceipt
	data, err := os.ReadFile(filepath.Join(out, "reconstruction.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := decodeStrictJSON(data, &receipt); err != nil || !receipt.Incident.Admitted {
		t.Fatalf("fresh process did not reconstruct: receipt=%+v err=%v", receipt, err)
	}
}

func TestRegressReconstructOutputAliasHelper(t *testing.T) {
	if os.Getenv("LIB_RECONSTRUCT_ALIAS_CHILD") != "1" {
		return
	}
	args := []string{
		"reconstruct",
		"-inventory", os.Getenv("LIB_RECONSTRUCT_ALIAS_INVENTORY"),
		"-inventory-sha256", os.Getenv("LIB_RECONSTRUCT_ALIAS_DIGEST"),
		"-o", os.Getenv("LIB_RECONSTRUCT_ALIAS_OUTPUT"),
	}
	if err := runRegressCommand(args, io.Discard); err != nil {
		t.Fatal(err)
	}
}

func TestRegressReconstructFreshProcessRefusesResolvedOutputAliases(t *testing.T) {
	for _, test := range []struct {
		name  string
		paths func(*testing.T, reconstructFixture) (string, string)
	}{
		{
			name: "output parent symlink into input",
			paths: func(t *testing.T, f reconstructFixture) (string, string) {
				alias := filepath.Join(t.TempDir(), "input-alias")
				if err := os.Symlink(f.dir, alias); err != nil {
					t.Skipf("symlink unavailable: %v", err)
				}
				return f.manifest, filepath.Join(alias, "new-output")
			},
		},
		{
			name: "input parent symlink",
			paths: func(t *testing.T, f reconstructFixture) (string, string) {
				alias := filepath.Join(t.TempDir(), "input-parent-alias")
				if err := os.Symlink(filepath.Dir(f.dir), alias); err != nil {
					t.Skipf("symlink unavailable: %v", err)
				}
				inventory := filepath.Join(alias, filepath.Base(f.dir), filepath.Base(f.manifest))
				return inventory, filepath.Join(f.dir, "new-output")
			},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			fixture := newReconstructFixture(t)
			inventory, outputDir := test.paths(t, fixture)
			before := reconstructTestTreeDigest(t, fixture.dir)
			ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestRegressReconstructOutputAliasHelper$")
			cmd.Env = []string{
				"LIB_RECONSTRUCT_ALIAS_CHILD=1",
				"LIB_RECONSTRUCT_ALIAS_INVENTORY=" + inventory,
				"LIB_RECONSTRUCT_ALIAS_DIGEST=" + fixture.digest,
				"LIB_RECONSTRUCT_ALIAS_OUTPUT=" + outputDir,
			}
			var output reconstructLimitedBuffer
			cmd.Stdout, cmd.Stderr = &output, &output
			err := cmd.Run()
			if ctx.Err() != nil {
				t.Fatalf("fresh process timed out: %v", ctx.Err())
			}
			if err == nil || !strings.Contains(output.String(), "output directory resolves inside the input inventory") {
				t.Fatalf("alias was not refused: child=%v output=%q", err, output.String())
			}
			if _, statErr := os.Lstat(filepath.Join(fixture.dir, "new-output")); !os.IsNotExist(statErr) {
				t.Fatalf("alias created output in input: %v", statErr)
			}
			if after := reconstructTestTreeDigest(t, fixture.dir); after != before {
				t.Fatalf("alias refusal changed input inventory: before=%s after=%s", before, after)
			}
		})
	}
}

type reconstructLimitedBuffer struct{ bytes.Buffer }

func (b *reconstructLimitedBuffer) Write(p []byte) (int, error) {
	if len(p) > 1<<20-b.Len() {
		return 0, errors.New("subprocess output limit exceeded")
	}
	return b.Buffer.Write(p)
}

type reconstructFixture struct {
	dir       string
	manifest  string
	digest    string
	inventory reconstructInventory
	files     map[string][]byte
}

func reconstructArgs(f reconstructFixture, output string) []string {
	return []string{"-inventory", f.manifest, "-inventory-sha256", f.digest, "-o", output}
}

func newReconstructFixture(t *testing.T) reconstructFixture {
	t.Helper()
	dir := t.TempDir()
	f := cudadiag.BuildSyntheticFixture(cudadiag.InjectionNone)
	traceBytes := []byte("synthetic independent trace receipt input")
	rawBytes := []byte("synthetic source observation bytes")
	transformBytes := []byte("synthetic identity transform bytes")
	traceHash, rawHash, transformHash := reconstructTestHash(traceBytes), reconstructTestHash(rawBytes), reconstructTestHash(transformBytes)
	f.Input.Receipt.InputSHA256 = traceHash
	f.Input.Receipt.CaptureBounds = &tracecontract.CaptureBounds{Start: 1000, End: 200000, Clock: "monotonic"}
	assignment := nvtelemetry.EntityAssignment{
		Process: nvtelemetry.EntityProcess{Process: f.Workload.Process, BootID: "synthetic-boot", NodeID: "synthetic-node", PIDNamespace: "synthetic-pidns", HasExecGeneration: true, ExecGeneration: 1},
		Device:  nvtelemetry.DeviceMetadata{UUID: "GPU-synthetic", Model: "synthetic device", DriverVersion: "synthetic driver", Index: 0, MIGScope: nvtelemetry.MIGScopeDisabled},
		Owner:   nvtelemetry.EntityOwner{AllocationID: "synthetic-allocation"}, Source: "synthetic fixture", SourceRevision: "1", State: "assigned", Authoritative: true,
		ClockDomain: "monotonic", ClockEpoch: "reconstruct-epoch", StartNanos: 0, EndNanos: 1_000_000, Sharing: "exclusive",
	}
	observer := nvtelemetry.Observer{ID: "synthetic-observer", Version: "1", Mode: "activity", Pass: "single", Population: "all synthetic records", Filter: "none", Replay: "none"}
	var readings []nvtelemetry.Reading
	add := func(name, family, kind string, span tracecontract.Interval) *nvtelemetry.Observation {
		uncertainty := int64(10)
		o := &nvtelemetry.Observation{Family: family, SourceClass: "synthetic", Producer: "synthetic-producer", ProducerVersion: "1", ProducerSchema: "synthetic.v1", RawSHA256: rawHash, Transform: "identity", TransformSHA256: transformHash, Assignment: &assignment, Workload: &f.Workload, OpID: f.OpID, Interval: &span, ClockEpoch: "reconstruct-epoch", UncertaintyNS: &uncertainty, Observer: observer, Kind: kind, TID: f.Input.Sched.TID}
		zero := float64(0)
		readings = append(readings, nvtelemetry.Reading{Name: name, Unit: "ns", DeviceUUID: "GPU-synthetic", Backend: "synthetic-producer", Evidence: nvtelemetry.EvidenceMeasured, Number: &nvtelemetry.Number{Float: &zero}, Observation: o})
		return o
	}
	host := add("cudaLaunchKernel", "host", "host_call", f.Input.HostCalls[0].Span)
	host.HasReturn = true
	contextID := cudaworkload.ContextStreamID{Context: 1, Stream: 2}
	host.Host = &cudaworkload.HostSubmission{Process: f.Workload.Process, OpID: f.OpID, Workload: f.Workload, NativeCorrelationID: 7, Provenance: cudaworkload.ProvenanceSynthetic, Context: contextID, TID: f.Input.Sched.TID, TimestampNs: uint64(host.Interval.Start), TimestampClock: "monotonic", TimestampEpoch: "reconstruct-epoch"}
	device := add("synthetic_kernel", "activity", "kernel", f.Input.Kernels[0].Span)
	device.Device = &cudaworkload.DeviceActivity{Process: f.Workload.Process, NativeCorrelationID: 7, ExternalCorrelationID: uint64(f.OpID), Context: contextID, StartNs: uint64(device.Interval.Start), EndNs: uint64(device.Interval.End), Kind: "kernel", Name: "synthetic_kernel"}
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
	observedEnvironment := cudaregress.Environment{DeviceIndex: 0, DeviceUUID: "GPU-synthetic", Model: "synthetic device", Driver: "synthetic driver", CUDA: "synthetic CUDA", Clock: "synthetic clock", Domain: "synthetic domain"}
	files := map[string][]byte{
		"telemetry.jsonl": streamBytes, "trace.bin": traceBytes, "raw.bin": rawBytes, "transform.bin": transformBytes,
		"source.go": []byte("synthetic source closure"), "binary.bin": []byte("synthetic binary closure"), "module.ptx": []byte("synthetic module closure"),
		"library.so": []byte("synthetic library closure"), "tool.bin": []byte("synthetic producer tool"), "input.bin": []byte("synthetic input bytes"),
		"oracle.bin": []byte("synthetic reference oracle"), "tree.txt": []byte("synthetic candidate tree"), "capture.json": []byte("synthetic capture configuration"),
	}
	refs := make([]cudaregress.ArtifactRef, 0, len(files))
	roles := map[string]string{"telemetry.jsonl": "telemetry_stream", "trace.bin": "trace", "raw.bin": "raw", "transform.bin": "transform", "source.go": "source", "binary.bin": "binary", "module.ptx": "module", "library.so": "library", "tool.bin": "tool", "input.bin": "input", "oracle.bin": "oracle", "tree.txt": "candidate_tree", "capture.json": "capture_config"}
	for name, data := range files {
		refs = append(refs, cudaregress.ArtifactRef{Name: name, Path: name, Digest: reconstructTestHash(data), SizeBytes: int64(len(data)), Role: roles[name]})
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
	files["trace.bin"] = traceBytes
	plan := cudaregress.PlanOptions{
		Rounds: 1, Order: "ABBA", Sources: []cudaregress.ArtifactRef{ref("source.go")}, Binaries: []cudaregress.ArtifactRef{ref("binary.bin")},
		Libraries:      []cudaregress.LibraryRef{{Name: "synthetic library", Path: "library.so", Version: "1", Digest: reconstructTestHash(files["library.so"])}},
		Inputs:         []cudaregress.InputRef{{Name: "synthetic input", Path: "input.bin", Digest: reconstructTestHash(files["input.bin"])}},
		Oracle:         cudaregress.CorrectnessOracle{Method: "digest-exact", ExpectedDigest: reconstructTestHash([]byte("expected output")), ReferencePath: "oracle.bin", ReferenceDigest: reconstructTestHash(files["oracle.bin"])},
		ExecutionScope: cudaregress.TimingScope{Channel: "wall_elapsed"}, StoppingRules: cudaregress.StoppingRules{MaxTrials: 5, MinSamples: 1},
	}
	inventory := reconstructInventory{
		Version: 1, Schema: reconstructInventorySchema,
		Producer:  reconstructProducer{Name: "synthetic-producer", Version: "1", Schema: "synthetic.v1", Tool: "synthetic-test", ToolVersion: "1", ToolSHA256: reconstructTestHash(files["tool.bin"]), ModuleSHA256s: []string{reconstructTestHash(files["module.ptx"])}, CandidateTreeSHA256: reconstructTestHash(files["tree.txt"]), CaptureConfigSHA256: reconstructTestHash(files["capture.json"])},
		Artifacts: refs, TelemetryStream: "telemetry.jsonl", RawArtifactPaths: map[string]string{traceHash: "trace.bin", rawHash: "raw.bin", transformHash: "transform.bin"},
		Telemetry: reconstructTelemetryOptions{MaxBytes: 1 << 20, Workload: f.Workload, OpID: f.OpID, Assignment: assignment, ClockEpoch: "reconstruct-epoch", Observer: observer, Receipt: &f.Input.Receipt,
			ExpectedSources: map[string]nvtelemetry.Observation{rawHash: expected}, WallInterval: f.Input.WallInterval, Thresholds: f.Input.Thresholds},
		Observed: observedEnvironment, Plan: plan,
	}
	fixture := reconstructFixture{dir: dir, manifest: filepath.Join(dir, "inventory.json"), inventory: inventory, files: files}
	fixture.write(t)
	return fixture
}

func mutateReconstructLoss(t *testing.T, f *reconstructFixture) {
	t.Helper()
	data := f.files[f.inventory.TelemetryStream]
	lines := bytes.Split(data, []byte{'\n'})
	var retained uint64
	for _, line := range lines {
		if len(line) == 0 {
			continue
		}
		var record nvtelemetry.StreamRecord
		if err := json.Unmarshal(line, &record); err != nil {
			t.Fatal(err)
		}
		if record.Manifest != nil && record.Manifest.LostRecordsByFamily["activity"].Total > retained {
			retained = record.Manifest.LostRecordsByFamily["activity"].Total
		}
	}
	loss := tracecontract.Loss{Dropped: 1, Total: retained + 1}
	var found bool
	for i, line := range lines {
		if len(line) == 0 {
			continue
		}
		var record nvtelemetry.StreamRecord
		if err := json.Unmarshal(line, &record); err != nil {
			t.Fatal(err)
		}
		if record.Manifest == nil {
			continue
		}
		found = true
		if record.Manifest.LostRecordsByFamily == nil {
			record.Manifest.LostRecordsByFamily = make(map[string]tracecontract.Loss)
		}
		record.Manifest.LostRecordsByFamily["activity"] = loss
		if record.Manifest.Receipt == nil {
			t.Fatal("independent stream receipt missing")
		}
		copyReceipt := *record.Manifest.Receipt
		if copyReceipt.LostRecords == nil {
			copyReceipt.LostRecords = make(map[string]tracecontract.Loss)
		}
		copyReceipt.LostRecords["activity"] = loss
		record.Manifest.Receipt = &copyReceipt
		lines[i], _ = json.Marshal(record)
	}
	if !found {
		t.Fatal("telemetry manifest not found")
	}
	copyReceipt := *f.inventory.Telemetry.Receipt
	if copyReceipt.LostRecords == nil {
		copyReceipt.LostRecords = make(map[string]tracecontract.Loss)
	}
	copyReceipt.LostRecords["activity"] = loss
	f.inventory.Telemetry.Receipt = &copyReceipt
	f.files[f.inventory.TelemetryStream] = bytes.Join(lines, []byte{'\n'})
	replaceReconstructArtifact(t, f, f.inventory.TelemetryStream, "telemetry_stream")
}

func mutateReconstructTraceReceipt(t *testing.T, f *reconstructFixture) {
	t.Helper()
	data := []byte("different independently retained trace bytes")
	digest := reconstructTestHash(data)
	f.files["trace.bin"] = data
	f.inventory.Telemetry.Receipt.InputSHA256 = digest
	f.inventory.RawArtifactPaths[digest] = "trace.bin"
	delete(f.inventory.RawArtifactPaths, reconstructTestHash([]byte("synthetic independent trace receipt input")))
	replaceReconstructArtifact(t, f, "trace.bin", "trace")
}

func mutateReconstructTransform(t *testing.T, f *reconstructFixture) {
	t.Helper()
	oldDigest := reconstructTestHash([]byte("synthetic identity transform bytes"))
	data := []byte("different retained transform bytes")
	digest := reconstructTestHash(data)
	f.files["transform.bin"] = data
	f.inventory.RawArtifactPaths[digest] = "transform.bin"
	delete(f.inventory.RawArtifactPaths, oldDigest)
	for key, source := range f.inventory.Telemetry.ExpectedSources {
		source.TransformSHA256 = digest
		f.inventory.Telemetry.ExpectedSources[key] = source
	}
	replaceReconstructArtifact(t, f, "transform.bin", "transform")
}

func mutateReconstructSchema(t *testing.T, f *reconstructFixture) {
	t.Helper()
	data := f.files[f.inventory.TelemetryStream]
	lines := bytes.Split(data, []byte{'\n'})
	for i, line := range lines {
		if len(line) == 0 {
			continue
		}
		var record nvtelemetry.StreamRecord
		if err := json.Unmarshal(line, &record); err != nil {
			t.Fatal(err)
		}
		if record.Manifest != nil {
			record.Manifest.SchemaVersion = nvtelemetry.SchemaVersionV5
			lines[i], _ = json.Marshal(record)
		}
	}
	f.files[f.inventory.TelemetryStream] = bytes.Join(lines, []byte{'\n'})
	replaceReconstructArtifact(t, f, f.inventory.TelemetryStream, "telemetry_stream")
}

func replaceReconstructArtifact(t *testing.T, f *reconstructFixture, name, role string) {
	t.Helper()
	data := f.files[name]
	for i := range f.inventory.Artifacts {
		if f.inventory.Artifacts[i].Path == name {
			f.inventory.Artifacts[i].Role = role
			f.inventory.Artifacts[i].Digest = reconstructTestHash(data)
			f.inventory.Artifacts[i].SizeBytes = int64(len(data))
			return
		}
	}
	t.Fatalf("artifact %q missing", name)
}

func (f *reconstructFixture) write(t *testing.T) {
	t.Helper()
	for name, data := range f.files {
		if err := os.WriteFile(filepath.Join(f.dir, name), data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	data, err := json.Marshal(f.inventory)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(f.manifest, data, 0600); err != nil {
		t.Fatal(err)
	}
	f.digest = reconstructTestHash(data)
}

func reconstructTestHash(data []byte) string {
	sum := sha256.Sum256(data)
	return fmt.Sprintf("%x", sum[:])
}

func reconstructTestTreeDigest(t *testing.T, dir string) string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	h := sha256.New()
	for _, entry := range entries {
		data, err := os.ReadFile(filepath.Join(dir, entry.Name()))
		if err != nil {
			t.Fatal(err)
		}
		_, _ = fmt.Fprintf(h, "%s:%s\n", entry.Name(), reconstructTestHash(data))
	}
	return fmt.Sprintf("%x", h.Sum(nil))
}

func TestRegressReconstructOutputContainment(t *testing.T) {
	const inside = "output directory resolves inside the input inventory"
	for _, test := range []struct {
		name string
		// output returns the -o path and, for accepted outputs, where the
		// directory is expected to be created.
		output          func(t *testing.T, base, input string) (output, created string)
		caseInsensitive bool
		refused         bool
	}{
		{
			name:    "input root",
			output:  func(t *testing.T, base, input string) (string, string) { return input, "" },
			refused: true,
		},
		{
			name: "child",
			output: func(t *testing.T, base, input string) (string, string) {
				return filepath.Join(input, "new-output"), ""
			},
			refused: true,
		},
		{
			name: "child named with leading dots",
			output: func(t *testing.T, base, input string) (string, string) {
				return filepath.Join(input, "..new-output"), ""
			},
			refused: true,
		},
		{
			name: "nested child",
			output: func(t *testing.T, base, input string) (string, string) {
				return filepath.Join(input, "sub", "new-output"), ""
			},
			refused: true,
		},
		{
			name: "child through lexical dot dot",
			output: func(t *testing.T, base, input string) (string, string) {
				return input + "/sub/../new-output", ""
			},
			refused: true,
		},
		{
			name: "symlink chain to input parent",
			output: func(t *testing.T, base, input string) (string, string) {
				first, second := filepath.Join(base, "alias-1"), filepath.Join(base, "alias-2")
				if err := os.Symlink(filepath.Dir(input), second); err != nil {
					t.Skipf("symlink unavailable: %v", err)
				}
				if err := os.Symlink(second, first); err != nil {
					t.Fatal(err)
				}
				return filepath.Join(first, filepath.Base(input), "new-output"), ""
			},
			refused: true,
		},
		{
			name: "case variant of input",
			output: func(t *testing.T, base, input string) (string, string) {
				return filepath.Join(filepath.Dir(input), strings.ToUpper(filepath.Base(input)), "new-output"), ""
			},
			caseInsensitive: true,
			refused:         true,
		},
		{
			name: "case variant of input ancestor",
			output: func(t *testing.T, base, input string) (string, string) {
				parent := filepath.Dir(input)
				return filepath.Join(filepath.Dir(parent), strings.ToUpper(filepath.Base(parent)), filepath.Base(input), "new-output"), ""
			},
			caseInsensitive: true,
			refused:         true,
		},
		{
			name: "sibling sharing input name prefix",
			output: func(t *testing.T, base, input string) (string, string) {
				return input + "new-output", input + "new-output"
			},
		},
		{
			name: "sibling named with leading dots",
			output: func(t *testing.T, base, input string) (string, string) {
				out := filepath.Join(filepath.Dir(input), "..new-output")
				return out, out
			},
		},
		{
			// The output path is cleaned before symlinks are resolved, so
			// alias/../x names x next to alias, not next to the alias target.
			name: "symlink then dot dot is cleaned lexically",
			output: func(t *testing.T, base, input string) (string, string) {
				alias := filepath.Join(base, "sub-alias")
				if err := os.Symlink(filepath.Join(input, "sub"), alias); err != nil {
					t.Skipf("symlink unavailable: %v", err)
				}
				return alias + "/../new-output", filepath.Join(base, "new-output")
			},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			base, err := filepath.EvalSymlinks(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			input := filepath.Join(base, "Parent", "Input")
			if err := os.MkdirAll(filepath.Join(input, "sub"), 0700); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(input, "inventory.json"), []byte("{}"), 0600); err != nil {
				t.Fatal(err)
			}
			if test.caseInsensitive && !reconstructTestCaseInsensitive(t, base) {
				t.Skipf("%s is on a case-sensitive volume; a case variant names a different directory", base)
			}
			output, created := test.output(t, base, input)
			before := reconstructTestTreeNames(t, input)
			snapshot := &reconstructSnapshot{
				root:          input,
				files:         map[string][]byte{"member.bin": []byte("member")},
				manifestBytes: []byte("{}"),
			}
			dir, _, cleanup, err := beginReconstructOutput(output, snapshot)
			if after := reconstructTestTreeNames(t, input); test.refused && after != before {
				t.Errorf("input tree changed:\nbefore:\n%s\nafter:\n%s", before, after)
			}
			if test.refused {
				if err == nil || !strings.Contains(err.Error(), inside) {
					t.Fatalf("beginReconstructOutput(%q) = %q, %v; want %q", output, dir, err, inside)
				}
				return
			}
			if err != nil {
				t.Fatalf("beginReconstructOutput(%q): %v", output, err)
			}
			if dir != created {
				t.Errorf("output directory = %q, want %q", dir, created)
			}
			if _, err := os.Stat(filepath.Join(created, "snapshot", "member.bin")); err != nil {
				t.Errorf("snapshot member not written: %v", err)
			}
			cleanup()
			if _, err := os.Lstat(created); !os.IsNotExist(err) {
				t.Errorf("cleanup left %s: %v", created, err)
			}
		})
	}
}

func TestRegressReconstructRefusesCaseVariantOutput(t *testing.T) {
	f := newReconstructFixture(t)
	parent := filepath.Dir(f.dir)
	if !reconstructTestCaseInsensitive(t, parent) {
		t.Skipf("%s is on a case-sensitive volume; a case variant names a different directory", parent)
	}
	upper := filepath.Join(filepath.Dir(parent), strings.ToUpper(filepath.Base(parent)), filepath.Base(f.dir))
	before := reconstructTestTreeDigest(t, f.dir)
	err := runRegressReconstructCommand(reconstructArgs(f, filepath.Join(upper, "new-output")), io.Discard)
	if err == nil || !strings.Contains(err.Error(), "output directory resolves inside the input inventory") {
		t.Fatalf("case-variant output was not refused: %v", err)
	}
	if _, err := os.Lstat(filepath.Join(f.dir, "new-output")); !os.IsNotExist(err) {
		t.Fatalf("case-variant output created inside input: %v", err)
	}
	if after := reconstructTestTreeDigest(t, f.dir); after != before {
		t.Fatalf("input inventory changed: before=%s after=%s", before, after)
	}
}

// reconstructTestCaseInsensitive reports whether dir is on a volume where
// names differing only in case refer to the same file.
func reconstructTestCaseInsensitive(t *testing.T, dir string) bool {
	t.Helper()
	probe, err := os.MkdirTemp(dir, "CaseProbe")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(probe)
	a, err := os.Stat(probe)
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.Stat(filepath.Join(dir, strings.ToLower(filepath.Base(probe))))
	return err == nil && os.SameFile(a, b)
}

func reconstructTestTreeNames(t *testing.T, dir string) string {
	t.Helper()
	var names []string
	err := filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		names = append(names, path)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return strings.Join(names, "\n")
}
