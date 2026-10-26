package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func commandHealthCapacityLimits() (nvtelemetry.HealthLimits, cudadiag.HealthCapacityLimits) {
	return nvtelemetry.HealthLimits{MaxBytes: 32 << 10, MaxDevices: 4, MaxSources: 8, MaxObservations: 32, MaxEdges: 8},
		cudadiag.HealthCapacityLimits{MaxBytes: 4 << 20, MaxFindings: 16, MaxWork: 16, MaxExternalRefs: 16}
}

func commandHealthCapacityFixture(injection cudadiag.InjectionType) (cudadiag.HealthCapacityInput, cudadiag.HealthCapacityLimits, error) {
	return commandHealthCapacityFixtureWithHealth(injection, nil)
}

func commandHealthCapacityFixtureWithHealth(injection cudadiag.InjectionType, mutate func(*nvtelemetry.HealthSnapshot)) (cudadiag.HealthCapacityInput, cudadiag.HealthCapacityLimits, error) {
	healthLimits, limits := commandHealthCapacityLimits()
	readings, telemetry := commandHealthCapacityTelemetry(injection)
	health := nvtelemetry.HealthSnapshot{
		Version: 1, NodeID: telemetry.Assignment.Process.NodeID, TopologyEpoch: "topology-protocol-1",
		ClockDomain: "monotonic", ClockEpoch: telemetry.ClockEpoch, AtNanos: 10000,
		DeviceStates: map[string]string{telemetry.Assignment.Device.UUID: "present"},
		Sources:      []nvtelemetry.HealthSource{{ID: "nvml-protocol", Kind: "nvml", Version: "12.6", State: "available", LastSuccessNanos: 10000, HasLastSuccess: true}},
		Observations: []nvtelemetry.HealthObservation{{
			SourceID: "nvml-protocol", Class: "raw",
			Reading: nvtelemetry.Reading{Name: "xid_event", Unit: "code", Backend: "nvml-protocol", DeviceUUID: telemetry.Assignment.Device.UUID, Evidence: nvtelemetry.EvidenceMeasured, IsInt: true, IntValue: 31, TimestampMonotonic: 10000},
		}},
		Complete: true,
	}
	if mutate != nil {
		mutate(&health)
	}
	healthJSON, err := json.Marshal(health)
	if err != nil {
		return cudadiag.HealthCapacityInput{}, limits, err
	}
	healthHash := sha256.Sum256(healthJSON)
	healthDigest := hex.EncodeToString(healthHash[:])

	assignmentJSON, err := json.Marshal(telemetry.Assignment)
	if err != nil {
		return cudadiag.HealthCapacityInput{}, limits, err
	}
	assignmentHash := sha256.Sum256(assignmentJSON)
	if telemetry.Receipt.Options == nil {
		telemetry.Receipt.Options = make(map[string]string)
	}
	telemetry.Receipt.Options["health_sha256"] = healthDigest
	telemetry.Receipt.Options["assignment_sha256"] = hex.EncodeToString(assignmentHash[:])
	telemetry.Receipt.Options["health_node_id"] = health.NodeID
	telemetry.Receipt.Options["health_topology_epoch"] = health.TopologyEpoch
	telemetry.Receipt.Options["health_clock_domain"] = health.ClockDomain
	telemetry.Receipt.Options["health_clock_epoch"] = health.ClockEpoch
	telemetry.Receipt.Options["assignment_node_id"] = telemetry.Assignment.Process.NodeID
	telemetry.Receipt.Options["assignment_boot_id"] = telemetry.Assignment.Process.BootID
	telemetry.Receipt.Options["assignment_pid"] = fmt.Sprint(telemetry.Assignment.Process.Process.PID)
	telemetry.Receipt.Options["assignment_process_generation"] = fmt.Sprint(telemetry.Assignment.Process.Process.Generation)
	telemetry.Receipt.Options["assignment_source"] = telemetry.Assignment.Source
	telemetry.Receipt.Options["assignment_source_revision"] = telemetry.Assignment.SourceRevision
	telemetry.Receipt.Options["assignment_start_ns"] = fmt.Sprint(telemetry.Assignment.StartNanos)
	telemetry.Receipt.Options["assignment_end_ns"] = fmt.Sprint(telemetry.Assignment.EndNanos)
	telemetry.Receipt.Options["device_uuid"] = telemetry.Assignment.Device.UUID
	telemetry.Receipt.Options["device_mig_scope"] = telemetry.Assignment.Device.MIGScope
	telemetry.Receipt.Options["device_mig_parent_uuid"] = telemetry.Assignment.Device.MIGParentUUID
	telemetry.Receipt.Options["device_mig_instance_id"] = fmt.Sprint(telemetry.Assignment.Device.MIGInstanceID)
	telemetry.Receipt.Options["device_has_mig_instance_id"] = fmt.Sprint(telemetry.Assignment.Device.HasMIGInstanceID)
	telemetry.Receipt.Options["device_mig_compute_instance_id"] = fmt.Sprint(telemetry.Assignment.Device.MIGComputeInstanceID)
	telemetry.Receipt.Options["device_has_mig_compute_instance_id"] = fmt.Sprint(telemetry.Assignment.Device.HasMIGComputeInstanceID)
	telemetry.Receipt.Options["mig_configuration_epoch"] = telemetry.Assignment.MIGConfigurationEpoch
	telemetry.Receipt.Options["allocation_id"] = telemetry.Assignment.Owner.AllocationID
	telemetry.Receipt.Options["allocation_attempt"] = telemetry.Assignment.Owner.Attempt
	telemetry.Receipt.Options["workload_run_id"] = telemetry.Workload.RunID
	telemetry.Receipt.Options["impact_clock_domain"] = telemetry.WallInterval.Clock
	telemetry.Receipt.Options["impact_clock_epoch"] = telemetry.ClockEpoch
	telemetry.Receipt.Options["impact_start_ns"] = fmt.Sprint(telemetry.WallInterval.Start)
	telemetry.Receipt.Options["impact_end_ns"] = fmt.Sprint(telemetry.WallInterval.End)
	telemetry.Receipt.Gates["health_identity_binding"] = tracecontract.Gate{Disposition: "passed", Reason: "protocol fixture binding", Provenance: "synthetic fixture"}
	telemetry.Receipt.Gates["health_freshness"] = tracecontract.Gate{Disposition: "passed", Reason: "protocol fixture freshness", Provenance: "synthetic fixture"}
	telemetryBytes, err := collectCommandHealthCapacityTelemetry(readings, telemetry)
	if err != nil {
		return cudadiag.HealthCapacityInput{}, limits, err
	}
	telemetry.MaxBytes = int64(len(telemetryBytes)) + 1

	finding := tracecontract.Finding{
		ID: "health-xid-1", Rule: "xid_event", Class: tracecontract.FindingObserved,
		Admission: tracecontract.Admission{Admitted: true, Status: "admitted", ReasonCode: tracecontract.ReasonAdmitted, Reason: "protocol fixture rule"},
		Artifacts: []string{healthDigest}, Intervals: []tracecontract.Interval{{Start: 10000, End: 10000, Clock: health.ClockDomain}},
	}
	work := nvtelemetry.AccountingWork{
		ID: "work-1", AllocationID: telemetry.Assignment.Owner.AllocationID, Units: 1000, Unit: "tokens",
		Success: true, Correct: true, Policy: "independent-output-check-v1", Evidence: "protocol-work-receipt",
		ClockDomain: health.ClockDomain, ClockEpoch: health.ClockEpoch, StartNanos: 2000, EndNanos: 30000,
	}
	return cudadiag.HealthCapacityInput{
		HealthJSON: healthJSON, HealthLimits: healthLimits, TelemetryJSONL: telemetryBytes,
		Telemetry: telemetry, HealthFindings: []tracecontract.Finding{finding}, Work: []nvtelemetry.AccountingWork{work},
	}, limits, nil
}

func commandHealthCapacityTelemetry(injection cudadiag.InjectionType) ([]nvtelemetry.Reading, cudadiag.TelemetryOptions) {
	f := cudadiag.BuildSyntheticFixture(injection)
	raw := []byte("synthetic health-capacity source observation")
	transform := []byte("identity transform protocol control")
	rawHash := fmt.Sprintf("%x", sha256.Sum256(raw))
	transformHash := fmt.Sprintf("%x", sha256.Sum256(transform))
	f.Input.Receipt.InputSHA256 = rawHash
	f.Input.Receipt.CaptureBounds = &tracecontract.CaptureBounds{Start: 1000, End: 350000, Clock: "monotonic"}
	assignment := nvtelemetry.EntityAssignment{
		Process: nvtelemetry.EntityProcess{Process: f.Workload.Process, BootID: "protocol-boot", NodeID: "protocol-node", PIDNamespace: "protocol-pidns", HasExecGeneration: true, ExecGeneration: 1},
		Device:  nvtelemetry.DeviceMetadata{UUID: "GPU-protocol", MIGScope: nvtelemetry.MIGScopeDisabled},
		Owner:   nvtelemetry.EntityOwner{AllocationID: "protocol-allocation"}, Source: "protocol-control", SourceRevision: "1",
		State: "assigned", Authoritative: true, ClockDomain: "monotonic", ClockEpoch: "reconstruct-epoch",
		StartNanos: 0, EndNanos: 1000000, Sharing: "exclusive",
	}
	observer := nvtelemetry.Observer{ID: "protocol-observer", Version: "1", Mode: "activity", Pass: "single", Population: "all protocol records", Filter: "none", Replay: "none"}
	opt := cudadiag.TelemetryOptions{
		Workload: f.Workload, OpID: f.OpID, Assignment: assignment, ClockEpoch: "reconstruct-epoch", Observer: observer,
		Receipt: &f.Input.Receipt, RawArtifacts: map[string][]byte{rawHash: raw, transformHash: transform},
		ExpectedSources: map[string]nvtelemetry.Observation{}, WallInterval: f.Input.WallInterval, Thresholds: f.Input.Thresholds,
	}
	var readings []nvtelemetry.Reading
	add := func(name, family, kind string, span tracecontract.Interval) *nvtelemetry.Observation {
		z := float64(0)
		unc := int64(10)
		o := &nvtelemetry.Observation{
			Family: family, SourceClass: "synthetic", Producer: "protocol-fixture", ProducerVersion: "1", ProducerSchema: "synthetic.v1",
			RawSHA256: rawHash, Transform: "identity", TransformSHA256: transformHash, Assignment: &assignment, Workload: &f.Workload,
			OpID: f.OpID, Interval: &span, ClockEpoch: "reconstruct-epoch", UncertaintyNS: &unc, Observer: observer, Kind: kind, TID: f.Input.Thresholds.TargetTID,
		}
		readings = append(readings, nvtelemetry.Reading{Name: name, Unit: "ns", DeviceUUID: "GPU-protocol", Backend: "protocol-fixture", Evidence: nvtelemetry.EvidenceMeasured, Number: &nvtelemetry.Number{Float: &z}, Observation: o})
		return o
	}
	contextID := cudaworkload.ContextStreamID{Context: 1, Stream: 2}
	host := add("cudaLaunchKernel", "host", "host_call", f.Input.HostCalls[0].Span)
	host.HasReturn = true
	host.Host = &cudaworkload.HostSubmission{Process: f.Workload.Process, OpID: f.OpID, Workload: f.Workload, NativeCorrelationID: 7, Provenance: cudaworkload.ProvenanceSynthetic, Context: contextID, TID: f.Input.Thresholds.TargetTID, TimestampNs: uint64(host.Interval.Start), TimestampClock: "monotonic", TimestampEpoch: "reconstruct-epoch"}
	device := add("protocol_kernel", "activity", "kernel", f.Input.Kernels[0].Span)
	device.Device = &cudaworkload.DeviceActivity{Process: f.Workload.Process, NativeCorrelationID: 7, ExternalCorrelationID: uint64(f.OpID), Context: contextID, StartNs: uint64(device.Interval.Start), EndNs: uint64(device.Interval.End), Kind: "kernel", Name: "protocol_kernel"}
	for _, iv := range f.Input.Sched.Intervals {
		add("scheduler", "sched", "sched_"+string(iv.State), iv.Span)
	}
	for _, family := range []string{"nvml", "dcgm", "managed", "topology", "workload", "nccl"} {
		add(family+"_zero", family, "gauge", f.Input.WallInterval)
	}
	opt.ExpectedSources[rawHash] = *host
	return readings, opt
}

func collectCommandHealthCapacityTelemetry(readings []nvtelemetry.Reading, opt cudadiag.TelemetryOptions) ([]byte, error) {
	var b bytes.Buffer
	clock := &reconstructTestClock{now: 1000}
	sampler := &reconstructTestSampler{readings: readings, clock: clock, device: opt.Assignment.Device}
	cfg := nvtelemetry.CollectorConfig{SchemaVersion: nvtelemetry.SchemaVersionV3, Receipt: opt.Receipt}
	collector := nvtelemetry.NewCollector(opt.Workload.RunID, sampler, clock, cfg, nvtelemetry.NewWriter(&b, cfg))
	if err := collector.InitStream(); err != nil {
		return nil, err
	}
	if err := collector.Step(context.Background()); err != nil {
		return nil, err
	}
	if _, err := collector.Stop(); err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}

func commandHealthFiles(t *testing.T) ([]string, string, healthCapacityCommandInput, []byte, []byte) {
	t.Helper()
	input, limits, err := commandHealthCapacityFixture(cudadiag.InjectionHostDelay)
	if err != nil {
		t.Fatal(err)
	}
	config := healthCapacityCommandInput{Telemetry: input.Telemetry, HealthLimits: input.HealthLimits, HealthFindings: input.HealthFindings, Work: input.Work, Limits: limits}
	configJSON, err := json.Marshal(config)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	paths := []string{filepath.Join(dir, "health.json"), filepath.Join(dir, "telemetry.jsonl"), filepath.Join(dir, "input.json")}
	for i, raw := range [][]byte{input.HealthJSON, input.TelemetryJSONL, configJSON} {
		if err := os.WriteFile(paths[i], raw, 0600); err != nil {
			t.Fatal(err)
		}
	}
	return []string{"-health", paths[0], "-telemetry", paths[1], "-input", paths[2]}, paths[2], config, input.HealthJSON, input.TelemetryJSONL
}

func decodeCommandHealthOutput(t *testing.T, raw []byte) healthCapacityCommandOutput {
	t.Helper()
	var output healthCapacityCommandOutput
	if err := json.Unmarshal(raw, &output); err != nil {
		t.Fatal(err)
	}
	return output
}

func TestHealthCapacityCommandPublicDispatch(t *testing.T) {
	args, configPath, _, health, telemetry := commandHealthFiles(t)
	command, ok := lookupCommand("health-capacity")
	if !ok {
		t.Fatal("health-capacity not registered")
	}
	output := captureStdout(t, func() error { return command.invoke(append([]string{"health-capacity"}, args...)) })
	t.Logf("command stdout: %s", output)
	result := decodeCommandHealthOutput(t, []byte(output))
	healthHash := sha256.Sum256(health)
	telemetryHash := sha256.Sum256(telemetry)
	configJSON, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	configHash := sha256.Sum256(configJSON)
	if result.ConfigurationSHA256 != hex.EncodeToString(configHash[:]) || result.Incident.HealthSHA256 != hex.EncodeToString(healthHash[:]) || result.Incident.TelemetrySHA256 != hex.EncodeToString(telemetryHash[:]) {
		t.Fatal("exact raw custody digests missing")
	}
	incident := result.Incident
	if incident.Version != 1 || !incident.Impact.Admitted || incident.ImpactClass != tracecontract.FindingAssociated || incident.WithheldReason != "" || incident.Diagnosis.Diagnosis.Mechanism != cudadiag.MechanismHostSubmissionDelay {
		t.Fatalf("command did not compose meaningful existing inputs: %+v", incident)
	}
	if len(incident.Health.Observations) != 1 || incident.Health.Observations[0].Reading.IntValue != 31 || len(incident.HealthFindings) != 1 || incident.HealthFindings[0].ID != "health-xid-1" || len(incident.Work) != 1 || incident.Work[0].ID != "work-1" || incident.Work[0].Units != 1000 || result.Validation != nil {
		t.Fatal("raw health/finding/useful work was not retained")
	}
	if !strings.Contains(usage, "cudaprof health-capacity") || command.usage != healthCapacityUsage {
		t.Fatal("registered help missing")
	}
}

func TestHealthCapacityCommandImportDoesNotAlterIncident(t *testing.T) {
	args, _, _, _, _ := commandHealthFiles(t)
	var before bytes.Buffer
	if err := runHealthCapacity(context.Background(), args, &before); err != nil {
		t.Fatal(err)
	}
	baseline := decodeCommandHealthOutput(t, before.Bytes())
	raw, err := os.ReadFile("testdata/health-capacity/nvsentinel-v1.26-request.json")
	if err != nil {
		t.Fatal(err)
	}
	hash := sha256.Sum256(raw)
	withImport := append(append([]string{}, args...), "-nvsentinel", "testdata/health-capacity/nvsentinel-v1.26-request.json", "-nvsentinel-sha256", hex.EncodeToString(hash[:]))
	command, ok := lookupCommand("health-capacity")
	if !ok {
		t.Fatal("missing command")
	}
	stdout := captureStdout(t, func() error { return command.invoke(append([]string{"health-capacity"}, withImport...)) })
	t.Logf("command import stdout: %s", stdout)
	imported := decodeCommandHealthOutput(t, []byte(stdout))
	if imported.Validation == nil || imported.Validation.SHA256 != hex.EncodeToString(hash[:]) || !imported.Validation.ExpectedDigestChecked {
		t.Fatal("missing producer parse/raw custody")
	}
	if len(imported.Incident.ExternalRefs) != 2 || imported.Incident.ExternalRefs[0].Kind != "request" || imported.Incident.ExternalRefs[1].Kind != "result" || imported.Incident.ExternalRefs[1].State != "producer_reported_Succeeded" {
		t.Fatal("opaque request/status refs lost")
	}
	for _, ref := range imported.Incident.ExternalRefs {
		if ref.Kind == "authorization" || ref.SHA256 != hex.EncodeToString(hash[:]) {
			t.Fatal("metadata promoted to authority")
		}
	}
	imported.Incident.ExternalRefs = nil
	if !reflect.DeepEqual(baseline.Incident, imported.Incident) {
		t.Fatal("optional validation changed health/workload association")
	}
	// A different request/node/provider object with a recomputed custody hash
	// remains entirely unverified; it cannot acquire an authoritative join.
	changed := bytes.ReplaceAll(raw, []byte("node-a"), []byte("node-other"))
	changed = bytes.ReplaceAll(changed, []byte("constructed-request"), []byte("other-request"))
	changed = bytes.ReplaceAll(changed, []byte("constructed-job"), []byte("other-provider-job"))
	path := filepath.Join(t.TempDir(), "changed-request.json")
	if err := os.WriteFile(path, changed, 0600); err != nil {
		t.Fatal(err)
	}
	changedHash := sha256.Sum256(changed)
	var after bytes.Buffer
	changedArgs := append(append([]string{}, args...), "-nvsentinel", path, "-nvsentinel-sha256", hex.EncodeToString(changedHash[:]))
	if err := runHealthCapacity(context.Background(), changedArgs, &after); err != nil {
		t.Fatal(err)
	}
	result := decodeCommandHealthOutput(t, after.Bytes())
	if result.Validation == nil {
		t.Fatal("changed request import lost")
	}
	if result.Validation.Authorization != "unavailable" || result.Validation.DeviceAssociation != "unavailable" || result.Validation.Coverage != "unverified" || result.Validation.OriginProof != "unavailable" {
		t.Fatal("recomputed bytes acquired proof")
	}
	result.Incident.ExternalRefs = nil
	if !reflect.DeepEqual(baseline.Incident, result.Incident) {
		t.Fatal("changed request altered passive incident")
	}
}

func TestHealthCapacityCommandRefusals(t *testing.T) {
	args, configPath, _, _, _ := commandHealthFiles(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := runHealthCapacity(ctx, args, io.Discard); !errors.Is(err, context.Canceled) {
		t.Fatal("cancellation hidden", err)
	}
	for _, extra := range [][]string{{"-max-bytes", "1"}, {"-output-bytes", "1"}, {"-nvsentinel-sha256", strings.Repeat("a", 64)}, {"extra"}, {"-unknown"}} {
		var output bytes.Buffer
		if err := runHealthCapacity(context.Background(), append(append([]string{}, args...), extra...), &output); err == nil || output.Len() != 0 {
			t.Fatal("refusal wrote output or succeeded", extra)
		}
	}
	var output bytes.Buffer
	if err := runHealthCapacity(context.Background(), args, operatorCommandShortWriter{}); !errors.Is(err, io.ErrShortWrite) {
		t.Fatal("short write lost", err)
	}
	broken := append([]string{}, args...)
	broken[1] = t.TempDir()
	if err := runHealthCapacity(context.Background(), broken, &output); err == nil {
		t.Fatal("nonregular input read")
	}
	original, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	for _, bad := range [][]byte{[]byte("{"), append(append([]byte{}, original...), []byte(" {}")...), bytes.Replace(original, []byte(`"telemetry":`), []byte(`"telemetry":{},"telemetry":`), 1)} {
		if err := os.WriteFile(configPath, bad, 0600); err != nil {
			t.Fatal(err)
		}
		if err := runHealthCapacity(context.Background(), args, &output); err == nil || output.Len() != 0 {
			t.Fatal("invalid config accepted")
		}
	}
}

func TestHealthCapacityCommandPassiveSource(t *testing.T) {
	for _, path := range []string{"health_capacity.go", "health_capacity_nvsentinel.go"} {
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		file, err := parser.ParseFile(token.NewFileSet(), path, raw, 0)
		if err != nil {
			t.Fatal(err)
		}
		allowed := map[string]bool{"bytes": true, "context": true, "crypto/sha256": true, "encoding/hex": true, "encoding/json": true, "errors": true, "flag": true, "fmt": true, "io": true, "os": true, "os/signal": true, "strings": true, "syscall": true, "unicode": true, "unicode/utf8": true, "slices": true, "time": true, "github.com/tmc/lib/nvidia/x/cudadiag": true, "github.com/tmc/lib/nvidia/x/cudaprof/tracecontract": true, "github.com/tmc/lib/nvidia/x/nvtelemetry": true}
		for _, imp := range file.Imports {
			if !allowed[strings.Trim(imp.Path.Value, `"`)] {
				t.Fatal("passive importer gained unexpected capability import", imp.Path.Value)
			}
		}
		ast.Inspect(file, func(node ast.Node) bool {
			if call, ok := node.(*ast.CallExpr); ok {
				if selector, ok := call.Fun.(*ast.SelectorExpr); ok {
					switch selector.Sel.Name {
					case "StartProcess", "Command", "CommandContext", "Start", "Dial", "Get", "Post", "Launch", "Reset", "Cordon", "Drain", "Init", "Query":
						t.Error("passive command contains action/network operation", selector.Sel.Name)
					}
				}
			}
			return true
		})
	}
}
