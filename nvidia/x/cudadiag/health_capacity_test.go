package cudadiag_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func healthCapacityLimits() (nvtelemetry.HealthLimits, cudadiag.HealthCapacityLimits) {
	return nvtelemetry.HealthLimits{MaxBytes: 32 << 10, MaxDevices: 4, MaxSources: 8, MaxObservations: 32, MaxEdges: 8},
		cudadiag.HealthCapacityLimits{MaxBytes: 4 << 20, MaxFindings: 16, MaxWork: 16, MaxExternalRefs: 16}
}

func healthCapacityFixture(injection cudadiag.InjectionType) (cudadiag.HealthCapacityInput, cudadiag.HealthCapacityLimits, error) {
	return healthCapacityFixtureWithHealth(injection, nil)
}

func healthCapacityFixtureWithHealth(injection cudadiag.InjectionType, mutate func(*nvtelemetry.HealthSnapshot)) (cudadiag.HealthCapacityInput, cudadiag.HealthCapacityLimits, error) {
	healthLimits, limits := healthCapacityLimits()
	readings, telemetry := healthCapacityTelemetry(injection)
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
	telemetryBytes, err := collectHealthCapacityTelemetry(readings, telemetry)
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

func healthCapacityTelemetry(injection cudadiag.InjectionType) ([]nvtelemetry.Reading, cudadiag.TelemetryOptions) {
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
		State: "assigned", Authoritative: true, ClockDomain: "monotonic", ClockEpoch: "protocol-epoch",
		StartNanos: 0, EndNanos: 1000000, Sharing: "exclusive",
	}
	observer := nvtelemetry.Observer{ID: "protocol-observer", Version: "1", Mode: "activity", Pass: "single", Population: "all protocol records", Filter: "none", Replay: "none"}
	opt := cudadiag.TelemetryOptions{
		Workload: f.Workload, OpID: f.OpID, Assignment: assignment, ClockEpoch: "protocol-epoch", Observer: observer,
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
			OpID: f.OpID, Interval: &span, ClockEpoch: "protocol-epoch", UncertaintyNS: &unc, Observer: observer, Kind: kind, TID: f.Input.Thresholds.TargetTID,
		}
		readings = append(readings, nvtelemetry.Reading{Name: name, Unit: "ns", DeviceUUID: "GPU-protocol", Backend: "protocol-fixture", Evidence: nvtelemetry.EvidenceMeasured, Number: &nvtelemetry.Number{Float: &z}, Observation: o})
		return o
	}
	contextID := cudaworkload.ContextStreamID{Context: 1, Stream: 2}
	host := add("cudaLaunchKernel", "host", "host_call", f.Input.HostCalls[0].Span)
	host.HasReturn = true
	host.Host = &cudaworkload.HostSubmission{Process: f.Workload.Process, OpID: f.OpID, Workload: f.Workload, NativeCorrelationID: 7, Provenance: cudaworkload.ProvenanceSynthetic, Context: contextID, TID: f.Input.Thresholds.TargetTID, TimestampNs: uint64(host.Interval.Start), TimestampClock: "monotonic", TimestampEpoch: "protocol-epoch"}
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

func collectHealthCapacityTelemetry(readings []nvtelemetry.Reading, opt cudadiag.TelemetryOptions) ([]byte, error) {
	var b bytes.Buffer
	clock := &interchangeClock{now: 1000}
	sampler := &interchangeSampler{readings: readings, clock: clock, device: opt.Assignment.Device}
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

func TestComposeHealthCapacityAdmitsAssociationOnly(t *testing.T) {
	in, limits, err := healthCapacityFixture(cudadiag.InjectionHostDelay)
	if err != nil {
		t.Fatal(err)
	}
	in.ExternalRefs = []cudadiag.HealthCapacityArtifactRef{{Kind: "request", ID: "req-1", SHA256: strings.Repeat("a", 64), State: "proposed"}, {Kind: "authorization", ID: "auth-1", SHA256: strings.Repeat("b", 64), State: "declined"}, {Kind: "result", ID: "result-1", SHA256: strings.Repeat("c", 64), State: "unexecuted"}}
	got, err := cudadiag.ComposeHealthCapacity(context.Background(), in, limits)
	if err != nil {
		t.Fatal(err)
	}
	if !got.Impact.Admitted || got.ImpactClass != tracecontract.FindingAssociated || got.WithheldReason != "" {
		t.Fatalf("association not admitted: %+v", got.Impact)
	}
	if got.Health.Observations[0].Class != "raw" || got.HealthFindings[0].Class != tracecontract.FindingObserved || got.Diagnosis.Diagnosis.Mechanism != cudadiag.MechanismHostSubmissionDelay {
		t.Fatal("source, rule finding, and diagnosis states were conflated")
	}
	if len(got.ExternalRefs) != 3 || got.ExternalRefs[1].State != "declined" || got.ReceiptSHA256 == "" || got.HealthSHA256 == "" || got.TelemetrySHA256 == "" {
		t.Fatal("independent refs or raw digests were lost")
	}
}

func TestComposeHealthCapacityRequiresAdmittedFindingInterval(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*cudadiag.HealthCapacityInput)
	}{
		{"unknown clock", func(in *cudadiag.HealthCapacityInput) { in.HealthFindings[0].Intervals[0].Clock = "" }},
		{"historical interval", func(in *cudadiag.HealthCapacityInput) {
			in.HealthFindings[0].Intervals[0] = tracecontract.Interval{Start: 1, End: 1, Clock: "monotonic"}
		}},
		{"reversed interval", func(in *cudadiag.HealthCapacityInput) {
			in.HealthFindings[0].Intervals[0] = tracecontract.Interval{Start: 10001, End: 10000, Clock: "monotonic"}
		}},
		{"missing interval", func(in *cudadiag.HealthCapacityInput) { in.HealthFindings[0].Intervals = nil }},
		{"duplicate finding ID", func(in *cudadiag.HealthCapacityInput) {
			in.HealthFindings = append(in.HealthFindings, in.HealthFindings[0])
		}},
		{"disjoint interval", func(in *cudadiag.HealthCapacityInput) {
			in.HealthFindings[0].Intervals[0] = tracecontract.Interval{Start: 40000, End: 40000, Clock: "monotonic"}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			in, limits, err := healthCapacityFixture(cudadiag.InjectionHostDelay)
			if err != nil {
				t.Fatal(err)
			}
			tc.mutate(&in)
			got, err := cudadiag.ComposeHealthCapacity(context.Background(), in, limits)
			if err != nil {
				t.Fatal(err)
			}
			if got.Impact.Admitted || got.ImpactClass != tracecontract.FindingUnavailable || got.WithheldReason == "" {
				t.Fatalf("invalid finding interval admitted: %+v", got.Impact)
			}
			if len(got.Health.Observations) != 1 || got.Health.Observations[0].Class != "raw" || len(got.HealthFindings) != len(in.HealthFindings) {
				t.Fatal("withholding erased source observations or supplied findings")
			}
		})
	}
}

func TestComposeHealthCapacityRequiresSameObservationIntersection(t *testing.T) {
	in, limits, err := healthCapacityFixtureWithHealth(cudadiag.InjectionHostDelay, func(health *nvtelemetry.HealthSnapshot) {
		first := health.Observations[0]
		first.Reading.TimestampMonotonic = 1
		second := health.Observations[0]
		second.Reading.TimestampMonotonic = 10000
		health.Observations = []nvtelemetry.HealthObservation{first, second}
	})
	if err != nil {
		t.Fatal(err)
	}
	in.HealthFindings[0].Intervals = []tracecontract.Interval{{Start: 1, End: 1, Clock: "monotonic"}}
	got, err := cudadiag.ComposeHealthCapacity(context.Background(), in, limits)
	if err != nil {
		t.Fatal(err)
	}
	if got.Impact.Admitted || got.ImpactClass != tracecontract.FindingUnavailable || got.WithheldReason == "" {
		t.Fatalf("separate observation matches admitted: %+v", got.Impact)
	}
	if len(got.Health.Observations) != 2 || got.Health.Observations[0].Reading.TimestampMonotonic != 1 || got.Health.Observations[1].Reading.TimestampMonotonic != 10000 {
		t.Fatal("withholding erased or altered raw source observations")
	}
}

func TestComposeHealthCapacityAllowsDistinctFindingIDs(t *testing.T) {
	in, limits, err := healthCapacityFixture(cudadiag.InjectionHostDelay)
	if err != nil {
		t.Fatal(err)
	}
	second := in.HealthFindings[0]
	second.ID = "health-xid-2"
	in.HealthFindings = append(in.HealthFindings, second)
	got, err := cudadiag.ComposeHealthCapacity(context.Background(), in, limits)
	if err != nil {
		t.Fatal(err)
	}
	if !got.Impact.Admitted || len(got.HealthFindings) != 2 {
		t.Fatalf("distinct findings did not retain admitted association: impact=%+v findings=%d", got.Impact, len(got.HealthFindings))
	}
}

func TestComposeHealthCapacityClonesWithheldDiagnosisWorkload(t *testing.T) {
	in, limits, err := healthCapacityFixture(cudadiag.InjectionHostDelay)
	if err != nil {
		t.Fatal(err)
	}
	in.Telemetry.Workload.Labels = map[string]string{"review": "before"}
	got, err := cudadiag.ComposeHealthCapacity(context.Background(), in, limits)
	if err != nil {
		t.Fatal(err)
	}
	if got.Diagnosis.Diagnosis.Admitted {
		t.Fatal("label mismatch did not exercise the withheld diagnosis path")
	}
	before, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	in.Telemetry.Workload.Labels["review"] = "after"
	after, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, after) || got.Diagnosis.Diagnosis.Workload.Labels["review"] != "before" || got.Workload.Labels["review"] != "before" {
		t.Fatal("caller mutation changed retained workload labels")
	}
}

func TestComposeHealthCapacityWithholdsIndependentMutants(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*cudadiag.HealthCapacityInput)
	}{
		{"missing identity gate", func(in *cudadiag.HealthCapacityInput) { delete(in.Telemetry.Receipt.Gates, "health_identity_binding") }},
		{"unchecked freshness gate", func(in *cudadiag.HealthCapacityInput) {
			g := in.Telemetry.Receipt.Gates["health_freshness"]
			g.Disposition = "unchecked"
			in.Telemetry.Receipt.Gates["health_freshness"] = g
		}},
		{"health digest mismatch", func(in *cudadiag.HealthCapacityInput) {
			in.Telemetry.Receipt.Options["health_sha256"] = strings.Repeat("0", 64)
		}},
		{"assignment boot mismatch", func(in *cudadiag.HealthCapacityInput) {
			in.Telemetry.Receipt.Options["assignment_boot_id"] = "other-boot"
		}},
		{"non-authoritative assignment", func(in *cudadiag.HealthCapacityInput) { in.Telemetry.Assignment.Authoritative = false }},
		{"wrong assigned device", func(in *cudadiag.HealthCapacityInput) { in.Telemetry.Assignment.Device.UUID = "GPU-other" }},
		{"clock mismatch", func(in *cudadiag.HealthCapacityInput) { in.Work[0].ClockEpoch = "other-epoch" }},
		{"incorrect work", func(in *cudadiag.HealthCapacityInput) { in.Work[0].Correct = false }},
		{"no health rule finding", func(in *cudadiag.HealthCapacityInput) { in.HealthFindings = nil }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			in, limits, err := healthCapacityFixture(cudadiag.InjectionHostDelay)
			if err != nil {
				t.Fatal(err)
			}
			tc.mutate(&in)
			got, err := cudadiag.ComposeHealthCapacity(context.Background(), in, limits)
			if err != nil {
				t.Fatal(err)
			}
			if got.Impact.Admitted || got.ImpactClass != tracecontract.FindingUnavailable || got.WithheldReason == "" {
				t.Fatalf("mutant admitted or lacked a reason: %+v", got)
			}
			if len(got.Health.Observations) != 1 || len(got.Diagnosis.Observations) == 0 {
				t.Fatal("withholding erased source evidence")
			}
		})
	}
}

func TestComposeHealthCapacityDoesNotInferHealthFromNoFault(t *testing.T) {
	in, limits, err := healthCapacityFixture(cudadiag.InjectionNone)
	if err != nil {
		t.Fatal(err)
	}
	got, err := cudadiag.ComposeHealthCapacity(context.Background(), in, limits)
	if err != nil {
		t.Fatal(err)
	}
	if got.Diagnosis.Diagnosis.Mechanism != cudadiag.MechanismNoFault || got.Impact.Admitted || got.WithheldReason == "" {
		t.Fatalf("no-fault input implied usable capacity: diagnosis=%+v impact=%+v", got.Diagnosis.Diagnosis, got.Impact)
	}
}

func TestComposeHealthCapacityDistinguishesMeasuredZeroAndUnavailable(t *testing.T) {
	measuredZero, limits, err := healthCapacityFixtureWithHealth(cudadiag.InjectionHostDelay, func(health *nvtelemetry.HealthSnapshot) {
		health.Observations[0].Reading.IntValue = 0
	})
	if err != nil {
		t.Fatal(err)
	}
	zero, err := cudadiag.ComposeHealthCapacity(context.Background(), measuredZero, limits)
	if err != nil {
		t.Fatal(err)
	}
	if !zero.Impact.Admitted || zero.Health.Observations[0].Reading.Evidence != nvtelemetry.EvidenceMeasured || zero.Health.Observations[0].Reading.IntValue != 0 {
		t.Fatal("valid measured zero was lost or treated as unavailable")
	}

	unavailable, limits, err := healthCapacityFixtureWithHealth(cudadiag.InjectionHostDelay, func(health *nvtelemetry.HealthSnapshot) {
		health.Observations[0].Reading.Evidence = nvtelemetry.EvidenceUnavailable
		health.Observations[0].Reading.UnavailableReason = "query_failure"
		health.Observations[0].Reading.IntValue = 0
	})
	if err != nil {
		t.Fatal(err)
	}
	got, err := cudadiag.ComposeHealthCapacity(context.Background(), unavailable, limits)
	if err != nil {
		t.Fatal(err)
	}
	if got.Impact.Admitted || got.Health.Observations[0].Reading.Evidence != nvtelemetry.EvidenceUnavailable || got.Health.Observations[0].Reading.UnavailableReason != "query_failure" {
		t.Fatal("unavailable health reading was collapsed into a measured value")
	}
}

func TestComposeHealthCapacityPreservesZeroUnavailableAndCopiesOutput(t *testing.T) {
	in, limits, err := healthCapacityFixture(cudadiag.InjectionHostDelay)
	if err != nil {
		t.Fatal(err)
	}
	in.ExternalRefs = []cudadiag.HealthCapacityArtifactRef{{Kind: "result", ID: "imported-result", SHA256: strings.Repeat("d", 64), State: "executed"}}
	got, err := cudadiag.ComposeHealthCapacity(context.Background(), in, limits)
	if err != nil {
		t.Fatal(err)
	}
	in.HealthFindings[0].Artifacts[0] = "mutated"
	in.Work[0].Units++
	in.ExternalRefs[0].ID = "mutated"
	if got.HealthFindings[0].Artifacts[0] == "mutated" || got.Work[0].Units == in.Work[0].Units || got.ExternalRefs[0].ID == "mutated" {
		t.Fatal("output aliases caller-owned records")
	}
	if got.Health.Observations[0].Reading.IsAvailable() != true || got.Health.Observations[0].Reading.IntValue != 31 {
		t.Fatal("measured raw value changed")
	}
	if got.Diagnosis.Observations[0].Reading.Evidence == nvtelemetry.EvidenceUnavailable {
		t.Fatal("available zero telemetry was changed into unavailable")
	}
}

func TestComposeHealthCapacityBoundsMalformedAndCancellation(t *testing.T) {
	in, limits, err := healthCapacityFixture(cudadiag.InjectionHostDelay)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := cudadiag.ComposeHealthCapacity(ctx, in, limits); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled composition error=%v", err)
	}

	bad := in
	bad.HealthJSON = []byte("{")
	if _, err := cudadiag.ComposeHealthCapacity(context.Background(), bad, limits); err == nil {
		t.Fatal("malformed health JSON admitted")
	}
	bad = in
	bad.TelemetryJSONL = bad.TelemetryJSONL[:len(bad.TelemetryJSONL)-20]
	if _, err := cudadiag.ComposeHealthCapacity(context.Background(), bad, limits); err == nil {
		t.Fatal("truncated telemetry stream admitted")
	}
	bad = in
	bad.HealthJSON = append(bad.HealthJSON, bytes.Repeat([]byte(" "), int(limits.MaxBytes))...)
	if _, err := cudadiag.ComposeHealthCapacity(context.Background(), bad, limits); err == nil {
		t.Fatal("oversized input admitted")
	}
	bad = in
	bad.ExternalRefs = []cudadiag.HealthCapacityArtifactRef{{Kind: "result", ID: "bad", SHA256: "not-a-digest", State: "executed"}}
	if _, err := cudadiag.ComposeHealthCapacity(context.Background(), bad, limits); err == nil {
		t.Fatal("malformed external ref admitted")
	}
}
