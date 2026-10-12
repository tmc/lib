package cudadiag_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

type interchangeClock struct{ now int64 }

func (c *interchangeClock) NowMonotonic() int64      { return c.now }
func (c *interchangeClock) NowUnix() int64           { return c.now }
func (c *interchangeClock) MonotonicDomain() string  { return "monotonic" }
func (c *interchangeClock) MonotonicEpochID() string { return "protocol-epoch" }
func (c *interchangeClock) WallDomain() string       { return "unix" }
func (c *interchangeClock) Provenance() string       { return "synthetic protocol control" }

type interchangeSampler struct {
	readings []nvtelemetry.Reading
	clock    *interchangeClock
	device   nvtelemetry.DeviceMetadata
}

func (s *interchangeSampler) Devices() []nvtelemetry.DeviceMetadata {
	return []nvtelemetry.DeviceMetadata{s.device}
}
func (s *interchangeSampler) Sample(context.Context) ([]nvtelemetry.Reading, error) {
	s.clock.now = 200000
	return s.readings, nil
}

func mixedProtocol() ([]nvtelemetry.Reading, cudadiag.TelemetryOptions) {
	f := cudadiag.BuildSyntheticFixture(cudadiag.InjectionNone)
	raw := []byte("synthetic mixed observation protocol control; no physical producer ran")
	transform := []byte("protocol identity transformation v1")
	hash := fmt.Sprintf("%x", sha256.Sum256(raw))
	transformHash := fmt.Sprintf("%x", sha256.Sum256(transform))
	f.Input.Receipt.InputSHA256 = hash
	f.Input.Receipt.CaptureBounds = &tracecontract.CaptureBounds{Start: 1000, End: 200000, Clock: "monotonic"}
	assignment := nvtelemetry.EntityAssignment{Process: nvtelemetry.EntityProcess{Process: f.Workload.Process, BootID: "protocol-boot", NodeID: "protocol-node", PIDNamespace: "protocol-pidns", HasExecGeneration: true, ExecGeneration: 1}, Device: nvtelemetry.DeviceMetadata{UUID: "GPU-protocol", MIGScope: nvtelemetry.MIGScopeDisabled}, Owner: nvtelemetry.EntityOwner{AllocationID: "protocol-allocation"}, Source: "protocol-control", SourceRevision: "1", State: "assigned", Authoritative: true, ClockDomain: "monotonic", ClockEpoch: "protocol-epoch", StartNanos: 0, EndNanos: 1000000, Sharing: "exclusive"}
	observer := nvtelemetry.Observer{ID: "protocol-observer", Version: "1", Mode: "activity", Pass: "single", Population: "all protocol records", Filter: "none", Replay: "none"}
	opt := cudadiag.TelemetryOptions{Workload: f.Workload, OpID: f.OpID, Assignment: assignment, ClockEpoch: "protocol-epoch", Observer: observer, Receipt: &f.Input.Receipt, RawArtifacts: map[string][]byte{hash: raw, transformHash: transform}, ExpectedSources: map[string]nvtelemetry.Observation{}, WallInterval: f.Input.WallInterval, Thresholds: f.Input.Thresholds}
	var readings []nvtelemetry.Reading
	add := func(name, family, kind string, span tracecontract.Interval) *nvtelemetry.Observation {
		z := float64(0)
		unc := int64(10)
		o := &nvtelemetry.Observation{Family: family, SourceClass: "synthetic", Producer: "protocol-fixture", ProducerVersion: "1", ProducerSchema: "synthetic.v1", RawSHA256: hash, Transform: "identity", TransformSHA256: transformHash, Assignment: &assignment, Workload: &f.Workload, OpID: f.OpID, Interval: &span, ClockEpoch: "protocol-epoch", UncertaintyNS: &unc, Observer: observer, Kind: kind, TID: f.Input.Thresholds.TargetTID}
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
	opt.ExpectedSources[hash] = *host
	return readings, opt
}

func collectMixed(t *testing.T, readings []nvtelemetry.Reading, opt cudadiag.TelemetryOptions) []byte {
	return collectMixedVersion(t, readings, opt, nvtelemetry.SchemaVersionV3)
}

func collectMixedVersion(t *testing.T, readings []nvtelemetry.Reading, opt cudadiag.TelemetryOptions, version int) []byte {
	t.Helper()
	var b bytes.Buffer
	clock := &interchangeClock{now: 1000}
	sampler := &interchangeSampler{readings: readings, clock: clock, device: opt.Assignment.Device}
	cfg := nvtelemetry.CollectorConfig{SchemaVersion: version, Receipt: opt.Receipt}
	c := nvtelemetry.NewCollector(opt.Workload.RunID, sampler, clock, cfg, nvtelemetry.NewWriter(&b, cfg))
	if err := c.InitStream(); err != nil {
		t.Fatal(err)
	}
	if err := c.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Stop(); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

func TestTelemetryV4ObserverCoverageSemanticAdmission(t *testing.T) {
	readings, opt := mixedProtocol()
	opt.Observer.SamplingCoverage = &nvtelemetry.SamplingCoverage{Status: "unknown", Basis: "time", Scope: "activity-window-v1"}
	for i := range readings {
		coverage := *opt.Observer.SamplingCoverage
		readings[i].Observation.Observer.SamplingCoverage = &coverage
	}
	// The expected source is a separately allocated, semantically equal value.
	expected := opt.ExpectedSources[readings[0].Observation.RawSHA256]
	coverage := *opt.Observer.SamplingCoverage
	expected.Observer.SamplingCoverage = &coverage
	opt.ExpectedSources[readings[0].Observation.RawSHA256] = expected
	raw := collectMixedVersion(t, readings, opt, nvtelemetry.SchemaVersionV4)
	got, err := cudadiag.DiagnoseTelemetry(context.Background(), bytes.NewReader(raw), opt)
	if err != nil || !got.Diagnosis.Admitted {
		t.Fatalf("explicit unknown erased point evidence: %+v, %v", got.Diagnosis, err)
	}
	if !reflect.DeepEqual(got.Observations[0].Reading.Observation.Observer.SamplingCoverage, opt.Observer.SamplingCoverage) {
		t.Fatal("V4 coverage descriptor was not retained")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := cudadiag.DiagnoseTelemetry(ctx, bytes.NewReader(raw), opt); !errors.Is(err, context.Canceled) {
		t.Fatalf("V4 cancellation ignored: %v", err)
	}
	bounded := opt
	bounded.MaxBytes = int64(len(raw) - 1)
	if _, err := cudadiag.DiagnoseTelemetry(context.Background(), bytes.NewReader(raw), bounded); !errors.Is(err, nvtelemetry.ErrBudgetExceeded) {
		t.Fatalf("V4 byte limit ignored: %v", err)
	}

	for _, lossCase := range []struct {
		name   string
		mutate func(map[string]tracecontract.Loss)
		cause  string
	}{
		{"loss-absent", func(loss map[string]tracecontract.Loss) { delete(loss, "nccl") }, "unassessed"},
		{"loss-dropped", func(loss map[string]tracecontract.Loss) {
			item := loss["nccl"]
			item.Total++
			item.Dropped++
			loss["nccl"] = item
		}, "records lost"},
		{"loss-unknown", func(loss map[string]tracecontract.Loss) {
			item := loss["nccl"]
			item.Unknown = true
			loss["nccl"] = item
		}, "unassessed"},
	} {
		t.Run(lossCase.name, func(t *testing.T) {
			var records []nvtelemetry.StreamRecord
			for _, line := range bytes.Split(bytes.TrimSpace(raw), []byte("\n")) {
				var record nvtelemetry.StreamRecord
				if err := json.Unmarshal(line, &record); err != nil {
					t.Fatal(err)
				}
				records = append(records, record)
			}
			lossCase.mutate(records[len(records)-1].Manifest.LostRecordsByFamily)
			var mutated bytes.Buffer
			for _, record := range records {
				if err := json.NewEncoder(&mutated).Encode(record); err != nil {
					t.Fatal(err)
				}
			}
			result, err := cudadiag.DiagnoseTelemetry(context.Background(), &mutated, opt)
			if err != nil {
				t.Fatal(err)
			}
			if !result.Diagnosis.Admitted {
				t.Fatalf("loss in one family erased unrelated diagnosis: %+v", result.Diagnosis)
			}
			for _, observation := range result.Observations {
				if observation.Reading.Observation.Family == "nccl" {
					if observation.Finding.Admission.Status != "withheld" || !strings.Contains(observation.Finding.Admission.Reason, lossCase.cause) {
						t.Fatalf("NCCL loss cause not preserved: %+v", observation.Finding.Admission)
					}
				} else if observation.Finding.Admission.Status != "admitted" {
					t.Fatalf("unrelated family withheld: %+v", observation.Finding.Admission)
				}
			}
		})
	}

	mutants := []struct {
		name string
		edit func(*cudadiag.TelemetryOptions)
	}{
		{"expected-source-scope", func(o *cudadiag.TelemetryOptions) {
			expect := o.ExpectedSources[readings[0].Observation.RawSHA256]
			bad := *expect.Observer.SamplingCoverage
			bad.Scope = "other-window"
			expect.Observer.SamplingCoverage = &bad
			o.ExpectedSources[readings[0].Observation.RawSHA256] = expect
		}},
		{"nil-versus-explicit-unknown", func(o *cudadiag.TelemetryOptions) { o.Observer.SamplingCoverage = nil }},
	}
	for _, tc := range mutants {
		t.Run(tc.name, func(t *testing.T) {
			mutated := opt
			mutated.ExpectedSources = make(map[string]nvtelemetry.Observation, len(opt.ExpectedSources))
			for key, value := range opt.ExpectedSources {
				mutated.ExpectedSources[key] = value
			}
			tc.edit(&mutated)
			result, err := cudadiag.DiagnoseTelemetry(context.Background(), bytes.NewReader(raw), mutated)
			if err != nil {
				t.Fatal(err)
			}
			if result.Observations[0].Finding.Admission.Status != "withheld" {
				t.Fatalf("observer mismatch admitted: %+v", result.Observations[0].Finding.Admission)
			}
		})
	}
}

func TestTelemetryV4CoverageAtPublicDiagnosisBoundary(t *testing.T) {
	for _, tc := range []struct {
		name     string
		coverage func(nvtelemetry.Reading) *nvtelemetry.SamplingCoverage
		check    func(*testing.T, *nvtelemetry.SamplingCoverage, nvtelemetry.Reading)
	}{
		{
			name: "measured-zero-time",
			coverage: func(reading nvtelemetry.Reading) *nvtelemetry.SamplingCoverage {
				span := uint64(reading.Observation.Interval.End - reading.Observation.Interval.Start)
				zero := uint64(0)
				return &nvtelemetry.SamplingCoverage{
					Status: "measured", Basis: "time", Scope: "activity-union/eligible-window-v1",
					Numerator: &nvtelemetry.Number{Uint: &zero}, Denominator: &nvtelemetry.Number{Uint: &span}, Unit: "ns",
				}
			},
			check: func(t *testing.T, got *nvtelemetry.SamplingCoverage, reading nvtelemetry.Reading) {
				t.Helper()
				span := uint64(reading.Observation.Interval.End - reading.Observation.Interval.Start)
				if got.Status != "measured" || got.Numerator == nil || got.Numerator.Uint == nil || *got.Numerator.Uint != 0 || got.Denominator == nil || got.Denominator.Uint == nil || *got.Denominator.Uint != span || got.Unit != "ns" {
					t.Fatalf("measured-zero coverage changed: %+v", got)
				}
			},
		},
		{
			name: "explicit-unknown",
			coverage: func(nvtelemetry.Reading) *nvtelemetry.SamplingCoverage {
				return &nvtelemetry.SamplingCoverage{Status: "unknown", Basis: "time", Scope: "activity-union/eligible-window-v1"}
			},
			check: func(t *testing.T, got *nvtelemetry.SamplingCoverage, _ nvtelemetry.Reading) {
				t.Helper()
				if got.Status != "unknown" || got.Numerator != nil || got.Denominator != nil || got.Unit != "" {
					t.Fatalf("explicit unknown coverage acquired a ratio: %+v", got)
				}
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			readings, opt := mixedProtocol()
			coverage := tc.coverage(readings[0])
			setProtocolSamplingCoverage(readings, &opt, coverage)
			// A single retained host record keeps the observer interval and exact
			// time denominator aligned. The diagnosis remains withheld because this
			// input does not establish the complete host/device correlation.
			raw := collectMixedVersion(t, readings[:1], opt, nvtelemetry.SchemaVersionV4)
			manifest, _, samples, err := nvtelemetry.ReadStreamStrict(bytes.NewReader(raw), 0)
			if err != nil {
				t.Fatalf("strict V4 reader: %v", err)
			}
			if manifest.SchemaVersion != nvtelemetry.SchemaVersionV4 || len(samples) != 1 || len(samples[0].Readings) != 1 {
				t.Fatalf("strict V4 stream shape: schema=%d samples=%d", manifest.SchemaVersion, len(samples))
			}
			strictReading := samples[0].Readings[0]
			strictCoverage := strictReading.Observation.Observer.SamplingCoverage
			tc.check(t, strictCoverage, strictReading)

			got, err := cudadiag.DiagnoseTelemetry(context.Background(), bytes.NewReader(raw), opt)
			if err != nil {
				t.Fatalf("public diagnosis: %v", err)
			}
			if len(got.Observations) != 1 || got.Observations[0].Finding.Admission.Status != "admitted" {
				t.Fatalf("valid retained reading was not individually admitted: %+v", got.Observations)
			}
			if !reflect.DeepEqual(got.Observations[0].Reading.Observation.Observer.SamplingCoverage, strictCoverage) {
				t.Fatal("diagnosis did not retain strict-reader sampling coverage")
			}
			if got.Diagnosis.Admitted || got.Diagnosis.Mechanism != cudadiag.MechanismWithheld {
				t.Fatalf("one retained reading established a whole diagnosis: %+v", got.Diagnosis)
			}
		})
	}
}

func TestTelemetryV4ExactReadingAndCoverageOperandsAtPublicDiagnosisBoundary(t *testing.T) {
	readings, opt := mixedProtocol()
	large := uint64(1<<53) + 12345
	denominator := large + 987
	coverage := &nvtelemetry.SamplingCoverage{
		Status: "measured", Basis: "event", Scope: "eligible-activity-events-v1", Unit: "events",
		Numerator: &nvtelemetry.Number{Uint: &large}, Denominator: &nvtelemetry.Number{Uint: &denominator},
	}
	setProtocolSamplingCoverage(readings, &opt, coverage)
	readings[0].Number = &nvtelemetry.Number{Uint: &large}
	raw := collectMixedVersion(t, readings[:1], opt, nvtelemetry.SchemaVersionV4)
	_, _, samples, err := nvtelemetry.ReadStreamStrict(bytes.NewReader(raw), 0)
	if err != nil {
		t.Fatalf("strict V4 reader: %v", err)
	}
	strictReading := samples[0].Readings[0]
	strictCoverage := strictReading.Observation.Observer.SamplingCoverage
	if strictReading.Number.Uint == nil || *strictReading.Number.Uint != large || strictReading.Number.Float != nil {
		t.Fatalf("strict reader rounded the exact point value: %+v", strictReading.Number)
	}
	if strictCoverage.Numerator.Uint == nil || *strictCoverage.Numerator.Uint != large || strictCoverage.Denominator.Uint == nil || *strictCoverage.Denominator.Uint != denominator {
		t.Fatalf("strict reader rounded exact coverage operands: %+v", strictCoverage)
	}
	got, err := cudadiag.DiagnoseTelemetry(context.Background(), bytes.NewReader(raw), opt)
	if err != nil {
		t.Fatalf("public diagnosis: %v", err)
	}
	if len(got.Observations) != 1 {
		t.Fatalf("retained observations = %d, want 1", len(got.Observations))
	}
	retained := got.Observations[0]
	if retained.Reading.Number.Uint == nil || *retained.Reading.Number.Uint != large || retained.Finding.Value != nil {
		t.Fatalf("diagnosis lost exact integer or converted it to float: %+v", retained)
	}
	retainedCoverage := retained.Reading.Observation.Observer.SamplingCoverage
	if retainedCoverage.Numerator.Uint == nil || *retainedCoverage.Numerator.Uint != large || retainedCoverage.Denominator.Uint == nil || *retainedCoverage.Denominator.Uint != denominator {
		t.Fatalf("diagnosis lost exact coverage operands: %+v", retainedCoverage)
	}
	if retained.Finding.Admission.Status != "admitted" || got.Diagnosis.Admitted {
		t.Fatalf("retained-value admission was conflated with whole-diagnosis admission: finding=%+v diagnosis=%+v", retained.Finding.Admission, got.Diagnosis)
	}
}

func TestTelemetryV4RequiredActivityLossWithholdsDependentDiagnosis(t *testing.T) {
	readings, opt := mixedProtocol()
	setProtocolSamplingCoverage(readings, &opt, &nvtelemetry.SamplingCoverage{Status: "unknown", Basis: "time", Scope: "activity-union/eligible-window-v1"})
	raw := collectMixedVersion(t, readings, opt, nvtelemetry.SchemaVersionV4)

	for _, tc := range []struct {
		name   string
		mutate func(map[string]tracecontract.Loss)
		want   string
	}{
		{
			name: "missing-activity-loss",
			mutate: func(loss map[string]tracecontract.Loss) {
				delete(loss, "activity")
			},
			want: "unassessed",
		},
		{
			name: "unknown-activity-loss",
			mutate: func(loss map[string]tracecontract.Loss) {
				item := loss["activity"]
				item.Unknown = true
				loss["activity"] = item
			},
			want: "unassessed",
		},
		{
			name: "dropped-activity-record",
			mutate: func(loss map[string]tracecontract.Loss) {
				item := loss["activity"]
				item.Total++
				item.Dropped++
				loss["activity"] = item
			},
			want: "required family records lost",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mutated := rewriteV4CompletionLoss(t, raw, tc.mutate)
			manifest, _, _, err := nvtelemetry.ReadStreamStrict(bytes.NewReader(mutated), 0)
			if err != nil {
				t.Fatalf("strict V4 reader: %v", err)
			}
			if _, ok := manifest.LostRecordsByFamily["activity"]; tc.name == "missing-activity-loss" && ok {
				t.Fatal("activity loss mutation was not applied")
			}
			got, err := cudadiag.DiagnoseTelemetry(context.Background(), bytes.NewReader(mutated), opt)
			if err != nil {
				t.Fatalf("public diagnosis: %v", err)
			}
			if got.Diagnosis.Admitted || got.Diagnosis.Mechanism != cudadiag.MechanismWithheld {
				t.Fatalf("required activity loss admitted diagnosis: %+v", got.Diagnosis)
			}
			if len(got.Observations) != len(readings) {
				t.Fatalf("withheld records disappeared: got %d observations, want %d", len(got.Observations), len(readings))
			}
			var sawActivity bool
			for _, observation := range got.Observations {
				admission := observation.Finding.Admission
				if observation.Reading.Observation.Family == "activity" {
					sawActivity = true
					if admission.Status != "withheld" || !strings.Contains(admission.Reason, tc.want) {
						t.Fatalf("activity dependency not withheld with %q: %+v", tc.want, admission)
					}
					if observation.Reading.Observation.Observer.SamplingCoverage == nil {
						t.Fatal("withheld activity record lost its sampling descriptor")
					}
				} else if admission.Status != "admitted" {
					t.Fatalf("unrelated retained reading was withheld: family=%s admission=%+v", observation.Reading.Observation.Family, admission)
				}
			}
			if !sawActivity {
				t.Fatal("stream contained no retained activity dependency")
			}
		})
	}
}

func setProtocolSamplingCoverage(readings []nvtelemetry.Reading, opt *cudadiag.TelemetryOptions, coverage *nvtelemetry.SamplingCoverage) {
	opt.Observer.SamplingCoverage = coverage
	for i := range readings {
		readings[i].Observation.Observer.SamplingCoverage = coverage
	}
	for key, expected := range opt.ExpectedSources {
		expected.Observer.SamplingCoverage = coverage
		opt.ExpectedSources[key] = expected
	}
}

func rewriteV4CompletionLoss(t *testing.T, raw []byte, mutate func(map[string]tracecontract.Loss)) []byte {
	t.Helper()
	lines := bytes.Split(bytes.TrimSpace(raw), []byte("\n"))
	var out bytes.Buffer
	for i, line := range lines {
		var record nvtelemetry.StreamRecord
		if err := json.Unmarshal(line, &record); err != nil {
			t.Fatal(err)
		}
		if i == len(lines)-1 {
			if record.Manifest == nil {
				t.Fatal("final stream record is not the completion manifest")
			}
			mutate(record.Manifest.LostRecordsByFamily)
		}
		if err := json.NewEncoder(&out).Encode(record); err != nil {
			t.Fatal(err)
		}
	}
	return out.Bytes()
}

func TestTelemetryPublicCollectorDiagnosis(t *testing.T) {
	r, opt := mixedProtocol()
	raw := collectMixed(t, r, opt)
	first, err := cudadiag.DiagnoseTelemetry(context.Background(), bytes.NewReader(raw), opt)
	if err != nil {
		t.Fatal(err)
	}
	if !first.Diagnosis.Admitted || first.Diagnosis.Mechanism != cudadiag.MechanismNoFault {
		t.Fatalf("mixed consumer withheld: %+v", first.Diagnosis)
	}
	if first.Diagnosis.PrimaryFinding.Class != tracecontract.FindingAssociated {
		t.Fatal("synthetic became physical observation")
	}
	if len(first.Observations) != len(r) {
		t.Fatal("observations lost")
	}
	for _, o := range first.Observations {
		if o.Finding.Admission.Status != "admitted" || o.Reading.Number == nil {
			t.Fatalf("missing roundtrip admission: %+v", o)
		}
	}
	second, err := cudadiag.DiagnoseTelemetry(context.Background(), bytes.NewReader(raw), opt)
	if err != nil || !reflect.DeepEqual(first, second) {
		t.Fatal("nondeterministic replay")
	}
}

func TestTelemetryIndependentAdmissionMutants(t *testing.T) {
	cases := []struct {
		name, want string
		mutate     func([]nvtelemetry.Reading, *cudadiag.TelemetryOptions)
	}{
		{"missing identity", "workload identity", func(r []nvtelemetry.Reading, o *cudadiag.TelemetryOptions) { r[0].Observation.Workload = nil }},
		{"missing epoch", "missing epoch", func(r []nvtelemetry.Reading, o *cudadiag.TelemetryOptions) { r[0].Observation.ClockEpoch = "" }},
		{"ambiguous clock", "missing epoch", func(r []nvtelemetry.Reading, o *cudadiag.TelemetryOptions) { r[0].Observation.Interval.Clock = "other" }},
		{"artifact corruption", "corrupt raw", func(r []nvtelemetry.Reading, o *cudadiag.TelemetryOptions) {
			o.RawArtifacts[r[0].Observation.RawSHA256] = []byte("corrupted")
		}},
		{"source relabel", "source provenance", func(r []nvtelemetry.Reading, o *cudadiag.TelemetryOptions) { r[0].Observation.SourceClass = "native" }},
		{"missing independent source", "source provenance", func(r []nvtelemetry.Reading, o *cudadiag.TelemetryOptions) { o.ExpectedSources = nil }},
		{"producer relabel", "source provenance", func(r []nvtelemetry.Reading, o *cudadiag.TelemetryOptions) { r[0].Observation.Producer = "nvml" }},
		{"PID reuse", "allocation identity", func(r []nvtelemetry.Reading, o *cudadiag.TelemetryOptions) {
			copy := *r[0].Observation.Assignment
			copy.Process.Process.Generation++
			r[0].Observation.Assignment = &copy
		}},
		{"MIG reuse", "allocation identity", func(r []nvtelemetry.Reading, o *cudadiag.TelemetryOptions) {
			copy := *r[0].Observation.Assignment
			copy.MIGConfigurationEpoch = "new-epoch"
			r[0].Observation.Assignment = &copy
		}},
		{"mode conflation", "observer mode", func(r []nvtelemetry.Reading, o *cudadiag.TelemetryOptions) { r[0].Observation.Observer.Mode = "range" }},
		{"pass conflation", "observer mode", func(r []nvtelemetry.Reading, o *cudadiag.TelemetryOptions) { r[0].Observation.Observer.Pass = "pass2" }},
		{"denominator absent", "denominator", func(r []nvtelemetry.Reading, o *cudadiag.TelemetryOptions) {
			r[0].Evidence = nvtelemetry.EvidenceDerived
		}},
		{"reset", "counter reset", func(r []nvtelemetry.Reading, o *cudadiag.TelemetryOptions) {
			r[0].Observation.Kind = "counter"
			r[0].Observation.CounterEpoch = "counter-1"
			r[0].Observation.CounterReset = true
		}},
		{"histogram boundary", "histogram boundary", func(r []nvtelemetry.Reading, o *cudadiag.TelemetryOptions) {
			r[0].Number = nil
			r[0].Observation.Kind = "histogram"
			r[0].Observation.Histogram = &nvtelemetry.Histogram{Bounds: []string{"1", "+Inf"}, Counts: []uint64{0, 1}}
			o.HistogramBounds = map[string][]string{r[0].Name: {"2", "+Inf"}}
		}},
		{"unknown family", "unassessed", func(r []nvtelemetry.Reading, o *cudadiag.TelemetryOptions) { r[0].Observation.Family = "unknown" }},
		{"missing capture receipt", "independent trace receipt", func(r []nvtelemetry.Reading, o *cudadiag.TelemetryOptions) { o.Receipt = nil }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r, opt := mixedProtocol()
			raw := collectMixed(t, r, opt) // freeze the independently collected original before mutating caller or stream
			var records []nvtelemetry.StreamRecord
			for _, line := range bytes.Split(bytes.TrimSpace(raw), []byte("\n")) {
				var rec nvtelemetry.StreamRecord
				if err := json.Unmarshal(line, &rec); err != nil {
					t.Fatal(err)
				}
				records = append(records, rec)
			}
			streamReadings := records[1].Sample.Readings
			tc.mutate(streamReadings, &opt)
			if tc.name == "unknown family" {
				delete(records[len(records)-1].Manifest.LostRecordsByFamily, "host")
			}
			// Family-loss mutant changes the family after collection, so no new report
			// of that family is supplied by the writer's original completion manifest.
			var mutated bytes.Buffer
			for _, rec := range records {
				if err := json.NewEncoder(&mutated).Encode(rec); err != nil {
					t.Fatal(err)
				}
			}
			got, err := cudadiag.DiagnoseTelemetry(context.Background(), &mutated, opt)
			if err != nil {
				t.Fatal(err)
			}
			finding := got.Observations[0].Finding
			if finding.Admission.Status != "withheld" || !strings.Contains(finding.Admission.Reason, tc.want) {
				t.Fatalf("mutant admitted or wrong cause: %+v", finding)
			}
			if len(got.Observations) != len(r) {
				t.Fatal("withheld reading disappeared")
			}
		})
	}
}

func TestTelemetryFamilyIsolationAndCancellation(t *testing.T) {
	r, opt := mixedProtocol()
	raw := collectMixed(t, r, opt)
	var records []nvtelemetry.StreamRecord
	for _, line := range bytes.Split(bytes.TrimSpace(raw), []byte("\n")) {
		var rec nvtelemetry.StreamRecord
		json.Unmarshal(line, &rec)
		records = append(records, rec)
	}
	last := records[len(records)-1].Manifest
	loss := last.LostRecordsByFamily["nccl"]
	loss.Dropped = 1
	loss.Total++
	last.LostRecordsByFamily["nccl"] = loss
	var b bytes.Buffer
	for _, rec := range records {
		json.NewEncoder(&b).Encode(rec)
	}
	got, err := cudadiag.DiagnoseTelemetry(context.Background(), bytes.NewReader(b.Bytes()), opt)
	if err != nil {
		t.Fatal(err)
	}
	if !got.Diagnosis.Admitted {
		t.Fatalf("unrelated loss erased diagnosis: %+v", got.Diagnosis)
	}
	for _, o := range got.Observations {
		if o.Reading.Observation.Family == "nccl" {
			if !strings.Contains(o.Finding.Admission.Reason, "records lost") {
				t.Fatal("loss erased")
			}
		} else if o.Finding.Admission.Status != "admitted" {
			t.Fatal("unrelated family withheld")
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := cudadiag.DiagnoseTelemetry(ctx, bytes.NewReader(raw), opt); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation ignored: %v", err)
	}
	opt.MaxBytes = int64(len(raw) - 1)
	if _, err := cudadiag.DiagnoseTelemetry(context.Background(), bytes.NewReader(raw), opt); !errors.Is(err, nvtelemetry.ErrBudgetExceeded) {
		t.Fatalf("consumer byte bound: %v", err)
	}
}

type cancellationReader struct{ cancel context.CancelFunc }

func (r cancellationReader) Read(p []byte) (int, error) { r.cancel(); return 0, context.Canceled }

func TestTelemetryMidReadCancellationAndExactValues(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	if _, err := cudadiag.DiagnoseTelemetry(ctx, cancellationReader{cancel}, cudadiag.TelemetryOptions{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("mid-read cancellation: %v", err)
	}
	r, opt := mixedProtocol()
	n := uint64(^uint64(0))
	r[len(r)-1].Number = &nvtelemetry.Number{Uint: &n}
	raw := collectMixed(t, r, opt)
	out, err := cudadiag.DiagnoseTelemetry(context.Background(), bytes.NewReader(raw), opt)
	if err != nil {
		t.Fatal(err)
	}
	last := out.Observations[len(out.Observations)-1]
	if last.Reading.Number.Uint == nil || *last.Reading.Number.Uint != n || last.Finding.Value != nil || last.Finding.Admission.Status != "admitted" {
		t.Fatalf("integer converted into float finding: %+v", last)
	}
}

func TestTelemetryUnknownLossAndTruncation(t *testing.T) {
	r, opt := mixedProtocol()
	raw := collectMixed(t, r, opt)
	var records []nvtelemetry.StreamRecord
	for _, line := range bytes.Split(bytes.TrimSpace(raw), []byte("\n")) {
		var rec nvtelemetry.StreamRecord
		if err := json.Unmarshal(line, &rec); err != nil {
			t.Fatal(err)
		}
		records = append(records, rec)
	}
	last := records[len(records)-1].Manifest
	loss := last.LostRecordsByFamily["activity"]
	loss.Unknown = true
	last.LostRecordsByFamily["activity"] = loss
	var b bytes.Buffer
	for _, rec := range records {
		json.NewEncoder(&b).Encode(rec)
	}
	got, err := cudadiag.DiagnoseTelemetry(context.Background(), bytes.NewReader(b.Bytes()), opt)
	if err != nil {
		t.Fatal(err)
	}
	if got.Diagnosis.Admitted || !strings.Contains(got.Observations[1].Finding.Admission.Reason, "unassessed") {
		t.Fatal("unknown activity loss admitted")
	}
	b.Reset()
	for _, rec := range records[:len(records)-1] {
		json.NewEncoder(&b).Encode(rec)
	}
	got, err = cudadiag.DiagnoseTelemetry(context.Background(), &b, opt)
	if err != nil {
		t.Fatal(err)
	}
	if got.Diagnosis.Admitted || !strings.Contains(got.Observations[0].Finding.Admission.Reason, "unassessed") {
		t.Fatal("incomplete stream admitted")
	}
}

func ExampleDiagnoseTelemetry() {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := cudadiag.DiagnoseTelemetry(ctx, strings.NewReader(""), cudadiag.TelemetryOptions{})
	fmt.Println(errors.Is(err, context.Canceled)) // Output: true
}
func ExampleTelemetryOptions() {
	o := cudadiag.TelemetryOptions{MaxBytes: 1 << 20, ClockEpoch: "boot-epoch"}
	fmt.Println(o.MaxBytes, o.ClockEpoch) // Output: 1048576 boot-epoch
}
func ExampleTelemetryFinding() {
	f := cudadiag.TelemetryFinding{Reading: nvtelemetry.Reading{Name: "counter"}, Finding: tracecontract.Finding{Class: tracecontract.FindingUnavailable}}
	fmt.Println(f.Reading.Name, f.Finding.Class) // Output: counter unavailable
}
func ExampleTelemetryDiagnosis() {
	d := cudadiag.TelemetryDiagnosis{Diagnosis: cudadiag.DiagnosisResult{Mechanism: cudadiag.MechanismWithheld}}
	fmt.Println(d.Diagnosis.Mechanism) // Output: withheld
}
