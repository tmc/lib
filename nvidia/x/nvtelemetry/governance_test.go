package nvtelemetry

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/dcgm"
)

func TestGovernedDirectDCGMHealthAndStream(t *testing.T) {
	config := dcgmTestConfig()
	config.Fields = DCGMIMEXFields()
	values := make([]dcgm.FieldValueV1, len(config.Fields))
	for i, f := range config.Fields {
		values[i] = dcgm.FieldValueV1{Version: dcgm.FieldValueVersion1, FieldId: uint16(f.ID), FieldType: dcgm.FieldTypeInt64, Ts: 123456}
		values[i].Value[0] = uint64(i + 1)
	}
	client := &fakeDCGMClient{values: values}
	clock := NewFakeClock(10, 100)
	direct, err := NewDCGMSampler(t.Context(), client, clock, config)
	if err != nil {
		t.Fatal(err)
	}
	defer direct.Close()
	p := operationPolicy(t)
	p.SourceID = config.SourceID
	p.AllowedDevices = []string{"GPU-a"}
	var registry CaptureRegistry
	governed, err := registry.NewSampler(direct, clock, p)
	if err != nil {
		t.Fatal(err)
	}
	defer governed.Close(t.Context())
	readings, err := governed.Sample(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	health := HealthSnapshot{Version: 1, NodeID: "node", TopologyEpoch: "topology", ClockDomain: clock.MonotonicDomain(), ClockEpoch: clock.MonotonicEpochID(), AtNanos: 10, DeviceStates: map[string]string{"GPU-a": "present"}, Sources: []HealthSource{{ID: config.SourceID, Kind: "native", Version: config.ProducerVersion, State: "available", HasLastSuccess: true, LastSuccessNanos: 10}}, Complete: true}
	for _, r := range readings {
		health.Observations = append(health.Observations, HealthObservation{SourceID: config.SourceID, Class: "raw", Scope: "physical/selected", Reading: r})
	}
	limits := HealthLimits{MaxBytes: 65536, MaxDevices: 1, MaxSources: 1, MaxObservations: 2, MaxEdges: 1}
	var healthBytes bytes.Buffer
	if err := WriteHealthSnapshot(&healthBytes, health, limits); err != nil {
		t.Fatal(err)
	}
	decoded, err := ReadHealthSnapshot(&healthBytes, limits)
	if err != nil {
		t.Fatal(err)
	}
	for _, o := range decoded.Observations {
		if o.Reading.HasSourceTimestamp || o.Reading.Source.Native.TimestampMicroseconds != 123456 || o.Reading.Source.Native.Status != 0 {
			t.Fatal("native cache timestamp became sensor time or was lost")
		}
	}
	var raw bytes.Buffer
	collector := NewCollector("governed-imex", governed, clock, p.Collector, NewWriter(&raw, p.Collector))
	if err := collector.InitStream(); err != nil {
		t.Fatal(err)
	}
	if err := collector.Step(t.Context()); err != nil {
		t.Fatal(err)
	}
	if _, err := collector.Stop(); err != nil {
		t.Fatal(err)
	}
	_, _, samples, err := ReadStreamStrict(&raw, 65536)
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != 1 || samples[0].Readings[0].Source.Native.TimestampMicroseconds != 123456 || samples[0].Readings[0].HasSourceTimestamp {
		t.Fatal("strict replay changed source time")
	}
	// An arbitrary sampler can borrow this valid native object; governance must
	// retain its own pointer and enforce native extents before allocating copies.
	for _, mutation := range []string{"clone", "field extent", "text extent", "budget"} {
		t.Run(mutation, func(t *testing.T) {
			input := cloneReadings(readings)
			policy := p
			switch mutation {
			case "field extent":
				input[0].Source.Native.FieldName = strings.Repeat("f", 257)
			case "text extent":
				input[0].Source.Native.RawText = strings.Repeat("x", 257)
			case "budget":
				policy.MaxSampleBytes = 100
			}
			fake := NewFakeSampler(DeviceMetadata{UUID: "GPU-a"})
			fake.Enqueue(input)
			var captures CaptureRegistry
			wrapped, err := captures.NewSampler(fake, clock, policy)
			if err != nil {
				t.Fatal(err)
			}
			defer wrapped.Close(t.Context())
			owned, err := wrapped.Sample(t.Context())
			if mutation != "clone" {
				if err == nil {
					t.Fatal("native extent/budget admitted")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			input[0].Source.Native.TimestampMicroseconds = 99
			input[0].Source.Native.FieldName = "mutated"
			if owned[0].Source.Native.TimestampMicroseconds != 123456 || owned[0].Source.Native.FieldName == "mutated" {
				t.Fatal("borrowed native pointer")
			}
			owned[0].Source.Native.RawBits = 77
			if input[0].Source.Native.RawBits == 77 {
				t.Fatal("native pointer leaked back")
			}
		})
	}
}

func TestGovernedDirectDCGMOpaquePrivacy(t *testing.T) {
	for _, kind := range []uint16{dcgm.FieldTypeString, uint16('b')} {
		t.Run(string(rune(kind)), func(t *testing.T) {
			config := dcgmTestConfig()
			config.Fields = DCGMIMEXFields()[:1]
			value := dcgm.FieldValueV1{Version: dcgm.FieldValueVersion1, FieldId: uint16(config.Fields[0].ID), FieldType: kind, Ts: 123456}
			// Even clearing a string's RawText would leave these first eight bytes
			// in RawBits/RawValue. Reject the complete opaque reading instead.
			value.Value[0] = 0x59454b5f495041 // API_KEY in the native union's low bytes
			client := &fakeDCGMClient{values: []dcgm.FieldValueV1{value}}
			clock := NewFakeClock(10, 100)
			direct, err := NewDCGMSampler(t.Context(), client, clock, config)
			if err != nil {
				t.Fatal(err)
			}
			defer direct.Close()
			rawReadings, err := direct.Sample(t.Context())
			if err != nil {
				t.Fatal(err)
			}
			if rawReadings[0].Source.Native.RawBits != value.Value[0] || rawReadings[0].Source.Native.TimestampMicroseconds != 123456 {
				t.Fatal("raw path fabricated bytes or time")
			}
			policy := operationPolicy(t)
			policy.SourceID = config.SourceID
			policy.AllowedDevices = []string{"GPU-a"}
			var registry CaptureRegistry
			governed, err := registry.NewSampler(direct, clock, policy)
			if err != nil {
				t.Fatal(err)
			}
			defer governed.Close(t.Context())
			var output bytes.Buffer
			collector := NewCollector("privacy", governed, clock, policy.Collector, NewWriter(&output, policy.Collector))
			if err := collector.InitStream(); err != nil {
				t.Fatal(err)
			}
			if err := collector.Step(t.Context()); err == nil || err.Error() != "opaque native diagnostics require explicit raw capture" {
				t.Fatalf("opaque native data admitted: %v", err)
			}
			if strings.Contains(output.String(), rawReadings[0].Source.RawValue) || strings.Contains(output.String(), "API_KEY") {
				t.Fatal("opaque raw bytes reached writer")
			}
			if len(collector.Samples()) != 0 {
				t.Fatal("opaque reading reached collector")
			}
		})
	}
}

func operationPolicy(t *testing.T) CapturePolicy {
	t.Helper()
	f, err := os.Open("testdata/governance_v1.json")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	p, err := ReadCapturePolicy(f, 4096)
	if err != nil {
		t.Fatal(err)
	}
	return *p
}

func TestGovernancePublicCollector(t *testing.T) {
	p := operationPolicy(t)
	clock := NewFakeClock(1, 1)
	source := NewFakeSampler(DeviceMetadata{UUID: "GPU-0"})
	var registry CaptureRegistry
	s, err := registry.NewSampler(source, clock, p)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close(context.Background())
	var raw bytes.Buffer
	c := NewCollector("governed", s, clock, p.Collector, NewWriter(&raw, p.Collector))
	if err := c.InitStream(); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		clock.Advance(time.Millisecond)
		if err := c.Step(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
	clock.Advance(time.Millisecond)
	if _, err := c.Stop(); err != nil {
		t.Fatal(err)
	}
	m, _, samples, err := ReadStreamStrict(&raw, int64(p.Collector.MaxByteBudget))
	if err != nil {
		t.Fatal(err)
	}
	if !m.IsComplete || len(samples) != 3 || s.Status().SuccessfulQueries != 3 {
		t.Fatalf("unexpected collector result %+v %+v", m, s.Status())
	}
	if err := s.Ready(clock.NowMonotonic(), int64(2*time.Millisecond)); err != nil {
		t.Fatal(err)
	}
	if err := s.Ready(clock.NowMonotonic(), 0); err == nil {
		t.Fatal("stale source remained ready")
	}
}

func TestGovernanceSelectionAndRollback(t *testing.T) {
	p := operationPolicy(t)
	clock := NewFakeClock(1, 1)
	var registry CaptureRegistry
	broad := NewFakeSampler(DeviceMetadata{UUID: "GPU-0"}, DeviceMetadata{UUID: "GPU-1"})
	if _, err := registry.NewSampler(broad, clock, p); err == nil {
		t.Fatal("broad capture admitted")
	}
	source := NewFakeSampler(DeviceMetadata{UUID: "GPU-0"})
	s, err := registry.NewSampler(source, clock, p)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := registry.NewSampler(source, clock, p); err == nil {
		t.Fatal("duplicate collector admitted")
	}
	p.MaxReadings = 0
	if _, err := registry.NewSampler(source, clock, p); err == nil {
		t.Fatal("bad reconfiguration admitted")
	}
	if _, err := s.Sample(context.Background()); err != nil {
		t.Fatal("rejected config changed running source", err)
	}
	source.Enqueue([]Reading{{DeviceUUID: "GPU-1"}})
	if _, err := s.Sample(context.Background()); err == nil {
		t.Fatal("cross-target batch admitted")
	}
	if err := s.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	p = operationPolicy(t)
	s2, err := registry.NewSampler(source, clock, p)
	if err != nil {
		t.Fatal("closed lease not released", err)
	}
	s2.Close(context.Background())
}

func TestGovernanceCloseRetainsBusyOwnership(t *testing.T) {
	p := operationPolicy(t)
	clock := NewFakeClock(1, 1)
	var registry CaptureRegistry
	source := NewFakeSampler(DeviceMetadata{UUID: "GPU-0"})
	entered := make(chan struct{})
	release := make(chan struct{})
	source.SetFallback(func(context.Context) ([]Reading, error) { close(entered); <-release; return nil, nil })
	s, err := registry.NewSampler(source, clock, p)
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { _, err := s.Sample(context.Background()); done <- err }()
	<-entered
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := s.Close(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("close=%v", err)
	}
	other := NewFakeSampler(DeviceMetadata{UUID: "GPU-0"})
	if _, err := registry.NewSampler(other, clock, p); err == nil {
		t.Fatal("busy lease released on cancellation")
	}
	close(release)
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("sample=%v", err)
	}
	if err := s.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	next, err := registry.NewSampler(other, clock, p)
	if err != nil {
		t.Fatal(err)
	}
	next.Close(context.Background())
}

func TestGovernanceOptionalUnavailableAndOutage(t *testing.T) {
	p := operationPolicy(t)
	clock := NewFakeClock(1, 1)
	var registry CaptureRegistry
	source := NewFakeSampler(DeviceMetadata{UUID: "GPU-0"})
	source.Enqueue([]Reading{{Name: MetricSMActivity, Unit: UnitPercent, DeviceUUID: "GPU-0", Evidence: EvidenceUnavailable, UnavailableReason: ReasonUnsupported}})
	s, err := registry.NewSampler(source, clock, p)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close(context.Background())
	if _, err := s.Sample(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := s.Ready(1, 0); err != nil {
		t.Fatal("optional field made source unready", err)
	}
	source.SetFallback(func(context.Context) ([]Reading, error) { return nil, errors.New("source disconnected") })
	clock.Advance(time.Second)
	if _, err := s.Sample(context.Background()); err == nil {
		t.Fatal("source outage hidden")
	}
	if err := s.Ready(clock.NowMonotonic(), 1); err == nil {
		t.Fatal("dead source remained ready")
	}
	if s.Status().FailedQueries != 1 {
		t.Fatal("source error not counted")
	}
}

func TestGovernanceBoundedGrowth(t *testing.T) {
	p := operationPolicy(t)
	clock := NewFakeClock(1, 1)
	var registry CaptureRegistry
	s, err := registry.NewSampler(NewFakeSampler(DeviceMetadata{UUID: "GPU-0"}), clock, p)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close(context.Background())
	var raw bytes.Buffer
	c := NewCollector("bounded", s, clock, p.Collector, NewWriter(&raw, p.Collector))
	if err := c.InitStream(); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 10000; i++ {
		clock.Advance(time.Millisecond)
		if err := c.Step(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := c.Stop(); err != nil {
		t.Fatal(err)
	}
	if raw.Len() > p.Collector.MaxByteBudget {
		t.Fatal("byte budget exceeded")
	}
	m, _, samples, err := ReadStream(&raw, int64(p.Collector.MaxByteBudget))
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) > p.Collector.MaxRecordBudget || m.IsComplete || m.LostRecords == 0 {
		t.Fatal("retention/loss contract failed")
	}
	if len(c.Samples()) > p.Collector.MaxBufferedRecords {
		t.Fatal("collector buffer grew past budget")
	}
}

func TestCapturePolicyAndLabels(t *testing.T) {
	b, err := os.ReadFile("testdata/governance_v1.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, input := range []string{string(b) + "{}", strings.Replace(string(b), `"version": 1`, `"version": 9`, 1), strings.Replace(string(b), `"version": 1`, `"version": 1, "version": 1`, 1), strings.Replace(string(b), `"version": 1`, `"version": 1, "Version": 2`, 1), strings.Replace(string(b), `"max_record_budget": 20`, `"max_record_budget": 0`, 1)} {
		if _, err := ReadCapturePolicy(strings.NewReader(input), 4096); err == nil {
			t.Fatal("bad config admitted")
		}
	}
	labels, err := SafeCaptureLabels(map[string]string{"job_id": "123", "api_key": "secret", "model_input": "user text"}, []string{"job_id"})
	if err != nil || len(labels) != 1 || labels["job_id"] != "123" {
		t.Fatal(labels, err)
	}
	for _, key := range []string{"api_key", "environment", "command", "model_input"} {
		if _, err := SafeCaptureLabels(map[string]string{key: "sensitive"}, []string{key}); err == nil {
			t.Fatal("sensitive key selected", key)
		}
	}
	if _, err := SafeCaptureLabels(map[string]string{"executable": "program --token=secret"}, []string{"executable"}); err == nil {
		t.Fatal("sensitive command admitted")
	}
}

func TestOperationsIndependentNegativeControls(t *testing.T) {
	if _, err := ReadCapturePolicy(nil, 4096); err == nil {
		t.Fatal("nil profile reader admitted")
	}
	if _, err := ReadHealthSnapshot(nil, operationHealthLimits()); err == nil {
		t.Fatal("nil health reader admitted")
	}
	b, err := os.ReadFile("testdata/governance_v1.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, input := range []string{strings.Replace(string(b), `"version": 1`, `"VERSION": 1`, 1), strings.Replace(string(b), `"source_id": "fixture-nvml"`, `"source_id": "first", "ſource_id": "fixture-nvml"`, 1), strings.Replace(string(b), `"version": 1`, `"version": null`, 1)} {
		if _, err := ReadCapturePolicy(strings.NewReader(input), 4096); err == nil {
			t.Fatal("alias/null profile admitted")
		}
	}
	p := operationPolicy(t)
	source := NewFakeSampler(DeviceMetadata{UUID: "GPU-0", Warnings: []string{"api_key=SECRET"}, MetadataFailures: []MetadataFailure{{Property: "uuid", Reason: ReasonQueryFailure, Error: "api_key=SECRET"}}})
	var registry CaptureRegistry
	s, err := registry.NewSampler(source, NewFakeClock(1, 1), p)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close(context.Background())
	if d := s.Devices()[0]; len(d.Warnings) != 0 || d.MetadataFailures[0].Error != "" || s.Status().SuppressedDiagnostics != 2 {
		t.Fatal("diagnostics not suppressed")
	}
	source.Enqueue([]Reading{{DeviceUUID: "GPU-0", Evidence: EvidenceUnavailable, UnavailableReason: "api_key=UNIQUE_SECRET_CONTROL"}})
	if readings, err := s.Sample(context.Background()); err == nil || readings != nil {
		t.Fatal("unsafe source diagnostic admitted")
	}
	clock := &operationExtremeClock{}
	var other CaptureRegistry
	s2, err := other.NewSampler(NewFakeSampler(DeviceMetadata{UUID: "GPU-0"}), clock, p)
	if err != nil {
		t.Fatal(err)
	}
	defer s2.Close(context.Background())
	if _, err := s2.Sample(context.Background()); err == nil || s2.Status().SuccessfulQueries != 0 || s2.Status().LastQueryDurationNanos < 0 {
		t.Fatal("clock overflow admitted")
	}
}

type operationExtremeClock struct{ calls int }

func (c *operationExtremeClock) NowMonotonic() int64 {
	c.calls++
	if c.calls == 1 {
		return math.MinInt64
	}
	return math.MaxInt64
}
func (*operationExtremeClock) NowUnix() int64 { return 0 }

func operationManagedReading(t *testing.T) Reading {
	t.Helper()
	sample, err := ReadManaged(context.Background(), strings.NewReader("# TYPE temp gauge\n"+`temp{UUID="GPU-0",job_id="123",api_key="PRIVATE_CONTROL"} 31`+"\n"), ManagedImport{SourceID: "fixture-nvml", Producer: "dcgm", ProducerVersion: "fixture-v1", NativeOrigin: "nvml", QueryStartNanos: 1, QueryEndNanos: 1, Fields: []ManagedField{{Metric: "temp", Name: MetricTemperatureC, Type: "gauge", Numerator: 1, Denominator: 1, SourceUnit: "celsius"}}, AllowedLabels: []string{"job_id", "api_key"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(sample.Readings) != 1 {
		t.Fatal("missing managed fixture")
	}
	return sample.Readings[0]
}

func TestGovernedSourcePrivacyAndOwnership(t *testing.T) {
	p := operationPolicy(t)
	p.AllowedSourceLabels = []string{"job_id"}
	reading := operationManagedReading(t)
	source := NewFakeSampler(DeviceMetadata{UUID: "GPU-0"})
	source.Enqueue([]Reading{reading})
	var registry CaptureRegistry
	s, err := registry.NewSampler(source, NewFakeClock(1, 1), p)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close(context.Background())
	got, err := s.Sample(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if got[0].Source.Labels["job_id"] != "123" || got[0].Source.Labels["api_key"] != "" || s.Status().SuppressedSourceLabels != 1 {
		t.Fatal("source policy did not suppress sensitive metadata")
	}
	reading.Source.RawValue = "999"
	reading.Source.Mapping.Name = "mutated"
	reading.Source.Labels["job_id"] = "changed"
	if got[0].Source.RawValue != "31" || got[0].Source.Mapping.Name != MetricTemperatureC || got[0].Source.Labels["job_id"] != "123" {
		t.Fatal("borrowed source metadata escaped sampler")
	}
	got[0].Source.Labels["UUID"] = "mutated"
	if reading.Source.Labels["UUID"] != "GPU-0" {
		t.Fatal("consumer mutated producer labels")
	}
}

func TestGovernedSourcePreallocationAndMappingControls(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*Reading)
	}{
		{"oversized_labels", func(r *Reading) {
			r.Source.Labels = make(map[string]string)
			for i := 0; i < 17; i++ {
				r.Source.Labels[strings.Repeat("x", i+1)] = "y"
			}
		}},
		{"oversized_raw", func(r *Reading) { r.Source.RawValue = strings.Repeat("1", 4096) }},
		{"oversized_mapping", func(r *Reading) { r.Source.Mapping.Metric = strings.Repeat("x", 4096) }},
		{"bad_source", func(r *Reading) { r.Source.SourceID = "other" }},
		{"bad_producer", func(r *Reading) { r.Source.Producer = "other" }},
		{"bad_uuid", func(r *Reading) { r.Source.Labels["UUID"] = "GPU-other" }},
		{"bad_scale", func(r *Reading) {
			r.Source.Mapping.Numerator = 2
			r.Source.MappingDigest = managedMappingDigest(r.Source.Mapping)
		}},
		{"bad_value", func(r *Reading) { r.Value = 32 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			reading := operationManagedReading(t)
			tc.mutate(&reading)
			source := NewFakeSampler(DeviceMetadata{UUID: "GPU-0"})
			source.Enqueue([]Reading{reading})
			var registry CaptureRegistry
			s, err := registry.NewSampler(source, NewFakeClock(1, 1), operationPolicy(t))
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close(context.Background())
			if got, err := s.Sample(context.Background()); err == nil || got != nil {
				t.Fatal("source contradiction admitted")
			}
		})
	}
	for _, text := range []string{"", "ASCII", "<>&", "\n\t\x00", "é", "\u2028\u2029", string([]byte{0xff})} {
		b, err := json.Marshal(text)
		if err != nil {
			t.Fatal(err)
		}
		size, ok := governedStringSize(text, len(b))
		if !ok || size != len(b) {
			t.Fatalf("escaped size %q got%d want%d", text, size, len(b))
		}
		if _, ok := governedStringSize(text, len(b)-1); ok {
			t.Fatal("escaped size budget ignored")
		}
	}
}
