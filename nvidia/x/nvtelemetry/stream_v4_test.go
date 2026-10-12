package nvtelemetry_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func v4Header() nvtelemetry.Manifest {
	return nvtelemetry.Manifest{RunID: "v4-protocol", SchemaVersion: nvtelemetry.SchemaVersionV4, Devices: []nvtelemetry.DeviceMetadata{{UUID: "GPU-v4"}}}
}

func v4Observer(c *nvtelemetry.SamplingCoverage) nvtelemetry.Observer {
	return nvtelemetry.Observer{ID: "observer", Version: "1", Mode: "activity", Pass: "pass-1", Population: "eligible activity intervals", SamplingCoverage: c}
}

func v4Reading(c *nvtelemetry.SamplingCoverage) nvtelemetry.Reading {
	zero := uint64(0)
	interval := tracecontract.Interval{Start: 10, End: 110, Clock: "mono"}
	return nvtelemetry.Reading{
		Name: "activity", Unit: "count", DeviceUUID: "GPU-v4", Evidence: nvtelemetry.EvidenceMeasured,
		Number:      &nvtelemetry.Number{Uint: &zero},
		Observation: &nvtelemetry.Observation{Family: "activity", SourceClass: "synthetic", Interval: &interval, ClockEpoch: "epoch", Observer: v4Observer(c), Kind: "kernel"},
	}
}

func writeV4(t *testing.T, readings ...nvtelemetry.Reading) []byte {
	t.Helper()
	var b bytes.Buffer
	w := nvtelemetry.NewWriter(&b, nvtelemetry.CollectorConfig{})
	m := v4Header()
	if err := w.WriteManifest(&m); err != nil {
		t.Fatal(err)
	}
	if err := w.WriteSample(&nvtelemetry.Sample{Seq: 1, Readings: readings}); err != nil {
		t.Fatal(err)
	}
	m.TotalSamples = 1
	m.LostRecordsByFamily = w.FamilyLoss()
	m.IsComplete = true
	if err := w.WriteManifest(&m); err != nil {
		t.Fatal(err)
	}
	return append([]byte(nil), b.Bytes()...)
}

func measuredTimeCoverage(numerator, denominator *nvtelemetry.Number) *nvtelemetry.SamplingCoverage {
	return &nvtelemetry.SamplingCoverage{Status: "measured", Basis: "time", Scope: "activity-union/eligible-window-v1", Numerator: numerator, Denominator: denominator, Unit: "ns"}
}

func TestV4ExactCoverageRoundtrip(t *testing.T) {
	zero, span := uint64(0), uint64(100)
	coverage := measuredTimeCoverage(&nvtelemetry.Number{Uint: &zero}, &nvtelemetry.Number{Uint: &span})
	raw := writeV4(t, v4Reading(coverage))
	m, _, samples, err := nvtelemetry.ReadStreamStrict(bytes.NewReader(raw), 0)
	if err != nil {
		t.Fatal(err)
	}
	if m.SchemaVersion != nvtelemetry.SchemaVersionV4 || len(samples) != 1 {
		t.Fatalf("schema/sample: %d/%d", m.SchemaVersion, len(samples))
	}
	if loss, ok := m.LostRecordsByFamily["activity"]; !ok || loss.Total != 1 || loss.Dropped != 0 || loss.Unknown {
		t.Fatalf("V4 family loss presence changed: %+v", m.LostRecordsByFamily)
	}
	got := samples[0].Readings[0].Observation.Observer.SamplingCoverage
	if !reflect.DeepEqual(got, coverage) || got.Numerator.Uint == nil || *got.Numerator.Uint != 0 {
		t.Fatalf("coverage changed: got %+v want %+v", got, coverage)
	}
	if !bytes.Contains(raw, []byte(`"numerator":{"uint":0}`)) {
		t.Fatalf("measured zero omitted: %s", raw)
	}
}

func TestV4CoverageStatesStayDistinct(t *testing.T) {
	for _, coverage := range []*nvtelemetry.SamplingCoverage{
		{Status: "unknown", Basis: "pc_sample", Scope: "mode-x-pass-y"},
		{Status: "unavailable", Basis: "time", Scope: "window-1", Reason: "clock_discontinuity"},
		{Status: "not_applicable", Basis: "event", Scope: "mode-does-not-count-events"},
	} {
		raw := writeV4(t, v4Reading(coverage))
		_, _, samples, err := nvtelemetry.ReadStream(bytes.NewReader(raw), 0)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(samples[0].Readings[0].Observation.Observer.SamplingCoverage, coverage) {
			t.Fatalf("state changed: got %+v want %+v", samples[0].Readings[0].Observation.Observer.SamplingCoverage, coverage)
		}
	}
}

func TestV4CoverageRejectsMalformedOperands(t *testing.T) {
	zero, one, two := uint64(0), uint64(1), uint64(2)
	negative := int64(-1)
	floating := float64(1)
	cases := []struct {
		name string
		c    *nvtelemetry.SamplingCoverage
	}{
		{"missing-descriptor", nil},
		{"float", &nvtelemetry.SamplingCoverage{Status: "measured", Basis: "event", Scope: "eligible-events-v1", Unit: "events", Numerator: &nvtelemetry.Number{Float: &floating}, Denominator: &nvtelemetry.Number{Uint: &one}}},
		{"negative", &nvtelemetry.SamplingCoverage{Status: "measured", Basis: "event", Scope: "eligible-events-v1", Unit: "events", Numerator: &nvtelemetry.Number{Int: &negative}, Denominator: &nvtelemetry.Number{Uint: &one}}},
		{"zero-denominator", &nvtelemetry.SamplingCoverage{Status: "measured", Basis: "event", Scope: "eligible-events-v1", Unit: "events", Numerator: &nvtelemetry.Number{Uint: &zero}, Denominator: &nvtelemetry.Number{Uint: &zero}}},
		{"numerator-over-denominator", &nvtelemetry.SamplingCoverage{Status: "measured", Basis: "event", Scope: "eligible-events-v1", Unit: "events", Numerator: &nvtelemetry.Number{Uint: &two}, Denominator: &nvtelemetry.Number{Uint: &one}}},
		{"time-denominator-mismatch", measuredTimeCoverage(&nvtelemetry.Number{Uint: &zero}, &nvtelemetry.Number{Uint: &two})},
		{"unavailable-no-reason", &nvtelemetry.SamplingCoverage{Status: "unavailable", Basis: "event"}},
		{"unknown-has-number", &nvtelemetry.SamplingCoverage{Status: "unknown", Numerator: &nvtelemetry.Number{Uint: &one}}},
		{"unknown-status", &nvtelemetry.SamplingCoverage{Status: "future"}},
		{"missing-measured-identity", &nvtelemetry.SamplingCoverage{Status: "measured", Basis: "event", Scope: "eligible-events-v1", Unit: "events", Numerator: &nvtelemetry.Number{Uint: &zero}, Denominator: &nvtelemetry.Number{Uint: &one}}},
		{"time-no-interval", measuredTimeCoverage(&nvtelemetry.Number{Uint: &zero}, &nvtelemetry.Number{Uint: &one})},
		{"oversized-scope", &nvtelemetry.SamplingCoverage{Status: "measured", Basis: "event", Scope: strings.Repeat("x", 4097), Unit: "events", Numerator: &nvtelemetry.Number{Uint: &zero}, Denominator: &nvtelemetry.Number{Uint: &one}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := v4Reading(tc.c)
			if tc.name == "missing-measured-identity" {
				r.Observation.Observer.ID = ""
			}
			if tc.name == "time-no-interval" {
				r.Observation.Interval = nil
			}
			w := nvtelemetry.NewWriter(&bytes.Buffer{}, nvtelemetry.CollectorConfig{})
			m := v4Header()
			if err := w.WriteManifest(&m); err != nil {
				t.Fatal(err)
			}
			if err := w.WriteSample(&nvtelemetry.Sample{Seq: 1, Readings: []nvtelemetry.Reading{r}}); !errors.Is(err, nvtelemetry.ErrMalformedStream) {
				t.Fatalf("got %v, want malformed stream", err)
			}
		})
	}
}

func TestV4OlderVersionsRejectCoveragePresence(t *testing.T) {
	for _, token := range []string{`null`, `{}`, `{"status":"unknown"}`} {
		for _, version := range []int{1, 2, 3} {
			t.Run(strings.Join([]string{string(rune('0' + version)), token}, "/"), func(t *testing.T) {
				raw := `{"manifest":{"schema_version":` + string(rune('0'+version)) + `,"devices":[{"uuid":"GPU-v4"}]}}` + "\n" +
					`{"sample":{"seq":1,"readings":[{"device_uuid":"GPU-v4","evidence":"measured","number":{"float":1},"observation":{"observer":{"sampling_coverage":` + token + `}}}]}}`
				_, _, _, err := nvtelemetry.ReadStream(strings.NewReader(raw), 0)
				if !errors.Is(err, nvtelemetry.ErrMalformedStream) {
					t.Fatalf("version %d accepted V4 key %s: %v", version, token, err)
				}
			})
		}
	}
}

func TestV4BoundedExactUint64(t *testing.T) {
	max := uint64(math.MaxUint64)
	c := &nvtelemetry.SamplingCoverage{Status: "measured", Basis: "event", Scope: "eligible-events-v1", Unit: "events", Numerator: &nvtelemetry.Number{Uint: &max}, Denominator: &nvtelemetry.Number{Uint: &max}}
	raw := writeV4(t, v4Reading(c))
	_, _, samples, err := nvtelemetry.ReadStream(bytes.NewReader(raw), 0)
	if err != nil {
		t.Fatal(err)
	}
	got := samples[0].Readings[0].Observation.Observer.SamplingCoverage
	if got.Numerator.Uint == nil || *got.Numerator.Uint != max || got.Denominator.Uint == nil || *got.Denominator.Uint != max {
		t.Fatalf("uint64 precision lost: %+v", got)
	}
	signedMax := int64(math.MaxInt64)
	c = &nvtelemetry.SamplingCoverage{Status: "measured", Basis: "event", Scope: "eligible-events-v1", Unit: "events", Numerator: &nvtelemetry.Number{Int: &signedMax}, Denominator: &nvtelemetry.Number{Int: &signedMax}}
	raw = writeV4(t, v4Reading(c))
	_, _, samples, err = nvtelemetry.ReadStream(bytes.NewReader(raw), 0)
	if err != nil {
		t.Fatal(err)
	}
	got = samples[0].Readings[0].Observation.Observer.SamplingCoverage
	if got.Numerator.Int == nil || *got.Numerator.Int != signedMax || got.Denominator.Int == nil || *got.Denominator.Int != signedMax {
		t.Fatalf("int64 precision lost: %+v", got)
	}
}

func TestV4FamilyLossPresenceAndStates(t *testing.T) {
	raw := writeV4(t, v4Reading(&nvtelemetry.SamplingCoverage{Status: "unknown"}))
	rewrite := func(t *testing.T, edit func(*nvtelemetry.Manifest)) []byte {
		t.Helper()
		lines := bytes.Split(bytes.TrimSpace(raw), []byte("\n"))
		var out bytes.Buffer
		for i, line := range lines {
			var record nvtelemetry.StreamRecord
			if err := json.Unmarshal(line, &record); err != nil {
				t.Fatal(err)
			}
			if i == len(lines)-1 {
				edit(record.Manifest)
			}
			if err := json.NewEncoder(&out).Encode(record); err != nil {
				t.Fatal(err)
			}
		}
		return out.Bytes()
	}
	for _, tc := range []struct {
		name string
		edit func(*nvtelemetry.Manifest)
		want func(map[string]tracecontract.Loss) bool
	}{
		{"absent-is-unassessed", func(m *nvtelemetry.Manifest) { delete(m.LostRecordsByFamily, "activity") }, func(loss map[string]tracecontract.Loss) bool { _, ok := loss["activity"]; return !ok }},
		{"dropped-preserved", func(m *nvtelemetry.Manifest) {
			loss := m.LostRecordsByFamily["activity"]
			loss.Total, loss.Dropped = 2, 1
			m.LostRecordsByFamily["activity"] = loss
		}, func(loss map[string]tracecontract.Loss) bool {
			v, ok := loss["activity"]
			return ok && v.Total == 2 && v.Dropped == 1 && !v.Unknown
		}},
		{"unknown-preserved", func(m *nvtelemetry.Manifest) {
			loss := m.LostRecordsByFamily["activity"]
			loss.Unknown = true
			m.LostRecordsByFamily["activity"] = loss
		}, func(loss map[string]tracecontract.Loss) bool { v, ok := loss["activity"]; return ok && v.Unknown }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mutated := rewrite(t, tc.edit)
			m, _, _, err := nvtelemetry.ReadStream(bytes.NewReader(mutated), 0)
			if err != nil {
				t.Fatal(err)
			}
			if !tc.want(m.LostRecordsByFamily) {
				t.Fatalf("loss state changed: %+v", m.LostRecordsByFamily)
			}
		})
	}
}

func TestV4CoverageExactKeysAndLegacyDefaults(t *testing.T) {
	for _, raw := range []string{
		`{"manifest":{"schema_version":4,"devices":[{"uuid":"GPU-v4"}],"unknown_contract":{}}}`,
		`{"manifest":{"schema_version":4,"devices":[{"uuid":"GPU-v4"}]}}` + "\n" + `{"sample":{"seq":1,"readings":[{"device_uuid":"GPU-v4","evidence":"measured","number":{"float":1},"observation":{"observer":{"sampling_coverage":{"Status":"unknown"}}}}]}}`,
		`{"manifest":{"schema_version":4,"devices":[{"uuid":"GPU-v4"}],"schema_version":4}}`,
		`{"manifest":{"schema_version":4,"devices":[{"uuid":"GPU-v4"}]}}` + "\n" + `{"sample":{"seq":1,"readings":[{"device_uuid":"GPU-v4","evidence":"measured","number":{"float":1},"observation":{"observer":{"sampling_coverage":{"status":"unknown","status":"unavailable","reason":"unknown"}}}}]}}`,
	} {
		_, _, _, err := nvtelemetry.ReadStream(strings.NewReader(raw), 0)
		if !errors.Is(err, nvtelemetry.ErrMalformedStream) {
			t.Fatalf("strict V4 keys accepted: %v", err)
		}
	}
	var b bytes.Buffer
	w := nvtelemetry.NewWriter(&b, nvtelemetry.CollectorConfig{})
	m := v4Header()
	m.SchemaVersion = nvtelemetry.SchemaVersionV2
	if err := w.WriteManifest(&m); err != nil {
		t.Fatal(err)
	}
	var header nvtelemetry.StreamRecord
	if err := json.Unmarshal(bytes.Split(b.Bytes(), []byte("\n"))[0], &header); err != nil {
		t.Fatal(err)
	}
	if header.Manifest.SchemaVersion != nvtelemetry.SchemaVersionV2 {
		t.Fatalf("default/version changed: %d", header.Manifest.SchemaVersion)
	}
}

func TestV3WriterRefusesV4CoverageWithoutSideEffects(t *testing.T) {
	var b bytes.Buffer
	w := nvtelemetry.NewWriter(&b, nvtelemetry.CollectorConfig{MaxRecordBudget: 1})
	m := v3Header()
	if err := w.WriteManifest(&m); err != nil {
		t.Fatal(err)
	}
	before := append([]byte(nil), b.Bytes()...)
	bytesBefore := w.BytesWritten()
	lostBefore := w.LostRecords()
	familyLossBefore := w.FamilyLoss()

	coverage := &nvtelemetry.SamplingCoverage{Status: "unknown"}
	reading := v3NumberReading(&nvtelemetry.Number{Float: new(float64)})
	reading.Observation = &nvtelemetry.Observation{
		Family:   "activity",
		Observer: nvtelemetry.Observer{SamplingCoverage: coverage},
	}
	err := w.WriteSample(&nvtelemetry.Sample{Seq: 1, Readings: []nvtelemetry.Reading{reading}})
	if !errors.Is(err, nvtelemetry.ErrMalformedStream) {
		t.Fatalf("V3 writer accepted V4 coverage: %v", err)
	}
	if !bytes.Equal(b.Bytes(), before) || w.BytesWritten() != bytesBefore {
		t.Fatal("rejected V4 coverage changed stream bytes")
	}
	if w.LostRecords() != lostBefore || !reflect.DeepEqual(w.FamilyLoss(), familyLossBefore) {
		t.Fatalf("rejected V4 coverage changed accounting: lost=%d family=%v", w.LostRecords(), w.FamilyLoss())
	}

	if err := w.WriteSample(&nvtelemetry.Sample{Seq: 2, Readings: []nvtelemetry.Reading{v3NumberReading(&nvtelemetry.Number{Float: new(float64)})}}); err != nil {
		t.Fatalf("rejected sample consumed the record budget: %v", err)
	}
}

type v4CoverageSampler struct{ readings []nvtelemetry.Reading }

func (s *v4CoverageSampler) Devices() []nvtelemetry.DeviceMetadata {
	return []nvtelemetry.DeviceMetadata{{UUID: "GPU-v4"}}
}

func (s *v4CoverageSampler) Sample(context.Context) ([]nvtelemetry.Reading, error) {
	return s.readings, nil
}

func TestV4CollectorCoverageCopyOwnership(t *testing.T) {
	numerator, denominator := uint64(0), uint64(100)
	coverage := measuredTimeCoverage(&nvtelemetry.Number{Uint: &numerator}, &nvtelemetry.Number{Uint: &denominator})
	sampler := &v4CoverageSampler{readings: []nvtelemetry.Reading{v4Reading(coverage)}}
	cfg := nvtelemetry.CollectorConfig{SchemaVersion: nvtelemetry.SchemaVersionV4}
	collector := nvtelemetry.NewCollector("v4-copy", sampler, nvtelemetry.NewFakeClock(1, 1), cfg, nil)
	collector.SetSampleHook(func(sample nvtelemetry.Sample) {
		reading := &sample.Readings[0]
		descriptor := reading.Observation.Observer.SamplingCoverage
		*descriptor.Numerator.Uint = 99
		descriptor.Scope = "hook-mutated"
		reading.Observation.Interval.Start = 20
		reading.Observation.Interval.End = 777
		reading.Observation.Interval.Clock = "hook-mutated"
	})
	if err := collector.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	if *sampler.readings[0].Observation.Observer.SamplingCoverage.Numerator.Uint != 0 || sampler.readings[0].Observation.Observer.SamplingCoverage.Scope != "activity-union/eligible-window-v1" {
		t.Fatal("collector or hook mutated sampler evidence")
	}
	// Mutating producer-owned state after Step must not change retained samples.
	*sampler.readings[0].Observation.Observer.SamplingCoverage.Numerator.Uint = 7
	sampler.readings[0].Observation.Observer.SamplingCoverage.Scope = "producer-mutated"
	sampler.readings[0].Observation.Interval.End = 999
	first := collector.Samples()
	reading := &first[0].Readings[0]
	descriptor := reading.Observation.Observer.SamplingCoverage
	if *descriptor.Numerator.Uint != 0 || descriptor.Scope != "activity-union/eligible-window-v1" || reading.Observation.Interval.Start != 10 || reading.Observation.Interval.End != 110 || reading.Observation.Interval.Clock != "mono" {
		t.Fatalf("retained sample aliased sampler or hook: reading=%+v", reading)
	}
	*descriptor.Numerator.Uint = 88
	descriptor.Scope = "caller-mutated"
	reading.Observation.Interval.Start = 20
	reading.Observation.Interval.End = 777
	reading.Observation.Interval.Clock = "caller-mutated"
	secondReading := &collector.Samples()[0].Readings[0]
	second := secondReading.Observation.Observer.SamplingCoverage
	if *second.Numerator.Uint != 0 || *second.Denominator.Uint != 100 || second.Scope != "activity-union/eligible-window-v1" {
		t.Fatalf("Samples exposed retained descriptor: %+v", second)
	}
	interval := secondReading.Observation.Interval
	if interval.Start != 10 || interval.End != 110 || interval.Clock != "mono" || uint64(interval.End-interval.Start) != *second.Denominator.Uint {
		t.Fatalf("Samples exposed retained time window: interval=%+v coverage=%+v", interval, second)
	}
}
