package nvtelemetry_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	n "github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func v5Uint(v uint64) *uint64    { return &v }
func v5Float(v float64) *float64 { return &v }
func v5String(v string) *string  { return &v }

func v5Reading() n.Reading {
	return n.Reading{Name: "latency", Unit: n.UnitSeconds, DeviceUUID: "GPU-v5", Evidence: n.EvidenceMeasured, TimestampUnix: 20,
		Observation: &n.Observation{Family: "histograms", SourceClass: "imported", ClockEpoch: "unix", Kind: "histogram",
			Observer:      n.Observer{SamplingCoverage: &n.SamplingCoverage{Status: "unknown"}},
			OTLPScopeName: v5String("github.com/tmc/lib/nvidia/x/nvtelemetry"), OTLPClock: &n.OTLPClockMetadata{},
			Histogram: &n.Histogram{Bounds: []string{"1", "+Inf"}, Counts: []uint64{2, 3}, Count: v5Uint(3), Sum: v5Float(4), Min: v5Float(1), Max: v5Float(2), Temporality: "delta"}}}
}

func v5Stream(t *testing.T, version int, r n.Reading) ([]byte, error) {
	t.Helper()
	var b bytes.Buffer
	w := n.NewWriter(&b, n.CollectorConfig{})
	if err := w.WriteManifest(&n.Manifest{SchemaVersion: version, Devices: []n.DeviceMetadata{{UUID: "GPU-v5"}}}); err != nil {
		return nil, err
	}
	err := w.WriteSample(&n.Sample{Seq: 1, QueryStartNanos: 1, QueryEndNanos: 2, Readings: []n.Reading{r}})
	return b.Bytes(), err
}

func TestV5StreamHistogram(t *testing.T) {
	for _, name := range []string{"distribution", "bucketless", "terminal-only", "known-zero", "no-recorded"} {
		t.Run(name, func(t *testing.T) {
			r := v5Reading()
			h := r.Observation.Histogram
			switch name {
			case "bucketless":
				h.Bounds, h.Counts = []string{}, []uint64{}
			case "terminal-only":
				h.Bounds, h.Counts = []string{"+Inf"}, []uint64{3}
			case "known-zero":
				h.Bounds, h.Counts = []string{}, []uint64{}
				h.Count = v5Uint(0)
				h.Sum = v5Float(0)
				h.Min, h.Max = nil, nil
			case "no-recorded":
				h.Bounds, h.Counts = []string{}, []uint64{}
				h.Count = v5Uint(0)
				h.Sum, h.Min, h.Max = nil, nil, nil
				h.Flags = 1
				r.Evidence = n.EvidenceUnavailable
				r.UnavailableReason = "no_recorded_value"
			}
			want, _ := json.Marshal(r)
			raw, err := v5Stream(t, n.SchemaVersionV5, r)
			if err != nil {
				t.Fatal(err)
			}
			m, _, samples, err := n.ReadStream(bytes.NewReader(raw), 0)
			if err != nil {
				t.Fatal(err)
			}
			got, _ := json.Marshal(samples[0].Readings[0])
			if m.SchemaVersion != n.SchemaVersionV5 || !bytes.Equal(got, want) {
				t.Fatalf("round trip: %s != %s", got, want)
			}
		})
	}
}

func TestV5OldWriterAtomicRefusal(t *testing.T) {
	mutations := []struct {
		name   string
		mutate func(*n.Observation)
	}{
		{"count", func(o *n.Observation) { o.Histogram.Count = v5Uint(0) }},
		{"sum", func(o *n.Observation) { o.Histogram.Sum = v5Float(0) }},
		{"min", func(o *n.Observation) { o.Histogram.Min = v5Float(0) }},
		{"max", func(o *n.Observation) { o.Histogram.Max = v5Float(0) }},
		{"temporality", func(o *n.Observation) { o.Histogram.Temporality = "delta" }},
		{"flags", func(o *n.Observation) { o.Histogram.Flags = 1 }},
		{"scope", func(o *n.Observation) { o.OTLPScopeName = v5String("") }},
		{"clock", func(o *n.Observation) { o.OTLPClock = &n.OTLPClockMetadata{} }},
		{"source-time", func(o *n.Observation) { o.OTLPSourceTime = &n.OTLPSourceTime{} }},
	}
	for version := 1; version <= 4; version++ {
		for _, test := range mutations {
			t.Run(fmt.Sprintf("V%d/%s", version, test.name), func(t *testing.T) {
				var b bytes.Buffer
				w := n.NewWriter(&b, n.CollectorConfig{MaxRecordBudget: 1, DropOnFull: true})
				if err := w.WriteManifest(&n.Manifest{SchemaVersion: version, Devices: []n.DeviceMetadata{{UUID: "GPU-v5"}}}); err != nil {
					t.Fatal(err)
				}
				o := &n.Observation{Kind: "histogram", Histogram: &n.Histogram{Bounds: []string{"1"}, Counts: []uint64{0}}}
				test.mutate(o)
				r := n.Reading{Name: "latency", Unit: n.UnitSeconds, DeviceUUID: "GPU-v5", Evidence: n.EvidenceMeasured, Observation: o}
				if version == 4 {
					o.Observer.SamplingCoverage = &n.SamplingCoverage{Status: "unknown"}
				}
				before := append([]byte(nil), b.Bytes()...)
				families := w.FamilyLoss()
				lost := w.LostRecords()
				size := w.BytesWritten()
				if err := w.WriteSample(&n.Sample{Seq: 1, Readings: []n.Reading{r}}); !errors.Is(err, n.ErrMalformedStream) {
					t.Fatalf("missing V5 typed refusal: %v", err)
				}
				if !bytes.Equal(b.Bytes(), before) || size != w.BytesWritten() || lost != w.LostRecords() || !reflect.DeepEqual(families, w.FamilyLoss()) {
					t.Fatal("refusal changed bytes/accounting")
				}
				valid := n.Reading{Name: n.MetricGPUUtilization, Unit: n.UnitPercent, DeviceUUID: "GPU-v5", Evidence: n.EvidenceMeasured}
				if version >= 3 {
					valid.Number = &n.Number{Uint: v5Uint(0)}
				}
				if err := w.WriteSample(&n.Sample{Seq: 2, Readings: []n.Reading{valid}}); err != nil {
					t.Fatalf("refusal consumed budget: %v", err)
				}
				if w.LostRecords() != lost {
					t.Fatal("refusal consumed remaining budget")
				}
			})
		}
	}
}

func TestV5OldRawPresenceRefusal(t *testing.T) {
	for version := 1; version <= 4; version++ {
		valid := n.Reading{Name: n.MetricGPUUtilization, Unit: n.UnitPercent, DeviceUUID: "GPU-v5", Evidence: n.EvidenceMeasured}
		if version >= 3 {
			valid.Number = &n.Number{Uint: v5Uint(0)}
		}
		raw, err := v5Stream(t, version, valid)
		if err != nil {
			t.Fatal(err)
		}
		if _, _, _, err := n.ReadStream(bytes.NewReader(raw), 0); err != nil {
			t.Fatalf("invalid old baseline V%d: %v", version, err)
		}
		lines := bytes.Split(bytes.TrimSpace(raw), []byte("\n"))
		for _, key := range []string{"count", "sum", "min", "max", "temporality", "flags", "otlp_scope_name", "otlp_clock_metadata", "otlp_source_time"} {
			for _, value := range []string{"null", "0", "\"\""} {
				t.Run(fmt.Sprintf("V%d/%s/%s", version, key, value), func(t *testing.T) {
					var envelope map[string]any
					json.Unmarshal(lines[1], &envelope)
					r := envelope["sample"].(map[string]any)["readings"].([]any)[0].(map[string]any)
					var v any
					json.Unmarshal([]byte(value), &v)
					o := map[string]any{}
					if strings.HasPrefix(key, "otlp_") {
						o[key] = v
					} else {
						o["histogram"] = map[string]any{key: v}
					}
					r["observation"] = o
					bad, _ := json.Marshal(envelope)
					bad = append(append(append([]byte(nil), lines[0]...), '\n'), append(bad, '\n')...)
					if _, _, _, err := n.ReadStream(bytes.NewReader(bad), 0); err == nil {
						t.Fatal("old raw V5 key accepted")
					}
				})
			}
		}
	}
}

func TestV5HistogramRefusals(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*n.Reading)
	}{
		{"missing-count", func(r *n.Reading) { r.Observation.Histogram.Count = nil }},
		{"decrease", func(r *n.Reading) { r.Observation.Histogram.Counts = []uint64{4, 3} }},
		{"total", func(r *n.Reading) { *r.Observation.Histogram.Count = 4 }},
		{"terminal", func(r *n.Reading) { r.Observation.Histogram.Bounds[1] = "2" }},
		{"flags", func(r *n.Reading) { r.Observation.Histogram.Flags = 2 }},
		{"recorded-flags", func(r *n.Reading) { r.Observation.Histogram.Flags = 1 }},
		{"summary", func(r *n.Reading) { *r.Observation.Histogram.Min = 3 }},
		{"zero-summary", func(r *n.Reading) {
			r.Observation.Histogram.Count = v5Uint(0)
			r.Observation.Histogram.Counts = []uint64{0, 0}
		}},
		{"nonfinite", func(r *n.Reading) { r.Observation.Histogram.Sum = v5Float(math.Inf(1)) }},
		{"underflow", func(r *n.Reading) { r.Observation.Histogram.Bounds[0] = "1e-500" }},
		{"rational", func(r *n.Reading) { r.Observation.Histogram.Bounds[0] = "1/2" }},
		{"nil-clock", func(r *n.Reading) { r.Observation.OTLPClock = nil }},
		{"nil-scope", func(r *n.Reading) { r.Observation.OTLPScopeName = nil }},
		{"missing-coverage", func(r *n.Reading) { r.Observation.Observer.SamplingCoverage = nil }},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			r := v5Reading()
			test.mutate(&r)
			if _, err := v5Stream(t, 5, r); err == nil {
				t.Fatal("invalid V5 reading accepted")
			}
		})
	}
}

func TestV5ExplicitMaximumStream(t *testing.T) {
	r := v5Reading()
	h := r.Observation.Histogram
	h.Bounds = nil
	h.Counts = nil
	for i := 0; i < 4096; i++ {
		h.Bounds = append(h.Bounds, fmt.Sprint(i))
		h.Counts = append(h.Counts, 3)
	}
	h.Bounds = append(h.Bounds, "+Inf")
	h.Counts = append(h.Counts, 3)
	raw, err := v5Stream(t, 5, r)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := n.ReadStream(bytes.NewReader(raw), 0); err != nil {
		t.Fatal(err)
	}
	old := v5Reading()
	old.Observation.SourceClass = "synthetic"
	old.Observation.OTLPClock = nil
	old.Observation.OTLPScopeName = nil
	old.Observation.Histogram = h
	h.Count, h.Sum, h.Min, h.Max = nil, nil, nil, nil
	h.Temporality = ""
	if _, err := v5Stream(t, 4, old); err == nil {
		t.Fatal("V5 maximum leaked into V4")
	}
}

type v5Sampler struct{ reading n.Reading }

func (s *v5Sampler) Devices() []n.DeviceMetadata                 { return []n.DeviceMetadata{{UUID: "GPU-v5"}} }
func (s *v5Sampler) Sample(context.Context) ([]n.Reading, error) { return []n.Reading{s.reading}, nil }

func TestV5CollectorCopyOwnership(t *testing.T) {
	r := v5Reading()
	r.Observation.Interval = &tracecontract.Interval{Clock: "unix", Start: 15, End: 20}
	r.Observation.OTLPSourceTime = &n.OTLPSourceTime{SourceClass: "synthetic", SourceClockEpoch: "mono-v1", SourceMonotonicEndNanos: 10, SourceIntervalPresent: true, SourceIntervalClock: "mono", SourceIntervalStartNanos: 5, SourceIntervalEndNanos: 10, MappingDomain: "mono", MappingEpoch: "mono-v1", MappingValidFromNanos: 0, MappingValidUntilNanos: 100, MappingUnixOffsetNanos: 10, MappingProvenance: "test"}
	want, _ := json.Marshal(r.Observation)
	mutations := []func(*n.Observation){
		func(o *n.Observation) { *o.Histogram.Count = 101 }, func(o *n.Observation) { *o.Histogram.Sum = 102 }, func(o *n.Observation) { *o.Histogram.Min = 103 }, func(o *n.Observation) { *o.Histogram.Max = 104 },
		func(o *n.Observation) { o.Histogram.Bounds[0] = "105" }, func(o *n.Observation) { o.Histogram.Counts[0] = 106 }, func(o *n.Observation) { *o.OTLPScopeName = "107" }, func(o *n.Observation) { o.OTLPClock.ClockDomain = "108" }, func(o *n.Observation) { o.OTLPSourceTime.MappingProvenance = "109" }, func(o *n.Observation) { o.Interval.End = 110 }, func(o *n.Observation) { o.Observer.SamplingCoverage.Reason = "111" },
	}
	s := &v5Sampler{r}
	c := n.NewCollector("v5-copy", s, n.NewFakeClock(1, 1), n.CollectorConfig{SchemaVersion: 5}, nil)
	assert := func(o *n.Observation) {
		t.Helper()
		got, _ := json.Marshal(o)
		if !bytes.Equal(got, want) {
			t.Fatalf("aliased observation: %s != %s", got, want)
		}
	}
	c.SetSampleHook(func(sample n.Sample) {
		for _, m := range mutations {
			m(sample.Readings[0].Observation)
			assert(s.reading.Observation)
		}
	})
	if err := c.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	assert(c.Samples()[0].Readings[0].Observation)
	for _, m := range mutations {
		m(s.reading.Observation)
		assert(c.Samples()[0].Readings[0].Observation)
	}
	for _, m := range mutations {
		snap := c.Samples()
		m(snap[0].Readings[0].Observation)
		assert(c.Samples()[0].Readings[0].Observation)
	}
}

func TestV5EvidenceAtomicRefusal(t *testing.T) {
	for _, evidence := range []string{"", "unknown", "invented"} {
		t.Run(fmt.Sprintf("%q", evidence), func(t *testing.T) {
			var output bytes.Buffer
			w := n.NewWriter(&output, n.CollectorConfig{MaxRecordBudget: 1, DropOnFull: true})
			if err := w.WriteManifest(&n.Manifest{SchemaVersion: 5, Devices: []n.DeviceMetadata{{UUID: "GPU-v5"}}}); err != nil {
				t.Fatal(err)
			}
			before := append([]byte(nil), output.Bytes()...)
			size, lost, families := w.BytesWritten(), w.LostRecords(), w.FamilyLoss()
			r := v5Reading()
			r.Evidence = evidence
			if err := w.WriteSample(&n.Sample{Seq: 1, Readings: []n.Reading{r}}); !errors.Is(err, n.ErrMalformedStream) {
				t.Fatalf("invalid V5 evidence accepted: %q: %v", evidence, err)
			}
			if !bytes.Equal(before, output.Bytes()) || w.BytesWritten() != size || w.LostRecords() != lost || !reflect.DeepEqual(families, w.FamilyLoss()) {
				t.Fatal("refusal changed bytes/accounting")
			}
			if err := w.WriteSample(&n.Sample{Seq: 2, Readings: []n.Reading{v5Reading()}}); err != nil || w.LostRecords() != lost {
				t.Fatalf("refusal consumed budget: %v", err)
			}
		})
	}
}

func TestV5RequiredMetadataMembers(t *testing.T) {
	r := v5Reading()
	r.TimestampUnix = 1100
	r.Observation.OTLPSourceTime = &n.OTLPSourceTime{SourceClass: "synthetic", SourceClockEpoch: "mono-v1", SourceMonotonicEndNanos: 100,
		MappingDomain: "mono", MappingEpoch: "mono-v1", MappingValidUntilNanos: 200, MappingUnixOffsetNanos: 1000, MappingProvenance: "caller-assertion"}
	raw, err := v5Stream(t, 5, r)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := n.ReadStream(bytes.NewReader(raw), 0); err != nil {
		t.Fatalf("explicit false/zero baseline: %v", err)
	}
	lines := bytes.Split(bytes.TrimSpace(raw), []byte("\n"))
	for _, object := range []string{"otlp_clock_metadata", "otlp_source_time"} {
		var sample map[string]any
		if err := json.Unmarshal(lines[1], &sample); err != nil {
			t.Fatal(err)
		}
		obs := sample["sample"].(map[string]any)["readings"].([]any)[0].(map[string]any)["observation"].(map[string]any)
		for key := range obs[object].(map[string]any) {
			for _, mode := range []string{"missing", "null", "alias"} {
				t.Run(object+"/"+key+"/"+mode, func(t *testing.T) {
					var envelope map[string]any
					if err := json.Unmarshal(lines[1], &envelope); err != nil {
						t.Fatal(err)
					}
					o := envelope["sample"].(map[string]any)["readings"].([]any)[0].(map[string]any)["observation"].(map[string]any)[object].(map[string]any)
					switch mode {
					case "missing":
						delete(o, key)
					case "null":
						o[key] = nil
					case "alias":
						o[strings.ToUpper(key)] = o[key]
						delete(o, key)
					}
					changed, _ := json.Marshal(envelope)
					bad := append(append(append([]byte(nil), lines[0]...), '\n'), changed...)
					bad = append(bad, '\n')
					if _, _, _, err := n.ReadStream(bytes.NewReader(bad), 0); !errors.Is(err, n.ErrMalformedStream) {
						t.Fatalf("required V5 member accepted: %v", err)
					}
				})
			}
		}
	}
}
