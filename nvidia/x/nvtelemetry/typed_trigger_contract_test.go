package nvtelemetry_test

import (
	"context"
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func triggerFloat(v float64) *float64 { return &v }
func triggerInt(v int64) *int64       { return &v }
func triggerUint(v uint64) *uint64    { return &v }

func TestTypedTriggerCollectorContract(t *testing.T) {
	for _, tt := range []struct {
		name    string
		version int
		reading nvtelemetry.Reading
		compare string
		limit   float64
		want    bool
	}{
		{"float", 3, nvtelemetry.Reading{Number: &nvtelemetry.Number{Float: triggerFloat(200.5)}}, ">", 100, true},
		{"signed", 3, nvtelemetry.Reading{Number: &nvtelemetry.Number{Int: triggerInt(-20)}}, "<", -10, true},
		{"unsigned", 3, nvtelemetry.Reading{Number: &nvtelemetry.Number{Uint: triggerUint(200)}}, ">=", 200, true},
		{"zero", 3, nvtelemetry.Reading{Number: &nvtelemetry.Number{Float: triggerFloat(0)}}, ">=", 0, true},
		{"typed below", 3, nvtelemetry.Reading{Number: &nvtelemetry.Number{Uint: triggerUint(200)}}, "<", 100, false},
		{"unavailable", 3, nvtelemetry.Reading{UnavailableReason: nvtelemetry.ReasonUnsupported}, "<=", 0, false},
		{"histogram", 3, nvtelemetry.Reading{Observation: &nvtelemetry.Observation{Kind: "histogram", Histogram: &nvtelemetry.Histogram{Bounds: []string{"+Inf"}, Counts: []uint64{1}}}}, "<=", 0, false},
		{"legacy float", 2, nvtelemetry.Reading{Value: 200.5}, ">", 100, true},
		{"legacy integer", 2, nvtelemetry.Reading{IsInt: true, IntValue: 200}, ">=", 200, true},
		{"normalized float", 3, nvtelemetry.Reading{Value: 200.5}, ">", 100, true},
		{"normalized integer", 3, nvtelemetry.Reading{IsInt: true, IntValue: 200}, ">=", 200, true},
		{"rounded integer", 3, nvtelemetry.Reading{Number: &nvtelemetry.Number{Uint: triggerUint(1<<53 + 1)}}, ">", 1 << 53, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := tt.reading
			r.Name, r.Evidence = "test_metric", nvtelemetry.EvidenceMeasured
			sampler := nvtelemetry.NewFakeSampler(nvtelemetry.DeviceMetadata{UUID: "GPU-test"})
			sampler.Enqueue([]nvtelemetry.Reading{r})
			c := nvtelemetry.NewCollector("typed-trigger", sampler, nvtelemetry.NewFakeClock(1, 1), nvtelemetry.CollectorConfig{SchemaVersion: tt.version}, nil)
			calls := 0
			c.SetSampleHook(func(s nvtelemetry.Sample) {
				calls++
				policy := nvtelemetry.TriggerPolicy{MetricName: r.Name, Threshold: tt.limit, Comparison: tt.compare}
				matched, _, ok := nvtelemetry.EvaluateSample(s, []nvtelemetry.TriggerPolicy{policy})
				if ok != tt.want {
					t.Fatalf("EvaluateSample = %v, want %v: %+v", ok, tt.want, s.Readings)
				}
				if tt.version >= 3 && ok && (matched.Number == nil || matched.Value != 0 || matched.IsInt || matched.IntValue != 0) {
					t.Fatalf("hook did not receive normalized Number: %+v", matched)
				}
				if tt.name == "rounded integer" && *s.Readings[0].Number.Uint != 1<<53+1 {
					t.Fatal("threshold rounding changed the exact reading")
				}
			})
			if err := c.Step(context.Background()); err != nil {
				t.Fatal(err)
			}
			if calls != 1 {
				t.Fatalf("sample hook calls = %d", calls)
			}
		})
	}
}

func TestTypedTriggerMalformedScalars(t *testing.T) {
	for _, tt := range []struct {
		name string
		r    nvtelemetry.Reading
	}{
		{"empty number", nvtelemetry.Reading{Number: &nvtelemetry.Number{}}},
		{"two arms", nvtelemetry.Reading{Number: &nvtelemetry.Number{Float: triggerFloat(0), Uint: triggerUint(0)}}},
		{"conflicting legacy", nvtelemetry.Reading{Value: 1, Number: &nvtelemetry.Number{Float: triggerFloat(0)}}},
		{"absent scalar", nvtelemetry.Reading{Observation: &nvtelemetry.Observation{Kind: "gauge"}}},
		{"histogram kind", nvtelemetry.Reading{Number: &nvtelemetry.Number{Float: triggerFloat(0)}, Observation: &nvtelemetry.Observation{Kind: "histogram"}}},
		{"nonfinite legacy", nvtelemetry.Reading{Value: math.Inf(1)}},
		{"nonfinite typed", nvtelemetry.Reading{Number: &nvtelemetry.Number{Float: triggerFloat(math.NaN())}}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := tt.r
			r.Name, r.Evidence = "test_metric", nvtelemetry.EvidenceMeasured
			for _, op := range []string{">", ">=", "<", "<="} {
				p := nvtelemetry.TriggerPolicy{MetricName: r.Name, Comparison: op}
				if p.Matches(r) {
					t.Fatalf("malformed scalar matched %q: %+v", op, r)
				}
			}
		})
	}
}
