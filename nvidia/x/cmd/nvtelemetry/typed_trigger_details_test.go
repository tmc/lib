package main

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func TestTypedTriggerDetails(t *testing.T) {
	f, i, u, zero := 200.5, int64(-20), uint64(1<<53+1), float64(0)
	for _, tt := range []struct {
		name string
		r    nvtelemetry.Reading
		want float64
	}{
		{"float", nvtelemetry.Reading{Number: &nvtelemetry.Number{Float: &f}}, f},
		{"signed", nvtelemetry.Reading{Number: &nvtelemetry.Number{Int: &i}}, float64(i)},
		{"unsigned", nvtelemetry.Reading{Number: &nvtelemetry.Number{Uint: &u}}, float64(u)},
		{"zero", nvtelemetry.Reading{Number: &nvtelemetry.Number{Float: &zero}}, 0},
		{"legacy float", nvtelemetry.Reading{Value: f}, f},
		{"legacy integer", nvtelemetry.Reading{IsInt: true, IntValue: u}, float64(u)},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := tt.r
			r.Name, r.Evidence = "detail_metric", nvtelemetry.EvidenceMeasured
			sampler := nvtelemetry.NewFakeSampler(nvtelemetry.DeviceMetadata{UUID: "GPU-test"})
			sampler.Enqueue([]nvtelemetry.Reading{r})
			c := nvtelemetry.NewCollector("details", sampler, nvtelemetry.NewFakeClock(1, 1), nvtelemetry.CollectorConfig{SchemaVersion: nvtelemetry.SchemaVersionV3}, nil)
			called := false
			c.SetSampleHook(func(s nvtelemetry.Sample) {
				policy := nvtelemetry.TriggerPolicy{MetricName: r.Name, Comparison: ">=", Threshold: tt.want}
				got, _, ok := nvtelemetry.EvaluateSample(s, []nvtelemetry.TriggerPolicy{policy})
				if !ok {
					t.Fatal("typed trigger not admitted")
				}
				called = true
				if value := triggerMetricValue(got); value != tt.want {
					t.Fatalf("CLI trigger detail = %v, want %v", value, tt.want)
				}
			})
			if err := c.Step(t.Context()); err != nil {
				t.Fatal(err)
			}
			if !called {
				t.Fatal("sample hook not called")
			}
		})
	}
}
