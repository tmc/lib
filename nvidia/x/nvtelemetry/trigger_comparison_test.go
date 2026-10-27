package nvtelemetry_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func TestTriggerComparisonOperators(t *testing.T) {
	for _, typed := range []bool{false, true} {
		representation := "legacy"
		if typed {
			representation = "typed"
		}
		for _, tt := range []struct {
			name, op string
			want     [3]bool
		}{
			{"default", "", [3]bool{false, false, true}},
			{"greater", ">", [3]bool{false, false, true}},
			{"greater_equal", ">=", [3]bool{false, true, true}},
			{"less", "<", [3]bool{true, false, false}},
			{"less_equal", "<=", [3]bool{true, true, false}},
			{"unknown", "disabled", [3]bool{}},
			{"typo", ">>", [3]bool{}},
			{"equality", "==", [3]bool{}},
			{"leading_space", " >", [3]bool{}},
			{"trailing_space", "> ", [3]bool{}},
		} {
			for i, value := range []float64{9, 10, 11} {
				t.Run(fmt.Sprintf("%s/%s/%g", representation, tt.name, value), func(t *testing.T) {
					r := nvtelemetry.Reading{Name: "trigger-control", Evidence: nvtelemetry.EvidenceMeasured, Value: value}
					if typed {
						r.Value = 0
						r.Number = &nvtelemetry.Number{Float: &value}
					}
					p := nvtelemetry.TriggerPolicy{MetricName: r.Name, Threshold: 10, Comparison: tt.op}
					if got := p.Matches(r); got != tt.want[i] {
						t.Fatalf("Matches(%q, %g)=%v, want %v", tt.op, value, got, tt.want[i])
					}
				})
			}
		}
	}
}

func TestEvaluateSampleSkipsUnknownComparison(t *testing.T) {
	sample := nvtelemetry.Sample{Readings: []nvtelemetry.Reading{{Name: "trigger-control", Evidence: nvtelemetry.EvidenceMeasured, Value: 11}}}
	invalid := nvtelemetry.TriggerPolicy{MetricName: "trigger-control", Threshold: 10, Comparison: ">>", Reason: "unsupported-control"}
	if _, _, ok := nvtelemetry.EvaluateSample(sample, []nvtelemetry.TriggerPolicy{invalid}); ok {
		t.Fatal("unsupported operator produced a positive trigger")
	}
	valid := invalid
	valid.Comparison = ">"
	valid.Reason = "configured-control"
	r, p, ok := nvtelemetry.EvaluateSample(sample, []nvtelemetry.TriggerPolicy{invalid, valid})
	if !ok || p.Reason != valid.Reason || r.Value != 11 {
		t.Fatalf("valid policy after unsupported policy = %+v, %+v, %v", r, p, ok)
	}
}

func TestCollectorTriggerComparisonContract(t *testing.T) {
	for _, version := range []int{2, 3} {
		t.Run(fmt.Sprintf("v%d", version), func(t *testing.T) {
			sampler := nvtelemetry.NewFakeSampler(nvtelemetry.DeviceMetadata{UUID: "GPU-trigger-control"})
			sampler.Enqueue([]nvtelemetry.Reading{{Name: "trigger-control", Evidence: nvtelemetry.EvidenceMeasured, Value: 11}})
			collector := nvtelemetry.NewCollector("comparison-control", sampler, nvtelemetry.NewFakeClock(1, 1), nvtelemetry.CollectorConfig{SchemaVersion: version}, nil)
			calls := 0
			collector.SetSampleHook(func(s nvtelemetry.Sample) {
				calls++
				for _, op := range []string{">>", "disabled"} {
					p := nvtelemetry.TriggerPolicy{MetricName: "trigger-control", Threshold: 10, Comparison: op}
					if _, _, ok := nvtelemetry.EvaluateSample(s, []nvtelemetry.TriggerPolicy{p}); ok {
						t.Errorf("collector reading produced a positive trigger for unsupported operator %q", op)
					}
				}
				p := nvtelemetry.TriggerPolicy{MetricName: "trigger-control", Threshold: 10}
				if _, _, ok := nvtelemetry.EvaluateSample(s, []nvtelemetry.TriggerPolicy{p}); !ok {
					t.Error("omitted operator no longer defaults to greater-than")
				}
			})
			if err := collector.Step(context.Background()); err != nil {
				t.Fatal(err)
			}
			if calls != 1 {
				t.Fatalf("sample hook calls=%d, want 1", calls)
			}
		})
	}
}

func ExampleTriggerPolicy_Matches() {
	r := nvtelemetry.Reading{Name: "queue_depth", Evidence: nvtelemetry.EvidenceMeasured, Value: 11}
	p := nvtelemetry.TriggerPolicy{MetricName: r.Name, Threshold: 10}
	fmt.Println(p.Matches(r))
	p.Comparison = ">>"
	fmt.Println(p.Matches(r))
	// Output:
	// true
	// false
}
