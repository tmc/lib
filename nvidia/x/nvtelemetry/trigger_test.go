package nvtelemetry_test

import (
	"context"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func TestTriggerPolicyMatches(t *testing.T) {
	policy := nvtelemetry.TriggerPolicy{
		MetricName: nvtelemetry.MetricGPUUtilization,
		Threshold:  80.0,
		Comparison: ">",
		TenantID:   "tenant-1",
		Reason:     "high_utilization",
	}

	matchReading := nvtelemetry.Reading{
		Name:     nvtelemetry.MetricGPUUtilization,
		Evidence: nvtelemetry.EvidenceMeasured,
		Value:    85.5,
	}
	if !policy.Matches(matchReading) {
		t.Errorf("expected reading value 85.5 to match threshold > 80.0")
	}

	noMatchReading := nvtelemetry.Reading{
		Name:     nvtelemetry.MetricGPUUtilization,
		Evidence: nvtelemetry.EvidenceMeasured,
		Value:    75.0,
	}
	if policy.Matches(noMatchReading) {
		t.Errorf("expected reading value 75.0 NOT to match threshold > 80.0")
	}

	unavailableReading := nvtelemetry.Reading{
		Name:              nvtelemetry.MetricGPUUtilization,
		Evidence:          nvtelemetry.EvidenceUnavailable,
		UnavailableReason: nvtelemetry.ReasonUnsupported,
		Value:             95.0,
	}
	if policy.Matches(unavailableReading) {
		t.Errorf("unavailable reading must never trigger escalation")
	}

	intReading := nvtelemetry.Reading{
		Name:     nvtelemetry.MetricPowerUsageMW,
		Evidence: nvtelemetry.EvidenceMeasured,
		IsInt:    true,
		IntValue: 350000,
	}
	intPolicy := nvtelemetry.TriggerPolicy{
		MetricName: nvtelemetry.MetricPowerUsageMW,
		Threshold:  300000,
		Comparison: ">=",
	}
	if !intPolicy.Matches(intReading) {
		t.Errorf("expected int reading 350000 to match threshold >= 300000")
	}
}

func TestEvaluateSample(t *testing.T) {
	policies := []nvtelemetry.TriggerPolicy{
		{
			MetricName: nvtelemetry.MetricTemperatureC,
			Threshold:  85.0,
			Comparison: ">",
			Reason:     "overheating",
		},
		{
			MetricName: nvtelemetry.MetricPowerUsageMW,
			Threshold:  400000,
			Comparison: ">",
			Reason:     "power_limit",
		},
	}

	sample := nvtelemetry.Sample{
		Seq: 1,
		Readings: []nvtelemetry.Reading{
			{
				Name:     nvtelemetry.MetricGPUUtilization,
				Evidence: nvtelemetry.EvidenceMeasured,
				Value:    50.0,
			},
			{
				Name:     nvtelemetry.MetricTemperatureC,
				Evidence: nvtelemetry.EvidenceMeasured,
				Value:    88.0,
			},
		},
	}

	reading, matchedPolicy, ok := nvtelemetry.EvaluateSample(sample, policies)
	if !ok {
		t.Fatalf("expected sample to match overheating policy")
	}
	if reading.Name != nvtelemetry.MetricTemperatureC || reading.Value != 88.0 {
		t.Errorf("wrong matched reading: %+v", reading)
	}
	if matchedPolicy.Reason != "overheating" {
		t.Errorf("wrong matched policy reason: %q", matchedPolicy.Reason)
	}
}

func TestCollectorSampleHook(t *testing.T) {
	sampler := nvtelemetry.NewFakeSampler(nvtelemetry.DeviceMetadata{UUID: "GPU-mock-0001"})
	sampler.Enqueue([]nvtelemetry.Reading{
		{
			Name:     nvtelemetry.MetricGPUUtilization,
			Evidence: nvtelemetry.EvidenceMeasured,
			Value:    92.0,
		},
	})

	cfg := nvtelemetry.CollectorConfig{
		IntervalNanos: int64(10 * time.Millisecond),
	}
	collector := nvtelemetry.NewCollector("test-hook", sampler, nvtelemetry.RealClock{}, cfg, nil)

	hookCalled := false
	var hookedSample nvtelemetry.Sample
	collector.SetSampleHook(func(s nvtelemetry.Sample) {
		hookCalled = true
		hookedSample = s
	})

	if err := collector.Step(context.Background()); err != nil {
		t.Fatalf("Step failed: %v", err)
	}

	if !hookCalled {
		t.Fatalf("expected sample hook to be called on Step")
	}
	if len(hookedSample.Readings) != 1 || hookedSample.Readings[0].Value != 92.0 {
		t.Fatalf("hooked sample mismatch: %+v", hookedSample)
	}
}
