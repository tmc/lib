package cuptiactivity_test

import (
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
	"github.com/tmc/lib/nvidia/x/cuptipmsampling"
)

func TestAttributePMSamples(t *testing.T) {
	records := []cuptiactivity.Record{
		{
			Kind:     cupti.ActivityKindMemcpy,
			Name:     "memcpy_d2h",
			Start:    500,
			End:      800,
			Complete: true,
		},
		{
			Kind:     cupti.ActivityKindConcurrentKernel,
			Name:     "kernel_a",
			Start:    1000,
			End:      3000,
			Complete: true,
		},
	}

	pmSamples := []cuptipmsampling.Sample{
		{
			Index:          0,
			StartTimestamp: 500,
			EndTimestamp:   800, // before kernel
			Values:         map[string]float64{"sm_util": 20.0},
		},
		{
			Index:          1,
			StartTimestamp: 900,
			EndTimestamp:   1500, // overlaps start
			Values:         map[string]float64{"sm_util": 80.0},
		},
		{
			Index:          2,
			StartTimestamp: 1500,
			EndTimestamp:   2500, // completely inside
			Values:         map[string]float64{"sm_util": 90.0},
		},
		{
			Index:          3,
			StartTimestamp: 3500,
			EndTimestamp:   4000, // after kernel
			Values:         map[string]float64{"sm_util": 10.0},
		},
	}

	attributed := cuptiactivity.AttributePMSamples(records, pmSamples)
	if len(attributed) != 1 {
		t.Fatalf("len(attributed) = %d, want 1 (memcpy should be skipped)", len(attributed))
	}

	ak := attributed[0]
	if ak.KernelName != "kernel_a" {
		t.Errorf("KernelName = %q, want kernel_a", ak.KernelName)
	}
	if ak.DurationNs != 2000 {
		t.Errorf("DurationNs = %d, want 2000", ak.DurationNs)
	}
	if len(ak.PMSamples) != 2 {
		t.Fatalf("len(ak.PMSamples) = %d, want 2 (only samples 1 and 2 overlap)", len(ak.PMSamples))
	}
	avgSM := ak.PMAverages["sm_util"]
	if math.Abs(avgSM-85.0) > 0.01 { // (80 + 90) / 2 = 85
		t.Errorf("PMAverages[sm_util] = %.2f, want 85.0", avgSM)
	}
}

func TestPMReaderInActivity(t *testing.T) {
	samples := []cuptiactivity.PMSample{
		{
			Index:          0,
			StartTimestamp: 100,
			EndTimestamp:   200,
			DurationNs:     100,
			Values:         map[string]float64{"dram__throughput": 55.5},
		},
	}
	r := cuptiactivity.NewPMReader(samples)
	s, err := r.ReadSample()
	if err != nil {
		t.Fatalf("ReadSample failed: %v", err)
	}
	if s.Values["dram__throughput"] != 55.5 {
		t.Errorf("unexpected sample: %+v", s)
	}
}
