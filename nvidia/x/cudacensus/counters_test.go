package cudacensus

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/nvml"
)

// recordedBlock builds a GPM result block as the driver would leave it: one
// entry per requested counter, each with its own return code, and a value field
// the driver may have written whether or not the metric succeeded.
func recordedBlock(t *testing.T, want []Counter, results []nvml.Result, values []float64) *nvml.GpmMetricsGetParams {
	t.Helper()
	if len(want) != len(results) || len(want) != len(values) {
		t.Fatalf("recordedBlock: %d counters, %d results, %d values", len(want), len(results), len(values))
	}
	var p nvml.GpmMetricsGetParams
	p.NumMetrics = uint32(len(want))
	for i, c := range want {
		p.Metrics[i].MetricId = uint32(counters[c].metric)
		p.Metrics[i].NvmlReturn = results[i]
		p.Metrics[i].Value = values[i]
	}
	return &p
}

// TestUnsupportedGPMMetricReportedAbsent: GPM reports support per metric inside
// the query, so a device that implements GPM can refuse an individual counter.
// That refusal is a fact about the device and has to survive into the reading.
func TestUnsupportedGPMMetricReportedAbsent(t *testing.T) {
	want := []Counter{SMActive, HMMATensorActive, DRAMBandwidthUtil}
	block := recordedBlock(t, want,
		[]nvml.Result{nvml.Success, nvml.ErrNotSupported, nvml.Success},
		[]float64{42.5, 99.9, 17.25})

	dst := make([]Reading, len(want))
	fillReadings(dst, want, block)

	if v, ok := dst[0].Get(); !ok || v != 42.5 {
		t.Errorf("sm_active: got (%v, %v), want (42.5, true)", v, ok)
	}
	if _, ok := dst[1].Get(); ok {
		t.Errorf("hmma_tensor_active: an unsupported metric reported a value")
	}
	if dst[1].Value != 0 {
		t.Errorf("hmma_tensor_active: absent reading carries value %v; the driver's 99.9 must be dropped", dst[1].Value)
	}
	if !strings.Contains(strings.ToLower(dst[1].Reason), "supported") {
		t.Errorf("hmma_tensor_active: reason %q does not say it is unsupported", dst[1].Reason)
	}
	if v, ok := dst[2].Get(); !ok || v != 17.25 {
		t.Errorf("dram_bandwidth_util: got (%v, %v), want (17.25, true)", v, ok)
	}
}

// TestAbsentCounterNeverZeroOrCarriedOver: the two ways this goes wrong are
// spelling absence as zero, and leaving the previous tick's value in the slice.
// The second is the dangerous one, because a carried-over counter is
// indistinguishable from a steady one.
func TestAbsentCounterNeverZeroOrCarriedOver(t *testing.T) {
	want := []Counter{SMActive, HMMATensorActive}
	dst := make([]Reading, len(want))

	// Tick one: both available.
	fillReadings(dst, want, recordedBlock(t, want,
		[]nvml.Result{nvml.Success, nvml.Success}, []float64{50, 60}))
	if v, ok := dst[1].Get(); !ok || v != 60 {
		t.Fatalf("tick one: hmma (%v, %v), want (60, true)", v, ok)
	}

	// Tick two: the driver refuses the tensor metric. The 60 must not survive.
	fillReadings(dst, want, recordedBlock(t, want,
		[]nvml.Result{nvml.Success, nvml.ErrNotSupported}, []float64{55, 60}))
	if v, ok := dst[1].Get(); ok {
		t.Errorf("tick two: hmma carried over as %v", v)
	}
	if dst[1].Value != 0 || dst[1].Available {
		t.Errorf("tick two: hmma reading is %+v, want an absence with no value", dst[1])
	}

	// A permission refusal is not the same fact as an unsupported metric, and
	// both are distinct from zero activity.
	fillReadings(dst, want, recordedBlock(t, want,
		[]nvml.Result{nvml.ErrNoPermission, nvml.Success}, []float64{0, 0}))
	if dst[0].Available {
		t.Error("a refused counter reported a value")
	}
	if dst[0].Reason == "" {
		t.Error("a refused counter carries no reason")
	}
	if v, ok := dst[1].Get(); !ok || v != 0 {
		t.Errorf("hmma: got (%v, %v); a genuine zero is a reading, not an absence", v, ok)
	}
}

// TestIdleDeviceReportsZeroWithConfidence: idle is a real state. A device that
// answers every counter with zero has told us something, and it must be
// distinguishable from a device that answered nothing.
func TestIdleDeviceReportsZeroWithConfidence(t *testing.T) {
	want := AllCounters()
	results := make([]nvml.Result, len(want))
	values := make([]float64, len(want))
	dst := make([]Reading, len(want))
	fillReadings(dst, want, recordedBlock(t, want, results, values))

	for i, r := range dst {
		v, ok := r.Get()
		if !ok {
			t.Errorf("%s: idle device reported it absent (%s)", want[i], r.Reason)
			continue
		}
		if v != 0 {
			t.Errorf("%s: got %v, want 0", want[i], v)
		}
		if r.Reason != "" {
			t.Errorf("%s: an available reading carries a reason %q", want[i], r.Reason)
		}
	}
}

// TestNoSummingAcrossGranularityLevels: utilization at one granularity is not an
// addend of utilization at another.
func TestNoSummingAcrossGranularityLevels(t *testing.T) {
	perSM := []Reading{
		{Counter: FP32Active, Value: 20, Available: true},
		{Counter: FP16Active, Value: 30, Available: true},
	}
	sum, g, err := SumByGranularity(perSM)
	if err != nil {
		t.Fatalf("summing two per-SM readings: %v", err)
	}
	if sum != 50 || g != PerSM {
		t.Errorf("got (%v, %s), want (50, %s)", sum, g, PerSM)
	}

	mixed := []Reading{
		{Counter: SMActive, Value: 20, Available: true},         // per SM
		{Counter: SMOccupancy, Value: 30, Available: true},      // per warp slot
		{Counter: DRAMBandwidthUtil, Value: 5, Available: true}, // device wide
	}
	if _, _, err := SumByGranularity(mixed); err == nil {
		t.Error("summed across granularities")
	} else if !strings.Contains(err.Error(), "cannot sum") {
		t.Errorf("error does not refuse the sum: %v", err)
	}

	// An absent reading must not contribute zero to a total: a sum that
	// silently drops a counter reports a lower utilization as a fact.
	withAbsence := []Reading{
		{Counter: FP32Active, Value: 20, Available: true},
		{Counter: FP16Active, Reason: "ErrNotSupported"},
	}
	if _, _, err := SumByGranularity(withAbsence); err == nil {
		t.Error("summed over an absent counter")
	} else if !strings.Contains(err.Error(), "absent") {
		t.Errorf("error does not name the absence: %v", err)
	}
}

// TestUtilizationNotReportedAsEfficiency: SMActive is the fraction of SMs that
// were busy, SMOccupancy the fraction of warp slots filled. Neither is an
// efficiency, and the package must not offer a conversion -- an SM issuing one
// instruction per kilocycle is busy.
func TestUtilizationNotReportedAsEfficiency(t *testing.T) {
	for _, c := range []Counter{SMActive, SMOccupancy, GraphicsActive, AnyTensorActive} {
		name := string(c)
		if strings.Contains(name, "efficiency") || strings.Contains(name, "flops") {
			t.Errorf("%s: an activity counter is named as a throughput", c)
		}
	}
	if g, err := GranularityOf(SMActive); err != nil || g != PerSM {
		t.Errorf("sm_active granularity (%s, %v), want %s", g, err, PerSM)
	}
	if g, err := GranularityOf(SMOccupancy); err != nil || g != PerWarp {
		t.Errorf("sm_occupancy granularity (%s, %v), want %s: occupancy is a residency of warp slots", g, err, PerWarp)
	}
}

// TestNoWallTimeSubstitution: the interval a sample reports is the span between
// the two GPM samples the values were differenced from. A sampler that has taken
// one sample has no interval, and must say so rather than reporting the host
// time since it started.
func TestNoWallTimeSubstitution(t *testing.T) {
	s := &gpmSampler{want: []Counter{SMActive}}
	dst := make([]Reading, 1)
	// Warm-up path, exercised without a device: no previous sample exists, so
	// no interval and no values.
	for i, c := range s.want {
		dst[i] = Reading{Counter: c, Reason: "warm-up: no previous sample"}
	}
	sample := Sample{Readings: dst}
	if sample.IntervalNs != 0 {
		t.Errorf("a warm-up sample reports an interval of %d ns", sample.IntervalNs)
	}
	if _, ok := dst[0].Get(); ok {
		t.Error("a warm-up sample reported a value")
	}
	if r, ok := sample.Reading(SMActive); !ok || !strings.Contains(r.Reason, "warm-up") {
		t.Errorf("warm-up reading is %+v", r)
	}
}

// TestSamplingLoopAllocationFree: the decode step runs on every tick, and a
// sampler that allocates per tick perturbs what it measures.
func TestSamplingLoopAllocationFree(t *testing.T) {
	want := AllCounters()
	results := make([]nvml.Result, len(want))
	values := make([]float64, len(want))
	for i := range results {
		results[i] = nvml.Success
		values[i] = float64(i)
	}
	block := recordedBlock(t, want, results, values)
	dst := make([]Reading, len(want))

	if n := testing.AllocsPerRun(200, func() { fillReadings(dst, want, block) }); n != 0 {
		t.Errorf("fillReadings allocates %v times per tick, want 0", n)
	}
}

// TestGPMStructVersionsMustBeSupplied: the nvml binding exposes no constant for
// the GPM parameter-block versions, and a hardcoded guess turns an ABI check into
// an ABI assumption. Zero is refused rather than defaulted.
func TestGPMStructVersionsMustBeSupplied(t *testing.T) {
	for _, v := range []StructVersions{{}, {GpmSupport: 1}, {GpmMetricsGet: 1}} {
		if err := v.validate(); err == nil {
			t.Errorf("%+v accepted", v)
		} else if !strings.Contains(err.Error(), "nvml.h") {
			t.Errorf("%+v: error does not say where the value comes from: %v", v, err)
		}
	}
	if err := (StructVersions{GpmSupport: 1, GpmMetricsGet: 1}).validate(); err != nil {
		t.Errorf("supplied versions rejected: %v", err)
	}
}

func TestUnknownCounterRejected(t *testing.T) {
	if _, err := GranularityOf(Counter("fp4_tensor_active")); err == nil {
		t.Error("an unknown counter was given a granularity")
	}
	if _, _, err := OpenSampler(0, []Counter{Counter("nope")}, StructVersions{GpmSupport: 1, GpmMetricsGet: 1}); err == nil {
		t.Error("OpenSampler accepted an unknown counter")
	}
}
