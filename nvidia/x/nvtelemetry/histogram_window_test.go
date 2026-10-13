package nvtelemetry_test

import (
	"bytes"
	"context"
	"fmt"
	"math"
	"reflect"
	"strconv"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	n "github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func hwU(v uint64) *uint64   { return &v }
func hwF(v float64) *float64 { return &v }
func hwI(v int64) *int64     { return &v }
func hwReading(start, end int64, counts []uint64, temp string) n.Reading {
	return n.Reading{Name: "window_latency", Unit: n.UnitSeconds, Backend: "fixture", DeviceUUID: "GPU-hw", Evidence: n.EvidenceMeasured, TimestampMonotonic: end, Observation: &n.Observation{Family: "histograms", SourceClass: "synthetic", Producer: "constructed", ProducerVersion: "1", ProducerSchema: "fixture.v5", Kind: "histogram", ClockEpoch: "epoch", UncertaintyNS: hwI(0), Interval: &tracecontract.Interval{Clock: "fixture", Start: start, End: end}, Observer: n.Observer{SamplingCoverage: &n.SamplingCoverage{Status: "unknown"}}, Histogram: &n.Histogram{Bounds: []string{"1", "+Inf"}, Counts: counts, Count: hwU(counts[len(counts)-1]), Temporality: temp}}}
}
func hwSamples(readings ...n.Reading) []n.Sample {
	out := make([]n.Sample, len(readings))
	for i, r := range readings {
		out[i] = n.Sample{Seq: uint64(i + 1), QueryStartNanos: int64(i + 1), QueryEndNanos: int64(i + 2), Readings: []n.Reading{r}}
	}
	return out
}
func hwManifest(samples []n.Sample, version int) *n.Manifest {
	return &n.Manifest{RunID: "histogram-window-control", SchemaVersion: version, Devices: []n.DeviceMetadata{{UUID: "GPU-hw"}}, ClockMonotonicDomain: "fixture", ClockMonotonicEpochID: "epoch", ClockWallDomain: "unix_utc", HasClockCalibration: true, ClockMappingValidFrom: 1, ClockMappingValidUntil: 100000, StartTimeMonotonic: 1, EndTimeMonotonic: 100000, StartTimeUnix: 101, EndTimeUnix: 100100, IsComplete: true, TotalSamples: uint64(len(samples))}
}
func hwCanonical(t *testing.T, samples []n.Sample, version int) (*n.Manifest, []n.Sample) {
	t.Helper()
	var b bytes.Buffer
	m := hwManifest(samples, version)
	m.IsComplete = false
	m.TotalSamples = 0
	w := n.NewWriter(&b, n.CollectorConfig{})
	if e := w.WriteManifest(m); e != nil {
		t.Fatal(e)
	}
	for i := range samples {
		if e := w.WriteSample(&samples[i]); e != nil {
			t.Fatal(e)
		}
	}
	m.IsComplete = true
	m.TotalSamples = uint64(len(samples))
	m.LostRecordsByFamily = w.FamilyLoss()
	if e := w.WriteManifest(m); e != nil {
		t.Fatal(e)
	}
	decoded, _, out, e := n.ReadStreamStrict(bytes.NewReader(b.Bytes()), 0)
	if e != nil {
		t.Fatal(e)
	}
	return decoded, out
}
func hwOptions(m *n.Manifest, p n.HistogramWindowPolicy) n.HistogramWindowOptions {
	return n.HistogramWindowOptions{Manifest: m, Policy: p, ClockDomain: "fixture", ClockEpoch: "epoch"}
}
func hwAggregate(t *testing.T, readings []n.Reading, p n.HistogramWindowPolicy) n.HistogramWindowSummary {
	t.Helper()
	m, s := hwCanonical(t, hwSamples(readings...), n.SchemaVersionV5)
	out, e := n.AggregateHistogramWindow("window", 10, 30, s, hwOptions(m, p))
	if e != nil || len(out) != 1 {
		t.Fatalf("aggregate %v %v", out, e)
	}
	return out[0]
}

func TestHistogramWindowAvailable(t *testing.T) {
	for _, policy := range []n.HistogramWindowPolicy{n.HistogramWindowDelta, n.HistogramWindowCumulativeDifference} {
		t.Run(string(policy), func(t *testing.T) {
			a, b := hwReading(10, 20, []uint64{1, 2}, "delta"), hwReading(20, 30, []uint64{2, 3}, "delta")
			want := []uint64{3, 5}
			if policy == n.HistogramWindowCumulativeDifference {
				a = hwReading(1, 10, []uint64{1, 2}, "cumulative")
				b = hwReading(1, 30, []uint64{3, 5}, "cumulative")
				want = []uint64{2, 3}
			}
			a.Observation.Histogram.Sum = hwF(-2)
			b.Observation.Histogram.Sum = hwF(0)
			a.Observation.Histogram.Min = hwF(-2)
			a.Observation.Histogram.Max = hwF(1)
			b.Observation.Histogram.Min = hwF(-1)
			b.Observation.Histogram.Max = hwF(1)
			out := hwAggregate(t, []n.Reading{a, b}, policy)
			if out.Status != n.HistogramWindowAvailable || !reflect.DeepEqual(out.Histogram.Counts, want) || *out.Histogram.Count != want[1] || out.CoverageStatus != n.HistogramWindowCompleteDeclaredIntervals || out.CoveredDurationNanos != 20 || out.ObserverCoverage.Status != "unknown" {
				t.Fatalf("result %+v", out)
			}
			if out.SumStatus != n.HistogramWindowFieldAvailable {
				t.Fatal(out)
			}
			if policy == n.HistogramWindowCumulativeDifference && (out.Histogram.Min != nil || out.MinStatus != n.HistogramWindowFieldNotDerivable) {
				t.Fatal("cumulative extrema", out)
			}
		})
	}
}

func TestHistogramWindowRefusals(t *testing.T) {
	rows := []struct {
		name   string
		mutate func(*n.Reading, *n.Reading)
		reason n.HistogramWindowReason
	}{
		{"gap", func(a, b *n.Reading) { b.Observation.Interval.Start = 21 }, n.HistogramWindowReasonGap},
		{"overlap", func(a, b *n.Reading) { b.Observation.Interval.Start = 19 }, n.HistogramWindowReasonOverlap},
		{"straddle", func(a, b *n.Reading) { a.Observation.Interval.Start = 9 }, n.HistogramWindowReasonStraddlingInterval},
		{"duplicate", func(a, b *n.Reading) {
			b.Observation.Interval.Start = 10
			b.Observation.Interval.End = 20
			b.TimestampMonotonic = 20
		}, n.HistogramWindowReasonDuplicateInterval},
		{"missing interval", func(a, b *n.Reading) { b.Observation.Interval = nil }, n.HistogramWindowReasonMissingInterval},
		{"unknown uncertainty", func(a, b *n.Reading) { b.Observation.UncertaintyNS = nil }, n.HistogramWindowReasonUnknownUncertainty},
		{"positive uncertainty", func(a, b *n.Reading) { b.Observation.UncertaintyNS = hwI(1) }, n.HistogramWindowReasonBoundaryUncertainty},
		{"clock", func(a, b *n.Reading) { b.Observation.Interval.Clock = "other" }, n.HistogramWindowReasonClockMismatch},
		{"epoch", func(a, b *n.Reading) { b.Observation.ClockEpoch = "other" }, n.HistogramWindowReasonClockMismatch},
		{"bounds", func(a, b *n.Reading) { b.Observation.Histogram.Bounds[0] = "2" }, n.HistogramWindowReasonSchemaChanged},
		{"producer", func(a, b *n.Reading) { b.Observation.ProducerVersion = "2" }, n.HistogramWindowReasonIdentityChanged},
		{"observer pass", func(a, b *n.Reading) { b.Observation.Observer.Pass = "other" }, n.HistogramWindowReasonIdentityChanged},
		{"temporal", func(a, b *n.Reading) { b.Observation.Histogram.Temporality = "cumulative" }, n.HistogramWindowReasonTemporalityMismatch},
		{"count-only", func(a, b *n.Reading) {
			a.Observation.Histogram.Bounds = []string{}
			a.Observation.Histogram.Counts = []uint64{}
			b.Observation.Histogram.Bounds = []string{}
			b.Observation.Histogram.Counts = []uint64{}
		}, n.HistogramWindowReasonMissingBuckets},
		{"no-recorded", func(a, b *n.Reading) {
			h := b.Observation.Histogram
			h.Bounds = []string{}
			h.Counts = []uint64{}
			h.Count = hwU(0)
			h.Flags = 1
			b.Evidence = n.EvidenceUnavailable
			b.UnavailableReason = "no_recorded_value"
		}, n.HistogramWindowReasonNoRecordedValue},
		{"unavailable", func(a, b *n.Reading) {
			b.Observation.Histogram = nil
			b.Evidence = n.EvidenceUnavailable
			b.UnavailableReason = "missing"
		}, n.HistogramWindowReasonSourceUnavailable},
		{"overflow", func(a, b *n.Reading) {
			a.Observation.Histogram.Counts = []uint64{math.MaxUint64, math.MaxUint64}
			a.Observation.Histogram.Count = hwU(math.MaxUint64)
		}, n.HistogramWindowReasonCountOverflow},
	}
	for _, row := range rows {
		t.Run(row.name, func(t *testing.T) {
			a := hwReading(10, 20, []uint64{1, 2}, "delta")
			b := hwReading(20, 30, []uint64{2, 3}, "delta")
			row.mutate(&a, &b)
			x := hwAggregate(t, []n.Reading{a, b}, n.HistogramWindowDelta)
			if x.Reason != row.reason || x.Histogram != nil || x.Status == n.HistogramWindowAvailable {
				t.Fatalf("%+v want %s", x, row.reason)
			}
		})
	}
}

func TestHistogramWindowCumulativeRefusals(t *testing.T) {
	for _, row := range []struct {
		name   string
		mutate func(*n.Reading, *n.Reading)
		reason n.HistogramWindowReason
	}{
		{"origin", func(a, b *n.Reading) { b.Observation.Interval.Start = 2 }, n.HistogramWindowReasonAggregationEpochChanged},
		{"missing endpoint", func(a, b *n.Reading) { a.Observation.Interval.End = 11; a.TimestampMonotonic = 11 }, n.HistogramWindowReasonMissingEndpoint},
		{"decrease", func(a, b *n.Reading) {
			b.Observation.Histogram.Counts = []uint64{0, 1}
			b.Observation.Histogram.Count = hwU(1)
		}, n.HistogramWindowReasonCounterDecrease},
		{"invalid interval cdf", func(a, b *n.Reading) {
			a.Observation.Histogram.Counts = []uint64{5, 10}
			a.Observation.Histogram.Count = hwU(10)
			b.Observation.Histogram.Counts = []uint64{9, 11}
			b.Observation.Histogram.Count = hwU(11)
		}, n.HistogramWindowReasonInvalidIntervalCDF},
	} {
		t.Run(row.name, func(t *testing.T) {
			a := hwReading(1, 10, []uint64{1, 2}, "cumulative")
			b := hwReading(1, 30, []uint64{3, 5}, "cumulative")
			row.mutate(&a, &b)
			x := hwAggregate(t, []n.Reading{a, b}, n.HistogramWindowCumulativeDifference)
			if x.Reason != row.reason {
				t.Fatalf("%+v want%s", x, row.reason)
			}
		})
	}
	// A recovered endpoint does not hide an observed intermediate decrease.
	a := hwReading(1, 10, []uint64{5, 10}, "cumulative")
	mid := hwReading(1, 20, []uint64{1, 2}, "cumulative")
	b := hwReading(1, 30, []uint64{9, 15}, "cumulative")
	if x := hwAggregate(t, []n.Reading{a, mid, b}, n.HistogramWindowCumulativeDifference); x.Reason != n.HistogramWindowReasonCounterDecrease {
		t.Fatal(x)
	}
}

func TestHistogramWindowOptionalAndEmpty(t *testing.T) {
	for _, policy := range []n.HistogramWindowPolicy{n.HistogramWindowDelta, n.HistogramWindowCumulativeDifference} {
		a, b := hwReading(10, 20, []uint64{0, 0}, "delta"), hwReading(20, 30, []uint64{0, 0}, "delta")
		if policy == n.HistogramWindowCumulativeDifference {
			a = hwReading(1, 10, []uint64{0, 0}, "cumulative")
			b = hwReading(1, 30, []uint64{0, 0}, "cumulative")
		}
		for _, r := range []*n.Reading{&a, &b} {
			r.Observation.Histogram.Bounds = []string{}
			r.Observation.Histogram.Counts = []uint64{}
		}
		x := hwAggregate(t, []n.Reading{a, b}, policy)
		if x.Status != n.HistogramWindowAvailable || *x.Histogram.Count != 0 || len(x.Histogram.Counts) != 0 {
			t.Fatal("empty schema", x)
		}
	}

	a := hwReading(10, 20, []uint64{0, 0}, "delta")
	b := hwReading(20, 30, []uint64{0, 0}, "delta")
	a.Observation.Histogram.Sum = hwF(0)
	b.Observation.Histogram.Sum = hwF(0)
	x := hwAggregate(t, []n.Reading{a, b}, n.HistogramWindowDelta)
	if x.Status != n.HistogramWindowAvailable || *x.Histogram.Count != 0 || x.Histogram.Sum == nil || *x.Histogram.Sum != 0 || x.MinStatus != n.HistogramWindowFieldEmpty {
		t.Fatal(x)
	}
	b.Observation.Histogram.Sum = nil
	x = hwAggregate(t, []n.Reading{a, b}, n.HistogramWindowDelta)
	if x.Histogram.Sum != nil || x.SumStatus != n.HistogramWindowFieldMissing {
		t.Fatal(x)
	}
	a = hwReading(10, 20, []uint64{1, 2}, "delta")
	b = hwReading(20, 30, []uint64{2, 3}, "delta")
	a.Observation.Histogram.Sum = hwF(math.MaxFloat64)
	b.Observation.Histogram.Sum = hwF(math.MaxFloat64)
	a.Observation.Histogram.Min = hwF(0)
	b.Observation.Histogram.Min = hwF(1)
	x = hwAggregate(t, []n.Reading{a, b}, n.HistogramWindowDelta)
	if x.Status != n.HistogramWindowAvailable || x.SumStatus != n.HistogramWindowFieldNonfinite || x.Histogram.Sum != nil || x.MinStatus != n.HistogramWindowFieldAvailable || x.MaxStatus != n.HistogramWindowFieldMissing {
		t.Fatal(x)
	}
	a.Observation.Histogram.Counts = []uint64{1<<53 + 1, 1<<53 + 1}
	a.Observation.Histogram.Count = hwU(1<<53 + 1)
	b.Observation.Histogram.Counts = []uint64{math.MaxUint64 - (1<<53 + 1), math.MaxUint64 - (1<<53 + 1)}
	b.Observation.Histogram.Count = hwU(math.MaxUint64 - (1<<53 + 1))
	x = hwAggregate(t, []n.Reading{a, b}, n.HistogramWindowDelta)
	if *x.Histogram.Count != math.MaxUint64 || x.Histogram.Counts[0] != math.MaxUint64 {
		t.Fatal("precision", x)
	}
}

func TestHistogramWindowImportedDirect(t *testing.T) {
	for _, temp := range []string{"delta", "cumulative"} {
		t.Run(temp, func(t *testing.T) {
			a := hwReading(10, 20, []uint64{1, 2}, temp)
			b := hwReading(20, 30, []uint64{2, 3}, temp)
			policy := n.HistogramWindowDelta
			if temp == "cumulative" {
				a = hwReading(1, 10, []uint64{1, 2}, temp)
				b = hwReading(1, 30, []uint64{3, 5}, temp)
				policy = n.HistogramWindowCumulativeDifference
			}
			// Direct imported fixture exercises the actual restricted importer separately
			// from the mapped export/import cell, which is blocked by its attribute cap.
			scope := "github.com/tmc/lib/nvidia/x/nvtelemetry"
			for _, p := range []*n.Reading{&a, &b} {
				o := p.Observation
				o.UncertaintyNS = nil
				o.SourceClass = "imported"
				o.ClockEpoch = "unix"
				o.OTLPScopeName = &scope
				o.OTLPClock = &n.OTLPClockMetadata{ClockDomain: "fixture", HasClockCalibration: true}
				p.TimestampUnix = p.TimestampMonotonic + 100
				p.TimestampMonotonic = 0
				o.Interval.Clock = "unix"
				o.Interval.Start += 100
				o.Interval.End += 100
			}
			mapping := &n.UnixTimeMapping{SourceClockDomain: "fixture", SourceClockEpoch: "epoch", SourceValidFromNanos: 1, SourceValidUntilNanos: 40, UnixOffsetNanos: 100, Provenance: "constructed"}
			var importedReadings []n.Reading
			for _, reading := range []n.Reading{a, b} {
				var raw bytes.Buffer
				if e := n.WriteOTLPMetrics(&raw, []n.Reading{reading}, n.ExportOptions{HistogramBindings: []n.HistogramBinding{{Name: "window_latency", Unit: n.UnitSeconds}}}); e != nil {
					t.Fatal(e)
				}
				imported, e := n.ReadOTLPHistogramMetrics(context.Background(), bytes.NewReader(raw.Bytes()), n.OTLPReadOptions{HistogramBindings: []n.HistogramBinding{{Name: "window_latency", Unit: n.UnitSeconds}}})
				if e != nil {
					t.Fatal(e)
				}
				importedReadings = append(importedReadings, imported.Readings...)
			}
			m, s := hwCanonical(t, hwSamples(importedReadings...), n.SchemaVersionV5)
			opts := hwOptions(m, policy)
			opts.ClockDomain = "unix"
			opts.ClockEpoch = "unix"
			x, e := n.AggregateHistogramWindow("window", 110, 130, s, opts)
			if e != nil || len(x) != 1 || x[0].Status != n.HistogramWindowAvailable || x[0].Sources[0].SourceClass != "imported" || x[0].ObserverCoverage.Status != "unknown" {
				t.Fatalf("%+v %v", x, e)
			}
			opts.UnixTimeMapping = mapping
			x, e = n.AggregateHistogramWindow("window", 110, 130, s, opts)
			if e != nil || x[0].Reason != n.HistogramWindowReasonClockMismatch {
				t.Fatal(x, e)
			}
		})
	}
}

func TestHistogramWindowLegacyAndOwnership(t *testing.T) {
	for _, version := range []int{n.SchemaVersionV3, n.SchemaVersionV4} {
		a := hwReading(10, 30, []uint64{1, 2}, "")
		a.Observation.Histogram.Count = nil
		if version == n.SchemaVersionV3 {
			a.Observation.Observer = n.Observer{}
		}
		m, s := hwCanonical(t, hwSamples(a), version)
		x, e := n.AggregateHistogramWindow("window", 10, 30, s, hwOptions(m, n.HistogramWindowDelta))
		if e != nil || x[0].Reason != n.HistogramWindowReasonTemporalUnknown {
			t.Fatal(x, e)
		}
	}
	m, s := hwCanonical(t, hwSamples(hwReading(10, 30, []uint64{1, 2}, "delta")), n.SchemaVersionV5)
	opts := hwOptions(m, n.HistogramWindowDelta)
	x, e := n.AggregateHistogramWindow("window", 10, 30, s, opts)
	if e != nil {
		t.Fatal(e)
	}
	x[0].Histogram.Counts[0] = 99
	*x[0].Histogram.Count = 99
	x[0].Sources[0].Interval.Start = 0
	x[0].Sources[0].Observer.SamplingCoverage.Status = "fake"
	again, e := n.AggregateHistogramWindow("window", 10, 30, s, opts)
	if e != nil || again[0].Histogram.Counts[0] != 1 || *again[0].Histogram.Count != 2 || again[0].Sources[0].Interval.Start != 10 || again[0].ObserverCoverage.Status != "unknown" {
		t.Fatal(again, e)
	}
	for _, row := range []struct {
		name   string
		mutate func(*n.HistogramWindowOptions)
		reason n.HistogramWindowReason
	}{
		{"missing manifest", func(o *n.HistogramWindowOptions) { o.Manifest = nil }, n.HistogramWindowReasonMissingManifest},
		{"incomplete", func(o *n.HistogramWindowOptions) { copy := *m; copy.IsComplete = false; o.Manifest = &copy }, n.HistogramWindowReasonIncompleteStream},
		{"unknown loss", func(o *n.HistogramWindowOptions) {
			copy := *m
			copy.LostRecordsByFamily = map[string]tracecontract.Loss{"histograms": {Unknown: true, Total: 1}}
			o.Manifest = &copy
		}, n.HistogramWindowReasonUnknownLoss},
		{"loss", func(o *n.HistogramWindowOptions) {
			copy := *m
			copy.LostRecordsByFamily = map[string]tracecontract.Loss{"histograms": {Total: 2, Dropped: 1}}
			o.Manifest = &copy
		}, n.HistogramWindowReasonLostReadings},
	} {
		t.Run(row.name, func(t *testing.T) {
			o := opts
			row.mutate(&o)
			x, e := n.AggregateHistogramWindow("window", 10, 30, s, o)
			if e != nil || x[0].Reason != row.reason {
				t.Fatal(x, e)
			}
		})
	}
	for _, limit := range []int{-1, 65537} {
		o := opts
		o.MaxReadings = limit
		if _, e := n.AggregateHistogramWindow("window", 10, 30, s, o); e == nil {
			t.Fatal("reading option limit", limit)
		}
	}
	for _, field := range []string{"readings", "series", "bounds"} {
		o := opts
		switch field {
		case "readings":
			o.MaxReadings = 1
		case "series":
			o.MaxSeries = 1
		case "bounds":
			o.MaxBounds = 1
		}
		_, e := n.AggregateHistogramWindow("window", 10, 30, s, o)
		if field == "bounds" && e == nil {
			t.Fatal("bound limit")
		}
		if field != "bounds" && e != nil {
			t.Fatal(e)
		}
	}
	t.Run("omitted samples", func(t *testing.T) {
		copy := *m
		copy.TotalSamples = 2
		if _, e := n.AggregateHistogramWindow("window", 10, 30, s, hwOptions(&copy, n.HistogramWindowDelta)); e == nil {
			t.Fatal("accounting")
		}
	})
}

func TestHistogramWindowLocalMappingAndMalformedReset(t *testing.T) {
	m, s := hwCanonical(t, hwSamples(hwReading(10, 20, []uint64{1, 2}, "delta"), hwReading(20, 30, []uint64{2, 3}, "delta")), n.SchemaVersionV5)
	opts := hwOptions(m, n.HistogramWindowDelta)
	opts.ClockDomain = "unix"
	opts.ClockEpoch = "unix"
	opts.UnixTimeMapping = &n.UnixTimeMapping{SourceClockDomain: "fixture", SourceClockEpoch: "epoch", SourceValidFromNanos: 1, SourceValidUntilNanos: 40, UnixOffsetNanos: 100, Provenance: "constructed"}
	x, e := n.AggregateHistogramWindow("window", 110, 130, s, opts)
	if e != nil || x[0].Status != n.HistogramWindowAvailable || x[0].Sources[0].AppliedUnixTimeMapping == nil || x[0].Sources[0].Interval.Clock != "fixture" || x[0].Sources[0].SourceClass != "synthetic" {
		t.Fatal(x, e)
	}
	for _, row := range []struct {
		name   string
		mutate func(*n.HistogramWindowOptions)
		reason n.HistogramWindowReason
	}{
		{"range", func(o *n.HistogramWindowOptions) {
			copy := *o.UnixTimeMapping
			copy.SourceValidUntilNanos = 29
			o.UnixTimeMapping = &copy
		}, n.HistogramWindowReasonClockMismatch},
		{"offset overflow", func(o *n.HistogramWindowOptions) {
			copy := *o.UnixTimeMapping
			copy.UnixOffsetNanos = math.MaxInt64
			o.UnixTimeMapping = &copy
		}, n.HistogramWindowReasonClockMismatch},
		{"calibration range", func(o *n.HistogramWindowOptions) { copy := *m; copy.ClockMappingValidUntil = 29; o.Manifest = &copy }, n.HistogramWindowReasonClockMismatch},
		{"unknown calibration", func(o *n.HistogramWindowOptions) {
			copy := *m
			copy.HasClockCalibration = false
			copy.ClockCalibrationBoundsNanos = 0
			o.Manifest = &copy
		}, n.HistogramWindowReasonUnknownUncertainty},
		{"nonzero calibration", func(o *n.HistogramWindowOptions) {
			copy := *m
			copy.ClockCalibrationBoundsNanos = 1
			o.Manifest = &copy
		}, n.HistogramWindowReasonBoundaryUncertainty},
	} {
		t.Run(row.name, func(t *testing.T) {
			o := opts
			row.mutate(&o)
			x, e := n.AggregateHistogramWindow("window", 110, 130, s, o)
			if e != nil || x[0].Reason != row.reason {
				t.Fatal(x, e)
			}
		})
	}
	rUnix := hwReading(10, 30, []uint64{1, 2}, "delta")
	rUnix.TimestampUnix = 129
	mu, su := hwCanonical(t, hwSamples(rUnix), n.SchemaVersionV5)
	ou := opts
	ou.Manifest = mu
	xu, eu := n.AggregateHistogramWindow("window", 110, 130, su, ou)
	if eu != nil || xu[0].Reason != n.HistogramWindowReasonClockMismatch {
		t.Fatal(xu, eu)
	}
	r := hwReading(1, 30, []uint64{1, 2}, "cumulative")
	r.Observation.CounterReset = true
	var raw bytes.Buffer
	w := n.NewWriter(&raw, n.CollectorConfig{})
	if e := w.WriteManifest(hwManifest(nil, n.SchemaVersionV5)); e != nil {
		t.Fatal(e)
	}
	if e := w.WriteSample(&n.Sample{Seq: 1, Readings: []n.Reading{r}}); e == nil {
		t.Fatal("illegal histogram reset flag serialized")
	}
}

func TestHistogramWindowLimits(t *testing.T) {
	// Canonical individual Sample.Readings is independently capped at4096.
	// Many bounded samples reach the new total-readings/series maxima.
	for _, row := range []struct {
		name                          string
		count, readLimit, seriesLimit int
		unique                        bool
		wantError                     bool
	}{
		{"default readings", 4096, 0, 0, false, false},
		{"default one over", 4097, 0, 0, false, true},
		{"custom readings", 4097, 4097, 0, false, false},
		{"maximum readings", 65536, 65536, 0, false, false},
		{"maximum one over", 65537, 65536, 0, false, true},
		{"default series", 4096, 4096, 0, true, false},
		{"default series one over", 4097, 4097, 0, true, true},
		{"custom series", 4097, 4097, 4097, true, false},
		{"maximum series", 65536, 65536, 65536, true, false},
	} {
		t.Run(row.name, func(t *testing.T) {
			var samples []n.Sample
			for i := 0; i < row.count; i++ {
				if i%4096 == 0 {
					samples = append(samples, n.Sample{Seq: uint64(len(samples) + 1), QueryStartNanos: int64(len(samples) + 1), QueryEndNanos: int64(len(samples) + 2)})
				}
				r := hwReading(10+int64(i), 11+int64(i), []uint64{0, 0}, "delta")
				if row.unique {
					r.Name = fmt.Sprintf("window_%d", i)
					r.Observation.Interval.Start = 10
					r.Observation.Interval.End = 30
					r.TimestampMonotonic = 30
				}
				samples[len(samples)-1].Readings = append(samples[len(samples)-1].Readings, r)
			}
			m, s := hwCanonical(t, samples, n.SchemaVersionV5)
			o := hwOptions(m, n.HistogramWindowDelta)
			o.MaxReadings = row.readLimit
			o.MaxSeries = row.seriesLimit
			end := int64(10 + row.count)
			if row.unique {
				end = 30
			}
			x, e := n.AggregateHistogramWindow("window", 10, end, s, o)
			if (e != nil) != row.wantError {
				t.Fatalf("limit error %v want%v", e, row.wantError)
			}
			if e == nil {
				if len(x) == 0 || x[0].Status != n.HistogramWindowAvailable {
					t.Fatal("reachable positive", x[0])
				}
				if row.unique && len(x) != row.count {
					t.Fatal(len(x))
				}
			}
		})
	}
	for _, row := range []struct {
		name         string
		count, limit int
		wantError    bool
	}{
		{"default bounds", 257, 0, false}, {"default bound one over", 258, 0, true}, {"custom bounds", 258, 258, false}, {"maximum bounds", 4097, 4097, false}, {"custom one over", 258, 257, true},
	} {
		t.Run(row.name, func(t *testing.T) {
			r := hwReading(10, 30, []uint64{0, 0}, "delta")
			h := r.Observation.Histogram
			h.Bounds = make([]string, row.count)
			h.Counts = make([]uint64, row.count)
			for i := 0; i < row.count-1; i++ {
				h.Bounds[i] = strconv.Itoa(i)
			}
			h.Bounds[row.count-1] = "+Inf"
			m, s := hwCanonical(t, hwSamples(r), n.SchemaVersionV5)
			o := hwOptions(m, n.HistogramWindowDelta)
			o.MaxBounds = row.limit
			x, e := n.AggregateHistogramWindow("window", 10, 30, s, o)
			if (e != nil) != row.wantError {
				t.Fatal(e, row.wantError)
			}
			if e == nil && x[0].Status != n.HistogramWindowAvailable {
				t.Fatal(x)
			}
		})
	}
}

func TestHistogramWindowIdentityAndMissing(t *testing.T) {
	for _, foreign := range []bool{false, true} {
		r := hwReading(10, 30, []uint64{1, 2}, "delta")
		clock := "fixture"
		if foreign {
			clock = "foreign"
		}
		r.Observation.Assignment = &n.EntityAssignment{Source: "constructed", SourceRevision: "1", State: "unknown", ClockDomain: clock, ClockEpoch: "epoch", StartNanos: 1, EndNanos: 40, Sharing: "unknown"}
		x := hwAggregate(t, []n.Reading{r}, n.HistogramWindowDelta)
		if foreign && x.Reason != n.HistogramWindowReasonClockMismatch || !foreign && x.Status != n.HistogramWindowAvailable {
			t.Fatal("assignment clock", x)
		}
	}

	for _, row := range []struct {
		name   string
		mutate func(*n.Reading)
		reason n.HistogramWindowReason
	}{
		{"name collision", func(r *n.Reading) { r.Name = n.MetricGPUUtilization; r.Unit = n.UnitPercent }, n.HistogramWindowReasonScalarNameCollision},
		{"missing producer", func(r *n.Reading) { r.Observation.Producer = "" }, n.HistogramWindowReasonMissingIdentity},
		{"missing clock", func(r *n.Reading) { r.Observation.Interval.Clock = "" }, n.HistogramWindowReasonMissingClock},
		{"missing epoch", func(r *n.Reading) { r.Observation.ClockEpoch = "" }, n.HistogramWindowReasonMissingEpoch},
	} {
		t.Run(row.name, func(t *testing.T) {
			r := hwReading(10, 30, []uint64{1, 2}, "delta")
			row.mutate(&r)
			x := hwAggregate(t, []n.Reading{r}, n.HistogramWindowDelta)
			if x.Reason != row.reason {
				t.Fatal(x)
			}
		})
	}
	for _, version := range []int{n.SchemaVersionV3, n.SchemaVersionV4} {
		r := hwReading(10, 30, []uint64{1, 2}, "")
		r.Name = n.MetricGPUUtilization
		r.Unit = n.UnitPercent
		r.Observation.Histogram.Count = nil
		if version == n.SchemaVersionV3 {
			r.Observation.Observer = n.Observer{}
		}
		m, s := hwCanonical(t, hwSamples(r), version)
		x, e := n.AggregateHistogramWindow("window", 10, 30, s, hwOptions(m, n.HistogramWindowDelta))
		if e != nil || x[0].Reason != n.HistogramWindowReasonTemporalUnknown {
			t.Fatal(x, e)
		}
	}
}

func hwProjected(r n.Reading) n.Reading {
	o := r.Observation
	i := o.Interval
	o.OTLPSourceTime = &n.OTLPSourceTime{SourceClass: o.SourceClass, SourceClockEpoch: o.ClockEpoch, SourceMonotonicEndNanos: r.TimestampMonotonic, SourceIntervalPresent: true, SourceIntervalClock: i.Clock, SourceIntervalStartNanos: i.Start, SourceIntervalEndNanos: i.End, MappingDomain: i.Clock, MappingEpoch: o.ClockEpoch, MappingValidFromNanos: 1, MappingValidUntilNanos: 100000, MappingUnixOffsetNanos: 100, MappingProvenance: "constructed"}
	o.UncertaintyNS = nil
	o.SourceClass = "imported"
	o.ClockEpoch = "unix"
	scope := "github.com/tmc/lib/nvidia/x/nvtelemetry"
	o.OTLPScopeName = &scope
	o.OTLPClock = &n.OTLPClockMetadata{ClockDomain: "fixture", HasClockCalibration: true}
	r.TimestampUnix = r.TimestampMonotonic + 100
	r.TimestampMonotonic = 0
	i.Start += 100
	i.End += 100
	i.Clock = "unix"
	return r
}

func TestHistogramWindowProjectedCanonicalEpoch(t *testing.T) {
	a := hwProjected(hwReading(1, 10, []uint64{1, 2}, "cumulative"))
	b := hwProjected(hwReading(1, 30, []uint64{3, 5}, "cumulative"))
	m, s := hwCanonical(t, hwSamples(a, b), n.SchemaVersionV5)
	o := hwOptions(m, n.HistogramWindowCumulativeDifference)
	o.ClockDomain = "unix"
	o.ClockEpoch = "unix"
	x, e := n.AggregateHistogramWindow("window", 110, 130, s, o)
	if e != nil || x[0].Status != n.HistogramWindowAvailable || x[0].Sources[0].OTLPSourceTime == nil {
		t.Fatal(x, e)
	}
	// Constructed canonical projected fields are not credit for the blocked OTLP export/import cell.
	b.Observation.OTLPSourceTime.SourceClockEpoch = "other"
	b.Observation.OTLPSourceTime.MappingEpoch = "other"
	m, s = hwCanonical(t, hwSamples(a, b), n.SchemaVersionV5)
	o.Manifest = m
	x, e = n.AggregateHistogramWindow("window", 110, 130, s, o)
	if e != nil || x[0].Reason != n.HistogramWindowReasonEpochMismatch {
		t.Fatal(x, e)
	}
}

func TestHistogramWindowConcurrentOwnership(t *testing.T) {
	m, s := hwCanonical(t, hwSamples(hwReading(10, 30, []uint64{1, 2}, "delta")), n.SchemaVersionV5)
	o := hwOptions(m, n.HistogramWindowDelta)
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 20; j++ {
				x, e := n.AggregateHistogramWindow("window", 10, 30, s, o)
				if e != nil || *x[0].Histogram.Count != 2 {
					t.Error(x, e)
					return
				}
				x[0].Histogram.Counts[0] = 99
				x[0].Sources[0].Interval.Start = 99
			}
		}()
	}
	wg.Wait()
	if s[0].Readings[0].Observation.Histogram.Counts[0] != 1 {
		t.Fatal("input changed")
	}
}

func TestHistogramWindowIntermediateCDFAndDiscontinuity(t *testing.T) {
	a := hwReading(1, 10, []uint64{5, 10}, "cumulative")
	mid := hwReading(1, 20, []uint64{9, 11}, "cumulative")
	b := hwReading(1, 30, []uint64{10, 20}, "cumulative")
	x := hwAggregate(t, []n.Reading{a, mid, b}, n.HistogramWindowCumulativeDifference)
	if x.Reason != n.HistogramWindowReasonInvalidIntervalCDF {
		t.Fatal(x)
	}
	samples := hwSamples(hwReading(10, 30, []uint64{1, 2}, "delta"))
	samples[0].ClockDiscontinuity = true
	samples[0].ClockDiscontinuityReason = "constructed step"
	m, s := hwCanonical(t, samples, n.SchemaVersionV5)
	out, e := n.AggregateHistogramWindow("window", 10, 30, s, hwOptions(m, n.HistogramWindowDelta))
	if e != nil || out[0].Reason != n.HistogramWindowReasonClockMismatch {
		t.Fatal(out, e)
	}
}
