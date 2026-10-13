package nvtelemetry

import (
	"fmt"
	"math"
	"reflect"
	"sort"

	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
)

// HistogramWindowPolicy selects declared temporal semantics; CDF counts do not select it.
type HistogramWindowPolicy string

const (
	HistogramWindowDelta                HistogramWindowPolicy = "delta"
	HistogramWindowCumulativeDifference HistogramWindowPolicy = "cumulative_difference"
)

// HistogramWindowOptions admits a bounded recorded distribution window. Manifest
// is the completed canonical manifest for all supplied samples. Zero limits
// select 4096 readings, 4096 series and 257 bounds; maxima are 65536, 65536 and 4097.
// At most 65536 samples are admitted, including empty samples. Exact boundary
// arithmetic supports declared known-zero uncertainty only. Mapping is an
// optional caller assertion from local source time to Unix, not calibration.
type HistogramWindowOptions struct {
	Manifest        *Manifest
	Policy          HistogramWindowPolicy
	ClockDomain     string
	ClockEpoch      string
	UnixTimeMapping *UnixTimeMapping
	MaxReadings     int
	MaxSeries       int
	MaxBounds       int
}

// HistogramWindowStatus describes distribution availability, not scalar precision.
type HistogramWindowStatus string

const (
	HistogramWindowAvailable   HistogramWindowStatus = "available"
	HistogramWindowUnavailable HistogramWindowStatus = "unavailable"
	HistogramWindowUnsupported HistogramWindowStatus = "unsupported"
	HistogramWindowRefused     HistogramWindowStatus = "refused"
)

// HistogramWindowReason explains a withheld distribution or scalar boundary.
type HistogramWindowReason string

const (
	HistogramWindowReasonMissingManifest         HistogramWindowReason = "missing_manifest"
	HistogramWindowReasonTemporalUnknown         HistogramWindowReason = "temporal_unknown"
	HistogramWindowReasonTemporalityMismatch     HistogramWindowReason = "temporality_mismatch"
	HistogramWindowReasonMissingInterval         HistogramWindowReason = "missing_interval"
	HistogramWindowReasonMissingClock            HistogramWindowReason = "missing_clock"
	HistogramWindowReasonClockMismatch           HistogramWindowReason = "clock_mismatch"
	HistogramWindowReasonMissingEpoch            HistogramWindowReason = "missing_epoch"
	HistogramWindowReasonEpochMismatch           HistogramWindowReason = "epoch_mismatch"
	HistogramWindowReasonUnknownUncertainty      HistogramWindowReason = "unknown_uncertainty"
	HistogramWindowReasonBoundaryUncertainty     HistogramWindowReason = "boundary_uncertainty"
	HistogramWindowReasonIncompleteStream        HistogramWindowReason = "incomplete_stream"
	HistogramWindowReasonUnknownLoss             HistogramWindowReason = "unknown_loss"
	HistogramWindowReasonLostReadings            HistogramWindowReason = "lost_readings"
	HistogramWindowReasonNoRecordedValue         HistogramWindowReason = "no_recorded_value"
	HistogramWindowReasonSourceUnavailable       HistogramWindowReason = "source_unavailable"
	HistogramWindowReasonMissingBuckets          HistogramWindowReason = "missing_buckets"
	HistogramWindowReasonMissingEndpoint         HistogramWindowReason = "missing_endpoint"
	HistogramWindowReasonGap                     HistogramWindowReason = "gap"
	HistogramWindowReasonOverlap                 HistogramWindowReason = "overlap"
	HistogramWindowReasonStraddlingInterval      HistogramWindowReason = "straddling_interval"
	HistogramWindowReasonDuplicateInterval       HistogramWindowReason = "duplicate_interval"
	HistogramWindowReasonScalarNameCollision     HistogramWindowReason = "scalar_name_collision"
	HistogramWindowReasonMixedValueKinds         HistogramWindowReason = "mixed_value_kinds"
	HistogramWindowReasonMissingIdentity         HistogramWindowReason = "missing_identity"
	HistogramWindowReasonIdentityChanged         HistogramWindowReason = "identity_changed"
	HistogramWindowReasonSourceProfileMismatch   HistogramWindowReason = "source_profile_mismatch"
	HistogramWindowReasonSchemaChanged           HistogramWindowReason = "schema_changed"
	HistogramWindowReasonAggregationEpochChanged HistogramWindowReason = "aggregation_epoch_changed"
	HistogramWindowReasonCounterDecrease         HistogramWindowReason = "counter_decrease"
	HistogramWindowReasonInvalidIntervalCDF      HistogramWindowReason = "invalid_interval_c_d_f"
	HistogramWindowReasonCountOverflow           HistogramWindowReason = "count_overflow"
)

// HistogramWindowFieldStatus describes optional summary presence and arithmetic.
type HistogramWindowFieldStatus string

const (
	HistogramWindowFieldAvailable    HistogramWindowFieldStatus = "available"
	HistogramWindowFieldMissing      HistogramWindowFieldStatus = "missing"
	HistogramWindowFieldEmpty        HistogramWindowFieldStatus = "empty"
	HistogramWindowFieldNotDerivable HistogramWindowFieldStatus = "not_derivable"
	HistogramWindowFieldNonfinite    HistogramWindowFieldStatus = "nonfinite"
	HistogramWindowFieldInconsistent HistogramWindowFieldStatus = "inconsistent"
)

// HistogramWindowCoverageStatus concerns declared intervals, not observer population.
type HistogramWindowCoverageStatus string

const (
	HistogramWindowCompleteDeclaredIntervals HistogramWindowCoverageStatus = "complete_declared_intervals"
	HistogramWindowIncompleteIntervals       HistogramWindowCoverageStatus = "incomplete"
	HistogramWindowUnknownIntervals          HistogramWindowCoverageStatus = "unknown"
)

// HistogramWindowSummary describes exact recorded CDF counts over declared
// intervals. An available Sum is finite float64 arithmetic, not an exact real
// sum. A nil Histogram with a status is not measured zero or a scalar mean.
type HistogramWindowSummary struct {
	WindowName, DeviceUUID, MetricName, Unit string
	Policy                                   HistogramWindowPolicy
	ClockDomain, ClockEpoch                  string
	StartNanos, EndNanos                     int64
	Status                                   HistogramWindowStatus
	Reason                                   HistogramWindowReason
	Histogram                                *Histogram
	SumStatus, MinStatus, MaxStatus          HistogramWindowFieldStatus
	InputCount                               int
	CoveredDurationNanos                     int64
	CoverageStatus                           HistogramWindowCoverageStatus
	ObserverCoverage                         *SamplingCoverage
	Sources                                  []HistogramWindowSource
}

// HistogramWindowSource owns original point lineage. Locators are not clock or
// physical identity proof. A mapping and calibration are caller assertions.
type HistogramWindowSource struct {
	SampleSeq                                              uint64
	ReadingIndex                                           int
	Interval                                               *tracecontract.Interval
	ClockEpoch                                             string
	SourceClass, Producer, ProducerVersion, ProducerSchema string
	RawSHA256, Transform, TransformSHA256                  string
	Evidence, UnavailableReason                            string
	ReadingSource                                          *ReadingSource
	AppliedUnixTimeMapping                                 *UnixTimeMapping
	Observer                                               Observer
	OTLPClock                                              *OTLPClockMetadata
	OTLPSourceTime                                         *OTLPSourceTime
	UncertaintyNS                                          *int64
}

type histogramWindowPoint struct {
	reading       Reading
	seq           uint64
	index         int
	start, end    int64
	discontinuity bool
}
type histogramWindowKey struct{ device, name string }

// AggregateHistogramWindow accumulates V5 delta intervals or differences V5
// cumulative endpoints in [startNanos,endNanos). It never substitutes collector
// query time, interpolates buckets, derives cumulative extrema, or estimates
// individual latency. V3/V4 temporal semantics are unsupported. A recovered
// reset between observations remains unobservable. Supplied samples must be
// the complete decoded manifest population; host fixtures do not authenticate
// a producer, calibration, observer coverage or physical execution.
func AggregateHistogramWindow(name string, start, end int64, samples []Sample, opts HistogramWindowOptions) ([]HistogramWindowSummary, error) {
	if !exportText(name) || name == "" || !exportText(opts.ClockDomain) || !exportText(opts.ClockEpoch) || start < 0 || end <= start || start > math.MaxInt64-(end-start) || opts.ClockDomain == "" || opts.ClockEpoch == "" || opts.ClockDomain == "unix" && start == 0 || opts.Policy != HistogramWindowDelta && opts.Policy != HistogramWindowCumulativeDifference {
		return nil, fmt.Errorf("histogram window: invalid window or clock policy")
	}
	limits := []struct {
		p        *int
		def, max int
	}{{&opts.MaxReadings, 4096, 65536}, {&opts.MaxSeries, 4096, 65536}, {&opts.MaxBounds, 257, 4097}}
	for _, l := range limits {
		n, e := histogramLimit(*l.p, l.def, l.max)
		if e != nil {
			return nil, fmt.Errorf("histogram window: %w", e)
		}
		*l.p = n
	}
	if len(samples) > 65536 {
		return nil, fmt.Errorf("histogram window: sample limit")
	}
	version := 0
	if opts.Manifest != nil {
		version = opts.Manifest.SchemaVersion
		if version < 1 || version > SchemaVersionV5 {
			return nil, fmt.Errorf("histogram window: unsupported schema version")
		}
		if e := validateVersionedRecord(StreamRecord{Manifest: opts.Manifest}, version); e != nil {
			return nil, e
		}
	}
	groups := map[histogramWindowKey][]histogramWindowPoint{}
	candidates := map[histogramWindowKey]bool{}
	families := map[string]uint64{}
	seqs := map[uint64]bool{}
	total := 0
	for _, sample := range samples {
		if len(sample.Readings) > opts.MaxReadings-total {
			return nil, fmt.Errorf("histogram window: reading limit")
		}
		total += len(sample.Readings)
		if seqs[sample.Seq] {
			return nil, fmt.Errorf("histogram window: duplicate sample sequence")
		}
		seqs[sample.Seq] = true
		if version != 0 {
			if e := validateVersionedRecord(StreamRecord{Sample: &sample}, version); e != nil {
				return nil, e
			}
		} else if e := boundedMetadata(reflect.ValueOf(sample), 0, true); e != nil {
			return nil, e
		}
		for i, r := range sample.Readings {
			k := histogramWindowKey{r.DeviceUUID, r.Name}
			o := r.Observation
			if o != nil {
				if families[o.Family] == math.MaxUint64 {
					return nil, fmt.Errorf("histogram window: family count overflow")
				}
				families[o.Family]++
				if o.Kind == "histogram" || o.Histogram != nil {
					candidates[k] = true
				}
				if o.Histogram != nil && len(o.Histogram.Bounds) > opts.MaxBounds {
					return nil, fmt.Errorf("histogram window: bound limit")
				}
			}
			groups[k] = append(groups[k], histogramWindowPoint{reading: r, seq: sample.Seq, index: i, discontinuity: sample.ClockDiscontinuity})
		}
	}
	if len(groups) > opts.MaxSeries {
		return nil, fmt.Errorf("histogram window: series limit")
	}
	if opts.Manifest != nil && opts.Manifest.IsComplete {
		if opts.Manifest.TotalSamples != uint64(len(samples)) {
			return nil, fmt.Errorf("histogram window: sample accounting mismatch")
		}
		for f, l := range opts.Manifest.LostRecordsByFamily {
			if !l.Unknown && (l.Dropped > l.Total || l.Total-l.Dropped != families[f]) {
				return nil, fmt.Errorf("histogram window: family accounting mismatch")
			}
		}
	}
	keys := make([]histogramWindowKey, 0, len(candidates))
	for k := range candidates {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].device == keys[j].device {
			return keys[i].name < keys[j].name
		}
		return keys[i].device < keys[j].device
	})
	out := make([]HistogramWindowSummary, 0, len(keys))
	for _, k := range keys {
		x := HistogramWindowSummary{WindowName: name, DeviceUUID: k.device, MetricName: k.name, Policy: opts.Policy, ClockDomain: opts.ClockDomain, ClockEpoch: opts.ClockEpoch, StartNanos: start, EndNanos: end, CoverageStatus: HistogramWindowUnknownIntervals, ObserverCoverage: &SamplingCoverage{Status: "unknown", Reason: "aggregate_population_coverage_unassessed"}}
		points := groups[k]
		if len(points) > 0 {
			x.Unit = points[0].reading.Unit
		}
		reason := aggregateHistogramGroup(&x, start, end, points, opts, version)
		if reason != "" {
			x.Histogram = nil
			x.SumStatus, x.MinStatus, x.MaxStatus = "", "", ""
			x.Reason = reason
			x.Status = histogramReasonStatus(reason)
		} else {
			x.Status = HistogramWindowAvailable
			x.CoverageStatus = HistogramWindowCompleteDeclaredIntervals
			x.CoveredDurationNanos = end - start
		}
		out = append(out, x)
	}
	return out, nil
}

func histogramReasonStatus(r HistogramWindowReason) HistogramWindowStatus {
	switch r {
	case HistogramWindowReasonTemporalUnknown, HistogramWindowReasonSourceProfileMismatch:
		return HistogramWindowUnsupported
	case HistogramWindowReasonMissingManifest, HistogramWindowReasonMissingInterval, HistogramWindowReasonMissingClock, HistogramWindowReasonMissingEpoch, HistogramWindowReasonUnknownUncertainty, HistogramWindowReasonIncompleteStream, HistogramWindowReasonUnknownLoss, HistogramWindowReasonLostReadings, HistogramWindowReasonNoRecordedValue, HistogramWindowReasonSourceUnavailable, HistogramWindowReasonMissingBuckets, HistogramWindowReasonMissingEndpoint, HistogramWindowReasonGap, HistogramWindowReasonMissingIdentity:
		return HistogramWindowUnavailable
	default:
		return HistogramWindowRefused
	}
}

func histogramSource(p histogramWindowPoint, m *UnixTimeMapping) HistogramWindowSource {
	r := cloneReadings([]Reading{p.reading})[0]
	o := r.Observation
	s := HistogramWindowSource{SampleSeq: p.seq, ReadingIndex: p.index, Evidence: r.Evidence, UnavailableReason: r.UnavailableReason, ReadingSource: r.Source}
	if o != nil {
		s.Interval = o.Interval
		s.ClockEpoch = o.ClockEpoch
		s.SourceClass = o.SourceClass
		s.Producer = o.Producer
		s.ProducerVersion = o.ProducerVersion
		s.ProducerSchema = o.ProducerSchema
		s.RawSHA256 = o.RawSHA256
		s.Transform = o.Transform
		s.TransformSHA256 = o.TransformSHA256
		s.Observer = o.Observer
		s.OTLPClock = o.OTLPClock
		s.OTLPSourceTime = o.OTLPSourceTime
		s.UncertaintyNS = o.UncertaintyNS
	}
	if m != nil {
		copy := *m
		s.AppliedUnixTimeMapping = &copy
	}
	return s
}

// identity excludes per-point values and time bounds, without discarding
// subject/workload/associated IDs or observer mode/pass identity.
func histogramIdentity(r Reading) *Observation {
	o := cloneReadings([]Reading{r})[0].Observation
	if o == nil {
		return nil
	}
	o.Histogram = nil
	o.Interval = nil
	o.RawSHA256 = ""
	o.TransformSHA256 = ""
	o.OTLPSourceTime = nil
	o.UncertaintyNS = nil
	o.CounterEpoch = ""
	o.Observer.SamplingCoverage = nil
	if a := o.Assignment; a != nil {
		a.StartNanos = 0
		a.EndNanos = 0
		for i := range a.Provenance {
			a.Provenance[i].StartNanos = 0
			a.Provenance[i].EndNanos = 0
		}
	}
	return o
}

func histogramPointTime(p *histogramWindowPoint, opts HistogramWindowOptions) HistogramWindowReason {
	r := p.reading
	o := r.Observation
	if p.discontinuity {
		return HistogramWindowReasonClockMismatch
	}
	if o == nil || o.Interval == nil {
		return HistogramWindowReasonMissingInterval
	}
	i := o.Interval
	if i.Clock == "" {
		return HistogramWindowReasonMissingClock
	}
	if o.ClockEpoch == "" {
		return HistogramWindowReasonMissingEpoch
	}
	p.start, p.end = i.Start, i.End
	if a := o.Assignment; a != nil {
		clock, epoch, lo, hi := i.Clock, o.ClockEpoch, i.Start, i.End
		if original := o.OTLPSourceTime; original != nil && original.SourceIntervalPresent {
			clock = original.SourceIntervalClock
			epoch = original.SourceClockEpoch
			lo = original.SourceIntervalStartNanos
			hi = original.SourceIntervalEndNanos
		}
		if a.ClockDomain != clock || a.ClockEpoch != epoch {
			return HistogramWindowReasonClockMismatch
		}
		if a.StartNanos > lo || a.EndNanos < hi {
			return HistogramWindowReasonIdentityChanged
		}
	}
	if o.SourceClass == "imported" {
		if opts.UnixTimeMapping != nil {
			return HistogramWindowReasonClockMismatch
		}
		if opts.ClockDomain != "unix" || opts.ClockEpoch != "unix" || i.Clock != "unix" || o.ClockEpoch != "unix" || r.TimestampUnix != i.End {
			return HistogramWindowReasonClockMismatch
		}
		c := o.OTLPClock
		if c == nil || !c.HasClockCalibration {
			return HistogramWindowReasonUnknownUncertainty
		}
		if c.ClockCalibrationBoundsNanos != 0 {
			return HistogramWindowReasonBoundaryUncertainty
		}
	} else {
		if o.UncertaintyNS == nil {
			return HistogramWindowReasonUnknownUncertainty
		}
		if r.TimestampMonotonic != i.End {
			return HistogramWindowReasonClockMismatch
		}
		if opts.Manifest.ClockMonotonicDomain != i.Clock || opts.Manifest.ClockMonotonicEpochID != o.ClockEpoch {
			return HistogramWindowReasonClockMismatch
		}
		m := opts.UnixTimeMapping
		if m == nil {
			if i.Clock != opts.ClockDomain {
				return HistogramWindowReasonClockMismatch
			}
			if o.ClockEpoch != opts.ClockEpoch {
				return HistogramWindowReasonEpochMismatch
			}
		} else {
			if opts.ClockDomain != "unix" || opts.ClockEpoch != "unix" || m.SourceClockDomain != i.Clock || m.SourceClockEpoch != o.ClockEpoch || m.Provenance == "" || !exportText(m.Provenance) || !exportText(m.SourceClockDomain) || !exportText(m.SourceClockEpoch) || m.SourceValidFromNanos > m.SourceValidUntilNanos || i.Start < m.SourceValidFromNanos || i.End > m.SourceValidUntilNanos {
				return HistogramWindowReasonClockMismatch
			}
			if !opts.Manifest.HasClockCalibration {
				return HistogramWindowReasonUnknownUncertainty
			}
			if opts.Manifest.ClockCalibrationBoundsNanos != 0 {
				return HistogramWindowReasonBoundaryUncertainty
			}
			if opts.Manifest.ClockMappingValidFrom > opts.Manifest.ClockMappingValidUntil || i.Start < opts.Manifest.ClockMappingValidFrom || i.End > opts.Manifest.ClockMappingValidUntil {
				return HistogramWindowReasonClockMismatch
			}
			var e error
			p.start, e = checkedUnixTime(i.Start, m.UnixOffsetNanos)
			if e != nil {
				return HistogramWindowReasonClockMismatch
			}
			p.end, e = checkedUnixTime(i.End, m.UnixOffsetNanos)
			if e != nil || r.TimestampUnix != 0 && r.TimestampUnix != p.end {
				return HistogramWindowReasonClockMismatch
			}
		}
	}
	if o.UncertaintyNS != nil && *o.UncertaintyNS != 0 || r.WallClockUncertaintyNanos != 0 {
		return HistogramWindowReasonBoundaryUncertainty
	}
	return ""
}

func aggregateHistogramGroup(x *HistogramWindowSummary, start, end int64, points []histogramWindowPoint, opts HistogramWindowOptions, version int) HistogramWindowReason {
	x.InputCount = len(points)
	for _, p := range points {
		x.Sources = append(x.Sources, histogramSource(p, opts.UnixTimeMapping))
	}
	if opts.Manifest == nil {
		return HistogramWindowReasonMissingManifest
	}
	if version < SchemaVersionV5 {
		return HistogramWindowReasonTemporalUnknown
	}
	if !opts.Manifest.IsComplete {
		return HistogramWindowReasonIncompleteStream
	}
	if _, found := LookupMetricKind(x.MetricName); found {
		return HistogramWindowReasonScalarNameCollision
	}
	var selected []histogramWindowPoint
	for _, p := range points {
		o := p.reading.Observation
		if o == nil || o.Kind != "histogram" || p.reading.Number != nil {
			return HistogramWindowReasonMixedValueKinds
		}
		if reason := histogramPointTime(&p, opts); reason != "" {
			return reason
		}
		if opts.Policy == HistogramWindowDelta {
			if p.end <= start || p.start >= end {
				continue
			}
		} else if p.end < start || p.end > end {
			continue
		}
		selected = append(selected, p)
	}
	if len(selected) == 0 {
		if opts.Policy == HistogramWindowCumulativeDifference {
			return HistogramWindowReasonMissingEndpoint
		}
		return HistogramWindowReasonGap
	}
	first := selected[0].reading
	base := first.Observation
	if first.DeviceUUID == "" || first.Name == "" || first.Unit == "" || first.Backend == "" || base.Family == "" || base.SourceClass == "" || base.Producer == "" || base.ProducerVersion == "" || base.ProducerSchema == "" {
		return HistogramWindowReasonMissingIdentity
	}
	identity := histogramIdentity(first)
	h := base.Histogram
	for _, p := range selected {
		r := p.reading
		o := r.Observation
		loss, ok := opts.Manifest.LostRecordsByFamily[o.Family]
		if !ok || loss.Unknown {
			return HistogramWindowReasonUnknownLoss
		}
		if loss.Dropped != 0 {
			return HistogramWindowReasonLostReadings
		}
		if r.Source != nil {
			return HistogramWindowReasonSourceProfileMismatch
		}
		if r.Unit != first.Unit {
			return HistogramWindowReasonSchemaChanged
		}
		if r.Backend != first.Backend || !reflect.DeepEqual(identity, histogramIdentity(r)) {
			return HistogramWindowReasonIdentityChanged
		}
		if !r.IsAvailable() {
			if o.Histogram != nil && o.Histogram.Flags == 1 {
				return HistogramWindowReasonNoRecordedValue
			}
			return HistogramWindowReasonSourceUnavailable
		}
		q := o.Histogram
		if q == nil {
			return HistogramWindowReasonMissingBuckets
		}
		if h == nil {
			h = q
		}
		if !reflect.DeepEqual(h.Bounds, q.Bounds) {
			return HistogramWindowReasonSchemaChanged
		}
		if q.Count == nil || len(q.Bounds) == 0 && *q.Count != 0 {
			return HistogramWindowReasonMissingBuckets
		}
		expected := "delta"
		if opts.Policy == HistogramWindowCumulativeDifference {
			expected = "cumulative"
		}
		if q.Temporality != expected {
			return HistogramWindowReasonTemporalityMismatch
		}
		bs, qs := base.OTLPSourceTime, o.OTLPSourceTime
		if (bs == nil) != (qs == nil) {
			return HistogramWindowReasonClockMismatch
		}
		if bs != nil {
			if bs.SourceClockEpoch != qs.SourceClockEpoch || bs.MappingEpoch != qs.MappingEpoch {
				return HistogramWindowReasonEpochMismatch
			}
			if bs.SourceClass != qs.SourceClass || bs.MappingDomain != qs.MappingDomain || bs.MappingUnixOffsetNanos != qs.MappingUnixOffsetNanos || bs.MappingProvenance != qs.MappingProvenance || bs.SourceIntervalClock != qs.SourceIntervalClock {
				return HistogramWindowReasonClockMismatch
			}
		}
		if o.CounterEpoch != base.CounterEpoch {
			return HistogramWindowReasonEpochMismatch
		}
	}
	sort.Slice(selected, func(i, j int) bool {
		if selected[i].end == selected[j].end {
			return selected[i].start < selected[j].start
		}
		return selected[i].end < selected[j].end
	})
	count := uint64(0)
	result := &Histogram{Bounds: append([]string(nil), h.Bounds...), Counts: make([]uint64, len(h.Counts)), Count: &count, Temporality: "delta"}
	x.SumStatus = HistogramWindowFieldMissing
	x.MinStatus = HistogramWindowFieldMissing
	x.MaxStatus = HistogramWindowFieldMissing
	if opts.Policy == HistogramWindowDelta {
		cursor := start
		for i, p := range selected {
			if p.start < start || p.end > end {
				return HistogramWindowReasonStraddlingInterval
			}
			if p.start >= p.end {
				return HistogramWindowReasonOverlap
			}
			if i > 0 && p.start == selected[i-1].start && p.end == selected[i-1].end {
				return HistogramWindowReasonDuplicateInterval
			}
			if p.start < cursor {
				return HistogramWindowReasonOverlap
			}
			if p.start > cursor {
				return HistogramWindowReasonGap
			}
			cursor = p.end
			q := p.reading.Observation.Histogram
			if *q.Count > math.MaxUint64-count {
				return HistogramWindowReasonCountOverflow
			}
			count += *q.Count
			for j, n := range q.Counts {
				if n > math.MaxUint64-result.Counts[j] {
					return HistogramWindowReasonCountOverflow
				}
				result.Counts[j] += n
			}
		}
		if cursor != end {
			return HistogramWindowReasonGap
		}
		histogramDeltaSummaries(x, result, selected)
	} else {
		if selected[0].end != start || selected[len(selected)-1].end != end {
			return HistogramWindowReasonMissingEndpoint
		}
		origin := selected[0].start
		if origin > start {
			return HistogramWindowReasonAggregationEpochChanged
		}
		for i, p := range selected {
			if p.start != origin {
				return HistogramWindowReasonAggregationEpochChanged
			}
			if i == 0 {
				continue
			}
			if p.end == selected[i-1].end {
				return HistogramWindowReasonDuplicateInterval
			}
			q := p.reading.Observation.Histogram
			prev := selected[i-1].reading.Observation.Histogram
			if *q.Count < *prev.Count {
				return HistogramWindowReasonCounterDecrease
			}
			for j, n := range q.Counts {
				if n < prev.Counts[j] {
					return HistogramWindowReasonCounterDecrease
				}
			}
			var previousDelta uint64
			for j, n := range q.Counts {
				delta := n - prev.Counts[j]
				if j > 0 && delta < previousDelta {
					return HistogramWindowReasonInvalidIntervalCDF
				}
				previousDelta = delta
			}
		}
		before := selected[0].reading.Observation.Histogram
		after := selected[len(selected)-1].reading.Observation.Histogram
		count = *after.Count - *before.Count
		for i, n := range after.Counts {
			result.Counts[i] = n - before.Counts[i]
		}
		if before.Sum != nil && after.Sum != nil {
			sum := *after.Sum - *before.Sum
			result.Sum = &sum
			x.SumStatus = HistogramWindowFieldAvailable
		}
		x.MinStatus = HistogramWindowFieldNotDerivable
		x.MaxStatus = HistogramWindowFieldNotDerivable
	}
	if len(result.Counts) > 0 && result.Counts[len(result.Counts)-1] != count {
		return HistogramWindowReasonInvalidIntervalCDF
	}
	for i := 1; i < len(result.Counts); i++ {
		if result.Counts[i] < result.Counts[i-1] {
			return HistogramWindowReasonInvalidIntervalCDF
		}
	}
	if result.Sum != nil {
		if math.IsNaN(*result.Sum) || math.IsInf(*result.Sum, 0) {
			result.Sum = nil
			x.SumStatus = HistogramWindowFieldNonfinite
		} else if count == 0 && *result.Sum != 0 {
			result.Sum = nil
			x.SumStatus = HistogramWindowFieldInconsistent
		}
	}
	if len(selected) == 1 {
		coverage := cloneReadings([]Reading{selected[0].reading})[0].Observation.Observer.SamplingCoverage
		if coverage != nil {
			x.ObserverCoverage = coverage
		}
	}
	x.Histogram = result
	return ""
}

func histogramDeltaSummaries(x *HistogramWindowSummary, h *Histogram, points []histogramWindowPoint) {
	sum := float64(0)
	sumKnown := true
	minKnown, maxKnown := true, true
	var min, max *float64
	for _, p := range points {
		q := p.reading.Observation.Histogram
		if q.Sum == nil {
			sumKnown = false
		} else {
			sum += *q.Sum
		}
		if *q.Count == 0 {
			continue
		}
		if q.Min == nil {
			minKnown = false
		} else if min == nil || *q.Min < *min {
			v := *q.Min
			min = &v
		}
		if q.Max == nil {
			maxKnown = false
		} else if max == nil || *q.Max > *max {
			v := *q.Max
			max = &v
		}
	}
	if sumKnown {
		h.Sum = &sum
		x.SumStatus = HistogramWindowFieldAvailable
	}
	if *h.Count == 0 {
		x.MinStatus = HistogramWindowFieldEmpty
		x.MaxStatus = HistogramWindowFieldEmpty
		return
	}
	if minKnown {
		h.Min = min
		x.MinStatus = HistogramWindowFieldAvailable
	}
	if maxKnown {
		h.Max = max
		x.MaxStatus = HistogramWindowFieldAvailable
	}
}
