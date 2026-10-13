package nvtelemetry

import (
	"math"
	"math/bits"
	"sort"
	"time"
)

// WindowOptions controls aggregation constraints.
type WindowOptions struct {
	MaxAllowedGapNanos int64 // maximum acceptable gap between samples before flagging coverage gap
}

// SamplePoint records a typed sample for window aggregation.
type SamplePoint struct {
	Timestamp int64
	Value     float64
	IntValue  uint64
	IsInt     bool
	Available bool
	Unit      string
}

// AggregateWindow computes time-weighted summaries for all device metrics within the window [startMono, endMono].
func AggregateWindow(windowName string, startMono, endMono int64, samples []Sample, opts WindowOptions) ([]WindowSummary, []EnergySummary) {
	windowDuration := endMono - startMono
	if windowDuration <= 0 {
		return nil, nil
	}

	type deviceMetricKey struct {
		deviceUUID string
		metricName string
	}

	grouped := make(map[deviceMetricKey][]windowPoint)

	for _, s := range samples {
		t := s.QueryStartNanos
		if t < startMono || t > endMono {
			continue
		}
		for _, r := range s.Readings {
			key := deviceMetricKey{deviceUUID: r.DeviceUUID, metricName: r.Name}
			grouped[key] = append(grouped[key], readingPoint(r, t))
		}
	}

	var summaries []WindowSummary
	var energySummaries []EnergySummary

	// Sort keys for deterministic output
	var keys []deviceMetricKey
	for k := range grouped {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].deviceUUID == keys[j].deviceUUID {
			return keys[i].metricName < keys[j].metricName
		}
		return keys[i].deviceUUID < keys[j].deviceUUID
	})

	for _, k := range keys {
		wpoints := grouped[k]
		if len(wpoints) == 0 {
			continue
		}

		// Reject numeric averaging of bitmask and event metrics.
		// Bitmasks and discrete error events must remain discrete evidence
		// and must not flow through generic arithmetic or time-weighted averages.
		if kind, found := LookupMetricKind(k.metricName); found && (kind == KindBitmask || kind == KindEvent) {
			continue
		}

		// Sort points chronologically
		sort.Slice(wpoints, func(i, j int) bool {
			return wpoints[i].Timestamp < wpoints[j].Timestamp
		})
		points := make([]SamplePoint, len(wpoints))
		meta := make([]counterMeta, len(wpoints))
		for i, p := range wpoints {
			points[i], meta[i] = p.SamplePoint, p.meta
		}

		unit := points[0].Unit

		// Calculate arithmetic stats across valid points
		var validCount int
		var arithSum float64
		minVal := math.MaxFloat64
		maxVal := -math.MaxFloat64
		var firstValid, lastValid *SamplePoint
		sampleValuesExact := true

		for i := range points {
			p := &points[i]
			if p.Available {
				validCount++
				if wpoints[i].inexactInteger {
					sampleValuesExact = false
				}
				arithSum += p.Value
				if p.Value < minVal {
					minVal = p.Value
				}
				if p.Value > maxVal {
					maxVal = p.Value
				}
				if firstValid == nil {
					firstValid = p
				}
				lastValid = p
			}
		}

		if validCount == 0 {
			continue
		}

		arithMean := arithSum / float64(validCount)

		// Time-weighted integration without bridging unavailable gaps
		var weightedSum float64
		var coveredNanos int64
		var maxGap int64

		// Boundary gaps
		if startGap := points[0].Timestamp - startMono; startGap > 0 {
			if startGap > maxGap {
				maxGap = startGap
			}
		}
		if endGap := endMono - points[len(points)-1].Timestamp; endGap > 0 {
			if endGap > maxGap {
				maxGap = endGap
			}
		}

		for i := 0; i < len(points)-1; i++ {
			p1 := points[i]
			p2 := points[i+1]
			dt := p2.Timestamp - p1.Timestamp
			if dt <= 0 {
				continue
			}
			if dt > maxGap {
				maxGap = dt
			}

			// An interval is ONLY covered if BOTH endpoints are available
			// and interval does not exceed MaxAllowedGapNanos when configured.
			if !p1.Available || !p2.Available {
				// Unavailable gap: do NOT bridge across it!
				continue
			}
			if opts.MaxAllowedGapNanos > 0 && dt > opts.MaxAllowedGapNanos {
				// Excessive gap: do NOT count as covered!
				continue
			}

			avgVal := (p1.Value + p2.Value) / 2.0
			weightedSum += avgVal * float64(dt)
			coveredNanos += dt
		}

		var timeWeighted float64
		if coveredNanos > 0 {
			timeWeighted = weightedSum / float64(coveredNanos)
		} else {
			timeWeighted = arithMean
		}

		coverageRatio := float64(coveredNanos) / float64(windowDuration)
		if coverageRatio > 1.0 {
			coverageRatio = 1.0
		}

		summary := WindowSummary{
			WindowName:           windowName,
			DeviceUUID:           k.deviceUUID,
			MetricName:           k.metricName,
			Unit:                 unit,
			TimeWeightedMean:     timeWeighted,
			ArithmeticMean:       arithMean,
			Min:                  minVal,
			Max:                  maxVal,
			StartValue:           firstValid.Value,
			EndValue:             lastValid.Value,
			CoveredDurationNanos: coveredNanos,
			TotalDurationNanos:   windowDuration,
			CoverageRatio:        coverageRatio,
			MaxGapNanos:          maxGap,
			SampleCount:          validCount,
		}
		if !sampleValuesExact {
			summary.Min = 0
			summary.Max = 0
			summary.StartValue = 0
			summary.EndValue = 0
			summary.SampleValuesStatus = WindowSampleValuesUnavailable
			summary.SampleValuesReason = ReasonIntegerPrecision
		}
		for _, point := range wpoints {
			if point.histogramBoundary {
				summary.HistogramBoundaryCount++
				summary.HistogramBoundaryReason = HistogramWindowReasonMixedValueKinds
				if _, found := LookupMetricKind(k.metricName); found {
					summary.HistogramBoundaryReason = HistogramWindowReasonScalarNameCollision
				}
			}
		}
		summaries = append(summaries, summary)

		// Energy calculation for cumulative energy counters
		if k.metricName == MetricTotalEnergyMJ {
			eSummary := cumulativeEnergy(windowName, k.deviceUUID, points, meta, startMono, endMono, opts)
			energySummaries = append(energySummaries, eSummary)
		}
	}

	return summaries, energySummaries
}

// windowPoint is a SamplePoint with its counter metadata.
type windowPoint struct {
	SamplePoint
	meta              counterMeta
	inexactInteger    bool
	histogramBoundary bool
}

// counterMeta holds the V3 metadata that decides whether a reading can serve
// as a cumulative energy reading. It is zero for legacy readings.
type counterMeta struct {
	v3    bool   // value came from the V3 Number arm
	kind  string // observation kind
	epoch string // counter epoch
	reset bool   // counter reset at this reading
}

// readingPoint converts r, observed at t, to a window point. Legacy readings
// use Value and IntValue. V3 readings use the Number arm: Uint and nonnegative
// Int keep their exact integer, and a reading with no Number (unavailable or
// histogram-only) is unavailable rather than zero. A negative energy counter
// is unavailable.
func readingPoint(r Reading, t int64) windowPoint {
	p := windowPoint{SamplePoint: SamplePoint{Timestamp: t, Available: r.IsAvailable(), Unit: r.Unit}}
	if r.Number == nil && r.Observation == nil {
		p.Value = r.Value
		p.inexactInteger = r.IsInt && !uint64ExactlyRepresentableInFloat64(r.IntValue)
		if r.IsInt && p.Value == 0 && r.IntValue > 0 {
			p.Value = float64(r.IntValue)
		}
		p.IntValue = r.IntValue
		p.IsInt = r.IsInt
		return p
	}
	p.histogramBoundary = r.Observation != nil && (r.Observation.Kind == "histogram" || r.Observation.Histogram != nil)
	p.meta.v3 = true
	if o := r.Observation; o != nil {
		p.meta.kind = o.Kind
		p.meta.epoch = o.CounterEpoch
		p.meta.reset = o.CounterReset
	}
	n := r.Number
	switch {
	case n == nil:
		p.Available = false
	case n.Uint != nil:
		p.IsInt = true
		p.IntValue = *n.Uint
		p.inexactInteger = !uint64ExactlyRepresentableInFloat64(*n.Uint)
		p.Value = float64(*n.Uint)
	case n.Int != nil:
		p.inexactInteger = !int64ExactlyRepresentableInFloat64(*n.Int)
		p.Value = float64(*n.Int)
		if *n.Int >= 0 {
			p.IsInt = true
			p.IntValue = uint64(*n.Int)
		} else if r.Name == MetricTotalEnergyMJ {
			p.Available = false
		}
	case n.Float != nil:
		p.Value = *n.Float
	default:
		p.Available = false
	}
	return p
}

func uint64ExactlyRepresentableInFloat64(value uint64) bool {
	length := bits.Len64(value)
	return length <= 53 || bits.TrailingZeros64(value) >= length-53
}

func int64ExactlyRepresentableInFloat64(value int64) bool {
	if value >= 0 {
		return uint64ExactlyRepresentableInFloat64(uint64(value))
	}
	magnitude := uint64(-(value + 1)) + 1
	return uint64ExactlyRepresentableInFloat64(magnitude)
}

// computeCumulativeEnergy derives energy from legacy cumulative counter points.
func computeCumulativeEnergy(windowName, deviceUUID string, points []SamplePoint, startMono, endMono int64, opts WindowOptions) EnergySummary {
	return cumulativeEnergy(windowName, deviceUUID, points, nil, startMono, endMono, opts)
}

// cumulativeEnergy derives energy from a cumulative counter. Meta, if non-nil,
// parallels points. A V3 reading counts only when its observation kind is
// counter with a nonempty counter epoch; a V3 counter reset or epoch change
// is reported as a reset, never subtracted across.
func cumulativeEnergy(windowName, deviceUUID string, points []SamplePoint, meta []counterMeta, startMono, endMono int64, opts WindowOptions) EnergySummary {
	if meta == nil {
		meta = make([]counterMeta, len(points))
	}
	windowDuration := endMono - startMono
	if len(points) < 2 {
		return EnergySummary{
			StartMonoNanos: startMono, EndMonoNanos: endMono,
			DeviceUUID:         deviceUUID,
			WindowName:         windowName,
			Method:             "cumulative_counter",
			TotalDurationNanos: windowDuration,
			Valid:              false,
			Reason:             "insufficient_samples",
		}
	}

	for i, p := range points {
		if p.Available && (!p.IsInt || meta[i].v3 && meta[i].kind != "counter") {
			return EnergySummary{WindowName: windowName, DeviceUUID: deviceUUID, Method: "cumulative_counter", Reason: "invalid_counter_type"}
		}
	}
	// Without a counter epoch, readings cannot be shown to share one counter.
	for i, p := range points {
		if p.Available && meta[i].v3 && meta[i].epoch == "" {
			return EnergySummary{
				StartMonoNanos: startMono, EndMonoNanos: endMono,
				DeviceUUID:         deviceUUID,
				WindowName:         windowName,
				Method:             "cumulative_counter",
				TotalDurationNanos: windowDuration,
				Valid:              false,
				Reason:             "missing_counter_epoch",
			}
		}
	}

	start := points[0]
	end := points[len(points)-1]

	if !start.Available || !end.Available {
		return EnergySummary{
			StartMonoNanos: startMono, EndMonoNanos: endMono,
			DeviceUUID:         deviceUUID,
			WindowName:         windowName,
			Method:             "cumulative_counter",
			TotalDurationNanos: windowDuration,
			Valid:              false,
			Reason:             ReasonBlank,
		}
	}

	covered := end.Timestamp - start.Timestamp
	coverage := float64(covered) / float64(windowDuration)
	if coverage > 1.0 {
		coverage = 1.0
	}

	// Check for gaps and resets across the sequence
	// A V3 stream compares each available reading with the previous
	// available one, so an unavailable reading is never read as zero.
	v3 := meta[0].v3
	resetDetected := v3 && meta[0].reset
	var gapExceeded bool
	prev := start
	for i := 0; i < len(points)-1 && !resetDetected; i++ {
		p1 := points[i]
		p2 := points[i+1]
		if !p1.Available || !p2.Available {
			gapExceeded = true
		}
		dt := p2.Timestamp - p1.Timestamp
		if opts.MaxAllowedGapNanos > 0 && dt > opts.MaxAllowedGapNanos {
			gapExceeded = true
		}
		if !v3 {
			if !p1.Available || !p2.Available {
				continue
			}
			resetDetected = p2.IntValue < p1.IntValue
			continue
		}
		if !p2.Available {
			continue
		}
		m := meta[i+1]
		resetDetected = p2.IntValue < prev.IntValue || m.reset || m.epoch != meta[0].epoch
		prev = p2
	}

	if resetDetected {
		return EnergySummary{
			StartMonoNanos: startMono, EndMonoNanos: endMono,
			DeviceUUID:           deviceUUID,
			WindowName:           windowName,
			Method:               "cumulative_counter",
			StartEnergyMJ:        start.IntValue,
			EndEnergyMJ:          end.IntValue,
			CoveredDurationNanos: covered,
			TotalDurationNanos:   windowDuration,
			CoverageRatio:        coverage,
			ResetDetected:        true,
			Valid:                false,
			Reason:               ReasonCounterReset,
		}
	}

	if gapExceeded {
		return EnergySummary{
			StartMonoNanos: startMono, EndMonoNanos: endMono,
			DeviceUUID:           deviceUUID,
			WindowName:           windowName,
			Method:               "cumulative_counter",
			StartEnergyMJ:        start.IntValue,
			EndEnergyMJ:          end.IntValue,
			CoveredDurationNanos: covered,
			TotalDurationNanos:   windowDuration,
			CoverageRatio:        coverage,
			Valid:                false,
			Reason:               ReasonGapExceeded,
		}
	}

	deltaMJ := end.IntValue - start.IntValue
	deltaJoules := float64(deltaMJ) / 1000.0

	return EnergySummary{
		StartMonoNanos: startMono, EndMonoNanos: endMono,
		DeviceUUID:           deviceUUID,
		WindowName:           windowName,
		Method:               "cumulative_counter",
		StartEnergyMJ:        start.IntValue,
		EndEnergyMJ:          end.IntValue,
		EnergyConsumedMJ:     deltaMJ,
		EnergyConsumedJoules: deltaJoules,
		CoveredDurationNanos: covered,
		TotalDurationNanos:   windowDuration,
		CoverageRatio:        coverage,
		ResetDetected:        false,
		Valid:                true,
	}
}

// IntegratePowerToEnergy computes energy by integrating power over time as a fallback.
func IntegratePowerToEnergy(windowName, deviceUUID string, powerReadings []SamplePoint, startMono, endMono int64, opts WindowOptions) EnergySummary {
	windowDuration := endMono - startMono
	if windowDuration <= 0 {
		return EnergySummary{
			StartMonoNanos: startMono, EndMonoNanos: endMono, WindowName: windowName, DeviceUUID: deviceUUID, Reason: "invalid_window_bounds"}
	}
	if len(powerReadings) < 2 {
		return EnergySummary{
			StartMonoNanos: startMono, EndMonoNanos: endMono,
			DeviceUUID:         deviceUUID,
			WindowName:         windowName,
			Method:             "integrated_power",
			TotalDurationNanos: windowDuration,
			Valid:              false,
			Reason:             "insufficient_samples",
		}
	}

	for _, p := range powerReadings {
		if p.Available && (p.Value < 0 || math.IsNaN(p.Value) || math.IsInf(p.Value, 0)) {
			return EnergySummary{WindowName: windowName, DeviceUUID: deviceUUID, Method: "integrated_power", Reason: "invalid_power_value"}
		}
	}

	var totalMJ float64
	var covered int64
	for i := 0; i < len(powerReadings)-1; i++ {
		p1 := powerReadings[i]
		p2 := powerReadings[i+1]
		dtNanos := p2.Timestamp - p1.Timestamp
		if p1.Timestamp < startMono || p2.Timestamp > endMono || dtNanos <= 0 {
			continue
		}
		if !p1.Available || !p2.Available {
			// Do not bridge across unavailable gap
			continue
		}
		if opts.MaxAllowedGapNanos > 0 && dtNanos > opts.MaxAllowedGapNanos {
			// Do not bridge excessive gap
			continue
		}
		avgMW := p1.Value/2 + p2.Value/2
		dtSec := float64(dtNanos) / float64(time.Second)
		totalMJ += avgMW * dtSec
		covered += dtNanos
	}

	coverage := float64(covered) / float64(windowDuration)
	if coverage > 1.0 {
		coverage = 1.0
	}

	// 2^64 is the first float outside the uint64 range. MaxUint64 rounds
	// to that same float, so comparison with MaxUint64 is insufficient.
	if math.IsNaN(totalMJ) || math.IsInf(totalMJ, 0) || totalMJ < 0 || totalMJ >= math.Ldexp(1, 64) {
		return EnergySummary{WindowName: windowName, DeviceUUID: deviceUUID, Method: "integrated_power", Reason: "energy_out_of_range"}
	}

	deltaJoules := totalMJ / 1000.0

	return EnergySummary{
		StartMonoNanos: startMono, EndMonoNanos: endMono,
		DeviceUUID:           deviceUUID,
		WindowName:           windowName,
		Method:               "integrated_power",
		EnergyConsumedMJ:     uint64(totalMJ),
		EnergyConsumedJoules: deltaJoules,
		CoveredDurationNanos: covered,
		TotalDurationNanos:   windowDuration,
		CoverageRatio:        coverage,
		Valid:                covered > 0,
	}
}

// AggregateWindowStrict aggregates metrics within the window, returning ErrBitmaskAveragingRefused
// if any metric within the window has bitmask or event kind.
func AggregateWindowStrict(windowName string, startMono, endMono int64, samples []Sample, opts WindowOptions) ([]WindowSummary, []EnergySummary, error) {
	for _, s := range samples {
		t := s.QueryStartNanos
		if t < startMono || t > endMono {
			continue
		}
		for _, r := range s.Readings {
			if err := ValidateMetricAveraging(r.Name); err != nil {
				return nil, nil, err
			}
		}
	}
	summaries, energySummaries := AggregateWindow(windowName, startMono, endMono, samples, opts)
	return summaries, energySummaries, nil
}
