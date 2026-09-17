package cudacmp

import (
	"fmt"
	"math"
	"sort"

	"github.com/tmc/lib/nvidia/x/cudacensus"
	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

// Interval represents a half-open time interval [StartNs, EndNs).
type Interval struct {
	StartNs int64
	EndNs   int64
}

// Duration returns the length of the interval.
func (i Interval) Duration() int64 {
	return i.EndNs - i.StartNs
}

// UnionDuration computes the total length of the union of half-open intervals [StartNs, EndNs).
// Overlapping intervals are merged so no time is double-counted.
// Returns an error if any interval has EndNs < StartNs, negative timestamps, or if duration overflows.
func UnionDuration(intervals []Interval) (int64, error) {
	if len(intervals) == 0 {
		return 0, nil
	}
	// Sort by StartNs, breaking ties with EndNs.
	sorted := make([]Interval, len(intervals))
	copy(sorted, intervals)
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i].StartNs == sorted[j].StartNs {
			return sorted[i].EndNs < sorted[j].EndNs
		}
		return sorted[i].StartNs < sorted[j].StartNs
	})

	var merged []Interval
	for _, iv := range sorted {
		if iv.EndNs < iv.StartNs {
			return 0, fmt.Errorf("cudacmp: invalid interval [%d, %d)", iv.StartNs, iv.EndNs)
		}
		if iv.StartNs < 0 || iv.EndNs < 0 {
			return 0, fmt.Errorf("cudacmp: negative timestamp in interval [%d, %d)", iv.StartNs, iv.EndNs)
		}
		if len(merged) == 0 {
			merged = append(merged, iv)
			continue
		}
		last := &merged[len(merged)-1]
		if iv.StartNs <= last.EndNs {
			if iv.EndNs > last.EndNs {
				last.EndNs = iv.EndNs
			}
		} else {
			merged = append(merged, iv)
		}
	}

	var total int64
	for _, m := range merged {
		dur := m.EndNs - m.StartNs
		if math.MaxInt64-dur < total {
			return 0, fmt.Errorf("cudacmp: duration overflow in interval union")
		}
		total += dur
	}
	return total, nil
}

// ExtractPhaseSpans returns all application spans in c associated with the given phase.
func ExtractPhaseSpans(c *cudacensus.Capture, phase Phase) []cudacensus.Span {
	if c == nil {
		return nil
	}
	target := string(phase)
	var out []cudacensus.Span
	for _, s := range c.Spans {
		if s.Name == target || s.Labels["phase"] == target {
			out = append(out, s)
			continue
		}
		// Decode phase may be demarcated by individual "token" spans labeled decode.
		if phase == PhaseDecode && s.Name == "token" && s.Labels["phase"] == "decode" {
			out = append(out, s)
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].StartNs < out[j].StartNs })
	return out
}

// ExtractPhaseLaunches finds all kernel launches falling within the time window
// of the specified phase.
func ExtractPhaseLaunches(c *cudacensus.Capture, phase Phase) ([]cudacensus.Launch, error) {
	if c == nil {
		return nil, fmt.Errorf("cudacmp: capture is nil")
	}
	spans := ExtractPhaseSpans(c, phase)
	if len(spans) == 0 {
		return nil, fmt.Errorf("cudacmp: no %s spans found in capture", phase)
	}

	// If there is an explicit outer span matching the phase name, use its exact window.
	for _, s := range spans {
		if s.Name == string(phase) {
			return c.Window(s.StartNs, s.EndNs), nil
		}
	}

	// Otherwise, compute the enclosing window across all matching spans.
	minStart := spans[0].StartNs
	maxEnd := spans[0].EndNs
	for _, s := range spans[1:] {
		if s.StartNs < minStart {
			minStart = s.StartNs
		}
		if s.EndNs > maxEnd {
			maxEnd = s.EndNs
		}
	}
	return c.Window(minStart, maxEnd), nil
}

// BuildPhaseCensus aggregates launches for a phase and groups them by architectural family.
// It computes both SummedKernelNs (sum of individual launch durations) and GPUBusyNs
// (duration of the union of GPU execution intervals).
func BuildPhaseCensus(phase Phase, launches []cudacensus.Launch, coster cudacensus.Coster, fc *FamilyClassifier) (PhaseCensus, error) {
	if fc == nil {
		fc = DefaultFamilyClassifier
	}
	entries := cudacensus.Census(launches, coster)

	var intervals []Interval
	var totalNs int64
	for _, l := range launches {
		if l.EndNs < l.StartNs {
			return PhaseCensus{}, fmt.Errorf("cudacmp: invalid launch interval [%d, %d)", l.StartNs, l.EndNs)
		}
		if l.StartNs < 0 || l.EndNs < 0 {
			return PhaseCensus{}, fmt.Errorf("cudacmp: negative timestamp in launch [%d, %d)", l.StartNs, l.EndNs)
		}
		dur := l.DurationNs()
		if math.MaxInt64-dur < totalNs {
			return PhaseCensus{}, fmt.Errorf("cudacmp: launch duration overflow")
		}
		totalNs += dur
		intervals = append(intervals, Interval{StartNs: l.StartNs, EndNs: l.EndNs})
	}

	gpuBusyNs, err := UnionDuration(intervals)
	if err != nil {
		return PhaseCensus{}, err
	}

	families := make(map[string]KernelMetric)
	for _, e := range entries {
		fam := fc.Classify(e.Name)
		km := families[fam]
		km.Family = fam
		km.Name = fam
		km.Launches += e.Launches
		km.Timing.DeviceNs += uint64(e.DeviceNs)
		km.BytesMoved += e.Bytes
		km.FLOPs += e.FLOPs

		if km.Residency == "" {
			km.Residency = e.Residency
		} else if km.Residency != e.Residency {
			km.Residency = cudacensus.UnknownResidency
		}
		families[fam] = km
	}

	return PhaseCensus{
		Phase:          phase,
		SummedKernelNs: totalNs,
		GPUBusyNs:      gpuBusyNs,
		Families:       families,
		Entries:        entries,
	}, nil
}

// ExtractPhaseCensus extracts launches from the capture for the given phase and builds
// its PhaseCensus.
func ExtractPhaseCensus(c *cudacensus.Capture, phase Phase, coster cudacensus.Coster, fc *FamilyClassifier) (PhaseCensus, error) {
	launches, err := ExtractPhaseLaunches(c, phase)
	if err != nil {
		return PhaseCensus{}, fmt.Errorf("extract phase %s: %w", phase, err)
	}
	return BuildPhaseCensus(phase, launches, coster, fc)
}

// ExtractPhases extracts both prefill and decode PhaseCensus from a capture.
func ExtractPhases(c *cudacensus.Capture, coster cudacensus.Coster, fc *FamilyClassifier) (prefill, decode PhaseCensus, err error) {
	prefill, err = ExtractPhaseCensus(c, PhasePrefill, coster, fc)
	if err != nil {
		return PhaseCensus{}, PhaseCensus{}, err
	}
	decode, err = ExtractPhaseCensus(c, PhaseDecode, coster, fc)
	if err != nil {
		return PhaseCensus{}, PhaseCensus{}, err
	}
	return prefill, decode, nil
}

// ComparePhases compares two PhaseCensus instances of the same phase and computes
// DifferentialReport entries across all architectural families.
// It returns an error if the phases are incompatible or inputs are invalid.
func ComparePhases(cPhase, goPhase PhaseCensus, nullDriftNs int64, model *cudaroofline.Model) ([]DifferentialReport, error) {
	if cPhase.Phase == "" || goPhase.Phase == "" {
		return nil, fmt.Errorf("cudacmp: empty phase name")
	}
	if cPhase.Phase != goPhase.Phase {
		return nil, fmt.Errorf("cudacmp: incompatible phases: %q vs %q", cPhase.Phase, goPhase.Phase)
	}
	if nullDriftNs < 0 {
		return nil, fmt.Errorf("cudacmp: invalid null drift threshold: %d (must be non-negative)", nullDriftNs)
	}

	// Collect all unique families across both censuses.
	seen := make(map[string]bool)
	for f := range cPhase.Families {
		seen[f] = true
	}
	for f := range goPhase.Families {
		seen[f] = true
	}

	// Order canonical families first, followed by any custom families.
	var order []string
	for _, fam := range CanonicalFamilies {
		if seen[fam] {
			order = append(order, fam)
			delete(seen, fam)
		}
	}
	var remaining []string
	for fam := range seen {
		remaining = append(remaining, fam)
	}
	sort.Strings(remaining)
	order = append(order, remaining...)

	reports := make([]DifferentialReport, 0, len(order))
	for _, fam := range order {
		cKm, cOk := cPhase.Families[fam]
		goKm, goOk := goPhase.Families[fam]

		rpt := DifferentialReport{
			Phase:       cPhase.Phase,
			Family:      fam,
			NullDriftNs: nullDriftNs,
		}

		switch {
		case cOk && goOk:
			rpt.Availability = AvailableBoth
			rpt.Available = true
			rpt.CObserved = true
			rpt.GoObserved = true

			cDur := int64(cKm.Timing.DeviceNs)
			goDur := int64(goKm.Timing.DeviceNs)
			deltaNs := goDur - cDur

			var speedup float64
			if goDur > 0 {
				speedup = float64(cDur) / float64(goDur)
			}

			absDelta := deltaNs
			if absDelta < 0 {
				absDelta = -absDelta
			}
			exceeds := absDelta > nullDriftNs

			rpt.CDurationNs = cDur
			rpt.GoDurationNs = goDur
			rpt.DeltaNs = deltaNs
			rpt.Speedup = speedup
			rpt.LaunchDeltaNs = int64(goKm.Timing.HostLaunchNs) - int64(cKm.Timing.HostLaunchNs)
			rpt.ExceedsDrift = exceeds
			rpt.StatisticallyValid = exceeds

			if model != nil {
				if cKm.Launches > 0 && cDur > 0 && cKm.Residency == model.Ceiling.Residency {
					pt := model.Evaluate(fam, cKm.FLOPs, cKm.BytesMoved, uint64(cDur))
					rpt.RooflineC = &pt
				}
				if goKm.Launches > 0 && goDur > 0 && goKm.Residency == model.Ceiling.Residency {
					pt := model.Evaluate(fam, goKm.FLOPs, goKm.BytesMoved, uint64(goDur))
					rpt.RooflineGo = &pt
				}
			}

		case cOk && !goOk:
			rpt.Availability = AvailableCOnly
			rpt.Available = false
			rpt.CObserved = true
			rpt.GoObserved = false
			cDur := int64(cKm.Timing.DeviceNs)
			rpt.CDurationNs = cDur
			// Delta, Speedup, and ExceedsDrift are not computed for one-sided families.
			if model != nil && cKm.Launches > 0 && cDur > 0 && cKm.Residency == model.Ceiling.Residency {
				pt := model.Evaluate(fam, cKm.FLOPs, cKm.BytesMoved, uint64(cDur))
				rpt.RooflineC = &pt
			}

		case !cOk && goOk:
			rpt.Availability = AvailableGoOnly
			rpt.Available = false
			rpt.CObserved = false
			rpt.GoObserved = true
			goDur := int64(goKm.Timing.DeviceNs)
			rpt.GoDurationNs = goDur
			// Delta, Speedup, and ExceedsDrift are not computed for one-sided families.
			if model != nil && goKm.Launches > 0 && goDur > 0 && goKm.Residency == model.Ceiling.Residency {
				pt := model.Evaluate(fam, goKm.FLOPs, goKm.BytesMoved, uint64(goDur))
				rpt.RooflineGo = &pt
			}
		}

		reports = append(reports, rpt)
	}
	return reports, nil
}

// CompareCaptures extracts phases from cCap and goCap, compares them, and returns
// differential reports for prefill and decode.
func CompareCaptures(cCap, goCap *cudacensus.Capture, coster cudacensus.Coster, fc *FamilyClassifier, nullDriftNs int64, model *cudaroofline.Model) ([]DifferentialReport, error) {
	cPrefill, cDecode, err := ExtractPhases(cCap, coster, fc)
	if err != nil {
		return nil, fmt.Errorf("c capture phases: %w", err)
	}
	goPrefill, goDecode, err := ExtractPhases(goCap, coster, fc)
	if err != nil {
		return nil, fmt.Errorf("go capture phases: %w", err)
	}

	prefillReports, err := ComparePhases(cPrefill, goPrefill, nullDriftNs, model)
	if err != nil {
		return nil, fmt.Errorf("compare prefill: %w", err)
	}
	decodeReports, err := ComparePhases(cDecode, goDecode, nullDriftNs, model)
	if err != nil {
		return nil, fmt.Errorf("compare decode: %w", err)
	}

	return append(prefillReports, decodeReports...), nil
}
