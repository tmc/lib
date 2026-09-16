package cudacmp

import (
	"fmt"
	"sort"

	"github.com/tmc/lib/nvidia/x/cudacensus"
	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

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
func BuildPhaseCensus(phase Phase, launches []cudacensus.Launch, coster cudacensus.Coster, fc *FamilyClassifier) PhaseCensus {
	if fc == nil {
		fc = DefaultFamilyClassifier
	}
	entries := cudacensus.Census(launches, coster)

	var totalNs int64
	for _, l := range launches {
		totalNs += l.DurationNs()
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
		Phase:    phase,
		TotalNs:  totalNs,
		Families: families,
		Entries:  entries,
	}
}

// ExtractPhaseCensus extracts launches from the capture for the given phase and builds
// its PhaseCensus.
func ExtractPhaseCensus(c *cudacensus.Capture, phase Phase, coster cudacensus.Coster, fc *FamilyClassifier) (PhaseCensus, error) {
	launches, err := ExtractPhaseLaunches(c, phase)
	if err != nil {
		return PhaseCensus{}, fmt.Errorf("extract phase %s: %w", phase, err)
	}
	return BuildPhaseCensus(phase, launches, coster, fc), nil
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
func ComparePhases(cPhase, goPhase PhaseCensus, nullDriftNs int64, model *cudaroofline.Model) []DifferentialReport {
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
		cKm := cPhase.Families[fam]
		goKm := goPhase.Families[fam]

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
		valid := absDelta > nullDriftNs

		var rptGo, rptC *cudaroofline.KernelPoint
		if model != nil {
			if cKm.Launches > 0 && cDur > 0 && cKm.Residency == model.Ceiling.Residency {
				pt := model.Evaluate(fam, cKm.FLOPs, cKm.BytesMoved, uint64(cDur))
				rptC = &pt
			}
			if goKm.Launches > 0 && goDur > 0 && goKm.Residency == model.Ceiling.Residency {
				pt := model.Evaluate(fam, goKm.FLOPs, goKm.BytesMoved, uint64(goDur))
				rptGo = &pt
			}
		}

		reports = append(reports, DifferentialReport{
			Phase:              cPhase.Phase,
			Family:             fam,
			CDurationNs:        cDur,
			GoDurationNs:       goDur,
			DeltaNs:            deltaNs,
			Speedup:            speedup,
			LaunchDeltaNs:      int64(goKm.Timing.HostLaunchNs) - int64(cKm.Timing.HostLaunchNs),
			NullDriftNs:        nullDriftNs,
			StatisticallyValid: valid,
			RooflineGo:         rptGo,
			RooflineC:          rptC,
		})
	}
	return reports
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

	prefillReports := ComparePhases(cPrefill, goPrefill, nullDriftNs, model)
	decodeReports := ComparePhases(cDecode, goDecode, nullDriftNs, model)

	return append(prefillReports, decodeReports...), nil
}
