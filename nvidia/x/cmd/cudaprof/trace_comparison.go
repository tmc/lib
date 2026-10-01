package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudaprof/isolation"
	"github.com/tmc/lib/nvidia/x/cudaprof/iteration"
	"github.com/tmc/lib/nvidia/x/cudaprof/shape"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
	"reflect"
	"regexp"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
)

// Comparison receipts preserve independent selected/full stationarity and both
// bound arm identities. A binding checks artifact consistency, not authenticity
// or workload equivalence. Missing shape receipts permit no operation delta.
func traceComparisonView(input *traceSession) (any, bool, error) {
	var alignment alignmentReceipt
	if err := json.Unmarshal(input.Members[".alignment.json"], &alignment); err != nil {
		return nil, false, fmt.Errorf("trace summary alignment: %w", err)
	}
	binding := false
	if data, ok := input.Members[".cudaprof-bindings.json"]; ok {
		var index tracecontract.Bindings
		if err := json.Unmarshal(data, &index); err != nil {
			return nil, false, fmt.Errorf("trace summary bindings: %w", err)
		}
		if err := index.Verify(input.Members); err != nil {
			return nil, false, err
		}
		binding = true
	}
	result := struct {
		Alignment      alignmentReceipt `json:"alignment_receipt"`
		Binding        string           `json:"artifact_binding"`
		Provenance     string           `json:"provenance"`
		Shape          string           `json:"shape_qualification"`
		OperationDelta *int64           `json:"operation_delta_ns"`
		Campaign       string           `json:"campaign_uncertainty"`
		Limits         []string         `json:"limits"`
	}{alignment, "unavailable", "supplied comparison receipt; gates are preserved and are not independently re-measured", "unavailable: matched application shape receipt absent", nil, "unavailable: independent run campaign receipt absent", []string{"launch distributions are not independent run uncertainty", "fixed-offset alignment does not synchronize independent runs", "hash binding establishes artifact consistency, not authenticity or application equivalence"}}
	if binding {
		result.Binding = "verified: exact members, manifest and alignment receipt"
		return result, true, nil
	}
	return result, false, &exitCodeError{Code: 2, Err: errors.New("trace summary: comparison member binding unavailable")}
}

// A qualified comparison uses the existing comparisonSummary model after
// independently recomputing shape and output-control evidence against exact
// archive-bound captures. It never constructs a second uncertainty model.
func qualifiedTraceComparison(input *traceSession, shapePath, campaignPath string, limits traceInputLimits) (any, map[string]string, error) {
	var alignment alignmentReceipt
	if err := json.Unmarshal(input.Members[".alignment.json"], &alignment); err != nil {
		return nil, nil, err
	}
	receipt, sources, evidence, err := validatedTraceShape(input, shapePath, limits)
	if err != nil {
		return nil, nil, err
	}
	gates := map[string]string{"shape": evidence.Disposition + ": " + evidence.Reason, "campaign": "unavailable: no bound campaign receipt"}
	if evidence.Disposition != "passed" {
		receipt.Shape.Pass = false
	}
	if receipt.Version == 0 {
		return nil, gates, errors.New("trace summary: bound shape evidence unavailable")
	}
	marker, err := regexp.Compile(alignment.Marker)
	if err != nil {
		return nil, gates, fmt.Errorf("trace comparison marker: %w", err)
	}
	states := [2]string{"unchecked", "unchecked"}
	for i, source := range sources {
		receipt.Arms[i].Completeness = "unavailable"
		receipt.Arms[i].Stationarity = iteration.Gate{Failures: []string{"full-capture stationarity unavailable"}}
		if source == nil || source.CollectorError != nil {
			continue
		}
		receipt.Arms[i].Completeness = "passed"
		observed := captureIsolation(source.Capture)
		states[i] = observed.State
		alignment.Arms[i].Isolation = observed
		alignment.Arms[i].Completeness = "passed"
		alignment.Arms[i].Stationarity = iteration.Gate{Failures: []string{"selected stationarity unavailable without retained iteration marker"}}
		alignment.Arms[i].FullGate = iteration.Gate{Failures: []string{"full stationarity unavailable without retained iteration marker"}}
		gates[fmt.Sprintf("arm_%d_isolation", i)] = observed.State + ": " + strings.Join(observed.Reasons, "; ")
		if alignment.Marker != "" {
			selected := alignment.Arms[i]
			if err := selectAlignmentWindow(&selected, source.Records, iterationWindow{marker: marker, skip: selected.Skip, count: selected.Iterations}); err != nil {
				return nil, gates, fmt.Errorf("trace comparison selected gate: %w", err)
			}
			if selected.Start != alignment.Arms[i].Start || selected.End != alignment.Arms[i].End {
				return nil, gates, errors.New("trace comparison: selected window differs from original capture")
			}
			alignment.Arms[i].Stationarity = selected.Stationarity
			alignment.Arms[i].FullGate = selected.FullGate
			alignment.Arms[i].MarkerVerdict = selected.MarkerVerdict
			alignment.Arms[i].Markers = selected.Markers

			report, e := iteration.Analyze(source.Records, iteration.Options{Marker: marker})
			if e == nil {
				receipt.Arms[i].Stationarity = report.Gate
				receipt.Arms[i].Markers = report.Markers
			}
		}
	}
	gates["isolation"] = "unchecked: at least one original arm lacks qualifying observer coverage"
	if states[0] == "clear" && states[1] == "clear" {
		gates["isolation"] = "clear: independently bound sampled-clear evidence on both original arms"
	}
	if states[0] == "contaminated" || states[1] == "contaminated" {
		gates["isolation"] = "contaminated: foreign activity observed on an original arm"
	}
	var campaign *campaignReport
	if campaignPath != "" {
		validated, state, e := validatedTraceCorrectness(input, campaignPath, limits)
		if e != nil {
			return nil, gates, e
		}
		gates["correctness"] = state.Disposition + ": " + state.Reason
		if state.Disposition == "passed" {
			if e := validateTraceCampaign(&validated, limits); e != nil {
				return nil, gates, e
			}
			campaign = &validated
			gates["campaign"] = "passed: source-bound output controls, decoded windows, shapes and existing block bootstrap recomputed; exchangeability remains unverified"
		}
	}
	summary, err := makeComparisonSummary(receipt, alignment, campaign)
	if err != nil {
		return nil, gates, err
	}
	summary.Limits += " Shape and full-capture stationarity were recomputed from original hash-matched source captures. Campaign output/source correspondence and profiled windows were independently revalidated when supplied; campaign intervals remain block-level estimates, never operation uncertainty."
	return summary, gates, nil
}

func validateTraceCampaign(campaign *campaignReport, limits traceInputLimits) error {
	if len(campaign.Blocks) < 3 || len(campaign.Blocks) > 100 || campaign.ExpectedMarkers[0] < 2 || campaign.ExpectedMarkers[1] < 2 || campaign.SkipIterations < 0 {
		return errors.New("trace campaign: invalid registered block/marker controls")
	}
	if campaign.Domain != "driver" && campaign.Domain != "runtime" {
		return errors.New("trace campaign: invalid domain")
	}
	marker, err := regexp.Compile(campaign.Marker)
	if err != nil || campaign.Marker == "" {
		return errors.New("trace campaign: invalid iteration marker")
	}
	var effects []float64
	for _, block := range campaign.Blocks {
		if len(block.Excluded) != 0 {
			continue
		}
		var windows [4]int64
		var records [4]timeline.Records
		count := 0
		for _, run := range block.Runs {
			if !run.Profiled {
				continue
			}
			if count >= 4 || run.Arm != qualifyArms[count] {
				return errors.New("trace campaign: profiled controls do not follow ABBA")
			}
			arm := 0
			if count == 1 || count == 2 {
				arm = 1
			}
			source, err := loadTraceSessionLimits(run.Capture, campaign.Domain, limits)
			if err != nil {
				return err
			}
			if source.Capture == nil || source.CollectorError != nil {
				return errors.New("trace campaign: incomplete profiled capture")
			}
			if err := isolation.Require(captureIsolation(source.Capture), false); err != nil {
				return err
			}
			report, err := iteration.Analyze(source.Records, iteration.Options{Marker: marker, Expect: campaign.ExpectedMarkers[arm], Skip: campaign.SkipIterations})
			if err != nil {
				return err
			}
			if !report.Gate.Pass {
				return fmt.Errorf("trace campaign: iteration gate failed: %v", report.Gate.Failures)
			}
			var start, end int64
			for _, it := range report.Iterations {
				if it.Steady {
					if start == 0 {
						start = it.Start
					}
					end = it.End
				}
			}
			records[count], err = campaignSteadyRecords(source.Records, start, end)
			if err != nil {
				return err
			}
			windows[count] = report.Steady.WindowNanos
			if run.WindowNanos != windows[count] {
				return errors.New("trace campaign: supplied window differs from decoded capture")
			}
			count++
		}
		if count != 4 || windows[0] <= 0 || windows[3] <= 0 {
			return errors.New("trace campaign: incomplete or empty ABBA window controls")
		}
		for _, pair := range [][2]int{{0, 1}, {3, 2}} {
			if !shape.Compare(records[pair[0]], records[pair[1]], campaign.Domain).Pass {
				return errors.New("trace campaign: selected shape qualification failed")
			}
		}
		effect := (float64(windows[1]-windows[0]) + float64(windows[2]-windows[3])) / float64(windows[0]+windows[3]) * 100
		if block.DeviceWindowEffectPct == nil || *block.DeviceWindowEffectPct != effect {
			return errors.New("trace campaign: supplied block effect differs from decoded windows")
		}
		effects = append(effects, effect)
	}
	var expected *campaignInterval
	if len(effects) >= 3 {
		value := campaignBootstrap(effects)
		expected = &value
	}
	if !reflect.DeepEqual(campaign.Interval, expected) {
		return errors.New("trace campaign: supplied interval differs from recomputed block bootstrap")
	}
	campaign.Interval = expected
	return nil
}
