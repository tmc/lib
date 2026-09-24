package main

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestAdvice_MarkdownAndJSON(t *testing.T) {
	report := &AdviceReport{
		Items: []AdviceItem{
			{
				Severity: SeverityCritical,
				Category: CategoryRegisterPressure,
				Location: Location{
					Kernel: "gemm",
					PC:     0x0040,
					File:   "gemm.cu",
					Line:   42,
				},
				Finding:                  "Register spill point detected at PC 0x0040 (STL)",
				ActionableRecommendation: "Reduce live variable ranges to eliminate local memory spills.",
			},
			{
				Severity: SeverityWarning,
				Category: CategoryMemory,
				Location: Location{
					Kernel: "matrix",
					File:   "matrix.cu",
					Line:   14,
				},
				Finding:                  "TMA instruction selection failed: non-contiguous stride",
				ActionableRecommendation: "Ensure innermost tensor dimension is contiguous.",
			},
		},
	}

	md := report.Markdown()
	if !strings.Contains(md, "# CUDA Kernel Advice Report") {
		t.Errorf("Markdown missing header: %s", md)
	}
	if !strings.Contains(md, "- **Read first**: 1") {
		t.Errorf("Markdown missing Read first count: %s", md)
	}
	if !strings.Contains(md, "- **Cost**: 1") {
		t.Errorf("Markdown missing Cost count: %s", md)
	}
	if !strings.Contains(md, "gemm.cu:42") {
		t.Errorf("Markdown missing location gemm.cu:42: %s", md)
	}

	jsonStr, err := report.JSON()
	if err != nil {
		t.Fatalf("JSON() failed: %v", err)
	}
	var decoded AdviceReport
	if err := json.Unmarshal([]byte(jsonStr), &decoded); err != nil {
		t.Fatalf("decode JSON failed: %v", err)
	}
	if len(decoded.Items) != 2 {
		t.Fatalf("len(decoded.Items) = %d, want 2", len(decoded.Items))
	}
	if decoded.Items[0].Severity != SeverityReadFirst {
		t.Errorf("item 0 severity = %s, want %s", decoded.Items[0].Severity, SeverityReadFirst)
	}
}

func TestAdvice_EvidenceLevels(t *testing.T) {
	rep := &AdviceReport{
		Items: []AdviceItem{
			{
				Severity:                 SeverityCritical,
				Category:                 CategoryArchitecture,
				EvidenceLevel:            EvidenceObserved,
				Finding:                  "Observed hardware violation",
				ActionableRecommendation: "Fix code",
				NextMeasurement:          "Verify with hardware timer",
			},
			{
				Severity:                 SeverityWarning,
				Category:                 CategoryMemory,
				EvidenceLevel:            EvidenceSuspected,
				Finding:                  "Suspected memory stall",
				ActionableRecommendation: "Check access pattern",
				NextMeasurement:          "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum or gld_transactions",
			},
			{
				Severity:                 SeverityInfo,
				Category:                 CategoryArchitecture,
				EvidenceLevel:            EvidenceUnmeasured,
				Finding:                  "Unmeasured launch telemetry",
				ActionableRecommendation: "Enable metrics",
				NextMeasurement:          "Capture with CUPTI activity records enabled",
			},
		},
	}

	md := rep.Markdown()
	for _, want := range []string{
		"- **Evidence**: observed",
		"- **Evidence**: suspected",
		"- **Evidence**: unmeasured",
		"- **Next Measurement**: Verify with hardware timer",
		"- **Next Measurement**: l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum or gld_transactions",
		"- **Next Measurement**: Capture with CUPTI activity records enabled",
	} {
		if !strings.Contains(md, want) {
			t.Errorf("Markdown output missing %q:\n%s", want, md)
		}
	}

	jsonStr, err := rep.JSON()
	if err != nil {
		t.Fatalf("JSON serialization failed: %v", err)
	}
	for _, want := range []string{
		`"evidence_level": "observed"`,
		`"evidence_level": "suspected"`,
		`"evidence_level": "unmeasured"`,
		`"next_measurement": "Verify with hardware timer"`,
		`"next_measurement": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum or gld_transactions"`,
	} {
		if !strings.Contains(jsonStr, want) {
			t.Errorf("JSON output missing %q:\n%s", want, jsonStr)
		}
	}
}

// TestCostHeaderStatesBothDenominators pins the two scopes a cost header
// prints, and pins that it prints only one when only one is known.
//
// "GPU time" named the busy union rather than the elapsed span. The two
// diverge by an order of magnitude on an idle capture: a ds4 Go prefill
// finding worth 59.74% of device time was 5.93% of the span it was captured
// in, and a reader with one number reads the larger as the smaller. A capture
// that never established a span has no second number to print, and must not
// invent one.
func TestCostHeaderStatesBothDenominators(t *testing.T) {
	item := AdviceItem{
		Severity:        SeverityCost,
		Category:        CategoryArchitecture,
		Finding:         "a family holds the device time",
		DeviceTimeNanos: 202_340_000,
		ShareOfProfile:  59.74,
	}

	withSpan := (&AdviceReport{Items: []AdviceItem{item}, SpanNanos: 3_410_950_000}).Markdown()
	const want = "### 1. [Cost] (59.74% of device time, 202.34 ms, 5.93% of the captured span)"
	if !strings.Contains(withSpan, want) {
		t.Errorf("header missing the span share\ngot:  %s\nwant: %s", withSpan, want)
	}

	// No span measured: one denominator, and no claim about the other.
	noSpan := (&AdviceReport{Items: []AdviceItem{item}}).Markdown()
	if !strings.Contains(noSpan, "### 1. [Cost] (59.74% of device time, 202.34 ms)") {
		t.Errorf("header without a span did not state device time alone: %s", noSpan)
	}
	if strings.Contains(noSpan, "captured span") {
		t.Errorf("header claimed a span share with no span measured: %s", noSpan)
	}
}
