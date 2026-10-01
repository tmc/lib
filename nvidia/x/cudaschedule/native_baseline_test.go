package cudaschedule_test

import (
	"fmt"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaschedule"
)

const scheduleReaderDelayThreshold = 8000

// Shared with the native campaign so host tests cover its acceptance checks.
func checkScheduleBaseline(report *cudaschedule.CampaignReport) error {
	if report == nil {
		return fmt.Errorf("missing baseline report")
	}
	if report.BaselineVerdict == cudaschedule.VerdictSkipped {
		return fmt.Errorf("baseline skipped: %s", report.Baseline.Stdout)
	}
	if report.BaselineVerdict != cudaschedule.VerdictPass || report.BaselineStatus != "Pass" {
		return fmt.Errorf("baseline failed (%s/%s, exit %d): %s", report.BaselineVerdict, report.BaselineStatus, report.Baseline.ExitCode, report.Baseline.Stdout)
	}
	if !strings.Contains(report.Baseline.Stdout, "PASS: shared handoff produced 1") {
		return fmt.Errorf("baseline missing expected PASS line (exit %d): %s", report.Baseline.ExitCode, report.Baseline.Stdout)
	}
	for _, line := range strings.Split(report.Baseline.Stdout, "\n") {
		if text, ok := strings.CutPrefix(line, "READER_DELAY_NS: "); ok {
			elapsed, err := strconv.ParseUint(text, 10, 64)
			if err != nil {
				return fmt.Errorf("baseline invalid reader delay: %w", err)
			}
			if elapsed < scheduleReaderDelayThreshold {
				return fmt.Errorf("baseline reader delay %d ns below threshold %d ns", elapsed, scheduleReaderDelayThreshold)
			}
			return nil
		}
	}
	return fmt.Errorf("baseline missing measured reader delay")
}

func TestScheduleNativeBaseline(t *testing.T) {
	for _, tc := range []struct {
		name, output, status string
		verdict              cudaschedule.Verdict
		want                 string
	}{
		{"accepted", "PASS: shared handoff produced 1\nREADER_DELAY_NS: 8000\n", "Pass", cudaschedule.VerdictPass, ""},
		{"skipped", "SKIP: unsupported architecture", "Declined", cudaschedule.VerdictSkipped, "baseline skipped"},
		{"pass without output", "", "Pass", cudaschedule.VerdictPass, "missing expected PASS line"},
		{"failed child exits zero", "", "Broken", cudaschedule.VerdictMismatch, "baseline failed (Mismatch/Broken, exit 0)"},
		{"missing delay", "PASS: shared handoff produced 1\n", "Pass", cudaschedule.VerdictPass, "missing measured reader delay"},
		{"short delay", "PASS: shared handoff produced 1\nREADER_DELAY_NS: 7999\n", "Pass", cudaschedule.VerdictPass, "reader delay 7999 ns below threshold 8000 ns"},
		{"invalid delay", "PASS: shared handoff produced 1\nREADER_DELAY_NS: invalid\n", "Pass", cudaschedule.VerdictPass, "invalid reader delay"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			report := &cudaschedule.CampaignReport{BaselineStatus: tc.status, BaselineVerdict: tc.verdict, Baseline: cudaschedule.RunRecord{Exited: true, ExitCode: 0, Stdout: tc.output}}
			err := checkScheduleBaseline(report)
			if tc.want == "" {
				if err != nil {
					t.Fatal(err)
				}
			} else if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want %q", err, tc.want)
			}
		})
	}
	if err := checkScheduleBaseline(nil); err == nil || !strings.Contains(err.Error(), "missing baseline") {
		t.Fatalf("nil report error = %v", err)
	}
}
