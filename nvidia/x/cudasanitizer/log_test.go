package cudasanitizer

import (
	"context"
	"math"
	"strings"
	"testing"
)

func TestParseLog(t *testing.T) {
	tests := []struct {
		name         string
		text         string
		summaryFound bool
		errs         int
		findings     int
		headline     string
	}{
		{
			name:         "empty",
			text:         "",
			summaryFound: false,
		},
		{
			name:         "clean",
			text:         cleanReport,
			summaryFound: true,
			errs:         0,
		},
		{
			name:         "errors",
			text:         errorReport,
			summaryFound: true,
			errs:         2,
			findings:     2,
			headline:     "Invalid __global__ write of size 4 bytes",
		},
		{
			name:         "truncated report has no verdict",
			text:         truncatedReport,
			summaryFound: false,
			findings:     1,
			headline:     "Invalid __global__ read of size 8 bytes",
		},
		{
			name:         "singular error",
			text:         "========= ERROR SUMMARY: 1 error\n",
			summaryFound: true,
			errs:         1,
		},
		{
			name: "application output interleaved with the report is ignored",
			text: "vector add ok\n========= ERROR SUMMARY: 0 errors\nall done\n",
			// Only prefixed lines are read, so a program that prints
			// "ERROR SUMMARY" itself cannot forge a verdict.
			summaryFound: true,
			errs:         0,
		},
		{
			name:         "forged summary without the prefix is not a verdict",
			text:         "ERROR SUMMARY: 0 errors\n",
			summaryFound: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := ParseLog(tt.text)
			if l.SummaryFound != tt.summaryFound {
				t.Errorf("SummaryFound = %v, want %v", l.SummaryFound, tt.summaryFound)
			}
			if l.Errors != tt.errs {
				t.Errorf("Errors = %d, want %d", l.Errors, tt.errs)
			}
			if len(l.Findings) != tt.findings {
				t.Fatalf("len(Findings) = %d, want %d (%+v)", len(l.Findings), tt.findings, l.Findings)
			}
			if tt.headline != "" && l.Findings[0].Headline != tt.headline {
				t.Errorf("Findings[0].Headline = %q, want %q", l.Findings[0].Headline, tt.headline)
			}
		})
	}
}

func TestParseLogLeakSummary(t *testing.T) {
	const text = `========= COMPUTE-SANITIZER
========= Leaked 1024 bytes at 0x7f0000000200
=========     Saved host backtrace up to driver entry point at error
=========
========= LEAK SUMMARY: 1024 bytes leaked in 1 allocations
========= ERROR SUMMARY: 1 error
`
	l := ParseLog(text)
	if !l.SummaryFound || l.Errors != 1 {
		t.Fatalf("SummaryFound = %v, Errors = %d; want true, 1", l.SummaryFound, l.Errors)
	}
	if l.LeakedBytes != 1024 || l.LeakedAllocations != 1 {
		t.Errorf("LeakedBytes = %d in %d allocations, want 1024 in 1",
			l.LeakedBytes, l.LeakedAllocations)
	}
	if len(l.Findings) != 1 || l.Findings[0].Headline != "Leaked 1024 bytes at 0x7f0000000200" {
		t.Errorf("Findings = %+v, want the leak block", l.Findings)
	}
}

// TestParseLogRacecheckOnly covers racecheck output that ends with its own
// summary instead of an ERROR SUMMARY line. Its error count is the verdict.
func TestParseLogRacecheckOnly(t *testing.T) {
	const text = `========= COMPUTE-SANITIZER
========= Race reported between Write access at 0x50 in kernel(int *)
=========     and Read access at 0x80 in kernel(int *)
=========
========= RACECHECK SUMMARY: 3 hazards displayed (2 errors, 1 warning)
`
	l := ParseLog(text)
	if !l.SummaryFound {
		t.Fatal("SummaryFound = false, want true: a racecheck summary is a verdict")
	}
	if l.Errors != 2 || l.Hazards != 3 || l.Warnings != 1 {
		t.Errorf("Errors = %d, Hazards = %d, Warnings = %d; want 2, 3, 1", l.Errors, l.Hazards, l.Warnings)
	}
}

func TestFindingString(t *testing.T) {
	l := ParseLog(errorReport)
	got := l.Findings[0].String()
	const want = "Invalid __global__ write of size 4 bytes\n" +
		"    at 0x70 in vectorAdd(int *, int *, int *, int)\n" +
		"    by thread (255,0,0) in block (3,0,0)\n" +
		"    Address 0x7f8a00000400 is out of bounds"
	if got != want {
		t.Errorf("Finding.String() =\n%s\nwant\n%s", got, want)
	}
}

// TestParseLogMultipleSummaries checks that a report carrying more than one
// summary totals the counts instead of keeping the last. compute-sanitizer
// prints one summary per process, so a run that forks -- or two runs
// concatenated into one file -- ends with several, and letting a later clean
// summary replace an earlier one would classify a report containing errors as
// clean. That is the single direction this parser must never get wrong.
func TestParseLogMultipleSummaries(t *testing.T) {
	const text = `========= COMPUTE-SANITIZER
========= Invalid __global__ read of size 4 bytes
=========     at addone(float*)+0x70
=========
========= ERROR SUMMARY: 8 errors
========= COMPUTE-SANITIZER
========= ERROR SUMMARY: 0 errors
`
	l := ParseLog(text)
	if !l.SummaryFound {
		t.Fatal("SummaryFound = false, want true")
	}
	if l.Errors != 8 {
		t.Errorf("Errors = %d, want 8: the second process's clean summary erased the first's", l.Errors)
	}
	if l.Summaries != 2 {
		t.Errorf("Summaries = %d, want 2", l.Summaries)
	}
	if len(l.Findings) != 1 {
		t.Errorf("Findings = %d, want 1", len(l.Findings))
	}
}

// A regular summary from another process must not erase a racecheck error.
func TestMixedSummaryFamilies(t *testing.T) {
	const race = "========= RACECHECK SUMMARY: 1 hazard displayed (1 error, 0 warnings)\n"
	const clean = "========= ERROR SUMMARY: 0 errors\n"
	for _, text := range []string{race + clean, clean + race} {
		log := ParseLog(text)
		if log.RaceErrors != 1 {
			t.Fatalf("RaceErrors = %d, want 1", log.RaceErrors)
		}
		result := &Result{Log: log, ExitCode: 0}
		result.classify(context.Background(), nil)
		if result.Outcome != OutcomeSanitizerErrors {
			t.Fatalf("mixed summaries classified as %v", result.Outcome)
		}
	}
}

func TestSummaryCountOverflow(t *testing.T) {
	const huge = "99999999999999999999999999999999999999999999"
	for _, line := range []string{
		"========= ERROR SUMMARY: " + huge + " errors\n",
		"========= RACECHECK SUMMARY: " + huge + " hazards displayed (" + huge + " errors, 0 warnings)\n",
	} {
		log := ParseLog(strings.Repeat(line, 2))
		if log.Errors != math.MaxInt {
			t.Fatalf("errors = %d, want saturated count", log.Errors)
		}
		result := &Result{Log: log, ExitCode: 0}
		result.classify(context.Background(), nil)
		if result.Outcome != OutcomeSanitizerErrors {
			t.Fatalf("overflow classified as %v", result.Outcome)
		}
	}
	log := ParseLog(strings.Repeat("========= LEAK SUMMARY: "+huge+" bytes leaked in "+huge+" allocations\n", 2))
	if log.LeakedBytes != math.MaxInt64 || log.LeakedAllocations != math.MaxInt {
		t.Fatalf("leak counts wrapped: %+v", log)
	}
}

func TestUnfinishedReportAfterSummary(t *testing.T) {
	const clean = "========= ERROR SUMMARY: 0 errors\n"
	const banner = "========= COMPUTE-SANITIZER\n"
	const finding = "========= Invalid __global__ write of size 4 bytes\n"
	for _, tt := range []struct {
		name, suffix string
		complete     bool
	}{
		{"banner", banner, false},
		{"finding", finding, false},
		{"banner and finding", banner + finding, false},
		{"completed second report", banner + finding + "========= ERROR SUMMARY: 1 error\n", true},
		{"completed clean second report", banner + clean, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			log := ParseLog(clean + tt.suffix)
			if log.SummaryFound != tt.complete {
				t.Errorf("SummaryFound = %v, want %v", log.SummaryFound, tt.complete)
			}
			result := &Result{Log: log, ExitCode: 0, ErrorExitCode: 66}
			result.classify(context.Background(), nil)
			if !tt.complete && result.Outcome != OutcomeToolFailure {
				t.Errorf("unfinished report classified as %v, want tool failure", result.Outcome)
			}
		})
	}
}
