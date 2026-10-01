package cudaschedule_test

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaschedule"
)

type nativeCampaignCounts struct {
	Evaluated, Executed, Empty, Skipped, HarnessErrors, Findings, Confirmed int
	ConfirmationAttempts, ConfirmationHarnessErrors, ConfirmationSkipped    int
}

func checkNativeCampaign(report *cudaschedule.CampaignReport, clean bool) (nativeCampaignCounts, error) {
	var counts nativeCampaignCounts
	if report == nil {
		return counts, errors.New("missing campaign report")
	}
	counts.Evaluated = len(report.Seeds)
	for _, seed := range report.Seeds {
		switch seed.Status {
		case cudaschedule.SeedExecuted:
			counts.Executed++
		case cudaschedule.SeedEmptyDraw:
			counts.Empty++
		case cudaschedule.SeedHarnessError:
			counts.HarnessErrors++
		}
		if seed.Verdict == cudaschedule.VerdictSkipped {
			counts.Skipped++
		}
		if c := seed.Confirmation; c != nil {
			counts.ConfirmationAttempts += len(c.Outcomes)
			for _, outcome := range c.Outcomes {
				if outcome == cudaschedule.VerdictHarnessError {
					counts.ConfirmationHarnessErrors++
				}
				if outcome == cudaschedule.VerdictSkipped {
					counts.ConfirmationSkipped++
				}
			}
		}
		if seed.Verdict.IsFinding() {
			counts.Findings++
			if confirmedNativeMismatch(seed, report.ConfirmRuns) {
				counts.Confirmed++
			}
		}
	}
	if report.SeedEnd <= report.SeedStart || uint64(counts.Evaluated) != report.SeedEnd-report.SeedStart {
		return counts, errors.New("incomplete seed interval")
	}
	if report.SeedsDrawn != counts.Evaluated || report.VariantsExecuted != counts.Executed || report.EmptyDraws != counts.Empty || report.HarnessErrors != counts.HarnessErrors || counts.Executed+counts.Empty+counts.HarnessErrors != counts.Evaluated {
		return counts, errors.New("inconsistent campaign counters")
	}
	if counts.Executed == 0 || counts.HarnessErrors != 0 || counts.Skipped != 0 || counts.ConfirmationHarnessErrors != 0 || counts.ConfirmationSkipped != 0 {
		return counts, errors.New("campaign has no executed variants or has harness errors or skips")
	}
	for i, seed := range report.Seeds {
		if seed.Seed != report.SeedStart+uint64(i) {
			return counts, errors.New("seed interval is not sequential")
		}
		if seed.Status == cudaschedule.SeedEmptyDraw && seed.Verdict == cudaschedule.VerdictEmptyDraw {
			continue
		}
		if seed.Status != cudaschedule.SeedExecuted || seed.Run == nil || cudaschedule.Classify(*seed.Run) != seed.Verdict {
			return counts, errors.New("executed outcome lacks a matching run receipt")
		}
		if seed.Verdict.IsFinding() {
			c := seed.Confirmation
			if c == nil || report.ConfirmRuns < 2 || c.Attempts != report.ConfirmRuns || len(c.Outcomes) != c.Attempts || c.Outcomes[0] != seed.Verdict {
				return counts, errors.New("incomplete confirmation outcomes")
			}
			findings := 0
			for _, outcome := range c.Outcomes {
				if outcome != cudaschedule.VerdictMismatch && outcome != cudaschedule.VerdictPass {
					return counts, fmt.Errorf("unexpected confirmation verdict %s", outcome)
				}
				if outcome == seed.Verdict {
					findings++
				}
			}
			if c.Findings != findings || c.Confirmed != (findings == report.ConfirmRuns) {
				return counts, errors.New("inconsistent confirmation counters")
			}
		} else if seed.Confirmation != nil {
			return counts, errors.New("nonfinding has confirmation outcomes")
		}
		if seed.Verdict != cudaschedule.VerdictPass && (clean || seed.Verdict != cudaschedule.VerdictMismatch) {
			return counts, fmt.Errorf("unexpected campaign verdict %s", seed.Verdict)
		}
	}
	return counts, nil
}

func confirmedNativeMismatch(seed cudaschedule.SeedResult, budget int) bool {
	c := seed.Confirmation
	if seed.Verdict != cudaschedule.VerdictMismatch || c == nil || !c.Confirmed || budget < 2 || c.Attempts != budget || c.Findings != budget || len(c.Outcomes) != budget {
		return false
	}
	for _, outcome := range c.Outcomes {
		if outcome != cudaschedule.VerdictMismatch {
			return false
		}
	}
	return true
}

func writeNativeCampaignReceipt(dir, name string, report *cudaschedule.CampaignReport, runErr error) error {
	if dir == "" {
		return nil
	}
	file, err := os.OpenFile(filepath.Join(dir, name+".json"), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	receipt := struct {
		Report *cudaschedule.CampaignReport `json:"report"`
		Error  string                       `json:"error,omitempty"`
	}{Report: report}
	if runErr != nil {
		receipt.Error = runErr.Error()
	}
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return errors.Join(encoder.Encode(receipt), file.Close())
}

func TestNativeCampaignEvidenceRefusesIncompleteControls(t *testing.T) {
	newReport := func() *cudaschedule.CampaignReport {
		run := cudaschedule.RunRecord{Exited: true, PTXDigest: strings.Repeat("1", 64), ExpectedPTXDigest: strings.Repeat("1", 64), PTXDigestSource: "runner_receipt"}
		return &cudaschedule.CampaignReport{SeedStart: 17, SeedEnd: 18, SeedsDrawn: 1, VariantsExecuted: 1, ConfirmRuns: 2, Seeds: []cudaschedule.SeedResult{{Seed: 17, Status: cudaschedule.SeedExecuted, Verdict: cudaschedule.VerdictPass, Run: &run}}}
	}
	if counts, err := checkNativeCampaign(newReport(), true); err != nil || counts.Executed != 1 || counts.Evaluated != 1 {
		t.Fatalf("healthy synthetic report counts=%+v err=%v", counts, err)
	}
	for _, tt := range []struct {
		name   string
		change func(*cudaschedule.CampaignReport)
	}{
		{"incomplete", func(r *cudaschedule.CampaignReport) { r.SeedEnd++ }},
		{"counters", func(r *cudaschedule.CampaignReport) { r.VariantsExecuted = 0 }},
		{"skipped", func(r *cudaschedule.CampaignReport) {
			r.Seeds[0].Verdict = cudaschedule.VerdictSkipped
			r.Seeds[0].Run.Stdout = "SKIPPING: unsupported"
		}},
		{"harness error", func(r *cudaschedule.CampaignReport) {
			r.Seeds[0].Status = cudaschedule.SeedHarnessError
			r.Seeds[0].Verdict = cudaschedule.VerdictHarnessError
			r.HarnessErrors = 1
			r.VariantsExecuted = 0
		}},
		{"empty only", func(r *cudaschedule.CampaignReport) {
			r.Seeds[0].Status = cudaschedule.SeedEmptyDraw
			r.Seeds[0].Verdict = cudaschedule.VerdictEmptyDraw
			r.EmptyDraws = 1
			r.VariantsExecuted = 0
		}},
		{"wrong seed", func(r *cudaschedule.CampaignReport) { r.Seeds[0].Seed++ }},
		{"missing receipt", func(r *cudaschedule.CampaignReport) { r.Seeds[0].Run.PTXDigest = "" }},
		{"finding", func(r *cudaschedule.CampaignReport) {
			r.Seeds[0].Verdict = cudaschedule.VerdictMismatch
			r.Seeds[0].Run.Stdout = "MISMATCH"
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := newReport()
			tt.change(r)
			if _, err := checkNativeCampaign(r, true); err == nil {
				t.Fatal("invalid clean control accepted")
			}
		})
	}
}

func TestNativeMismatchRequiresRepeatedConfirmation(t *testing.T) {
	seed := cudaschedule.SeedResult{Verdict: cudaschedule.VerdictMismatch, Confirmation: &cudaschedule.ConfirmationSummary{Attempts: 2, Findings: 2, Confirmed: true, Outcomes: []cudaschedule.Verdict{cudaschedule.VerdictMismatch, cudaschedule.VerdictMismatch}}}
	if !confirmedNativeMismatch(seed, 2) {
		t.Fatal("synthetic repeated mismatch rejected")
	}
	seed.Confirmation.Outcomes[1] = cudaschedule.VerdictPass
	if confirmedNativeMismatch(seed, 2) {
		t.Fatal("one-off mismatch accepted")
	}
	seed.Confirmation.Outcomes = []cudaschedule.Verdict{cudaschedule.VerdictMismatch}
	seed.Confirmation.Attempts, seed.Confirmation.Findings = 1, 1
	if confirmedNativeMismatch(seed, 1) {
		t.Fatal("single execution accepted as native confirmation")
	}
}

func TestNativeReceiptDoesNotOverwriteEvidence(t *testing.T) {
	dir := t.TempDir()
	if err := writeNativeCampaignReceipt(dir, "clean", nil, errors.New("synthetic failure")); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "clean.json")
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := writeNativeCampaignReceipt(dir, "clean", nil, nil); !errors.Is(err, os.ErrExist) {
		t.Fatalf("repeat receipt error=%v, want existing file", err)
	}
	after, err := os.ReadFile(path)
	if err != nil || string(after) != string(before) || !strings.Contains(string(after), "synthetic failure") {
		t.Fatalf("receipt changed or lost error: data=%q err=%v", after, err)
	}
}

func TestNativeCampaignConfirmationEvidence(t *testing.T) {
	newReport := func() *cudaschedule.CampaignReport {
		digest := strings.Repeat("1", 64)
		run := &cudaschedule.RunRecord{Exited: true, Stdout: "MISMATCH", PTXDigest: digest, ExpectedPTXDigest: digest, PTXDigestSource: "runner_receipt"}
		confirmation := &cudaschedule.ConfirmationSummary{Attempts: 2, Findings: 2, Confirmed: true, Outcomes: []cudaschedule.Verdict{cudaschedule.VerdictMismatch, cudaschedule.VerdictMismatch}}
		return &cudaschedule.CampaignReport{SeedStart: 17, SeedEnd: 18, SeedsDrawn: 1, VariantsExecuted: 1, ConfirmRuns: 2, Seeds: []cudaschedule.SeedResult{{Seed: 17, Status: cudaschedule.SeedExecuted, Verdict: cudaschedule.VerdictMismatch, Run: run, Confirmation: confirmation}}}
	}
	counts, err := checkNativeCampaign(newReport(), false)
	if err != nil || counts.Confirmed != 1 || counts.ConfirmationAttempts != 2 {
		t.Fatalf("repeated synthetic finding: counts=%+v err=%v", counts, err)
	}
	for _, tc := range []struct {
		name   string
		change func(*cudaschedule.ConfirmationSummary)
	}{
		{"missing attempts", func(c *cudaschedule.ConfirmationSummary) { c.Attempts = 1 }},
		{"missing outcome", func(c *cudaschedule.ConfirmationSummary) { c.Outcomes = c.Outcomes[:1] }},
		{"wrong first outcome", func(c *cudaschedule.ConfirmationSummary) { c.Outcomes[0] = cudaschedule.VerdictPass }},
		{"inflated findings", func(c *cudaschedule.ConfirmationSummary) { c.Findings = 3 }},
		{"wrong confirmed flag", func(c *cudaschedule.ConfirmationSummary) { c.Confirmed = false }},
		{"harness error", func(c *cudaschedule.ConfirmationSummary) { c.Outcomes[1] = cudaschedule.VerdictHarnessError }},
		{"skipped", func(c *cudaschedule.ConfirmationSummary) { c.Outcomes[1] = cudaschedule.VerdictSkipped }},
		{"wedged", func(c *cudaschedule.ConfirmationSummary) { c.Outcomes[1] = cudaschedule.VerdictGpuWedged }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			report := newReport()
			tc.change(report.Seeds[0].Confirmation)
			if _, err := checkNativeCampaign(report, false); err == nil {
				t.Fatal("invalid confirmation accepted")
			}
		})
	}
	report := newReport()
	c := report.Seeds[0].Confirmation
	c.Outcomes[1], c.Findings, c.Confirmed = cudaschedule.VerdictPass, 1, false
	if counts, err := checkNativeCampaign(report, false); err != nil || counts.Confirmed != 0 || counts.Findings != 1 {
		t.Fatalf("honest one-off: counts=%+v err=%v", counts, err)
	}
}
