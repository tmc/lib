package cudaschedule_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaschedule"
)

type corpusRecord struct {
	Provenance      string                 `json:"provenance"`
	ExpectedVerdict string                 `json:"expected_verdict"`
	Record          cudaschedule.RunRecord `json:"record"`
}

func TestClassify_EveryKindFromCorpus(t *testing.T) {
	files, err := filepath.Glob("testdata/*.json")
	if err != nil {
		t.Fatalf("Glob testdata/*.json: %v", err)
	}
	if len(files) < 8 {
		t.Fatalf("expected at least 8 corpus records in testdata/, found %d", len(files))
	}

	seenVerdicts := make(map[cudaschedule.Verdict]bool)

	for _, file := range files {
		data, err := os.ReadFile(file)
		if err != nil {
			t.Fatalf("ReadFile(%q): %v", file, err)
		}
		var cr corpusRecord
		if err := json.Unmarshal(data, &cr); err != nil {
			t.Fatalf("Unmarshal(%q): %v", file, err)
		}

		verdict := cudaschedule.Classify(cr.Record)
		if verdict.String() != cr.ExpectedVerdict {
			t.Errorf("file %q classified as %s (%d), want %s",
				file, verdict, verdict, cr.ExpectedVerdict)
		}
		seenVerdicts[verdict] = true
	}

	// Verify all 8 kinds were covered by the corpus
	allKinds := []cudaschedule.Verdict{
		cudaschedule.VerdictPass,
		cudaschedule.VerdictSkipped,
		cudaschedule.VerdictHang,
		cudaschedule.VerdictCrash,
		cudaschedule.VerdictMismatch,
		cudaschedule.VerdictOutputChanged,
		cudaschedule.VerdictGpuWedged,
		cudaschedule.VerdictHarnessError,
	}

	for _, k := range allKinds {
		if !seenVerdicts[k] {
			t.Errorf("verdict %s (%d) was not covered in corpus", k, k)
		}
	}
}

func TestClassify_WedgeRequiresPristineFailure(t *testing.T) {
	// Hang with passing pristine re-run stays Hang
	hangWithGoodPristine := cudaschedule.RunRecord{
		TimedOut: true,
		Pristine: &cudaschedule.RunRecord{
			Exited:   true,
			ExitCode: 0,
			Stdout:   "PASS: pristine re-run succeeded\n",
		},
	}
	if v := cudaschedule.Classify(hangWithGoodPristine); v != cudaschedule.VerdictHang {
		t.Errorf("Hang with good pristine classified as %s, want Hang", v)
	}

	// The same Hang with failing pristine re-run (e.g. also timed out or crashed) becomes GpuWedged
	hangWithFailedPristine := cudaschedule.RunRecord{
		TimedOut: true,
		Pristine: &cudaschedule.RunRecord{
			TimedOut: true,
			Stderr:   "device unresponsive\n",
		},
	}
	if v := cudaschedule.Classify(hangWithFailedPristine); v != cudaschedule.VerdictGpuWedged {
		t.Errorf("Hang with failed pristine classified as %s, want GpuWedged", v)
	}
}

func TestControl_ClassifierDistinguishesMismatch(t *testing.T) {
	mismatchRecord := cudaschedule.RunRecord{
		Exited:   true,
		ExitCode: 0,
		Stdout:   "MISMATCH at index 5: computed 42.0 != expected 0.0\n",
	}

	verdict := cudaschedule.Classify(mismatchRecord)
	if verdict == cudaschedule.VerdictPass {
		t.Fatalf("dead classifier: Mismatch classified as Pass!")
	}
	if verdict != cudaschedule.VerdictMismatch {
		t.Fatalf("classified as %s, want Mismatch", verdict)
	}
	if !verdict.IsFinding() {
		t.Fatalf("Mismatch must be a schedule finding")
	}
}

func TestControl_OneOffReported(t *testing.T) {
	// Confirmation summary representing 1 reproduction out of 5 confirmation runs
	summary := cudaschedule.ConfirmationSummary{
		Attempts:  5,
		Findings:  1,
		Confirmed: false,
		Outcomes: []cudaschedule.Verdict{
			cudaschedule.VerdictMismatch,
			cudaschedule.VerdictPass,
			cudaschedule.VerdictPass,
			cudaschedule.VerdictPass,
			cudaschedule.VerdictPass,
		},
	}

	if summary.Confirmed {
		t.Fatalf("one-off finding was marked as confirmed!")
	}
	if summary.Findings != 1 {
		t.Fatalf("findings count = %d, want 1", summary.Findings)
	}
	if summary.Attempts != 5 {
		t.Fatalf("attempts count = %d, want 5", summary.Attempts)
	}
}

func TestControl_HarnessErrorNotFinding(t *testing.T) {
	harnessErr := cudaschedule.RunRecord{
		HarnessErr: "ptx patch failed: objcopy error",
	}

	verdict := cudaschedule.Classify(harnessErr)
	if verdict != cudaschedule.VerdictHarnessError {
		t.Fatalf("classified as %s, want HarnessError", verdict)
	}
	if verdict.IsFinding() {
		t.Fatalf("HarnessError was incorrectly counted as a schedule finding")
	}

	report := cudaschedule.CampaignReport{
		Seeds: []cudaschedule.SeedResult{
			{
				Seed:    1,
				Status:  cudaschedule.SeedHarnessError,
				Verdict: cudaschedule.VerdictHarnessError,
				Run:     &harnessErr,
			},
		},
	}
	if report.FindingCount() != 0 {
		t.Fatalf("report finding count with only HarnessError was %d, want 0", report.FindingCount())
	}
}

func TestVerdict_FindingTaxonomyPartition(t *testing.T) {
	findings := map[cudaschedule.Verdict]bool{
		cudaschedule.VerdictMismatch:      true,
		cudaschedule.VerdictOutputChanged: true,
		cudaschedule.VerdictHang:          true,
		cudaschedule.VerdictCrash:         true,
		cudaschedule.VerdictGpuWedged:     true,
	}
	nonFindings := map[cudaschedule.Verdict]bool{
		cudaschedule.VerdictPass:         true,
		cudaschedule.VerdictSkipped:      true,
		cudaschedule.VerdictHarnessError: true,
		cudaschedule.VerdictEmptyDraw:    true,
	}

	for v := range findings {
		if !v.IsFinding() {
			t.Errorf("verdict %s should be a finding", v)
		}
		if v.FindingLabel() == "" {
			t.Errorf("verdict %s missing finding label", v)
		}
	}

	for v := range nonFindings {
		if v.IsFinding() {
			t.Errorf("verdict %s should NOT be a finding", v)
		}
		if v.FindingLabel() != "" {
			t.Errorf("non-finding %s has unexpected finding label: %q", v, v.FindingLabel())
		}
	}
}
