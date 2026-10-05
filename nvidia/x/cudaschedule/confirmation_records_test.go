package cudaschedule_test

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaschedule"
)

func TestCampaignRetainsEachConfirmationRecord(t *testing.T) {
	calls := 0
	opts := cudaschedule.CampaignOptions{
		PTXSource: CAMPAIGN_PTX, SeedStart: 2, SeedEnd: 3,
		Intensity: 2, ConfirmRuns: 3,
		RunVariant: func(_ context.Context, _ uint64, pristine bool, ptx string) (cudaschedule.RunRecord, error) {
			rec := cudaschedule.RunRecord{Exited: true, Stdout: "PASS"}
			if !pristine && strings.Contains(ptx, "nanosleep.u32") {
				calls++
				rec.Stdout = fmt.Sprintf("MISMATCH attempt=%d", calls)
				rec.StderrPath = fmt.Sprintf("attempt-%d.stderr", calls)
			}
			return rec, nil
		},
	}
	report, err := runSyntheticCampaign(context.Background(), opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Seeds) != 1 || report.Seeds[0].Confirmation == nil {
		t.Fatalf("missing finding: %+v", report)
	}
	c := report.Seeds[0].Confirmation
	if len(c.Runs) != 3 || len(c.Outcomes) != 3 {
		t.Fatalf("retained %d runs, %d outcomes", len(c.Runs), len(c.Outcomes))
	}
	for i, run := range c.Runs {
		if run.Stdout != fmt.Sprintf("MISMATCH attempt=%d", i+1) || run.StderrPath != fmt.Sprintf("attempt-%d.stderr", i+1) {
			t.Fatalf("attempt %d lost its record: %+v", i+1, run)
		}
		if run.PTXDigest == "" || run.PTXDigest != run.ExpectedPTXDigest || cudaschedule.Classify(run) != c.Outcomes[i] {
			t.Fatalf("attempt %d lost loaded/classified evidence: %+v", i+1, run)
		}
	}
}
