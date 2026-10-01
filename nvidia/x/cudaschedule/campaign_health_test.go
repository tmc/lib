package cudaschedule_test

import (
	"context"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaschedule"
)

func TestCampaignHookHealthTimeoutRecordsGpuWedge(t *testing.T) {
	pristineCalls := 0
	r, err := runSyntheticCampaign(context.Background(), cudaschedule.CampaignOptions{
		PTXSource: CAMPAIGN_PTX, SeedStart: 20, SeedEnd: 21, Intensity: 2,
		ConfirmRuns: 3, Timeout: 8 * time.Second, PristineTimeout: 8 * time.Second,
		RunVariant: func(ctx context.Context, _ uint64, pristine bool, _ string) (cudaschedule.RunRecord, error) {
			if pristine {
				pristineCalls++
				if pristineCalls == 1 {
					return cudaschedule.RunRecord{Exited: true}, nil
				}
				<-ctx.Done()
				return cudaschedule.RunRecord{}, nil
			}
			// Synthetic variant timeout; the health timeout comes from its context.
			return cudaschedule.RunRecord{TimedOut: true}, nil
		},
	})
	if err != nil || r == nil || len(r.Seeds) != 1 || r.Seeds[0].Verdict != cudaschedule.VerdictGpuWedged {
		t.Fatalf("report=%+v err=%v", r, err)
	}
	health := r.Seeds[0].Run.Pristine
	if pristineCalls != 2 || health == nil || !health.TimedOut || !health.PristineRun {
		t.Fatalf("calls=%d health=%+v", pristineCalls, health)
	}
}

func TestCampaignCompareOutputAppliesToConfirmationRuns(t *testing.T) {
	for _, compare := range []bool{false, true} {
		name := "disabled"
		if compare {
			name = "enabled"
		}
		t.Run(name, func(t *testing.T) {
			variants := 0
			r, err := runSyntheticCampaign(context.Background(), cudaschedule.CampaignOptions{
				PTXSource: CAMPAIGN_PTX, SeedStart: 20, SeedEnd: 21, Intensity: 2,
				ConfirmRuns: 3, CompareOutput: compare, Timeout: 8 * time.Second, PristineTimeout: 8 * time.Second,
				RunVariant: func(_ context.Context, _ uint64, pristine bool, _ string) (cudaschedule.RunRecord, error) {
					output := "reference"
					if !pristine {
						variants++
						switch variants {
						case 1:
							output = "MISMATCH initial finding"
						case 3:
							output = "changed confirmation"
						}
					}
					return cudaschedule.RunRecord{Exited: true, Stdout: output}, nil
				},
			})
			if err != nil || r == nil || len(r.Seeds) != 1 || r.Seeds[0].Confirmation == nil {
				t.Fatalf("report=%+v err=%v", r, err)
			}
			c := r.Seeds[0].Confirmation
			last := cudaschedule.VerdictPass
			if compare {
				last = cudaschedule.VerdictOutputChanged
			}
			want := []cudaschedule.Verdict{cudaschedule.VerdictMismatch, cudaschedule.VerdictPass, last}
			if variants != 3 || c.Attempts != 3 || c.Findings != 1 || c.Confirmed || len(c.Outcomes) != 3 {
				t.Fatalf("calls=%d confirmation=%+v", variants, c)
			}
			for i, v := range want {
				if c.Outcomes[i] != v {
					t.Fatalf("CompareOutput=%t outcomes=%v want%v", compare, c.Outcomes, want)
				}
			}
		})
	}
}
