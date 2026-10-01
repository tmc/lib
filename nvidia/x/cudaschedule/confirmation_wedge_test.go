package cudaschedule_test

import (
	"context"
	"crypto/sha256"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaschedule"
)

func TestCampaignConfirmationWedgeStopsDeviceReuse(t *testing.T) {
	data, err := os.ReadFile("testdata/schedule_race.ptx")
	if err != nil {
		t.Fatal(err)
	}
	for _, phase := range []string{"main", "planted"} {
		for _, keepGoing := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/keepGoing=%t", phase, keepGoing), func(t *testing.T) {
				variants := map[string]int{}
				calls := map[string]int{}
				wedged, postWedge := false, 0
				r, err := cudaschedule.RunCampaign(context.Background(), cudaschedule.CampaignOptions{
					PTXSource: string(data), SeedStart: 20, SeedEnd: 22, Intensity: 2,
					ConfirmRuns: 3, KeepGoing: keepGoing, Timeout: 8 * time.Second, PristineTimeout: 8 * time.Second,
					Arms: &cudaschedule.CampaignArms{PlantedBugPTXSource: string(data) + "\n// planted", CleanPTXSource: string(data) + "\n// clean"},
					RunVariant: func(_ context.Context, _ uint64, pristine bool, ptx string) (cudaschedule.RunRecord, error) {
						kind := "main"
						if strings.Contains(ptx, "// planted") {
							kind = "planted"
						} else if strings.Contains(ptx, "// clean") {
							kind = "clean"
						}
						calls[kind]++
						if wedged {
							postWedge++
						}
						// These receipts model the supplied host-only PTX workload;
						// they are not proof of a CUDA load or physical device state.
						rec := cudaschedule.RunRecord{Exited: true, PTXDigest: fmt.Sprintf("%x", sha256.Sum256([]byte(ptx)))}
						if kind == phase {
							if pristine && variants[kind] == 2 {
								rec.ExitCode = 1
								wedged = true
							} else if !pristine {
								variants[kind]++
								switch variants[kind] {
								case 1:
									rec.Stdout = "MISMATCH initial finding"
								case 2:
									rec.TimedOut = true
								}
							}
						}
						return rec, nil
					},
				})
				if err != nil || r == nil || r.Arms == nil || !wedged {
					t.Fatalf("report=%+v err=%v wedged=%t", r, err, wedged)
				}
				affected := r
				if phase == "planted" {
					affected = r.Arms.PlantedBug.Report
				}
				if affected == nil || len(affected.Seeds) == 0 || affected.Seeds[0].Verdict != cudaschedule.VerdictMismatch {
					t.Fatalf("initial verdict not retained: %+v", affected)
				}
				c := affected.Seeds[0].Confirmation
				attempts := 2
				if keepGoing {
					attempts = 3
				}
				if c == nil || c.Attempts != attempts || len(c.Outcomes) != attempts || c.Outcomes[1] != cudaschedule.VerdictGpuWedged || c.Confirmed || c.Findings != 1 {
					t.Fatalf("confirmation=%+v, want %d attempts with nested wedge", c, attempts)
				}
				if keepGoing {
					if postWedge == 0 || variants[phase] != 4 || len(affected.Seeds) != 2 || calls["clean"] == 0 || calls["planted"] == 0 || c.Outcomes[2] != cudaschedule.VerdictPass {
						t.Fatalf("explicit continuation lost: post=%d variants=%v calls=%v seeds=%d", postWedge, variants, calls, len(affected.Seeds))
					}
				} else {
					if postWedge != 0 || variants[phase] != 2 || len(affected.Seeds) != 1 || calls["clean"] != 0 || (phase == "main" && calls["planted"] != 0) {
						t.Fatalf("device reused after wedge: post=%d variants=%v calls=%v seeds=%d", postWedge, variants, calls, len(affected.Seeds))
					}
					if r.Arms.CleanControl.Outcome != "not_run_after_gpu_wedge" {
						t.Fatalf("clean outcome=%q", r.Arms.CleanControl.Outcome)
					}
				}
				if r.CalibrationValidated {
					t.Fatal("confirmation wedge must not validate this calibration")
				}
			})
		}
	}
}
