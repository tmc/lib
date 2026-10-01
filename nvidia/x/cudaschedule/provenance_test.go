package cudaschedule_test

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaschedule"
)

func TestClassifyRequiresLoadedReceipt(t *testing.T) {
	digest := strings.Repeat("1", 64)
	valid := cudaschedule.RunRecord{Exited: true, PTXDigest: digest, ExpectedPTXDigest: digest, PTXDigestSource: "runner_receipt"}
	for _, tc := range []struct {
		name   string
		change func(*cudaschedule.RunRecord)
		want   cudaschedule.Verdict
	}{
		{"valid", func(*cudaschedule.RunRecord) {}, cudaschedule.VerdictPass},
		{"missing loaded", func(r *cudaschedule.RunRecord) { r.PTXDigest = "" }, cudaschedule.VerdictHarnessError},
		{"missing expected", func(r *cudaschedule.RunRecord) { r.ExpectedPTXDigest = "" }, cudaschedule.VerdictHarnessError},
		{"missing both", func(r *cudaschedule.RunRecord) { r.PTXDigest = ""; r.ExpectedPTXDigest = "" }, cudaschedule.VerdictHarnessError},
		{"wrong loaded", func(r *cudaschedule.RunRecord) { r.PTXDigest = strings.Repeat("2", 64) }, cudaschedule.VerdictHarnessError},
		{"invalid matching", func(r *cudaschedule.RunRecord) { r.PTXDigest = "bad"; r.ExpectedPTXDigest = "bad" }, cudaschedule.VerdictHarnessError},
		{"invalid hex", func(r *cudaschedule.RunRecord) {
			r.PTXDigest = strings.Repeat("z", 64)
			r.ExpectedPTXDigest = r.PTXDigest
		}, cudaschedule.VerdictHarnessError},
		{"controller input", func(r *cudaschedule.RunRecord) { r.PTXDigestSource = "campaign_input" }, cudaschedule.VerdictHarnessError},
		{"timeout before receipt", func(r *cudaschedule.RunRecord) { r.TimedOut = true; r.PTXDigest = "" }, cudaschedule.VerdictHarnessError},
		{"timeout after receipt", func(r *cudaschedule.RunRecord) { r.TimedOut = true }, cudaschedule.VerdictHang},
		{"oracle missing", func(r *cudaschedule.RunRecord) { r.OracleAttached = true }, cudaschedule.VerdictHarnessError},
		{"oracle complete", func(r *cudaschedule.RunRecord) {
			r.OracleAttached = true
			r.OraclePID = 1
			r.OraclePatchedModules = 1
			r.OracleInstrumentedAccesses = 1
		}, cudaschedule.VerdictPass},
		{"oracle zero pid", func(r *cudaschedule.RunRecord) {
			r.OracleAttached = true
			r.OraclePatchedModules = 1
			r.OracleInstrumentedAccesses = 1
		}, cudaschedule.VerdictHarnessError},
		{"oracle zero modules", func(r *cudaschedule.RunRecord) {
			r.OracleAttached = true
			r.OraclePID = 1
			r.OracleInstrumentedAccesses = 1
		}, cudaschedule.VerdictHarnessError},
		{"oracle zero accesses", func(r *cudaschedule.RunRecord) { r.OracleAttached = true; r.OraclePID = 1; r.OraclePatchedModules = 1 }, cudaschedule.VerdictHarnessError},
		{"invalid comparison baseline", func(r *cudaschedule.RunRecord) {
			r.CompareOutput = true
			r.Baseline = &cudaschedule.RunRecord{Exited: true}
		}, cudaschedule.VerdictHarnessError},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := valid
			tc.change(&r)
			if got := cudaschedule.Classify(r); got != tc.want {
				t.Fatalf("Classify=%v want %v", got, tc.want)
			}
		})
	}
}

func TestCampaignMissingReceiptIsNotFabricated(t *testing.T) {
	for _, role := range []string{"baseline", "variant", "health", "wrong baseline", "wrong health"} {
		t.Run(role, func(t *testing.T) {
			phase := strings.TrimPrefix(role, "wrong ")
			pristineCalls := 0
			report, err := cudaschedule.RunCampaign(context.Background(), cudaschedule.CampaignOptions{
				PTXSource: CAMPAIGN_PTX, SeedStart: 20, SeedEnd: 21, Intensity: 2, ConfirmRuns: 1, Timeout: 10 * time.Millisecond, PristineTimeout: 8 * time.Second,
				RunVariant: func(ctx context.Context, _ uint64, pristine bool, ptx string) (cudaschedule.RunRecord, error) {
					if pristine {
						pristineCalls++
					}
					r := cudaschedule.RunRecord{Exited: true, PTXDigest: testPTXDigest(ptx)}
					if (phase == "baseline" && pristineCalls == 1) || (phase == "variant" && !pristine) || (phase == "health" && pristine && pristineCalls == 2) {
						r.PTXDigest = ""
						if strings.HasPrefix(role, "wrong ") {
							r.PTXDigest = strings.Repeat("0", 64)
						}
					}
					if phase == "health" && !pristine {
						<-ctx.Done()
						r.TimedOut = true
					}
					return r, nil
				},
			})
			if phase == "baseline" {
				if err == nil || report == nil || report.BaselineVerdict != cudaschedule.VerdictHarnessError {
					t.Fatalf("report=%+v err=%v", report, err)
				}
				return
			}
			if err != nil || len(report.Seeds) != 1 || report.Seeds[0].Verdict != cudaschedule.VerdictHarnessError {
				t.Fatalf("report=%+v err=%v", report, err)
			}
			if role == "variant" && report.Seeds[0].Run.PTXDigest != "" {
				t.Fatal("missing loaded digest was fabricated")
			}
		})
	}
}

const executableReceiptScript = `for path do :; done
if command -v sha256sum >/dev/null 2>&1; then digest=$(sha256sum "$path"); else digest=$(shasum -a 256 "$path"); fi
digest=${digest%% *}
`

func TestCampaignExecutableReceiptRefusals(t *testing.T) {
	for _, mode := range []string{"missing", "wrong", "duplicate", "invalid", "oracle missing", "oracle zero", "oracle wrong pid"} {
		t.Run(mode, func(t *testing.T) {
			script := executableReceiptScript + `if grep -q ptx_schedule "$path"; then
case "$1" in
missing) ;;
wrong) printf 'CUDASCHEDULE_LOADED_PTX_SHA256:%064d\n' 0 ;;
duplicate) printf 'CUDASCHEDULE_LOADED_PTX_SHA256:%s\nCUDASCHEDULE_LOADED_PTX_SHA256:%s\n' "$digest" "$digest" ;;
invalid) printf 'CUDASCHEDULE_LOADED_PTX_SHA256:bad\n' ;;
*) printf 'CUDASCHEDULE_LOADED_PTX_SHA256:%s\n' "$digest" ;;
esac
if [ "$1" = 'oracle zero' ]; then printf 'CUDASCHEDULE_ORACLE_LIVENESS:%d 1 0\n' $$; fi
if [ "$1" = 'oracle wrong pid' ]; then printf 'CUDASCHEDULE_ORACLE_LIVENESS:%d 1 1\n' $(( $$ + 1 )); fi
else
printf 'CUDASCHEDULE_LOADED_PTX_SHA256:%s\n' "$digest"
printf 'CUDASCHEDULE_ORACLE_LIVENESS:%d 1 1\n' $$
fi
echo PASS`
			r, err := cudaschedule.RunCampaign(context.Background(), cudaschedule.CampaignOptions{
				PTXSource: CAMPAIGN_PTX, SeedStart: 20, SeedEnd: 21, Intensity: 2, ConfirmRuns: 1, Timeout: 5 * time.Second,
				Executable: "/bin/sh", Args: []string{"-c", script, "receipt-test", mode}, DeviceProbe: func() bool { return true }, OracleAttached: strings.HasPrefix(mode, "oracle"),
			})
			if err != nil || r == nil || len(r.Seeds) != 1 || r.Seeds[0].Verdict != cudaschedule.VerdictHarnessError {
				t.Fatalf("report=%+v err=%v", r, err)
			}
		})
	}
}

func TestCampaignRetainingArtifactsPreservesChildExit(t *testing.T) {
	script := executableReceiptScript + `printf 'CUDASCHEDULE_LOADED_PTX_SHA256:%s\n' "$digest"
if grep -q ptx_schedule "$path"; then exit 7; fi
echo PASS`
	r, err := cudaschedule.RunCampaign(context.Background(), cudaschedule.CampaignOptions{
		PTXSource: CAMPAIGN_PTX, SeedStart: 20, SeedEnd: 21, Intensity: 2, ConfirmRuns: 1, Timeout: 5 * time.Second,
		Executable: "/bin/sh", Args: []string{"-c", script, "retention-test"}, DeviceProbe: func() bool { return true }, KeepArtifacts: true, ArtifactDir: t.TempDir(),
	})
	if err != nil || r == nil || len(r.Seeds) != 1 || r.Seeds[0].Verdict != cudaschedule.VerdictCrash || r.Seeds[0].Run.ExitCode != 7 {
		t.Fatalf("report=%+v err=%v", r, err)
	}
	for _, p := range []string{r.Seeds[0].Run.PTXPath, r.Seeds[0].Run.StdoutPath, r.Seeds[0].Run.StderrPath, r.Seeds[0].Run.ReplayScriptPath} {
		if _, err := os.Stat(p); err != nil {
			t.Fatal(err)
		}
	}
}

func TestCampaignCalibrationRequiresConfirmedFindingAndCleanRuns(t *testing.T) {
	for _, mode := range []string{"confirmed", "one-off", "clean harness", "no findings", "no executions", "clean skipped", "mixed skipped"} {
		t.Run(mode, func(t *testing.T) {
			plantedCalls := 0
			opts := cudaschedule.CampaignOptions{PTXSource: CAMPAIGN_PTX, SeedStart: 20, SeedEnd: 22, Intensity: 2, ConfirmRuns: 2,
				Arms: &cudaschedule.CampaignArms{PlantedBugPTXSource: CAMPAIGN_PTX + "\n// planted", CleanPTXSource: CAMPAIGN_PTX + "\n// clean"},
				RunVariant: func(_ context.Context, seed uint64, pristine bool, ptx string) (cudaschedule.RunRecord, error) {
					r := cudaschedule.RunRecord{Exited: true}
					if pristine {
						return r, nil
					}
					if strings.Contains(ptx, "// clean") && mode == "clean harness" {
						r.HarnessErr = "broken control"
					}
					if strings.Contains(ptx, "// clean") && (mode == "clean skipped" || (mode == "mixed skipped" && seed == 21)) {
						r.Stdout = "skipping: control unavailable"
					}
					if strings.Contains(ptx, "// planted") {
						plantedCalls++
						if mode != "no findings" && (mode != "one-off" || plantedCalls == 1) {
							r.Stdout = "MISMATCH"
						}
					}
					return r, nil
				},
			}
			if mode == "no executions" {
				opts.Intensity = 0
			}
			r, err := runSyntheticCampaign(context.Background(), opts)
			if err != nil || r == nil || r.Arms == nil {
				t.Fatalf("report=%+v err=%v", r, err)
			}
			if r.CalibrationValidated != (mode == "confirmed") {
				t.Fatalf("calibration=%t arms=%+v", r.CalibrationValidated, r.Arms)
			}
			if mode == "one-off" && r.Arms.PlantedBug.ConfirmedFindings != 0 {
				t.Fatal("one-off marked confirmed")
			}
			if mode == "clean harness" && (r.Arms.CleanControl.HarnessErrors != 2 || r.Arms.CleanControl.Outcome == "clean") {
				t.Fatalf("clean=%+v", r.Arms.CleanControl)
			}
			if (mode == "no executions" || strings.HasSuffix(mode, "skipped")) && r.Arms.CleanControl.Outcome != "incomplete" {
				t.Fatalf("empty control=%+v", r.Arms.CleanControl)
			}
		})
	}
}

func TestCampaignArmsStopAfterGpuWedge(t *testing.T) {
	for _, phase := range []string{"main", "planted"} {
		t.Run(phase, func(t *testing.T) {
			pristineCalls := map[string]int{}
			cleanCalls := 0
			plantedCalls := 0
			r, err := runSyntheticCampaign(context.Background(), cudaschedule.CampaignOptions{
				PTXSource: CAMPAIGN_PTX, SeedStart: 20, SeedEnd: 21, Intensity: 2, ConfirmRuns: 2,
				Arms: &cudaschedule.CampaignArms{PlantedBugPTXSource: CAMPAIGN_PTX + "\n// planted", CleanPTXSource: CAMPAIGN_PTX + "\n// clean"},
				RunVariant: func(_ context.Context, _ uint64, pristine bool, ptx string) (cudaschedule.RunRecord, error) {
					kind := "main"
					if strings.Contains(ptx, "// planted") {
						kind = "planted"
						plantedCalls++
					}
					if strings.Contains(ptx, "// clean") {
						kind = "clean"
						cleanCalls++
					}
					rec := cudaschedule.RunRecord{Exited: true}
					if pristine {
						pristineCalls[kind]++
						if kind == phase && pristineCalls[kind] > 1 {
							rec.ExitCode = 1
						}
						return rec, nil
					}
					if kind == phase {
						rec.TimedOut = true
					}
					return rec, nil
				},
			})
			if err != nil || r == nil || r.Arms == nil || r.CalibrationValidated || cleanCalls != 0 || (phase == "main" && plantedCalls != 0) {
				t.Fatalf("report=%+v err=%v clean=%d planted=%d", r, err, cleanCalls, plantedCalls)
			}
			if r.Arms.CleanControl.Outcome != "not_run_after_gpu_wedge" {
				t.Fatalf("clean outcome=%q", r.Arms.CleanControl.Outcome)
			}
		})
	}
}
