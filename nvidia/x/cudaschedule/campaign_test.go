package cudaschedule_test

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudaschedule"
)

const CAMPAIGN_PTX = `.version 8.7
.target sm_90
.address_size 64

.visible .entry simple_kernel(
    .param .u64 data
)
{
    .reg .pred %p0;
    .reg .b32 %r0;
    .reg .b64 %rd1;
L_loop:
    atom.global.add.u32 %r0, [%rd1], 1;
    @%p0 bra L_loop;
    ret;
}
`

func TestCampaign_RejectsZeroConfirmRuns(t *testing.T) {
	ctx := context.Background()
	opts := cudaschedule.CampaignOptions{
		PTXSource:   CAMPAIGN_PTX,
		SeedStart:   0,
		SeedEnd:     10,
		ConfirmRuns: 0, // Must be rejected
	}

	_, err := cudaschedule.RunCampaign(ctx, opts)
	if err == nil {
		t.Fatalf("RunCampaign succeeded with ConfirmRuns=0, want error")
	}
	if !strings.Contains(err.Error(), "confirm_runs must be at least 1") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestCampaign_RejectsInvalidSeedRange(t *testing.T) {
	ctx := context.Background()
	opts := cudaschedule.CampaignOptions{
		PTXSource:   CAMPAIGN_PTX,
		SeedStart:   10,
		SeedEnd:     10, // Invalid: start must be less than end
		ConfirmRuns: 1,
	}

	_, err := cudaschedule.RunCampaign(ctx, opts)
	if err == nil {
		t.Fatalf("RunCampaign succeeded with SeedStart >= SeedEnd, want error")
	}
	if !strings.Contains(err.Error(), "invalid seed range") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestCampaign_PendingWithoutDevice(t *testing.T) {
	ctx := context.Background()
	opts := cudaschedule.CampaignOptions{
		PTXSource:   CAMPAIGN_PTX,
		SeedStart:   0,
		SeedEnd:     10,
		ConfirmRuns: 1,
		DeviceProbe: func() bool { return false }, // Explicitly no device
	}

	report, err := cudaschedule.RunCampaign(ctx, opts)
	if !errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Fatalf("RunCampaign without device returned %v, want cudaerrors.ErrHardwarePending", err)
	}
	if report != nil && len(report.Seeds) > 0 {
		t.Fatalf("expected no verdicts reported when device is pending, got %d seeds", len(report.Seeds))
	}
}

func TestCampaign_PluggableRunnerSweep(t *testing.T) {
	ctx := context.Background()

	// Seed 2 causes Mismatch; others Pass
	runner := func(ctx context.Context, seed uint64, _ bool, ptx string) (cudaschedule.RunRecord, error) {
		if strings.Contains(ptx, "nanosleep.u32") && strings.Contains(ptx, "seed=2") {
			return cudaschedule.RunRecord{
				Exited:   true,
				ExitCode: 0,
				Stdout:   "MISMATCH detected in race interleaving\n",
			}, nil
		}
		return cudaschedule.RunRecord{
			Exited:   true,
			ExitCode: 0,
			Stdout:   "PASS: all checks passed\n",
		}, nil
	}

	opts := cudaschedule.CampaignOptions{
		PTXSource:   CAMPAIGN_PTX,
		SeedStart:   0,
		SeedEnd:     5,
		Intensity:   2.0,
		ConfirmRuns: 3,
		RunVariant:  runner,
	}

	report, err := cudaschedule.RunCampaign(ctx, opts)
	if err != nil {
		t.Fatalf("RunCampaign failed: %v", err)
	}

	if len(report.Seeds) != 5 {
		t.Fatalf("report contains %d seeds, want 5", len(report.Seeds))
	}

	findingCount := report.FindingCount()
	if findingCount != 1 {
		t.Fatalf("report.FindingCount() = %d, want 1", findingCount)
	}

	// Verify seed 2 result
	seed2 := report.Seeds[2]
	if seed2.Verdict != cudaschedule.VerdictMismatch {
		t.Errorf("seed 2 verdict = %v, want Mismatch", seed2.Verdict)
	}
	if seed2.Confirmation == nil {
		t.Fatalf("seed 2 missing confirmation summary")
	}
	if !seed2.Confirmation.Confirmed {
		t.Errorf("seed 2 expected to be confirmed across 3 confirmation runs")
	}
	if seed2.Confirmation.Findings != 3 {
		t.Errorf("seed 2 findings = %d, want 3", seed2.Confirmation.Findings)
	}
}

func TestCampaign_DefaultRunnerUsesVariant(t *testing.T) {
	ctx := context.Background()
	builds := 0
	opts := cudaschedule.CampaignOptions{
		PTXSource: CAMPAIGN_PTX,
		SeedStart: 1, SeedEnd: 4, Intensity: 2, ConfirmRuns: 1,
		DeviceProbe: func() bool { return true },
		BuildVariant: func(_ context.Context, _ uint64, _ bool, ptxPath string) (string, []string, error) {
			builds++
			return "/bin/sh", []string{"-c", `cksum < "$1"`, "variant-hash"}, nil
		},
	}
	report, err := cudaschedule.RunCampaign(ctx, opts)
	if err != nil {
		t.Fatal(err)
	}
	if report.BaselineVerdict != cudaschedule.VerdictPass || len(report.Seeds) != 3 || builds != 4 {
		t.Fatalf("baseline/seeds/builds = %v/%d/%d, want Pass/3/4", report.BaselineVerdict, len(report.Seeds), builds)
	}
	seen := make(map[string]uint64)
	for _, seed := range report.Seeds {
		if seed.Run.Stdout == report.Baseline.Stdout {
			t.Errorf("seed %d executable saw pristine PTX digest: %q", seed.Seed, seed.Run.Stdout)
		}
		if previous, ok := seen[seed.Run.Stdout]; ok {
			t.Errorf("seeds %d and %d produced the same variant content digest %q", previous, seed.Seed, seed.Run.Stdout)
		}
		seen[seed.Run.Stdout] = seed.Seed
	}
}

func TestCampaign_CompareOutputUsesBaselineAtCampaignLevel(t *testing.T) {
	for _, compare := range []bool{false, true} {
		t.Run(fmt.Sprintf("compare=%t", compare), func(t *testing.T) {
			report, err := cudaschedule.RunCampaign(context.Background(), cudaschedule.CampaignOptions{
				PTXSource: CAMPAIGN_PTX, SeedStart: 1, SeedEnd: 2, Intensity: 2,
				ConfirmRuns: 1, CompareOutput: compare,
				RunVariant: func(_ context.Context, _ uint64, pristine bool, _ string) (cudaschedule.RunRecord, error) {
					output := "variant stdout"
					if pristine {
						output = "baseline stdout"
					}
					return cudaschedule.RunRecord{Exited: true, Stdout: output}, nil
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			if len(report.Seeds) != 1 {
				t.Fatalf("seed rows = %d, want 1", len(report.Seeds))
			}
			want := cudaschedule.VerdictPass
			if compare {
				want = cudaschedule.VerdictOutputChanged
			}
			if got := report.Seeds[0].Verdict; got != want {
				t.Fatalf("CompareOutput=%t verdict = %v, want %v", compare, got, want)
			}
		})
	}
}

func TestCampaign_ExecPathWatchdogRecordsTimeout(t *testing.T) {
	report, err := cudaschedule.RunCampaign(context.Background(), cudaschedule.CampaignOptions{
		PTXSource: CAMPAIGN_PTX, SeedStart: 3, SeedEnd: 4, Intensity: 2,
		ConfirmRuns: 1, Timeout: 100 * time.Millisecond, PristineTimeout: 10 * time.Second,
		DeviceProbe: func() bool { return true },
		BuildVariant: func(_ context.Context, _ uint64, pristine bool, _ string) (string, []string, error) {
			if pristine {
				return "/bin/sh", []string{"-c", "echo PASS"}, nil
			}
			return "/bin/sh", []string{"-c", "exec sleep 1"}, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Seeds) != 1 || report.Seeds[0].Verdict != cudaschedule.VerdictHang || report.Seeds[0].Run == nil || !report.Seeds[0].Run.TimedOut {
		t.Fatalf("exec-path watchdog result = %+v", report.Seeds)
	}
}

func TestCampaign_BuildTimeIsOutsideRunTimeout(t *testing.T) {
	var variantBuilds int
	report, err := cudaschedule.RunCampaign(context.Background(), cudaschedule.CampaignOptions{
		PTXSource: CAMPAIGN_PTX, SeedStart: 20, SeedEnd: 21, Intensity: 2,
		ConfirmRuns: 1, Timeout: 2 * time.Second, PristineTimeout: 8 * time.Second, BuildTimeout: 12 * time.Second,
		DeviceProbe: func() bool { return true },
		Executable:  "/bin/sh",
		Args:        []string{"-c", "echo PASS"},
		BuildVariant: func(_ context.Context, _ uint64, pristine bool, _ string) (string, []string, error) {
			if !pristine {
				variantBuilds++
				time.Sleep(4 * time.Second)
			}
			return "/bin/sh", []string{"-c", "echo PASS"}, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if variantBuilds != 1 || report.BaselineVerdict != cudaschedule.VerdictPass || len(report.Seeds) != 1 || report.Seeds[0].Verdict != cudaschedule.VerdictPass {
		t.Fatalf("slow build was included in run timeout: builds=%d report=%+v", variantBuilds, report)
	}
}

func TestCampaign_BuildFailureIsHarnessErrorAndSweepContinues(t *testing.T) {
	report, err := cudaschedule.RunCampaign(context.Background(), cudaschedule.CampaignOptions{
		PTXSource: CAMPAIGN_PTX, SeedStart: 30, SeedEnd: 32, Intensity: 2,
		ConfirmRuns: 1, Timeout: time.Second, DeviceProbe: func() bool { return true },
		BuildVariant: func(_ context.Context, seed uint64, pristine bool, _ string) (string, []string, error) {
			if !pristine && seed == 30 {
				return "", nil, fmt.Errorf("compiler rejected variant")
			}
			return "/bin/sh", []string{"-c", "echo PASS"}, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Seeds) != 2 || report.Seeds[0].Status != cudaschedule.SeedHarnessError || report.Seeds[0].Verdict != cudaschedule.VerdictHarnessError || report.Seeds[0].Verdict.IsFinding() {
		t.Fatalf("build failure seed = %+v", report.Seeds)
	}
	if report.Seeds[1].Verdict != cudaschedule.VerdictPass || report.FindingCount() != 0 {
		t.Fatalf("sweep did not continue after build error: %+v", report.Seeds)
	}
	if report.SeedsDrawn != 2 || report.HarnessErrors != 1 || report.VariantsExecuted != 1 || report.EmptyDraws != 0 {
		t.Fatalf("seed partition counters = %+v", report)
	}
}

func TestCampaign_BuildWithoutRunnableArtifactIsHarnessError(t *testing.T) {
	report, err := cudaschedule.RunCampaign(context.Background(), cudaschedule.CampaignOptions{
		PTXSource: CAMPAIGN_PTX, SeedStart: 40, SeedEnd: 41, Intensity: 2,
		ConfirmRuns: 1, Timeout: time.Second, DeviceProbe: func() bool { return true },
		BuildVariant: func(_ context.Context, _ uint64, pristine bool, _ string) (string, []string, error) {
			if pristine {
				return "/bin/sh", []string{"-c", "echo PASS"}, nil
			}
			return "/path/that/does/not/exist/cudaschedule-runner", nil, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Seeds) != 1 || report.Seeds[0].Status != cudaschedule.SeedHarnessError || report.Seeds[0].Verdict != cudaschedule.VerdictHarnessError || report.VariantsExecuted != 0 {
		t.Fatalf("missing runnable artifact was counted as execution: %+v", report)
	}
}

func TestCampaign_BuildHookSeesPristineAndDistinctImages(t *testing.T) {
	type build struct {
		seed     uint64
		pristine bool
		path     string
		ptx      string
	}
	var builds []build
	report, err := cudaschedule.RunCampaign(context.Background(), cudaschedule.CampaignOptions{
		PTXSource: CAMPAIGN_PTX, SeedStart: 50, SeedEnd: 51, Intensity: 2,
		ConfirmRuns: 1, Timeout: 2 * time.Second, PristineTimeout: 8 * time.Second, DeviceProbe: func() bool { return true },
		BuildVariant: func(_ context.Context, seed uint64, pristine bool, path string) (string, []string, error) {
			data, err := os.ReadFile(path)
			if err != nil {
				return "", nil, err
			}
			builds = append(builds, build{seed: seed, pristine: pristine, path: path, ptx: string(data)})
			return "/bin/sh", []string{"-c", `grep -q nanosleep "$1" && exec sleep 10 || echo PASS`, "image-check"}, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(builds) != 3 || len(report.Seeds) != 1 || report.Seeds[0].Run == nil || report.Seeds[0].Run.Pristine == nil {
		t.Fatalf("build/run records = %+v / %+v", builds, report.Seeds)
	}
	if !builds[0].pristine || builds[0].ptx != CAMPAIGN_PTX {
		t.Fatalf("baseline build was not marked pristine: %+v", builds[0])
	}
	if builds[1].pristine || !strings.Contains(builds[1].ptx, "nanosleep.u32") {
		t.Fatalf("variant build flag/content disagree: %+v", builds[1])
	}
	if !builds[2].pristine || builds[2].ptx != CAMPAIGN_PTX {
		t.Fatalf("health build was not marked pristine: %+v", builds[2])
	}
	if builds[0].path == builds[1].path || builds[1].path == builds[2].path {
		t.Fatalf("pristine and variant paths collide: %+v", builds)
	}
}

func TestCampaign_PristineRunUsesSeparateTimeout(t *testing.T) {
	pristineRuns := 0
	report, err := cudaschedule.RunCampaign(context.Background(), cudaschedule.CampaignOptions{
		PTXSource: CAMPAIGN_PTX, SeedStart: 51, SeedEnd: 52, Intensity: 10,
		ConfirmRuns: 1, Timeout: time.Second, PristineTimeout: 5 * time.Second,
		RunVariant: func(ctx context.Context, _ uint64, pristine bool, _ string) (cudaschedule.RunRecord, error) {
			if pristine {
				pristineRuns++
				time.Sleep(2 * time.Second)
				return cudaschedule.RunRecord{Exited: true, Stdout: "PASS"}, nil
			}
			<-ctx.Done()
			return cudaschedule.RunRecord{}, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if pristineRuns != 2 || report.BaselineVerdict != cudaschedule.VerdictPass || len(report.Seeds) != 1 || report.Seeds[0].Verdict != cudaschedule.VerdictHang {
		t.Fatalf("separate baseline and health timeouts were not applied: pristine runs=%d report=%+v", pristineRuns, report)
	}
	if report.Seeds[0].Run.Pristine == nil || report.Seeds[0].Run.Pristine.TimedOut || !report.Seeds[0].Run.Pristine.PristineRun {
		t.Fatalf("health re-run did not pass under the pristine timeout: %+v", report.Seeds[0].Run)
	}
}

func TestCampaign_RunnerReceivesSeed(t *testing.T) {
	type call struct {
		seed uint64
		ptx  string
	}
	var calls []call
	firstVariant := true
	runner := func(_ context.Context, seed uint64, _ bool, ptx string) (cudaschedule.RunRecord, error) {
		calls = append(calls, call{seed, ptx})
		if ptx != CAMPAIGN_PTX && seed == 5 && firstVariant {
			firstVariant = false
			return cudaschedule.RunRecord{Exited: true, Stdout: "MISMATCH"}, nil
		}
		if ptx != CAMPAIGN_PTX && seed == 6 {
			return cudaschedule.RunRecord{TimedOut: true}, nil
		}
		return cudaschedule.RunRecord{Exited: true, Stdout: "PASS"}, nil
	}
	report, err := cudaschedule.RunCampaign(context.Background(), cudaschedule.CampaignOptions{
		PTXSource: CAMPAIGN_PTX, SeedStart: 5, SeedEnd: 8, Intensity: 2,
		ConfirmRuns: 3, RunVariant: runner,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Seeds) != 3 || !report.Baseline.PristineRun {
		t.Fatalf("report = %+v; want three seeds and a marked pristine baseline", report)
	}
	if report.Seeds[0].Seed != 5 || report.Seeds[0].Confirmation == nil || report.Seeds[0].Confirmation.Attempts != 3 {
		t.Fatalf("seed 5 confirmation = %+v", report.Seeds[0])
	}
	seenVariants := map[uint64]int{}
	for _, c := range calls {
		if c.ptx != CAMPAIGN_PTX {
			seenVariants[c.seed]++
		}
	}
	if seenVariants[5] != 3 || seenVariants[6] != 3 || seenVariants[7] != 1 {
		t.Fatalf("variant calls by seed = %v, want 5:3, 6:3, 7:1", seenVariants)
	}
	if report.Seeds[1].Run.Pristine == nil || !report.Seeds[1].Run.Pristine.PristineRun {
		t.Fatalf("seed 6 timeout lacks marked pristine health check: %+v", report.Seeds[1].Run)
	}
	pristineSeed6 := 0
	for _, c := range calls {
		if c.seed == 6 && c.ptx == CAMPAIGN_PTX {
			pristineSeed6++
		}
	}
	if pristineSeed6 != 3 {
		t.Fatalf("seed 6 pristine health calls = %d, want one per timeout attempt", pristineSeed6)
	}
	for _, seed := range report.Seeds {
		if seed.Run.PristineRun {
			t.Errorf("variant seed %d was marked pristine", seed.Seed)
		}
	}
}

func TestCampaign_RaceFixtureSweepReplayWithFakeRunner(t *testing.T) {
	readFixture := func(name string) string {
		t.Helper()
		data, err := os.ReadFile(filepath.Join("testdata", name))
		if err != nil {
			t.Fatal(err)
		}
		return string(data)
	}
	racePTX := readFixture("schedule_race.ptx")
	cleanPTX := readFixture("schedule_race_clean.ptx")
	const start, end = uint64(17), uint64(49)
	var findingSeed uint64
	for seed := start; seed < end; seed++ {
		rewrite, err := cudaschedule.Perturb(racePTX, cudaschedule.Options{Seed: seed, Intensity: 2})
		if err != nil {
			t.Fatal(err)
		}
		maxDelay := uint64(2 * rewrite.Report.SitesTotal * cudaschedule.DefaultMaxSleepNs)
		if rewrite.Report.InjectedNsPerVisit > maxDelay {
			t.Fatalf("seed %d injected delay %d ns exceeds the per-site ceiling", seed, rewrite.Report.InjectedNsPerVisit)
		}
		if rewrite.Report.SitesInjected > 0 {
			findingSeed = seed
			break
		}
	}
	if findingSeed == 0 {
		t.Fatal("fixed seed interval has no non-empty race draw")
	}

	run := func(source string, mismatchSeed uint64, images map[uint64]string) *cudaschedule.CampaignReport {
		report, err := cudaschedule.RunCampaign(context.Background(), cudaschedule.CampaignOptions{
			PTXSource: source, SeedStart: start, SeedEnd: end, Intensity: 2,
			ConfirmRuns: 1, Timeout: 3 * time.Second, RunVariant: func(_ context.Context, seed uint64, pristine bool, ptx string) (cudaschedule.RunRecord, error) {
				if !pristine {
					images[seed] = ptx
					if seed == mismatchSeed && strings.Contains(ptx, "nanosleep.u32") {
						return cudaschedule.RunRecord{Exited: true, Stdout: "MISMATCH: shared handoff read before write"}, nil
					}
				}
				return cudaschedule.RunRecord{Exited: true, Stdout: "PASS"}, nil
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		return report
	}
	cleanImages := make(map[uint64]string)
	clean := run(cleanPTX, 0, cleanImages)
	if clean.FindingCount() != 0 || len(clean.Seeds) != int(end-start) {
		t.Fatalf("synchronized control campaign found a defect: %+v", clean.Seeds)
	}

	sweepImages := make(map[uint64]string)
	sweep := run(racePTX, findingSeed, sweepImages)
	var selected *cudaschedule.SeedResult
	for i := range sweep.Seeds {
		if sweep.Seeds[i].Seed == findingSeed {
			selected = &sweep.Seeds[i]
			break
		}
	}
	if selected == nil || selected.Verdict != cudaschedule.VerdictMismatch {
		t.Fatalf("selected race seed %d did not produce the planted mismatch: %+v", findingSeed, sweep.Seeds)
	}
	replay, err := cudaschedule.RunCampaign(context.Background(), cudaschedule.CampaignOptions{
		PTXSource: racePTX, SeedStart: findingSeed, SeedEnd: findingSeed + 1,
		Intensity: 2, ConfirmRuns: 1, Timeout: 3 * time.Second,
		RunVariant: func(_ context.Context, seed uint64, pristine bool, ptx string) (cudaschedule.RunRecord, error) {
			if !pristine {
				sweepImages[seed+1000] = ptx
				return cudaschedule.RunRecord{Exited: true, Stdout: "MISMATCH: shared handoff read before write"}, nil
			}
			return cudaschedule.RunRecord{Exited: true, Stdout: "PASS"}, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(replay.Seeds) != 1 || replay.Seeds[0].Verdict != cudaschedule.VerdictMismatch || sweepImages[findingSeed] != sweepImages[findingSeed+1000] {
		t.Fatalf("seed %d replay differs: sweep=%+v replay=%+v", findingSeed, selected, replay.Seeds)
	}
}

func TestCampaign_BrokenBaselineRunsNoVariants(t *testing.T) {
	var variantRuns int
	report, err := cudaschedule.RunCampaign(context.Background(), cudaschedule.CampaignOptions{
		PTXSource: CAMPAIGN_PTX, SeedStart: 1, SeedEnd: 4, Intensity: 2, ConfirmRuns: 1,
		RunVariant: func(_ context.Context, _ uint64, _ bool, _ string) (cudaschedule.RunRecord, error) {
			variantRuns++
			return cudaschedule.RunRecord{Exited: true, Stdout: "MISMATCH"}, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if report.BaselineVerdict != cudaschedule.VerdictMismatch || report.BaselineStatus != "Broken" || len(report.Seeds) != 0 || variantRuns != 1 {
		t.Fatalf("baseline=%v/%s seeds=%d calls=%d, want Mismatch/Broken/0/1", report.BaselineVerdict, report.BaselineStatus, len(report.Seeds), variantRuns)
	}
}

func TestCampaign_BrokenBaselinePrecedesAnalysis(t *testing.T) {
	report, err := cudaschedule.RunCampaign(context.Background(), cudaschedule.CampaignOptions{
		PTXSource: "not valid PTX", SeedStart: 1, SeedEnd: 2, ConfirmRuns: 1,
		RunVariant: func(_ context.Context, _ uint64, _ bool, _ string) (cudaschedule.RunRecord, error) {
			return cudaschedule.RunRecord{Exited: true, Stdout: "MISMATCH"}, nil
		},
	})
	if err != nil {
		t.Fatalf("RunCampaign error = %v, want baseline report before PTX analysis", err)
	}
	if report.BaselineStatus != "Broken" || report.BaselineVerdict != cudaschedule.VerdictMismatch || len(report.Seeds) != 0 {
		t.Fatalf("malformed broken baseline report = %+v", report)
	}
}

func TestCampaign_SkippedBaselineIsDeclined(t *testing.T) {
	report, err := cudaschedule.RunCampaign(context.Background(), cudaschedule.CampaignOptions{
		PTXSource: CAMPAIGN_PTX, SeedStart: 1, SeedEnd: 2, ConfirmRuns: 1,
		RunVariant: func(_ context.Context, _ uint64, _ bool, _ string) (cudaschedule.RunRecord, error) {
			return cudaschedule.RunRecord{Exited: true, Stdout: "skipping: unsupported target"}, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if report.BaselineVerdict != cudaschedule.VerdictSkipped || report.BaselineStatus != "Declined" || len(report.Seeds) != 0 {
		t.Fatalf("skipped baseline = %+v, want Declined and no seeds", report)
	}
}

func TestCampaign_EmptyDrawNotAFinding(t *testing.T) {
	const oneSite = `.version 8.7
.target sm_90
.address_size 64
.visible .entry only(){ atom.global.add.u32 %r, [%rd], 1; ret; }
`
	seed := uint64(0)
	for ; seed < 10000; seed++ {
		r, err := cudaschedule.Perturb(oneSite, cudaschedule.Options{Seed: seed, Intensity: 0.5})
		if err != nil {
			t.Fatal(err)
		}
		if r.Report.SitesInjected == 0 {
			break
		}
	}
	if seed == 10000 {
		t.Fatal("failed to find non-trivial empty draw")
	}
	var variantRuns int
	report, err := cudaschedule.RunCampaign(context.Background(), cudaschedule.CampaignOptions{
		PTXSource: oneSite, SeedStart: seed, SeedEnd: seed + 1, Intensity: 0.5, ConfirmRuns: 1,
		RunVariant: func(_ context.Context, _ uint64, _ bool, _ string) (cudaschedule.RunRecord, error) {
			variantRuns++
			return cudaschedule.RunRecord{Exited: true}, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Seeds) != 1 || report.Seeds[0].Report.SitesInjected != 0 || report.Seeds[0].Verdict != cudaschedule.VerdictEmptyDraw || variantRuns != 1 {
		t.Fatalf("empty draw result=%+v variant calls=%d, want recorded non-finding and baseline only", report.Seeds, variantRuns)
	}
	if report.Seeds[0].Run != nil || report.Seeds[0].Status != cudaschedule.SeedEmptyDraw {
		t.Fatalf("empty draw unexpectedly ran: %+v", report.Seeds[0])
	}
	if report.SeedsDrawn != 1 || report.EmptyDraws != 1 || report.VariantsExecuted != 0 || report.HarnessErrors != 0 {
		t.Fatalf("empty draw counters = %+v", report)
	}
}

func TestCampaign_OneOffReported(t *testing.T) {
	variantRuns := 0
	report, err := cudaschedule.RunCampaign(context.Background(), cudaschedule.CampaignOptions{
		PTXSource: CAMPAIGN_PTX, SeedStart: 2, SeedEnd: 3, Intensity: 2, ConfirmRuns: 5,
		RunVariant: func(_ context.Context, _ uint64, _ bool, ptx string) (cudaschedule.RunRecord, error) {
			if ptx == CAMPAIGN_PTX {
				return cudaschedule.RunRecord{Exited: true, Stdout: "PASS"}, nil
			}
			variantRuns++
			if variantRuns == 1 {
				return cudaschedule.RunRecord{Exited: true, Stdout: "MISMATCH"}, nil
			}
			return cudaschedule.RunRecord{Exited: true, Stdout: "PASS"}, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	s := report.Seeds[0].Confirmation
	if s == nil || s.Confirmed || s.Findings != 1 || s.Attempts != 5 || len(s.Outcomes) != 5 || variantRuns != 5 {
		t.Fatalf("one-off confirmation = %+v, variant calls=%d", s, variantRuns)
	}
}

func TestCampaign_HarnessErrorNotFinding(t *testing.T) {
	report, err := cudaschedule.RunCampaign(context.Background(), cudaschedule.CampaignOptions{
		PTXSource: CAMPAIGN_PTX, SeedStart: 3, SeedEnd: 6, Intensity: 2, ConfirmRuns: 1,
		RunVariant: func(_ context.Context, seed uint64, _ bool, ptx string) (cudaschedule.RunRecord, error) {
			if ptx == CAMPAIGN_PTX {
				return cudaschedule.RunRecord{Exited: true, Stdout: "PASS"}, nil
			}
			if seed == 3 {
				return cudaschedule.RunRecord{}, fmt.Errorf("compile failed")
			}
			return cudaschedule.RunRecord{Exited: true, Stdout: "PASS"}, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Seeds) != 3 || report.Seeds[0].Verdict != cudaschedule.VerdictHarnessError || report.FindingCount() != 0 {
		t.Fatalf("campaign results = %+v, finding count %d", report.Seeds, report.FindingCount())
	}
}

func TestCampaign_WedgeAndKeepGoing(t *testing.T) {
	run := func(keepGoing bool) *cudaschedule.CampaignReport {
		report, err := cudaschedule.RunCampaign(context.Background(), cudaschedule.CampaignOptions{
			PTXSource: CAMPAIGN_PTX, SeedStart: 0, SeedEnd: 3, Intensity: 2,
			ConfirmRuns: 1, KeepGoing: keepGoing,
			RunVariant: func(_ context.Context, seed uint64, _ bool, ptx string) (cudaschedule.RunRecord, error) {
				if ptx == CAMPAIGN_PTX {
					if seed == 0 {
						return cudaschedule.RunRecord{Exited: true, Stdout: "PASS"}, nil
					}
					return cudaschedule.RunRecord{Exited: true, ExitCode: 1}, nil
				}
				return cudaschedule.RunRecord{TimedOut: true}, nil
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		return report
	}
	stop := run(false)
	if len(stop.Seeds) != 2 || stop.Seeds[0].Verdict != cudaschedule.VerdictHang || stop.Seeds[1].Verdict != cudaschedule.VerdictGpuWedged {
		t.Fatalf("stop-on-wedge results = %+v", stop.Seeds)
	}
	keep := run(true)
	if len(keep.Seeds) != 3 || keep.Seeds[1].Verdict != cudaschedule.VerdictGpuWedged {
		t.Fatalf("keep-going results = %+v", keep.Seeds)
	}
	if got := cudaschedule.Classify(cudaschedule.RunRecord{TimedOut: true, Pristine: &cudaschedule.RunRecord{HarnessErr: "probe failed"}}); got != cudaschedule.VerdictHarnessError {
		t.Fatalf("timeout with pristine harness error = %v, want HarnessError", got)
	}
}

func TestCampaign_CancellationHasNoSeedVerdict(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	report, err := cudaschedule.RunCampaign(ctx, cudaschedule.CampaignOptions{
		PTXSource: CAMPAIGN_PTX, SeedStart: 2, SeedEnd: 3, Intensity: 2, ConfirmRuns: 1,
		RunVariant: func(_ context.Context, _ uint64, _ bool, ptx string) (cudaschedule.RunRecord, error) {
			if ptx == CAMPAIGN_PTX {
				return cudaschedule.RunRecord{Exited: true, Stdout: "PASS"}, nil
			}
			cancel()
			return cudaschedule.RunRecord{Exited: true, ExitCode: 1}, nil
		},
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("RunCampaign error = %v, want context.Canceled", err)
	}
	if report == nil || len(report.Seeds) != 0 {
		t.Fatalf("canceled run recorded a seed verdict: %+v", report)
	}
}

func TestCampaign_DeadlineHasNoSeedVerdict(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	report, err := cudaschedule.RunCampaign(ctx, cudaschedule.CampaignOptions{
		PTXSource: CAMPAIGN_PTX, SeedStart: 4, SeedEnd: 5, Intensity: 2, ConfirmRuns: 1,
		RunVariant: func(ctx context.Context, _ uint64, _ bool, ptx string) (cudaschedule.RunRecord, error) {
			if ptx == CAMPAIGN_PTX {
				return cudaschedule.RunRecord{Exited: true, Stdout: "PASS"}, nil
			}
			<-ctx.Done()
			return cudaschedule.RunRecord{}, nil
		},
	})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("RunCampaign error = %v, want context deadline", err)
	}
	if report == nil || len(report.Seeds) != 0 {
		t.Fatalf("deadline recorded a seed verdict: %+v", report)
	}
}

func TestCampaign_ParentDeadlineIsNotWatchdog(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
	defer cancel()
	report, err := cudaschedule.RunCampaign(ctx, cudaschedule.CampaignOptions{
		PTXSource: CAMPAIGN_PTX, Executable: "/bin/sh",
		Args:      []string{"-c", `grep -q nanosleep "$1" && exec sleep 5 || echo PASS`, "parent-deadline"},
		SeedStart: 10, SeedEnd: 11, Intensity: 2, ConfirmRuns: 1, Timeout: 5 * time.Second,
		DeviceProbe: func() bool { return true },
	})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("RunCampaign error = %v, want parent deadline", err)
	}
	if report == nil || len(report.Seeds) != 0 {
		t.Fatalf("parent deadline was recorded as a seed verdict: %+v", report)
	}
}

func TestCampaign_WaitDelayClosesInheritedPipe(t *testing.T) {
	start := time.Now()
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	tmp := filepath.Join(home, "tmp")
	if err := os.MkdirAll(tmp, 0o700); err != nil {
		t.Fatal(err)
	}
	pid, err := os.CreateTemp(tmp, "cudaschedule-child-*.pid")
	if err != nil {
		t.Fatal(err)
	}
	pidFile := pid.Name()
	if err := pid.Close(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if data, readErr := os.ReadFile(pidFile); readErr == nil {
			if childPID, parseErr := strconv.Atoi(strings.TrimSpace(string(data))); parseErr == nil {
				if child, findErr := os.FindProcess(childPID); findErr == nil {
					_ = child.Kill()
				}
			}
		}
		_ = os.Remove(pidFile)
	})
	report, err := cudaschedule.RunCampaign(context.Background(), cudaschedule.CampaignOptions{
		PTXSource: CAMPAIGN_PTX,
		SeedStart: 9, SeedEnd: 10, Intensity: 2, ConfirmRuns: 1, Timeout: 5 * time.Second, PristineTimeout: 8 * time.Second,
		DeviceProbe: func() bool { return true },
		BuildVariant: func(_ context.Context, _ uint64, _ bool, _ string) (string, []string, error) {
			return "/bin/sh", []string{"-c", `grep -q nanosleep "$2" && { sleep 30 & echo $! > "$1"; wait; } || echo PASS`, "wait-delay", pidFile}, nil
		},
	})
	elapsed := time.Since(start)
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Seeds) != 1 || report.Seeds[0].Verdict != cudaschedule.VerdictHang {
		t.Fatalf("wait-delay campaign = %+v", report.Seeds)
	}
	if report.Seeds[0].Run.Pristine == nil || !report.Seeds[0].Run.Pristine.PristineRun {
		t.Fatalf("timeout health check is not marked pristine: %+v", report.Seeds[0].Run)
	}
	if elapsed > 15*time.Second {
		t.Fatalf("watchdog plus WaitDelay took %s", elapsed)
	}
	if data, err := os.ReadFile(pidFile); err != nil || len(strings.TrimSpace(string(data))) == 0 {
		t.Fatalf("grandchild pid file outside campaign directory is missing: %q, %v", data, err)
	}
}
