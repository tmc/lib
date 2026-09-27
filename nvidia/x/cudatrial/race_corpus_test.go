package cudatrial

import (
	"context"
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudaliveness"
)

func TestCorpus_AsyncCopyPipelineRace(t *testing.T) {
	corpus := NewRaceCorpus()
	res, err := corpus.EvaluateCase("memcpy/shared_readwrite_race", false /* FIXED=0 */, "sm_80")
	if err != nil {
		t.Fatalf("EvaluateCase: %v", err)
	}
	if !res.HasRace {
		t.Errorf("expected async copy pipeline race detected on FIXED=0")
	}
	if !res.Passed() {
		t.Errorf("expected test case to pass validation on detecting race in FIXED=0")
	}
}

func TestCorpus_AsyncCopyPipelineSynchronized(t *testing.T) {
	corpus := NewRaceCorpus()
	res, err := corpus.EvaluateCase("memcpy/shared_readwrite_race", true /* FIXED=1 */, "sm_80")
	if err != nil {
		t.Fatalf("EvaluateCase: %v", err)
	}
	if res.HasRace {
		t.Errorf("expected synchronized async copy pipeline in FIXED=1 to have no races")
	}
	if !res.Passed() {
		t.Errorf("expected test case to pass validation in FIXED=1")
	}
}

func TestCorpus_DSMEMClusterSharedRace(t *testing.T) {
	corpus := NewRaceCorpus()
	res, err := corpus.EvaluateCase("dsmem/shared_readwrite_race", false /* FIXED=0 */, "sm_90")
	if err != nil {
		t.Fatalf("EvaluateCase: %v", err)
	}
	if !res.HasRace {
		t.Errorf("expected DSMEM cluster race detected on FIXED=0")
	}
}

func TestCorpus_SubwarpDivergenceRace(t *testing.T) {
	corpus := NewRaceCorpus()
	resRace, err := corpus.EvaluateCase("intersubwarp/shared_readwrite_race", false /* FIXED=0 */, "sm_80")
	if err != nil {
		t.Fatalf("EvaluateCase FIXED=0: %v", err)
	}
	if !resRace.HasRace {
		t.Errorf("expected inter-subwarp divergence race on FIXED=0")
	}

	resFixed, err := corpus.EvaluateCase("intersubwarp/shared_readwrite_race", true /* FIXED=1 */, "sm_80")
	if err != nil {
		t.Fatalf("EvaluateCase FIXED=1: %v", err)
	}
	if resFixed.HasRace {
		t.Errorf("expected no race in fixed inter-subwarp case")
	}
}

func TestCorpus_UVMHostDeviceCoherenceRace(t *testing.T) {
	corpus := NewRaceCorpus()
	resRace, err := corpus.EvaluateCase("hostdevice/global_readwrite_race", false /* FIXED=0 */, "sm_80")
	if err != nil {
		t.Fatalf("EvaluateCase FIXED=0: %v", err)
	}
	if !resRace.HasRace {
		t.Errorf("expected UVM host-device race on FIXED=0")
	}

	resFixed, err := corpus.EvaluateCase("hostdevice/global_readwrite_race", true /* FIXED=1 */, "sm_80")
	if err != nil {
		t.Fatalf("EvaluateCase FIXED=1: %v", err)
	}
	if resFixed.HasRace {
		t.Errorf("expected no race on synchronized UVM host-device case")
	}
}

func TestCorpus_ByteIndependenceFalsePositive(t *testing.T) {
	corpus := NewRaceCorpus()
	res, err := corpus.EvaluateCase("false_positives/shared_bytewise_readwrite", false, "sm_90")
	if err != nil {
		t.Fatalf("EvaluateCase false_positives: %v", err)
	}
	if res.HasRace {
		t.Errorf("false positive: byte-independent accesses incorrectly reported as data race")
	}
	if !res.Passed() {
		t.Errorf("expected false positive test case to pass validation")
	}
}

func TestCorpus_HardwareRequirementPendingReport(t *testing.T) {
	corpus := NewRaceCorpus()
	res, err := corpus.EvaluateCase("dsmem/shared_readwrite_race", false, "sm_80")
	if err == nil {
		t.Fatalf("expected error evaluating sm_90+ DSMEM case on sm_80 target")
	}
	if !errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Errorf("expected ErrHardwarePending, got %v", err)
	}
	if !res.HardwarePending {
		t.Errorf("expected HardwarePending == true on evaluation result")
	}
}

func TestControl_MutantMaskedSyncFailsOnFixed0(t *testing.T) {
	// Omitting a required barrier or cp.async.wait_group in a FIXED=0 candidate
	// must trigger a race detection failure.
	corpus := NewRaceCorpus()
	res, err := corpus.EvaluateCase("bulkcpy/shared_readwrite_race", false /* FIXED=0 */, "sm_90")
	if err != nil {
		t.Fatalf("EvaluateCase: %v", err)
	}
	if !res.HasRace {
		t.Errorf("mutant masked sync failed to detect race in FIXED=0 candidate")
	}
	if res.ReportedDefects < 1 {
		t.Errorf("expected at least 1 reported defect on FIXED=0, got %d", res.ReportedDefects)
	}
}

func TestControl_FalsePositiveWordOverlapClearsOnFixed1(t *testing.T) {
	// Sub-word byte accesses on adjacent bytes within a 32-bit word under FIXED=1
	// must yield zero race findings.
	corpus := NewRaceCorpus()
	res, err := corpus.EvaluateCase("false_positives/shared_bytewise_writewrite", true /* FIXED=1 */, "sm_90")
	if err != nil {
		t.Fatalf("EvaluateCase: %v", err)
	}
	if res.HasRace {
		t.Errorf("sub-word byte accesses produced false positive race finding")
	}
	if res.ReportedDefects != 0 {
		t.Errorf("expected 0 reported defects, got %d", res.ReportedDefects)
	}
}

func TestControl_HardwarePendingFalsifier(t *testing.T) {
	// Executing sm_90+ cluster or DSMEM test cases on sm_80 physical GPUs must
	// return ErrHardwarePending and must never report false success.
	corpus := NewRaceCorpus()
	for _, tc := range corpus.TestCases() {
		if tc.MinArch == "sm_90" {
			res, err := corpus.EvaluateCase(tc.Name, false, "sm_80")
			if err == nil {
				t.Errorf("case %s: expected error on sm_80, got nil", tc.Name)
			}
			if !errors.Is(err, cudaerrors.ErrHardwarePending) {
				t.Errorf("case %s: expected ErrHardwarePending, got %v", tc.Name, err)
			}
			if res.Passed() {
				t.Errorf("case %s: falsely reported passed on unsupported hardware", tc.Name)
			}
		}
	}
}

func TestControl_DetectorNeverReachedIsNotACleanSweep(t *testing.T) {
	// Point the runner at a stub RUN_CMD that always exits 0 and never invokes any detector.
	// The run must FAIL. Without this control a 27/27 clean sweep is unfalsifiable.
	corpus := NewRaceCorpus()
	ctx := context.Background()

	// Stub command that always exits 0 without detecting any races
	stubCmd := "true"

	results, err := corpus.RunExternal(ctx, stubCmd, false /* FIXED=0 */, "sm_90")
	if err == nil {
		t.Fatalf("stub RUN_CMD exiting 0 on FIXED=0 must FAIL, but returned nil error")
	}

	if !errors.Is(err, ErrDetectorNeverReached) && !errors.Is(err, cudaliveness.ErrNotLive) {
		t.Errorf("expected ErrDetectorNeverReached or ErrNotLive, got %v", err)
	}

	// Verify that the conformance rate for a 0-exit stub on FIXED=0 is NOT 100%.
	passed, total, rate := corpus.ConformanceRate(results)
	if passed == total || rate == 1.0 {
		t.Errorf("falsifier failed: stub runner reported 100%% sweep (%d/%d, rate=%.2f)", passed, total, rate)
	}
}
