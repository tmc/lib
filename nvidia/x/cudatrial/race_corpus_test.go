package cudatrial

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudaliveness"
	"github.com/tmc/lib/nvidia/x/cudashared"
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
	tc, _ := corpus.Lookup("false_positives/shared_bytewise_readwrite")
	accesses, err := caseAccesses(tc, false)
	if err != nil {
		t.Fatal(err)
	}
	m := cudashared.NewHazardMatrix("sm_90")
	for _, a := range accesses {
		if err := m.AddAccess(a); err != nil {
			t.Fatal(err)
		}
	}
	records, err := m.Evaluate()
	if err != nil {
		t.Fatal(err)
	}
	if len(records) != 1 || !records[0].WordConflict || records[0].ByteConflict {
		t.Fatalf("false-positive model records = %+v, want exactly one word-only conflict", records)
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
			for _, fixed := range []bool{false, true} {
				res, err := corpus.EvaluateCase(tc.Name, fixed, "sm_80")
				if err == nil {
					t.Errorf("case %s: expected error on sm_80, got nil", tc.Name)
				}
				if !errors.Is(err, cudaerrors.ErrHardwarePending) {
					t.Errorf("case %s: expected ErrHardwarePending, got %v", tc.Name, err)
				}
				if res.Passed() {
					t.Errorf("case %s fixed=%v: falsely reported passed on unsupported hardware", tc.Name, fixed)
				}
			}
		}
	}
}

func TestControl_DetectorNeverReachedIsNotACleanSweep(t *testing.T) {
	// Point the runner at a stub RUN_CMD that always exits 0 and never invokes any detector.
	// The run must FAIL. Without this control a 27/27 clean sweep is unfalsifiable.
	corpus := NewRaceCorpus()
	ctx := context.Background()
	root := makeCorpusBinaries(t, corpus, "sm_90")
	t.Setenv("CUHADRON_BINARY_ROOT", root)

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
	passed, total, rate, rateErr := corpus.ConformanceRate(results)
	if rateErr == nil {
		t.Fatal("ConformanceRate without lane 52 exec-control succeeded")
	}
	if total != 0 || rate != 0 || passed != 0 {
		t.Errorf("rate reported without exec-control: %d/%d, rate=%.2f", passed, total, rate)
	}
}

func TestControl_EmptyArchIsPending(t *testing.T) {
	c := NewRaceCorpus()
	for _, name := range []string{"dsmem/shared_readwrite_race", "bulkcpy/shared_readwrite_race"} {
		res, err := c.EvaluateCase(name, false, "")
		if !errors.Is(err, cudaerrors.ErrHardwarePending) || !res.HardwarePending || res.Passed() {
			t.Errorf("%s empty arch = (%+v, %v), want pending refusal", name, res, err)
		}
	}
	root := makeCorpusBinaries(t, c, "")
	t.Setenv("CUHADRON_BINARY_ROOT", root)
	if _, err := c.RunExternal(context.Background(), "true", false, ""); !errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Fatalf("RunExternal empty arch = %v, want ErrHardwarePending", err)
	}
}

func TestCorpus_EveryCaseEvaluatedByModel(t *testing.T) {
	c := NewRaceCorpus()
	for _, tc := range c.TestCases() {
		for _, fixed := range []bool{false, true} {
			res, err := c.EvaluateCase(tc.Name, fixed, "sm_90")
			if tc.Scope == cudashared.ScopeMultiGPU {
				if err == nil || res.Passed() {
					t.Errorf("%s fixed=%v was not refused as unmodelled", tc.Name, fixed)
				}
				continue
			}
			if err != nil || res.EvaluatedPairs < 1 {
				t.Errorf("%s fixed=%v: pairs=%d err=%v", tc.Name, fixed, res.EvaluatedPairs, err)
			}
			if tc.FixedSupported && !fixed && !res.HasRace {
				t.Errorf("race case %s fixed=0 was relabelled or evaluated clean", tc.Name)
			}
			if !tc.FixedSupported && res.HasRace {
				t.Errorf("false-positive case %s reports a race", tc.Name)
			}
			accesses, err := caseAccesses(tc, fixed)
			if err != nil {
				t.Fatal(err)
			}
			if len(accesses) != 2 || accesses[0].Space != tc.Memory || accesses[0].Scope != tc.Scope {
				t.Errorf("%s model does not match declared taxonomy", tc.Name)
			}
			m := cudashared.NewHazardMatrix("sm_90")
			for _, a := range accesses {
				if err := m.AddAccess(a); err != nil {
					t.Fatal(err)
				}
			}
			gotHazard := cudashared.HazardNone
			if accesses[0].IsWrite && accesses[1].IsWrite {
				gotHazard = cudashared.HazardWriteWrite
			} else if accesses[0].IsWrite {
				gotHazard = cudashared.HazardWriteRead
			} else if accesses[1].IsWrite {
				gotHazard = cudashared.HazardReadWrite
			}
			if gotHazard != tc.Hazard {
				t.Errorf("%s taxonomy hazard=%v, access model=%v", tc.Name, tc.Hazard, gotHazard)
			}
		}
	}
}

func TestRunExternalUsesFixedBinaryAndKnownRaceControl(t *testing.T) {
	c := NewRaceCorpus()
	root := makeCorpusBinaries(t, c, "sm_90")
	t.Setenv("CUHADRON_BINARY_ROOT", root)
	results, err := c.RunExternal(context.Background(), `sh -c 'case "$1" in */FIXED=0/*) exit 1;; *) exit 0;; esac' sh`, true, "sm_90")
	if err != nil {
		t.Fatal(err)
	}
	if len(results) == 0 || results[0].BinaryFixed || !results[0].HasRace || results[0].Fixed != true {
		t.Fatalf("fixed=1 sweep did not run the independent FIXED=0 control: %+v", results[0])
	}
	var sample EvaluationResult
	for _, r := range results {
		if r.TestCase == "memcpy/shared_readwrite_race" {
			sample = r
			break
		}
	}
	if sample.BinaryFixed != true || !strings.Contains(sample.BinaryPath, "FIXED=1") {
		t.Fatalf("candidate path lacks FIXED=1 provenance: %+v", sample)
	}
	results = append(results, EvaluationResult{TestCase: "lane52/exec-control", IsExecControl: true, ExecControlPass: true, Configuration: "FIXED=1;ARCH=sm_90"})
	if _, _, _, err := c.ConformanceRate(results); err != nil {
		t.Fatalf("matching lane52 exec-control rejected: %v", err)
	}
}

func TestControl_DetectorNeverReachedFixed1(t *testing.T) {
	c := NewRaceCorpus()
	t.Setenv("CUHADRON_BINARY_ROOT", makeCorpusBinaries(t, c, "sm_90"))
	_, err := c.RunExternal(context.Background(), "true", true, "sm_90")
	if !errors.Is(err, ErrDetectorNeverReached) {
		t.Fatalf("fixed=1 clean stub = %v, want ErrDetectorNeverReached", err)
	}
}

func TestHazardMatrix_SubwarpAndHostDeviceControls(t *testing.T) {
	c := NewRaceCorpus()
	for _, name := range []string{"intersubwarp/shared_readwrite_race", "hostdevice/global_readwrite_race"} {
		tc, _ := c.Lookup(name)
		for _, fixed := range []bool{false, true} {
			accesses, err := caseAccesses(tc, fixed)
			if err != nil {
				t.Fatal(err)
			}
			m := cudashared.NewHazardMatrix("sm_90")
			for _, a := range accesses {
				if err := m.AddAccess(a); err != nil {
					t.Fatal(err)
				}
			}
			hasRace, err := m.HasRace()
			if err != nil {
				t.Fatal(err)
			}
			if hasRace == fixed {
				t.Errorf("%s fixed=%v hasRace=%v", name, fixed, hasRace)
			}
		}
	}
}

func TestControl_NonOneExitIsNotDetection(t *testing.T) {
	c := NewRaceCorpus()
	root := makeCorpusBinaries(t, c, "sm_90")
	t.Setenv("CUHADRON_BINARY_ROOT", root)
	cmd := `sh -c 'case "$1" in */FIXED=0/*) exit 2;; *) exit 0;; esac' sh`
	results, err := c.RunExternal(context.Background(), cmd, false, "sm_90")
	if err == nil {
		t.Fatal("exit 2 was accepted as a race result")
	}
	if len(results) == 0 || results[0].ExitCode != 2 || results[0].Passed() {
		t.Fatalf("unknown control exit treated as detection: %+v", results[0])
	}
	if _, err := c.RunExternal(context.Background(), "/definitely/missing/cuhadron-runner", false, "sm_90"); err == nil {
		t.Fatal("missing RUN_CMD succeeded")
	}
}

func TestConformanceRateRejectsMismatchedExecConfiguration(t *testing.T) {
	c := NewRaceCorpus()
	results := []EvaluationResult{
		{TestCase: "lane52/exec-control", IsExecControl: true, ExecControlPass: true, Configuration: "FIXED=0;ARCH=sm_90"},
		{TestCase: "memcpy/shared_readwrite_race", Category: CategoryMemcpy, Fixed: true, Configuration: "FIXED=1;ARCH=sm_90"},
	}
	if _, _, _, err := c.ConformanceRate(results); err == nil {
		t.Fatal("mismatched exec-control configuration accepted")
	}
}

func makeCorpusBinaries(t *testing.T, c *RaceCorpus, arch string) string {
	t.Helper()
	root := t.TempDir()
	for _, fixed := range []string{"FIXED=0", "FIXED=1"} {
		for _, tc := range c.TestCases() {
			path := filepath.Join(root, fixed, tc.Name+"."+arch+".out")
			if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path, nil, 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}
	return root
}
