package cudatrial

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudashared"
	"github.com/tmc/lib/nvidia/x/cudatma"
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
	for _, tc := range NewRaceCorpus().TestCases() {
		if tc.Category != CategoryFalsePositives {
			continue
		}
		t.Run(tc.Name, func(t *testing.T) {
			res, err := NewRaceCorpus().EvaluateCase(tc.Name, false, "sm_90")
			if err != nil {
				t.Fatalf("EvaluateCase: %v", err)
			}
			if res.HasRace || !res.Passed() {
				t.Fatalf("false-positive result = %+v", res)
			}
			accesses, err := caseAccesses(tc.Name, false)
			if err != nil {
				t.Fatal(err)
			}
			if accesses[0].ThreadID != 0 || accesses[1].ThreadID != 32 || accesses[0].SizeBytes != 1 || accesses[1].SizeBytes != 1 || accesses[1].Offset != accesses[0].Offset+1 || accesses[0].Offset&^3 != accesses[1].Offset&^3 {
				t.Fatalf("access placement = %+v, want threads 0/32 on adjacent bytes in one word", accesses)
			}
			records, err := evaluateRecords("sm_90", accesses)
			if err != nil {
				t.Fatal(err)
			}
			if len(records) != 1 || !records[0].WordConflict || records[0].ByteConflict || records[0].IsRace {
				t.Fatalf("false-positive model records = %+v, want one word-only non-race", records)
			}
		})
	}
}

func evaluateRecords(arch string, accesses []cudashared.MemoryAccess) ([]cudashared.HazardRecord, error) {
	m := cudashared.NewHazardMatrix(arch)
	for _, a := range accesses {
		if err := m.AddAccess(a); err != nil {
			return nil, err
		}
	}
	return m.Evaluate()
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
	if res.Error == nil || !errors.Is(res.Error, cudaerrors.ErrHardwarePending) {
		t.Errorf("pending result error = %v, want ErrHardwarePending", res.Error)
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

func TestControl_NeverReachedRunHasNoRate(t *testing.T) {
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

	if !errors.Is(err, ErrDetectorNeverReached) {
		t.Errorf("expected ErrDetectorNeverReached, got %v", err)
	}

	// Matching lane 52 evidence cannot excuse the failed planted-race control.
	control := makeExecControl(t, "FIXED=0;ARCH=sm_90")
	_, rateErr := corpus.ConformanceRate(results, control)
	if rateErr == nil {
		t.Fatal("ConformanceRate accepted a run whose planted race was not detected")
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
				if err == nil || res.Passed() || !res.Unmodelled {
					t.Errorf("%s fixed=%v was not refused as unmodelled", tc.Name, fixed)
				}
				continue
			}
			if err != nil {
				t.Errorf("%s fixed=%v: pairs=%d err=%v", tc.Name, fixed, res.EvaluatedPairs, err)
			}
			if tc.FixedSupported && !fixed && !res.HasRace {
				t.Errorf("race case %s fixed=0 was relabelled or evaluated clean", tc.Name)
			}
			if !tc.FixedSupported && res.HasRace {
				t.Errorf("false-positive case %s reports a race", tc.Name)
			}
			accesses, err := caseAccesses(tc.Name, fixed)
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
			records, err := m.Evaluate()
			if err != nil {
				t.Fatal(err)
			}
			if len(records) != res.EvaluatedPairs {
				t.Errorf("%s fixed=%v evaluated %d pairs, matrix returned %d records", tc.Name, fixed, res.EvaluatedPairs, len(records))
			}
			wantRace := false
			for _, record := range records {
				wantRace = wantRace || record.IsRace
			}
			if res.HasRace != wantRace {
				t.Errorf("%s fixed=%v verdict=%v, hazard records imply %v", tc.Name, fixed, res.HasRace, wantRace)
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

func TestCasewiseModelTracksUpstreamFixMechanisms(t *testing.T) {
	c := NewRaceCorpus()
	disjoint := []string{
		"interkernel/global_writewrite_race",
		"interwave/global_writewrite_race",
		"bulkcpy/global_writewrite_race",
		"memcpy/shared_writewrite_race",
	}
	for _, name := range disjoint {
		accesses, err := caseAccesses(name, true)
		if err != nil {
			t.Fatal(err)
		}
		matrix := cudashared.NewHazardMatrix("sm_121a")
		for _, access := range accesses {
			if err := matrix.AddAccess(access); err != nil {
				t.Fatal(err)
			}
		}
		records, err := matrix.Evaluate()
		if err != nil {
			t.Fatal(err)
		}
		if len(records) != 0 {
			t.Errorf("%s FIXED=1 records = %+v, want disjoint accesses", name, records)
		}
	}
	for name, model := range accessModels {
		if model.fix != "sync" {
			continue
		}
		accesses, err := caseAccesses(name, true)
		if err != nil {
			t.Fatal(err)
		}
		matrix := cudashared.NewHazardMatrix("sm_121a")
		for _, access := range accesses {
			if err := matrix.AddAccess(access); err != nil {
				t.Fatal(err)
			}
		}
		records, err := matrix.Evaluate()
		if err != nil {
			t.Fatal(err)
		}
		if len(records) != 1 || records[0].AccessA.Stage == records[0].AccessB.Stage || !records[0].WordConflict {
			t.Errorf("%s FIXED=1 records = %+v, want overlapping accesses in different stages", name, records)
		}
	}
	for _, name := range []string{"bulkcpy/global_writewrite_race", "interwave/global_writewrite_race"} {
		accesses, err := caseAccesses(name, false)
		if err != nil {
			t.Fatal(err)
		}
		if accesses[0].BlockID == accesses[1].BlockID {
			t.Errorf("%s did not model distinct blocks", name)
		}
	}
	interkernel, err := caseAccesses("interkernel/global_writewrite_race", false)
	if err != nil {
		t.Fatal(err)
	}
	if interkernel[0].AgentID == interkernel[1].AgentID || interkernel[0].StreamID == interkernel[1].StreamID || interkernel[0].ThreadID != interkernel[1].ThreadID || interkernel[0].BlockID != interkernel[1].BlockID {
		t.Fatal("interkernel accesses do not isolate different agents at the same execution coordinates")
	}
	dsmem, err := caseAccesses("dsmem/shared_readwrite_race", false)
	if err != nil {
		t.Fatal(err)
	}
	if dsmem[0].BlockID == dsmem[1].BlockID || dsmem[0].ClusterID != dsmem[1].ClusterID {
		t.Fatal("DSMEM case does not model two CTAs in one cluster")
	}
	bulkAccesses, err := caseAccesses("bulkcpy/shared_readwrite_race", false)
	if err != nil || bulkAccesses[0].IsWrite || !bulkAccesses[1].IsWrite {
		t.Fatalf("bulkcpy shared-source case is not read-then-write: accesses=%+v err=%v", bulkAccesses, err)
	}
	if len(accessModels) != len(c.TestCases()) {
		t.Fatalf("case model table has %d entries for %d corpus cases", len(accessModels), len(c.TestCases()))
	}
}

func TestBulkcpySharedReadWriteUsesWARGenerator(t *testing.T) {
	bad, err := cudatma.GenerateTMASharedReadWriteScenario(false)
	if err != nil {
		t.Fatal(err)
	}
	if len(bad.Hazards) != 1 || bad.Hazards[0].Kind != "WAR" {
		t.Fatalf("bulkcpy/shared_readwrite_race generator has hazards %+v, want WAR", bad.Hazards)
	}
	wrong, err := cudatma.GenerateTMAScenario(false)
	if err != nil {
		t.Fatal(err)
	}
	if len(wrong.Hazards) == 0 || wrong.Hazards[0].Kind != "RAW" {
		t.Fatalf("negative control has hazards %+v, want RAW", wrong.Hazards)
	}
}

func TestConformanceRateValidatesExactCorpusAndControls(t *testing.T) {
	c := NewRaceCorpus()
	control := makeExecControl(t, "FIXED=1;ARCH=sm_90")
	good := validConformanceResults(c, control.Configuration())
	tests := []struct {
		name   string
		mutate func([]EvaluationResult) []EvaluationResult
	}{
		{"duplicate case", func(r []EvaluationResult) []EvaluationResult { return append(r, r[1]) }},
		{"unknown case", func(r []EvaluationResult) []EvaluationResult {
			return append(r, EvaluationResult{TestCase: "not/in/corpus", Configuration: control.Configuration()})
		}},
		{"missing case", func(r []EvaluationResult) []EvaluationResult { return append(r[:1], r[2:]...) }},
		{"failing row renamed to control", func(r []EvaluationResult) []EvaluationResult {
			r[1].TestCase = "cudatrial/planted-race-control"
			r[1].Error = errors.New("candidate failed")
			return r
		}},
		{"only exec control", func([]EvaluationResult) []EvaluationResult { return nil }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := c.ConformanceRate(tc.mutate(append([]EvaluationResult(nil), good...)), control); err == nil {
				t.Fatal("invalid corpus/control results were accepted")
			}
		})
	}
	if _, err := (&RaceCorpus{}).ConformanceRate(nil, control); err == nil {
		t.Fatal("empty corpus was accepted")
	}
}

func TestConformanceRateReportsDisjointCounts(t *testing.T) {
	c := NewRaceCorpus()
	configuration := "FIXED=1;ARCH=sm_90"
	control := makeExecControl(t, configuration)
	results := validConformanceResults(c, configuration)
	results[1].HardwarePending = true
	results[2].Unmodelled = true
	results[3].UnknownExit = true
	results[3].ExitCode = 2
	results[3].Error = errors.New("unknown detector exit")
	results[4].Error = errors.New("ordinary failure")
	summary, err := c.ConformanceRate(results, control)
	if err != nil {
		t.Fatal(err)
	}
	if summary.Total != 27 || summary.Passed != 23 || summary.HardwarePending != 1 || summary.Unmodelled != 1 || summary.UnknownExit != 1 || summary.Failed != 1 {
		t.Fatalf("disjoint conformance counts = %+v", summary)
	}
	if summary.Passed+summary.Failed+summary.HardwarePending+summary.Unmodelled+summary.UnknownExit != summary.Total {
		t.Fatalf("counts do not sum to total: %+v", summary)
	}
}

func TestConformanceRateReportsFixed0MissesAfterControlPasses(t *testing.T) {
	c := NewRaceCorpus()
	configuration := "FIXED=0;ARCH=sm_90"
	control := makeExecControl(t, configuration)
	results := []EvaluationResult{{
		TestCase: "cudatrial/planted-race-control", IsLivenessControl: true,
		HasRace: true, ExitCode: 1, Configuration: configuration,
	}}
	for _, tc := range c.TestCases() {
		results = append(results, EvaluationResult{
			TestCase: tc.Name, Category: tc.Category, Fixed: false,
			Configuration: configuration,
		})
	}
	summary, err := c.ConformanceRate(results, control)
	if err != nil {
		t.Fatal(err)
	}
	if summary.Total != 27 || summary.Failed == 0 || summary.Rate >= 1 {
		t.Fatalf("passing only the planted control looked like corpus conformance: %+v", summary)
	}
}

func TestRunExternalPendingRowsCoverEveryUnsupportedCase(t *testing.T) {
	c := NewRaceCorpus()
	root := makeCorpusBinaries(t, c, "sm_80")
	t.Setenv("CUHADRON_BINARY_ROOT", root)
	cmd := `sh -c 'case "$1" in */FIXED=0/*) exit 1;; *) exit 0;; esac' sh`
	results, err := c.RunExternal(context.Background(), cmd, false, "sm_80")
	if err != nil {
		t.Fatal(err)
	}
	seen := make(map[string]int)
	pending := 0
	for _, result := range results {
		if result.TestCase == "cudatrial/planted-race-control" {
			continue
		}
		seen[result.TestCase]++
		if result.HardwarePending {
			pending++
			if !errors.Is(result.Error, cudaerrors.ErrHardwarePending) {
				t.Errorf("pending row %s has error %v", result.TestCase, result.Error)
			}
		}
	}
	if pending != 8 || len(seen) != 27 {
		t.Fatalf("sm_80 sweep has %d pending rows and %d case names, want 8 and 27", pending, len(seen))
	}
	for _, tc := range c.TestCases() {
		if seen[tc.Name] != 1 {
			t.Errorf("sm_80 sweep has %d rows for %s", seen[tc.Name], tc.Name)
		}
		if tc.MinArch == "sm_90" {
			foundPending := false
			for _, result := range results {
				if result.TestCase == tc.Name {
					foundPending = result.HardwarePending
				}
			}
			if !foundPending {
				t.Errorf("sm_90 case %s was not reported pending", tc.Name)
			}
		}
	}
	control := makeExecControl(t, "FIXED=0;ARCH=sm_80")
	summary, err := c.ConformanceRate(results, control)
	if err != nil {
		t.Fatal(err)
	}
	if summary.Total != 27 || summary.HardwarePending != 8 || summary.Passed+summary.Failed+summary.HardwarePending+summary.Unmodelled+summary.UnknownExit != 27 {
		t.Fatalf("sm_80 conformance counts = %+v", summary)
	}
}

func TestMinArchValueGatesEvaluationAndExternalSweep(t *testing.T) {
	c := NewRaceCorpus()
	c.cases[0].MinArch = "sm_100"
	if result, err := c.EvaluateCase(c.cases[0].Name, false, "sm_90"); err == nil || !result.HardwarePending {
		t.Fatalf("sm_100 case on sm_90 = (%+v, %v), want pending", result, err)
	}
	root := makeCorpusBinaries(t, c, "sm_90")
	t.Setenv("CUHADRON_BINARY_ROOT", root)
	cmd := `sh -c 'case "$1" in */FIXED=0/*) exit 1;; *) exit 0;; esac' sh`
	results, err := c.RunExternal(context.Background(), cmd, false, "sm_90")
	if err != nil {
		t.Fatal(err)
	}
	for _, result := range results {
		if result.TestCase == c.cases[0].Name {
			if !result.HardwarePending {
				t.Fatalf("sm_100 case on sm_90 was not pending: %+v", result)
			}
			return
		}
	}
	t.Fatal("RunExternal omitted the sm_100 case")
}

func validConformanceResults(c *RaceCorpus, configuration string) []EvaluationResult {
	results := []EvaluationResult{{TestCase: "cudatrial/planted-race-control", IsLivenessControl: true, HasRace: true, ExitCode: 1, Category: CategoryMemcpy, Configuration: configuration}}
	for _, tc := range c.TestCases() {
		result := EvaluationResult{TestCase: tc.Name, Category: tc.Category, Fixed: true, Configuration: configuration}
		results = append(results, result)
	}
	return results
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
	control := makeExecControl(t, "FIXED=1;ARCH=sm_90")
	summary, err := c.ConformanceRate(results, control)
	if err != nil {
		t.Fatalf("matching lane52 exec-control rejected: %v", err)
	}
	if summary.Total != 27 || summary.Passed+summary.Failed+summary.HardwarePending+summary.Unmodelled+summary.UnknownExit != summary.Total {
		t.Fatalf("conformance counts do not partition 27 cases: %+v", summary)
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

func TestConformanceRateRejectsMissingPlantedControlAlone(t *testing.T) {
	c := NewRaceCorpus()
	root := makeCorpusBinaries(t, c, "sm_90")
	t.Setenv("CUHADRON_BINARY_ROOT", root)
	results, runErr := c.RunExternal(context.Background(), "true", false, "sm_90")
	if !errors.Is(runErr, ErrDetectorNeverReached) {
		t.Fatalf("clean planted-control run error = %v, want ErrDetectorNeverReached", runErr)
	}
	if len(results) != 28 {
		t.Fatalf("results contain %d rows, want 27 corpus rows plus control", len(results))
	}
	for i, result := range results {
		if result.TestCase == "cudatrial/planted-race-control" {
			results = append(results[:i], results[i+1:]...)
			break
		}
	}
	control := makeExecControl(t, "FIXED=0;ARCH=sm_90")
	if _, err := c.ConformanceRate(results, control); err == nil {
		t.Fatal("27 complete corpus rows without planted control were accepted")
	}
}

func TestHazardMatrix_SubwarpAndHostDeviceControls(t *testing.T) {
	c := NewRaceCorpus()
	for _, name := range []string{"intersubwarp/shared_readwrite_race", "hostdevice/global_readwrite_race"} {
		tc, _ := c.Lookup(name)
		for _, fixed := range []bool{false, true} {
			accesses, err := caseAccesses(tc.Name, fixed)
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

func TestRunExternalCandidateUnknownExitAndOutput(t *testing.T) {
	c := NewRaceCorpus()
	root := makeCorpusBinaries(t, c, "sm_90")
	t.Setenv("CUHADRON_BINARY_ROOT", root)
	cmd := `sh -c 'case "$1" in */FIXED=0/memcpy/shared_readwrite_race.sm90.out) echo planted; exit 1;; */FIXED=0/memcpy/shared_writewrite_race.sm90.out) echo candidate-output; exit 2;; *) exit 0;; esac' sh`
	results, err := c.RunExternal(context.Background(), cmd, false, "sm_90")
	if err == nil {
		t.Fatal("candidate exit 2 was accepted")
	}
	for _, result := range results {
		switch result.TestCase {
		case "cudatrial/planted-race-control":
			if !result.HasRace || result.Output != "planted\n" {
				t.Fatalf("control result = %+v", result)
			}
		case "memcpy/shared_writewrite_race":
			if !result.UnknownExit || result.Error == nil || result.HasRace || result.Output != "candidate-output\n" {
				t.Fatalf("candidate exit 2 result = %+v", result)
			}
		}
	}
}

func TestRunExternalMultigpuRowsAreUnmodelledWithoutExecution(t *testing.T) {
	c := NewRaceCorpus()
	root := makeCorpusBinaries(t, c, "sm_90")
	t.Setenv("CUHADRON_BINARY_ROOT", root)
	results, err := c.RunExternal(context.Background(), `sh -c 'case "$1" in */multigpu/*) echo multigpu-ran; exit 1;; *) exit 1;; esac' sh`, false, "sm_90")
	if err != nil {
		t.Fatal(err)
	}
	for _, result := range results {
		if result.Category != CategoryMultiGPU {
			continue
		}
		if !result.Unmodelled || result.HasRace || result.Output != "" {
			t.Errorf("multigpu row ran or was modeled: %+v", result)
		}
	}
}

func TestEvaluateAllJoinsCaseErrorsAndSuffixArchitectures(t *testing.T) {
	c := NewRaceCorpus()
	_, err := c.EvaluateAll(false, "sm_80")
	if err == nil || !strings.Contains(err.Error(), "multigpu/global_writewrite_race") || !errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Fatalf("sm_80 errors = %v, want joined multigpu and pending errors", err)
	}
	for _, arch := range []string{"sm_90a", "sm_121a"} {
		rows, _ := c.EvaluateAll(false, arch)
		for _, row := range rows {
			if row.HardwarePending {
				t.Errorf("EvaluateAll(%q) marked %s pending: %v", arch, row.TestCase, row.Error)
			}
		}
	}
}

func TestPassedRejectsFalsePositiveRace(t *testing.T) {
	result := EvaluationResult{Category: CategoryFalsePositives, HasRace: true}
	if result.Passed() {
		t.Fatal("false-positive row with a race was accepted")
	}
}

func TestConformanceRateRejectsMismatchedExecConfiguration(t *testing.T) {
	c := NewRaceCorpus()
	configuration := "FIXED=0;ARCH=sm_90"
	control := makeExecControl(t, configuration)
	results := validConformanceResults(c, configuration)
	for i := range results {
		if results[i].TestCase != "cudatrial/planted-race-control" {
			results[i].Configuration = "FIXED=1;ARCH=sm_90"
			break
		}
	}
	if _, err := c.ConformanceRate(results, control); err == nil {
		t.Fatal("mismatched exec-control configuration accepted")
	}
}

func TestConformanceRateRejectsFixedFieldThatDisagreesWithConfiguration(t *testing.T) {
	c := NewRaceCorpus()
	configuration := "FIXED=1;ARCH=sm_90"
	results := validConformanceResults(c, configuration)
	for i := range results {
		if results[i].TestCase == c.cases[0].Name {
			results[i].Fixed = false
			results[i].HasRace = true
			break
		}
	}
	if _, err := c.ConformanceRate(results, makeExecControl(t, configuration)); err == nil {
		t.Fatal("row with FIXED=0 and a race was accepted under FIXED=1 configuration")
	}
}

func TestReadExecControlRejectsInvalidReceipts(t *testing.T) {
	for _, tt := range []struct {
		name string
		data string
	}{
		{"passed false", `{"configuration":"FIXED=1;ARCH=sm_90","passed":false}`},
		{"empty configuration", `{"configuration":"  ","passed":true}`},
		{"malformed JSON", `{"configuration":`},
	} {
		t.Run(tt.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "exec-control.json")
			if err := os.WriteFile(path, []byte(tt.data), 0o600); err != nil {
				t.Fatal(err)
			}
			if _, err := ReadExecControl(path); err == nil {
				t.Fatal("invalid exec-control receipt was accepted")
			}
		})
	}
}

func TestConformanceRateRevalidatesExecControlEvidence(t *testing.T) {
	c := NewRaceCorpus()
	control := makeExecControl(t, "FIXED=1;ARCH=sm_90")
	results := validConformanceResults(c, control.Configuration())
	if err := os.Remove(control.EvidencePath()); err != nil {
		t.Fatal(err)
	}
	if _, err := c.ConformanceRate(results, control); err == nil {
		t.Fatal("removed exec-control evidence was accepted")
	}
	if err := os.WriteFile(control.EvidencePath(), []byte(`{"configuration":"FIXED=1;ARCH=sm_90","passed":false}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := c.ConformanceRate(results, control); err == nil {
		t.Fatal("exec-control evidence changed to passed=false was accepted")
	}
}

func makeExecControl(t *testing.T, configuration string) ExecControl {
	t.Helper()
	path := filepath.Join(t.TempDir(), "exec-control.json")
	data := `{"configuration":"` + configuration + `","passed":true}`
	if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
		t.Fatal(err)
	}
	control, err := ReadExecControl(path)
	if err != nil {
		t.Fatal(err)
	}
	return control
}

func makeCorpusBinaries(t *testing.T, c *RaceCorpus, arch string) string {
	t.Helper()
	root := t.TempDir()
	for _, fixed := range []string{"FIXED=0", "FIXED=1"} {
		for _, tc := range c.TestCases() {
			path := filepath.Join(root, fixed, tc.Name+"."+binaryArchSuffix(arch)+".out")
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
