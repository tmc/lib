package cudatrial

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudaliveness"
	"github.com/tmc/lib/nvidia/x/cudashared"
	"github.com/tmc/lib/nvidia/x/cudatma"
)

// ErrDetectorNeverReached indicates that a detector runner or external command
// exited 0 without detecting a planted race, indicating the detector was never reached.
var ErrDetectorNeverReached = errors.New("cudatrial: detector was never reached: runner reported clean exit on planted race")

// RaceCategory classifies the benchmark category derived from cuHadron.
type RaceCategory string

const (
	CategoryMemcpy         RaceCategory = "memcpy"
	CategoryBulkcpy        RaceCategory = "bulkcpy"
	CategoryInterkernel    RaceCategory = "interkernel"
	CategoryHostDevice     RaceCategory = "hostdevice"
	CategoryAsyncMemcpy    RaceCategory = "asyncmemcpy"
	CategoryIntersubwarp   RaceCategory = "intersubwarp"
	CategoryDSMEM          RaceCategory = "dsmem"
	CategoryMultiGPU       RaceCategory = "multigpu"
	CategoryFalsePositives RaceCategory = "false_positives"
	CategoryInterwave      RaceCategory = "interwave"
)

// TestCase describes one benchmark candidate within the race conformance corpus.
type TestCase struct {
	Name           string
	Category       RaceCategory
	Memory         cudashared.MemorySpace
	Scope          cudashared.ExecutionScope
	Hazard         cudashared.HazardType
	MinArch        string
	FixedSupported bool
	Description    string
}

// EvaluationResult records the outcome of evaluating a test case.
type EvaluationResult struct {
	TestCase          string
	Category          RaceCategory
	Fixed             bool
	BinaryFixed       bool
	HasRace           bool
	ExitCode          int
	Error             error
	ReportedDefects   int
	HardwarePending   bool
	EvaluatedPairs    int
	BinaryPath        string
	Output            string
	Configuration     string
	IsExecControl     bool
	ExecControlPass   bool
	IsLivenessControl bool
}

// Passed reports whether the evaluation correctly matched the expected outcome:
// For false positives: never a race.
// For FIXED=0: race must be detected.
// For FIXED=1: race must be absent.
func (r EvaluationResult) Passed() bool {
	if r.IsExecControl {
		return r.ExecControlPass && r.Error == nil && r.ExitCode == 0
	}
	if r.IsLivenessControl {
		return r.Error == nil && r.ExitCode == 1 && r.HasRace
	}
	if r.HardwarePending || r.Error != nil || (r.ExitCode != 0 && r.ExitCode != 1) {
		return false
	}
	if r.Category == CategoryFalsePositives {
		return !r.HasRace
	}
	if r.Fixed {
		return !r.HasRace
	}
	return r.HasRace
}

// RaceCorpus holds the 27 data-race cases derived from NVlabs/cuHadron commit
// 0a6c6d5 (Apache-2.0).
type RaceCorpus struct {
	cases []TestCase
}

// NewRaceCorpus constructs the canonical cuHadron conformance corpus comprising all 27 benchmark cases.
func NewRaceCorpus() *RaceCorpus {
	return &RaceCorpus{
		cases: []TestCase{
			// memcpy (cp.async)
			{Name: "memcpy/shared_writewrite_race", Category: CategoryMemcpy, Memory: cudashared.MemoryShared, Scope: cudashared.ScopeCTA, Hazard: cudashared.HazardWriteWrite, MinArch: "sm_80", FixedSupported: true, Description: "Multiple threads write same shared locations via cp.async"},
			{Name: "memcpy/global_readwrite_race", Category: CategoryMemcpy, Memory: cudashared.MemoryGlobal, Scope: cudashared.ScopeDevice, Hazard: cudashared.HazardReadWrite, MinArch: "sm_80", FixedSupported: true, Description: "One warp reads via cp.async while another writes global memory"},
			{Name: "memcpy/shared_readwrite_race", Category: CategoryMemcpy, Memory: cudashared.MemoryShared, Scope: cudashared.ScopeCTA, Hazard: cudashared.HazardReadWrite, MinArch: "sm_80", FixedSupported: true, Description: "Consumer reads shared memory before cp.async completes"},

			// bulkcpy (cp.async.bulk, sm_90+)
			{Name: "bulkcpy/shared_readwrite_race", Category: CategoryBulkcpy, Memory: cudashared.MemoryShared, Scope: cudashared.ScopeCTA, Hazard: cudashared.HazardWriteRead, MinArch: "sm_90", FixedSupported: true, Description: "Bulk copy reads shared memory while other threads write"},
			{Name: "bulkcpy/global_writewrite_race", Category: CategoryBulkcpy, Memory: cudashared.MemoryGlobal, Scope: cudashared.ScopeDevice, Hazard: cudashared.HazardWriteWrite, MinArch: "sm_90", FixedSupported: true, Description: "Multiple blocks write same global location via bulk copy"},
			{Name: "bulkcpy/global_readwrite_race", Category: CategoryBulkcpy, Memory: cudashared.MemoryGlobal, Scope: cudashared.ScopeDevice, Hazard: cudashared.HazardReadWrite, MinArch: "sm_90", FixedSupported: true, Description: "Read before cp.async.bulk completes"},
			{Name: "bulkcpy/shared_writeread_race_g2s", Category: CategoryBulkcpy, Memory: cudashared.MemoryShared, Scope: cudashared.ScopeCTA, Hazard: cudashared.HazardWriteRead, MinArch: "sm_90", FixedSupported: true, Description: "Bulk copy global to shared write-read race"},
			{Name: "bulkcpy/shared_writewrite_race_g2s", Category: CategoryBulkcpy, Memory: cudashared.MemoryShared, Scope: cudashared.ScopeCTA, Hazard: cudashared.HazardWriteWrite, MinArch: "sm_90", FixedSupported: true, Description: "Bulk copy global to shared write-write race"},
			{Name: "bulkcpy/global_readwrite_race_g2s", Category: CategoryBulkcpy, Memory: cudashared.MemoryGlobal, Scope: cudashared.ScopeDevice, Hazard: cudashared.HazardReadWrite, MinArch: "sm_90", FixedSupported: true, Description: "Bulk copy global to shared read-write race"},

			// interkernel (streams)
			{Name: "interkernel/global_writewrite_race", Category: CategoryInterkernel, Memory: cudashared.MemoryGlobal, Scope: cudashared.ScopeDevice, Hazard: cudashared.HazardWriteWrite, MinArch: "sm_80", FixedSupported: true, Description: "Two kernels on different streams write same addresses"},
			{Name: "interkernel/global_readwrite_race", Category: CategoryInterkernel, Memory: cudashared.MemoryGlobal, Scope: cudashared.ScopeDevice, Hazard: cudashared.HazardReadWrite, MinArch: "sm_80", FixedSupported: true, Description: "Consumer kernel reads before producer kernel completes"},

			// asyncmemcpy (cudaMemcpyAsync vs kernel)
			{Name: "asyncmemcpy/memcpy_htod_kernel_race", Category: CategoryAsyncMemcpy, Memory: cudashared.MemoryGlobal, Scope: cudashared.ScopeHostDevice, Hazard: cudashared.HazardReadWrite, MinArch: "sm_80", FixedSupported: true, Description: "Kernel reads before cudaMemcpyAsync HtoD completes"},
			{Name: "asyncmemcpy/kernel_memcpy_dtoh_race", Category: CategoryAsyncMemcpy, Memory: cudashared.MemoryGlobal, Scope: cudashared.ScopeHostDevice, Hazard: cudashared.HazardReadWrite, MinArch: "sm_80", FixedSupported: true, Description: "cudaMemcpyAsync DtoH reads before kernel completes"},

			// hostdevice (CPU-GPU UVM)
			{Name: "hostdevice/global_writewrite_race", Category: CategoryHostDevice, Memory: cudashared.MemoryUnified, Scope: cudashared.ScopeHostDevice, Hazard: cudashared.HazardWriteWrite, MinArch: "sm_80", FixedSupported: true, Description: "CPU and GPU write same unified addresses concurrently"},
			{Name: "hostdevice/global_readwrite_race", Category: CategoryHostDevice, Memory: cudashared.MemoryUnified, Scope: cudashared.ScopeHostDevice, Hazard: cudashared.HazardReadWrite, MinArch: "sm_80", FixedSupported: true, Description: "CPU reads unified memory before GPU kernel finishes"},
			{Name: "hostdevice/global_writeread_race", Category: CategoryHostDevice, Memory: cudashared.MemoryUnified, Scope: cudashared.ScopeHostDevice, Hazard: cudashared.HazardWriteRead, MinArch: "sm_80", FixedSupported: true, Description: "GPU reads while CPU updates unified memory"},

			// intersubwarp (divergent threads within warp)
			{Name: "intersubwarp/shared_readwrite_race", Category: CategoryIntersubwarp, Memory: cudashared.MemoryShared, Scope: cudashared.ScopeSubwarp, Hazard: cudashared.HazardReadWrite, MinArch: "sm_80", FixedSupported: true, Description: "Divergent subwarps in warp read and write same shared address"},
			{Name: "intersubwarp/shared_writewrite_race", Category: CategoryIntersubwarp, Memory: cudashared.MemoryShared, Scope: cudashared.ScopeSubwarp, Hazard: cudashared.HazardWriteWrite, MinArch: "sm_80", FixedSupported: true, Description: "Divergent subwarps in warp write same shared address"},

			// dsmem (distributed shared memory, sm_90+)
			{Name: "dsmem/shared_readwrite_race", Category: CategoryDSMEM, Memory: cudashared.MemoryDSMEM, Scope: cudashared.ScopeCluster, Hazard: cudashared.HazardReadWrite, MinArch: "sm_90", FixedSupported: true, Description: "CTAs in cluster read and write DSMEM without cluster sync"},
			{Name: "dsmem/shared_writewrite_race", Category: CategoryDSMEM, Memory: cudashared.MemoryDSMEM, Scope: cudashared.ScopeCluster, Hazard: cudashared.HazardWriteWrite, MinArch: "sm_90", FixedSupported: true, Description: "CTAs in cluster write DSMEM without cluster sync"},

			// multigpu
			{Name: "multigpu/global_writewrite_race", Category: CategoryMultiGPU, Memory: cudashared.MemoryGlobal, Scope: cudashared.ScopeMultiGPU, Hazard: cudashared.HazardWriteWrite, MinArch: "sm_80", FixedSupported: true, Description: "Multiple GPUs write same addresses without peer sync"},
			{Name: "multigpu/global_readwrite_race", Category: CategoryMultiGPU, Memory: cudashared.MemoryGlobal, Scope: cudashared.ScopeMultiGPU, Hazard: cudashared.HazardReadWrite, MinArch: "sm_80", FixedSupported: true, Description: "GPU reads before other GPU write completes"},

			// false_positives (byte independence: no race under byte granularity)
			{Name: "false_positives/shared_bytewise_readwrite", Category: CategoryFalsePositives, Memory: cudashared.MemoryShared, Scope: cudashared.ScopeCTA, Hazard: cudashared.HazardReadWrite, MinArch: "sm_80", FixedSupported: false, Description: "Thread 0 reads byte 0, thread 32 writes byte 1 in same 32-bit word"},
			{Name: "false_positives/shared_bytewise_writewrite", Category: CategoryFalsePositives, Memory: cudashared.MemoryShared, Scope: cudashared.ScopeCTA, Hazard: cudashared.HazardWriteWrite, MinArch: "sm_80", FixedSupported: false, Description: "Adjacent threads write distinct bytes in same shared 32-bit word"},
			{Name: "false_positives/global_bytewise_readwrite", Category: CategoryFalsePositives, Memory: cudashared.MemoryGlobal, Scope: cudashared.ScopeCTA, Hazard: cudashared.HazardReadWrite, MinArch: "sm_80", FixedSupported: false, Description: "Thread 0 reads byte 0, thread 32 writes byte 1 in same global 32-bit word"},
			{Name: "false_positives/global_bytewise_writewrite", Category: CategoryFalsePositives, Memory: cudashared.MemoryGlobal, Scope: cudashared.ScopeCTA, Hazard: cudashared.HazardWriteWrite, MinArch: "sm_80", FixedSupported: false, Description: "Adjacent threads write distinct bytes in same global 32-bit word"},

			// interwave
			{Name: "interwave/global_writewrite_race", Category: CategoryInterwave, Memory: cudashared.MemoryGlobal, Scope: cudashared.ScopeDevice, Hazard: cudashared.HazardWriteWrite, MinArch: "sm_80", FixedSupported: true, Description: "Blocks in different execution waves write same address without grid sync"},
		},
	}
}

// TestCases returns a copy of all registered benchmark test cases.
func (c *RaceCorpus) TestCases() []TestCase {
	if c == nil {
		return nil
	}
	return append([]TestCase(nil), c.cases...)
}

// Lookup finds a test case by exact name.
func (c *RaceCorpus) Lookup(name string) (TestCase, bool) {
	if c == nil {
		return TestCase{}, false
	}
	for _, tc := range c.cases {
		if tc.Name == name {
			return tc, true
		}
	}
	return TestCase{}, false
}

// EvaluateCase evaluates a single benchmark case under the model for the specified FIXED setting and target architecture.
// If the case requires sm_90+ (e.g. DSMEM or bulk copy) and targetArch does not support clusters, it reports
// cudaerrors.ErrHardwarePending rather than simulating success.
func (c *RaceCorpus) EvaluateCase(name string, fixed bool, targetArch string) (EvaluationResult, error) {
	tc, ok := c.Lookup(name)
	if !ok {
		return EvaluationResult{}, fmt.Errorf("unknown test case %q", name)
	}
	base := EvaluationResult{TestCase: name, Category: tc.Category, Fixed: fixed}
	if tc.MinArch == "sm_90" {
		if targetArch == "" {
			err := fmt.Errorf("empty target architecture: %w", cudaerrors.ErrHardwarePending)
			base.HardwarePending, base.Error = true, err
			return base, err
		}
		if err := cudashared.CheckClusterHardware(targetArch); err != nil {
			base.HardwarePending, base.Error = true, err
			return base, err
		}
	}
	accesses, err := caseAccesses(tc, fixed)
	if err != nil {
		base.Error = fmt.Errorf("unmodelled case %s: %w", name, err)
		return base, base.Error
	}
	m := cudashared.NewHazardMatrix(targetArch)
	for _, a := range accesses {
		if err := m.AddAccess(a); err != nil {
			base.Error = err
			return base, err
		}
	}
	records, err := m.Evaluate()
	if err != nil {
		base.HardwarePending, base.Error = errors.Is(err, cudaerrors.ErrHardwarePending), err
		return base, err
	}
	base.EvaluatedPairs = len(accesses) * (len(accesses) - 1) / 2
	for _, record := range records {
		base.HasRace = base.HasRace || record.IsRace
	}
	if tc.Category == CategoryMemcpy || tc.Category == CategoryBulkcpy {
		var scenario cudatma.PipelineScenario
		if tc.Name == "bulkcpy/shared_readwrite_race" {
			scenario, err = cudatma.GenerateTMASharedReadWriteScenario(fixed)
		} else if tc.Category == CategoryBulkcpy {
			scenario, err = cudatma.GenerateTMAScenario(fixed)
		} else {
			scenario, err = cudatma.GenerateAsyncCopyScenario(fixed)
		}
		if err != nil {
			base.Error = err
			return base, err
		}
		base.HasRace = base.HasRace || scenario.HasHazards
	}
	if base.HasRace {
		base.ReportedDefects = 1
	}
	return base, nil
}

// caseAccesses is the host model transcribed from each corpus shape. A nil
// access list is never a successful evaluation.
func caseAccesses(tc TestCase, fixed bool) ([]cudashared.MemoryAccess, error) {
	if tc.Scope == cudashared.ScopeMultiGPU {
		return nil, fmt.Errorf("multi-GPU ownership is not represented by MemoryAccess")
	}
	first, second := cudashared.MemoryAccess{Scope: tc.Scope, ThreadID: 0, BlockID: 0, ClusterID: 0, Space: tc.Memory, Offset: 0x1000, SizeBytes: 4, Stage: 0}, cudashared.MemoryAccess{Scope: tc.Scope, ThreadID: 1, BlockID: 0, ClusterID: 0, Space: tc.Memory, Offset: 0x1000, SizeBytes: 4, Stage: 0}
	switch tc.Hazard {
	case cudashared.HazardReadWrite:
		second.IsWrite = true
	case cudashared.HazardWriteRead:
		first.IsWrite = true
	case cudashared.HazardWriteWrite:
		first.IsWrite, second.IsWrite = true, true
	default:
		return nil, fmt.Errorf("unsupported hazard type %v", tc.Hazard)
	}
	if tc.Scope == cudashared.ScopeSubwarp {
		second.ThreadID = 16
		if fixed {
			second.Offset += 16
		}
	}
	if tc.Scope == cudashared.ScopeCluster {
		second.BlockID = 1
	}
	if tc.Category == CategoryFalsePositives {
		first.SizeBytes, second.SizeBytes, second.Offset = 1, 1, first.Offset+1
	}
	if fixed && tc.Category != CategoryFalsePositives && tc.Scope != cudashared.ScopeSubwarp {
		second.Stage = 1
	}
	return []cudashared.MemoryAccess{first, second}, nil
}

// EvaluateAll evaluates all test cases in the corpus for the given FIXED setting.
func (c *RaceCorpus) EvaluateAll(fixed bool, targetArch string) ([]EvaluationResult, error) {
	if c == nil {
		return nil, fmt.Errorf("nil race corpus")
	}
	results := make([]EvaluationResult, 0, len(c.cases))
	var errs []error
	for _, tc := range c.cases {
		res, err := c.EvaluateCase(tc.Name, fixed, targetArch)
		results = append(results, res)
		if err != nil {
			errs = append(errs, err)
		}
	}
	return results, errors.Join(errs...)
}

// RunExternal executes an external detector RUN_CMD against the corpus test cases.
// It verifies detector liveness using [cudaliveness.Planted] controls: in FIXED=0 mode,
// planted race cases MUST result in a detected race (non-zero exit code).
// If a stub RUN_CMD always exits 0, the liveness control asserts failure, preventing
// an unfalsifiable 27/27 clean sweep.
func (c *RaceCorpus) RunExternal(ctx context.Context, runCmd string, fixed bool, targetArch string) ([]EvaluationResult, error) {
	if c == nil {
		return nil, fmt.Errorf("nil race corpus")
	}
	if strings.TrimSpace(runCmd) == "" {
		return nil, fmt.Errorf("empty RUN_CMD")
	}
	if targetArch == "" {
		var pending []EvaluationResult
		var errs []error
		fixedValue := "0"
		if fixed {
			fixedValue = "1"
		}
		for _, tc := range c.cases {
			if tc.MinArch != "sm_90" {
				continue
			}
			err := fmt.Errorf("%s requires a target architecture: %w", tc.Name, cudaerrors.ErrHardwarePending)
			pending = append(pending, EvaluationResult{TestCase: tc.Name, Category: tc.Category, Fixed: fixed, HardwarePending: true, Error: err, Configuration: "FIXED=" + fixedValue + ";ARCH="})
			errs = append(errs, err)
		}
		return pending, errors.Join(errs...)
	}

	root := os.Getenv("CUHADRON_BINARY_ROOT")
	if root == "" {
		return nil, fmt.Errorf("CUHADRON_BINARY_ROOT is required")
	}
	fixedName := "FIXED=0"
	if fixed {
		fixedName = "FIXED=1"
	}
	configuration := fixedName + ";ARCH=" + targetArch
	results := make([]EvaluationResult, 0, len(c.cases)+1)
	var plantedControls []cudaliveness.Control
	var runErrors []error

	// This binary is built with FIXED=0 and is run on every sweep, including a
	// FIXED=1 sweep, so detector liveness is never inferred from labels.
	controlPath := filepath.Join(root, "FIXED=0", "memcpy", "shared_readwrite_race."+targetArch+".out")
	control, _ := runDetector(ctx, runCmd, controlPath)
	control.TestCase, control.Category, control.Configuration, control.Fixed, control.BinaryFixed, control.IsLivenessControl = "cudatrial/planted-race-control", CategoryMemcpy, configuration, fixed, false, true
	results = append(results, control)
	if control.Error != nil {
		runErrors = append(runErrors, control.Error)
	}
	if !control.Passed() || !control.HasRace {
		runErrors = append(runErrors, fmt.Errorf("%w: planted FIXED=0 control was not detected", ErrDetectorNeverReached))
	}
	plantedControls = append(plantedControls, cudaliveness.Planted{Site: control.TestCase, Reported: control.ReportedDefects, Ran: control.BinaryPath != ""})

	for _, tc := range c.cases {
		if err := ctx.Err(); err != nil {
			return results, err
		}

		// Architecture check. An empty target is unknown, not a default sm_90.
		if tc.MinArch == "sm_90" {
			if err := cudashared.CheckClusterHardware(targetArch); err != nil {
				results = append(results, EvaluationResult{
					TestCase:        tc.Name,
					Category:        tc.Category,
					Fixed:           fixed,
					HardwarePending: true,
					Error:           err,
					Configuration:   configuration,
				})
				continue
			}
		}

		caseRoot := filepath.Join(root, fixedName)
		binaryPath := filepath.Join(caseRoot, tc.Name+"."+targetArch+".out")
		res, runErr := runDetector(ctx, runCmd, binaryPath)
		res.TestCase, res.Category, res.Fixed, res.BinaryFixed, res.Configuration = tc.Name, tc.Category, fixed, fixed, configuration
		if runErr != nil {
			res.Error = runErr
			runErrors = append(runErrors, runErr)
		}
		results = append(results, res)

		// Record liveness control for planted race candidates in FIXED=0.
		if runErr != nil || (!fixed && tc.Category != CategoryFalsePositives) {
			plantedControls = append(plantedControls, cudaliveness.Planted{
				Site:     tc.Name,
				Reported: res.ReportedDefects,
				Ran:      res.BinaryPath != "",
			})
		}
	}

	// Assert the planted binary was run even when the evaluated candidates are fixed.
	if len(plantedControls) > 0 {
		if err := cudaliveness.Assert(plantedControls...); err != nil {
			runErrors = append(runErrors, fmt.Errorf("%w: %w", ErrDetectorNeverReached, err))
		}
	}

	return results, errors.Join(runErrors...)
}

func runDetector(ctx context.Context, runCmd, binaryPath string) (EvaluationResult, error) {
	res := EvaluationResult{BinaryPath: binaryPath}
	if _, err := os.Stat(binaryPath); err != nil {
		res.Error = fmt.Errorf("stat detector input %s: %w", binaryPath, err)
		return res, res.Error
	}
	cmd := exec.CommandContext(ctx, "sh", "-c", "exec "+runCmd+" \"$1\"", "cudatrial", binaryPath)
	out, err := cmd.CombinedOutput()
	res.Output = string(out)
	res.ExitCode = 0
	if err != nil {
		var exitErr *exec.ExitError
		if !errors.As(err, &exitErr) {
			res.Error = fmt.Errorf("start detector: %w", err)
			return res, res.Error
		}
		res.ExitCode = exitErr.ExitCode()
	}
	switch res.ExitCode {
	case 0:
		res.HasRace = false
	case 1:
		res.HasRace, res.ReportedDefects = true, 1
	default:
		res.Error = fmt.Errorf("detector exited %d: %s", res.ExitCode, res.Output)
		return res, res.Error
	}
	return res, nil
}

// ConformanceRate computes the number and percentage of test cases passing conformance criteria.
func (c *RaceCorpus) ConformanceRate(results []EvaluationResult) (passed int, total int, rate float64, err error) {
	configuration := ""
	for _, r := range results {
		if r.IsExecControl && r.TestCase == "lane52/exec-control" && r.Passed() {
			configuration = r.Configuration
			break
		}
	}
	if configuration == "" {
		return 0, 0, 0, fmt.Errorf("conformance rate requires a passing lane 52 exec-control result")
	}
	for _, r := range results {
		if r.TestCase != "lane52/exec-control" && r.Configuration != configuration {
			return 0, 0, 0, fmt.Errorf("exec-control configuration %q does not match %q", configuration, r.Configuration)
		}
	}
	for _, r := range results {
		if r.IsExecControl || r.TestCase == "cudatrial/planted-race-control" {
			continue
		}
		total++
	}
	if total == 0 {
		return 0, 0, 0.0, nil
	}
	for _, r := range results {
		if r.IsExecControl || r.TestCase == "cudatrial/planted-race-control" {
			continue
		}
		if r.Passed() {
			passed++
		}
	}
	rate = float64(passed) / float64(total)
	return passed, total, rate, nil
}
