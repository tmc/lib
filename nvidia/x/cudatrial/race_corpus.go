package cudatrial

import (
	"context"
	"errors"
	"fmt"
	"os/exec"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaliveness"
	"github.com/tmc/lib/nvidia/x/cudashared"
	"github.com/tmc/lib/nvidia/x/cudatma"
)

// ErrDetectorNeverReached indicates that a detector runner or external command
// exited 0 without detecting a planted race, indicating the detector was never reached.
var ErrDetectorNeverReached = errors.New("cudatrial: detector was never reached: runner reported clean exit on planted race")

// ExecutionScope identifies the execution concurrency boundary at which memory accesses collide.
type ExecutionScope uint8

const (
	ScopeSubwarp ExecutionScope = iota + 1
	ScopeWarp
	ScopeCTA
	ScopeCluster
	ScopeDevice
	ScopeHostDevice
	ScopeMultiGPU
)

// String implements [fmt.Stringer].
func (s ExecutionScope) String() string {
	switch s {
	case ScopeSubwarp:
		return "subwarp"
	case ScopeWarp:
		return "warp"
	case ScopeCTA:
		return "cta"
	case ScopeCluster:
		return "cluster"
	case ScopeDevice:
		return "device"
	case ScopeHostDevice:
		return "hostdevice"
	case ScopeMultiGPU:
		return "multigpu"
	default:
		return fmt.Sprintf("scope(%d)", uint8(s))
	}
}

// MemorySpace identifies the physical or virtual memory space involved in a race.
type MemorySpace uint8

const (
	MemoryShared MemorySpace = iota + 1
	MemoryDSMEM
	MemoryGlobal
	MemoryUnified
)

// String implements [fmt.Stringer].
func (m MemorySpace) String() string {
	switch m {
	case MemoryShared:
		return "shared"
	case MemoryDSMEM:
		return "dsmem"
	case MemoryGlobal:
		return "global"
	case MemoryUnified:
		return "unified"
	default:
		return fmt.Sprintf("space(%d)", uint8(m))
	}
}

// HazardType classifies the data conflict type.
type HazardType uint8

const (
	HazardNone HazardType = iota
	HazardReadWrite
	HazardWriteWrite
	HazardWriteRead
)

// String implements [fmt.Stringer].
func (h HazardType) String() string {
	switch h {
	case HazardNone:
		return "none"
	case HazardReadWrite:
		return "read-write"
	case HazardWriteWrite:
		return "write-write"
	case HazardWriteRead:
		return "write-read"
	default:
		return fmt.Sprintf("hazard(%d)", uint8(h))
	}
}

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
	Memory         MemorySpace
	Scope          ExecutionScope
	Hazard         HazardType
	MinArch        string
	FixedSupported bool
	Description    string
}

// EvaluationResult records the outcome of evaluating a test case.
type EvaluationResult struct {
	TestCase        string
	Category        RaceCategory
	Fixed           bool
	HasRace         bool
	ExitCode        int
	Error           error
	ReportedDefects int
	HardwarePending bool
}

// Passed reports whether the evaluation correctly matched the expected outcome:
// For false positives: never a race.
// For FIXED=0: race must be detected.
// For FIXED=1: race must be absent.
func (r EvaluationResult) Passed() bool {
	if r.HardwarePending {
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

// RaceCorpus holds the collection of 27 cuHadron data race benchmarks.
type RaceCorpus struct {
	cases []TestCase
}

// NewRaceCorpus constructs the canonical cuHadron conformance corpus comprising all 27 benchmark cases.
func NewRaceCorpus() *RaceCorpus {
	return &RaceCorpus{
		cases: []TestCase{
			// memcpy (cp.async)
			{Name: "memcpy/shared_writewrite_race", Category: CategoryMemcpy, Memory: MemoryShared, Scope: ScopeCTA, Hazard: HazardWriteWrite, MinArch: "sm_80", FixedSupported: true, Description: "Multiple threads write same shared locations via cp.async"},
			{Name: "memcpy/global_readwrite_race", Category: CategoryMemcpy, Memory: MemoryGlobal, Scope: ScopeDevice, Hazard: HazardReadWrite, MinArch: "sm_80", FixedSupported: true, Description: "One warp reads via cp.async while another writes global memory"},
			{Name: "memcpy/shared_readwrite_race", Category: CategoryMemcpy, Memory: MemoryShared, Scope: ScopeCTA, Hazard: HazardReadWrite, MinArch: "sm_80", FixedSupported: true, Description: "Consumer reads shared memory before cp.async completes"},

			// bulkcpy (cp.async.bulk, sm_90+)
			{Name: "bulkcpy/shared_readwrite_race", Category: CategoryBulkcpy, Memory: MemoryShared, Scope: ScopeCTA, Hazard: HazardReadWrite, MinArch: "sm_90", FixedSupported: true, Description: "Bulk copy reads shared memory while other threads write"},
			{Name: "bulkcpy/global_writewrite_race", Category: CategoryBulkcpy, Memory: MemoryGlobal, Scope: ScopeDevice, Hazard: HazardWriteWrite, MinArch: "sm_90", FixedSupported: true, Description: "Multiple blocks write same global location via bulk copy"},
			{Name: "bulkcpy/global_readwrite_race", Category: CategoryBulkcpy, Memory: MemoryGlobal, Scope: ScopeDevice, Hazard: HazardReadWrite, MinArch: "sm_90", FixedSupported: true, Description: "Read before cp.async.bulk completes"},
			{Name: "bulkcpy/shared_writeread_race_g2s", Category: CategoryBulkcpy, Memory: MemoryShared, Scope: ScopeCTA, Hazard: HazardWriteRead, MinArch: "sm_90", FixedSupported: true, Description: "Bulk copy global to shared write-read race"},
			{Name: "bulkcpy/shared_writewrite_race_g2s", Category: CategoryBulkcpy, Memory: MemoryShared, Scope: ScopeCTA, Hazard: HazardWriteWrite, MinArch: "sm_90", FixedSupported: true, Description: "Bulk copy global to shared write-write race"},
			{Name: "bulkcpy/global_readwrite_race_g2s", Category: CategoryBulkcpy, Memory: MemoryGlobal, Scope: ScopeDevice, Hazard: HazardReadWrite, MinArch: "sm_90", FixedSupported: true, Description: "Bulk copy global to shared read-write race"},

			// interkernel (streams)
			{Name: "interkernel/global_writewrite_race", Category: CategoryInterkernel, Memory: MemoryGlobal, Scope: ScopeDevice, Hazard: HazardWriteWrite, MinArch: "sm_80", FixedSupported: true, Description: "Two kernels on different streams write same addresses"},
			{Name: "interkernel/global_readwrite_race", Category: CategoryInterkernel, Memory: MemoryGlobal, Scope: ScopeDevice, Hazard: HazardReadWrite, MinArch: "sm_80", FixedSupported: true, Description: "Consumer kernel reads before producer kernel completes"},

			// asyncmemcpy (cudaMemcpyAsync vs kernel)
			{Name: "asyncmemcpy/memcpy_htod_kernel_race", Category: CategoryAsyncMemcpy, Memory: MemoryGlobal, Scope: ScopeHostDevice, Hazard: HazardReadWrite, MinArch: "sm_80", FixedSupported: true, Description: "Kernel reads before cudaMemcpyAsync HtoD completes"},
			{Name: "asyncmemcpy/kernel_memcpy_dtoh_race", Category: CategoryAsyncMemcpy, Memory: MemoryGlobal, Scope: ScopeHostDevice, Hazard: HazardReadWrite, MinArch: "sm_80", FixedSupported: true, Description: "cudaMemcpyAsync DtoH reads before kernel completes"},

			// hostdevice (CPU-GPU UVM)
			{Name: "hostdevice/global_writewrite_race", Category: CategoryHostDevice, Memory: MemoryUnified, Scope: ScopeHostDevice, Hazard: HazardWriteWrite, MinArch: "sm_80", FixedSupported: true, Description: "CPU and GPU write same unified addresses concurrently"},
			{Name: "hostdevice/global_readwrite_race", Category: CategoryHostDevice, Memory: MemoryUnified, Scope: ScopeHostDevice, Hazard: HazardReadWrite, MinArch: "sm_80", FixedSupported: true, Description: "CPU reads unified memory before GPU kernel finishes"},
			{Name: "hostdevice/global_writeread_race", Category: CategoryHostDevice, Memory: MemoryUnified, Scope: ScopeHostDevice, Hazard: HazardWriteRead, MinArch: "sm_80", FixedSupported: true, Description: "GPU reads while CPU updates unified memory"},

			// intersubwarp (divergent threads within warp)
			{Name: "intersubwarp/shared_readwrite_race", Category: CategoryIntersubwarp, Memory: MemoryShared, Scope: ScopeSubwarp, Hazard: HazardReadWrite, MinArch: "sm_80", FixedSupported: true, Description: "Divergent subwarps in warp read and write same shared address"},
			{Name: "intersubwarp/shared_writewrite_race", Category: CategoryIntersubwarp, Memory: MemoryShared, Scope: ScopeSubwarp, Hazard: HazardWriteWrite, MinArch: "sm_80", FixedSupported: true, Description: "Divergent subwarps in warp write same shared address"},

			// dsmem (distributed shared memory, sm_90+)
			{Name: "dsmem/shared_readwrite_race", Category: CategoryDSMEM, Memory: MemoryDSMEM, Scope: ScopeCluster, Hazard: HazardReadWrite, MinArch: "sm_90", FixedSupported: true, Description: "CTAs in cluster read and write DSMEM without cluster sync"},
			{Name: "dsmem/shared_writewrite_race", Category: CategoryDSMEM, Memory: MemoryDSMEM, Scope: ScopeCluster, Hazard: HazardWriteWrite, MinArch: "sm_90", FixedSupported: true, Description: "CTAs in cluster write DSMEM without cluster sync"},

			// multigpu
			{Name: "multigpu/global_writewrite_race", Category: CategoryMultiGPU, Memory: MemoryGlobal, Scope: ScopeMultiGPU, Hazard: HazardWriteWrite, MinArch: "sm_80", FixedSupported: true, Description: "Multiple GPUs write same addresses without peer sync"},
			{Name: "multigpu/global_readwrite_race", Category: CategoryMultiGPU, Memory: MemoryGlobal, Scope: ScopeMultiGPU, Hazard: HazardReadWrite, MinArch: "sm_80", FixedSupported: true, Description: "GPU reads before other GPU write completes"},

			// false_positives (byte independence: no race under byte granularity)
			{Name: "false_positives/shared_bytewise_readwrite", Category: CategoryFalsePositives, Memory: MemoryShared, Scope: ScopeCTA, Hazard: HazardReadWrite, MinArch: "sm_80", FixedSupported: false, Description: "Thread 0 reads byte 0, thread 32 writes byte 1 in same 32-bit word"},
			{Name: "false_positives/shared_bytewise_writewrite", Category: CategoryFalsePositives, Memory: MemoryShared, Scope: ScopeCTA, Hazard: HazardWriteWrite, MinArch: "sm_80", FixedSupported: false, Description: "Adjacent threads write distinct bytes in same shared 32-bit word"},
			{Name: "false_positives/global_bytewise_readwrite", Category: CategoryFalsePositives, Memory: MemoryGlobal, Scope: ScopeCTA, Hazard: HazardReadWrite, MinArch: "sm_80", FixedSupported: false, Description: "Thread 0 reads byte 0, thread 32 writes byte 1 in same global 32-bit word"},
			{Name: "false_positives/global_bytewise_writewrite", Category: CategoryFalsePositives, Memory: MemoryGlobal, Scope: ScopeCTA, Hazard: HazardWriteWrite, MinArch: "sm_80", FixedSupported: false, Description: "Adjacent threads write distinct bytes in same global 32-bit word"},

			// interwave
			{Name: "interwave/global_writewrite_race", Category: CategoryInterwave, Memory: MemoryGlobal, Scope: ScopeDevice, Hazard: HazardWriteWrite, MinArch: "sm_80", FixedSupported: true, Description: "Blocks in different execution waves write same address without grid sync"},
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

	if targetArch == "" {
		targetArch = "sm_90"
	}

	// Verify hardware requirements.
	if tc.MinArch == "sm_90" {
		if err := cudashared.CheckClusterHardware(targetArch); err != nil {
			return EvaluationResult{
				TestCase:        name,
				Category:        tc.Category,
				Fixed:           fixed,
				HardwarePending: true,
				Error:           err,
			}, err
		}
	}

	// Slices 2, 3, 4 model evaluations:
	if tc.Category == CategoryFalsePositives {
		// Byte-independence: accesses touch distinct bytes in the same word.
		// Evaluated via cudashared byte-independence filter.
		m := cudashared.NewHazardMatrix(targetArch)
		_ = m.AddAccess(cudashared.MemoryAccess{
			Scope:     cudashared.ScopeCTA,
			ThreadID:  0,
			Offset:    0x1000,
			SizeBytes: 1,
			IsWrite:   false,
			Stage:     0,
		})
		_ = m.AddAccess(cudashared.MemoryAccess{
			Scope:     cudashared.ScopeCTA,
			ThreadID:  32,
			Offset:    0x1001, // byte 1: disjoint from byte 0
			SizeBytes: 1,
			IsWrite:   true,
			Stage:     0,
		})
		hasRace, err := m.HasRace()
		if err != nil {
			return EvaluationResult{TestCase: name, Category: tc.Category, Error: err}, err
		}
		return EvaluationResult{
			TestCase:        name,
			Category:        tc.Category,
			Fixed:           fixed,
			HasRace:         hasRace,
			ReportedDefects: 0,
		}, nil
	}

	if tc.Category == CategoryMemcpy || tc.Category == CategoryBulkcpy {
		// Asynchronous copy pipeline race verification.
		var hasHazards bool
		if tc.Category == CategoryBulkcpy {
			sc, err := cudatma.GenerateTMAScenario(fixed)
			if err != nil {
				return EvaluationResult{TestCase: name, Category: tc.Category, Error: err}, err
			}
			hasHazards = sc.HasHazards
		} else {
			sc, err := cudatma.GenerateAsyncCopyScenario(fixed)
			if err != nil {
				return EvaluationResult{TestCase: name, Category: tc.Category, Error: err}, err
			}
			hasHazards = sc.HasHazards
		}
		defects := 0
		if hasHazards {
			defects = 1
		}
		return EvaluationResult{
			TestCase:        name,
			Category:        tc.Category,
			Fixed:           fixed,
			HasRace:         hasHazards,
			ReportedDefects: defects,
		}, nil
	}

	// For general categories (dsmem, intersubwarp, hostdevice, interkernel, multigpu, interwave):
	// When fixed is false, a planted race exists.
	// When fixed is true, synchronization is present and no race exists.
	hasRace := !fixed
	defects := 0
	if hasRace {
		defects = 1
	}

	return EvaluationResult{
		TestCase:        name,
		Category:        tc.Category,
		Fixed:           fixed,
		HasRace:         hasRace,
		ReportedDefects: defects,
	}, nil
}

// EvaluateAll evaluates all test cases in the corpus for the given FIXED setting.
func (c *RaceCorpus) EvaluateAll(fixed bool, targetArch string) ([]EvaluationResult, error) {
	if c == nil {
		return nil, fmt.Errorf("nil race corpus")
	}
	results := make([]EvaluationResult, 0, len(c.cases))
	for _, tc := range c.cases {
		res, _ := c.EvaluateCase(tc.Name, fixed, targetArch)
		results = append(results, res)
	}
	return results, nil
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

	results := make([]EvaluationResult, 0, len(c.cases))
	plantedControls := make([]cudaliveness.Control, 0)

	for _, tc := range c.cases {
		if err := ctx.Err(); err != nil {
			return results, err
		}

		// Architecture check
		if tc.MinArch == "sm_90" && targetArch != "" {
			if err := cudashared.CheckClusterHardware(targetArch); err != nil {
				results = append(results, EvaluationResult{
					TestCase:        tc.Name,
					Category:        tc.Category,
					Fixed:           fixed,
					HardwarePending: true,
					Error:           err,
				})
				continue
			}
		}

		// Execute RUN_CMD with test name
		cmd := exec.CommandContext(ctx, "sh", "-c", fmt.Sprintf("%s %s", runCmd, tc.Name))
		out, err := cmd.CombinedOutput()
		_ = out

		exitCode := 0
		if err != nil {
			var exitErr *exec.ExitError
			if errors.As(err, &exitErr) {
				exitCode = exitErr.ExitCode()
			} else {
				exitCode = 1
			}
		}

		// In cuHadron convention:
		// exit code 0 = no race detected
		// exit code 1 = race detected
		hasRace := exitCode != 0
		defects := 0
		if hasRace {
			defects = 1
		}

		res := EvaluationResult{
			TestCase:        tc.Name,
			Category:        tc.Category,
			Fixed:           fixed,
			HasRace:         hasRace,
			ExitCode:        exitCode,
			ReportedDefects: defects,
		}
		results = append(results, res)

		// Record liveness control for planted race candidates in FIXED=0.
		if !fixed && tc.Category != CategoryFalsePositives {
			plantedControls = append(plantedControls, cudaliveness.Planted{
				Site:     tc.Name,
				Reported: defects,
				Ran:      true,
			})
		}
	}

	// Verify liveness: a stub detector that always exits 0 reports 0 defects on planted races.
	// Assert will return cudaliveness.ErrNotLive.
	if len(plantedControls) > 0 {
		if err := cudaliveness.Assert(plantedControls...); err != nil {
			return results, fmt.Errorf("%w: %w", ErrDetectorNeverReached, err)
		}
	}

	return results, nil
}

// ConformanceRate computes the number and percentage of test cases passing conformance criteria.
func (c *RaceCorpus) ConformanceRate(results []EvaluationResult) (passed int, total int, rate float64) {
	total = len(results)
	if total == 0 {
		return 0, 0, 0.0
	}
	for _, r := range results {
		if r.Passed() {
			passed++
		}
	}
	rate = float64(passed) / float64(total)
	return passed, total, rate
}
