package cudatrial

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudashared"
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
	Unmodelled        bool
	UnknownExit       bool
	EvaluatedPairs    int
	BinaryPath        string
	Output            string
	Configuration     string
	IsLivenessControl bool
}

// Passed reports whether the evaluation correctly matched the expected outcome:
// For false positives: never a race.
// For FIXED=0: race must be detected.
// For FIXED=1: race must be absent.
func (r EvaluationResult) Passed() bool {
	if r.IsLivenessControl {
		return r.Error == nil && r.ExitCode == 1 && r.HasRace
	}
	if r.HardwarePending || r.Unmodelled || r.UnknownExit || r.Error != nil || (r.ExitCode != 0 && r.ExitCode != 1) {
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

// ExecControl is a validated lane 52 execution-control receipt. Its fields are
// private so a result row cannot claim control status with a struct literal.
// Lane 52 writes JSON with `configuration` and `passed` fields, then callers
// pass the value returned by [ReadExecControl] to [RaceCorpus.ConformanceRate].
// ConformanceRate re-reads and validates the file on each call.
type ExecControl struct {
	configuration string
	evidencePath  string
}

type execControlReceipt struct {
	Configuration string `json:"configuration"`
	Passed        bool   `json:"passed"`
}

// ReadExecControl validates a lane 52 JSON receipt and retains its path as
// evidence. The required JSON shape is
// `{"configuration":"FIXED=1;ARCH=sm_90","passed":true}`.
func ReadExecControl(path string) (ExecControl, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return ExecControl{}, fmt.Errorf("read exec-control evidence: %w", err)
	}
	var receipt execControlReceipt
	if err := json.Unmarshal(data, &receipt); err != nil {
		return ExecControl{}, fmt.Errorf("parse exec-control evidence: %w", err)
	}
	if strings.TrimSpace(receipt.Configuration) == "" || !receipt.Passed {
		return ExecControl{}, fmt.Errorf("exec-control evidence is not a passing configured receipt")
	}
	return ExecControl{configuration: receipt.Configuration, evidencePath: path}, nil
}

// Configuration returns the run configuration named by the validated receipt.
func (c ExecControl) Configuration() string { return c.configuration }

// EvidencePath returns the file that supplied the validated receipt.
func (c ExecControl) EvidencePath() string { return c.evidencePath }

// ConformanceSummary contains disjoint result counts for the corpus.
type ConformanceSummary struct {
	Passed          int
	Failed          int
	HardwarePending int
	Unmodelled      int
	UnknownExit     int
	Total           int
	Rate            float64
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
			{Name: "memcpy/global_readwrite_race", Category: CategoryMemcpy, Memory: cudashared.MemoryGlobal, Scope: cudashared.ScopeCTA, Hazard: cudashared.HazardReadWrite, MinArch: "sm_80", FixedSupported: true, Description: "One warp reads via cp.async while another writes global memory"},
			{Name: "memcpy/shared_readwrite_race", Category: CategoryMemcpy, Memory: cudashared.MemoryShared, Scope: cudashared.ScopeCTA, Hazard: cudashared.HazardReadWrite, MinArch: "sm_80", FixedSupported: true, Description: "Consumer reads shared memory before cp.async completes"},

			// bulkcpy (cp.async.bulk, sm_90+)
			{Name: "bulkcpy/shared_readwrite_race", Category: CategoryBulkcpy, Memory: cudashared.MemoryShared, Scope: cudashared.ScopeCTA, Hazard: cudashared.HazardReadWrite, MinArch: "sm_90", FixedSupported: true, Description: "Bulk copy reads shared memory before other threads write"},
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

func architectureNumber(arch string) (int, error) {
	arch = strings.ToLower(strings.TrimSpace(arch))
	if !strings.HasPrefix(arch, "sm_") {
		return 0, fmt.Errorf("unrecognized architecture %q", arch)
	}
	value := strings.TrimPrefix(arch, "sm_")
	value = strings.TrimRight(value, "abcdefghijklmnopqrstuvwxyz")
	if value == "" {
		return 0, fmt.Errorf("unrecognized architecture %q", arch)
	}
	n, err := strconv.Atoi(value)
	if err != nil {
		return 0, fmt.Errorf("unrecognized architecture %q: %w", arch, err)
	}
	return n, nil
}

func hardwarePending(tc TestCase, targetArch string) error {
	if targetArch == "" {
		return fmt.Errorf("%s requires a target architecture: %w", tc.Name, cudaerrors.ErrHardwarePending)
	}
	target, err := architectureNumber(targetArch)
	if err != nil {
		return fmt.Errorf("%s target is unknown: %w: %w", tc.Name, cudaerrors.ErrHardwarePending, err)
	}
	minimum, err := architectureNumber(tc.MinArch)
	if err != nil {
		return fmt.Errorf("%s minimum architecture is invalid: %w", tc.Name, err)
	}
	if target < minimum {
		return fmt.Errorf("%s requires %s; target %s is below it: %w", tc.Name, tc.MinArch, targetArch, cudaerrors.ErrHardwarePending)
	}
	if tc.Memory == cudashared.MemoryDSMEM || tc.Scope == cudashared.ScopeCluster {
		return cudashared.CheckClusterHardware(targetArch)
	}
	return nil
}

func binaryArchSuffix(targetArch string) string {
	return strings.ReplaceAll(strings.TrimSpace(targetArch), "_", "")
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
	if err := hardwarePending(tc, targetArch); err != nil {
		base.HardwarePending, base.Error = errors.Is(err, cudaerrors.ErrHardwarePending), err
		return base, err
	}
	accesses, err := caseAccesses(tc.Name, fixed)
	if err != nil {
		base.Unmodelled = true
		base.Error = fmt.Errorf("unmodelled case %s: %w", name, err)
		return base, base.Error
	}
	base.HasRace, base.EvaluatedPairs, err = evaluateAccesses(targetArch, accesses)
	if err != nil {
		base.HardwarePending, base.Error = errors.Is(err, cudaerrors.ErrHardwarePending), err
		return base, err
	}
	if base.HasRace {
		base.ReportedDefects = 1
	}
	return base, nil
}

func evaluateAccesses(targetArch string, accesses []cudashared.MemoryAccess) (bool, int, error) {
	m := cudashared.NewHazardMatrix(targetArch)
	for _, access := range accesses {
		if err := m.AddAccess(access); err != nil {
			return false, 0, err
		}
	}
	records, err := m.Evaluate()
	if err != nil {
		return false, 0, err
	}
	hasRace := false
	for _, record := range records {
		hasRace = hasRace || record.IsRace
	}
	return hasRace, len(records), nil
}

type caseAccessModel struct {
	scope  cudashared.ExecutionScope
	space  cudashared.MemorySpace
	hazard cudashared.HazardType
	fix    string
	kind   string
}

var accessModels = map[string]caseAccessModel{
	"memcpy/shared_writewrite_race":              {cudashared.ScopeCTA, cudashared.MemoryShared, cudashared.HazardWriteWrite, "disjoint", "threads"},
	"memcpy/global_readwrite_race":               {cudashared.ScopeCTA, cudashared.MemoryGlobal, cudashared.HazardReadWrite, "sync", "threads"},
	"memcpy/shared_readwrite_race":               {cudashared.ScopeCTA, cudashared.MemoryShared, cudashared.HazardReadWrite, "sync", "threads"},
	"bulkcpy/shared_readwrite_race":              {cudashared.ScopeCTA, cudashared.MemoryShared, cudashared.HazardReadWrite, "sync", "threads"},
	"bulkcpy/global_writewrite_race":             {cudashared.ScopeDevice, cudashared.MemoryGlobal, cudashared.HazardWriteWrite, "disjoint", "blocks"},
	"bulkcpy/global_readwrite_race":              {cudashared.ScopeDevice, cudashared.MemoryGlobal, cudashared.HazardReadWrite, "sync", "threads"},
	"bulkcpy/shared_writeread_race_g2s":          {cudashared.ScopeCTA, cudashared.MemoryShared, cudashared.HazardWriteRead, "sync", "threads"},
	"bulkcpy/shared_writewrite_race_g2s":         {cudashared.ScopeCTA, cudashared.MemoryShared, cudashared.HazardWriteWrite, "sync", "threads"},
	"bulkcpy/global_readwrite_race_g2s":          {cudashared.ScopeDevice, cudashared.MemoryGlobal, cudashared.HazardReadWrite, "sync", "threads"},
	"interkernel/global_writewrite_race":         {cudashared.ScopeDevice, cudashared.MemoryGlobal, cudashared.HazardWriteWrite, "disjoint", "kernels"},
	"interkernel/global_readwrite_race":          {cudashared.ScopeDevice, cudashared.MemoryGlobal, cudashared.HazardReadWrite, "sync", "kernels"},
	"asyncmemcpy/memcpy_htod_kernel_race":        {cudashared.ScopeHostDevice, cudashared.MemoryGlobal, cudashared.HazardReadWrite, "sync", "agents"},
	"asyncmemcpy/kernel_memcpy_dtoh_race":        {cudashared.ScopeHostDevice, cudashared.MemoryGlobal, cudashared.HazardReadWrite, "sync", "agents"},
	"hostdevice/global_writewrite_race":          {cudashared.ScopeHostDevice, cudashared.MemoryUnified, cudashared.HazardWriteWrite, "sync", "agents"},
	"hostdevice/global_readwrite_race":           {cudashared.ScopeHostDevice, cudashared.MemoryUnified, cudashared.HazardReadWrite, "sync", "agents"},
	"hostdevice/global_writeread_race":           {cudashared.ScopeHostDevice, cudashared.MemoryUnified, cudashared.HazardWriteRead, "sync", "agents"},
	"intersubwarp/shared_readwrite_race":         {cudashared.ScopeSubwarp, cudashared.MemoryShared, cudashared.HazardReadWrite, "disjoint", "subwarps"},
	"intersubwarp/shared_writewrite_race":        {cudashared.ScopeSubwarp, cudashared.MemoryShared, cudashared.HazardWriteWrite, "disjoint", "subwarps"},
	"dsmem/shared_readwrite_race":                {cudashared.ScopeCluster, cudashared.MemoryDSMEM, cudashared.HazardReadWrite, "sync", "cluster"},
	"dsmem/shared_writewrite_race":               {cudashared.ScopeCluster, cudashared.MemoryDSMEM, cudashared.HazardWriteWrite, "sync", "cluster"},
	"false_positives/shared_bytewise_readwrite":  {cudashared.ScopeCTA, cudashared.MemoryShared, cudashared.HazardReadWrite, "bytes", "false-positive"},
	"false_positives/shared_bytewise_writewrite": {cudashared.ScopeCTA, cudashared.MemoryShared, cudashared.HazardWriteWrite, "bytes", "false-positive"},
	"false_positives/global_bytewise_readwrite":  {cudashared.ScopeCTA, cudashared.MemoryGlobal, cudashared.HazardReadWrite, "bytes", "false-positive"},
	"false_positives/global_bytewise_writewrite": {cudashared.ScopeCTA, cudashared.MemoryGlobal, cudashared.HazardWriteWrite, "bytes", "false-positive"},
	"interwave/global_writewrite_race":           {cudashared.ScopeDevice, cudashared.MemoryGlobal, cudashared.HazardWriteWrite, "disjoint", "blocks"},
	"multigpu/global_writewrite_race":            {cudashared.ScopeMultiGPU, cudashared.MemoryGlobal, cudashared.HazardWriteWrite, "unmodelled", "multigpu"},
	"multigpu/global_readwrite_race":             {cudashared.ScopeMultiGPU, cudashared.MemoryGlobal, cudashared.HazardReadWrite, "unmodelled", "multigpu"},
}

func caseAccesses(name string, fixed bool) ([]cudashared.MemoryAccess, error) {
	model, ok := accessModels[name]
	if !ok {
		return nil, fmt.Errorf("no access model for case %q", name)
	}
	if model.kind == "multigpu" {
		return nil, fmt.Errorf("multi-GPU ownership is not represented by MemoryAccess")
	}
	a := cudashared.MemoryAccess{Scope: model.scope, Space: model.space, AgentID: 1, ThreadID: 0, BlockID: 0, ClusterID: 0, Offset: 0x1000, SizeBytes: 4, Stage: 0}
	b := cudashared.MemoryAccess{Scope: model.scope, Space: model.space, AgentID: 1, ThreadID: 1, BlockID: 0, ClusterID: 0, Offset: 0x1000, SizeBytes: 4, Stage: 0}
	switch model.hazard {
	case cudashared.HazardReadWrite:
		b.IsWrite = true
	case cudashared.HazardWriteRead:
		a.IsWrite = true
	case cudashared.HazardWriteWrite:
		a.IsWrite, b.IsWrite = true, true
	default:
		return nil, fmt.Errorf("case %q has invalid access model", name)
	}
	switch model.kind {
	case "threads":
	case "kernels":
		a.AgentID, b.AgentID = 1, 2
		b.ThreadID = a.ThreadID
		a.StreamID, b.StreamID = 1, 2
	case "agents":
		a.AgentID, b.AgentID = 1, 2
		b.ThreadID, b.BlockID, b.ClusterID = a.ThreadID, a.BlockID, a.ClusterID
	case "subwarps":
		b.ThreadID = 16
		if fixed {
			b.Offset += 16
		}
	case "cluster":
		b.BlockID = 1
	case "blocks":
		b.ThreadID, b.BlockID = 0, 1
	case "false-positive":
		b.ThreadID = 32
		a.SizeBytes, b.SizeBytes, b.Offset = 1, 1, a.Offset+1
	default:
		return nil, fmt.Errorf("case %q has unknown model kind %q", name, model.kind)
	}
	if fixed {
		switch model.fix {
		case "disjoint":
			b.Offset += 4
		case "sync":
			b.Stage = 1
		case "bytes":
		case "unmodelled":
		default:
			return nil, fmt.Errorf("case %q has unknown fixed mechanism %q", name, model.fix)
		}
	}
	return []cudashared.MemoryAccess{a, b}, nil
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
// It verifies detector liveness with an independent FIXED=0 planted-race
// binary on every sweep. A stub RUN_CMD that exits 0 on that input fails the
// run rather than producing an unfalsifiable clean sweep.
func (c *RaceCorpus) RunExternal(ctx context.Context, runCmd string, fixed bool, targetArch string) ([]EvaluationResult, error) {
	if c == nil {
		return nil, fmt.Errorf("nil race corpus")
	}
	if strings.TrimSpace(runCmd) == "" {
		return nil, fmt.Errorf("empty RUN_CMD")
	}
	if targetArch == "" {
		pending := make([]EvaluationResult, 0, len(c.cases))
		var errs []error
		fixedValue := "0"
		if fixed {
			fixedValue = "1"
		}
		for _, tc := range c.cases {
			err := hardwarePending(tc, targetArch)
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
	var runErrors []error

	// This binary is built with FIXED=0 and is run on every sweep, including a
	// FIXED=1 sweep, so detector liveness is never inferred from labels.
	archSuffix := binaryArchSuffix(targetArch)
	controlPath := filepath.Join(root, "FIXED=0", "memcpy", "shared_readwrite_race."+archSuffix+".out")
	control, _ := runDetector(ctx, runCmd, controlPath)
	control.TestCase, control.Category, control.Configuration, control.Fixed, control.BinaryFixed, control.IsLivenessControl = "cudatrial/planted-race-control", CategoryMemcpy, configuration, fixed, false, true
	results = append(results, control)
	if control.Error != nil {
		runErrors = append(runErrors, control.Error)
	}
	if !control.Passed() || !control.HasRace {
		runErrors = append(runErrors, fmt.Errorf("%w: planted FIXED=0 control was not detected", ErrDetectorNeverReached))
	}

	for _, tc := range c.cases {
		if err := ctx.Err(); err != nil {
			return results, err
		}
		if model, ok := accessModels[tc.Name]; ok && model.kind == "multigpu" {
			results = append(results, EvaluationResult{
				TestCase:      tc.Name,
				Category:      tc.Category,
				Fixed:         fixed,
				Unmodelled:    true,
				Error:         fmt.Errorf("multi-GPU ownership is not represented by MemoryAccess"),
				Configuration: configuration,
			})
			continue
		}

		// Architecture check. An empty target is unknown, not a default sm_90.
		if err := hardwarePending(tc, targetArch); err != nil {
			if errors.Is(err, cudaerrors.ErrHardwarePending) {
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
			res := EvaluationResult{TestCase: tc.Name, Category: tc.Category, Fixed: fixed, Error: err, Configuration: configuration}
			results = append(results, res)
			runErrors = append(runErrors, err)
			continue
		}

		caseRoot := filepath.Join(root, fixedName)
		binaryPath := filepath.Join(caseRoot, tc.Name+"."+archSuffix+".out")
		res, runErr := runDetector(ctx, runCmd, binaryPath)
		res.TestCase, res.Category, res.Fixed, res.BinaryFixed, res.Configuration = tc.Name, tc.Category, fixed, fixed, configuration
		if runErr != nil {
			res.Error = runErr
			runErrors = append(runErrors, runErr)
		}
		results = append(results, res)

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
		res.UnknownExit = true
		res.Error = fmt.Errorf("detector exited %d: %s", res.ExitCode, res.Output)
		return res, res.Error
	}
	return res, nil
}

// ConformanceRate requires a separately validated lane 52 execution receipt
// and exactly one result for each corpus case.
func (c *RaceCorpus) ConformanceRate(results []EvaluationResult, control ExecControl) (ConformanceSummary, error) {
	var summary ConformanceSummary
	if c == nil || len(c.cases) == 0 {
		return summary, fmt.Errorf("conformance rate requires a non-empty corpus")
	}
	if control.configuration == "" || control.evidencePath == "" {
		return summary, fmt.Errorf("conformance rate requires validated exec-control evidence")
	}
	current, err := ReadExecControl(control.evidencePath)
	if err != nil {
		return summary, fmt.Errorf("exec-control evidence is no longer valid: %w", err)
	}
	if current.configuration != control.configuration {
		return summary, fmt.Errorf("exec-control evidence configuration changed from %q to %q", control.configuration, current.configuration)
	}
	want := make(map[string]bool, len(c.cases))
	for _, tc := range c.cases {
		want[tc.Name] = true
	}
	seen := make(map[string]bool, len(c.cases))
	plantedCount := 0
	for _, r := range results {
		if r.TestCase == "cudatrial/planted-race-control" {
			plantedCount++
			if !r.IsLivenessControl || r.Configuration != control.configuration || !r.Passed() || !r.HasRace {
				return summary, fmt.Errorf("planted-race control is missing, failed, or mismatched")
			}
			continue
		}
		if !want[r.TestCase] {
			return summary, fmt.Errorf("result %q is not a corpus case", r.TestCase)
		}
		if seen[r.TestCase] {
			return summary, fmt.Errorf("duplicate corpus result %q", r.TestCase)
		}
		seen[r.TestCase] = true
		if r.Configuration != control.configuration {
			return summary, fmt.Errorf("result %q configuration %q does not match exec-control %q", r.TestCase, r.Configuration, control.configuration)
		}
		summary.Total++
		switch {
		case r.HardwarePending:
			summary.HardwarePending++
		case r.Unmodelled:
			summary.Unmodelled++
		case r.UnknownExit:
			summary.UnknownExit++
		case r.Passed():
			summary.Passed++
		default:
			summary.Failed++
		}
	}
	if plantedCount != 1 {
		return ConformanceSummary{}, fmt.Errorf("conformance rate requires exactly one passing planted-race control, got %d", plantedCount)
	}
	for name := range want {
		if !seen[name] {
			return ConformanceSummary{}, fmt.Errorf("missing corpus result %q", name)
		}
	}
	if summary.Total != len(c.cases) {
		return ConformanceSummary{}, fmt.Errorf("corpus result count %d, want %d", summary.Total, len(c.cases))
	}
	summary.Rate = float64(summary.Passed) / float64(summary.Total)
	return summary, nil
}
