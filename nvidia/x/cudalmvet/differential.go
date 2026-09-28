package cudalmvet

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudaref"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// ErrContaminatedAccumulator indicates that the device accumulator was not reset
// to FNVOffsetBasis before a launch.
var ErrContaminatedAccumulator = errors.New("cudalmvet: device accumulator was not reset prior to launch (contaminated state)")

type stageFailure struct {
	stage FailureStage
	err   error
}

func (e stageFailure) Error() string              { return e.err.Error() }
func (e stageFailure) Unwrap() error              { return e.err }
func atStage(stage FailureStage, err error) error { return stageFailure{stage: stage, err: err} }
func failureStage(err error, fallback FailureStage) FailureStage {
	var failure stageFailure
	if errors.As(err, &failure) {
		return failure.stage
	}
	return fallback
}

// ComparisonStatus describes the outcome of comparing CPU reference and GPU device trace hashes.
type ComparisonStatus int

const (
	// StatusUnmeasured indicates the device side could not run (e.g. no GPU device present).
	StatusUnmeasured ComparisonStatus = iota

	// StatusAgreed indicates the CPU reference and GPU device trace hashes match exactly.
	StatusAgreed

	// StatusDisagreed indicates the CPU reference and GPU device trace hashes diverge.
	StatusDisagreed
)

func (s ComparisonStatus) String() string {
	switch s {
	case StatusUnmeasured:
		return "Unmeasured"
	case StatusAgreed:
		return "Agreed"
	case StatusDisagreed:
		return "Disagreed"
	default:
		return fmt.Sprintf("ComparisonStatus(%d)", int(s))
	}
}

// ComparisonResult holds the results of comparing CPU and GPU trace executions.
type ComparisonResult struct {
	Status     ComparisonStatus
	Stage      FailureStage
	CPUHash    uint64
	DeviceHash uint64
	Triage     *TriageReport
	Err        error
}

// FailureStage identifies the stage that made a device result unavailable.
type FailureStage uint8

const (
	StageNone FailureStage = iota
	StageCompile
	StageLoad
	StageLaunch
	StageReadback
)

func (r *ComparisonResult) String() string {
	if r == nil {
		return "ComparisonResult(nil)"
	}
	switch r.Status {
	case StatusAgreed:
		return fmt.Sprintf("Agreed (hash: 0x%016x)", r.CPUHash)
	case StatusDisagreed:
		msg := fmt.Sprintf("Disagreed (CPU: 0x%016x, Device: 0x%016x)", r.CPUHash, r.DeviceHash)
		if r.Triage != nil {
			msg += "\n" + r.Triage.String()
		}
		return msg
	case StatusUnmeasured:
		if r.Err != nil {
			return fmt.Sprintf("Unmeasured: %v", r.Err)
		}
		return fmt.Sprintf("Unmeasured (CPU reference hash: 0x%016x)", r.CPUHash)
	default:
		return fmt.Sprintf("Unknown status %v", r.Status)
	}
}

// CompareConfig holds configuration options for Compare.
type CompareConfig struct {
	Inputs           map[string][]float32
	Device           *cuda.Device
	Policy           FloatPolicy
	KernelSource     string
	EntryPoint       string
	CompilerOpts     []nvrtcx.Option
	JITOptions       cudakernel.JITOptions
	TriageOnMismatch bool
	VerifyCleanReset bool
	perturbation     *Perturbation
	compileKernel    func(string, ...nvrtcx.Option) (*nvrtcx.CompiledModule, error)
}

// CompareOption configures a Compare execution.
type CompareOption func(*CompareConfig)

// WithInputs sets the input tensor buffers for graph evaluation.
func WithInputs(inputs map[string][]float32) CompareOption {
	return func(c *CompareConfig) {
		c.Inputs = inputs
	}
}

// WithCudaDevice specifies the physical CUDA device to execute on.
func WithCudaDevice(dev *cuda.Device) CompareOption {
	return func(c *CompareConfig) {
		c.Device = dev
	}
}

// WithFloatPolicy sets the float normalization policy for the accumulator.
func WithFloatPolicy(p FloatPolicy) CompareOption {
	return func(c *CompareConfig) {
		c.Policy = p
	}
}

// WithKernelSource supplies an explicit CUDA C++ kernel source and entry point.
func WithKernelSource(src, entryPoint string) CompareOption {
	return func(c *CompareConfig) {
		c.KernelSource = src
		c.EntryPoint = entryPoint
	}
}

// WithCompilerOptions sets NVRTC compilation options.
func WithCompilerOptions(opts ...nvrtcx.Option) CompareOption {
	return func(c *CompareConfig) {
		c.CompilerOpts = append(c.CompilerOpts, opts...)
	}
}

// WithJITOpts sets driver JIT options.
func WithJITOpts(jit cudakernel.JITOptions) CompareOption {
	return func(c *CompareConfig) {
		c.JITOptions = jit
	}
}

// WithTriage enables recording accumulator triage on hash mismatch.
func WithTriage(enable bool) CompareOption {
	return func(c *CompareConfig) {
		c.TriageOnMismatch = enable
	}
}

// WithVerifyCleanReset enforces checking that the device accumulator is at
// FNVOffsetBasis before launch.
func WithVerifyCleanReset(verify bool) CompareOption {
	return func(c *CompareConfig) {
		c.VerifyCleanReset = verify
	}
}

func withPerturbation(p Perturbation) CompareOption {
	return func(c *CompareConfig) { c.perturbation = &p }
}

func withCompileKernel(fn func(string, ...nvrtcx.Option) (*nvrtcx.CompiledModule, error)) CompareOption {
	return func(c *CompareConfig) { c.compileKernel = fn }
}

// EvaluateCPUGraph executes a computation graph on the CPU using cudaref and folds
// all intermediate tensor values in declared topological order into an Accumulator.
func EvaluateCPUGraph(g *cudair.Graph, inputs map[string][]float32, policy FloatPolicy, recording bool) (*Accumulator, error) {
	if g == nil {
		return nil, errors.New("cudalmvet: nil graph")
	}

	// Ensure all non-input nodes are in outputs to inspect all intermediate values
	origOutputs := g.Outputs()
	defer func() {
		_ = g.SetOutputs(origOutputs...)
	}()

	var intermediateNodes []*cudair.Node
	for _, n := range g.Nodes() {
		if n.Kind != cudair.KindInput {
			intermediateNodes = append(intermediateNodes, n)
		}
	}
	if len(intermediateNodes) > 0 {
		if err := g.SetOutputs(intermediateNodes...); err != nil {
			return nil, fmt.Errorf("cudalmvet: configure graph outputs: %w", err)
		}
	}

	snap, err := g.Snapshot()
	if err != nil {
		return nil, fmt.Errorf("cudalmvet: snapshot graph: %w", err)
	}

	results, err := cudaref.EvaluateSnapshot(snap, inputs)
	if err != nil {
		return nil, fmt.Errorf("cudalmvet: evaluate CPU reference: %w", err)
	}

	return foldGraphResults(snap, results, policy, recording), nil
}

func foldGraphResults(snap *cudair.Snapshot, results map[string][]float32, policy FloatPolicy, recording bool) *Accumulator {
	acc := NewAccumulator(WithPolicy(policy), WithRecording(recording))
	// Sequential execution along topological order
	for _, n := range snap.Order() {
		if n.Kind == cudair.KindInput {
			continue
		}
		buf, ok := results[n.Name]
		if !ok || len(buf) == 0 {
			continue // Unit/empty values pruned per contract
		}
		acc.FoldFloat32sNamed(n.Name, buf)
	}
	return acc
}

// CheckDeviceReset verifies that the loaded module's trace accumulator is clean
// (equal to FNVOffsetBasis). If not, it returns ErrContaminatedAccumulator.
func CheckDeviceReset(mod driver.Module) error {
	val, err := nvrtcx.ReadDeviceTrace(mod)
	if err != nil {
		return err
	}
	next, err := nvrtcx.ReadDeviceTraceNext(mod)
	if err != nil {
		return err
	}
	last, err := nvrtcx.ReadDeviceTraceLastBits(mod)
	if err != nil {
		return err
	}
	return validateAccumulatorState(val, next, last)
}

func validateAccumulatorState(hash uint64, next, last uint32) error {
	if hash != nvrtcx.TraceHashOffsetBasis || next != 0 || last != 0 {
		return fmt.Errorf("%w: hash=0x%016x next-ticket=%d last-bits=0x%08x", ErrContaminatedAccumulator, hash, next, last)
	}
	return nil
}

func deviceMismatchTriage(cpuHash, deviceHash uint64) *TriageReport {
	return &TriageReport{
		DiffIndex: -1,
		HashA:     cpuHash,
		HashB:     deviceHash,
		Message:   fmt.Sprintf("trace hash divergence: CPU reference produced 0x%016x, device produced 0x%016x; first differing fold is unknown", cpuHash, deviceHash),
	}
}

type deviceTrace struct {
	hash     uint64
	lastBits uint32
}

func localizeDeviceMismatch(events []TraceEvent, runPrefix func(int) (deviceTrace, error)) (*TriageReport, error) {
	if len(events) == 0 {
		return deviceMismatchTriage(0, 0), nil
	}
	low, high := 0, len(events)
	for low+1 < high {
		mid := low + (high-low)/2
		got, err := runPrefix(mid)
		if err != nil {
			return nil, err
		}
		if got.hash == events[mid-1].Hash {
			low = mid
		} else {
			high = mid
		}
	}
	got, err := runPrefix(high)
	if err != nil {
		return nil, err
	}
	event := events[high-1]
	value := math.Float32frombits(got.lastBits)
	return &TriageReport{
		DiffIndex: high - 1,
		NodeName:  event.NodeName,
		ValueA:    event.Value,
		ValueB:    float64(value),
		BitsA:     event.Bits,
		BitsB:     uint64(got.lastBits),
		HashA:     event.Hash,
		HashB:     got.hash,
		Message:   fmt.Sprintf("first divergence at fold #%d (node %q): CPU value %v (0x%08x), device value %v (0x%08x)", high-1, event.NodeName, event.Value, uint32(event.Bits), value, got.lastBits),
	}, nil
}

// Compare evaluates the CPU reference and GPU device executions for a computation
// graph, reporting agreement, disagreement with triage, or Unmeasured if device
// execution is unavailable.
func Compare(ctx context.Context, g *cudair.Graph, opts ...CompareOption) (*ComparisonResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("cudalmvet.Compare: %w", err)
	}
	cfg := &CompareConfig{
		Policy:           PolicyRawBits,
		TriageOnMismatch: true,
		VerifyCleanReset: true,
	}
	for _, opt := range opts {
		opt(cfg)
	}

	// 1. CPU Reference Evaluation
	accCPU, err := EvaluateCPUGraph(g, cfg.Inputs, cfg.Policy, cfg.TriageOnMismatch)
	if err != nil {
		return nil, fmt.Errorf("cudalmvet.Compare: CPU reference failed: %w", err)
	}
	cpuHash := accCPU.Hash()

	// 2. Check Device Availability
	if cfg.Device == nil {
		return &ComparisonResult{
			Status:  StatusUnmeasured,
			CPUHash: cpuHash,
		}, nil
	}

	// 3. Generate or use device kernel
	var kernelSrc string
	var entryPoint string
	generatedKernel := false

	if cfg.KernelSource != "" {
		kernelSrc = cfg.KernelSource
		entryPoint = cfg.EntryPoint
	} else {
		generatedKernel = true
		// Synthesize instrumented kernel from graph
		src, ep, err := synthesizeGraphKernel(g)
		if err != nil {
			return &ComparisonResult{
				Status:  StatusUnmeasured,
				Stage:   StageCompile,
				CPUHash: cpuHash,
				Err:     fmt.Errorf("synthesize kernel: %w", err),
			}, nil
		}
		kernelSrc = src
		entryPoint = ep
	}
	executionSnapshot, err := g.Snapshot()
	if err != nil {
		return &ComparisonResult{Status: StatusUnmeasured, Stage: StageCompile, CPUHash: cpuHash, Err: err}, nil
	}

	// 4. Compile with nvrtcx
	compileOpts := append([]nvrtcx.Option{
		nvrtcx.WithStandard("c++17"),
		nvrtcx.WithJITOptions(cfg.JITOptions),
		nvrtcx.WithNoCache(),
	}, cfg.CompilerOpts...)
	compileFlags, err := nvrtcx.CompileFlags(compileOpts...)
	if err != nil {
		return &ComparisonResult{Status: StatusUnmeasured, Stage: StageCompile, CPUHash: cpuHash, Err: err}, nil
	}

	compile := cfg.compileKernel
	if compile == nil {
		compile = nvrtcx.Compile
	}
	cmod, err := compile(kernelSrc, compileOpts...)
	if err != nil {
		return &ComparisonResult{
			Status:  StatusUnmeasured,
			Stage:   StageCompile,
			CPUHash: cpuHash,
			Err:     fmt.Errorf("nvrtcx compile: %w", err),
		}, nil
	}
	if cfg.perturbation != nil {
		if err := validateCompiledPerturbation(*cfg.perturbation, compileFlags, cmod); err != nil {
			return &ComparisonResult{Status: StatusUnmeasured, Stage: StageCompile, CPUHash: cpuHash, Err: err}, nil
		}
	}

	// 5. Load and execute on device
	devMod, err := cmod.Load(cfg.Device)
	if err != nil {
		return &ComparisonResult{
			Status:  StatusUnmeasured,
			Stage:   StageLoad,
			CPUHash: cpuHash,
			Err:     fmt.Errorf("load module: %w", err),
		}, nil
	}
	defer cfg.Device.WithContext(func() error { return devMod.Close() })

	// Optional check for clean reset
	if cfg.VerifyCleanReset {
		var resetErr error
		if err := cfg.Device.WithContext(func() error { return CheckDeviceReset(devMod) }); err != nil {
			resetErr = err
		}
		if resetErr != nil {
			return &ComparisonResult{Status: StatusUnmeasured, Stage: StageLaunch, CPUHash: cpuHash, Err: resetErr}, nil
		}
	}

	trace, err := executeDeviceKernel(ctx, devMod, entryPoint, cfg, executionSnapshot, ^uint32(0), false, generatedKernel)
	if err != nil {
		return &ComparisonResult{
			Status:  StatusUnmeasured,
			Stage:   failureStage(err, StageLaunch),
			CPUHash: cpuHash,
			Err:     fmt.Errorf("execute device kernel: %w", err),
		}, nil
	}
	devHash := trace.hash

	if devHash == cpuHash {
		return &ComparisonResult{
			Status:     StatusAgreed,
			CPUHash:    cpuHash,
			DeviceHash: devHash,
		}, nil
	}

	// Mismatch triage
	res := &ComparisonResult{
		Status:     StatusDisagreed,
		CPUHash:    cpuHash,
		DeviceHash: devHash,
	}
	if cfg.TriageOnMismatch {
		res.Triage = deviceMismatchTriage(cpuHash, devHash)
		if generatedKernel && len(accCPU.Events()) != 0 {
			localized, err := localizeDeviceMismatch(accCPU.Events(), func(prefix int) (deviceTrace, error) {
				return executeDeviceKernel(ctx, devMod, entryPoint, cfg, executionSnapshot, uint32(prefix), true, true)
			})
			if err == nil {
				res.Triage = localized
			} else {
				res.Triage.Message += fmt.Sprintf("; device fold localization failed: %v", err)
			}
		}
	}
	return res, nil
}

// synthesizeGraphKernel creates a basic elementwise CUDA C++ kernel for a graph.
func synthesizeGraphKernel(g *cudair.Graph) (src string, entryPoint string, err error) {
	snap, err := g.Snapshot()
	if err != nil {
		return "", "", err
	}

	var sb strings.Builder
	sb.WriteString(nvrtcx.TraceHeader())
	sb.WriteString("\nextern \"C\" __global__ void eval_graph_kernel(int num_elements, unsigned int trace_fold_limit")

	// Parameter order is shared with the host argument binder.
	for _, arg := range kernelArgumentLayout(snap) {
		if arg.Output {
			sb.WriteString(fmt.Sprintf(", float *__restrict__ out_%s", arg.Name))
		} else {
			sb.WriteString(fmt.Sprintf(", const float *__restrict__ in_%s", arg.Name))
		}
	}
	outputs := snap.Outputs()
	sb.WriteString(") {\n")
	sb.WriteString("    int idx = threadIdx.x;\n")
	sb.WriteString("    if (idx >= num_elements) return;\n\n")

	// Emit sequentially in topological order
	foldPlan := buildTraceFoldPlan(snap)
	ticketOffsets := make(map[string]int)
	for _, event := range foldPlan {
		if _, exists := ticketOffsets[event.Node]; !exists {
			ticketOffsets[event.Node] = event.Ticket
		}
	}
	for _, n := range snap.Order() {
		if n.Kind == cudair.KindInput {
			continue
		}
		var expr string
		switch n.Kind {
		case cudair.KindConstant:
			if len(n.ConstValues) > 1 {
				name := "constant_" + n.Name
				sb.WriteString("    const float " + name + "[] = {")
				for i, value := range n.ConstValues {
					if i != 0 {
						sb.WriteString(", ")
					}
					sb.WriteString(float32Literal(value))
				}
				sb.WriteString("};\n")
				expr = fmt.Sprintf("%s[idx %% %d]", name, len(n.ConstValues))
			} else if len(n.ConstValues) == 1 {
				expr = float32Literal(n.ConstValues[0])
			} else {
				expr = "0.0f"
			}
		case cudair.KindUnary:
			inName := n.Inputs[0].Name
			var valRef string
			if n.Inputs[0].Kind == cudair.KindInput {
				valRef = fmt.Sprintf("in_%s[idx]", inName)
			} else {
				valRef = fmt.Sprintf("var_%s", inName)
			}
			switch n.Op {
			case "relu":
				expr = fmt.Sprintf("(%s > 0.0f ? %s : 0.0f)", valRef, valRef)
			case "neg":
				expr = fmt.Sprintf("(-%s)", valRef)
			case "exp":
				expr = fmt.Sprintf("expf(%s)", valRef)
			case "log":
				expr = fmt.Sprintf("logf(%s)", valRef)
			case "sqrt":
				expr = fmt.Sprintf("sqrtf(%s)", valRef)
			case "rsqrt":
				expr = fmt.Sprintf("rsqrtf(%s)", valRef)
			default:
				return "", "", fmt.Errorf("synthesize: unsupported unary op %q", n.Op)
			}
		case cudair.KindBinary:
			aName := n.Inputs[0].Name
			bName := n.Inputs[1].Name
			var aRef, bRef string
			if n.Inputs[0].Kind == cudair.KindInput {
				aRef = fmt.Sprintf("in_%s[idx]", aName)
			} else {
				aRef = fmt.Sprintf("var_%s", aName)
			}
			if n.Inputs[1].Kind == cudair.KindInput {
				bRef = fmt.Sprintf("in_%s[idx]", bName)
			} else {
				bRef = fmt.Sprintf("var_%s", bName)
			}

			switch n.Op {
			case "add":
				expr = fmt.Sprintf("(%s + %s)", aRef, bRef)
			case "sub":
				expr = fmt.Sprintf("(%s - %s)", aRef, bRef)
			case "mul":
				expr = fmt.Sprintf("(%s * %s)", aRef, bRef)
			case "div":
				expr = fmt.Sprintf("(%s / %s)", aRef, bRef)
			default:
				return "", "", fmt.Errorf("synthesize: unsupported binary op %q", n.Op)
			}
		default:
			return "", "", fmt.Errorf("synthesize: unsupported op kind %v", n.Kind)
		}

		sb.WriteString(fmt.Sprintf("    float var_%s = %s;\n", n.Name, expr))
		count := n.NumElements()
		if count > 0 {
			ticket := fmt.Sprintf("%dU + (unsigned int)idx", ticketOffsets[n.Name])
			sb.WriteString(fmt.Sprintf("    if (idx < %d) { %s", count, nvrtcx.EmitOrderedDumpVar(fmt.Sprintf("var_%s", n.Name), ticket, "trace_fold_limit")))
			sb.WriteString("    }\n")
		}
	}

	// Write outputs
	for _, out := range outputs {
		sb.WriteString(fmt.Sprintf("    out_%s[idx] = var_%s;\n", out.Name, out.Name))
	}
	sb.WriteString("}\n")

	return sb.String(), "eval_graph_kernel", nil
}

func float32Literal(v float32) string {
	return strconv.FormatFloat(float64(v), 'g', 9, 32) + "f"
}

type foldTicket struct {
	Node    string
	Element int
	Ticket  int
}

func buildTraceFoldPlan(snap *cudair.Snapshot) []foldTicket {
	var plan []foldTicket
	ticket := 0
	for _, n := range snap.Order() {
		if n.Kind == cudair.KindInput {
			continue
		}
		for i := 0; i < n.NumElements(); i++ {
			plan = append(plan, foldTicket{Node: n.Name, Element: i, Ticket: ticket})
			ticket++
		}
	}
	return plan
}

type kernelArgument struct {
	Name   string
	Output bool
}

func kernelArgumentLayout(snap *cudair.Snapshot) []kernelArgument {
	var args []kernelArgument
	for _, n := range snap.Inputs() {
		args = append(args, kernelArgument{Name: n.Name})
	}
	for _, n := range snap.Outputs() {
		args = append(args, kernelArgument{Name: n.Name, Output: true})
	}
	return args
}

// executeDeviceKernel allocates buffers, launches the kernel, and reads back the trace hash.
func executeDeviceKernel(ctx context.Context, mod driver.Module, entryPoint string, cfg *CompareConfig, snap *cudair.Snapshot, foldLimit uint32, reset, generated bool) (deviceTrace, error) {
	if err := ctx.Err(); err != nil {
		return deviceTrace{}, err
	}
	var trace deviceTrace
	err := cfg.Device.WithContext(func() error {
		var err error
		trace, err = executeDeviceKernelCurrent(ctx, mod, entryPoint, cfg, snap, foldLimit, reset, generated)
		return err
	})
	return trace, err
}

func executeDeviceKernelCurrent(ctx context.Context, mod driver.Module, entryPoint string, cfg *CompareConfig, snap *cudair.Snapshot, foldLimit uint32, reset, generated bool) (deviceTrace, error) {
	fn, err := mod.Function(entryPoint)
	if err != nil {
		return deviceTrace{}, fmt.Errorf("get function %q: %w", entryPoint, err)
	}

	numElems := 0
	for _, arg := range kernelArgumentLayout(snap) {
		if arg.Output {
			continue
		}
		buf := cfg.Inputs[arg.Name]
		if len(buf) > numElems {
			numElems = len(buf)
		}
	}
	for _, node := range snap.Nodes() {
		if node.NumElements() > numElems {
			numElems = node.NumElements()
		}
	}
	if numElems == 0 {
		numElems = 1
	}
	if numElems > 1024 {
		return deviceTrace{}, fmt.Errorf("cudalmvet: generated ordered kernel supports at most 1024 elements, got %d", numElems)
	}

	// Allocate input device buffers
	var inDevPtrs []driver.DevicePtr
	defer func() {
		for _, ptr := range inDevPtrs {
			_ = ptr.Free()
		}
	}()

	var args []unsafe.Pointer
	nVal := int32(numElems)
	args = append(args, unsafe.Pointer(&nVal))
	if generated {
		args = append(args, unsafe.Pointer(&foldLimit))
	}

	for _, arg := range kernelArgumentLayout(snap) {
		if arg.Output {
			continue
		}
		buf, ok := cfg.Inputs[arg.Name]
		if !ok {
			return deviceTrace{}, fmt.Errorf("missing input %q", arg.Name)
		}
		byteCount := uint(len(buf) * 4)
		ptr, err := driver.MemAllocBytes(byteCount)
		if err != nil {
			return deviceTrace{}, fmt.Errorf("alloc input: %w", err)
		}
		inDevPtrs = append(inDevPtrs, ptr)
		if byteCount != 0 {
			if err := ptr.CopyFromHostPtr(unsafe.Pointer(unsafe.SliceData(buf)), byteCount); err != nil {
				return deviceTrace{}, fmt.Errorf("copy input: %w", err)
			}
		}
		args = append(args, unsafe.Pointer(&ptr))
	}

	// The generated signature has one output buffer per declared graph output.
	outPtrs := make([]driver.DevicePtr, len(snap.Outputs()))
	defer func() {
		for _, ptr := range outPtrs {
			_ = ptr.Free()
		}
	}()
	outputIndex := 0
	for _, arg := range kernelArgumentLayout(snap) {
		if !arg.Output {
			continue
		}
		output := snap.Outputs()[outputIndex]
		outputIndex++
		bytes := uint(output.NumElements() * 4)
		if bytes == 0 {
			bytes = 4
		}
		ptr, err := driver.MemAllocBytes(bytes)
		if err != nil {
			return deviceTrace{}, fmt.Errorf("alloc output %q: %w", output.Name, err)
		}
		outPtrs[outputIndex-1] = ptr
		args = append(args, unsafe.Pointer(&outPtrs[outputIndex-1]))
	}

	if reset {
		if err := nvrtcx.ResetDeviceTrace(mod); err != nil {
			return deviceTrace{}, fmt.Errorf("reset trace for triage: %w", err)
		}
	}
	if err := ctx.Err(); err != nil {
		return deviceTrace{}, err
	}
	blockSize := uint32(1024)
	gridSize := uint32(1)

	gridDim := driver.Dim3{X: gridSize, Y: 1, Z: 1}
	blockDim := driver.Dim3{X: blockSize, Y: 1, Z: 1}

	if err := fn.LaunchKernel(gridDim, blockDim, 0, 0, args); err != nil {
		return deviceTrace{}, atStage(StageLaunch, fmt.Errorf("launch kernel %q: %w", entryPoint, err))
	}

	// Synchronize device
	if err := driver.CtxSynchronize(); err != nil {
		return deviceTrace{}, atStage(StageLaunch, fmt.Errorf("synchronize: %w", err))
	}

	// Readback trace
	var trace deviceTrace
	trace.hash, err = nvrtcx.ReadDeviceTrace(mod)
	if err != nil {
		return deviceTrace{}, atStage(StageReadback, err)
	}
	trace.lastBits, err = nvrtcx.ReadDeviceTraceLastBits(mod)
	if err != nil {
		return deviceTrace{}, atStage(StageReadback, err)
	}
	return trace, nil
}
