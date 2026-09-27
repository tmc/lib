package cudalmvet

import (
	"context"
	"errors"
	"fmt"
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
	CPUHash    uint64
	DeviceHash uint64
	Triage     *TriageReport
	Err        error
}

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

// EvaluateCPUGraph executes a computation graph on the CPU using cudaref and folds
// all intermediate tensor values in declared topological order into an Accumulator.
func EvaluateCPUGraph(g *cudair.Graph, inputs map[string][]float32, policy FloatPolicy, recording bool) (*Accumulator, error) {
	if g == nil {
		return nil, errors.New("cudalmvet: nil graph")
	}

	acc := NewAccumulator(WithPolicy(policy), WithRecording(recording))

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

	return acc, nil
}

// CheckDeviceReset verifies that the loaded module's trace accumulator is clean
// (equal to FNVOffsetBasis). If not, it returns ErrContaminatedAccumulator.
func CheckDeviceReset(mod driver.Module) error {
	val, err := nvrtcx.ReadDeviceTrace(mod)
	if err != nil {
		return err
	}
	if val != nvrtcx.TraceHashOffsetBasis {
		return fmt.Errorf("%w: trace accumulator is 0x%016x, expected offset basis 0x%016x",
			ErrContaminatedAccumulator, val, nvrtcx.TraceHashOffsetBasis)
	}
	return nil
}

// Compare evaluates the CPU reference and GPU device executions for a computation
// graph, reporting agreement, disagreement with triage, or Unmeasured if device
// execution is unavailable.
func Compare(ctx context.Context, g *cudair.Graph, opts ...CompareOption) (*ComparisonResult, error) {
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

	if cfg.KernelSource != "" {
		kernelSrc = cfg.KernelSource
		entryPoint = cfg.EntryPoint
	} else {
		// Synthesize instrumented kernel from graph
		src, ep, err := synthesizeGraphKernel(g)
		if err != nil {
			return &ComparisonResult{
				Status:  StatusUnmeasured,
				CPUHash: cpuHash,
				Err:     fmt.Errorf("synthesize kernel: %w", err),
			}, nil
		}
		kernelSrc = src
		entryPoint = ep
	}

	// 4. Compile with nvrtcx
	compileOpts := append([]nvrtcx.Option{
		nvrtcx.WithStandard("c++17"),
		nvrtcx.WithJITOptions(cfg.JITOptions),
	}, cfg.CompilerOpts...)

	cmod, err := nvrtcx.Compile(kernelSrc, compileOpts...)
	if err != nil {
		return &ComparisonResult{
			Status:  StatusUnmeasured,
			CPUHash: cpuHash,
			Err:     fmt.Errorf("nvrtcx compile: %w", err),
		}, nil
	}

	// 5. Load and execute on device
	devMod, err := cmod.Load(cfg.Device)
	if err != nil {
		return &ComparisonResult{
			Status:  StatusUnmeasured,
			CPUHash: cpuHash,
			Err:     fmt.Errorf("load module: %w", err),
		}, nil
	}

	// Optional check for clean reset
	if cfg.VerifyCleanReset {
		if err := CheckDeviceReset(devMod); err != nil {
			// Try resetting it
			if rerr := nvrtcx.ResetDeviceTrace(devMod); rerr != nil {
				return nil, fmt.Errorf("cudalmvet: reset device trace: %w", rerr)
			}
		}
	}

	devHash, err := executeDeviceKernel(ctx, devMod, entryPoint, cfg)
	if err != nil {
		return &ComparisonResult{
			Status:  StatusUnmeasured,
			CPUHash: cpuHash,
			Err:     fmt.Errorf("execute device kernel: %w", err),
		}, nil
	}

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
		res.Triage = &TriageReport{
			HashA:   cpuHash,
			HashB:   devHash,
			Message: fmt.Sprintf("trace hash divergence: CPU reference produced 0x%016x, device produced 0x%016x", cpuHash, devHash),
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
	sb.WriteString("\nextern \"C\" __global__ void eval_graph_kernel(int num_elements")

	// Input parameters
	inputs := snap.Inputs()
	for _, in := range inputs {
		sb.WriteString(fmt.Sprintf(", const float *__restrict__ in_%s", in.Name))
	}
	// Output parameters
	outputs := snap.Outputs()
	for _, out := range outputs {
		sb.WriteString(fmt.Sprintf(", float *__restrict__ out_%s", out.Name))
	}
	sb.WriteString(") {\n")
	sb.WriteString("    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n")
	sb.WriteString("    if (idx >= num_elements) return;\n\n")

	// Emit sequentially in topological order
	for _, n := range snap.Order() {
		if n.Kind == cudair.KindInput {
			continue
		}
		var expr string
		switch n.Kind {
		case cudair.KindConstant:
			if len(n.ConstValues) > 0 {
				expr = fmt.Sprintf("%ff", n.ConstValues[0])
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
		sb.WriteString(fmt.Sprintf("    %s", nvrtcx.EmitDumpVar(fmt.Sprintf("var_%s", n.Name))))
	}

	// Write outputs
	for _, out := range outputs {
		sb.WriteString(fmt.Sprintf("    out_%s[idx] = var_%s;\n", out.Name, out.Name))
	}
	sb.WriteString("}\n")

	return sb.String(), "eval_graph_kernel", nil
}

// executeDeviceKernel allocates buffers, launches the kernel, and reads back the trace hash.
func executeDeviceKernel(ctx context.Context, mod driver.Module, entryPoint string, cfg *CompareConfig) (uint64, error) {
	fn, err := mod.Function(entryPoint)
	if err != nil {
		return 0, fmt.Errorf("get function %q: %w", entryPoint, err)
	}

	numElems := 0
	for _, buf := range cfg.Inputs {
		if len(buf) > numElems {
			numElems = len(buf)
		}
	}
	if numElems == 0 {
		numElems = 1
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

	for _, buf := range cfg.Inputs {
		byteCount := uint(len(buf) * 4)
		ptr, err := driver.MemAllocBytes(byteCount)
		if err != nil {
			return 0, fmt.Errorf("alloc input: %w", err)
		}
		inDevPtrs = append(inDevPtrs, ptr)
		if err := ptr.CopyFromHostPtr(unsafe.Pointer(unsafe.SliceData(buf)), byteCount); err != nil {
			return 0, fmt.Errorf("copy input: %w", err)
		}
		args = append(args, unsafe.Pointer(&ptr))
	}

	// Output buffer
	outBytes := uint(numElems * 4)
	outPtr, err := driver.MemAllocBytes(outBytes)
	if err != nil {
		return 0, fmt.Errorf("alloc output: %w", err)
	}
	defer outPtr.Free()
	args = append(args, unsafe.Pointer(&outPtr))

	// Reset device trace before launch
	if err := nvrtcx.ResetDeviceTrace(mod); err != nil {
		return 0, fmt.Errorf("reset trace: %w", err)
	}

	blockSize := uint32(256)
	gridSize := uint32((numElems + int(blockSize) - 1) / int(blockSize))
	if gridSize == 0 {
		gridSize = 1
	}

	gridDim := driver.Dim3{X: gridSize, Y: 1, Z: 1}
	blockDim := driver.Dim3{X: blockSize, Y: 1, Z: 1}

	if err := fn.LaunchKernel(gridDim, blockDim, 0, 0, args); err != nil {
		return 0, fmt.Errorf("launch kernel %q: %w", entryPoint, err)
	}

	// Synchronize device
	if err := driver.CtxSynchronize(); err != nil {
		return 0, fmt.Errorf("synchronize: %w", err)
	}

	// Readback trace
	return nvrtcx.ReadDeviceTrace(mod)
}
