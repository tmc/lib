package egraph

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// AutotuneBudget is a search deadline checked between candidates.
// Compiling or benchmarking a candidate can exceed the remaining budget.
type AutotuneBudget time.Duration

const (
	// BudgetInstant skips background candidate micro-benchmarking (instant startup).
	BudgetInstant AutotuneBudget = 0

	// BudgetQuick requests a 50 ms search budget.
	BudgetQuick = AutotuneBudget(50 * time.Millisecond)

	// BudgetThorough requests a one-second search budget.
	BudgetThorough = AutotuneBudget(1 * time.Second)
)

// TuningCandidate represents an AST expression paired with candidate tile/launch parameters.
type TuningCandidate struct {
	Expr       *Expr
	BlockDim   int
	GridDim    int
	SharedMem  int
	KernelName string
	CUDASource string
}

// Key returns a unique composite identifier for the candidate expression and launch configuration.
func (c TuningCandidate) Key() string {
	exprStr := ""
	if c.Expr != nil {
		exprStr = c.Expr.String()
	}
	return fmt.Sprintf("%s::b%d::g%d", exprStr, c.BlockDim, c.GridDim)
}

// AutotuneResult stores the benchmarked latency of a candidate kernel.
type AutotuneResult struct {
	Candidate TuningCandidate
	Latency   time.Duration
	TFLOPS    float64
}

// Tuner evaluates candidate AST expressions against real GPU hardware within a time budget.
type Tuner struct {
	mu         sync.RWMutex
	dev        *cuda.Device
	cache      map[string]TuningCandidate
	bestScores map[string]time.Duration
}

// NewTuner creates a new hardware-in-the-loop kernel tuner.
func NewTuner(dev *cuda.Device) *Tuner {
	return &Tuner{
		dev:        dev,
		cache:      make(map[string]TuningCandidate),
		bestScores: make(map[string]time.Duration),
	}
}

// benchmarkSizes are the problem dimensions the autotuner measures at. They are
// small enough to keep a tuning sweep quick and large enough that the block
// size being swept actually changes the work per thread.
const (
	benchElements = 1024 // elementwise length
	benchInDim    = 256  // rmsnorm input width
	benchOutDim   = 128  // rmsnorm output rows
	benchHidden   = 256  // swiglu hidden width
	benchInter    = 512  // swiglu intermediate width
)

// launchPlan describes how to call one generated kernel: the float32 buffers to
// allocate, in parameter order, then the trailing scalar arguments.
//
// The autotuner used to launch every candidate with two inputs, an output and
// an element count, which is the add kernel's signature. Both fused kernels
// take seven parameters, so those launches were rejected outright and every
// candidate for them was discarded -- autotuning silently did nothing for the
// only two non-trivial operators this package generates.
type launchPlan struct {
	buffers []int // element count of each buffer parameter, in order
	scalars []any // remaining arguments, in order
	shared  uint32
	grid    func(block uint32) uint32
}

func benchmarkPlan(cand TuningCandidate) (launchPlan, error) {
	op := ""
	if cand.Expr != nil {
		op = cand.Expr.Op
	}
	switch op {
	case "add":
		return launchPlan{
			buffers: []int{benchElements, benchElements, benchElements},
			scalars: []any{int32(benchElements)},
			grid: func(block uint32) uint32 {
				return (benchElements + block - 1) / block
			},
		}, nil
	case "fused_rmsnorm_gemv":
		// x, weight, W_proj, out, in_dim, out_dim, eps; one block per row.
		return launchPlan{
			buffers: []int{benchInDim, benchInDim, benchOutDim * benchInDim, benchOutDim},
			scalars: []any{int32(benchInDim), int32(benchOutDim), float32(1e-5)},
			grid:    func(uint32) uint32 { return benchOutDim },
		}, nil
	case "fused_swiglu_gemv":
		// x, W_gate, W_up, W_down, out, hidden_dim, intermediate_dim. Every
		// block fills the whole activation buffer, so one block is the only
		// shape the kernel's shared-memory indexing supports.
		return launchPlan{
			buffers: []int{benchHidden, benchInter * benchHidden, benchInter * benchHidden, benchHidden * benchInter, benchHidden},
			scalars: []any{int32(benchHidden), int32(benchInter)},
			shared:  benchInter * 4,
			grid:    func(uint32) uint32 { return 1 },
		}, nil
	case "fused_swiglu_q8_0":
		return launchPlan{
			buffers: []int{benchHidden * 4, benchInter * benchHidden, benchInter * benchHidden, benchHidden * benchInter, benchHidden * 4},
			scalars: []any{int32(benchHidden), int32(benchInter)},
			shared:  benchInter * 4,
			grid:    func(uint32) uint32 { return 1 },
		}, nil
	case "q8_0_matvec":
		return launchPlan{
			buffers: []int{benchOutDim * benchInDim, benchInDim * 4, benchOutDim * 4},
			scalars: []any{int32(benchOutDim), int32(benchInDim)},
			grid:    func(block uint32) uint32 { return (benchOutDim*32 + block - 1) / block },
		}, nil
	case "u2_simd", "u2_scalar":
		return launchPlan{
			buffers: []int{benchElements * 4, benchElements * 4, benchElements * 4},
			scalars: []any{float32(1.0), int32(benchElements)},
			grid:    func(block uint32) uint32 { return (benchElements + block - 1) / block },
		}, nil
	case "q2_k_matvec":
		return launchPlan{
			buffers: []int{benchOutDim * (benchInDim / 256) * 84, benchInDim * 4, benchOutDim * 4},
			scalars: []any{int32(benchOutDim), int32(benchInDim)},
			grid:    func(block uint32) uint32 { return (benchOutDim*32 + block - 1) / block },
		}, nil
	case "fused_rope_store_kv":
		return launchPlan{
			buffers: []int{benchHidden * 4, benchHidden * 4, benchHidden * 4, benchHidden * 4, benchHidden * 4, benchHidden * 4},
			scalars: []any{int32(0), float32(10000.0), int32(benchHidden), int32(benchHidden), int32(128), int32(2048)},
			grid:    func(block uint32) uint32 { return (benchHidden + block - 1) / block },
		}, nil
	}
	return launchPlan{}, fmt.Errorf("egraph: no benchmark launch plan for operator %q", op)
}

// BenchmarkCandidate compiles, loads, warms up, and benchmarks the candidate on the GPU,
// returning average host wall time per iteration across launches and stream
// synchronization. Compilation, loading and warmup precede the measured interval.
func BenchmarkCandidate(dev *cuda.Device, cand TuningCandidate, iters int) (time.Duration, error) {
	if dev == nil {
		return 0, fmt.Errorf("egraph: nil CUDA device for benchmark")
	}
	if iters <= 0 {
		iters = 5
	}

	major, minor, err := dev.ComputeCapability()
	if err != nil {
		return 0, fmt.Errorf("egraph: compute capability: %w", err)
	}

	mod, err := nvrtcx.Compile(cand.CUDASource, nvrtcx.WithArch(major, minor))
	if err != nil {
		return 0, fmt.Errorf("egraph: compile candidate: %w", err)
	}

	loadedMod, err := mod.Load(dev)
	if err != nil {
		return 0, fmt.Errorf("egraph: load module: %w", err)
	}
	defer loadedMod.Unload()

	fn, err := loadedMod.GetFunction(cand.KernelName)
	if err != nil {
		return 0, fmt.Errorf("egraph: get kernel function %q: %w", cand.KernelName, err)
	}

	stream, err := dev.NewStream()
	if err != nil {
		return 0, fmt.Errorf("egraph: create stream: %w", err)
	}
	defer stream.Destroy()

	plan, err := benchmarkPlan(cand)
	if err != nil {
		return 0, err
	}
	bufs := make([]cudamem.Slice[float32], len(plan.buffers))
	for i, n := range plan.buffers {
		b, err := cudamem.Alloc[float32](dev, n)
		if err != nil {
			for _, done := range bufs[:i] {
				_ = done.Free()
			}
			return 0, fmt.Errorf("egraph: alloc benchmark buffer %d: %w", i, err)
		}
		bufs[i] = b
	}
	defer func() {
		for _, b := range bufs {
			_ = b.Free()
		}
	}()

	args := make([]any, 0, len(bufs)+len(plan.scalars))
	for _, b := range bufs {
		args = append(args, b)
	}
	args = append(args, plan.scalars...)

	block := uint32(cand.BlockDim)
	if block == 0 {
		block = 128
	}
	grid := uint32(cand.GridDim)
	if grid == 0 {
		grid = plan.grid(block)
	}
	shared := uint32(cand.SharedMem)
	if shared == 0 {
		shared = plan.shared
	}

	launcher := cudalaunch.Kernel(fn).Grid(grid, 1).Block(block).SharedMem(shared).On(stream)

	// 1. Warmup Execution
	if err := launcher.Launch(args...); err != nil {
		return 0, fmt.Errorf("egraph: warmup launch: %w", err)
	}
	if err := stream.Synchronize(context.Background()); err != nil {
		return 0, fmt.Errorf("egraph: warmup sync: %w", err)
	}

	// 2. Host elapsed time for launches and synchronization.
	t0 := time.Now()
	for i := 0; i < iters; i++ {
		if err := launcher.Launch(args...); err != nil {
			return 0, fmt.Errorf("egraph: benchmark launch: %w", err)
		}
	}
	if err := stream.Synchronize(context.Background()); err != nil {
		return 0, fmt.Errorf("egraph: benchmark sync: %w", err)
	}
	elapsed := time.Since(t0) / time.Duration(iters)

	return elapsed, nil
}

// SelectBest extracts the instant default plan and, if budget > 0, benchmarks candidate
// tile/kernel configurations on the device to pick the fastest kernel.
func (t *Tuner) SelectBest(ctx context.Context, g *EGraph, root Id, budget AutotuneBudget) (*TuningCandidate, error) {
	// 1. Instant extraction (zero delay)
	defaultExpr, _, err := ExtractBest(g, root, FusedMegakernelCost{})
	if err != nil {
		return nil, fmt.Errorf("egraph: extract default: %w", err)
	}

	defaultCand := TuningCandidate{
		Expr:       defaultExpr,
		BlockDim:   128,
		KernelName: "Kernel_Auto_Default",
	}

	if budget == BudgetInstant || t.dev == nil {
		return &defaultCand, nil
	}

	cacheKey := fmt.Sprintf("%s::budget=%v", defaultExpr.String(), budget)
	t.mu.RLock()
	cached, ok := t.cache[cacheKey]
	t.mu.RUnlock()
	if ok {
		return &cached, nil
	}

	// 2. Multi-Candidate Tile Space Exploration within Budget
	tileSizes := []int{64, 128, 256, 512}
	deadline := time.Now().Add(time.Duration(budget))

	bestCand := defaultCand
	bestDur := time.Duration(1<<63 - 1)
	// Why every candidate was discarded, if every candidate was. Returning
	// the unbenchmarked default and reporting success hid a total failure to
	// tune behind a candidate that carried no source at all.
	var lastErr error

	for _, blockSize := range tileSizes {
		if time.Now().After(deadline) {
			break
		}

		cand := TuningCandidate{
			Expr:       defaultExpr,
			BlockDim:   blockSize,
			KernelName: fmt.Sprintf("Kernel_Auto_Block%d", blockSize),
		}

		src, err := GenerateCUDA(defaultExpr, CodegenOptions{
			KernelName: cand.KernelName,
			Precision:  "float",
		})
		if err != nil {
			lastErr = err
			continue
		}
		cand.CUDASource = src

		// Measure real GPU execution latency on hardware
		execDur, err := BenchmarkCandidate(t.dev, cand, 5)
		if err != nil {
			lastErr = err
			continue
		}

		if execDur < bestDur {
			bestDur = execDur
			bestCand = cand
		}
	}

	if lastErr != nil && bestCand.CUDASource == "" {
		return nil, fmt.Errorf("egraph: no candidate could be benchmarked for %s: %w", defaultExpr.String(), lastErr)
	}

	t.mu.Lock()
	t.cache[cacheKey] = bestCand
	t.bestScores[cacheKey] = bestDur
	t.mu.Unlock()

	return &bestCand, nil
}
