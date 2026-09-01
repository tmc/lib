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

// AutotuneBudget controls the tradeoff between instant startup and deep kernel search.
type AutotuneBudget time.Duration

const (
	// BudgetInstant skips background candidate micro-benchmarking (instant startup).
	BudgetInstant AutotuneBudget = 0

	// BudgetQuick searches up to ~50ms of candidate tile configurations.
	BudgetQuick = AutotuneBudget(50 * time.Millisecond)

	// BudgetThorough searches up to ~1s of candidate kernels for maximum sustained decode throughput.
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

// BenchmarkCandidate compiles, loads, warms up, and benchmarks the candidate on the GPU,
// measuring pure execution duration (excluding compilation/load time).
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

	const testElements = 1024
	inA, err := cudamem.Alloc[float32](dev, testElements)
	if err != nil {
		return 0, fmt.Errorf("egraph: alloc input A: %w", err)
	}
	defer inA.Free()

	inB, err := cudamem.Alloc[float32](dev, testElements)
	if err != nil {
		return 0, fmt.Errorf("egraph: alloc input B: %w", err)
	}
	defer inB.Free()

	outBuf, err := cudamem.Alloc[float32](dev, testElements)
	if err != nil {
		return 0, fmt.Errorf("egraph: alloc output buffer: %w", err)
	}
	defer outBuf.Free()

	grid := uint32(cand.GridDim)
	if grid == 0 {
		grid = (testElements + uint32(cand.BlockDim) - 1) / uint32(cand.BlockDim)
		if grid == 0 {
			grid = 1
		}
	}
	block := uint32(cand.BlockDim)
	if block == 0 {
		block = 128
	}

	launcher := cudalaunch.Kernel(fn).Grid(grid).Block(block).SharedMem(uint32(cand.SharedMem)).On(stream)

	// 1. Warmup Execution
	if err := launcher.Launch(inA.Ptr(), inB.Ptr(), outBuf.Ptr(), uint32(testElements)); err != nil {
		return 0, fmt.Errorf("egraph: warmup launch: %w", err)
	}
	if err := stream.Synchronize(context.Background()); err != nil {
		return 0, fmt.Errorf("egraph: warmup sync: %w", err)
	}

	// 2. Timed GPU Microbenchmark (pure execution time)
	t0 := time.Now()
	for i := 0; i < iters; i++ {
		if err := launcher.Launch(inA.Ptr(), inB.Ptr(), outBuf.Ptr(), uint32(testElements)); err != nil {
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
			continue
		}
		cand.CUDASource = src

		// Measure real GPU execution latency on hardware
		execDur, err := BenchmarkCandidate(t.dev, cand, 5)
		if err != nil {
			continue
		}

		if execDur < bestDur {
			bestDur = execDur
			bestCand = cand
		}
	}

	t.mu.Lock()
	t.cache[cacheKey] = bestCand
	t.bestScores[cacheKey] = bestDur
	t.mu.Unlock()

	return &bestCand, nil
}
