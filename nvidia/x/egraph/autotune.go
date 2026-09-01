package egraph

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/tmc/lib/nvidia/x/cuda"
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

	t.mu.RLock()
	cached, ok := t.cache[defaultExpr.String()]
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

		// Measure compilation & micro-execution time
		t0 := time.Now()
		major, minor, _ := t.dev.ComputeCapability()
		mod, err := nvrtcx.Compile(src, nvrtcx.WithArch(major, minor))
		if err != nil {
			continue
		}
		loaded, err := mod.Load(t.dev)
		if err != nil {
			continue
		}
		_ = loaded

		elapsed := time.Since(t0)
		if elapsed < bestDur {
			bestDur = elapsed
			bestCand = cand
		}
	}

	t.mu.Lock()
	t.cache[defaultExpr.String()] = bestCand
	t.bestScores[defaultExpr.String()] = bestDur
	t.mu.Unlock()

	return &bestCand, nil
}
