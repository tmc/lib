package egraph

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"strings"
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

var autotuneBlockSizes = [...]int{64, 128, 256, 512}

// TuningCandidate represents an AST expression paired with candidate tile/launch parameters.
type TuningCandidate struct {
	Expr       *Expr
	BlockDim   int
	GridDim    int
	SharedMem  int
	KernelName string
	CUDASource string
}

// CandidateEvaluator measures and validates one generated CUDA candidate.
// It may update launch metadata on candidate (for example GridDim or
// SharedMem) to reflect the measured workload. An error rejects this candidate;
// return context cancellation to stop the search. Implementations must honor
// ctx and must not mutate Expr or CUDASource.
type CandidateEvaluator func(ctx context.Context, candidate *TuningCandidate) (time.Duration, error)

// ExpressionWorkloadEvaluator compiles, validates, and measures one materialized
// graph choice using a caller-defined workload. It must return the fully formed
// candidate that it measured and a positive duration. It must honor ctx and
// must not mutate expr. Errors reject the candidate; return context cancellation
// to stop the search.
type ExpressionWorkloadEvaluator func(ctx context.Context, expr *Expr) (TuningCandidate, time.Duration, error)

// Key returns a unique composite identifier for the candidate expression,
// kernel, and launch configuration.
func (c TuningCandidate) Key() string {
	exprStr := ""
	if c.Expr != nil {
		exprStr = c.Expr.String()
	}
	return fmt.Sprintf("%s::%s::b%d::g%d", exprStr, c.KernelName, c.BlockDim, c.GridDim)
}

// AutotuneResult stores the benchmarked latency of a candidate kernel.
type AutotuneResult struct {
	Candidate TuningCandidate
	Latency   time.Duration
	TFLOPS    float64
}

// RankedTuningResult is one successful candidate from measured choice search.
// Results are ordered from shortest to longest measured latency. Search is
// bounded by its deadline and attempt limit, so results cover only choices
// actually measured; each expression contributes its best launch configuration.
type RankedTuningResult struct {
	Candidate TuningCandidate
	Latency   time.Duration
}

// Tuner evaluates candidate AST expressions against real GPU hardware within a time budget.
type Tuner struct {
	mu         sync.RWMutex
	dev        *cuda.Device
	cache      map[string]TuningCandidate
	ranked     map[string][]RankedTuningResult
	bestScores map[string]time.Duration
	pgo        *PGOTracker
}

// NewTuner creates a new hardware-in-the-loop kernel tuner.
func NewTuner(dev *cuda.Device) *Tuner {
	return &Tuner{
		dev:        dev,
		cache:      make(map[string]TuningCandidate),
		ranked:     make(map[string][]RankedTuningResult),
		bestScores: make(map[string]time.Duration),
		pgo:        &PGOTracker{},
	}
}

// SelectRankedWithEvaluator searches reachable e-graph alternatives and
// returns successfully measured candidates found before the search budget or
// attempt limit in latency order. Each distinct expression contributes its
// fastest successful launch configuration. workloadID has the same identity
// and cache requirements as SelectBestWithEvaluator. The returned expressions
// have cloned expression nodes; payload data follows the graph's cloning rules
// and must be treated as immutable.
func (t *Tuner) SelectRankedWithEvaluator(ctx context.Context, g *EGraph, root Id, budget AutotuneBudget, workloadID string, evaluator CandidateEvaluator) ([]RankedTuningResult, error) {
	if evaluator == nil {
		return nil, fmt.Errorf("egraph: nil candidate evaluator for autotuning")
	}
	measure := func(ctx context.Context, expr *Expr, deadline time.Time) (TuningCandidate, time.Duration, error) {
		return measureExpressionWithEvaluator(ctx, expr, deadline, evaluator)
	}
	return t.selectRankedWithWorkload(ctx, g, root, budget, workloadID, "candidate", measure)
}

// SelectRankedWithExpressionEvaluator searches reachable alternatives with a
// caller-defined evaluator and returns successfully measured choices found
// before the search budget or attempt limit in latency order. Its workload
// identity and cache behavior match SelectBestWithExpressionEvaluator.
func (t *Tuner) SelectRankedWithExpressionEvaluator(ctx context.Context, g *EGraph, root Id, budget AutotuneBudget, workloadID string, evaluator ExpressionWorkloadEvaluator) ([]RankedTuningResult, error) {
	if evaluator == nil {
		return nil, fmt.Errorf("egraph: nil expression workload evaluator")
	}
	measure := func(ctx context.Context, expr *Expr, deadline time.Time) (TuningCandidate, time.Duration, error) {
		candidateCtx, cancel := context.WithDeadline(ctx, deadline)
		input := cloneTuningExpr(expr)
		before := cloneTuningExpr(input)
		candidate, duration, err := evaluator(candidateCtx, input)
		ctxErr := candidateCtx.Err()
		cancel()
		if ctxErr != nil {
			return TuningCandidate{}, 0, ctxErr
		}
		if err != nil {
			return TuningCandidate{}, 0, err
		}
		if !reflect.DeepEqual(input, before) {
			return TuningCandidate{}, 0, fmt.Errorf("expression workload evaluator mutated its expression input")
		}
		if !reflect.DeepEqual(candidate.Expr, before) {
			return TuningCandidate{}, 0, fmt.Errorf("expression workload evaluator returned a candidate for a different expression")
		}
		if duration <= 0 {
			return TuningCandidate{}, 0, fmt.Errorf("expression workload evaluator returned non-positive duration")
		}
		candidate.Expr = cloneTuningExpr(before)
		return candidate, duration, nil
	}
	return t.selectRankedWithWorkload(ctx, g, root, budget, workloadID, "expression", measure)
}

func (t *Tuner) selectRankedWithWorkload(ctx context.Context, g *EGraph, root Id, budget AutotuneBudget, workloadID, evaluatorKind string, measure expressionEvaluator) ([]RankedTuningResult, error) {
	if t == nil {
		return nil, fmt.Errorf("egraph: nil tuner for autotuning")
	}
	if ctx == nil {
		return nil, fmt.Errorf("egraph: nil context for autotuning")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if g == nil {
		return nil, fmt.Errorf("egraph: nil E-Graph for autotuning")
	}
	if strings.TrimSpace(workloadID) == "" {
		return nil, fmt.Errorf("egraph: empty workload identity for autotuning")
	}
	if budget == BudgetInstant {
		return nil, fmt.Errorf("egraph: measured candidate search requires a positive budget")
	}
	deadline := time.Now().Add(time.Duration(budget))
	if !time.Now().Before(deadline) {
		return nil, fmt.Errorf("egraph: autotune budget expired before choice search")
	}
	if err := g.RebuildChecked(); err != nil {
		return nil, fmt.Errorf("egraph: rebuild before measured choice search: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if !time.Now().Before(deadline) {
		return nil, fmt.Errorf("egraph: autotune budget expired while rebuilding graph")
	}
	cacheKey := fmt.Sprintf("evaluator=%s:%q::graph=%p::revision=%d::root=%d::budget=%v", evaluatorKind, workloadID, g, g.revision, g.Find(root), budget)
	t.mu.RLock()
	cached, ok := t.ranked[cacheKey]
	t.mu.RUnlock()
	if ok {
		return cloneRankedTuningResults(cached), nil
	}
	var results []RankedTuningResult
	best, duration, err := t.searchChoices(ctx, g, root, deadline, workloadID, measure, &results)
	if err != nil {
		return nil, fmt.Errorf("egraph: rank reachable choices for workload %q: %w", workloadID, err)
	}
	if len(results) == 0 {
		return nil, fmt.Errorf("egraph: measured choice search returned no ranked candidates")
	}
	t.mu.Lock()
	t.ranked[cacheKey] = cloneRankedTuningResults(results)
	t.cache[cacheKey] = cloneTuningCandidate(best)
	t.bestScores[cacheKey] = duration
	t.mu.Unlock()
	return cloneRankedTuningResults(results), nil
}

func cloneRankedTuningResults(results []RankedTuningResult) []RankedTuningResult {
	cloned := make([]RankedTuningResult, len(results))
	for i, result := range results {
		cloned[i] = RankedTuningResult{
			Candidate: cloneTuningCandidate(result.Candidate),
			Latency:   result.Latency,
		}
	}
	return cloned
}

// PGO returns the tracker's live measurements. It is safe for a recorder
// consumer to call Record while SelectBest is running.
func (t *Tuner) PGO() *PGOTracker {
	if t == nil {
		return nil
	}
	return t.pgo
}

// RecordObservation feeds a measured execution into the tuner's live PGO
// tracker. It is safe to call from a background recorder consumer.
func (t *Tuner) RecordObservation(o Observation) {
	if t == nil {
		return
	}
	t.pgo.RecordObservation(o)
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
	case "fused_rmsnorm_gemv", "tile_rmsnorm_qkv", "tile_rmsnorm_proj":
		// x, weight, W_proj, out, in_dim, out_dim, eps; one block per row.
		return launchPlan{
			buffers: []int{benchInDim, benchInDim, benchOutDim * benchInDim, benchOutDim},
			scalars: []any{int32(benchInDim), int32(benchOutDim), float32(1e-5)},
			grid:    func(uint32) uint32 { return benchOutDim },
		}, nil
	case "fused_swiglu_gemv", "tile_swiglu":
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
	case "iq2_xxs_matvec":
		return launchPlan{
			buffers: []int{benchOutDim * (benchInDim / 256) * 66, benchInDim * 4, benchOutDim * 4},
			scalars: []any{int32(benchOutDim), int32(benchInDim)},
			grid:    func(block uint32) uint32 { return (benchOutDim*32 + block - 1) / block },
		}, nil
	case "fused_rope_store_kv":
		return launchPlan{
			buffers: []int{benchHidden * 4, benchHidden * 4, benchHidden * 4, benchHidden * 4, benchHidden * 4, benchHidden * 4},
			scalars: []any{int32(0), float32(10000.0), int32(benchHidden), int32(benchHidden), int32(128), int32(2048)},
			grid:    func(block uint32) uint32 { return (benchHidden + block - 1) / block },
		}, nil
	case "pdl_gemv_pipeline", "fused_gemv_pipeline":
		return launchPlan{
			buffers: []int{benchHidden, benchInter * benchHidden, benchHidden * benchInter, benchHidden},
			scalars: []any{int32(benchHidden), int32(benchInter), int32(benchHidden)},
			shared:  benchInter * 4,
			grid:    func(block uint32) uint32 { return (benchHidden + block - 1) / block },
		}, nil
	case "pdl_split", "pdl_pipeline":
		return launchPlan{
			buffers: []int{benchInDim, benchOutDim * benchInDim, benchOutDim, benchOutDim},
			scalars: []any{int32(benchOutDim), int32(benchInDim)},
			grid:    func(block uint32) uint32 { return (benchOutDim + block - 1) / block },
		}, nil
	}
	return launchPlan{}, fmt.Errorf("egraph: no benchmark launch plan for operator %q", op)
}

// BenchmarkCandidate compiles, loads, warms up, and benchmarks the candidate on the GPU,
// returning average host wall time per iteration across launches and stream
// synchronization. Compilation, loading and warmup precede the measured interval.
func BenchmarkCandidate(dev *cuda.Device, cand TuningCandidate, iters int) (time.Duration, error) {
	return benchmarkCandidateContext(context.Background(), dev, cand, iters)
}

func benchmarkCandidateContext(ctx context.Context, dev *cuda.Device, cand TuningCandidate, iters int) (_ time.Duration, retErr error) {
	if ctx == nil {
		return 0, fmt.Errorf("egraph: nil context for benchmark")
	}
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	if dev == nil {
		return 0, fmt.Errorf("egraph: nil CUDA device for benchmark")
	}
	plan, err := benchmarkPlan(cand)
	if err != nil {
		return 0, err
	}
	if iters <= 0 {
		iters = 5
	}

	mod, err := nvrtcx.CompileContext(ctx, cand.CUDASource, nvrtcx.WithDeviceArch(dev))
	if err != nil {
		return 0, fmt.Errorf("egraph: compile candidate: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return 0, err
	}

	loadedMod, err := mod.Load(dev)
	if err != nil {
		return 0, fmt.Errorf("egraph: load module: %w", err)
	}
	defer loadedMod.Close()
	if err := ctx.Err(); err != nil {
		return 0, err
	}

	fn, err := loadedMod.Function(cand.KernelName)
	if err != nil {
		return 0, fmt.Errorf("egraph: get kernel function %q: %w", cand.KernelName, err)
	}

	stream, err := dev.NewStream()
	if err != nil {
		return 0, fmt.Errorf("egraph: create stream: %w", err)
	}

	bufs := make([]cudamem.Slice[float32], len(plan.buffers))
	for i, n := range plan.buffers {
		b, err := cudamem.Alloc[float32](dev, n)
		if err != nil {
			for _, done := range bufs[:i] {
				_ = done.Free()
			}
			if closeErr := stream.Close(); closeErr != nil {
				return 0, errors.Join(fmt.Errorf("egraph: alloc benchmark buffer %d: %w", i, err), closeErr)
			}
			return 0, fmt.Errorf("egraph: alloc benchmark buffer %d: %w", i, err)
		}
		bufs[i] = b
	}
	// Close synchronizes any work still in flight. A cancelled Synchronize
	// does not cancel GPU work; if Close cannot establish completion, retain
	// the buffers rather than freeing memory that queued kernels may still use.
	defer func() {
		if closeErr := stream.Close(); closeErr != nil {
			retErr = errors.Join(retErr, fmt.Errorf("egraph: close benchmark stream: %w", closeErr))
			return
		}
		for _, b := range bufs {
			if freeErr := b.Free(); freeErr != nil {
				retErr = errors.Join(retErr, fmt.Errorf("egraph: free benchmark buffer: %w", freeErr))
			}
		}
	}()
	// Kernels are measured with a reproducible all-zero input and output state.
	// The benchmark measures launch behavior, not numerical quality; zeroed
	// quantization scales also keep packed-weight fixtures finite and repeatable.
	for i, b := range bufs {
		if err := b.MemsetAsync(0, stream); err != nil {
			return 0, fmt.Errorf("egraph: initialize benchmark buffer %d: %w", i, err)
		}
	}
	if err := stream.Synchronize(ctx); err != nil {
		return 0, fmt.Errorf("egraph: synchronize benchmark buffer initialization: %w", err)
	}

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

	launcher := cudalaunch.Kernel(fn).Grid(grid, 1, 1).Block(block, 1, 1).SharedMem(shared).On(stream)

	// 1. Warmup Execution
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	if err := launcher.Launch(args...); err != nil {
		return 0, fmt.Errorf("egraph: warmup launch: %w", err)
	}
	if err := stream.Synchronize(ctx); err != nil {
		return 0, fmt.Errorf("egraph: warmup sync: %w", err)
	}
	if cand.Expr != nil && cand.Expr.Op == "iq2_xxs_matvec" {
		out := make([]float32, benchOutDim)
		if err := bufs[2].CopyToHost(out); err != nil {
			return 0, fmt.Errorf("egraph: read iq2 benchmark output: %w", err)
		}
		for i, v := range out {
			if math.IsNaN(float64(v)) {
				return 0, fmt.Errorf("egraph: iq2 benchmark output is nan at index %d", i)
			}
		}
	}

	// 2. Host elapsed time for launches and synchronization.
	t0 := time.Now()
	for i := 0; i < iters; i++ {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		if err := launcher.Launch(args...); err != nil {
			return 0, fmt.Errorf("egraph: benchmark launch: %w", err)
		}
	}
	if err := stream.Synchronize(ctx); err != nil {
		return 0, fmt.Errorf("egraph: benchmark sync: %w", err)
	}
	elapsed := time.Since(t0) / time.Duration(iters)

	return elapsed, nil
}

// SelectBest extracts the instant default plan and, if budget > 0, benchmarks candidate
// tile/kernel configurations on the device to pick the fastest kernel.
func (t *Tuner) SelectBest(ctx context.Context, g *EGraph, root Id, budget AutotuneBudget) (*TuningCandidate, error) {
	if t == nil {
		return nil, fmt.Errorf("egraph: nil tuner for autotuning")
	}
	if ctx == nil {
		return nil, fmt.Errorf("egraph: nil context for autotuning")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if g == nil {
		return nil, fmt.Errorf("egraph: nil E-Graph for autotuning")
	}
	var deadline time.Time
	tuningEnabled := budget != BudgetInstant && t.dev != nil
	if tuningEnabled {
		deadline = time.Now().Add(time.Duration(budget))
		if !time.Now().Before(deadline) {
			return nil, fmt.Errorf("egraph: autotune budget expired before default extraction")
		}
	}
	// 1. Instant extraction (zero delay)
	defaultExpr, _, err := ExtractBest(g, root, FusedMegakernelCost{})
	if err != nil {
		return nil, fmt.Errorf("egraph: extract default: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if tuningEnabled && !time.Now().Before(deadline) {
		return nil, fmt.Errorf("egraph: autotune budget expired during default extraction")
	}

	defaultCand := TuningCandidate{
		Expr:       defaultExpr,
		BlockDim:   128,
		KernelName: "Kernel_Auto_Default",
	}

	if !tuningEnabled {
		return &defaultCand, nil
	}
	cacheKey := fmt.Sprintf("graph=%p::revision=%d::root=%d::budget=%v", g, g.revision, g.Find(root), budget)
	t.mu.RLock()
	cached, ok := t.cache[cacheKey]
	t.mu.RUnlock()
	if ok {
		copy := cloneTuningCandidate(cached)
		return &copy, nil
	}

	bestCand, bestDur, err := t.searchChoices(ctx, g, root, deadline, "synthetic", t.measureExpression, nil)
	if err != nil {
		return nil, fmt.Errorf("egraph: tune reachable graph choices for %s: %w", defaultExpr.String(), err)
	}

	t.mu.Lock()
	t.cache[cacheKey] = cloneTuningCandidate(bestCand)
	t.bestScores[cacheKey] = bestDur
	t.mu.Unlock()

	bestCand = cloneTuningCandidate(bestCand)
	return &bestCand, nil
}

// SelectBestWithEvaluator searches reachable e-graph alternatives using a
// caller-supplied workload evaluator. workloadID must be a stable identity for
// the exact shapes, inputs, correctness checks, and hard constraints measured
// by evaluator; it participates in both the random seed and result cache key.
// Errors from evaluator reject that candidate, while context cancellation
// stops the search. The evaluator must honor ctx because it cannot be forcibly
// interrupted.
func (t *Tuner) SelectBestWithEvaluator(ctx context.Context, g *EGraph, root Id, budget AutotuneBudget, workloadID string, evaluator CandidateEvaluator) (*TuningCandidate, error) {
	if evaluator == nil {
		return nil, fmt.Errorf("egraph: nil candidate evaluator for autotuning")
	}
	measure := func(ctx context.Context, expr *Expr, deadline time.Time) (TuningCandidate, time.Duration, error) {
		return measureExpressionWithEvaluator(ctx, expr, deadline, evaluator)
	}
	return t.selectBestWithWorkload(ctx, g, root, budget, workloadID, "candidate", measure)
}

// SelectBestWithExpressionEvaluator searches reachable alternatives with a
// caller-defined workload evaluator. The callback receives each materialized
// expression directly, without generic CUDA generation or launch-parameter
// sweeps. workloadID must stably identify the shapes, inputs, correctness
// checks, and hard constraints; it participates in the deterministic search
// seed and result cache key. The callback must honor ctx because it cannot be
// forcibly interrupted. Callback errors reject that expression candidate.
func (t *Tuner) SelectBestWithExpressionEvaluator(ctx context.Context, g *EGraph, root Id, budget AutotuneBudget, workloadID string, evaluator ExpressionWorkloadEvaluator) (*TuningCandidate, error) {
	if evaluator == nil {
		return nil, fmt.Errorf("egraph: nil expression workload evaluator")
	}
	measure := func(ctx context.Context, expr *Expr, deadline time.Time) (TuningCandidate, time.Duration, error) {
		candidateCtx, cancel := context.WithDeadline(ctx, deadline)
		input := cloneTuningExpr(expr)
		before := cloneTuningExpr(input)
		candidate, duration, err := evaluator(candidateCtx, input)
		ctxErr := candidateCtx.Err()
		cancel()
		if ctxErr != nil {
			return TuningCandidate{}, 0, ctxErr
		}
		if err != nil {
			return TuningCandidate{}, 0, err
		}
		if !reflect.DeepEqual(input, before) {
			return TuningCandidate{}, 0, fmt.Errorf("expression workload evaluator mutated its expression input")
		}
		if !reflect.DeepEqual(candidate.Expr, before) {
			return TuningCandidate{}, 0, fmt.Errorf("expression workload evaluator returned a candidate for a different expression")
		}
		if duration <= 0 {
			return TuningCandidate{}, 0, fmt.Errorf("expression workload evaluator returned non-positive duration")
		}
		candidate.Expr = cloneTuningExpr(before)
		return candidate, duration, nil
	}
	return t.selectBestWithWorkload(ctx, g, root, budget, workloadID, "expression", measure)
}

func (t *Tuner) selectBestWithWorkload(ctx context.Context, g *EGraph, root Id, budget AutotuneBudget, workloadID, evaluatorKind string, measure expressionEvaluator) (*TuningCandidate, error) {
	if t == nil {
		return nil, fmt.Errorf("egraph: nil tuner for autotuning")
	}
	if ctx == nil {
		return nil, fmt.Errorf("egraph: nil context for autotuning")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if g == nil {
		return nil, fmt.Errorf("egraph: nil E-Graph for autotuning")
	}
	if strings.TrimSpace(workloadID) == "" {
		return nil, fmt.Errorf("egraph: empty workload identity for autotuning")
	}
	if budget == BudgetInstant {
		return nil, fmt.Errorf("egraph: measured candidate search requires a positive budget")
	}
	deadline := time.Now().Add(time.Duration(budget))
	if !time.Now().Before(deadline) {
		return nil, fmt.Errorf("egraph: autotune budget expired before choice search")
	}
	if err := g.RebuildChecked(); err != nil {
		return nil, fmt.Errorf("egraph: rebuild before measured choice search: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if !time.Now().Before(deadline) {
		return nil, fmt.Errorf("egraph: autotune budget expired while rebuilding graph")
	}
	cacheKey := fmt.Sprintf("evaluator=%s:%q::graph=%p::revision=%d::root=%d::budget=%v", evaluatorKind, workloadID, g, g.revision, g.Find(root), budget)
	t.mu.RLock()
	cached, ok := t.cache[cacheKey]
	t.mu.RUnlock()
	if ok {
		copy := cloneTuningCandidate(cached)
		return &copy, nil
	}
	best, duration, err := t.searchChoices(ctx, g, root, deadline, workloadID, measure, nil)
	if err != nil {
		return nil, fmt.Errorf("egraph: evaluate reachable choices for workload %q: %w", workloadID, err)
	}
	t.mu.Lock()
	t.cache[cacheKey] = cloneTuningCandidate(best)
	t.bestScores[cacheKey] = duration
	t.mu.Unlock()
	best = cloneTuningCandidate(best)
	return &best, nil
}

func measureExpressionWithEvaluator(ctx context.Context, expr *Expr, deadline time.Time, evaluator CandidateEvaluator) (TuningCandidate, time.Duration, error) {
	var best TuningCandidate
	var bestDuration time.Duration
	benchmarked := false
	var lastErr error
	fingerprint := expr.Fingerprint()
	for _, blockSize := range autotuneBlockSizes {
		if err := ctx.Err(); err != nil {
			return TuningCandidate{}, 0, err
		}
		if !time.Now().Before(deadline) {
			break
		}
		name := fmt.Sprintf("Kernel_Auto_%s_Block%d", sanitizeName(expr.Op), blockSize)
		source, err := GenerateCUDA(expr, CodegenOptions{KernelName: name, Precision: "float"})
		if err != nil {
			lastErr = err
			continue
		}
		candidate := TuningCandidate{Expr: expr, BlockDim: blockSize, KernelName: name, CUDASource: source}
		candidateCtx, cancel := context.WithDeadline(ctx, deadline)
		latency, evalErr := evaluator(candidateCtx, &candidate)
		cancel()
		if errors.Is(evalErr, context.Canceled) || errors.Is(evalErr, context.DeadlineExceeded) {
			return TuningCandidate{}, 0, evalErr
		}
		if err := ctx.Err(); err != nil {
			return TuningCandidate{}, 0, err
		}
		if !time.Now().Before(deadline) {
			lastErr = fmt.Errorf("autotune budget expired during workload evaluation")
			break
		}
		if evalErr != nil {
			lastErr = evalErr
			continue
		}
		if candidate.Expr != expr || candidate.CUDASource != source ||
			candidate.KernelName != name || candidate.BlockDim != blockSize ||
			expr.Fingerprint() != fingerprint {
			lastErr = fmt.Errorf("candidate evaluator changed expression, source, or candidate identity")
			continue
		}
		if latency <= 0 {
			lastErr = fmt.Errorf("candidate evaluator returned non-positive latency")
			continue
		}
		if !benchmarked || latency < bestDuration {
			best, bestDuration, benchmarked = candidate, latency, true
		}
	}
	if !benchmarked {
		if err := ctx.Err(); err != nil {
			return TuningCandidate{}, 0, err
		}
		if lastErr == nil {
			lastErr = fmt.Errorf("no workload evaluation completed")
		}
		return TuningCandidate{}, 0, lastErr
	}
	return best, bestDuration, nil
}
