package egraph

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// ErrThrottled is returned when shadow execution is prevented by the throttle gate
// due to high memory bus saturation or memory pressure from the primary stream.
var ErrThrottled = errors.New("egraph: shadow execution throttled by memory pressure")

// ThrottleGate determines whether speculative shadow execution is permitted.
type ThrottleGate interface {
	Allow() bool
}

// MemoryThrottle gates speculative shadow execution based on DRAM memory bus saturation,
// ensuring shadow kernels only run when the bus is not saturated by the primary inference stream.
// This prevents tail latency spikes during memory-bandwidth-bound decode phases.
type MemoryThrottle struct {
	Threshold float64        // saturation threshold in [0.0, 1.0]; shadow execution is blocked at or above this value
	Sampler   func() float64 // returns current DRAM bus utilization in [0.0, 1.0]
}

// NewMemoryThrottle creates a MemoryThrottle with the given saturation threshold and sampler.
func NewMemoryThrottle(threshold float64, sampler func() float64) *MemoryThrottle {
	return &MemoryThrottle{
		Threshold: threshold,
		Sampler:   sampler,
	}
}

// Allow reports whether current memory utilization is strictly below Threshold.
func (m *MemoryThrottle) Allow() bool {
	if m == nil || m.Sampler == nil {
		return true
	}
	return m.Sampler() < m.Threshold
}

// ShadowExecutor executes a candidate kernel plan on isolated shadow buffers
// using an isolated CUDA stream and returns the measured execution duration.
type ShadowExecutor interface {
	ExecuteShadow(ctx context.Context, s *cuda.Stream, cand TuningCandidate) (time.Duration, error)
}

// DeviceShadowExecutor implements ShadowExecutor on real CUDA hardware,
// executing candidate kernels on isolated shadow device buffers and measuring
// durations with cuda.Event markers.
type DeviceShadowExecutor struct {
	dev *cuda.Device
}

// NewDeviceShadowExecutor creates a DeviceShadowExecutor for dev.
func NewDeviceShadowExecutor(dev *cuda.Device) *DeviceShadowExecutor {
	return &DeviceShadowExecutor{dev: dev}
}

// ExecuteShadow compiles, loads, allocates shadow buffers, and benchmarks cand on s.
func (e *DeviceShadowExecutor) ExecuteShadow(ctx context.Context, s *cuda.Stream, cand TuningCandidate) (time.Duration, error) {
	if e.dev == nil {
		return 0, errors.New("egraph: nil device for shadow execution")
	}
	if s == nil {
		return 0, errors.New("egraph: nil shadow stream")
	}

	mod, err := nvrtcx.Compile(cand.CUDASource, nvrtcx.WithDeviceArch(e.dev))
	if err != nil {
		return 0, fmt.Errorf("egraph: compile candidate: %w", err)
	}

	loadedMod, err := mod.Load(e.dev)
	if err != nil {
		return 0, fmt.Errorf("egraph: load module: %w", err)
	}
	defer loadedMod.Unload()

	fn, err := loadedMod.GetFunction(cand.KernelName)
	if err != nil {
		return 0, fmt.Errorf("egraph: get kernel function %q: %w", cand.KernelName, err)
	}

	plan, err := benchmarkPlan(cand)
	if err != nil {
		return 0, err
	}

	bufs := make([]cudamem.Slice[float32], len(plan.buffers))
	for i, n := range plan.buffers {
		b, err := cudamem.Alloc[float32](e.dev, n)
		if err != nil {
			for _, done := range bufs[:i] {
				_ = done.Free()
			}
			return 0, fmt.Errorf("egraph: alloc shadow buffer %d: %w", i, err)
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

	launcher := cudalaunch.Kernel(fn).Grid(grid, 1).Block(block).SharedMem(shared).On(s)

	start, err := e.dev.NewEvent()
	if err != nil {
		return 0, fmt.Errorf("egraph: create start event: %w", err)
	}
	defer start.Destroy()

	stop, err := e.dev.NewEvent()
	if err != nil {
		return 0, fmt.Errorf("egraph: create stop event: %w", err)
	}
	defer stop.Destroy()

	if err := start.Record(s); err != nil {
		return 0, fmt.Errorf("egraph: record start event: %w", err)
	}
	if err := launcher.Launch(args...); err != nil {
		return 0, fmt.Errorf("egraph: launch shadow kernel: %w", err)
	}
	if err := stop.Record(s); err != nil {
		return 0, fmt.Errorf("egraph: record stop event: %w", err)
	}
	if err := stop.Synchronize(ctx); err != nil {
		return 0, fmt.Errorf("egraph: sync stop event: %w", err)
	}

	elapsed, err := stop.ElapsedSince(start)
	if err != nil {
		return 0, fmt.Errorf("egraph: measure event elapsed time: %w", err)
	}
	return elapsed, nil
}

// ShadowTuner manages speculative shadow execution of alternative candidate kernel plans
// on an isolated stream and coordinates in-place CUDA graph hot-swapping.
type ShadowTuner struct {
	mu         sync.RWMutex
	dev        *cuda.Device
	stream     *cuda.Stream
	ownsStream bool
	tracker    *PGOTracker
	throttle   ThrottleGate
	executor   ShadowExecutor
}

// ShadowTunerOption configures a ShadowTuner.
type ShadowTunerOption func(*ShadowTuner)

// WithThrottle configures the throttle gate for shadow execution.
func WithThrottle(gate ThrottleGate) ShadowTunerOption {
	return func(st *ShadowTuner) {
		st.throttle = gate
	}
}

// WithPGOTracker sets an explicit PGOTracker instance.
func WithPGOTracker(tracker *PGOTracker) ShadowTunerOption {
	return func(st *ShadowTuner) {
		st.tracker = tracker
	}
}

// WithShadowExecutor sets a custom ShadowExecutor backend.
func WithShadowExecutor(exec ShadowExecutor) ShadowTunerOption {
	return func(st *ShadowTuner) {
		st.executor = exec
	}
}

// WithShadowStream sets an explicit isolated CUDA stream for shadow execution.
func WithShadowStream(s *cuda.Stream) ShadowTunerOption {
	return func(st *ShadowTuner) {
		st.stream = s
		st.ownsStream = false
	}
}

// NewShadowTuner creates a new speculative shadow tuner.
func NewShadowTuner(dev *cuda.Device, opts ...ShadowTunerOption) (*ShadowTuner, error) {
	st := &ShadowTuner{
		dev:     dev,
		tracker: &PGOTracker{},
	}
	for _, opt := range opts {
		opt(st)
	}
	if st.executor == nil && dev != nil {
		st.executor = NewDeviceShadowExecutor(dev)
	}
	if st.stream == nil && dev != nil {
		s, err := dev.NewStream()
		if err != nil {
			return nil, fmt.Errorf("egraph: create shadow stream: %w", err)
		}
		st.stream = s
		st.ownsStream = true
	}
	return st, nil
}

// Close releases resources held by the tuner, destroying the shadow stream if owned.
func (st *ShadowTuner) Close() error {
	if st == nil {
		return nil
	}
	st.mu.Lock()
	defer st.mu.Unlock()
	if st.ownsStream && st.stream != nil {
		err := st.stream.Destroy()
		st.stream = nil
		st.ownsStream = false
		return err
	}
	return nil
}

// PGO returns the live PGOTracker containing recorded execution profiles.
func (st *ShadowTuner) PGO() *PGOTracker {
	if st == nil {
		return nil
	}
	return st.tracker
}

// Stream returns the isolated shadow CUDA stream, or nil if none is configured.
func (st *ShadowTuner) Stream() *cuda.Stream {
	if st == nil {
		return nil
	}
	st.mu.RLock()
	defer st.mu.RUnlock()
	return st.stream
}

// SetThrottle updates the throttle gate dynamically.
func (st *ShadowTuner) SetThrottle(gate ThrottleGate) {
	if st == nil {
		return
	}
	st.mu.Lock()
	defer st.mu.Unlock()
	st.throttle = gate
}

// Evaluate executes cand on the isolated shadow stream if permitted by the throttle gate,
// measures its latency, and feeds the observation directly into the PGOTracker.
// If the memory bus is saturated according to the throttle gate, it returns ErrThrottled.
func (st *ShadowTuner) Evaluate(ctx context.Context, cand TuningCandidate) (time.Duration, error) {
	if st == nil {
		return 0, errors.New("egraph: nil shadow tuner")
	}
	st.mu.RLock()
	throttle := st.throttle
	st.mu.RUnlock()

	if throttle != nil && !throttle.Allow() {
		return 0, ErrThrottled
	}

	st.mu.RLock()
	executor := st.executor
	stream := st.stream
	st.mu.RUnlock()

	if executor == nil {
		return 0, errors.New("egraph: no shadow executor configured")
	}

	dur, err := executor.ExecuteShadow(ctx, stream, cand)
	if err != nil {
		return 0, fmt.Errorf("egraph: shadow execute: %w", err)
	}

	st.tracker.RecordObservation(Observation{
		Candidate: cand.Key(),
		Duration:  dur,
	})
	return dur, nil
}

// EvaluateCandidates evaluates a series of candidate plans under throttle gating,
// collecting results for successfully executed candidates and returning ErrThrottled
// if throttled.
func (st *ShadowTuner) EvaluateCandidates(ctx context.Context, candidates []TuningCandidate) ([]AutotuneResult, error) {
	var results []AutotuneResult
	for _, cand := range candidates {
		dur, err := st.Evaluate(ctx, cand)
		if err != nil {
			if errors.Is(err, ErrThrottled) {
				return results, ErrThrottled
			}
			return results, err
		}
		results = append(results, AutotuneResult{
			Candidate: cand,
			Latency:   dur,
		})
	}
	return results, nil
}

// CandidatePlansForExpr returns candidate kernel plans for expr across the given block sizes.
// If blockSizes is empty, default block sizes (128, 256) are used.
func CandidatePlansForExpr(expr *Expr, blockSizes []int) []TuningCandidate {
	if expr == nil {
		return nil
	}
	if len(blockSizes) == 0 {
		blockSizes = []int{128, 256}
	}
	candidates := make([]TuningCandidate, 0, len(blockSizes))
	op := expr.Op
	for _, b := range blockSizes {
		name := fmt.Sprintf("Kernel_%s_Block%d", sanitizeName(op), b)
		cand := TuningCandidate{
			Expr:       expr,
			BlockDim:   b,
			KernelName: name,
		}
		if plan, err := benchmarkPlan(cand); err == nil {
			cand.GridDim = int(plan.grid(uint32(b)))
			cand.SharedMem = int(plan.shared)
		}
		if IsTerminalOp(op) {
			if src, err := GenerateCUDA(expr, CodegenOptions{
				KernelName: name,
				Precision:  "float",
			}); err == nil {
				cand.CUDASource = src
			}
		}
		candidates = append(candidates, cand)
	}
	return candidates
}

// ExtractCandidatePlans generates alternative candidate kernel plans from an E-Graph
// rooted at root, exploring different AST rewrites (such as sub-byte dequantization fusion,
// tiling sizes, and megakernel fusion) paired with thread block geometries.
func ExtractCandidatePlans(g *EGraph, root Id, blockSizes []int) ([]TuningCandidate, error) {
	if g == nil {
		return nil, errors.New("egraph: nil E-graph")
	}
	if len(blockSizes) == 0 {
		blockSizes = []int{128, 256}
	}
	topASTs, err := ExtractTopK(g, root, 4, FusedMegakernelCost{})
	if err != nil {
		return nil, fmt.Errorf("egraph: extract candidate plans: %w", err)
	}

	var candidates []TuningCandidate
	for _, ast := range topASTs {
		candidates = append(candidates, CandidatePlansForExpr(ast.Expr, blockSizes)...)
	}
	return candidates, nil
}

func sanitizeName(s string) string {
	var b strings.Builder
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' {
			b.WriteRune(r)
		} else {
			b.WriteRune('_')
		}
	}
	return b.String()
}

// PromotionPolicy specifies criteria for confirming a candidate plan is faster
// before promoting it into the active execution graph.
type PromotionPolicy struct {
	MinObservations int           // Minimum number of observations required for candidate (default: 1)
	MinImprovement  time.Duration // Minimum absolute duration improvement required
	ImprovementFrac float64       // Minimum fractional improvement required (e.g. 0.05 for 5%)
}

// ShouldPromote reports whether candidate is confirmed faster than current according to
// profile tracker scores and the promotion policy.
func (st *ShadowTuner) ShouldPromote(current, candidate TuningCandidate, policy PromotionPolicy) bool {
	if st == nil || st.tracker == nil {
		return false
	}
	currentScore, okCurrent := st.tracker.Score(current.Key())
	candScore, okCand := st.tracker.Score(candidate.Key())
	if !okCand {
		return false
	}
	minObs := policy.MinObservations
	if minObs <= 0 {
		minObs = 1
	}
	if st.tracker.Count(candidate.Key()) < minObs {
		return false
	}
	if !okCurrent {
		return true
	}
	if candScore >= currentScore {
		return false
	}
	diff := currentScore - candScore
	if diff < policy.MinImprovement {
		return false
	}
	if policy.ImprovementFrac > 0 {
		if float64(diff)/float64(currentScore) < policy.ImprovementFrac {
			return false
		}
	}
	return true
}

// GraphNodeUpdater is the minimal interface for updating a kernel node in an instantiated CUDA graph in-place.
// *cudagraph.Exec implements this interface directly.
type GraphNodeUpdater interface {
	SetKernelNodeGrid(node *cudagraph.Node, grid cudalaunch.Dim3) error
	SetKernelNodeParams(node *cudagraph.Node, params *driver.KernelNodeParams) error
	UpdateKernelNode(node *cudagraph.Node, grid cudalaunch.Dim3, args ...any) error
	ApplyKernelNodeUpdate(node *cudagraph.Node, update cudagraph.KernelNodeUpdate) error
}

// HotSwapNode updates an instantiated graph kernel node in-place to use candidate launch
// configuration (grid dimension, block dimension, and kernel arguments) without
// graph re-instantiation or topology validation overhead.
func HotSwapNode(exec GraphNodeUpdater, node *cudagraph.Node, cand TuningCandidate, args ...any) error {
	if exec == nil {
		return errors.New("egraph: nil graph updater")
	}
	if node == nil {
		return errors.New("egraph: nil graph node")
	}
	grid := cudalaunch.Dim3{
		X: uint32(cand.GridDim),
		Y: 1,
		Z: 1,
	}
	if grid.X == 0 && cand.BlockDim > 0 {
		grid.X = 1
	}
	if len(args) > 0 {
		return exec.UpdateKernelNode(node, grid, args...)
	}
	if grid.X > 0 {
		return exec.SetKernelNodeGrid(node, grid)
	}
	return nil
}

// HotSwapKernelNodeUpdates applies a sequence of typed KernelNodeUpdates to node on exec in-place
// via ApplyKernelNodeUpdate, matching the semantics of cudaGraphKernelNodeUpdatesApply.
func HotSwapKernelNodeUpdates(exec GraphNodeUpdater, node *cudagraph.Node, updates ...cudagraph.KernelNodeUpdate) error {
	if exec == nil {
		return errors.New("egraph: nil graph updater")
	}
	if node == nil {
		return errors.New("egraph: nil graph node")
	}
	for _, u := range updates {
		if err := exec.ApplyKernelNodeUpdate(node, u); err != nil {
			return fmt.Errorf("egraph: apply kernel node update: %w", err)
		}
	}
	return nil
}

// PromoteAndApply checks whether candidate is confirmed faster than current according to
// policy, and if so, performs an in-place hot-swap of the kernel node in exec without
// re-instantiating the graph. It reports whether the promotion and hot-swap occurred.
func (st *ShadowTuner) PromoteAndApply(
	exec GraphNodeUpdater,
	node *cudagraph.Node,
	current, candidate TuningCandidate,
	policy PromotionPolicy,
	args ...any,
) (bool, error) {
	if !st.ShouldPromote(current, candidate, policy) {
		return false, nil
	}
	if err := HotSwapNode(exec, node, candidate, args...); err != nil {
		return false, fmt.Errorf("egraph: hot-swap kernel node: %w", err)
	}
	return true, nil
}
