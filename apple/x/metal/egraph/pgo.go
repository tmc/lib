// SPDX-License-Identifier: Apache-2.0

package egraph

import (
	"fmt"
	"math"
	"sort"
	"sync"
	"time"
)

// PGOFeedbackCostModel calculates E-Graph extraction costs derived truthfully
// from runtime execution measurements (e.g. from FlightRecorder).
type PGOFeedbackCostModel struct {
	mu           sync.RWMutex
	opLatencies  map[string]*OpLatencyStat
	fallbackCost CostModel
}

// OpLatencyStat tracks execution time statistics for an operator or fused kernel.
type OpLatencyStat struct {
	Samples  int
	TotalNs  int64
	MinNs    int64
	MaxNs    int64
	SumSqNs  float64
	LastSeen time.Time
}

// NewPGOFeedbackCostModel creates a cost model that uses live measured latencies.
func NewPGOFeedbackCostModel(fallback CostModel) *PGOFeedbackCostModel {
	if fallback == nil {
		fallback = FusedMegakernelCost{}
	}
	return &PGOFeedbackCostModel{
		opLatencies:  make(map[string]*OpLatencyStat),
		fallbackCost: fallback,
	}
}

// RecordMeasurement updates the observed latency for a given operation.
func (m *PGOFeedbackCostModel) RecordMeasurement(op string, dur time.Duration) {
	if m == nil || op == "" || dur <= 0 {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()

	ns := dur.Nanoseconds()
	if ns <= 0 {
		return
	}
	stat, ok := m.opLatencies[op]
	if !ok {
		stat = &OpLatencyStat{
			MinNs: ns,
			MaxNs: ns,
		}
		m.opLatencies[op] = stat
	}

	stat.Samples++
	stat.TotalNs += ns
	stat.SumSqNs += float64(ns) * float64(ns)
	if ns < stat.MinNs {
		stat.MinNs = ns
	}
	if ns > stat.MaxNs {
		stat.MaxNs = ns
	}
	stat.LastSeen = time.Now()
}

// NodeCost implements CostModel by looking up measured execution latency.
func (m *PGOFeedbackCostModel) NodeCost(n ENode, childCosts []float64) float64 {
	if m == nil {
		return AstSizeCost{}.NodeCost(n, childCosts)
	}
	m.mu.RLock()
	stat, ok := m.opLatencies[n.Op]
	if ok {
		copy := *stat
		stat = &copy
	}
	m.mu.RUnlock()

	if ok && stat.Samples > 0 {
		// Mean execution time in microseconds as the cost metric
		total := float64(stat.TotalNs) / float64(stat.Samples) / 1000.0
		for _, cost := range childCosts {
			total += cost
		}
		return total
	}

	// Preserve the fallback model's complete child-cost semantics. Some cost
	// models are not additive, so evaluating the node cost without children and
	// summing them here would change their result (or panic).
	fallback := m.fallbackCost
	if fallback == nil {
		fallback = AstSizeCost{}
	}
	return evaluateCostModel(fallback, n, childCosts)
}

// AutoTuningTrial records a single parameter search trial for a kernel threadgroup.
type AutoTuningTrial struct {
	Kernel          string    `json:"kernel"`
	ThreadgroupSize [3]int    `json:"threadgroup_size"`
	DurationUS      float64   `json:"duration_us"`
	Samples         int       `json:"samples"`
	Timestamp       time.Time `json:"timestamp"`
}

// HotSwapEvent records a dynamic kernel pipeline state replacement.
type HotSwapEvent struct {
	KernelName string    `json:"kernel_name"`
	FromDesc   string    `json:"from_desc"`
	ToDesc     string    `json:"to_desc"`
	Reason     string    `json:"reason"`
	Timestamp  time.Time `json:"timestamp"`
}

// OptimizationIteration records a progressive re-optimization milestone.
type OptimizationIteration struct {
	Index      int       `json:"index"`
	Name       string    `json:"name"`
	LatencyMs  float64   `json:"latency_ms"`
	SpeedupPct float64   `json:"speedup_pct"`
	Active     bool      `json:"active"`
	Timestamp  time.Time `json:"timestamp"`
}

// PGOTelemetryData represents the complete M5 Profile-Guided Optimization runtime state.
type PGOTelemetryData struct {
	ActiveTarget      string                  `json:"active_target"`
	CostModel         string                  `json:"cost_model"`
	History           []OptimizationIteration `json:"history"`
	AutoTuningTrials  []AutoTuningTrial       `json:"autotuning_trials"`
	HotSwaps          []HotSwapEvent          `json:"hot_swaps"`
	Convergence       ConvergenceStatus       `json:"convergence"`
	CumulativeSpeedup float64                 `json:"cumulative_speedup_x"`
}

// ConvergenceStatus reports tuner state and measured variant summaries.
type ConvergenceStatus struct {
	Locked             bool                  `json:"locked"`
	OptimalKernel      string                `json:"optimal_kernel,omitempty"`
	OptimalTG          [3]int                `json:"optimal_threadgroup"` // Best observed mean when one kernel is present.
	ConfidenceScore    float64               `json:"confidence_score"`
	ConfidenceMeasured bool                  `json:"confidence_measured"`
	EvaluatedConfigs   int                   `json:"evaluated_configs"`
	BestByKernel       []KernelVariantResult `json:"best_by_kernel,omitempty"`
}

// KernelVariantResult reports the lowest sample-weighted mean threadgroup for
// one kernel.
type KernelVariantResult struct {
	Kernel       string  `json:"kernel"`
	Threadgroup  [3]int  `json:"threadgroup_size"`
	MeanDuration float64 `json:"mean_duration_us"` // Weighted by each trial's Samples.
	Trials       int     `json:"trials"`           // Number of recorded trial rows.
}

// PGOTracker collects runtime telemetry, tracks auto-tuning trials, and provides
// structured PGO data for dashboard and diagnostic endpoints.
type PGOTracker struct {
	mu           sync.RWMutex
	activeTarget string
	costModel    string
	history      []OptimizationIteration
	trials       []AutoTuningTrial
	hotSwaps     []HotSwapEvent
	baselineMs   float64
	locked       bool
}

// NewPGOTracker initializes a PGO telemetry tracker.
func NewPGOTracker(target string) *PGOTracker {
	return &PGOTracker{
		activeTarget: target,
		costModel:    "Hardware Feedback Driven (FlightRecorder Latency Sampling)",
		history:      make([]OptimizationIteration, 0, 16),
		trials:       make([]AutoTuningTrial, 0, 64),
		hotSwaps:     make([]HotSwapEvent, 0, 16),
	}
}

// RecordIteration logs an optimization step with observed latency. It ignores
// a nil tracker, empty name, or non-positive or non-finite latency.
func (t *PGOTracker) RecordIteration(name string, latencyMs float64, active bool) {
	if t == nil || name == "" || latencyMs <= 0 || math.IsNaN(latencyMs) || math.IsInf(latencyMs, 0) {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	idx := len(t.history)
	if idx == 0 {
		t.baselineMs = latencyMs
	}

	speedupPct := 0.0
	if t.baselineMs > 0 && idx > 0 {
		speedupPct = ((t.baselineMs - latencyMs) / t.baselineMs) * 100.0
	}

	if active {
		for i := range t.history {
			t.history[i].Active = false
		}
	}

	t.history = append(t.history, OptimizationIteration{
		Index:      idx,
		Name:       name,
		LatencyMs:  latencyMs,
		SpeedupPct: speedupPct,
		Active:     active,
		Timestamp:  time.Now(),
	})
}

// RecordTrial logs a threadgroup auto-tuning trial. durUS is the trial's mean
// duration across samples. It ignores invalid kernel names, non-positive
// dimensions or duration, non-finite duration, and non-positive sample counts.
func (t *PGOTracker) RecordTrial(kernel string, tg [3]int, durUS float64, samples int) {
	if t == nil || kernel == "" || durUS <= 0 || math.IsNaN(durUS) || math.IsInf(durUS, 0) || samples <= 0 {
		return
	}
	for _, size := range tg {
		if size <= 0 {
			return
		}
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	t.trials = append(t.trials, AutoTuningTrial{
		Kernel:          kernel,
		ThreadgroupSize: tg,
		DurationUS:      durUS,
		Samples:         samples,
		Timestamp:       time.Now(),
	})
}

// RecordHotSwap logs a kernel hot-swap event. It ignores a nil tracker,
// empty kernel name, and events whose source and destination are identical.
func (t *PGOTracker) RecordHotSwap(kernelName, fromDesc, toDesc, reason string) {
	if t == nil || kernelName == "" || fromDesc == toDesc {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	t.hotSwaps = append(t.hotSwaps, HotSwapEvent{
		KernelName: kernelName,
		FromDesc:   fromDesc,
		ToDesc:     toDesc,
		Reason:     reason,
		Timestamp:  time.Now(),
	})
}

// LockConvergence marks the optimization search space as converged.
func (t *PGOTracker) LockConvergence(locked bool) {
	if t == nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.locked = locked
}

// Telemetry returns a snapshot of truthful PGO telemetry data.
func (t *PGOTracker) Telemetry() PGOTelemetryData {
	if t == nil {
		return PGOTelemetryData{}
	}
	t.mu.RLock()
	defer t.mu.RUnlock()

	historyCopy := make([]OptimizationIteration, len(t.history))
	copy(historyCopy, t.history)

	trialsCopy := make([]AutoTuningTrial, len(t.trials))
	copy(trialsCopy, t.trials)

	hotSwapsCopy := make([]HotSwapEvent, len(t.hotSwaps))
	copy(hotSwapsCopy, t.hotSwaps)
	convergence := summarizeTrials(t.trials)
	convergence.Locked = t.locked

	var cumulative float64 = 1.0
	if len(t.history) > 0 && t.baselineMs > 0 {
		currentActive := t.baselineMs
		for _, h := range t.history {
			if h.Active && h.LatencyMs > 0 {
				currentActive = h.LatencyMs
				break
			}
		}
		cumulative = t.baselineMs / currentActive
	}

	return PGOTelemetryData{
		ActiveTarget:      t.activeTarget,
		CostModel:         t.costModel,
		History:           historyCopy,
		AutoTuningTrials:  trialsCopy,
		HotSwaps:          hotSwapsCopy,
		CumulativeSpeedup: cumulative,
		Convergence:       convergence,
	}
}

func summarizeTrials(trials []AutoTuningTrial) ConvergenceStatus {
	type config struct {
		kernel string
		tg     [3]int
	}
	type aggregate struct {
		mean         float64
		sampleWeight float64
		trials       int
	}
	byConfig := make(map[config]aggregate)
	for _, trial := range trials {
		key := config{kernel: trial.Kernel, tg: trial.ThreadgroupSize}
		stat := byConfig[key]
		weight := float64(trial.Samples)
		stat.sampleWeight += weight
		stat.mean += (trial.DurationUS - stat.mean) * weight / stat.sampleWeight
		stat.trials++
		byConfig[key] = stat
	}
	kernels := make([]string, 0)
	byKernel := make(map[string][]KernelVariantResult)
	for key, stat := range byConfig {
		if _, ok := byKernel[key.kernel]; !ok {
			kernels = append(kernels, key.kernel)
		}
		byKernel[key.kernel] = append(byKernel[key.kernel], KernelVariantResult{
			Kernel:       key.kernel,
			Threadgroup:  key.tg,
			MeanDuration: stat.mean,
			Trials:       stat.trials,
		})
	}
	sort.Strings(kernels)
	result := ConvergenceStatus{EvaluatedConfigs: len(byConfig)}
	for _, kernel := range kernels {
		variants := byKernel[kernel]
		sort.Slice(variants, func(i, j int) bool {
			if variants[i].MeanDuration != variants[j].MeanDuration {
				return variants[i].MeanDuration < variants[j].MeanDuration
			}
			for axis := range variants[i].Threadgroup {
				if variants[i].Threadgroup[axis] != variants[j].Threadgroup[axis] {
					return variants[i].Threadgroup[axis] < variants[j].Threadgroup[axis]
				}
			}
			return false
		})
		result.BestByKernel = append(result.BestByKernel, variants[0])
	}
	if len(result.BestByKernel) == 1 {
		best := result.BestByKernel[0]
		result.OptimalKernel = best.Kernel
		result.OptimalTG = best.Threadgroup
	}
	return result
}

// String returns human-readable summary of PGO state.
func (t *PGOTracker) String() string {
	data := t.Telemetry()
	return fmt.Sprintf("PGOTracker[Target=%s, Iterations=%d, Trials=%d, Speedup=%.2fx, Locked=%v]",
		data.ActiveTarget, len(data.History), len(data.AutoTuningTrials), data.CumulativeSpeedup, data.Convergence.Locked)
}
