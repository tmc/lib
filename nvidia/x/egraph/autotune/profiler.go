package autotune

import (
	"encoding/json"
	"fmt"
	"math"
	"sync"
)

// StallProfile models aggregated CUPTI PC-sampling warp-stall reasons
// (cuptisample) collected over a kernel execution window.
type StallProfile struct {
	TotalSamples          uint64 `json:"total_samples"`
	StallMIOThrottle      uint64 `json:"stall_mio_throttle"`
	StallMathPipeThrottle uint64 `json:"stall_math_pipe_throttle"`
	StallLongScoreboard   uint64 `json:"stall_long_scoreboard"`
	StallShortScoreboard  uint64 `json:"stall_short_scoreboard"`
	StallBarrier          uint64 `json:"stall_barrier"`
	StallNotSelected      uint64 `json:"stall_not_selected"`
}

func (s StallProfile) effectiveTotal() uint64 {
	sum := s.StallMIOThrottle +
		s.StallMathPipeThrottle +
		s.StallLongScoreboard +
		s.StallShortScoreboard +
		s.StallBarrier +
		s.StallNotSelected
	if s.TotalSamples >= sum && s.TotalSamples > 0 {
		return s.TotalSamples
	}
	return sum
}

// RangeMetrics models CUPTI range profiler telemetry (cuptirange) and PTXAS/SASS
// resource allocation counters for a kernel launch.
type RangeMetrics struct {
	DurationNS           uint64  `json:"duration_ns"`
	AchievedTFLOPS       float64 `json:"achieved_tflops"`
	AchievedHBMBWGBps    float64 `json:"achieved_hbm_bw_gbps"`
	AchievedNVLinkBWGBps float64 `json:"achieved_nvlink_bw_gbps"`
	RegisterCount        int     `json:"register_count"`
	LocalSpillLoadBytes  int64   `json:"local_spill_load_bytes"`
	LocalSpillStoreBytes int64   `json:"local_spill_store_bytes"`
}

// BottleneckKind classifies the primary execution bottleneck diagnosed from
// warp-stall PC sampling and range counters.
type BottleneckKind string

const (
	BottleneckRegisterSpill   BottleneckKind = "register_spill"
	BottleneckMemoryLatency   BottleneckKind = "memory_latency"
	BottleneckMIOSMEMConflict BottleneckKind = "mio_smem_conflict"
	BottleneckBarrierSkew     BottleneckKind = "barrier_skew"
	BottleneckMathPipeBound   BottleneckKind = "math_pipe_bound"
	BottleneckBalanced        BottleneckKind = "balanced"
)

// Recommendation describes the e-graph schedule and warp-specialized register
// budget adjustments recommended by [DiagnoseBottleneck].
type Recommendation struct {
	Bottleneck                     BottleneckKind `json:"bottleneck"`
	SuggestedStages                int            `json:"suggested_stages"`
	SuggestedCTAGroup              int            `json:"suggested_cta_group"`
	EnableEarlyReleaseSyncRestrict bool           `json:"enable_early_release_sync_restrict"`
	EnableSparse2to4               bool           `json:"enable_sparse_2to4"`
	PreferSub4BitLUT               bool           `json:"prefer_sub_4bit_lut"`
	ProducerMaxRegs                int            `json:"producer_max_regs"`
	ConsumerMaxRegs                int            `json:"consumer_max_regs"`
	Rationale                      string         `json:"rationale"`
}

// DiagnoseBottleneck classifies the dominant runtime bottleneck for a kernel
// running on twin given PC-sampling stalls and range metrics, returning a
// concrete schedule [Recommendation].
func DiagnoseBottleneck(twin HardwareTwin, stalls StallProfile, metrics RangeMetrics) Recommendation {
	if metrics.LocalSpillLoadBytes > 0 || metrics.LocalSpillStoreBytes > 0 || metrics.RegisterCount > 255 {
		return Recommendation{
			Bottleneck:        BottleneckRegisterSpill,
			SuggestedStages:   2,
			SuggestedCTAGroup: 1,
			ProducerMaxRegs:   40,
			ConsumerMaxRegs:   168,
			Rationale: fmt.Sprintf(
				"local memory register spill detected (load=%dB, store=%dB, regs=%d); rebalance setmaxnreg to producer=40 consumer=168",
				metrics.LocalSpillLoadBytes, metrics.LocalSpillStoreBytes, metrics.RegisterCount,
			),
		}
	}

	total := stalls.effectiveTotal()
	if total == 0 {
		return Recommendation{
			Bottleneck:        BottleneckBalanced,
			SuggestedStages:   2,
			SuggestedCTAGroup: 1,
			ProducerMaxRegs:   48,
			ConsumerMaxRegs:   192,
			Rationale:         "no warp stall samples recorded; keep balanced baseline schedule",
		}
	}

	invTotal := 1.0 / float64(total)
	longSBRatio := float64(stalls.StallLongScoreboard) * invTotal
	mathRatio := float64(stalls.StallMathPipeThrottle) * invTotal
	mioRatio := float64(stalls.StallMIOThrottle+stalls.StallShortScoreboard) * invTotal
	barrierRatio := float64(stalls.StallBarrier) * invTotal

	// Find dominant stall category above the 35% significance threshold.
	if longSBRatio > 0.35 && longSBRatio >= mathRatio && longSBRatio >= mioRatio && longSBRatio >= barrierRatio {
		earlyRelease := twin.SupportsSyncRestrictReadA || twin.Target == "sm_107a" || twin.Target == "sm_107f"
		return Recommendation{
			Bottleneck:                     BottleneckMemoryLatency,
			SuggestedStages:                4,
			SuggestedCTAGroup:              1,
			EnableEarlyReleaseSyncRestrict: earlyRelease,
			PreferSub4BitLUT:               true,
			ProducerMaxRegs:                56,
			ConsumerMaxRegs:                184,
			Rationale: fmt.Sprintf(
				"long scoreboard stalls dominate (%.1f%%); deepen TMA pipeline to 4 stages, prefer sub-4-bit LUT weights, early-release sync_restrict=%v",
				longSBRatio*100.0, earlyRelease,
			),
		}
	}

	if mathRatio > 0.35 && mathRatio >= mioRatio && mathRatio >= barrierRatio {
		ctaGroup := 1
		if twin.TMEMColumns >= 512 {
			ctaGroup = 2
		}
		return Recommendation{
			Bottleneck:        BottleneckMathPipeBound,
			SuggestedStages:   3,
			SuggestedCTAGroup: ctaGroup,
			EnableSparse2to4:  true,
			ProducerMaxRegs:   48,
			ConsumerMaxRegs:   208,
			Rationale: fmt.Sprintf(
				"math pipe throttle dominates (%.1f%%); enable 2:4 structured sparsity and cta_group::%d",
				mathRatio*100.0, ctaGroup,
			),
		}
	}

	if mioRatio > 0.35 && mioRatio >= barrierRatio {
		return Recommendation{
			Bottleneck:        BottleneckMIOSMEMConflict,
			SuggestedStages:   2,
			SuggestedCTAGroup: 1,
			ProducerMaxRegs:   56,
			ConsumerMaxRegs:   176,
			Rationale: fmt.Sprintf(
				"MIO / short scoreboard stalls dominate (%.1f%%); pad/swizzle shared memory and stage TMEM loads",
				mioRatio*100.0,
			),
		}
	}

	if barrierRatio > 0.35 {
		return Recommendation{
			Bottleneck:        BottleneckBarrierSkew,
			SuggestedStages:   3,
			SuggestedCTAGroup: 1,
			ProducerMaxRegs:   48,
			ConsumerMaxRegs:   168,
			Rationale: fmt.Sprintf(
				"barrier stalls dominate (%.1f%%); decouple producer/consumer mbarrier phases and balance cluster arrival",
				barrierRatio*100.0,
			),
		}
	}

	return Recommendation{
		Bottleneck:        BottleneckBalanced,
		SuggestedStages:   3,
		SuggestedCTAGroup: 1,
		ProducerMaxRegs:   48,
		ConsumerMaxRegs:   192,
		Rationale:         "warp stalls balanced across pipes; no single stall exceeds 35% threshold",
	}
}

// ScheduleCandidate describes a candidate e-graph kernel schedule evaluated by
// [OnlineCostModel].
type ScheduleCandidate struct {
	Name             string  `json:"name"`
	Target           string  `json:"target"`
	TileM            int     `json:"tile_m"`
	TileN            int     `json:"tile_n"`
	TileK            int     `json:"tile_k"`
	Stages           int     `json:"stages"`
	CTAGroup         int     `json:"cta_group"`
	UseLUTB          bool    `json:"use_lut_b"`
	UseSparse2to4    bool    `json:"use_sparse_2to4"`
	UseEarlyRelease  bool    `json:"use_early_release"`
	AnalyticalCostUS float64 `json:"analytical_cost_us"`
	ProducerMaxRegs  int     `json:"producer_max_regs,omitempty"`  // setmaxnreg.dec target; 0 leaves the allocation
	ConsumerMaxRegs  int     `json:"consumer_max_regs,omitempty"`  // setmaxnreg.inc target; 0 leaves the allocation
	Swizzle128B      bool    `json:"swizzle_128b,omitempty"`       // 128-byte TMA shared memory swizzle
	DoubleBufferTMEM bool    `json:"double_buffer_tmem,omitempty"` // accumulators in TMEM columns [0,256) and [256,512)
}

type candidateRecord struct {
	Candidate      ScheduleCandidate `json:"candidate"`
	Observations   int               `json:"observations"`
	EMALatencyUS   float64           `json:"ema_latency_us"`
	LastStalls     StallProfile      `json:"last_stalls"`
	LastMetrics    RangeMetrics      `json:"last_metrics"`
	EMALongSBRate  float64           `json:"ema_long_sb_rate"`
	EMAMathRate    float64           `json:"ema_math_rate"`
	EMAMIORate     float64           `json:"ema_mio_rate"`
	EMABarrierRate float64           `json:"ema_barrier_rate"`
}

type shapeHistory struct {
	Candidates      map[string]*candidateRecord `json:"candidates"`
	WorstStalls     StallProfile                `json:"worst_stalls"`
	WorstMetrics    RangeMetrics                `json:"worst_metrics"`
	HasObservations bool                        `json:"has_observations"`
}

// OnlineCostModel maintains per-shape online CUPTI telemetry observations and
// combines analytical roofline latency estimates with Exponential Moving
// Average (EMA) measured latency and stall-driven adjustments.
type OnlineCostModel struct {
	mu     sync.RWMutex
	alpha  float64
	shapes map[string]*shapeHistory
}

// ProfileCache is an alias for [OnlineCostModel] used for JSON-serializable
// profile persistence across autotuning sessions.
type ProfileCache = OnlineCostModel

// NewOnlineCostModel returns an initialized [OnlineCostModel] with default EMA
// smoothing factor (0.5).
func NewOnlineCostModel() *OnlineCostModel {
	return &OnlineCostModel{
		alpha:  0.5,
		shapes: make(map[string]*shapeHistory),
	}
}

// NewProfileCache returns an initialized [ProfileCache].
func NewProfileCache() *ProfileCache {
	return NewOnlineCostModel()
}

func (m *OnlineCostModel) ensureInitLocked() {
	if m.alpha <= 0 || m.alpha > 1 {
		m.alpha = 0.5
	}
	if m.shapes == nil {
		m.shapes = make(map[string]*shapeHistory)
	}
}

// RecordObservation records a runtime CUPTI stall profile and range metrics
// observation for cand executing problem shape shapeKey.
func (m *OnlineCostModel) RecordObservation(shapeKey string, cand ScheduleCandidate, stalls StallProfile, metrics RangeMetrics) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.ensureInitLocked()

	sh, ok := m.shapes[shapeKey]
	if !ok {
		sh = &shapeHistory{
			Candidates: make(map[string]*candidateRecord),
		}
		m.shapes[shapeKey] = sh
	}

	measuredUS := float64(metrics.DurationNS) / 1000.0
	if measuredUS <= 0 {
		measuredUS = cand.AnalyticalCostUS
	}

	total := stalls.effectiveTotal()
	var longSB, mathRate, mioRate, barRate float64
	if total > 0 {
		inv := 1.0 / float64(total)
		longSB = float64(stalls.StallLongScoreboard) * inv
		mathRate = float64(stalls.StallMathPipeThrottle) * inv
		mioRate = float64(stalls.StallMIOThrottle+stalls.StallShortScoreboard) * inv
		barRate = float64(stalls.StallBarrier) * inv
	}

	rec, exists := sh.Candidates[cand.Name]
	if !exists || rec.Observations == 0 {
		rec = &candidateRecord{
			Candidate:      cand,
			Observations:   1,
			EMALatencyUS:   measuredUS,
			LastStalls:     stalls,
			LastMetrics:    metrics,
			EMALongSBRate:  longSB,
			EMAMathRate:    mathRate,
			EMAMIORate:     mioRate,
			EMABarrierRate: barRate,
		}
		sh.Candidates[cand.Name] = rec
	} else {
		a := m.alpha
		rec.Candidate = cand
		rec.Observations++
		rec.EMALatencyUS = a*measuredUS + (1.0-a)*rec.EMALatencyUS
		rec.LastStalls = stalls
		rec.LastMetrics = metrics
		rec.EMALongSBRate = a*longSB + (1.0-a)*rec.EMALongSBRate
		rec.EMAMathRate = a*mathRate + (1.0-a)*rec.EMAMathRate
		rec.EMAMIORate = a*mioRate + (1.0-a)*rec.EMAMIORate
		rec.EMABarrierRate = a*barRate + (1.0-a)*rec.EMABarrierRate
	}

	if !sh.HasObservations || isMoreSevere(stalls, metrics, sh.WorstStalls, sh.WorstMetrics) {
		sh.WorstStalls = stalls
		sh.WorstMetrics = metrics
		sh.HasObservations = true
	}
}

func isMoreSevere(stallsA StallProfile, metricsA RangeMetrics, stallsB StallProfile, metricsB RangeMetrics) bool {
	spillA := metricsA.LocalSpillLoadBytes + metricsA.LocalSpillStoreBytes
	spillB := metricsB.LocalSpillLoadBytes + metricsB.LocalSpillStoreBytes
	if spillA != spillB {
		return spillA > spillB
	}
	if (metricsA.RegisterCount > 255) != (metricsB.RegisterCount > 255) {
		return metricsA.RegisterCount > 255
	}
	return metricsA.DurationNS >= metricsB.DurationNS
}

// SelectBest evaluates candidates for shapeKey on twin by combining analytical
// roofline cost with EMA measured latency and stall-based adjustments
// (spillPenalty, longSBAdjustment, and mathPipeAdjustment).
func (m *OnlineCostModel) SelectBest(shapeKey string, twin HardwareTwin, candidates []ScheduleCandidate) (ScheduleCandidate, Recommendation, error) {
	if len(candidates) == 0 {
		return ScheduleCandidate{}, Recommendation{}, fmt.Errorf("select best: empty schedule candidates")
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	var sh *shapeHistory
	if m.shapes != nil {
		sh = m.shapes[shapeKey]
	}

	var rec Recommendation
	if sh != nil && sh.HasObservations {
		rec = DiagnoseBottleneck(twin, sh.WorstStalls, sh.WorstMetrics)
	} else {
		rec = DiagnoseBottleneck(twin, StallProfile{}, RangeMetrics{})
	}

	bestIdx := 0
	bestScore := math.Inf(1)

	for i, cand := range candidates {
		score := m.scoreCandidate(cand, twin, sh, rec)
		if score < bestScore {
			bestScore = score
			bestIdx = i
		}
	}

	chosen := candidates[bestIdx]
	if sh != nil {
		if obs, ok := sh.Candidates[chosen.Name]; ok && obs.Observations > 0 {
			chosenDiag := DiagnoseBottleneck(twin, obs.LastStalls, obs.LastMetrics)
			if rec.Bottleneck == BottleneckBalanced && chosenDiag.Bottleneck != BottleneckBalanced {
				rec = chosenDiag
			}
		}
	}
	return chosen, rec, nil
}

func (m *OnlineCostModel) scoreCandidate(cand ScheduleCandidate, twin HardwareTwin, sh *shapeHistory, shapeRec Recommendation) float64 {
	baseCost := cand.AnalyticalCostUS
	if baseCost <= 0 {
		baseCost = 1.0
	}

	var (
		spillPenalty       float64
		longSBAdjustment   float64
		mathPipeAdjustment float64
	)

	if sh != nil {
		if obs, ok := sh.Candidates[cand.Name]; ok && obs.Observations > 0 {
			// Blend analytical roofline cost with EMA observed latency (80% EMA, 20% analytical).
			baseCost = 0.8*obs.EMALatencyUS + 0.2*baseCost

			spillBytes := obs.LastMetrics.LocalSpillLoadBytes + obs.LastMetrics.LocalSpillStoreBytes
			if spillBytes > 0 || obs.LastMetrics.RegisterCount > 255 {
				spillPenalty = 50.0 + float64(spillBytes)*0.05
			}

			if obs.EMALongSBRate > 0.35 {
				longSBAdjustment += baseCost * obs.EMALongSBRate
			}
			if obs.EMAMathRate > 0.35 {
				mathPipeAdjustment += baseCost * obs.EMAMathRate * 0.5
			}
		}
	}

	// Apply bottleneck-guided feature adjustments so schedules addressing the
	// diagnosed bottleneck are promoted when baseline candidates stall.
	switch shapeRec.Bottleneck {
	case BottleneckMemoryLatency:
		if cand.Stages >= shapeRec.SuggestedStages {
			longSBAdjustment -= baseCost * 0.18
		}
		if cand.UseEarlyRelease && (twin.SupportsSyncRestrictReadA || twin.Target == "sm_107a" || twin.Target == "sm_107f") {
			longSBAdjustment -= baseCost * 0.15
		}
		if cand.UseLUTB && twin.SupportsLUTDecompressB {
			longSBAdjustment -= baseCost * 0.15
		} else if cand.UseLUTB {
			longSBAdjustment -= baseCost * 0.08
		}
	case BottleneckMathPipeBound:
		if cand.UseSparse2to4 {
			mathPipeAdjustment -= baseCost * 0.25
		}
		if cand.CTAGroup >= shapeRec.SuggestedCTAGroup && shapeRec.SuggestedCTAGroup > 1 {
			mathPipeAdjustment -= baseCost * 0.15
		}
	case BottleneckRegisterSpill:
		if cand.Stages <= shapeRec.SuggestedStages && cand.Stages > 0 {
			spillPenalty -= baseCost * 0.10
		}
	}

	return baseCost + spillPenalty + longSBAdjustment + mathPipeAdjustment
}

type serializedCostModel struct {
	Alpha  float64                  `json:"alpha"`
	Shapes map[string]*shapeHistory `json:"shapes"`
}

// MarshalJSON serializes the online cost model's accumulated profile telemetry
// to JSON.
func (m *OnlineCostModel) MarshalJSON() ([]byte, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	alpha := m.alpha
	if alpha <= 0 || alpha > 1 {
		alpha = 0.5
	}
	payload := serializedCostModel{
		Alpha:  alpha,
		Shapes: m.shapes,
	}
	if payload.Shapes == nil {
		payload.Shapes = make(map[string]*shapeHistory)
	}
	return json.Marshal(payload)
}

// UnmarshalJSON restores the online cost model's accumulated profile telemetry
// from JSON.
func (m *OnlineCostModel) UnmarshalJSON(data []byte) error {
	var payload serializedCostModel
	if err := json.Unmarshal(data, &payload); err != nil {
		return fmt.Errorf("unmarshal online cost model: %w", err)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if payload.Alpha <= 0 || payload.Alpha > 1 {
		m.alpha = 0.5
	} else {
		m.alpha = payload.Alpha
	}
	if payload.Shapes == nil {
		m.shapes = make(map[string]*shapeHistory)
	} else {
		m.shapes = payload.Shapes
	}
	return nil
}
