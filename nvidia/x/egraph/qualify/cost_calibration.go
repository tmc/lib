package qualify

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"sync"

	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/egraph/autotune"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/ptxasinfo"
)

// OpCostFunc computes the extraction cost of an e-graph node given the best
// costs of its child e-classes. OpCostFunc implements [egraph.CostModel].
type OpCostFunc func(n egraph.ENode, childCosts []float64) float64

// NodeCost implements [egraph.CostModel].
func (f OpCostFunc) NodeCost(n egraph.ENode, childCosts []float64) float64 {
	if f == nil {
		return (hopper.Cost{}).NodeCost(n, childCosts)
	}
	return f(n, childCosts)
}

// PTXASReport aliases [ptxasinfo.Report] for compile-time register and spill
// profile ingestion during cost calibration and PTXAS/SASS refinement loops.
type PTXASReport = ptxasinfo.Report

// CostObservation records a single empirical or compile-time observation for an
// operator and schedule configuration.
type CostObservation struct {
	Op                   string  `json:"op"`
	ScheduleKey          string  `json:"schedule_key"`
	AnalyticalCost       float64 `json:"analytical_cost"`
	MeasuredHoldoutRatio float64 `json:"measured_holdout_ratio"`
	RegistersPerThread   int     `json:"registers_per_thread"`
	TotalSpillBytes      int     `json:"total_spill_bytes"`
}

// PGOCacheEntry pairs a [autotune.PGOCacheKey] and hardware-verified
// [autotune.PGOPromotionProof] from a [autotune.VerifiedPGOCache] with optional
// operator, schedule, analytical cost, and PTXAS resource metadata.
type PGOCacheEntry struct {
	Key                autotune.PGOCacheKey       `json:"key"`
	Proof              autotune.PGOPromotionProof `json:"proof"`
	Op                 string                     `json:"op,omitempty"`
	ScheduleKey        string                     `json:"schedule_key,omitempty"`
	AnalyticalCost     float64                    `json:"analytical_cost,omitempty"`
	RegistersPerThread int                        `json:"registers_per_thread,omitempty"`
	TotalSpillBytes    int                        `json:"total_spill_bytes,omitempty"`
	PTXAS              *PTXASReport               `json:"-"`
}

const (
	defaultCalibrationLearningRate = 0.55
	defaultCalibrationPriorWeight  = 0.50
	minCalibratedMultiplier        = 0.05
	maxCalibratedMultiplier        = 50.0
)

type calibrationBucket struct {
	count         int
	emaLog        float64
	sumLog        float64
	lastAnalytic  float64
	lastHoldout   float64
	lastRegisters int
	lastSpills    int
}

func (b *calibrationBucket) update(targetMult, eta float64, obs CostObservation) {
	logTarget := math.Log(clampFloat(targetMult, minCalibratedMultiplier, maxCalibratedMultiplier))
	b.count++
	b.sumLog += logTarget
	if b.count == 1 {
		b.emaLog = logTarget
	} else {
		stepEta := max(eta/math.Sqrt(1.0+0.25*float64(b.count-1)), 0.15)
		b.emaLog = (1.0-stepEta)*b.emaLog + stepEta*logTarget
	}
	if obs.AnalyticalCost > 0 {
		b.lastAnalytic = obs.AnalyticalCost
	}
	if obs.MeasuredHoldoutRatio > 0 {
		b.lastHoldout = obs.MeasuredHoldoutRatio
	}
	if obs.RegistersPerThread > 0 {
		b.lastRegisters = obs.RegistersPerThread
	}
	if obs.TotalSpillBytes >= 0 {
		b.lastSpills = obs.TotalSpillBytes
	}
}

func (b *calibrationBucket) multiplier(priorWeight float64) float64 {
	if b == nil || b.count == 0 {
		return 1.0
	}
	if priorWeight <= 0 {
		priorWeight = defaultCalibrationPriorWeight
	}
	n := float64(b.count)
	// Combine damped online Multiplicative Weights (emaLog) with conjugate
	// Gaussian log-space posterior shrinkage (sumLog / (n + priorWeight)).
	bayesLog := b.sumLog / (n + priorWeight)
	dampedLog := (n / (n + priorWeight)) * b.emaLog
	blendedLog := 0.65*dampedLog + 0.35*bayesLog
	return clampFloat(math.Exp(blendedLog), minCalibratedMultiplier, maxCalibratedMultiplier)
}

// CalibratedCostModel learns empirical operator and schedule cost multipliers
// from hardware-verified [autotune.VerifiedPGOCache] entries and [PTXASReport]
// register/spill profiles using damped online Multiplicative Weights with
// Bayesian prior smoothing.
type CalibratedCostModel struct {
	mu           sync.RWMutex
	base         OpCostFunc
	learningRate float64
	priorWeight  float64
	costConfig   hopper.CostConfig
	outDType     hopper.DType

	opBuckets    map[string]*calibrationBucket
	schedBuckets map[string]*calibrationBucket
	jointBuckets map[string]*calibrationBucket
	observations []CostObservation
}

// NewCalibratedCostModel returns a [CalibratedCostModel] wrapping base. When
// base is nil, [hopper.Cost.NodeCost] is used as the uncalibrated baseline.
func NewCalibratedCostModel(base OpCostFunc) *CalibratedCostModel {
	if base == nil {
		base = (hopper.Cost{}).NodeCost
	}
	return &CalibratedCostModel{
		base:         base,
		learningRate: defaultCalibrationLearningRate,
		priorWeight:  defaultCalibrationPriorWeight,
		opBuckets:    make(map[string]*calibrationBucket),
		schedBuckets: make(map[string]*calibrationBucket),
		jointBuckets: make(map[string]*calibrationBucket),
	}
}

// Observe updates c with a single empirical or compile-time [CostObservation].
// Schedules with verified holdout wins (MeasuredHoldoutRatio > 1.0) receive
// lower effective cost multipliers (< 1.0), whereas schedules with local memory
// spills (TotalSpillBytes > 0) or high register pressure receive proportional
// multiplicative penalties (> 1.0).
func (c *CalibratedCostModel) Observe(obs CostObservation) {
	if c == nil {
		return
	}
	target := observationTargetMultiplier(obs)
	op := normalizeCalibrationKey(obs.Op)
	sched := normalizeCalibrationKey(obs.ScheduleKey)

	c.mu.Lock()
	defer c.mu.Unlock()

	if c.opBuckets == nil {
		c.opBuckets = make(map[string]*calibrationBucket)
	}
	if c.schedBuckets == nil {
		c.schedBuckets = make(map[string]*calibrationBucket)
	}
	if c.jointBuckets == nil {
		c.jointBuckets = make(map[string]*calibrationBucket)
	}
	eta := c.learningRate
	if eta <= 0 || eta > 1 {
		eta = defaultCalibrationLearningRate
	}

	c.observations = append(c.observations, obs)

	if op != "" {
		for _, alias := range opAliases(op) {
			b := c.opBuckets[alias]
			if b == nil {
				b = &calibrationBucket{}
				c.opBuckets[alias] = b
			}
			b.update(target, eta, obs)
		}
	}
	if sched != "" {
		b := c.schedBuckets[sched]
		if b == nil {
			b = &calibrationBucket{}
			c.schedBuckets[sched] = b
		}
		b.update(target, eta, obs)
	}
	if op != "" && sched != "" {
		for _, alias := range opAliases(op) {
			jKey := alias + "|" + sched
			b := c.jointBuckets[jKey]
			if b == nil {
				b = &calibrationBucket{}
				c.jointBuckets[jKey] = b
			}
			b.update(target, eta, obs)
		}
	}
}

// ObservePTXAS extracts per-function register and spill profiles from rep and
// records them as [CostObservation] entries for op and scheduleKey.
func (c *CalibratedCostModel) ObservePTXAS(op, scheduleKey string, rep PTXASReport) {
	if c == nil {
		return
	}
	var maxRegs, totalSpills int
	for _, fn := range rep.Functions() {
		if fn == nil {
			continue
		}
		if fn.Registers > maxRegs {
			maxRegs = fn.Registers
		}
		totalSpills += fn.SpillStores + fn.SpillLoads
	}
	c.Observe(CostObservation{
		Op:                 op,
		ScheduleKey:        scheduleKey,
		RegistersPerThread: maxRegs,
		TotalSpillBytes:    totalSpills,
	})
}

// IngestPGOCache ingests a slice of hardware-verified [PGOCacheEntry] records
// into c. Entries that carry a valid holdout ratio (> 0) or PTXAS resource
// profile update the corresponding operator and schedule multipliers.
func (c *CalibratedCostModel) IngestPGOCache(entries []PGOCacheEntry) {
	if c == nil {
		return
	}
	for _, entry := range entries {
		for _, obs := range observationsFromPGOCacheEntry(entry) {
			c.Observe(obs)
		}
	}
}

// IngestVerifiedPGOCache loads all verified entries from cache and ingests
// them into c.
func (c *CalibratedCostModel) IngestVerifiedPGOCache(cache *autotune.VerifiedPGOCache) error {
	entries, err := LoadPGOCacheEntries(cache)
	if err != nil {
		return err
	}
	c.IngestPGOCache(entries)
	return nil
}

// LoadPGOCacheEntries enumerates and cryptographically verifies all entries in
// cache, returning them as a slice of [PGOCacheEntry] values.
func LoadPGOCacheEntries(cache *autotune.VerifiedPGOCache) ([]PGOCacheEntry, error) {
	if cache == nil {
		return nil, errors.New("load pgo cache entries: nil cache")
	}
	keys, err := cache.List()
	if err != nil {
		return nil, err
	}
	entries := make([]PGOCacheEntry, 0, len(keys))
	for _, key := range keys {
		proof, _, err := cache.LoadVerified(key)
		if err != nil {
			return nil, err
		}
		entries = append(entries, PGOCacheEntry{
			Key:   key,
			Proof: proof,
		})
	}
	return entries, nil
}

// Multiplier returns the learned cost multiplier for (op, scheduleKey). A value
// below 1.0 indicates an empirically faster operator/schedule; a value above
// 1.0 indicates a penalized operator/schedule. Unobserved pairs return 1.0.
func (c *CalibratedCostModel) Multiplier(op, scheduleKey string) float64 {
	if c == nil {
		return 1.0
	}
	opKey := normalizeCalibrationKey(op)
	schedKey := normalizeCalibrationKey(scheduleKey)

	c.mu.RLock()
	defer c.mu.RUnlock()

	pw := c.priorWeight
	if pw <= 0 {
		pw = defaultCalibrationPriorWeight
	}

	if opKey != "" && schedKey != "" {
		if b, ok := c.jointBuckets[opKey+"|"+schedKey]; ok && b.count > 0 {
			return b.multiplier(pw)
		}
	}
	if schedKey != "" {
		if b, ok := c.schedBuckets[schedKey]; ok && b.count > 0 {
			if opKey != "" {
				if ob, ok := c.opBuckets[opKey]; ok && ob.count > 0 {
					return clampFloat(math.Sqrt(b.multiplier(pw)*ob.multiplier(pw)), minCalibratedMultiplier, maxCalibratedMultiplier)
				}
			}
			return b.multiplier(pw)
		}
	}
	if opKey != "" {
		if b, ok := c.opBuckets[opKey]; ok && b.count > 0 {
			return b.multiplier(pw)
		}
	}
	return 1.0
}

// OpCostFunc returns an [OpCostFunc] that scales each e-graph node's local
// operator cost by its calibrated empirical multiplier while preserving
// monotonicity with respect to child e-class costs.
func (c *CalibratedCostModel) OpCostFunc() OpCostFunc {
	return func(n egraph.ENode, childCosts []float64) float64 {
		baseFn := OpCostFunc((hopper.Cost{}).NodeCost)
		if c != nil && c.base != nil {
			baseFn = c.base
		}
		rawCost := baseFn.NodeCost(n, childCosts)

		var childSum, childMax float64
		for _, cc := range childCosts {
			if !math.IsNaN(cc) && !math.IsInf(cc, 0) && cc > 0 {
				childSum += cc
				if cc > childMax {
					childMax = cc
				}
			}
		}
		if math.IsNaN(rawCost) || math.IsInf(rawCost, 0) {
			return childSum + 1.0
		}

		mult := 1.0
		if c != nil {
			mult = c.nodeMultiplier(n)
		}

		if rawCost >= childSum {
			localCost := rawCost - childSum
			if localCost == 0 && n.Op != "" && len(childCosts) > 0 && mult != 1.0 {
				localCost = 1.0
			}
			return childSum + localCost*mult
		}
		localCost := max(rawCost-childMax, 0.0)
		return childMax + localCost*mult
	}
}

func (c *CalibratedCostModel) nodeMultiplier(n egraph.ENode) float64 {
	schedKey := nodeScheduleKey(n)
	op := n.Op
	if strings.HasPrefix(op, "fn:") {
		op = hopper.OpGEMM
	}
	m := c.Multiplier(op, schedKey)
	if n.Op == hopper.OpGEMM && len(n.Children) > 3 {
		// Check whether fused epilogue GEMM has a dedicated schedule observation.
		if fm := c.Multiplier(op, "fused_epilogue"); fm != 1.0 {
			m *= fm
		}
	}
	if n.Op == hopper.OpGEMM && len(n.Children) <= 3 {
		if um := c.Multiplier(op, "unfused_gemm"); um != 1.0 {
			m *= um
		}
	}
	return clampFloat(m, minCalibratedMultiplier, maxCalibratedMultiplier)
}

func nodeScheduleKey(n egraph.ENode) string {
	if strings.HasPrefix(n.Op, "fn:") {
		return strings.TrimPrefix(n.Op, "fn:")
	}
	if s, ok := n.Data.(string); ok && s != "" {
		return s
	}
	if n.Op == hopper.OpGEMM {
		if len(n.Children) > 3 {
			return "fused_epilogue"
		}
		return "unfused_gemm"
	}
	return ""
}

// ExtractCalibratedStructuralBeam extracts up to k structurally distinct
// [StructuralBeamCandidate] plans from g rooted at root, using cal both during
// e-graph extraction and when scoring lowered candidate schedules.
func ExtractCalibratedStructuralBeam(g *egraph.EGraph, root egraph.Id, k int, cal *CalibratedCostModel) ([]StructuralBeamCandidate, error) {
	if g == nil {
		return nil, errors.New("extract calibrated structural beam: nil e-graph")
	}
	if k <= 0 {
		return nil, nil
	}
	if cal == nil {
		cal = NewCalibratedCostModel(nil)
	}

	cal.mu.RLock()
	cfg := cal.costConfig
	outDType := cal.outDType
	cal.mu.RUnlock()

	normCfg, err := cfg.Norm()
	if err != nil {
		normCfg = cfg
	}
	if outDType == 0 {
		outDType = inferEGraphOutputDType(g)
	}
	name := "calibrated_beam"

	calOpCost := cal.OpCostFunc()
	baseDiversified := []diversifiedCostModel{
		{
			epilogueArityPenalty: 6000,
			nonIdentityPenalty:   4000,
			mapCost:              100,
			reduceCost:           150,
			modelFusedBonus:      -1500,
		},
		{
			epilogueArityPenalty: 2500,
			nonIdentityPenalty:   0,
			mapCost:              1800,
			reduceCost:           1800,
			modelFusedBonus:      500,
		},
		{
			epilogueArityPenalty: 0,
			nonIdentityPenalty:   0,
			mapCost:              4000,
			reduceCost:           4000,
			modelFusedBonus:      1000,
		},
	}

	models := make([]egraph.CostModel, 0, 2+len(baseDiversified)*2)
	models = append(models, calOpCost)
	for _, dm := range baseDiversified {
		dmCopy := dm
		models = append(models, NewCalibratedCostModel(dmCopy.NodeCost).withSharedStateFrom(cal).OpCostFunc())
	}
	models = append(models, hopper.Cost{})
	for _, dm := range baseDiversified {
		models = append(models, dm)
	}

	var exprs []*egraph.Expr
	seenExpr := make(map[string]bool)
	addExpr := func(e *egraph.Expr) {
		if e == nil {
			return
		}
		fp := e.Fingerprint()
		if seenExpr[fp] {
			return
		}
		seenExpr[fp] = true
		exprs = append(exprs, e)
	}

	harvestSnapshot := func(snap *egraph.EGraph) {
		if err := snap.RebuildChecked(); err != nil {
			return
		}
		limits := egraph.ExtractionLimits{MaxWork: 1 << 30, MaxNodes: 1 << 24, MaxDepth: 1 << 20}
		topKLimit := max(k*4, 12)
		for _, m := range models {
			if e, _, _, err := egraph.ExtractSegmented(context.Background(), snap, root, m, limits); err == nil {
				addExpr(e)
			}
			if cands, err := egraph.ExtractTopK(snap, root, topKLimit, m); err == nil {
				for _, c := range cands {
					addExpr(c.Expr)
				}
			}
		}
	}

	work := g.Clone()
	harvestSnapshot(work)

	for range 8 {
		if _, err := egraph.SaturateChecked(work, hopper.Rules(), egraph.SaturationOptions{MaxIterations: 16}); err != nil {
			return nil, fmt.Errorf("extract calibrated structural beam: saturate: %w", err)
		}
		harvestSnapshot(work)

		n, err := hopper.FuseEpilogues(work)
		if err != nil {
			return nil, fmt.Errorf("extract calibrated structural beam: fuse epilogues: %w", err)
		}
		harvestSnapshot(work)
		if n == 0 {
			break
		}
	}

	dtypes := []hopper.DType{outDType}
	if outDType != hopper.BF16 {
		dtypes = append(dtypes, hopper.BF16)
	}
	if outDType != hopper.F32 {
		dtypes = append(dtypes, hopper.F32)
	}

	var (
		candidates []StructuralBeamCandidate
		seenSig    = make(map[string]bool)
	)
	addCandidate := func(expr *egraph.Expr, p *hopper.Plan, mega *hopper.Megakernel, tag string) {
		if p == nil || len(p.Kernels) == 0 {
			return
		}
		if _, err := p.Source(); err != nil {
			return
		}
		if mega != nil {
			if _, err := mega.Source(); err != nil {
				return
			}
		}
		sig := ComputePlanSignature(p, mega)
		if sig == "" || seenSig[sig] {
			return
		}
		seenSig[sig] = true
		candName := fmt.Sprintf("%s_%s_%d", name, tag, len(candidates)+1)
		renamed := renamePlanForCandidate(p, candName)
		var megaCopy *hopper.Megakernel
		if mega != nil {
			if m, err := renamed.Megakernel(candName + "_mega"); err == nil {
				megaCopy = &m
			} else {
				mc := *mega
				mc.Name = candName + "_mega"
				megaCopy = &mc
			}
		}
		rawNs := estimateCandidateNs(renamed, megaCopy, normCfg)
		calNs := cal.calibrateCandidateNs(expr, renamed, megaCopy, tag, sig, rawNs, normCfg)
		candidates = append(candidates, StructuralBeamCandidate{
			Name:          candName,
			Expr:          expr,
			Plan:          renamed,
			Megakernel:    megaCopy,
			PlanSignature: sig,
			EstimatedNs:   calNs,
			CostConfig:    normCfg,
		})
	}

	for _, expr := range exprs {
		var compiled bool
		for _, dt := range dtypes {
			if scheduled, err := hopper.CompileWithCost(expr, name, dt, normCfg); err == nil {
				compiled = true
				addCandidate(expr, scheduled, nil, "sched")

				if len(scheduled.Kernels) >= 2 {
					if compat, mega, ok := promotePlanToMegakernel(scheduled, name+"_mega"); ok {
						addCandidate(expr, compat, &mega, "mega")
					}
				}
				if len(scheduled.Kernels) >= 2 {
					tailPlan := clonePlan(scheduled)
					if tailPlan.FuseTails() > 0 {
						addCandidate(expr, tailPlan, nil, "tail")
					}
				}
				for _, v := range calibratedScheduleVariants(scheduled) {
					addCandidate(expr, v.plan, nil, v.tag)
				}
			}
			if basePlan, err := hopper.Compile(expr, name, dt); err == nil {
				compiled = true
				addCandidate(expr, basePlan, nil, "base")
				if len(basePlan.Kernels) >= 2 {
					if compat, mega, ok := promotePlanToMegakernel(basePlan, name+"_base_mega"); ok {
						addCandidate(expr, compat, &mega, "base_mega")
					}
				}
			}
			if compiled {
				break
			}
		}
	}

	if len(candidates) == 0 {
		return nil, fmt.Errorf("extract calibrated structural beam: no valid lowerable candidates for root %d", root)
	}

	slices.SortStableFunc(candidates, compareBeamCandidates)
	if len(candidates) > k {
		candidates = candidates[:k]
	}
	return candidates, nil
}

type scheduleVariantPlan struct {
	plan *hopper.Plan
	tag  string
}

func calibratedScheduleVariants(p *hopper.Plan) []scheduleVariantPlan {
	if p == nil || len(p.Kernels) == 0 {
		return nil
	}
	var out []scheduleVariantPlan
	for i, step := range p.Kernels {
		switch {
		case step.Tile != nil:
			t := *step.Tile
			if t.WGMMA {
				cp := clonePlan(p)
				cand := t
				cand.WGMMA = false
				cand.WarpSpecialized = false
				if normed, err := cand.Norm(); err == nil {
					cp.Kernels[i].Tile = &normed
					out = append(out, scheduleVariantPlan{plan: cp, tag: "mma_sync"})
				}
			}
			if !t.Swizzle128B {
				cp := clonePlan(p)
				cand := t
				cand.Swizzle128B = true
				if normed, err := cand.Norm(); err == nil {
					cp.Kernels[i].Tile = &normed
					out = append(out, scheduleVariantPlan{plan: cp, tag: "swizzle128b"})
				}
			}
		case step.GEMM != nil:
			g := *step.GEMM
			if g.Prefetch > 0 {
				cp := clonePlan(p)
				cand := g
				cand.Prefetch = 0
				if normed, err := cand.Norm(); err == nil {
					cp.Kernels[i].GEMM = &normed
					out = append(out, scheduleVariantPlan{plan: cp, tag: "no_prefetch"})
				}
			} else {
				cp := clonePlan(p)
				cand := g
				cand.Prefetch = 2
				if normed, err := cand.Norm(); err == nil {
					cp.Kernels[i].GEMM = &normed
					out = append(out, scheduleVariantPlan{plan: cp, tag: "prefetch"})
				}
			}
		}
	}
	return out
}

func (c *CalibratedCostModel) withSharedStateFrom(src *CalibratedCostModel) *CalibratedCostModel {
	if src == nil {
		return c
	}
	src.mu.RLock()
	defer src.mu.RUnlock()
	c.learningRate = src.learningRate
	c.priorWeight = src.priorWeight
	c.costConfig = src.costConfig
	c.outDType = src.outDType
	c.opBuckets = src.opBuckets
	c.schedBuckets = src.schedBuckets
	c.jointBuckets = src.jointBuckets
	return c
}

func (c *CalibratedCostModel) calibrateCandidateNs(expr *egraph.Expr, plan *hopper.Plan, mega *hopper.Megakernel, tag, sig string, rawNs float64, cfg hopper.CostConfig) float64 {
	if c == nil || plan == nil {
		return rawNs
	}

	// 1. Scale each step's analytical latency contribution by its learned
	// operator/schedule multiplier.
	var stepWeightedNs float64
	for _, step := range plan.Kernels {
		single := &hopper.Plan{
			Name:    plan.Name,
			Inputs:  plan.Inputs,
			Temps:   plan.Temps,
			Output:  plan.Output,
			Kernels: []hopper.PlanKernel{step},
		}
		stepNs := single.EstimateNs(cfg)
		stepMult := c.stepMultiplier(step, tag)
		stepWeightedNs += stepNs * stepMult
	}

	if mega != nil && len(plan.Kernels) > 1 {
		launchNs := cfg.LaunchNs
		if launchNs <= 0 {
			launchNs = 2000.0
		}
		const deviceBarrierNs = 150.0
		savedPerTransition := max(launchNs-deviceBarrierNs, 0)
		stepWeightedNs = max(stepWeightedNs-float64(len(plan.Kernels)-1)*savedPerTransition, 1.0)
	}
	if stepWeightedNs <= 0 {
		stepWeightedNs = rawNs
	}

	// 2. Apply whole-candidate or structural topology multipliers when observed.
	candMult := c.candidateLevelMultiplier(expr, plan, mega, tag, sig)
	return max(stepWeightedNs*candMult, 1.0)
}

func (c *CalibratedCostModel) stepMultiplier(step hopper.PlanKernel, tag string) float64 {
	op := stepOpKind(step)
	keys := stepScheduleKeys(step, tag)

	c.mu.RLock()
	defer c.mu.RUnlock()

	pw := c.priorWeight
	if pw <= 0 {
		pw = defaultCalibrationPriorWeight
	}

	// Prefer the most specific joint (op, scheduleKey) or scheduleKey match.
	var matched []float64
	for _, k := range keys {
		nk := normalizeCalibrationKey(k)
		if nk == "" {
			continue
		}
		if op != "" {
			if b, ok := c.jointBuckets[normalizeCalibrationKey(op)+"|"+nk]; ok && b.count > 0 {
				matched = append(matched, b.multiplier(pw))
				continue
			}
		}
		if b, ok := c.schedBuckets[nk]; ok && b.count > 0 {
			matched = append(matched, b.multiplier(pw))
		}
	}
	if len(matched) > 0 {
		prod := 1.0
		for _, m := range matched {
			prod *= m
		}
		mean := math.Pow(prod, 1.0/float64(len(matched)))
		return clampFloat(mean, minCalibratedMultiplier, maxCalibratedMultiplier)
	}
	if op != "" {
		if b, ok := c.opBuckets[normalizeCalibrationKey(op)]; ok && b.count > 0 {
			return b.multiplier(pw)
		}
	}
	return 1.0
}

func (c *CalibratedCostModel) candidateLevelMultiplier(expr *egraph.Expr, plan *hopper.Plan, mega *hopper.Megakernel, tag, sig string) float64 {
	c.mu.RLock()
	defer c.mu.RUnlock()

	pw := c.priorWeight
	if pw <= 0 {
		pw = defaultCalibrationPriorWeight
	}

	var candKeys []string
	if sig != "" {
		candKeys = append(candKeys, sig)
	}
	if tag != "" {
		candKeys = append(candKeys, tag)
	}
	if plan != nil && plan.Name != "" {
		candKeys = append(candKeys, plan.Name)
	}
	if mega != nil {
		candKeys = append(candKeys, "megakernel")
		if mega.Name != "" {
			candKeys = append(candKeys, mega.Name)
		}
	} else if plan != nil && len(plan.Kernels) > 1 {
		candKeys = append(candKeys, "multi_launch", "unfused")
	}
	if plan != nil && len(plan.Kernels) == 1 {
		k0 := plan.Kernels[0]
		if (k0.GEMM != nil && !k0.GEMM.Epilogue.IsZero() && !k0.GEMM.Epilogue.IsIdentity()) ||
			(k0.Tile != nil && !k0.Tile.Epilogue.IsZero() && !k0.Tile.Epilogue.IsIdentity()) {
			candKeys = append(candKeys, "fused_epilogue", "fused")
		}
	}
	if expr != nil && expr.Op != "" {
		candKeys = append(candKeys, expr.Op)
	}

	mult := 1.0
	for _, ck := range candKeys {
		nck := normalizeCalibrationKey(ck)
		if nck == "" {
			continue
		}
		if b, ok := c.schedBuckets[nck]; ok && b.count > 0 {
			mult *= b.multiplier(pw)
		}
	}
	return clampFloat(mult, minCalibratedMultiplier, maxCalibratedMultiplier)
}

func stepScheduleKeys(k hopper.PlanKernel, tag string) []string {
	var keys []string
	add := func(s string) {
		s = strings.TrimSpace(s)
		if s != "" && !slices.Contains(keys, s) {
			keys = append(keys, s)
		}
	}
	if name := stepKernelName(k); name != "" {
		add(name)
	}
	if tag != "" {
		add(tag)
	}
	switch {
	case k.GEMM != nil:
		g := *k.GEMM
		if normed, err := g.Norm(); err == nil {
			g = normed
		}
		hasPrefetch := g.Prefetch > 0
		add(fmt.Sprintf("skinny_gemm:rpb=%d,warps=%d,splitk=%d,prefetch=%d,bfp4=%t,tail=%t",
			g.RowsPerBlock, g.Warps, g.SplitK, g.Prefetch, g.BlackwellFP4, g.Tail != nil))
		add(fmt.Sprintf("rpb=%d,warps=%d,splitk=%d,prefetch=%t", g.RowsPerBlock, g.Warps, g.SplitK, hasPrefetch))
		add(fmt.Sprintf("prefetch=%t", hasPrefetch))
		add(fmt.Sprintf("prefetch=%d", g.Prefetch))
		add(fmt.Sprintf("splitk=%d", g.SplitK))
		add(fmt.Sprintf("warps=%d", g.Warps))
		add(fmt.Sprintf("rpb=%d", g.RowsPerBlock))
		if g.BlackwellFP4 {
			add("blackwell_fp4")
		}
		if g.Tail != nil {
			add("tail_fused")
		}
		if !g.Epilogue.IsZero() && !g.Epilogue.IsIdentity() {
			add("fused_epilogue")
		} else {
			add("unfused_gemm")
		}

	case k.Tile != nil:
		t := *k.Tile
		if normed, err := t.Norm(); err == nil {
			t = normed
		}
		add(fmt.Sprintf("tile_gemm:bm=%d,bn=%d,stages=%d,wgmma=%t,swizzle128b=%t,warpspec=%t,bfp4=%t",
			t.BM, t.BN, t.Stages, t.WGMMA, t.Swizzle128B, t.WarpSpecialized, t.BlackwellFP4))
		add(fmt.Sprintf("bm=%d,bn=%d,stages=%d,wgmma=%t,swizzle128b=%t",
			t.BM, t.BN, t.Stages, t.WGMMA, t.Swizzle128B))
		add(fmt.Sprintf("wgmma=%t", t.WGMMA))
		add(fmt.Sprintf("swizzle128b=%t", t.Swizzle128B))
		add(fmt.Sprintf("stages=%d", t.Stages))
		if t.WGMMA {
			add("wgmma")
		} else {
			add("mma_sync")
		}
		if t.Swizzle128B {
			add("swizzle128b")
		}
		if !t.Epilogue.IsZero() && !t.Epilogue.IsIdentity() {
			add("fused_epilogue")
		} else {
			add("unfused_gemm")
		}

	case k.Row != nil:
		r := *k.Row
		add(fmt.Sprintf("row:cluster=%t", r.Cluster))
		add(fmt.Sprintf("cluster=%t", r.Cluster))

	case k.Attention != nil:
		a := *k.Attention
		if normed, err := a.Norm(); err == nil {
			a = normed
		}
		add(fmt.Sprintf("paged_attention:chunk=%d,stages=%d,wgmma=%t,swizzle=%t,splits=%t",
			a.Chunk, a.Stages, a.WGMMA, a.Swizzle, a.DeviceSplits))
		add(fmt.Sprintf("wgmma=%t", a.WGMMA))
		add(fmt.Sprintf("swizzle=%t", a.Swizzle))
	}
	return keys
}

func observationTargetMultiplier(obs CostObservation) float64 {
	target := 1.0
	if !math.IsNaN(obs.MeasuredHoldoutRatio) && !math.IsInf(obs.MeasuredHoldoutRatio, 0) && obs.MeasuredHoldoutRatio > 0 {
		ratio := clampFloat(obs.MeasuredHoldoutRatio, 0.10, 10.0)
		// Verified holdout wins (> 1.0x) reduce effective extraction cost;
		// regressions (< 1.0x) increase effective extraction cost.
		target /= math.Pow(ratio, 1.35)
	}
	if obs.TotalSpillBytes > 0 {
		spillBytes := float64(obs.TotalSpillBytes)
		spillPenalty := 1.0 + 0.35*math.Log1p(spillBytes/4.0) + 0.03*spillBytes
		target *= spillPenalty
	}
	if obs.RegistersPerThread > 128 {
		excess := float64(obs.RegistersPerThread-128) / 128.0
		regPenalty := 1.0 + 0.65*excess
		if obs.RegistersPerThread > 192 {
			regPenalty += 0.75 * float64(obs.RegistersPerThread-192) / 64.0
		}
		target *= regPenalty
	}
	return clampFloat(target, minCalibratedMultiplier, maxCalibratedMultiplier)
}

func observationsFromPGOCacheEntry(entry PGOCacheEntry) []CostObservation {
	ratio := entry.Proof.HoldoutRatio
	if ratio <= 0 && entry.Proof.ScreenRatio > 0 {
		ratio = entry.Proof.ScreenRatio
	}
	regs := entry.RegistersPerThread
	spills := entry.TotalSpillBytes
	if entry.PTXAS != nil {
		for _, fn := range entry.PTXAS.Functions() {
			if fn == nil {
				continue
			}
			if fn.Registers > regs {
				regs = fn.Registers
			}
			spills += fn.SpillStores + fn.SpillLoads
		}
	}

	op := strings.TrimSpace(entry.Op)
	schedKey := strings.TrimSpace(entry.ScheduleKey)
	analytic := entry.AnalyticalCost

	var extraKeys []string
	if len(entry.Proof.ScheduleJSON) > 0 {
		var meta struct {
			Op                 string  `json:"op"`
			ScheduleKey        string  `json:"schedule_key"`
			PlanSignature      string  `json:"plan_signature"`
			Tag                string  `json:"tag"`
			AnalyticalCost     float64 `json:"analytical_cost"`
			RegistersPerThread int     `json:"registers_per_thread"`
			TotalSpillBytes    int     `json:"total_spill_bytes"`
		}
		if err := json.Unmarshal(entry.Proof.ScheduleJSON, &meta); err == nil {
			if op == "" && meta.Op != "" {
				op = meta.Op
			}
			if schedKey == "" && meta.ScheduleKey != "" {
				schedKey = meta.ScheduleKey
			}
			if analytic <= 0 && meta.AnalyticalCost > 0 {
				analytic = meta.AnalyticalCost
			}
			if regs <= 0 && meta.RegistersPerThread > 0 {
				regs = meta.RegistersPerThread
			}
			if spills <= 0 && meta.TotalSpillBytes > 0 {
				spills = meta.TotalSpillBytes
			}
			if meta.PlanSignature != "" {
				extraKeys = append(extraKeys, meta.PlanSignature)
			}
			if meta.Tag != "" {
				extraKeys = append(extraKeys, meta.Tag)
			}
		}
	}

	if schedKey == "" {
		if entry.Key.PlanKey != "" {
			schedKey = entry.Key.PlanKey
		} else if entry.Proof.CandidateName != "" {
			schedKey = entry.Proof.CandidateName
		}
	} else {
		if entry.Key.PlanKey != "" && entry.Key.PlanKey != schedKey {
			extraKeys = append(extraKeys, entry.Key.PlanKey)
		}
	}
	if entry.Proof.CandidateName != "" && entry.Proof.CandidateName != schedKey {
		extraKeys = append(extraKeys, entry.Proof.CandidateName)
	}

	if op == "" {
		op = inferOpFromScheduleString(schedKey + " " + entry.Key.PlanKey + " " + entry.Proof.CandidateName)
	}

	baseObs := CostObservation{
		Op:                   op,
		ScheduleKey:          schedKey,
		AnalyticalCost:       analytic,
		MeasuredHoldoutRatio: ratio,
		RegistersPerThread:   regs,
		TotalSpillBytes:      spills,
	}
	out := []CostObservation{baseObs}
	for _, ek := range extraKeys {
		ek = strings.TrimSpace(ek)
		if ek == "" || ek == schedKey {
			continue
		}
		out = append(out, CostObservation{
			Op:                   op,
			ScheduleKey:          ek,
			AnalyticalCost:       analytic,
			MeasuredHoldoutRatio: ratio,
			RegistersPerThread:   regs,
			TotalSpillBytes:      spills,
		})
	}
	return out
}

func inferOpFromScheduleString(s string) string {
	low := strings.ToLower(s)
	switch {
	case strings.Contains(low, "tile_gemm") || strings.Contains(low, "wgmma"):
		return "tile_gemm"
	case strings.Contains(low, "skinny_gemm") || strings.Contains(low, "prefetch"):
		return "skinny_gemm"
	case strings.Contains(low, "grouped_gemm"):
		return hopper.OpGroupedGEMM
	case strings.Contains(low, "paged_attn") || strings.Contains(low, "paged_attention"):
		return hopper.OpPagedAttn
	case strings.Contains(low, "row") || strings.Contains(low, "rmsnorm"):
		return "row"
	case strings.Contains(low, "gemm"):
		return hopper.OpGEMM
	default:
		return ""
	}
}

func opAliases(op string) []string {
	op = normalizeCalibrationKey(op)
	switch op {
	case "gemm", "skinny_gemm", "tile_gemm":
		return uniqueStrings([]string{op, "gemm", "skinny_gemm", "tile_gemm"})
	case "map", "row", "reduce_sum", "reduce_max":
		return uniqueStrings([]string{op, "row", "map"})
	case "paged_attn", "paged_attention", "paged_attn_gated":
		return uniqueStrings([]string{op, "paged_attn", "paged_attention"})
	case "grouped_gemm", "grouped_matmul", "grouped_gemm_glu", "grouped_gemm_sumsq":
		return uniqueStrings([]string{op, "grouped_gemm"})
	default:
		return []string{op}
	}
}

func uniqueStrings(in []string) []string {
	var out []string
	for _, s := range in {
		if s != "" && !slices.Contains(out, s) {
			out = append(out, s)
		}
	}
	return out
}

func normalizeCalibrationKey(s string) string {
	return strings.ToLower(strings.TrimSpace(s))
}

func inferEGraphOutputDType(g *egraph.EGraph) hopper.DType {
	if g == nil {
		return hopper.BF16
	}
	var sawF32, sawBF16 bool
	for _, cl := range g.Classes() {
		if cl == nil {
			continue
		}
		for _, n := range cl.Nodes {
			if t, ok := n.Data.(hopper.Tensor); ok {
				switch t.DType {
				case hopper.BF16:
					sawBF16 = true
				case hopper.F32:
					sawF32 = true
				}
			}
		}
	}
	if sawBF16 {
		return hopper.BF16
	}
	if sawF32 {
		return hopper.F32
	}
	return hopper.BF16
}

func clampFloat(v, lo, hi float64) float64 {
	if math.IsNaN(v) {
		return 1.0
	}
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
