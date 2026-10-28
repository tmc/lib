// SPDX-License-Identifier: Apache-2.0

package cudamtp

import (
	"container/heap"
	"errors"
	"fmt"
	"math"
	"sort"
)

const (
	// MaxWindowSize is the maximum number of runtime observations retained per tier
	// in ForwardRuntimeEstimator (matching csrc/engine/mtp_varlen.hpp).
	MaxWindowSize = 5000

	// MinSamplesForReliableEstimate is the minimum sample count required in a bucket
	// before its median latency estimate is treated as calibrated.
	MinSamplesForReliableEstimate = 10

	// SamplerMedianWindowSize is the sliding window length for SamplerRuntimeEstimator.
	SamplerMedianWindowSize = 9
)

var (
	// ErrInvalidVarlenConfig indicates invalid VarlenScheduleConfig parameters.
	ErrInvalidVarlenConfig = errors.New("cudamtp: invalid varlen schedule config")

	// ErrEmptySequenceBatch indicates an empty sequence slice passed to Schedule.
	ErrEmptySequenceBatch = errors.New("cudamtp: empty sequence batch")
)

type float64MinHeap []float64

func (h float64MinHeap) Len() int           { return len(h) }
func (h float64MinHeap) Less(i, j int) bool { return h[i] < h[j] }
func (h float64MinHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *float64MinHeap) Push(x any)        { *h = append(*h, x.(float64)) }
func (h *float64MinHeap) Pop() any {
	old := *h
	n := len(old)
	v := old[n-1]
	*h = old[:n-1]
	return v
}

type float64MaxHeap []float64

func (h float64MaxHeap) Len() int           { return len(h) }
func (h float64MaxHeap) Less(i, j int) bool { return h[i] > h[j] }
func (h float64MaxHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *float64MaxHeap) Push(x any)        { *h = append(*h, x.(float64)) }
func (h *float64MaxHeap) Pop() any {
	old := *h
	n := len(old)
	v := old[n-1]
	*h = old[:n-1]
	return v
}

// MedianTracker maintains an online running median using a max-heap (lo) for the
// lower half of samples and a min-heap (hi) for the upper half, matching the
// two-partition MedianTracker in DeepInfer's csrc/engine/mtp_varlen.hpp.
type MedianTracker struct {
	lo float64MaxHeap
	hi float64MinHeap
}

// NewMedianTracker returns an initialized MedianTracker.
// The zero value of MedianTracker is also ready to use.
func NewMedianTracker() *MedianTracker {
	return &MedianTracker{}
}

// Update inserts x into the two-heap tracker and rebalances the heaps so that
// len(lo) == len(hi) or len(lo) == len(hi)+1.
func (m *MedianTracker) Update(x float64) {
	if math.IsNaN(x) {
		return
	}
	if len(m.lo) == 0 || x <= m.lo[0] {
		heap.Push(&m.lo, x)
	} else {
		heap.Push(&m.hi, x)
	}
	m.rebalance()
}

func (m *MedianTracker) rebalance() {
	for len(m.lo) > len(m.hi)+1 {
		v := heap.Pop(&m.lo).(float64)
		heap.Push(&m.hi, v)
	}
	for len(m.lo) < len(m.hi) {
		v := heap.Pop(&m.hi).(float64)
		heap.Push(&m.lo, v)
	}
}

// Median returns the exact median of all observed values, or 0 if empty.
func (m *MedianTracker) Median() float64 {
	if len(m.lo) == 0 {
		return 0
	}
	if len(m.lo) > len(m.hi) {
		return m.lo[0]
	}
	return (m.lo[0] + m.hi[0]) * 0.5
}

// Count returns the total number of samples currently in the tracker.
func (m *MedianTracker) Count() int {
	return len(m.lo) + len(m.hi)
}

// Clear removes all samples from the tracker.
func (m *MedianTracker) Clear() {
	m.lo = m.lo[:0]
	m.hi = m.hi[:0]
}

// PerfPoint records the observed latency of one forward decode step at a given
// total query token count and total KV cache token count.
type PerfPoint struct {
	TotalSeqlensQ int
	TotalSeqlensK int
	DurationMS    float64
}

type bucketStats struct {
	duration MedianTracker
	seqlensK MedianTracker
	minK     int
	minKMS   float64
	maxK     int
	maxKMS   float64
}

func (b *bucketStats) update(pt PerfPoint) {
	n := b.duration.Count()
	b.duration.Update(pt.DurationMS)
	b.seqlensK.Update(float64(pt.TotalSeqlensK))
	if n == 0 || pt.TotalSeqlensK < b.minK {
		b.minK = pt.TotalSeqlensK
		b.minKMS = pt.DurationMS
	}
	if n == 0 || pt.TotalSeqlensK > b.maxK {
		b.maxK = pt.TotalSeqlensK
		b.maxKMS = pt.DurationMS
	}
}

// ForwardRuntimeEstimator estimates forward pass latency as a function of
// total query tokens (seqlensQ) and total KV cache tokens (seqlensK).
// It uses a two-tier (recent + old) sliding window of size MaxWindowSize,
// per-bucket MedianTrackers, and a dynamically estimated kvCacheSlope
// (dt / d_seqlens_k).
type ForwardRuntimeEstimator struct {
	buckets      []int
	recent       map[int]*bucketStats
	old          map[int]*bucketStats
	recentSlope  MedianTracker
	oldSlope     MedianTracker
	recentCount  int
	kvCacheSlope float64
}

// NewForwardRuntimeEstimator constructs a ForwardRuntimeEstimator for the given
// CUDA graph token buckets.
func NewForwardRuntimeEstimator(buckets []int) *ForwardRuntimeEstimator {
	sorted := make([]int, 0, len(buckets))
	for _, b := range buckets {
		if b > 0 {
			sorted = append(sorted, b)
		}
	}
	sort.Ints(sorted)
	dedup := sorted[:0]
	for _, b := range sorted {
		if len(dedup) == 0 || dedup[len(dedup)-1] != b {
			dedup = append(dedup, b)
		}
	}
	return &ForwardRuntimeEstimator{
		buckets: dedup,
		recent:  make(map[int]*bucketStats),
		old:     make(map[int]*bucketStats),
	}
}

func (e *ForwardRuntimeEstimator) ensureInit() {
	if e.recent == nil {
		e.recent = make(map[int]*bucketStats)
	}
	if e.old == nil {
		e.old = make(map[int]*bucketStats)
	}
}

func (e *ForwardRuntimeEstimator) mapBucket(seqlensQ int) int {
	if seqlensQ <= 1 {
		seqlensQ = 1
	}
	if len(e.buckets) == 0 {
		return seqlensQ
	}
	idx := sort.SearchInts(e.buckets, seqlensQ)
	if idx < len(e.buckets) {
		return e.buckets[idx]
	}
	return e.buckets[len(e.buckets)-1]
}

// Record adds a runtime observation to the recent window, rotating recent to old
// when recentCount reaches MaxWindowSize.
func (e *ForwardRuntimeEstimator) Record(pt PerfPoint) {
	if pt.TotalSeqlensQ <= 0 || pt.DurationMS <= 0 || math.IsNaN(pt.DurationMS) {
		return
	}
	e.ensureInit()

	if e.recentCount >= MaxWindowSize {
		e.old = e.recent
		e.recent = make(map[int]*bucketStats)
		e.oldSlope = e.recentSlope
		e.recentSlope = MedianTracker{}
		e.recentCount = 0
	}

	b := e.mapBucket(pt.TotalSeqlensQ)
	st, ok := e.recent[b]
	if !ok {
		st = &bucketStats{}
		e.recent[b] = st
	}
	st.update(pt)
	e.recentCount++

	if st.maxK > st.minK {
		slope := (st.maxKMS - st.minKMS) / float64(st.maxK-st.minK)
		if slope > 0 {
			e.recentSlope.Update(slope)
		}
	}
	if e.recentSlope.Count() > 0 {
		e.kvCacheSlope = math.Max(0, e.recentSlope.Median())
	} else if e.oldSlope.Count() > 0 {
		e.kvCacheSlope = math.Max(0, e.oldSlope.Median())
	}
}

// Update is an alias for Record.
func (e *ForwardRuntimeEstimator) Update(pt PerfPoint) {
	e.Record(pt)
}

// KVCacheSlope returns the estimated latency slope (ms per KV token).
func (e *ForwardRuntimeEstimator) KVCacheSlope() float64 {
	return e.kvCacheSlope
}

type calibratedBucket struct {
	bsz      int
	baseMS   float64
	medianK  float64
	reliable bool
}

func (e *ForwardRuntimeEstimator) collectBuckets() []calibratedBucket {
	seen := make(map[int]struct{})
	for b := range e.recent {
		seen[b] = struct{}{}
	}
	for b := range e.old {
		seen[b] = struct{}{}
	}
	keys := make([]int, 0, len(seen))
	for b := range seen {
		keys = append(keys, b)
	}
	sort.Ints(keys)

	var reliable, fallback []calibratedBucket
	for _, b := range keys {
		rec := e.recent[b]
		old := e.old[b]
		var dur, medK float64
		var count int
		if rec != nil && rec.duration.Count() >= MinSamplesForReliableEstimate {
			dur = rec.duration.Median()
			medK = rec.seqlensK.Median()
			count = rec.duration.Count()
		} else if old != nil && old.duration.Count() >= MinSamplesForReliableEstimate {
			if rec != nil && rec.duration.Count() > 0 {
				wRec := float64(rec.duration.Count())
				wOld := float64(old.duration.Count())
				dur = (rec.duration.Median()*wRec + old.duration.Median()*wOld) / (wRec + wOld)
				medK = (rec.seqlensK.Median()*wRec + old.seqlensK.Median()*wOld) / (wRec + wOld)
			} else {
				dur = old.duration.Median()
				medK = old.seqlensK.Median()
			}
			count = old.duration.Count()
		} else {
			var wRec, wOld float64
			if rec != nil && rec.duration.Count() > 0 {
				wRec = float64(rec.duration.Count())
				dur += rec.duration.Median() * wRec
				medK += rec.seqlensK.Median() * wRec
				count += rec.duration.Count()
			}
			if old != nil && old.duration.Count() > 0 {
				wOld = float64(old.duration.Count())
				dur += old.duration.Median() * wOld
				medK += old.seqlensK.Median() * wOld
				count += old.duration.Count()
			}
			if count > 0 {
				dur /= (wRec + wOld)
				medK /= (wRec + wOld)
			}
		}
		if count == 0 {
			continue
		}
		baseAtZeroK := dur - e.kvCacheSlope*medK
		if baseAtZeroK < 0.01*dur {
			baseAtZeroK = 0.01 * dur
		}
		cb := calibratedBucket{
			bsz:      b,
			baseMS:   baseAtZeroK,
			medianK:  medK,
			reliable: count >= MinSamplesForReliableEstimate,
		}
		if cb.reliable {
			reliable = append(reliable, cb)
		} else {
			fallback = append(fallback, cb)
		}
	}
	if len(reliable) > 0 {
		return reliable
	}
	return fallback
}

// Estimate returns the estimated forward pass duration in milliseconds for
// seqlensQ total query tokens and seqlensK total KV cache tokens, linearly
// interpolating between calibrated bucket medians and adding kvCacheSlope * seqlensK.
func (e *ForwardRuntimeEstimator) Estimate(seqlensQ int, seqlensK int) float64 {
	if seqlensQ <= 0 {
		seqlensQ = 1
	}
	if seqlensK < 0 {
		seqlensK = 0
	}
	e.ensureInit()

	cals := e.collectBuckets()
	if len(cals) == 0 {
		// Cold-start analytical fallback: sublinear base cost + small per-Q and per-K terms.
		return 1.0 + 0.05*float64(seqlensQ) + 0.0005*float64(seqlensK)
	}

	q := float64(seqlensQ)
	var baseMS float64
	if len(cals) == 1 {
		c := cals[0]
		if seqlensQ <= c.bsz {
			baseMS = c.baseMS
		} else {
			baseMS = c.baseMS * (q / float64(c.bsz))
		}
	} else if seqlensQ <= cals[0].bsz {
		baseMS = cals[0].baseMS
	} else if seqlensQ >= cals[len(cals)-1].bsz {
		last := cals[len(cals)-1]
		prev := cals[len(cals)-2]
		slopeQ := (last.baseMS - prev.baseMS) / float64(last.bsz-prev.bsz)
		if slopeQ < 0 {
			slopeQ = 0
		}
		baseMS = last.baseMS + slopeQ*float64(seqlensQ-last.bsz)
	} else {
		for i := 0; i < len(cals)-1; i++ {
			lo, hi := cals[i], cals[i+1]
			if seqlensQ >= lo.bsz && seqlensQ <= hi.bsz {
				t := float64(seqlensQ-lo.bsz) / float64(hi.bsz-lo.bsz)
				baseMS = lo.baseMS + t*(hi.baseMS-lo.baseMS)
				break
			}
		}
	}

	est := baseMS + e.kvCacheSlope*float64(seqlensK)
	if est <= 1e-6 {
		est = 1e-6
	}
	return est
}

// SamplerRuntimeEstimator estimates token sampling latency using a 9-sample
// median filter for base overhead plus linear scaling in totalTokens.
type SamplerRuntimeEstimator struct {
	samples      [SamplerMedianWindowSize]float64
	tokens       [SamplerMedianWindowSize]int
	size         int
	next         int
	medianMS     float64
	perTokenMS   float64
	baseOverhead float64
}

// NewSamplerRuntimeEstimator constructs a 9-sample median filter estimator for sampler latency.
func NewSamplerRuntimeEstimator() *SamplerRuntimeEstimator {
	return &SamplerRuntimeEstimator{
		baseOverhead: 0.05,
		perTokenMS:   0.002,
	}
}

// Record inserts a sampler runtime measurement for totalTokens into the 9-sample ring buffer.
func (s *SamplerRuntimeEstimator) Record(totalTokens int, durationMS float64) {
	if durationMS <= 0 || math.IsNaN(durationMS) {
		return
	}
	if totalTokens <= 0 {
		totalTokens = 1
	}
	s.samples[s.next] = durationMS
	s.tokens[s.next] = totalTokens
	s.next = (s.next + 1) % SamplerMedianWindowSize
	if s.size < SamplerMedianWindowSize {
		s.size++
	}

	buf := make([]float64, s.size)
	copy(buf, s.samples[:s.size])
	sort.Float64s(buf)
	mid := s.size / 2
	if s.size%2 == 0 {
		s.medianMS = (buf[mid-1] + buf[mid]) * 0.5
	} else {
		s.medianMS = buf[mid]
	}

	// Estimate linear token slope when multiple distinct token counts have been observed.
	minTok, maxTok := s.tokens[0], s.tokens[0]
	minMS, maxMS := s.samples[0], s.samples[0]
	for i := 1; i < s.size; i++ {
		if s.tokens[i] < minTok {
			minTok = s.tokens[i]
			minMS = s.samples[i]
		}
		if s.tokens[i] > maxTok {
			maxTok = s.tokens[i]
			maxMS = s.samples[i]
		}
	}
	if maxTok > minTok && maxMS > minMS {
		s.perTokenMS = (maxMS - minMS) / float64(maxTok-minTok)
		base := minMS - s.perTokenMS*float64(minTok)
		if base > 0 {
			s.baseOverhead = base
		} else {
			s.baseOverhead = s.medianMS * 0.5
		}
	} else {
		s.baseOverhead = s.medianMS
		s.perTokenMS = 0.001
	}
}

// Update records a sampler latency observation with a default token count of 1.
func (s *SamplerRuntimeEstimator) Update(durationMS float64) {
	s.Record(1, durationMS)
}

// Estimate returns the predicted sampler latency in milliseconds for totalTokens.
func (s *SamplerRuntimeEstimator) Estimate(totalTokens int) float64 {
	if totalTokens < 0 {
		totalTokens = 0
	}
	if s.size == 0 {
		base := s.baseOverhead
		if base <= 0 {
			base = 0.05
		}
		slope := s.perTokenMS
		if slope <= 0 {
			slope = 0.002
		}
		return base + slope*float64(totalTokens)
	}
	return s.baseOverhead + s.perTokenMS*float64(totalTokens)
}

// SequenceMTPState holds per-sequence state for confidence-scheduled variable-length MTP.
type SequenceMTPState struct {
	SeqID       int64
	ContextLen  int
	AcceptProbs []float64
	EWMAProb    float64
	FixedLen    int
}

// VarlenScheduleConfig configures the dynamic variable-length MTP scheduler.
type VarlenScheduleConfig struct {
	NumSpecSteps   int
	Buckets        []int
	EWMADecay      float64
	MinDynamicProb float64
	StepOverheadMS float64
}

// VarlenScheduleDecision reports the selected token budget, bucket size, per-sequence
// query lengths, and throughput metrics for one decode step.
type VarlenScheduleDecision struct {
	TargetTotalQ            int
	BucketBsz               int
	SeqlensQ                []int
	EstimatedForwardMS      float64
	EstimatedExpectedAccept float64
	ThroughputScore         float64
}

// VarlenScheduler dynamically schedules per-sequence speculative draft verification
// lengths to maximize expected accepted tokens per millisecond across CUDA graph buckets.
type VarlenScheduler struct {
	cfg        VarlenScheduleConfig
	forwardEst *ForwardRuntimeEstimator
	samplerEst *SamplerRuntimeEstimator
	ewmaBySeq  map[int64]float64
}

// NewVarlenScheduler validates cfg, applies defaults (EWMADecay = 0.9, MinDynamicProb = 0.3),
// and returns an initialized VarlenScheduler.
func NewVarlenScheduler(cfg VarlenScheduleConfig) (*VarlenScheduler, error) {
	if cfg.NumSpecSteps <= 0 {
		return nil, fmt.Errorf("%w: NumSpecSteps must be > 0, got %d", ErrInvalidVarlenConfig, cfg.NumSpecSteps)
	}
	if len(cfg.Buckets) == 0 {
		return nil, fmt.Errorf("%w: Buckets must not be empty", ErrInvalidVarlenConfig)
	}
	for _, b := range cfg.Buckets {
		if b <= 0 {
			return nil, fmt.Errorf("%w: bucket size must be > 0, got %d", ErrInvalidVarlenConfig, b)
		}
	}
	if cfg.EWMADecay == 0 {
		cfg.EWMADecay = 0.9
	}
	if cfg.EWMADecay < 0 || cfg.EWMADecay >= 1.0 {
		return nil, fmt.Errorf("%w: EWMADecay must be in [0, 1), got %v", ErrInvalidVarlenConfig, cfg.EWMADecay)
	}
	if cfg.MinDynamicProb == 0 {
		cfg.MinDynamicProb = 0.3
	}
	if cfg.MinDynamicProb < 0 || cfg.MinDynamicProb > 1.0 {
		return nil, fmt.Errorf("%w: MinDynamicProb must be in [0, 1], got %v", ErrInvalidVarlenConfig, cfg.MinDynamicProb)
	}
	if cfg.StepOverheadMS < 0 {
		return nil, fmt.Errorf("%w: StepOverheadMS must be >= 0, got %v", ErrInvalidVarlenConfig, cfg.StepOverheadMS)
	}

	sortedBuckets := make([]int, len(cfg.Buckets))
	copy(sortedBuckets, cfg.Buckets)
	sort.Ints(sortedBuckets)
	dedup := sortedBuckets[:0]
	for _, b := range sortedBuckets {
		if len(dedup) == 0 || dedup[len(dedup)-1] != b {
			dedup = append(dedup, b)
		}
	}
	cfg.Buckets = dedup

	return &VarlenScheduler{
		cfg:        cfg,
		forwardEst: NewForwardRuntimeEstimator(cfg.Buckets),
		samplerEst: NewSamplerRuntimeEstimator(),
		ewmaBySeq:  make(map[int64]float64),
	}, nil
}

// UpdateAcceptance updates the per-sequence EWMA acceptance probability using
// ewma = decay*old + (1-decay)*(accepted/draftLen).
func (s *VarlenScheduler) UpdateAcceptance(seqID int64, draftLen int, accepted int) {
	if draftLen <= 0 {
		return
	}
	if s.ewmaBySeq == nil {
		s.ewmaBySeq = make(map[int64]float64)
	}
	rate := float64(accepted) / float64(draftLen)
	if rate < 0 {
		rate = 0
	} else if rate > 1 {
		rate = 1
	}
	old := s.ewmaBySeq[seqID]
	s.ewmaBySeq[seqID] = s.cfg.EWMADecay*old + (1.0-s.cfg.EWMADecay)*rate
}

// EWMAProb returns the tracked EWMA acceptance probability for seqID.
func (s *VarlenScheduler) EWMAProb(seqID int64) (float64, bool) {
	v, ok := s.ewmaBySeq[seqID]
	return v, ok
}

// RecordStep updates the internal ForwardRuntimeEstimator with a completed step observation.
func (s *VarlenScheduler) RecordStep(pt PerfPoint) {
	if s.forwardEst == nil {
		s.forwardEst = NewForwardRuntimeEstimator(s.cfg.Buckets)
	}
	s.forwardEst.Record(pt)
}

// RecordSampler updates the internal SamplerRuntimeEstimator with a completed sampler observation.
func (s *VarlenScheduler) RecordSampler(totalTokens int, durationMS float64) {
	if s.samplerEst == nil {
		s.samplerEst = NewSamplerRuntimeEstimator()
	}
	s.samplerEst.Record(totalTokens, durationMS)
}

type candidateToken struct {
	seqIdx  int
	stepIdx int
	cumProb float64
}

// Schedule evaluates candidate token budgets across CUDA graph buckets using each
// sequence's EWMAProb (step N-2 signal) to choose TargetTotalQ maximizing expected
// accepted tokens per millisecond, and greedily allocates SeqlensQ up to 1+NumSpecSteps
// using step N-1 marginal cumulative acceptance probabilities.
func (s *VarlenScheduler) Schedule(seqs []SequenceMTPState) (VarlenScheduleDecision, error) {
	if len(seqs) == 0 {
		return VarlenScheduleDecision{}, ErrEmptySequenceBatch
	}
	numSeqs := len(seqs)
	maxPerSeq := 1 + s.cfg.NumSpecSteps
	maxTotalQ := numSeqs * maxPerSeq

	var totalK int
	baseTotalQ := 0
	minQ := make([]int, numSeqs)
	maxQ := make([]int, numSeqs)

	for i, seq := range seqs {
		if seq.ContextLen > 0 {
			totalK += seq.ContextLen
		}
		if seq.FixedLen > 0 {
			fl := seq.FixedLen
			if fl > maxPerSeq {
				fl = maxPerSeq
			}
			minQ[i] = fl
			maxQ[i] = fl
		} else {
			minQ[i] = 1
			maxQ[i] = maxPerSeq
		}
		baseTotalQ += minQ[i]
	}

	// Build step N-2 candidate queue using EWMAProb to select TargetTotalQ.
	var bszQueue []candidateToken
	for i, seq := range seqs {
		ewma := seq.EWMAProb
		if tracked, ok := s.ewmaBySeq[seq.SeqID]; ok {
			if ewma <= 0 {
				ewma = tracked
			} else {
				// Seed tracker if caller provided an explicit prior in SequenceMTPState.
				s.ewmaBySeq[seq.SeqID] = ewma
			}
		} else if ewma > 0 {
			if s.ewmaBySeq == nil {
				s.ewmaBySeq = make(map[int64]float64)
			}
			s.ewmaBySeq[seq.SeqID] = ewma
		}
		if ewma < 0 {
			ewma = 0
		} else if ewma > 1 {
			ewma = 1
		}
		cum := 1.0
		for step := 1; step < maxQ[i]; step++ {
			cum *= ewma
			if step < minQ[i] {
				continue
			}
			if ewma < s.cfg.MinDynamicProb {
				continue
			}
			bszQueue = append(bszQueue, candidateToken{
				seqIdx:  i,
				stepIdx: step,
				cumProb: cum,
			})
		}
	}

	sort.Slice(bszQueue, func(i, j int) bool {
		if bszQueue[i].cumProb != bszQueue[j].cumProb {
			return bszQueue[i].cumProb > bszQueue[j].cumProb
		}
		if bszQueue[i].seqIdx != bszQueue[j].seqIdx {
			return bszQueue[i].seqIdx < bszQueue[j].seqIdx
		}
		return bszQueue[i].stepIdx < bszQueue[j].stepIdx
	})

	// Build step N-1 candidate queue using cumulative product of AcceptProbs (or EWMAProb fallback).
	var allocQueue []candidateToken
	var baseExpectedAccept float64
	for i, seq := range seqs {
		ewma := seq.EWMAProb
		if ewma <= 0 {
			ewma = s.ewmaBySeq[seq.SeqID]
		}
		if ewma < 0 {
			ewma = 0
		} else if ewma > 1 {
			ewma = 1
		}
		baseExpectedAccept += 1.0 // verified/bonus base token
		cum := 1.0
		for step := 1; step < maxQ[i]; step++ {
			p := ewma
			if step-1 < len(seq.AcceptProbs) {
				p = seq.AcceptProbs[step-1]
			}
			if p < 0 {
				p = 0
			} else if p > 1 {
				p = 1
			}
			cum *= p
			if step < minQ[i] {
				baseExpectedAccept += cum
				continue
			}
			allocQueue = append(allocQueue, candidateToken{
				seqIdx:  i,
				stepIdx: step,
				cumProb: cum,
			})
		}
	}

	sort.Slice(allocQueue, func(i, j int) bool {
		if allocQueue[i].cumProb != allocQueue[j].cumProb {
			return allocQueue[i].cumProb > allocQueue[j].cumProb
		}
		if allocQueue[i].seqIdx != allocQueue[j].seqIdx {
			return allocQueue[i].seqIdx < allocQueue[j].seqIdx
		}
		return allocQueue[i].stepIdx < allocQueue[j].stepIdx
	})

	selectBucket := func(totalQ int) int {
		if len(s.cfg.Buckets) == 0 {
			return totalQ
		}
		idx := sort.SearchInts(s.cfg.Buckets, totalQ)
		if idx < len(s.cfg.Buckets) {
			return s.cfg.Buckets[idx]
		}
		return s.cfg.Buckets[len(s.cfg.Buckets)-1]
	}

	evalBudget := func(curBsz int, curExpected float64) (bucket int, fwdMS, score float64) {
		bucket = selectBucket(curBsz)
		fwdMS = s.forwardEst.Estimate(bucket, totalK+curBsz)
		sampMS := s.samplerEst.Estimate(curBsz)
		totalMS := fwdMS + sampMS + s.cfg.StepOverheadMS
		if totalMS <= 1e-9 {
			totalMS = 1e-9
		}
		score = curExpected / totalMS
		return bucket, fwdMS, score
	}

	bestBsz := baseTotalQ
	bestBucket, bestFwdMS, bestScore := evalBudget(baseTotalQ, baseExpectedAccept)
	curExpected := baseExpectedAccept
	curBsz := baseTotalQ

	maxBucket := maxTotalQ
	if len(s.cfg.Buckets) > 0 && s.cfg.Buckets[len(s.cfg.Buckets)-1] < maxBucket {
		maxBucket = s.cfg.Buckets[len(s.cfg.Buckets)-1]
	}

	for _, cand := range bszQueue {
		if curBsz+1 > maxBucket {
			break
		}
		curBsz++
		curExpected += cand.cumProb
		bkt, fwd, sc := evalBudget(curBsz, curExpected)
		if sc > bestScore {
			bestScore = sc
			bestBsz = curBsz
			bestBucket = bkt
			bestFwdMS = fwd
		}
	}

	// Greedily allocate extra draft tokens up to bestBsz using step N-1 marginal cumulative probabilities.
	seqlensQ := make([]int, numSeqs)
	copy(seqlensQ, minQ)
	extraNeeded := bestBsz - baseTotalQ
	expectedAccept := baseExpectedAccept
	for i := 0; i < len(allocQueue) && extraNeeded > 0; i++ {
		seqIdx := allocQueue[i].seqIdx
		if seqlensQ[seqIdx] < maxQ[seqIdx] {
			seqlensQ[seqIdx]++
			expectedAccept += allocQueue[i].cumProb
			extraNeeded--
		}
	}

	actualTotalQ := 0
	for _, q := range seqlensQ {
		actualTotalQ += q
	}
	bestBucket, bestFwdMS, bestScore = evalBudget(actualTotalQ, expectedAccept)

	return VarlenScheduleDecision{
		TargetTotalQ:            actualTotalQ,
		BucketBsz:               bestBucket,
		SeqlensQ:                seqlensQ,
		EstimatedForwardMS:      bestFwdMS,
		EstimatedExpectedAccept: expectedAccept,
		ThroughputScore:         bestScore,
	}, nil
}
