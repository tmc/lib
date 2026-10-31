package autotune

import (
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"sync"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// Canonical fleet GPU architectures supported by [FleetThroughputSurrogate].
var supportedFleetArchs = map[hopper.Arch]int{
	hopper.TargetSM90A:  0,
	hopper.TargetSM100A: 1,
	hopper.TargetSM120A: 2,
	hopper.TargetSM121A: 3,
	hopper.TargetSM107A: 4,
	hopper.TargetSM103A: 1,
	hopper.TargetSM107F: 4,
}

// FleetTelemetrySample captures the static PTXAS resource footprint, runtime
// CUPTI stall ratios, and held-out speedup / latency of a compiled plan or
// megakernel on a target architecture.
type FleetTelemetrySample struct {
	Arch                     hopper.Arch `json:"arch"`
	BatchM                   int         `json:"batch_m"`
	NumStages                int         `json:"num_stages"`
	RegsPerThread            int         `json:"regs_per_thread"`
	SharedBytes              int         `json:"shared_bytes"`
	SpillBytes               int         `json:"spill_bytes"`
	LongScoreboardStallRatio float64     `json:"long_scoreboard_stall_ratio"`
	BarrierStallRatio        float64     `json:"barrier_stall_ratio"`
	HoldoutSpeedup           float64     `json:"holdout_speedup"`
	ObservedLatencyUs        float64     `json:"observed_latency_us,omitempty"`
	SourceSHA256             [32]byte    `json:"source_sha256,omitempty"`
}

// Validate checks that s has a supported architecture and well-formed feature
// values.
func (s FleetTelemetrySample) Validate() error {
	arch := normalizeFleetArch(s.Arch)
	if _, ok := supportedFleetArchs[arch]; !ok {
		return fmt.Errorf("fleet telemetry sample: unsupported arch %q", s.Arch)
	}
	if s.BatchM <= 0 {
		return fmt.Errorf("fleet telemetry sample: invalid BatchM %d", s.BatchM)
	}
	if s.NumStages <= 0 {
		return fmt.Errorf("fleet telemetry sample: invalid NumStages %d", s.NumStages)
	}
	if s.RegsPerThread < 0 || s.RegsPerThread > 256 {
		return fmt.Errorf("fleet telemetry sample: invalid RegsPerThread %d", s.RegsPerThread)
	}
	if s.SharedBytes < 0 {
		return fmt.Errorf("fleet telemetry sample: invalid SharedBytes %d", s.SharedBytes)
	}
	if s.SpillBytes < 0 {
		return fmt.Errorf("fleet telemetry sample: invalid SpillBytes %d", s.SpillBytes)
	}
	if math.IsNaN(s.LongScoreboardStallRatio) || s.LongScoreboardStallRatio < 0 || s.LongScoreboardStallRatio > 1.0 {
		return fmt.Errorf("fleet telemetry sample: LongScoreboardStallRatio %v outside [0, 1]", s.LongScoreboardStallRatio)
	}
	if math.IsNaN(s.BarrierStallRatio) || s.BarrierStallRatio < 0 || s.BarrierStallRatio > 1.0 {
		return fmt.Errorf("fleet telemetry sample: BarrierStallRatio %v outside [0, 1]", s.BarrierStallRatio)
	}
	return nil
}

// PGOCacheEntry bundles a verified [PGOCacheKey], [PGOPromotionProof], cubin
// payload, and optional [FleetTelemetrySample] receipt for fleet-wide PGO
// exchange and surrogate fitting.
type PGOCacheEntry struct {
	Key            PGOCacheKey          `json:"key"`
	Proof          PGOPromotionProof    `json:"proof"`
	Cubin          []byte               `json:"cubin,omitempty"`
	Telemetry      FleetTelemetrySample `json:"telemetry,omitempty"`
	HoldoutSpeedup float64              `json:"holdout_speedup"`
}

func normalizeFleetArch(arch hopper.Arch) hopper.Arch {
	a := strings.ToLower(strings.TrimSpace(arch))
	switch a {
	case "sm_90", "sm_90a":
		return hopper.TargetSM90A
	case "sm_100", "sm_100a":
		return hopper.TargetSM100A
	case "sm_103", "sm_103a":
		return hopper.TargetSM103A
	case "sm_107", "sm_107a":
		return hopper.TargetSM107A
	case "sm_107f":
		return hopper.TargetSM107F
	case "sm_120", "sm_120a":
		return hopper.TargetSM120A
	case "sm_121", "sm_121a":
		return hopper.TargetSM121A
	default:
		return a
	}
}

// Feature layout for the 18-dimensional ridge-regularized linear + interaction
// digital-twin throughput surrogate:
//
//	0: intercept (1.0)
//	1: arch == sm_90a
//	2: arch == sm_100a / sm_103a
//	3: arch == sm_120a
//	4: arch == sm_121a
//	5: arch == sm_107a / sm_107f
//	6: arch TMEM capacity indicator (1.0 on sm_100a/sm_103a/sm_107a, 0.0 otherwise)
//	7: log2(max(BatchM, 1)) / 6.0
//	8: NumStages / 16.0
//	9: RegsPerThread / 255.0
//	10: SharedBytes / 163840.0
//	11: log1p(SpillBytes) / 8.0
//	12: LongScoreboardStallRatio
//	13: BarrierStallRatio
//	14: interaction: (RegsPerThread / 255.0) * (log1p(SpillBytes) / 8.0)
//	15: interaction: (NumStages / 16.0) * BarrierStallRatio
//	16: interaction: (log2(BatchM) / 6.0) * (1.0 - LongScoreboardStallRatio)
//	17: interaction: (SharedBytes / 163840.0) * (RegsPerThread / 255.0)
const surrogateFeatureDim = 18

func defaultPhysicsPriorWeights() [surrogateFeatureDim]float64 {
	return [surrogateFeatureDim]float64{
		0:  1.35,  // baseline speedup intercept
		1:  0.04,  // sm_90a
		2:  0.14,  // sm_100a
		3:  0.08,  // sm_120a
		4:  0.10,  // sm_121a
		5:  0.22,  // sm_107a
		6:  0.08,  // TMEM availability bonus
		7:  0.20,  // BatchM Tensor-Core utilization gain
		8:  -0.08, // stage count overhead
		9:  -0.12, // register pressure
		10: -0.10, // dynamic shared memory occupancy pressure
		11: -0.55, // local memory register spill penalty
		12: -0.45, // long scoreboard memory-stall penalty
		13: -0.35, // barrier stall penalty
		14: -0.40, // high regs x spill interaction penalty
		15: -0.25, // many stages x barrier stall interaction penalty
		16: 0.25,  // high BatchM x low memory-stall synergy
		17: -0.18, // high shared mem x high regs occupancy cliff
	}
}

func featurizeSample(s FleetTelemetrySample) ([surrogateFeatureDim]float64, error) {
	if err := s.Validate(); err != nil {
		return [surrogateFeatureDim]float64{}, err
	}
	arch := normalizeFleetArch(s.Arch)
	archBucket := supportedFleetArchs[arch]

	var x [surrogateFeatureDim]float64
	x[0] = 1.0
	x[1+archBucket] = 1.0
	if archBucket == 1 || archBucket == 4 {
		x[6] = 1.0 // TMEM-capable architecture (sm_100a, sm_103a, sm_107a)
	}

	batchNorm := math.Log2(float64(max(s.BatchM, 1))) / 6.0
	stagesNorm := float64(s.NumStages) / 16.0
	regsNorm := float64(s.RegsPerThread) / 255.0
	smemNorm := float64(s.SharedBytes) / 163840.0
	spillNorm := math.Log1p(float64(s.SpillBytes)) / 8.0
	longSB := s.LongScoreboardStallRatio
	barrier := s.BarrierStallRatio

	x[7] = batchNorm
	x[8] = stagesNorm
	x[9] = regsNorm
	x[10] = smemNorm
	x[11] = spillNorm
	x[12] = longSB
	x[13] = barrier

	// Feature interactions.
	x[14] = regsNorm * spillNorm
	x[15] = stagesNorm * barrier
	x[16] = batchNorm * (1.0 - longSB)
	x[17] = smemNorm * regsNorm
	return x, nil
}

// FleetThroughputSurrogate is a ridge-regularized linear and feature-interaction
// hardware digital-twin throughput surrogate trained on federated PGO receipts
// across GPU architectures (sm_90a, sm_100a, sm_120a, sm_121a, sm_107a).
type FleetThroughputSurrogate struct {
	mu          sync.RWMutex
	RidgeLambda float64
	weights     [surrogateFeatureDim]float64
	samples     []FleetTelemetrySample
}

// NewFleetThroughputSurrogate returns a [FleetThroughputSurrogate] initialized
// with physics-informed roofline priors and ridge regularization parameter
// lambda (defaulting to 0.05 when lambda <= 0).
func NewFleetThroughputSurrogate(lambda float64) *FleetThroughputSurrogate {
	if lambda <= 0 || math.IsNaN(lambda) || math.IsInf(lambda, 0) {
		lambda = 0.05
	}
	return &FleetThroughputSurrogate{
		RidgeLambda: lambda,
		weights:     defaultPhysicsPriorWeights(),
	}
}

// Samples returns a snapshot of the telemetry samples used to fit s.
func (s *FleetThroughputSurrogate) Samples() []FleetTelemetrySample {
	if s == nil {
		return nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return slices.Clone(s.samples)
}

// Weights returns a copy of the fitted surrogate parameter vector.
func (s *FleetThroughputSurrogate) Weights() []float64 {
	if s == nil {
		return nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]float64, surrogateFeatureDim)
	copy(out, s.weights[:])
	return out
}

// Fit fits the ridge-regularized linear / feature-interaction surrogate
//
//	(X^T X + lambda I) w = X^T y + lambda w_prior
//
// over samples, where y is each sample's HoldoutSpeedup (or 1000 / ObservedLatencyUs
// when HoldoutSpeedup <= 0).
func (s *FleetThroughputSurrogate) Fit(samples []FleetTelemetrySample) error {
	if s == nil {
		return errors.New("fleet throughput surrogate fit: nil surrogate")
	}
	if len(samples) == 0 {
		return errors.New("fleet throughput surrogate fit: empty sample slice")
	}

	lambda := s.RidgeLambda
	if lambda <= 0 || math.IsNaN(lambda) || math.IsInf(lambda, 0) {
		lambda = 0.05
	}
	prior := defaultPhysicsPriorWeights()

	var xtx [surrogateFeatureDim][surrogateFeatureDim]float64
	var xty [surrogateFeatureDim]float64
	for d := range surrogateFeatureDim {
		xtx[d][d] = lambda
		xty[d] = lambda * prior[d]
	}

	validSamples := make([]FleetTelemetrySample, 0, len(samples))
	for i, smp := range samples {
		feats, err := featurizeSample(smp)
		if err != nil {
			return fmt.Errorf("fleet throughput surrogate fit sample %d: %w", i, err)
		}
		target := smp.HoldoutSpeedup
		if target <= 0 && smp.ObservedLatencyUs > 0 {
			target = 100.0 / smp.ObservedLatencyUs
		}
		if math.IsNaN(target) || math.IsInf(target, 0) || target <= 0 {
			return fmt.Errorf("fleet throughput surrogate fit sample %d: invalid HoldoutSpeedup %v", i, smp.HoldoutSpeedup)
		}
		smp.HoldoutSpeedup = target
		validSamples = append(validSamples, smp)

		for r := range surrogateFeatureDim {
			xty[r] += feats[r] * target
			for c := range surrogateFeatureDim {
				xtx[r][c] += feats[r] * feats[c]
			}
		}
	}

	w, err := solveLinearSystem18(xtx, xty)
	if err != nil {
		return fmt.Errorf("fleet throughput surrogate solve: %w", err)
	}

	s.mu.Lock()
	s.weights = w
	s.samples = validSamples
	s.mu.Unlock()
	return nil
}

// IngestPGOCacheEntries extracts [FleetTelemetrySample] records from verified
// federated [PGOCacheEntry] receipts and fits the surrogate across all samples.
func (s *FleetThroughputSurrogate) IngestPGOCacheEntries(entries []PGOCacheEntry) error {
	if s == nil {
		return errors.New("fleet throughput surrogate ingest: nil surrogate")
	}
	if len(entries) == 0 {
		return errors.New("fleet throughput surrogate ingest: empty entries")
	}
	var samples []FleetTelemetrySample
	for i, e := range entries {
		smp := e.Telemetry
		if smp.Arch == "" {
			smp.Arch = e.Key.Arch
		}
		if smp.BatchM <= 0 {
			smp.BatchM = max(e.Key.ShapeBucket, 1)
		}
		if smp.NumStages <= 0 {
			smp.NumStages = 4
		}
		if smp.RegsPerThread <= 0 {
			smp.RegsPerThread = 128
		}
		if smp.HoldoutSpeedup <= 0 {
			if e.HoldoutSpeedup > 0 {
				smp.HoldoutSpeedup = e.HoldoutSpeedup
			} else {
				smp.HoldoutSpeedup = e.Proof.HoldoutRatio
			}
		}
		if smp.SourceSHA256 == ([32]byte{}) {
			smp.SourceSHA256 = e.Proof.SourceSHA256
		}
		if err := smp.Validate(); err != nil {
			return fmt.Errorf("ingest pgo cache entry %d: %w", i, err)
		}
		samples = append(samples, smp)
	}
	s.mu.RLock()
	combined := append(slices.Clone(s.samples), samples...)
	s.mu.RUnlock()
	return s.Fit(combined)
}

// IngestVerifiedCache loads all verified entries from cache, parses any
// embedded schedule telemetry metadata in ScheduleJSON, and fits the surrogate.
func (s *FleetThroughputSurrogate) IngestVerifiedCache(cache *VerifiedPGOCache) (int, error) {
	if s == nil {
		return 0, errors.New("ingest verified cache: nil surrogate")
	}
	if cache == nil {
		return 0, errors.New("ingest verified cache: nil cache")
	}
	keys, err := cache.List()
	if err != nil {
		return 0, err
	}
	var entries []PGOCacheEntry
	for _, k := range keys {
		proof, cubin, err := cache.LoadVerified(k)
		if err != nil {
			return 0, err
		}
		var tel FleetTelemetrySample
		if len(proof.ScheduleJSON) > 0 {
			_ = json.Unmarshal(proof.ScheduleJSON, &tel)
		}
		tel.Arch = k.Arch
		if tel.BatchM <= 0 {
			tel.BatchM = k.ShapeBucket
		}
		if tel.NumStages <= 0 {
			tel.NumStages = 4
		}
		if tel.RegsPerThread <= 0 {
			tel.RegsPerThread = 128
		}
		tel.HoldoutSpeedup = proof.HoldoutRatio
		tel.SourceSHA256 = proof.SourceSHA256
		entries = append(entries, PGOCacheEntry{
			Key:            k,
			Proof:          proof,
			Cubin:          cubin,
			Telemetry:      tel,
			HoldoutSpeedup: proof.HoldoutRatio,
		})
	}
	if len(entries) == 0 {
		return 0, nil
	}
	if err := s.IngestPGOCacheEntries(entries); err != nil {
		return 0, err
	}
	return len(entries), nil
}

// PredictSpeedup predicts the holdout speedup ratio for sample using the fitted
// ridge-regularized surrogate.
func (s *FleetThroughputSurrogate) PredictSpeedup(sample FleetTelemetrySample) (float64, error) {
	if s == nil {
		return 0, errors.New("predict speedup: nil surrogate")
	}
	feats, err := featurizeSample(sample)
	if err != nil {
		return 0, err
	}
	s.mu.RLock()
	w := s.weights
	if w == ([surrogateFeatureDim]float64{}) {
		w = defaultPhysicsPriorWeights()
	}
	s.mu.RUnlock()

	var pred float64
	for i := range surrogateFeatureDim {
		pred += w[i] * feats[i]
	}
	if pred < 0.05 {
		pred = 0.05
	}
	return pred, nil
}

// PredictLatencyUs predicts execution latency in microseconds given baselineLatencyUs.
func (s *FleetThroughputSurrogate) PredictLatencyUs(sample FleetTelemetrySample, baselineLatencyUs float64) (float64, error) {
	if baselineLatencyUs <= 0 || math.IsNaN(baselineLatencyUs) || math.IsInf(baselineLatencyUs, 0) {
		return 0, fmt.Errorf("predict latency: invalid baselineLatencyUs %v", baselineLatencyUs)
	}
	speedup, err := s.PredictSpeedup(sample)
	if err != nil {
		return 0, err
	}
	return baselineLatencyUs / speedup, nil
}

// ExtractPlanTelemetryFeatures derives a static [FleetTelemetrySample] feature
// vector `(Arch, BatchM, NumStages, RegsPerThread, SharedBytes, SpillBytes,
// LongScoreboardStallRatio, BarrierStallRatio)` from a candidate [hopper.Plan]
// targeting arch.
func ExtractPlanTelemetryFeatures(arch hopper.Arch, p *hopper.Plan) (FleetTelemetrySample, error) {
	normArch := normalizeFleetArch(arch)
	if _, ok := supportedFleetArchs[normArch]; !ok {
		return FleetTelemetrySample{}, fmt.Errorf("extract plan telemetry features: unsupported arch %q", arch)
	}
	if p == nil || len(p.Kernels) == 0 {
		return FleetTelemetrySample{}, errors.New("extract plan telemetry features: empty plan")
	}

	dispatchPlan, _ := hopper.SelectArchDispatchPlan(normArch, nil)
	maxRegsBudget := dispatchPlan.ConsumerMaxRegs
	if maxRegsBudget <= 0 {
		maxRegsBudget = 192
	}

	batchM := 1
	maxRegs := 64
	maxShared := 0
	spillBytes := 0
	var longSBAcc, barrierAcc float64

	for _, k := range p.Kernels {
		m := max(k.M, k.Rows, k.Batch, 1)
		if m > batchM {
			batchM = m
		}
		switch {
		case k.GEMM != nil:
			g := *k.GEMM
			normed, err := g.Norm()
			if err != nil {
				normed = g
			}
			regs := 96 + 16*normed.Prefetch
			if normed.BlackwellFP4 {
				regs += 12
			}
			if regs > maxRegs {
				maxRegs = regs
			}
			// Unprefetched global loads suffer higher long-scoreboard stall ratio.
			switch {
			case normed.Prefetch >= 3:
				longSBAcc += 0.14
			case normed.Prefetch == 2:
				longSBAcc += 0.22
			default:
				longSBAcc += 0.42
			}
			if normed.SplitK > 1 {
				barrierAcc += 0.18 * float64(normed.SplitK-1)
			} else {
				barrierAcc += 0.06
			}
			if regs > maxRegsBudget {
				spillBytes += (regs - maxRegsBudget) * 16
			}

		case k.Tile != nil:
			t := *k.Tile
			normed, err := t.Norm()
			if err != nil {
				normed = t
			}
			smem := normed.SharedBytes()
			if smem > maxShared {
				maxShared = smem
			}
			regs := 128
			if normed.BM >= 128 && normed.BN >= 128 {
				regs = 184
			}
			if normed.Stages >= 4 {
				regs += 24
			}
			if regs > maxRegs {
				maxRegs = regs
			}
			if normed.WGMMA || normed.BlackwellFP4 {
				longSBAcc += 0.12
			} else {
				longSBAcc += 0.36
			}
			if normed.WarpSpecialized {
				barrierAcc += 0.08
			} else {
				barrierAcc += 0.16
			}
			if regs > maxRegsBudget {
				spillBytes += (regs - maxRegsBudget) * 24
			}

		case k.Attention != nil:
			a := *k.Attention
			normed, err := a.Norm()
			if err != nil {
				normed = a
			}
			smem := normed.SharedBytes()
			if smem > maxShared {
				maxShared = smem
			}
			regs := 136
			if regs > maxRegs {
				maxRegs = regs
			}
			if normed.Stages >= 2 {
				longSBAcc += 0.18
			} else {
				longSBAcc += 0.35
			}
			barrierAcc += 0.12

		case k.Row != nil:
			regs := 64
			if regs > maxRegs {
				maxRegs = regs
			}
			longSBAcc += 0.20
			if k.Row.Cluster {
				barrierAcc += 0.15
			} else {
				barrierAcc += 0.05
			}

		default:
			longSBAcc += 0.22
			barrierAcc += 0.08
		}
	}

	nStages := len(p.Kernels)
	if maxRegs > 255 {
		spillBytes += (maxRegs - 255) * 32
		maxRegs = 255
	}

	longSBRatio := min(0.95, max(0.02, longSBAcc/float64(nStages)))
	// Unfused multi-stage plans also incur cross-stage synchronization overhead.
	barrierRatio := min(0.95, max(0.02, barrierAcc/float64(nStages)+0.015*float64(max(nStages-1, 0))))

	return FleetTelemetrySample{
		Arch:                     normArch,
		BatchM:                   batchM,
		NumStages:                nStages,
		RegsPerThread:            maxRegs,
		SharedBytes:              maxShared,
		SpillBytes:               spillBytes,
		LongScoreboardStallRatio: longSBRatio,
		BarrierStallRatio:        barrierRatio,
	}, nil
}

// RankCandidatesZeroShot evaluates candidates on arch without running GPU
// hardware benchmarks, returning the candidate indices sorted from highest
// predicted speedup to lowest along with their predicted speedup scores.
func (s *FleetThroughputSurrogate) RankCandidatesZeroShot(
	arch hopper.Arch,
	candidates []*hopper.Plan,
) ([]int, []float64, error) {
	if s == nil {
		return nil, nil, errors.New("rank candidates zero-shot: nil surrogate")
	}
	if len(candidates) == 0 {
		return nil, nil, errors.New("rank candidates zero-shot: empty candidates")
	}
	normArch := normalizeFleetArch(arch)
	if _, ok := supportedFleetArchs[normArch]; !ok {
		return nil, nil, fmt.Errorf("rank candidates zero-shot: unsupported arch %q", arch)
	}

	type scoredCandidate struct {
		idx   int
		score float64
	}
	scored := make([]scoredCandidate, len(candidates))
	for i, cand := range candidates {
		if cand == nil || len(cand.Kernels) == 0 {
			return nil, nil, fmt.Errorf("rank candidates zero-shot: candidate %d is empty", i)
		}
		sample, err := ExtractPlanTelemetryFeatures(normArch, cand)
		if err != nil {
			return nil, nil, fmt.Errorf("rank candidates zero-shot candidate %d: %w", i, err)
		}
		score, err := s.PredictSpeedup(sample)
		if err != nil {
			return nil, nil, fmt.Errorf("rank candidates zero-shot candidate %d: %w", i, err)
		}
		scored[i] = scoredCandidate{idx: i, score: score}
	}

	slices.SortFunc(scored, func(a, b scoredCandidate) int {
		if c := cmp.Compare(b.score, a.score); c != 0 {
			return c
		}
		return cmp.Compare(a.idx, b.idx)
	})

	indices := make([]int, len(scored))
	scores := make([]float64, len(scored))
	for i, sc := range scored {
		indices[i] = sc.idx
		scores[i] = sc.score
	}
	return indices, scores, nil
}

// MergeFederatedPGOCaches merges verified PGO cache entries from remote into
// local across a fleet of machines.
//
// An entry in remote is admitted into local if and only if:
//  1. It passes full cryptographic cubin SHA-256 and statistical holdout gate
//     verification via [VerifiedPGOCache.LoadVerified].
//  2. Its promotion proof carries a non-zero SourceSHA256 digest.
//  3. Either local does not yet contain the key (counted in added), or remote's
//     HoldoutRatio (HoldoutSpeedup) is strictly greater than local's existing
//     HoldoutRatio (counted in updated).
func MergeFederatedPGOCaches(local, remote *VerifiedPGOCache) (added, updated int, err error) {
	if local == nil || local.Dir == "" {
		return 0, 0, errors.New("merge federated pgo caches: invalid local cache")
	}
	if remote == nil || remote.Dir == "" {
		return 0, 0, errors.New("merge federated pgo caches: invalid remote cache")
	}

	remoteKeys, err := remote.List()
	if err != nil {
		return 0, 0, fmt.Errorf("merge federated pgo caches: list remote: %w", err)
	}

	for _, key := range remoteKeys {
		remoteProof, remoteCubin, err := remote.LoadVerified(key)
		if err != nil {
			return added, updated, fmt.Errorf("merge federated pgo caches: verify remote key %s: %w", key.CacheID(), err)
		}
		// Require a valid non-zero SHA-256 source digest.
		if remoteProof.SourceSHA256 == ([32]byte{}) {
			continue
		}

		localProof, _, localErr := local.LoadVerified(key)
		if localErr != nil {
			// Not present locally -> admit as new federated entry.
			if err := local.Store(key, remoteProof, remoteCubin); err != nil {
				return added, updated, fmt.Errorf("merge federated pgo caches: store new key %s: %w", key.CacheID(), err)
			}
			added++
			continue
		}

		// Present locally -> only overwrite if remote has strictly higher HoldoutRatio
		// and matching source digest when local also specifies a source digest.
		if localProof.SourceSHA256 != ([32]byte{}) && remoteProof.SourceSHA256 != localProof.SourceSHA256 {
			continue
		}
		if remoteProof.HoldoutRatio > localProof.HoldoutRatio {
			if err := local.Store(key, remoteProof, remoteCubin); err != nil {
				return added, updated, fmt.Errorf("merge federated pgo caches: update key %s: %w", key.CacheID(), err)
			}
			updated++
		}
	}

	return added, updated, nil
}

func solveLinearSystem18(a [surrogateFeatureDim][surrogateFeatureDim]float64, b [surrogateFeatureDim]float64) ([surrogateFeatureDim]float64, error) {
	var aug [surrogateFeatureDim][surrogateFeatureDim + 1]float64
	for r := range surrogateFeatureDim {
		for c := range surrogateFeatureDim {
			aug[r][c] = a[r][c]
		}
		aug[r][surrogateFeatureDim] = b[r]
	}

	for col := 0; col < surrogateFeatureDim; col++ {
		pivot := col
		maxAbs := math.Abs(aug[col][col])
		for r := col + 1; r < surrogateFeatureDim; r++ {
			if v := math.Abs(aug[r][col]); v > maxAbs {
				maxAbs = v
				pivot = r
			}
		}
		if maxAbs < 1e-12 {
			return [surrogateFeatureDim]float64{}, fmt.Errorf("singular matrix at column %d", col)
		}
		if pivot != col {
			aug[col], aug[pivot] = aug[pivot], aug[col]
		}
		invDiag := 1.0 / aug[col][col]
		for c := col; c <= surrogateFeatureDim; c++ {
			aug[col][c] *= invDiag
		}
		for r := 0; r < surrogateFeatureDim; r++ {
			if r == col {
				continue
			}
			factor := aug[r][col]
			if factor == 0 {
				continue
			}
			for c := col; c <= surrogateFeatureDim; c++ {
				aug[r][c] -= factor * aug[col][c]
			}
		}
	}

	var x [surrogateFeatureDim]float64
	for r := range surrogateFeatureDim {
		x[r] = aug[r][surrogateFeatureDim]
	}
	return x, nil
}
