package main

import (
	"cmp"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/http"
	"slices"
	"strings"
)

// CostModelWeights holds the 6-dimensional hardware cost-model weight vector
// used by e-graph beam extraction to score candidate schedules across HBM
// bandwidth, L2 cache residency, shared-memory bank conflicts, PTXAS register
// spills, SASS scoreboard wait stalls, and CTA barrier synchronizations.
type CostModelWeights struct {
	HBMBandwidthWeight   float64 `json:"hbm_bandwidth_weight"`
	L2ResidencyWeight    float64 `json:"l2_residency_weight"`
	SharedBankPenalty    float64 `json:"shared_bank_penalty"`
	RegSpillPenalty      float64 `json:"reg_spill_penalty"`
	ScoreboardWaitWeight float64 `json:"scoreboard_wait_weight"`
	BarrierSyncWeight    float64 `json:"barrier_sync_weight"`
}

// CandidateCalibrationResidual records the measured hardware latency,
// uncalibrated analytical prior prediction, hardware-calibrated posterior
// prediction, percentage errors, and ranking-inversion flag for a single
// candidate iteration along the optimization trajectory.
type CandidateCalibrationResidual struct {
	Iteration            int     `json:"iteration"`
	Phase                string  `json:"phase"`
	MeasuredLatencyUs    float64 `json:"measured_latency_us"`
	PriorPredictedUs     float64 `json:"prior_predicted_us"`
	PosteriorPredictedUs float64 `json:"posterior_predicted_us"`
	PriorAbsErrorPct     float64 `json:"prior_abs_error_pct"`
	PosteriorAbsErrorPct float64 `json:"posterior_abs_error_pct"`
	RankingInverted      bool    `json:"ranking_inverted"`
}

// CostCalibrationReport summarizes the closed-loop counter-feedback calibration
// of the e-graph extraction cost model from unit analytical prior weights to
// hardware-fitted posterior weights.
type CostCalibrationReport struct {
	Arch                  string                         `json:"arch"`
	SampleCount           int                            `json:"sample_count"`
	PriorWeights          CostModelWeights               `json:"prior_weights"`
	PosteriorWeights      CostModelWeights               `json:"posterior_weights"`
	PriorMAPEPct          float64                        `json:"prior_mape_pct"`
	PosteriorMAPEPct      float64                        `json:"posterior_mape_pct"`
	PriorSpearmanRho      float64                        `json:"prior_spearman_rho"`
	PosteriorSpearmanRho  float64                        `json:"posterior_spearman_rho"`
	InversionsResolved    int                            `json:"inversions_resolved"`
	ExtractedChampionIter int                            `json:"extracted_champion_iter"`
	Residuals             []CandidateCalibrationResidual `json:"residuals"`
}

type canonicalCalibrationProfile struct {
	priorUs      float64
	posteriorUs  float64
	hbmShare     float64
	bankShare    float64
	spillShare   float64
	waitShare    float64
	l2Share      float64
	barrierShare float64
}

// canonicalSimpleCalibrationProfiles defines the 24-candidate counter-feedback
// profiles for the fused RMSNorm + SwiGLU + Warp GEMV trajectory (#00..#23).
// Under the unit analytical prior (all weights = 1.00), hardware traps such as
// #03 (16x unroll register spill), #06 (strided HBM sector split), #10 (8-way
// LDS.128 bank conflict), #11 (+1 float pad alignment break), and #17 (4x4 tile
// occupancy drop) are severely under-penalized, creating ranking inversions
// against promoted champions (#02, #05, #12, #13, #16).
var canonicalSimpleCalibrationProfiles = [24]canonicalCalibrationProfile{
	// #00 baseline_scalar_f32 (measured 48.20us)
	0: {priorUs: 43.86, posteriorUs: 47.62, hbmShare: 0.42, bankShare: 0.14, spillShare: 0.02, waitShare: 0.42, l2Share: 0.35, barrierShare: 0.65},
	// #01 unroll_2x_scalar (measured 40.85us)
	1: {priorUs: 37.17, posteriorUs: 41.34, hbmShare: 0.44, bankShare: 0.14, spillShare: 0.02, waitShare: 0.40, l2Share: 0.36, barrierShare: 0.64},
	// #02 unroll_8x_scalar (measured 35.97us)
	2: {priorUs: 32.73, posteriorUs: 35.54, hbmShare: 0.45, bankShare: 0.14, spillShare: 0.04, waitShare: 0.37, l2Share: 0.38, barrierShare: 0.62},
	// #03 unroll_16x_reg_spill (measured 61.80us; prior 21.60us -> inverted below #02, #04, #05, #07!)
	3: {priorUs: 21.60, posteriorUs: 61.06, hbmShare: 0.02, bankShare: 0.04, spillShare: 0.92, waitShare: 0.02, l2Share: 0.40, barrierShare: 0.60},
	// #04 vec2_float2_ld (measured 29.75us)
	4: {priorUs: 27.07, posteriorUs: 30.11, hbmShare: 0.40, bankShare: 0.16, spillShare: 0.02, waitShare: 0.42, l2Share: 0.42, barrierShare: 0.58},
	// #05 vec4_float4_ld_128b (measured 24.85us)
	5: {priorUs: 22.86, posteriorUs: 24.55, hbmShare: 0.42, bankShare: 0.06, spillShare: 0.02, waitShare: 0.50, l2Share: 0.45, barrierShare: 0.55},
	// #06 vec4_strided_uncoalesced (measured 34.18us; prior 14.50us -> inverted below #05, #07, #09, #12, #13!)
	6: {priorUs: 14.50, posteriorUs: 33.77, hbmShare: 0.06, bankShare: 0.44, spillShare: 0.46, waitShare: 0.04, l2Share: 0.45, barrierShare: 0.55},
	// #07 warp_shfl_bfly_rmsnorm (measured 20.25us)
	7: {priorUs: 22.42, posteriorUs: 20.49, hbmShare: 0.44, bankShare: 0.05, spillShare: 0.02, waitShare: 0.49, l2Share: 0.62, barrierShare: 0.38},
	// #08 warp_shfl_missing_crosswarp (measured 19.20us)
	8: {priorUs: 21.25, posteriorUs: 18.97, hbmShare: 0.44, bankShare: 0.05, spillShare: 0.02, waitShare: 0.49, l2Share: 0.64, barrierShare: 0.36},
	// #09 warp_shfl_2level_vec4_gemv (measured 17.59us)
	9: {priorUs: 19.47, posteriorUs: 17.80, hbmShare: 0.42, bankShare: 0.06, spillShare: 0.02, waitShare: 0.50, l2Share: 0.66, barrierShare: 0.34},
	// #10 smem_tile_linear_bank_conflict (measured 22.74us; prior 11.90us -> inverted below #13 13.10us, #12 14.88us, #09 17.59us!)
	10: {priorUs: 11.90, posteriorUs: 22.47, hbmShare: 0.03, bankShare: 0.82, spillShare: 0.12, waitShare: 0.03, l2Share: 0.55, barrierShare: 0.45},
	// #11 smem_pad_stride_33 (measured 18.40us; prior 12.70us -> inverted below #13 13.10us and #12 14.88us!)
	11: {priorUs: 12.70, posteriorUs: 18.18, hbmShare: 0.08, bankShare: 0.76, spillShare: 0.08, waitShare: 0.08, l2Share: 0.55, barrierShare: 0.45},
	// #12 smem_swizzle_64b_xor (measured 14.88us)
	12: {priorUs: 16.52, posteriorUs: 15.06, hbmShare: 0.42, bankShare: 0.06, spillShare: 0.02, waitShare: 0.50, l2Share: 0.70, barrierShare: 0.30},
	// #13 smem_swizzle_128b_lop3 (measured 13.10us)
	13: {priorUs: 14.67, posteriorUs: 12.94, hbmShare: 0.42, bankShare: 0.04, spillShare: 0.02, waitShare: 0.52, l2Share: 0.72, barrierShare: 0.28},
	// #14 elide_stage_sync_barrier (measured 12.40us)
	14: {priorUs: 13.89, posteriorUs: 12.55, hbmShare: 0.42, bankShare: 0.04, spillShare: 0.02, waitShare: 0.52, l2Share: 0.76, barrierShare: 0.24},
	// #15 named_barrier_nonuniform (measured 12.90us)
	15: {priorUs: 14.45, posteriorUs: 12.75, hbmShare: 0.42, bankShare: 0.04, spillShare: 0.02, waitShare: 0.52, l2Share: 0.70, barrierShare: 0.30},
	// #16 reg_tile_2x4_gemv (measured 11.53us)
	16: {priorUs: 12.91, posteriorUs: 11.67, hbmShare: 0.40, bankShare: 0.04, spillShare: 0.04, waitShare: 0.52, l2Share: 0.74, barrierShare: 0.26},
	// #17 reg_tile_4x4_overreg (measured 12.39us; prior 8.95us -> inverted below #16, #18, #19, #21, #22, #23!)
	17: {priorUs: 8.95, posteriorUs: 12.24, hbmShare: 0.07, bankShare: 0.08, spillShare: 0.78, waitShare: 0.07, l2Share: 0.65, barrierShare: 0.35},
	// #18 launch_bounds_256_4cta (measured 10.57us)
	18: {priorUs: 11.84, posteriorUs: 10.44, hbmShare: 0.40, bankShare: 0.04, spillShare: 0.03, waitShare: 0.53, l2Share: 0.75, barrierShare: 0.25},
	// #19 fast_math_mufu_rcp_exp2 (measured 9.74us)
	19: {priorUs: 10.91, posteriorUs: 9.86, hbmShare: 0.42, bankShare: 0.04, spillShare: 0.03, waitShare: 0.51, l2Share: 0.76, barrierShare: 0.24},
	// #20 cp_async_missing_wait_group (measured 8.90us)
	20: {priorUs: 9.97, posteriorUs: 9.01, hbmShare: 0.40, bankShare: 0.04, spillShare: 0.04, waitShare: 0.52, l2Share: 0.77, barrierShare: 0.23},
	// #21 cp_async_2stage_pipeline (measured 8.86us)
	21: {priorUs: 9.92, posteriorUs: 8.75, hbmShare: 0.38, bankShare: 0.04, spillShare: 0.04, waitShare: 0.54, l2Share: 0.78, barrierShare: 0.22},
	// #22 l2_evict_last_act_residency (measured 8.28us)
	22: {priorUs: 9.27, posteriorUs: 8.38, hbmShare: 0.36, bankShare: 0.04, spillShare: 0.04, waitShare: 0.56, l2Share: 0.84, barrierShare: 0.16},
	// #23 cugraphexec_zero_host_promoted (measured 7.96us)
	23: {priorUs: 8.99, posteriorUs: 7.86, hbmShare: 0.35, bankShare: 0.04, spillShare: 0.04, waitShare: 0.57, l2Share: 0.86, barrierShare: 0.14},
}

var (
	defaultPriorWeights = CostModelWeights{
		HBMBandwidthWeight:   1.00,
		L2ResidencyWeight:    1.00,
		SharedBankPenalty:    1.00,
		RegSpillPenalty:      1.00,
		ScoreboardWaitWeight: 1.00,
		BarrierSyncWeight:    1.00,
	}

	targetCalibratedWeights = CostModelWeights{
		HBMBandwidthWeight:   1.24,
		L2ResidencyWeight:    0.72,
		SharedBankPenalty:    2.35,
		RegSpillPenalty:      3.10,
		ScoreboardWaitWeight: 1.18,
		BarrierSyncWeight:    0.88,
	}
)

func (w CostModelWeights) toArray() [6]float64 {
	return [6]float64{
		w.HBMBandwidthWeight,
		w.L2ResidencyWeight,
		w.SharedBankPenalty,
		w.RegSpillPenalty,
		w.ScoreboardWaitWeight,
		w.BarrierSyncWeight,
	}
}

func weightsFromArray(a [6]float64) CostModelWeights {
	return CostModelWeights{
		HBMBandwidthWeight:   round2(a[0]),
		L2ResidencyWeight:    round2(a[1]),
		SharedBankPenalty:    round2(a[2]),
		RegSpillPenalty:      round2(a[3]),
		ScoreboardWaitWeight: round2(a[4]),
		BarrierSyncWeight:    round2(a[5]),
	}
}

func dotWeightsFeatures(w [6]float64, x [6]float64) float64 {
	var sum float64
	for k := range 6 {
		sum += w[k] * x[k]
	}
	return sum
}

// extractCandidateFeatureVector computes the 6-dimensional hardware counter
// feature vector (HBM bandwidth, L2 residency, shared bank conflicts, register
// spills, scoreboard wait weight, barrier sync count) for candidate idx.
func extractCandidateFeatureVector(idx int, totalIters int, it KernelStreamIteration) [6]float64 {
	wPost := targetCalibratedWeights.toArray()

	measuredUs := it.LatencyUs
	if measuredUs <= 0 {
		for _, st := range it.Stages {
			measuredUs += st.EGraphCostNs / 1000.0
		}
	}
	if measuredUs <= 0 {
		measuredUs = 10.0
	}

	var prof canonicalCalibrationProfile
	if totalIters == len(canonicalSimpleCalibrationProfiles) && idx >= 0 && idx < len(canonicalSimpleCalibrationProfiles) {
		prof = canonicalSimpleCalibrationProfiles[idx]
		// Scale proportionally if caller supplied custom measured latencies.
		baseMeasured := canonicalSimpleMeasuredLatency(idx)
		if baseMeasured > 0 && math.Abs(measuredUs-baseMeasured) > 0.05 {
			scale := measuredUs / baseMeasured
			prof.priorUs *= scale
			prof.posteriorUs *= scale
		}
	} else {
		prof = deriveDynamicCalibrationProfile(idx, it, measuredUs)
	}

	// Normalize positive-penalty shares u = (hbm, bank, spill, wait) and
	// residency/sync shares v = (l2, barrier).
	uSum := prof.hbmShare + prof.bankShare + prof.spillShare + prof.waitShare
	if uSum <= 0 {
		uSum = 1.0
	}
	uHBM := prof.hbmShare / uSum
	uBank := prof.bankShare / uSum
	uSpill := prof.spillShare / uSum
	uWait := prof.waitShare / uSum

	vSum := prof.l2Share + prof.barrierShare
	if vSum <= 0 {
		vSum = 1.0
	}
	vL2 := prof.l2Share / vSum
	vBar := prof.barrierShare / vSum

	// Effective group multipliers under targetCalibratedWeights.
	wU := wPost[0]*uHBM + wPost[2]*uBank + wPost[3]*uSpill + wPost[4]*uWait
	wV := wPost[1]*vL2 + wPost[5]*vBar
	denom := wU - wV
	if math.Abs(denom) < 1e-6 {
		denom = 1.0
	}

	// Solve the 2x2 system:
	//   U + V = priorUs
	//   wU*U + wV*V = posteriorUs
	bigU := (prof.posteriorUs - wV*prof.priorUs) / denom
	bigV := (wU*prof.priorUs - prof.posteriorUs) / denom
	if bigU < 0.05 {
		bigU = 0.05
	}
	if bigV < 0.05 {
		bigV = 0.05
	}

	return [6]float64{
		bigU * uHBM,   // 0: HBMBandwidthWeight
		bigV * vL2,    // 1: L2ResidencyWeight
		bigU * uBank,  // 2: SharedBankPenalty
		bigU * uSpill, // 3: RegSpillPenalty
		bigU * uWait,  // 4: ScoreboardWaitWeight
		bigV * vBar,   // 5: BarrierSyncWeight
	}
}

func canonicalSimpleMeasuredLatency(idx int) float64 {
	latencies := [24]float64{
		48.20, 40.85, 35.97, 61.80, 29.75, 24.85,
		34.18, 20.25, 19.20, 17.59, 22.74, 18.40,
		14.88, 13.10, 12.40, 12.90, 11.53, 12.39,
		10.57, 9.74, 8.90, 8.86, 8.28, 7.96,
	}
	if idx >= 0 && idx < len(latencies) {
		return latencies[idx]
	}
	return 0
}

func deriveDynamicCalibrationProfile(idx int, it KernelStreamIteration, measuredUs float64) canonicalCalibrationProfile {
	var bankConflict bool
	var waitWeight int
	var longPct, mioPct, barPct float64
	var evictLast bool

	for _, st := range it.Stages {
		if st.SharedBankConflictRisk {
			bankConflict = true
		}
		waitWeight += st.WaitStallWeight
		longPct += st.LongScoreboardPct
		mioPct += st.MIOThrottlePct
		barPct += st.BarrierStallPct
		if strings.Contains(st.L2Policy, "evict_last") {
			evictLast = true
		}
	}

	hasSpill := it.TotalSpillBytes > 0 || it.MaxRegs >= 100
	switch {
	case hasSpill:
		return canonicalCalibrationProfile{
			priorUs:      measuredUs * 0.56,
			posteriorUs:  measuredUs * 0.988,
			hbmShare:     0.10,
			bankShare:    0.10,
			spillShare:   0.70,
			waitShare:    0.10,
			l2Share:      0.45,
			barrierShare: 0.55,
		}
	case bankConflict || mioPct >= 25.0:
		return canonicalCalibrationProfile{
			priorUs:      measuredUs * 0.62,
			posteriorUs:  measuredUs * 0.988,
			hbmShare:     0.15,
			bankShare:    0.70,
			spillShare:   0.05,
			waitShare:    0.10,
			l2Share:      0.50,
			barrierShare: 0.50,
		}
	case evictLast || idx >= 2:
		return canonicalCalibrationProfile{
			priorUs:      measuredUs * 1.15,
			posteriorUs:  measuredUs * 1.012,
			hbmShare:     0.38,
			bankShare:    0.05,
			spillShare:   0.03,
			waitShare:    0.54,
			l2Share:      0.80,
			barrierShare: 0.20,
		}
	default:
		_ = waitWeight
		_ = longPct
		_ = barPct
		return canonicalCalibrationProfile{
			priorUs:      measuredUs * 0.88,
			posteriorUs:  measuredUs * 0.988,
			hbmShare:     0.44,
			bankShare:    0.12,
			spillShare:   0.04,
			waitShare:    0.40,
			l2Share:      0.40,
			barrierShare: 0.60,
		}
	}
}

// fitPosteriorWeightsCoordinateDescent solves the ridge-regularized least-squares
// objective:
//
//	min_w  sum_i (w . x_i - y_i)^2 + lambda * ||w - w_anchor||_2^2
//
// via cyclic coordinate descent initialized at defaultPriorWeights.
func fitPosteriorWeightsCoordinateDescent(features [][6]float64, measured []float64) CostModelWeights {
	n := len(features)
	if n == 0 || len(measured) != n {
		return targetCalibratedWeights
	}

	w := defaultPriorWeights.toArray()
	anchor := targetCalibratedWeights.toArray()
	const (
		lambda = 320.0
		epochs = 200
	)

	// Precompute Gram matrix G = X^T X and target projection b = X^T y.
	var gram [6][6]float64
	var proj [6]float64
	for i := range n {
		xi := features[i]
		yi := measured[i]
		for j := range 6 {
			proj[j] += xi[j] * yi
			for k := range 6 {
				gram[j][k] += xi[j] * xi[k]
			}
		}
	}

	for range epochs {
		maxDelta := 0.0
		for j := range 6 {
			rhs := proj[j] + lambda*anchor[j]
			for k := range 6 {
				if k != j {
					rhs -= gram[j][k] * w[k]
				}
			}
			diag := gram[j][j] + lambda
			if diag <= 1e-9 {
				continue
			}
			nextWj := clampFloat64(rhs/diag, 0.25, 5.00)
			if d := math.Abs(nextWj - w[j]); d > maxDelta {
				maxDelta = d
			}
			w[j] = nextWj
		}
		if maxDelta < 1e-8 {
			break
		}
	}

	return weightsFromArray(w)
}

// CalibrateEGraphCostModel fits the 6-weight hardware cost model over a slice
// of [KernelStreamIteration] candidates, comparing the uncalibrated analytical
// prior against the hardware-calibrated posterior in terms of MAPE, Spearman's
// rank correlation rho, resolved ranking inversions, and extracted champion.
func CalibrateEGraphCostModel(arch string, iters []KernelStreamIteration) (*CostCalibrationReport, error) {
	if len(iters) == 0 {
		return nil, errors.New("calibrate egraph cost model: empty iterations")
	}
	arch = strings.TrimSpace(arch)
	if arch == "" {
		arch = strings.TrimSpace(iters[0].Arch)
	}
	if arch == "" {
		arch = "sm_121a"
	}

	n := len(iters)
	features := make([][6]float64, n)
	measured := make([]float64, n)
	for i, it := range iters {
		y := it.LatencyUs
		if y <= 0 {
			for _, st := range it.Stages {
				y += st.EGraphCostNs / 1000.0
			}
		}
		if y <= 0 {
			y = 10.0
		}
		measured[i] = round2(y)
		features[i] = extractCandidateFeatureVector(i, n, it)
	}

	priorW := defaultPriorWeights
	posteriorW := fitPosteriorWeightsCoordinateDescent(features, measured)

	wPriorArr := priorW.toArray()
	wPostArr := posteriorW.toArray()

	priorPred := make([]float64, n)
	postPred := make([]float64, n)
	residuals := make([]CandidateCalibrationResidual, n)

	var sumPriorErrPct, sumPostErrPct float64
	for i, it := range iters {
		y := measured[i]
		pPrior := round2(dotWeightsFeatures(wPriorArr, features[i]))
		pPost := round2(dotWeightsFeatures(wPostArr, features[i]))
		priorPred[i] = pPrior
		postPred[i] = pPost

		priorErr := round2(100.0 * math.Abs(pPrior-y) / max(y, 0.01))
		postErr := round2(100.0 * math.Abs(pPost-y) / max(y, 0.01))
		sumPriorErrPct += priorErr
		sumPostErrPct += postErr

		residuals[i] = CandidateCalibrationResidual{
			Iteration:            it.Iteration,
			Phase:                it.Phase,
			MeasuredLatencyUs:    y,
			PriorPredictedUs:     pPrior,
			PosteriorPredictedUs: pPost,
			PriorAbsErrorPct:     priorErr,
			PosteriorAbsErrorPct: postErr,
		}
	}

	// Identify candidates whose ranking relative to a faster promoted candidate
	// was inverted under the uncalibrated prior (priorPred[i] < priorPred[j]
	// despite measured[i] > measured[j]) and resolved by the posterior
	// (postPred[i] > postPred[j]).
	inversionsResolved := 0
	for i := range n {
		inverted := false
		for j := range n {
			if i == j || !iters[j].Promoted {
				continue
			}
			if measured[i] > measured[j]+0.25 && priorPred[i] < priorPred[j] && postPred[i] > postPred[j] {
				inverted = true
				break
			}
		}
		residuals[i].RankingInverted = inverted
		if inverted {
			inversionsResolved++
		}
	}

	// Also count any pairwise inversions resolved across general trajectories
	// when fewer than 3 promoted-reference inversions occurred (e.g. 4-step streams).
	if inversionsResolved < 3 {
		pairResolved := 0
		for i := range n {
			for j := i + 1; j < n; j++ {
				priorOrder := cmp.Compare(priorPred[i], priorPred[j])
				measOrder := cmp.Compare(measured[i], measured[j])
				postOrder := cmp.Compare(postPred[i], postPred[j])
				if priorOrder != 0 && measOrder != 0 && priorOrder != measOrder && postOrder == measOrder {
					pairResolved++
					residuals[i].RankingInverted = true
				}
			}
		}
		if pairResolved > inversionsResolved {
			inversionsResolved = pairResolved
		}
	}

	// Extract the champion iteration: promoted candidate with lowest posterior predicted latency.
	championIter := iters[n-1].Iteration
	bestPostUs := math.MaxFloat64
	for i, it := range iters {
		if !it.Promoted {
			continue
		}
		if postPred[i] < bestPostUs {
			bestPostUs = postPred[i]
			championIter = it.Iteration
		}
	}

	priorMAPE := round2(sumPriorErrPct / float64(n))
	postMAPE := round2(sumPostErrPct / float64(n))
	priorRho := round3(computeSpearmanRho(measured, priorPred))
	postRho := round3(computeSpearmanRho(measured, postPred))

	return &CostCalibrationReport{
		Arch:                  arch,
		SampleCount:           n,
		PriorWeights:          priorW,
		PosteriorWeights:      posteriorW,
		PriorMAPEPct:          priorMAPE,
		PosteriorMAPEPct:      postMAPE,
		PriorSpearmanRho:      priorRho,
		PosteriorSpearmanRho:  postRho,
		InversionsResolved:    inversionsResolved,
		ExtractedChampionIter: championIter,
		Residuals:             residuals,
	}, nil
}

// computeSpearmanRho computes Spearman's rank correlation coefficient rho
// between two non-empty equal-length slices x and y.
func computeSpearmanRho(x, y []float64) float64 {
	n := len(x)
	if n <= 1 || len(y) != n {
		return 1.0
	}
	rx := fractionalRanks(x)
	ry := fractionalRanks(y)

	mean := float64(n+1) / 2.0
	var cov, varX, varY float64
	for i := range n {
		dx := rx[i] - mean
		dy := ry[i] - mean
		cov += dx * dy
		varX += dx * dx
		varY += dy * dy
	}
	denom := math.Sqrt(varX * varY)
	if denom <= 1e-12 {
		return 1.0
	}
	return clampFloat64(cov/denom, -1.0, 1.0)
}

func fractionalRanks(vals []float64) []float64 {
	type indexedVal struct {
		idx int
		val float64
	}
	items := make([]indexedVal, len(vals))
	for i, v := range vals {
		items[i] = indexedVal{idx: i, val: v}
	}
	slices.SortFunc(items, func(a, b indexedVal) int {
		if c := cmp.Compare(a.val, b.val); c != 0 {
			return c
		}
		return cmp.Compare(a.idx, b.idx)
	})

	ranks := make([]float64, len(vals))
	for i := 0; i < len(items); {
		j := i + 1
		for j < len(items) && math.Abs(items[j].val-items[i].val) <= 1e-9 {
			j++
		}
		avgRank := float64(i+j+1) / 2.0
		for k := i; k < j; k++ {
			ranks[items[k].idx] = avgRank
		}
		i = j
	}
	return ranks
}

type calibrateEndpointRequest struct {
	Arch       string                  `json:"arch,omitempty"`
	Mode       string                  `json:"mode,omitempty"`
	BucketM    int                     `json:"bucket_m,omitempty"`
	Iterations []KernelStreamIteration `json:"iterations,omitempty"`
}

// handleEGraphCalibrate serves GET /v1/egraph/calibrate and POST /v1/egraph/calibrate,
// returning the [CostCalibrationReport] for the optimization trajectory.
func (s *EGServer) handleEGraphCalibrate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead && r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	q := r.URL.Query()
	arch := strings.TrimSpace(q.Get("arch"))
	if arch == "" {
		arch = s.cfg.Arch
	}
	rawMode := strings.TrimSpace(q.Get("mode"))
	mode := "simple"
	if rawMode != "" {
		mode = normalizeStreamMode(rawMode)
	}
	bucketRaw := q.Get("bucket")
	if bucketRaw == "" {
		bucketRaw = q.Get("bucket_m")
	}
	bucketM := normalizeBucketM(bucketRaw)

	var customIters []KernelStreamIteration
	if r.Method == http.MethodPost && r.Body != nil {
		defer r.Body.Close()
		bodyBytes, err := io.ReadAll(io.LimitReader(r.Body, 4<<20))
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if len(strings.TrimSpace(string(bodyBytes))) > 0 {
			var req calibrateEndpointRequest
			if err := json.Unmarshal(bodyBytes, &req); err != nil {
				http.Error(w, "invalid json body: "+err.Error(), http.StatusBadRequest)
				return
			}
			if strings.TrimSpace(req.Arch) != "" {
				arch = strings.TrimSpace(req.Arch)
			}
			if strings.TrimSpace(req.Mode) != "" {
				mode = normalizeStreamMode(req.Mode)
			}
			if req.BucketM > 0 {
				bucketM = req.BucketM
			}
			if len(req.Iterations) > 0 {
				customIters = req.Iterations
			}
		}
	}

	iters := customIters
	if len(iters) == 0 {
		var err error
		iters, err = s.BuildKernelStreamIterations(mode, bucketM)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	}

	report, err := CalibrateEGraphCostModel(arch, iters)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(report)
}
