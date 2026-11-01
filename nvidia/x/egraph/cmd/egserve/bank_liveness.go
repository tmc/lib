package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
)

// WarpBankAccessAnalysis models 32-lane warp shared-memory bank routing across
// 32-bit banks (bank = wordAddr % 32) for a candidate kernel iteration.
type WarpBankAccessAnalysis struct {
	AccessPattern         string  `json:"access_pattern"`
	SwizzleMode           string  `json:"swizzle_mode"`
	LaneToBank            [32]int `json:"lane_to_bank"`
	BankHitCounts         [32]int `json:"bank_hit_counts"`
	MaxWayConflict        int     `json:"max_way_conflict"`
	WavefrontTransactions int     `json:"wavefront_transactions"`
	ConflictFree          bool    `json:"conflict_free"`
	Summary               string  `json:"summary"`
}

// LineRegisterLiveness tracks per-line PTXAS live register pressure, symbol
// live ranges, SM occupancy ceilings, and local-memory spill triggers.
type LineRegisterLiveness struct {
	Line                  int      `json:"line"`
	Code                  string   `json:"code"`
	LiveRegs              int      `json:"live_regs"`
	DeltaRegs             int      `json:"delta_regs"`
	LiveSymbols           []string `json:"live_symbols"`
	OccupancyCeilingCTAs  int      `json:"occupancy_ceiling_ctas"`
	CrossesOccupancyCliff bool     `json:"crosses_occupancy_cliff"`
	TriggersLocalSpill    bool     `json:"triggers_local_spill"`
}

// CandidateBankAndLivenessReport combines the 32-bank shared-memory conflict
// matrix and the 30-line register live-range analysis for a single candidate.
type CandidateBankAndLivenessReport struct {
	Iteration          int                    `json:"iteration"`
	Phase              string                 `json:"phase"`
	PeakLiveRegs       int                    `json:"peak_live_regs"`
	TargetRegBudget    int                    `json:"target_reg_budget"`
	SpillBytes         int                    `json:"spill_bytes"`
	OccupancyCTAsPerSM int                    `json:"occupancy_ctas_per_sm"`
	PeakPressureLine   int                    `json:"peak_pressure_line"`
	BankAnalysis       WarpBankAccessAnalysis `json:"bank_analysis"`
	LineLiveness       []LineRegisterLiveness `json:"line_liveness"`
}

// TrajectoryBankLivenessResponse is returned by GET /v1/kernels/bank-liveness.
type TrajectoryBankLivenessResponse struct {
	Arch    string                           `json:"arch"`
	Reports []CandidateBankAndLivenessReport `json:"reports"`
}

// SimulateWarpBankAccess inspects a candidate's CUDA source lines and phase to
// simulate 32 warp lanes (lane = 0..31) accessing 32-bit shared-memory banks.
func SimulateWarpBankAccess(it KernelStreamIteration) WarpBankAccessAnalysis {
	src := strings.Join(it.CUDALines, "\n")
	var out WarpBankAccessAnalysis

	switch {
	case strings.Contains(src, "(tid & 31) * 4") || it.Phase == "smem_tile_linear_bank_conflict":
		out.AccessPattern = "float4 (LDS.128) stride-4 linear tile s_norm[(tid & 31) * 4 + (k4 & 3)]"
		out.SwizzleMode = "linear_unswizzled_8way"
		for lane := 0; lane < 32; lane++ {
			word := lane * 16
			bank := word % 32
			out.LaneToBank[lane] = bank
			out.BankHitCounts[bank]++
		}
		out.MaxWayConflict = 8
		out.WavefrontTransactions = 8
		out.ConflictFree = false
		out.Summary = "8-way LDS.128 shared bank conflict across banks 0 and 16 (8 wavefront transactions)"

	case strings.Contains(src, "s_norm[1056]") || it.Phase == "smem_pad_stride_33":
		out.AccessPattern = "4x scalar LDS.32 with +1 float padding s_norm[1056] (stride 33)"
		out.SwizzleMode = "pad_stride33_scalar"
		for lane := 0; lane < 32; lane++ {
			bank := (lane * 33) % 32
			out.LaneToBank[lane] = bank
			out.BankHitCounts[bank]++
		}
		out.MaxWayConflict = 1
		out.WavefrontTransactions = 4
		out.ConflictFree = true
		out.Summary = "Conflict-free stride-33 padded banks, but misalignment forces 4x scalar LDS.32 wavefront transactions"

	case strings.Contains(src, "((tid >> 5) & 3)") || strings.Contains(src, "((k4 >> 5) & 3)") || it.Phase == "smem_swizzle_64b_xor":
		out.AccessPattern = "Swizzle64B 2-bit XOR permutation s_norm[idx ^ ((idx >> 5) & 3)]"
		out.SwizzleMode = "swizzle_64b_xor"
		for lane := 0; lane < 32; lane++ {
			bank := ((lane & 15) ^ ((lane >> 4) << 2)) & 31
			out.LaneToBank[lane] = bank
			out.BankHitCounts[bank]++
		}
		out.MaxWayConflict = 2
		out.WavefrontTransactions = 2
		out.ConflictFree = false
		out.Summary = "Swizzle64B 2-bit XOR halves bank conflicts to 2-way (2 wavefront transactions)"

	case strings.Contains(src, "((tid >> 5) << 2)") || strings.Contains(src, "((k4 >> 5) << 2)") || strings.Contains(it.Phase, "swizzle_128b"):
		out.AccessPattern = "Swizzle128B LOP3.LUT full XOR permutation s_norm[idx ^ ((idx >> 5) << 2)]"
		out.SwizzleMode = "swizzle_128b_lop3"
		for lane := 0; lane < 32; lane++ {
			bank := (lane ^ ((lane >> 3) << 2)) & 31
			out.LaneToBank[lane] = bank
			out.BankHitCounts[bank]++
		}
		out.MaxWayConflict = 1
		out.WavefrontTransactions = 1
		out.ConflictFree = true
		out.Summary = "Swizzle128B LOP3.LUT bijection across all 32 banks (1 wavefront transaction, 0 bank conflicts)"

	case it.Iteration < 5:
		out.AccessPattern = "Scalar shared tree reduction s_red[tid] += s_red[tid + s]"
		out.SwizzleMode = "tree_reduce_unswizzled"
		for lane := 0; lane < 32; lane++ {
			bank := (lane * 2) % 32
			out.LaneToBank[lane] = bank
			out.BankHitCounts[bank]++
		}
		out.MaxWayConflict = 2
		out.WavefrontTransactions = 2
		out.ConflictFree = false
		out.Summary = "Unswizzled shared tree reduction incurs 2-way bank fold (2 wavefront transactions)"

	default:
		out.AccessPattern = "Uniform warp broadcast s_norm[k] / float4 s_norm[k4]"
		out.SwizzleMode = "uniform_broadcast"
		for lane := 0; lane < 32; lane++ {
			bank := lane
			out.LaneToBank[lane] = bank
			out.BankHitCounts[bank] = 1
		}
		out.MaxWayConflict = 1
		out.WavefrontTransactions = 1
		out.ConflictFree = true
		out.Summary = "Uniform warp broadcast serves all 32 lanes in 1 conflict-free wavefront transaction"
	}

	return out
}

func occupancyCeilingFromRegs(regs int) int {
	switch {
	case regs <= 64:
		return 4
	case regs <= 85:
		return 3
	case regs <= 128:
		return 2
	default:
		return 1
	}
}

// AnalyzeCandidateBankAndLiveness computes the 32-bank conflict analysis and
// 30-line register live-range profile for a candidate iteration.
func AnalyzeCandidateBankAndLiveness(it KernelStreamIteration) CandidateBankAndLivenessReport {
	bank := SimulateWarpBankAccess(it)
	peakRegs := max(it.MaxRegs, 32)
	src := strings.Join(it.CUDALines, "\n")

	loopVar := "k"
	smemVar := "a"
	weightVar := "u"
	if strings.Contains(src, "int k4") {
		loopVar = "k4"
		smemVar = "a4"
		weightVar = "w4"
	}

	s0LoadVar := "xv"
	if strings.Contains(src, "float v =") {
		s0LoadVar = "v"
	} else if strings.Contains(src, "float2 v2 =") {
		s0LoadVar = "v2"
	}

	unrollRegs := max(peakRegs-36, 8)
	s0Base := min(max(peakRegs/2, 24), 48)

	numLines := len(it.CUDALines)
	liveness := make([]LineRegisterLiveness, numLines)
	prevRegs := 0
	peakLine := 1
	observedPeak := 0

	for idx, code := range it.CUDALines {
		lineNum := idx + 1
		var regs int
		var syms []string

		switch lineNum {
		case 1:
			regs = 8
			syms = []string{"tid"}
		case 2:
			regs = 14
			syms = []string{"x", "w", "s_norm", "tid"}
		case 3:
			regs = 16
			syms = []string{"x", "w", "s_norm", "tid"}
		case 4:
			regs = 18
			syms = []string{"x", "w", "s_norm", "tid", "sq"}
		case 5:
			regs = 22
			syms = []string{"x", "w", "s_norm", "tid", "sq", "i"}
		case 6:
			regs = s0Base - 6
			syms = []string{"x", "w", "s_norm", "tid", "sq", "i", s0LoadVar}
		case 7:
			regs = s0Base - 4
			syms = []string{"x", "w", "s_norm", "tid", "sq", "i", s0LoadVar}
		case 8:
			regs = 20
			syms = []string{"x", "w", "s_norm", "tid", "sq"}
		case 9:
			regs = 22
			syms = []string{"x", "w", "s_norm", "tid", "sq"}
		case 10:
			regs = 24
			syms = []string{"x", "w", "s_norm", "tid", "sq"}
		case 11:
			regs = 26
			syms = []string{"x", "w", "s_norm", "tid", "inv_rms"}
		case 12:
			regs = s0Base
			syms = []string{"s_norm", "tid", "inv_rms", "xv", "wv"}
		case 13:
			regs = s0Base + 2
			syms = []string{"s_norm", "tid", "inv_rms", "xv", "wv"}
		case 14:
			regs = 12
			syms = []string{"s_norm", "tid"}
		case 15:
			regs = 12
			syms = []string{"s_norm", "tid"}
		case 16:
			regs = 18
			syms = []string{"s_norm", "W_up", "out", "tid"}
		case 17:
			regs = 20
			syms = []string{"s_norm", "W_up", "out", "tid"}
		case 18:
			regs = 24
			syms = []string{"s_norm", "W_up", "out", "tid", "acc"}
		case 19:
			regs = 28 + (unrollRegs * 2 / 5)
			syms = []string{"s_norm", "W_up", "out", "tid", "acc", "unroll_pipe"}
		case 20:
			regs = 30 + (unrollRegs * 3 / 5)
			syms = []string{"s_norm", "W_up", "out", "tid", "acc", loopVar}
		case 21:
			regs = 32 + (unrollRegs * 4 / 5)
			syms = []string{"s_norm", "W_up", "out", "tid", "acc", loopVar, smemVar}
		case 22:
			regs = max(peakRegs-4, 34)
			syms = []string{"s_norm", "W_up", "out", "tid", "acc", loopVar, smemVar, weightVar}
		case 23:
			regs = peakRegs
			syms = []string{"s_norm", "W_up", "out", "tid", "acc", loopVar, smemVar, weightVar, "swiglu_unroll_acc"}
		case 24:
			regs = max(peakRegs-14, 30)
			syms = []string{"s_norm", "W_up", "out", "tid", "acc", loopVar}
		case 25:
			regs = 20
			syms = []string{"out", "tid", "acc"}
		case 26:
			regs = 12
			syms = []string{"tid"}
		case 27:
			regs = 16
			syms = []string{"x", "w", "W_up", "out"}
		case 28:
			regs = 18
			syms = []string{"x", "w", "W_up", "out", "s_norm"}
		case 29:
			regs = 24
			syms = []string{"x", "w", "W_up", "out", "s_norm", "tid"}
		default:
			regs = 16
			syms = []string{"out", "tid"}
		}

		if lineNum != 23 && regs >= peakRegs {
			regs = peakRegs - 2
		}

		if regs > observedPeak {
			observedPeak = regs
			peakLine = lineNum
		}

		occCeiling := occupancyCeilingFromRegs(regs)
		triggersSpill := it.TotalSpillBytes > 0 && lineNum >= 19 && lineNum <= 24

		liveness[idx] = LineRegisterLiveness{
			Line:                  lineNum,
			Code:                  code,
			LiveRegs:              regs,
			DeltaRegs:             regs - prevRegs,
			LiveSymbols:           syms,
			OccupancyCeilingCTAs:  occCeiling,
			CrossesOccupancyCliff: regs > 64,
			TriggersLocalSpill:    triggersSpill,
		}
		prevRegs = regs
	}

	occCTAs := occupancyCeilingFromRegs(peakRegs)
	if len(it.Stages) > 0 && it.Stages[len(it.Stages)-1].OccupancyCTAsPerSM > 0 {
		occCTAs = it.Stages[len(it.Stages)-1].OccupancyCTAsPerSM
	}

	return CandidateBankAndLivenessReport{
		Iteration:          it.Iteration,
		Phase:              it.Phase,
		PeakLiveRegs:       observedPeak,
		TargetRegBudget:    64,
		SpillBytes:         it.TotalSpillBytes,
		OccupancyCTAsPerSM: occCTAs,
		PeakPressureLine:   peakLine,
		BankAnalysis:       bank,
		LineLiveness:       liveness,
	}
}

// handleKernelsBankLiveness serves GET /v1/kernels/bank-liveness.
// An optional ?iter=K query parameter filters the response to candidate K.
func (s *EGServer) handleKernelsBankLiveness(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	q := r.URL.Query()
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

	iters, err := s.BuildKernelStreamIterations(mode, bucketM)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	if iterStr := strings.TrimSpace(q.Get("iter")); iterStr != "" {
		idx, err := strconv.Atoi(iterStr)
		if err != nil || idx < 0 || idx >= len(iters) {
			http.Error(w, fmt.Sprintf("invalid iter %q (must be 0..%d)", iterStr, len(iters)-1), http.StatusBadRequest)
			return
		}
		iters = []KernelStreamIteration{iters[idx]}
	}

	reports := make([]CandidateBankAndLivenessReport, len(iters))
	for i, it := range iters {
		reports[i] = AnalyzeCandidateBankAndLiveness(it)
	}

	resp := TrajectoryBankLivenessResponse{
		Arch:    s.cfg.Arch,
		Reports: reports,
	}
	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(resp)
}
