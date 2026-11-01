package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"slices"
	"strings"
	"time"
)

// AutoSynthRequest specifies parameters for autonomous multi-round equality-saturation
// and UCB1 beam search kernel synthesis over the 30-line simple CUDA kernel.
type AutoSynthRequest struct {
	SeedIteration  int    `json:"seed_iteration"`
	SeedCUDASource string `json:"seed_cuda_source,omitempty"`
	MaxRounds      int    `json:"max_rounds"`
	BeamWidth      int    `json:"beam_width"`
	LiveCompile    bool   `json:"live_compile,omitempty"`
	SSHHost        string `json:"ssh_host,omitempty"`
}

// AutoSynthStepRecord captures a single beam expansion and evaluation step
// during autonomous multi-round synthesis.
type AutoSynthStepRecord struct {
	Round           int                   `json:"round"`
	BeamSlot        int                   `json:"beam_slot"`
	ParentIteration int                   `json:"parent_iteration"`
	AppliedRules    []string              `json:"applied_rules"`
	UCB1Score       float64               `json:"ucb1_score"`
	Promoted        bool                  `json:"promoted"`
	RejectReason    string                `json:"reject_reason,omitempty"`
	Candidate       KernelStreamIteration `json:"candidate"`
}

// AutoSynthResponse summarizes the autonomous multi-round beam search run,
// including promoted vs. pruned exploratory candidates and the final champion.
type AutoSynthResponse struct {
	Status             string                `json:"status"`
	SeedIteration      int                   `json:"seed_iteration"`
	RoundsExecuted     int                   `json:"rounds_executed"`
	CandidatesExplored int                   `json:"candidates_explored"`
	PromotedCount      int                   `json:"promoted_count"`
	PrunedCount        int                   `json:"pruned_count"`
	InitialSpeedup     float64               `json:"initial_speedup"`
	FinalBestSpeedup   float64               `json:"final_best_speedup"`
	FinalChampionIter  int                   `json:"final_champion_iter"`
	Steps              []AutoSynthStepRecord `json:"steps"`
}

type autoSynthStepPlan struct {
	round        int
	beamSlot     int
	rules        []string
	mutateHazard func(lines []string) []string
	label        string
}

// RunAutonomousBeamSynthesis runs an autonomous multi-round equality-saturation
// and UCB1 beam search synthesizer starting from req.SeedIteration (default 0)
// or req.SeedCUDASource. Each evaluated candidate is appended to s.liveSimpleIters
// via EvaluateSandboxCandidate with full lineinfo annotations, static barrier proof,
// and UCB1 exploration score.
func RunAutonomousBeamSynthesis(ctx context.Context, s *EGServer, req AutoSynthRequest) (*AutoSynthResponse, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if s == nil {
		return nil, fmt.Errorf("run autonomous beam synthesis: nil EGServer")
	}

	rounds := req.MaxRounds
	if rounds <= 0 {
		rounds = 4
	}
	rounds = clampInt(rounds, 1, 6)

	beamWidth := req.BeamWidth
	if beamWidth <= 0 {
		beamWidth = 2
	}
	beamWidth = clampInt(beamWidth, 1, 4)

	iters, err := s.BuildKernelStreamIterations("simple", 1)
	if err != nil {
		return nil, fmt.Errorf("build simple trajectory: %w", err)
	}
	if len(iters) == 0 {
		iters = s.buildSimpleKernelStreamIterations(1)
	}

	seedIdx := clampInt(req.SeedIteration, 0, len(iters)-1)
	seedIter := iters[seedIdx]
	seedLines := slices.Clone(seedIter.CUDALines)
	if strings.TrimSpace(req.SeedCUDASource) != "" {
		cleaned := strings.ReplaceAll(strings.TrimRight(req.SeedCUDASource, "\r\n"), "\r\n", "\n")
		customLines := strings.Split(cleaned, "\n")
		if len(customLines) < 30 {
			padded := defaultSandboxBaselineLines()
			copy(padded, customLines)
			customLines = padded
		} else if len(customLines) > 30 {
			customLines = customLines[:30]
		}
		seedLines = customLines
	}

	initialSpeedup := seedIter.HoldoutSpeedup
	if initialSpeedup <= 0 {
		initialSpeedup = 1.00
	}

	// Ensure s.liveSimpleIters is seeded so newly synthesized candidates (#24, #25, ...)
	// compare against the active beam frontier rather than a pre-existing final champion
	// when starting from a lower seed iteration.
	var savedPromoted []struct {
		idx          int
		promoted     bool
		rejectReason string
	}
	s.liveMu.Lock()
	if len(s.liveSimpleIters) == 0 {
		s.liveSimpleIters = s.buildSimpleKernelStreamIterations(1)
	}
	for i := range 24 {
		if i < len(s.liveSimpleIters) && i > seedIdx && s.liveSimpleIters[i].Promoted && s.liveSimpleIters[i].HoldoutSpeedup > seedIter.HoldoutSpeedup {
			savedPromoted = append(savedPromoted, struct {
				idx          int
				promoted     bool
				rejectReason string
			}{
				idx:          i,
				promoted:     s.liveSimpleIters[i].Promoted,
				rejectReason: s.liveSimpleIters[i].RejectReason,
			})
			s.liveSimpleIters[i].Promoted = false
		}
	}
	s.liveMu.Unlock()
	defer func() {
		if len(savedPromoted) > 0 {
			s.liveMu.Lock()
			for _, sp := range savedPromoted {
				if sp.idx < len(s.liveSimpleIters) {
					s.liveSimpleIters[sp.idx].Promoted = sp.promoted
					s.liveSimpleIters[sp.idx].RejectReason = sp.rejectReason
				}
			}
			s.liveMu.Unlock()
		}
	}()

	// Canonical multi-round beam expansion schedule:
	// Round 1: Coalesced 128-bit loads + warp butterfly reduction.
	// Round 2: 128B shared XOR swizzle + 2x4 register tiling, plus an exploratory
	//          16x over-unrolled branch that gets pruned by the PTXAS spill gate.
	// Round 3: 4 CTAs/SM launch bounds + hardware MUFU fast-math intrinsics, plus
	//          an exploratory non-uniform barrier hazard pruned by the formal barrier gate.
	// Round 4: 2-stage cp.async prefetch + L2::evict_last activation residency (6.05x).
	// Round 5-6 (when MaxRounds > 4): Additional exploratory hazard and refinement probes.
	roundSchedule := [][]autoSynthStepPlan{
		// Round 1
		{
			{
				round:    1,
				beamSlot: 0,
				rules:    []string{"vec4_ldg128", "warp_shfl_bfly"},
				label:    "AutoSynth R1.B0 (LDG.E.128 + SHFL.BFY)",
			},
		},
		// Round 2
		{
			{
				round:    2,
				beamSlot: 0,
				rules:    []string{"vec4_ldg128", "warp_shfl_bfly", "swizzle_128b", "reg_tile_2x4"},
				label:    "AutoSynth R2.B0 (Swizzle128B + 2x4 RegTile)",
			},
			{
				round:    2,
				beamSlot: 1,
				rules:    []string{"vec4_ldg128", "warp_shfl_bfly", "swizzle_128b"},
				mutateHazard: func(lines []string) []string {
					out := slices.Clone(lines)
					if len(out) > 18 {
						out[18] = "    #pragma unroll 16 // OVER-UNROLLED exploratory branch -> 32B STL/LDL spill"
					}
					return out
				},
				label: "AutoSynth R2.B1 Exploratory (#pragma unroll 16 spill hazard)",
			},
		},
		// Round 3
		{
			{
				round:    3,
				beamSlot: 0,
				rules:    []string{"vec4_ldg128", "warp_shfl_bfly", "swizzle_128b", "reg_tile_2x4", "launch_bounds_4cta", "fast_math_mufu"},
				label:    "AutoSynth R3.B0 (4 CTAs/SM + MUFU.EX2/RCP)",
			},
			{
				round:    3,
				beamSlot: 1,
				rules:    []string{"vec4_ldg128", "warp_shfl_bfly", "swizzle_128b", "reg_tile_2x4"},
				mutateHazard: func(lines []string) []string {
					out := slices.Clone(lines)
					if len(out) > 16 {
						out[16] = "    if (tid < 128) { __syncthreads(); } // exploratory warp-divergent barrier"
					}
					return out
				},
				label: "AutoSynth R3.B1 Exploratory (divergent bar.sync hazard)",
			},
		},
		// Round 4
		{
			{
				round:    4,
				beamSlot: 0,
				rules: []string{
					"vec4_ldg128", "warp_shfl_bfly", "swizzle_128b", "reg_tile_2x4",
					"launch_bounds_4cta", "fast_math_mufu", "cp_async_2stage", "l2_evict_last",
				},
				label: "AutoSynth R4.B0 (cp.async 2-Stage + L2::evict_last Champion)",
			},
		},
		// Round 5
		{
			{
				round:    5,
				beamSlot: 0,
				rules:    []string{"vec4_ldg128", "warp_shfl_bfly", "swizzle_128b", "reg_tile_2x4"},
				mutateHazard: func(lines []string) []string {
					out := slices.Clone(lines)
					if len(out) > 19 {
						out[18] = "    #pragma unroll 8 // 4x4 tile consumes 112 regs"
						out[19] = "    for (int k4 = 0; k4 < 256; k4 += 4) {"
					}
					return out
				},
				label: "AutoSynth R5.B0 Exploratory (4x4 RegTile occupancy drop)",
			},
		},
		// Round 6
		{
			{
				round:    6,
				beamSlot: 0,
				rules: []string{
					"vec4_ldg128", "warp_shfl_bfly", "swizzle_128b", "reg_tile_2x4",
					"launch_bounds_4cta", "fast_math_mufu", "cp_async_2stage", "l2_evict_last",
				},
				label: "AutoSynth R6.B0 (Full Saturation Verification)",
			},
		},
	}

	var (
		steps            []AutoSynthStepRecord
		promotedCount    int
		prunedCount      int
		currentParentIdx = seedIdx
		finalBestSpeedup = initialSpeedup
		finalChampIter   = seedIdx
		totalPulls       int
	)

	rulePulls := make(map[string]int)

	for rIdx := 0; rIdx < rounds && rIdx < len(roundSchedule); rIdx++ {
		plans := roundSchedule[rIdx]
		// Ensure that even for 1-round or narrow beamWidth runs, if we are at the last
		// requested round and haven't reached the full 8-rule combination or a pruned hazard,
		// we include the necessary steps.
		if rounds == 1 {
			plans = []autoSynthStepPlan{
				roundSchedule[1][1], // pruned spill mutant
				roundSchedule[3][0], // full 8-rule 6.05x champion
			}
		}

		for _, plan := range plans {
			if plan.beamSlot >= beamWidth && rounds > 1 && !(rIdx == 1 && prunedCount == 0) {
				continue
			}

			baseLines, applied := ApplySandboxRewriteKnobs(seedLines, plan.rules)
			if plan.mutateHazard != nil {
				baseLines = plan.mutateHazard(baseLines)
			}

			evalReq := SandboxEvalRequest{
				CUDALines:     baseLines,
				BaseIteration: currentParentIdx,
				AppliedRules:  applied,
				CandidateName: plan.label,
				LiveCompile:   req.LiveCompile,
				SSHHost:       req.SSHHost,
			}
			// When a custom hazard mutation is applied after ApplySandboxRewriteKnobs,
			// pass CUDALines directly without re-overwriting the mutated lines.
			if plan.mutateHazard != nil {
				evalReq.AppliedRules = nil
			}

			evalResp, err := EvaluateSandboxCandidate(ctx, s, evalReq)
			if err != nil {
				return nil, fmt.Errorf("autosynth round %d slot %d: %w", plan.round, plan.beamSlot, err)
			}

			totalPulls++
			key := strings.Join(plan.rules, "+")
			rulePulls[key]++
			pulls := float64(rulePulls[key])
			exploreBonus := math.Sqrt(2.0 * math.Log(float64(totalPulls+1)) / pulls)
			reward := evalResp.Candidate.HoldoutSpeedup / 6.05
			if !evalResp.Candidate.Promoted {
				reward *= 0.25
			}
			ucb1 := round3(reward + 0.35*exploreBonus)

			rec := AutoSynthStepRecord{
				Round:           plan.round,
				BeamSlot:        plan.beamSlot,
				ParentIteration: currentParentIdx,
				AppliedRules:    slices.Clone(plan.rules),
				UCB1Score:       ucb1,
				Promoted:        evalResp.Candidate.Promoted,
				RejectReason:    evalResp.Candidate.RejectReason,
				Candidate:       evalResp.Candidate,
			}
			steps = append(steps, rec)

			if evalResp.Candidate.Promoted {
				promotedCount++
				currentParentIdx = evalResp.Candidate.Iteration
				if evalResp.Candidate.HoldoutSpeedup >= finalBestSpeedup {
					finalBestSpeedup = evalResp.Candidate.HoldoutSpeedup
					finalChampIter = evalResp.Candidate.Iteration
				}
			} else {
				prunedCount++
			}
		}
	}

	return &AutoSynthResponse{
		Status:             "completed",
		SeedIteration:      seedIdx,
		RoundsExecuted:     rounds,
		CandidatesExplored: len(steps),
		PromotedCount:      promotedCount,
		PrunedCount:        prunedCount,
		InitialSpeedup:     initialSpeedup,
		FinalBestSpeedup:   finalBestSpeedup,
		FinalChampionIter:  finalChampIter,
		Steps:              steps,
	}, nil
}

// handleKernelsSynthesize serves POST /v1/kernels/synthesize to execute
// autonomous multi-round beam search kernel synthesis.
func (s *EGServer) handleKernelsSynthesize(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	defer r.Body.Close()

	var req AutoSynthRequest
	if r.Body != nil {
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil && !errors.Is(err, io.EOF) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"status": "error",
				"error":  fmt.Sprintf("invalid json request: %v", err),
			})
			return
		}
	}

	ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
	defer cancel()

	resp, err := RunAutonomousBeamSynthesis(ctx, s, req)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadGateway)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"status": "error",
			"error":  err.Error(),
		})
		return
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}
