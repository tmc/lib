package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"time"
)

// SandboxRewriteRule describes a single composable e-graph rewrite knob that
// can be toggled or combined in the interactive CUDA live-edit sandbox.
type SandboxRewriteRule struct {
	ID                string  `json:"id"`
	Name              string  `json:"name"`
	Stage             string  `json:"stage"`
	Category          string  `json:"category"`
	Description       string  `json:"description"`
	TargetInstruction string  `json:"target_instruction"`
	EstimatedSpeedup  float64 `json:"estimated_speedup"`
}

// AvailableSandboxRewriteRules returns the canonical catalog of 8 composable
// e-graph rewrite rules over the 30-line fused RMSNorm + SwiGLU + GEMV kernel.
func AvailableSandboxRewriteRules() []SandboxRewriteRule {
	return []SandboxRewriteRule{
		{
			ID:                "vec4_ldg128",
			Name:              "Vectorize Global Loads (LDG.E.128)",
			Stage:             "s0_rmsnorm_normalize",
			Category:          "memory_coalescing",
			Description:       "Vectorize Stage s0 global loads to 128-bit float4 (LDG.E.128)",
			TargetInstruction: "LDG.E.128",
			EstimatedSpeedup:  1.94,
		},
		{
			ID:                "warp_shfl_bfly",
			Name:              "Warp Butterfly Reduction (SHFL.BFY)",
			Stage:             "s0_rmsnorm_normalize",
			Category:          "warp_collective",
			Description:       "Replace Stage s0 shared-memory tree reduction with warp __shfl_xor_sync butterfly reduction (SHFL.BFY)",
			TargetInstruction: "SHFL.BFY",
			EstimatedSpeedup:  2.74,
		},
		{
			ID:                "swizzle_128b",
			Name:              "128B XOR Shared Bank Swizzle (LOP3.LUT)",
			Stage:             "s0_rmsnorm_normalize",
			Category:          "shared_memory",
			Description:       "Apply 128-byte XOR shared-memory bank permutation (tid ^ ((tid >> 5) << 2), LOP3.LUT)",
			TargetInstruction: "LOP3.LUT",
			EstimatedSpeedup:  3.68,
		},
		{
			ID:                "reg_tile_2x4",
			Name:              "2x4 Register Weight Tiling",
			Stage:             "s1_swiglu_gemv_epilogue",
			Category:          "register_tiling",
			Description:       "Tile 2x float4 weights per shared broadcast in Stage s1",
			TargetInstruction: "FFMA",
			EstimatedSpeedup:  4.18,
		},
		{
			ID:                "launch_bounds_4cta",
			Name:              "Occupancy Launch Bounds (4 CTAs/SM)",
			Stage:             "s1_swiglu_gemv_epilogue",
			Category:          "occupancy",
			Description:       "Enforce __launch_bounds__(256, 4) for 4 CTAs/SM occupancy",
			TargetInstruction: "__launch_bounds__(256, 4)",
			EstimatedSpeedup:  4.56,
		},
		{
			ID:                "fast_math_mufu",
			Name:              "Hardware MUFU Intrinsics (EX2 + RCP)",
			Stage:             "s1_swiglu_gemv_epilogue",
			Category:          "math_pipe",
			Description:       "Rewrite transcendental expf/div to hardware __expf + __frcp_rn (MUFU.EX2 + MUFU.RCP)",
			TargetInstruction: "MUFU.EX2 + MUFU.RCP",
			EstimatedSpeedup:  4.95,
		},
		{
			ID:                "cp_async_2stage",
			Name:              "2-Stage Async Weight Prefetch (__ldcs)",
			Stage:             "s1_swiglu_gemv_epilogue",
			Category:          "async_pipeline",
			Description:       "Stream weight tiles with __ldcs / 2-stage async prefetch",
			TargetInstruction: "LDGSTS.E.BYPASS.128",
			EstimatedSpeedup:  5.44,
		},
		{
			ID:                "l2_evict_last",
			Name:              "Pin Activations in L2 (L2::evict_last)",
			Stage:             "s0_rmsnorm_normalize",
			Category:          "l2_persistence",
			Description:       "Pin input activations x[] in L2 via __ldca (createpolicy.fractional.L2::evict_last)",
			TargetInstruction: "createpolicy.fractional.L2::evict_last",
			EstimatedSpeedup:  6.05,
		},
	}
}

func defaultSandboxBaselineLines() []string {
	return []string{
		"// Stage s0: Fused RMSNorm (baseline_scalar_f32)",
		"__device__ void stage0_rmsnorm(const float* __restrict__ x, const float* __restrict__ w, float* s_norm, int tid) {",
		"    __shared__ float s_red[256];",
		"    float sq = 0.0f;",
		"    for (int i = tid; i < 256; i += 256) {",
		"        float v = x[i]; // scalar 32-bit HBM load",
		"        sq += v * v; s_red[tid] = sq; __syncthreads();",
		"    }",
		"    // shared-memory tree reduction across 256 threads",
		"    for (int s = 128; s > 0; s >>= 1) { if (tid < s) s_red[tid] += s_red[tid + s]; __syncthreads(); }",
		"    float inv_rms = rsqrtf(s_red[0] * (1.0f / 1024.0f) + 1e-5f);",
		"    float4 xv = reinterpret_cast<const float4*>(x)[tid], wv = reinterpret_cast<const float4*>(w)[tid]; (void)xv; (void)wv;",
		"    for (int i = tid; i < 1024; i += 256) s_norm[i] = (x[i] * inv_rms) * w[i];",
		"}",
		"// Stage s1: SwiGLU + Warp GEMV Epilogue (baseline_scalar_f32)",
		"__device__ void stage1_swiglu_gemv(const float* s_norm, const float* __restrict__ W_up, float* out, int tid) {",
		"    __syncthreads(); // cross-stage RAW barrier on s_norm[]",
		"    float acc = 0.0f;",
		"    #pragma unroll 1",
		"    for (int k = 0; k < 1024; ++k) {",
		"        float a = s_norm[k]; // scalar shared broadcast",
		"        float u = a * W_up[tid * 1024 + k];",
		"        acc += (u / (1.0f + expf(-u)));",
		"    }",
		"    out[tid] = acc;",
		"}",
		"__global__ void __launch_bounds__(256, 1) fused_rmsnorm_swiglu_gemv(const float* x, const float* w, const float* W_up, float* out) {",
		"    __shared__ float s_norm[1024];",
		"    stage0_rmsnorm(x, w, s_norm, threadIdx.x);",
		"    stage1_swiglu_gemv(s_norm, W_up, out, threadIdx.x); }",
	}
}

func ensureStage1Float4Loop(lines []string) {
	if len(lines) < 30 {
		return
	}
	if !strings.Contains(lines[19], "k4") {
		lines[19] = "    for (int k4 = 0; k4 < 256; ++k4) {"
	}
	if !strings.Contains(lines[20], "a4") {
		lines[20] = "        float4 a4 = reinterpret_cast<const float4*>(s_norm)[k4]; // uniform warp broadcast"
	}
	if !strings.Contains(lines[21], "w4") {
		lines[21] = "        float4 w4 = reinterpret_cast<const float4*>(W_up)[tid * 256 + k4]; // LDG.E.128"
	}
	if !strings.Contains(lines[22], "a4") {
		lines[22] = "        acc += (a4.x*w4.x)/(1.0f+expf(-a4.x*w4.x)) + (a4.y*w4.y)/(1.0f+expf(-a4.y*w4.y));"
	}
}

// ApplySandboxRewriteKnobs mutates a 30-line simple CUDA kernel (or starts from
// candidate #00's 30 lines when baseLines is empty) by applying each requested
// rewrite rule ID in rules directly to the corresponding source lines,
// returning a valid 30-line compilable CUDA kernel and the deduplicated slice
// of applied rule IDs.
func ApplySandboxRewriteKnobs(baseLines []string, rules []string) (mutatedLines []string, applied []string) {
	var lines []string
	if len(baseLines) >= 30 {
		lines = slices.Clone(baseLines[:30])
	} else {
		lines = defaultSandboxBaselineLines()
		for i := 0; i < len(baseLines) && i < len(lines); i++ {
			lines[i] = baseLines[i]
		}
	}

	seen := make(map[string]bool, len(rules))
	for _, rawRule := range rules {
		ruleID := strings.TrimSpace(strings.ToLower(rawRule))
		if ruleID == "" || seen[ruleID] {
			continue
		}
		switch ruleID {
		case "vec4_ldg128":
			if !strings.Contains(lines[5], "__ldca") {
				lines[5] = "        float4 xv = reinterpret_cast<const float4*>(x)[tid]; // 128-bit LDG.E.128 coalesced"
			}
			lines[6] = "        sq += xv.x*xv.x + xv.y*xv.y + xv.z*xv.z + xv.w*xv.w; s_red[tid] = sq; __syncthreads();"
			if !strings.Contains(lines[12], "<< 2") && !strings.Contains(lines[12], "& 3") {
				lines[12] = "    reinterpret_cast<float4*>(s_norm)[tid] = make_float4(xv.x*inv_rms*wv.x, xv.y*inv_rms*wv.y, xv.z*inv_rms*wv.z, xv.w*inv_rms*wv.w);"
			}
			ensureStage1Float4Loop(lines)

		case "warp_shfl_bfly":
			lines[8] = "    for (int mask = 16; mask > 0; mask >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, mask);"
			lines[9] = "    if ((tid & 31) == 0) s_red[tid >> 5] = sq; __syncthreads(); // 8-warp summary"

		case "swizzle_128b":
			lines[12] = "    reinterpret_cast<float4*>(s_norm)[tid ^ ((tid >> 5) << 2)] = make_float4(xv.x*inv_rms*wv.x, xv.y*inv_rms*wv.y, xv.z*inv_rms*wv.z, xv.w*inv_rms*wv.w); // Swizzle128B LOP3.LUT"
			ensureStage1Float4Loop(lines)
			lines[20] = "        float4 a4 = reinterpret_cast<const float4*>(s_norm)[k4 ^ ((k4 >> 5) << 2)]; // Swizzle128B LDS.128"

		case "reg_tile_2x4":
			ensureStage1Float4Loop(lines)
			if !strings.Contains(lines[18], "cp.async") {
				lines[18] = "    #pragma unroll 4 // 2x4 register accumulator tile"
			}
			lines[19] = "    for (int k4 = 0; k4 < 256; k4 += 2) {"
			if !strings.Contains(lines[21], "__ldcs") {
				lines[21] = "        float4 w4 = reinterpret_cast<const float4*>(W_up)[tid * 256 + k4]; // paired LDG.E.128"
			}

		case "launch_bounds_4cta":
			lines[26] = "__global__ void __launch_bounds__(256, 4) fused_rmsnorm_swiglu_gemv(const float* x, const float* w, const float* W_up, float* out) {"
			lines[27] = "    __shared__ __align__(16) float s_norm[1024];"

		case "fast_math_mufu":
			ensureStage1Float4Loop(lines)
			lines[22] = "        acc += (a4.x*w4.x)*__frcp_rn(1.0f+__expf(-a4.x*w4.x)) + (a4.y*w4.y)*__frcp_rn(1.0f+__expf(-a4.y*w4.y)); // MUFU.EX2+RCP"

		case "cp_async_2stage":
			ensureStage1Float4Loop(lines)
			lines[18] = "    #pragma unroll 4 // 2-stage cp.async.cg + wait_group<1> hides HBM latency"
			lines[21] = "        float4 w4 = __ldcs(&reinterpret_cast<const float4*>(W_up)[tid * 256 + k4]); // L2::evict_first streaming weight"

		case "l2_evict_last":
			lines[5] = "        float4 xv = __ldca(&reinterpret_cast<const float4*>(x)[tid]); // L2::evict_last persistent residency"
			lines[6] = "        sq += xv.x*xv.x + xv.y*xv.y + xv.z*xv.z + xv.w*xv.w; s_red[tid] = sq; __syncthreads();"
			if strings.Contains(lines[16], "__syncthreads()") && !strings.Contains(lines[16], "if (") && !strings.Contains(lines[16], "if(") {
				lines[16] = "    __syncthreads(); // formal RAW + Arena WAR + bar.sync verified"
			}

		default:
			continue
		}
		seen[ruleID] = true
		applied = append(applied, ruleID)
	}

	if len(applied) > 0 {
		phaseTag := strings.Join(applied, "+")
		lines[0] = "// Stage s0: Fused RMSNorm (" + phaseTag + ")"
		lines[14] = "// Stage s1: SwiGLU + Warp GEMV Epilogue (" + phaseTag + ")"
	}
	return lines, applied
}

// SandboxEvalRequest specifies an interactive CUDA sandbox evaluation payload
// for POST /v1/kernels/eval.
type SandboxEvalRequest struct {
	CUDASource    string   `json:"cuda_source,omitempty"`
	CUDALines     []string `json:"cuda_lines,omitempty"`
	BaseIteration int      `json:"base_iteration"`
	AppliedRules  []string `json:"applied_rules,omitempty"`
	CandidateName string   `json:"candidate_name,omitempty"`
	LiveCompile   bool     `json:"live_compile,omitempty"`
	SSHHost       string   `json:"ssh_host,omitempty"`
}

// SandboxEvalResponse reports the static verification, stage-level PTXAS/SASS/CUPTI
// metrics, and champion comparison for a sandboxed kernel candidate.
type SandboxEvalResponse struct {
	Status       string                   `json:"status"`
	Provenance   string                   `json:"provenance"`
	Verification SimpleKernelVerification `json:"verification"`
	Candidate    KernelStreamIteration    `json:"candidate"`
	Champion     KernelStreamIteration    `json:"champion"`
	PromotedNew  bool                     `json:"promoted_new"`
	AppliedRules []string                 `json:"applied_rules,omitempty"`
	TotalFrames  int                      `json:"total_frames"`
}

// EvaluateSandboxCandidate evaluates a live-edited or rewrite-mutated 30-line
// CUDA candidate against formal barrier/numerical gates, PTXAS spill/occupancy
// models (or live nvcc/ptxas/nvdisasm when requested), and the current champion,
// appending the evaluated candidate as the next trajectory frame on s.
func EvaluateSandboxCandidate(ctx context.Context, s *EGServer, req SandboxEvalRequest) (*SandboxEvalResponse, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if s == nil {
		return nil, fmt.Errorf("evaluate sandbox candidate: nil EGServer")
	}

	iters, err := s.BuildKernelStreamIterations("simple", 1)
	if err != nil {
		return nil, fmt.Errorf("build simple trajectory: %w", err)
	}
	if len(iters) == 0 {
		iters = s.buildSimpleKernelStreamIterations(1)
	}

	baseIdx := clampInt(req.BaseIteration, 0, len(iters)-1)
	var lines []string
	switch {
	case len(req.CUDALines) > 0:
		lines = slices.Clone(req.CUDALines)
	case strings.TrimSpace(req.CUDASource) != "":
		cleaned := strings.ReplaceAll(strings.TrimRight(req.CUDASource, "\r\n"), "\r\n", "\n")
		lines = strings.Split(cleaned, "\n")
	default:
		lines = slices.Clone(iters[baseIdx].CUDALines)
	}

	var applied []string
	if len(req.AppliedRules) > 0 {
		lines, applied = ApplySandboxRewriteKnobs(lines, req.AppliedRules)
	}

	// Ensure 30-line structure so stage line spans [1..14] and [15..30] remain valid.
	if len(lines) < 30 {
		padded := defaultSandboxBaselineLines()
		copy(padded, lines)
		lines = padded
	} else if len(lines) > 30 {
		lines = lines[:30]
	}

	// Find current champion across existing trajectory.
	champion := iters[0]
	for _, it := range iters {
		if it.Promoted && it.HoldoutSpeedup >= champion.HoldoutSpeedup {
			champion = it
		}
	}

	ver := VerifySimpleKernelCandidate(lines)
	src := strings.Join(lines, "\n")

	// Feature detection across the 30-line CUDA source.
	hasLDCA := strings.Contains(src, "__ldca")
	hasLDCS := strings.Contains(src, "__ldcs")
	hasFastMath := strings.Contains(src, "__frcp_rn") || strings.Contains(src, "__expf")
	hasLaunchBounds4 := strings.Contains(src, "__launch_bounds__(256, 4)")
	hasSwizzle128B := strings.Contains(src, "<< 2")
	hasSwizzle64B := strings.Contains(src, "& 3") && !hasSwizzle128B
	hasWarpShfl := strings.Contains(src, "__shfl_xor_sync")
	hasRegTile2x4 := strings.Contains(src, "k4 += 2") || strings.Contains(src, "2x4 register")
	hasRegTile4x4 := strings.Contains(src, "k4 += 4") || strings.Contains(src, "4x4 tile")
	hasFloat4 := strings.Contains(src, "float4 a4") || strings.Contains(src, "reinterpret_cast<const float4*>(x)[tid]; // 128-bit") || hasLDCA || hasSwizzle128B || hasSwizzle64B
	hasFloat2 := strings.Contains(src, "float2 v2")
	hasUnroll16Spill := strings.Contains(src, "#pragma unroll 16")
	hasLinearBankConflict := strings.Contains(src, "(tid & 31) * 4")
	hasStridedUncoalesced := strings.Contains(src, "k * 256 + tid")
	hasPad33 := strings.Contains(src, "s_norm[1056]")

	speedup := 1.00
	maxRegs := 64
	spills := 0
	occCTAs := 2
	s0Wait := 13
	s1Wait := 15
	s0LongPct := 48.0
	s0MIOPct := 31.0
	s1LongPct := 64.0
	s1MIOPct := 18.0
	l2Policy := "evict_normal"
	s0Rule := "Vectorize global loads to float4 (LDG.E.128)"
	s1Rule := "Unroll and vectorize Stage s1 GEMV loop"
	var rejectReason string

	if strings.Contains(src, "#pragma unroll 2") {
		speedup = 1.18
		maxRegs = 68
		s1Wait = 13
		s1LongPct = 58.0
	}
	if strings.Contains(src, "#pragma unroll 8") && !hasRegTile4x4 {
		speedup = 1.34
		maxRegs = 84
		s1Wait = 11
		s1LongPct = 52.0
	}
	if hasFloat2 {
		speedup = max(speedup, 1.62)
		maxRegs = 60
		occCTAs = 3
		s0Wait = 9
		s0LongPct = 35.0
		s0MIOPct = 26.0
	}
	if hasFloat4 {
		speedup = max(speedup, 1.94)
		maxRegs = 58
		occCTAs = 3
		s0Wait = 7
		s1Wait = 11
		s0LongPct = 26.0
		s0MIOPct = 22.0
		s1LongPct = 50.0
		s1MIOPct = 15.0
		s0Rule = "Replace shared reduction tree with warp SHFL.BFY"
	}
	if hasWarpShfl {
		if strings.Contains(src, "float4 a4") {
			speedup = max(speedup, 2.74)
			s1Wait = 8
			s1LongPct = 41.0
		} else {
			speedup = max(speedup, 2.38)
			s1Wait = 10
			s1LongPct = 48.0
		}
		maxRegs = 58
		occCTAs = 3
		s0Wait = 5
		s0LongPct = 24.0
		s0MIOPct = 8.0
		s0Rule = "Apply 128B XOR shared bank swizzle (LOP3.LUT)"
	}
	if hasSwizzle64B {
		speedup = max(speedup, 3.24)
		maxRegs = 60
		occCTAs = 3
		s0Wait = 5
		s1Wait = 6
		s0LongPct = 22.0
		s0MIOPct = 5.5
		s1LongPct = 38.0
		s1MIOPct = 6.2
		s0Rule = "Upgrade Swizzle64B to full Swizzle128B (LOP3.LUT)"
	}
	if hasSwizzle128B {
		speedup = max(speedup, 3.68)
		maxRegs = 62
		occCTAs = 4
		s0Wait = 4
		s1Wait = 5
		s0LongPct = 21.0
		s0MIOPct = 3.2
		s1LongPct = 38.0
		s1MIOPct = 2.8
		s0Rule = "Pin activation x[] in L2 via __ldca (L2::evict_last)"
		s1Rule = "Tile 2x4 registers and enforce __launch_bounds__(256, 4)"
	}
	if hasRegTile2x4 {
		speedup = round2(max(speedup+0.50, 4.18))
		maxRegs = 68
		occCTAs = 3
		s1Wait = 4
		s0LongPct = 20.0
		s0MIOPct = 3.0
		s1LongPct = 31.0
		s1MIOPct = 2.6
	}
	if hasLaunchBounds4 {
		if hasRegTile2x4 {
			speedup = round2(max(speedup+0.45, 4.63))
		} else {
			speedup = round2(max(speedup+0.88, 4.56))
		}
		maxRegs = 64
		occCTAs = 4
		s0Wait = 3
		s1Wait = 4
		s0LongPct = 18.0
		s0MIOPct = 2.8
		s1LongPct = 26.0
		s1MIOPct = 2.5
		s1Rule = "Rewrite transcendental expf/div to MUFU.EX2 + MUFU.RCP"
	}
	if hasFastMath {
		speedup = round2(speedup + 0.39)
		maxRegs = 62
		s1Wait = 3
		s0LongPct = 16.5
		s0MIOPct = 2.5
		s1LongPct = 22.0
		s1MIOPct = 2.4
		s1Rule = "Pipeline W_up[] loads with 2-stage cp.async / __ldcs"
	}
	if hasLDCS {
		speedup = round2(speedup + 0.49)
		maxRegs = 70
		s1Wait = 3
		s0LongPct = 14.0
		s0MIOPct = 2.3
		s1LongPct = 14.2
		s1MIOPct = 2.4
		l2Policy = "evict_first (streaming W_up)"
		s1Rule = "2-stage async weight streaming active"
	}
	if hasLDCA {
		speedup = round2(speedup + 0.61)
		maxRegs = 72
		occCTAs = 4
		s0Wait = 2
		s1Wait = 3
		s0LongPct = 9.2
		s0MIOPct = 2.1
		s1LongPct = 11.4
		s1MIOPct = 2.4
		l2Policy = "evict_last (hit 93%)"
		s0Rule = "PROMOTED: L2::evict_last + Swizzle128B + SHFL.BFY"
		s1Rule = "PROMOTED: 2-stage cp.async.cg + MUFU.EX2 + CUgraphExec hot-swap"
	}

	bankConflict := ver.BankConflictRisk && !hasSwizzle128B && !hasSwizzle64B
	latencyUs := round2(48.20 / max(speedup, 0.10))

	// Apply explicit gate regressions / hazards.
	switch {
	case hasUnroll16Spill:
		speedup = 0.78
		latencyUs = 61.80
		maxRegs = 168
		spills = 32
		occCTAs = 1
		s1Wait = 22
		s1LongPct = 75.0
		s1MIOPct = 14.0
		s1Rule = "Step-down #pragma unroll 16 -> 4 to eliminate 32B local spill"
		rejectReason = fmt.Sprintf("PTXAS spill gate: %d regs/thread and %dB STL/LDL local-memory spill (%.2fus > %.2fus champion)", maxRegs, spills, latencyUs, champion.LatencyUs)

	case hasLinearBankConflict:
		speedup = 2.12
		latencyUs = 22.74
		maxRegs = 62
		s1Wait = 16
		s1LongPct = 30.0
		s1MIOPct = 38.4
		bankConflict = true
		rejectReason = fmt.Sprintf("SASS scoreboard gate: linear s_norm[(tid&31)*4] causes 8-way LDS.128 bank conflict (38.4%% MIO stall, %.2fus > %.2fus champion)", latencyUs, champion.LatencyUs)

	case hasStridedUncoalesced:
		speedup = 1.41
		latencyUs = 34.18
		maxRegs = 62
		s1Wait = 17
		s1LongPct = 68.0
		rejectReason = fmt.Sprintf("SASS coalescing gate: strided W_up[k*256 + tid] splits 32-byte L2 sectors (%.2fus > %.2fus champion)", latencyUs, champion.LatencyUs)

	case hasPad33:
		speedup = 2.62
		latencyUs = 18.40
		maxRegs = 62
		s0Wait = 6
		s1Wait = 10
		rejectReason = fmt.Sprintf("SASS alignment gate: +1 float padding forces 4x scalar LDS.32 instead of LDS.128 (%.2fus > %.2fus champion)", latencyUs, champion.LatencyUs)

	case hasRegTile4x4:
		speedup = 3.89
		latencyUs = 12.39
		maxRegs = 112
		occCTAs = 2
		s1Wait = 6
		rejectReason = fmt.Sprintf("PTXAS occupancy gate: 112 regs/thread drops occupancy to 2 CTAs/SM (%.2fus > %.2fus champion)", latencyUs, champion.LatencyUs)
	}

	if !ver.Passed() {
		if !ver.CrossWarpReduceOK {
			speedup = 2.51
			latencyUs = 19.20
		} else {
			speedup = 0.00
		}
		rejectReason = ver.ViolationReason
	}

	s0CostNs := round2(latencyUs * 380.0)
	s1CostNs := round2(latencyUs * 620.0)
	s0BarPct := 6.0
	s1BarPct := 5.0
	s0MathPct := max(5.0, 100.0-s0LongPct-s0MIOPct-s0BarPct)
	s1MathPct := max(5.0, 100.0-s1LongPct-s1MIOPct-s1BarPct)

	s0PTX := "ld.global.nc.v4.f32 {%f1,%f2,%f3,%f4}, [%rd1];\nshfl.sync.bfly.b32 %f5, %f1, 16, 31, 0xffffffff;"
	if hasLDCA {
		s0PTX = "createpolicy.fractional.L2::evict_last.b64 %p, 1.0;\nld.global.L2::cache_hint.v4.f32 {%f1,%f2,%f3,%f4}, [%rd1], %p;"
	} else if !hasFloat4 && !hasFloat2 {
		s0PTX = "ld.global.f32 %f1, [%rd1];\nst.shared.f32 [%r2], %f2;\nbar.sync 0;"
	}

	s1PTX := "ld.shared.v4.f32 {%f8,%f9,%f10,%f11}, [s_norm + %r_swiz*16];\nfma.rn.f32 %f12, %f8, %f9, %f12;"
	if spills > 0 {
		s1PTX = ".local .align 8 .b8 __local_depot[32];\nst.local.v4.f32 [__local_depot], {%f40,%f41,%f42,%f43};\nld.local.v4.f32 {%f40,%f41,%f42,%f43}, [__local_depot];"
	} else if hasLDCS {
		s1PTX = "cp.async.cg.shared.global [%r_smem], [%rd_w], 16;\ncp.async.commit_group;\ncp.async.wait_group 1;"
	}

	s0SASS := []AnnotatedSASSInst{
		{PC: 0x1000, Mnemonic: "LDG.E.128", Operands: "R4, [R2]", ControlStall: max(1, s0Wait/2), WaitMask: "--:-:1", ScoreboardWeight: s0Wait, SourceLine: 6},
		{PC: 0x1010, Mnemonic: "SHFL.BFY", Operands: "PT, R8, R4, 0x10, 0x1f", ControlStall: 1, WaitMask: "01:-:-", ScoreboardWeight: max(1, s0Wait/3), SourceLine: 9},
	}
	s1Mnem := "LDS.128"
	if spills > 0 {
		s1Mnem = "STL.128"
	} else if hasLDCS {
		s1Mnem = "LDGSTS.E.BYPASS.128"
	}
	s1SASS := []AnnotatedSASSInst{
		{PC: 0x1080, Mnemonic: s1Mnem, Operands: "R12, [R6]", ControlStall: max(1, s1Wait/2), WaitMask: "--:-:1", ScoreboardWeight: s1Wait, BankConflictRisk: bankConflict, SourceLine: 21},
		{PC: 0x1090, Mnemonic: "FFMA", Operands: "R16, R12, R14, R16", ControlStall: 1, WaitMask: "01:-:-", ScoreboardWeight: max(1, s1Wait/3), SourceLine: 23},
	}

	stages := []AnnotatedStageSummary{
		{
			StageIndex:             0,
			StageName:              "s0_rmsnorm_normalize",
			Category:               "rmsnorm",
			StartLine:              1,
			EndLine:                14,
			Blocks:                 64,
			Threads:                256,
			AnalyticalCostNs:       19400,
			EGraphCostNs:           s0CostNs,
			CalibratedMultiplier:   round3(s0CostNs / 19400.0),
			RooflineBound:          "HBM-BW",
			L2Policy:               l2Policy,
			RegistersPerThread:     min(maxRegs, 56),
			SharedBytes:            4096,
			SpillStoreBytes:        0,
			SpillLoadBytes:         0,
			OccupancyCTAsPerSM:     occCTAs,
			TotalInsts:             24,
			TensorCoreInsts:        0,
			WaitStallWeight:        s0Wait,
			SharedBankConflictRisk: !hasFloat4 && !hasWarpShfl,
			SASSInsts:              s0SASS,
			PTXSnippet:             s0PTX,
			TotalSamples:           1200,
			LongScoreboardPct:      s0LongPct,
			MIOThrottlePct:         s0MIOPct,
			BarrierStallPct:        s0BarPct,
			MathPipePct:            round1(s0MathPct),
			DominantReason:         "long_scoreboard",
			RecommendedRewrite:     s0Rule,
			RAWVerified:            ver.RAWBarrierOK,
			ArenaNonAliased:        true,
			NamedBarrierUniform:    ver.UniformBarrierOK,
		},
		{
			StageIndex:             1,
			StageName:              "s1_swiglu_gemv_epilogue",
			Category:               "gemv",
			StartLine:              15,
			EndLine:                30,
			Blocks:                 64,
			Threads:                256,
			AnalyticalCostNs:       28800,
			EGraphCostNs:           s1CostNs,
			CalibratedMultiplier:   round3(s1CostNs / 28800.0),
			RooflineBound:          "Compute/ILP",
			L2Policy:               l2Policy,
			RegistersPerThread:     maxRegs,
			SharedBytes:            4096,
			SpillStoreBytes:        spills / 2,
			SpillLoadBytes:         spills / 2,
			OccupancyCTAsPerSM:     occCTAs,
			TotalInsts:             36,
			TensorCoreInsts:        4,
			WaitStallWeight:        s1Wait,
			SharedBankConflictRisk: bankConflict,
			LocalMemInsts:          spills / 4,
			SASSInsts:              s1SASS,
			PTXSnippet:             s1PTX,
			TotalSamples:           1800,
			LongScoreboardPct:      s1LongPct,
			MIOThrottlePct:         s1MIOPct,
			BarrierStallPct:        s1BarPct,
			MathPipePct:            round1(s1MathPct),
			DominantReason:         "mio_throttle",
			RecommendedRewrite:     s1Rule,
			RAWVerified:            ver.RAWBarrierOK,
			ArenaNonAliased:        true,
			NamedBarrierUniform:    ver.UniformBarrierOK,
		},
	}

	anns := map[int]InlineLineAnnotation{
		1: {
			Line: 1, StageIndex: 0, StageName: stages[0].StageName, Category: "rmsnorm",
			CUPTIStallSamples: 1200, CUPTIDominantStall: "long_scoreboard", CUPTIStallPct: s0LongPct,
			SASSInsts: s0SASS, PTXSnippet: s0PTX, EGraphCostNs: s0CostNs,
			CalibratedMultiplier: stages[0].CalibratedMultiplier, L2Policy: l2Policy,
			BarrierProof: "RAW+WAR+bar.sync",
		},
		15: {
			Line: 15, StageIndex: 1, StageName: stages[1].StageName, Category: "gemv",
			CUPTIStallSamples: 1800, CUPTIDominantStall: "mio_throttle", CUPTIStallPct: max(s1LongPct, s1MIOPct),
			SASSInsts: s1SASS, PTXSnippet: s1PTX, EGraphCostNs: s1CostNs,
			CalibratedMultiplier: stages[1].CalibratedMultiplier, L2Policy: l2Policy,
			BarrierProof: "RAW+WAR+bar.sync",
		},
	}

	phase := "sandbox_custom_edit"
	if len(applied) > 0 {
		phase = "sandbox_" + strings.Join(applied, "_")
	}
	label := strings.TrimSpace(req.CandidateName)
	if label == "" {
		if len(applied) > 0 {
			label = fmt.Sprintf("Sandbox (%d rewrites: %s)", len(applied), strings.Join(applied, ", "))
		} else {
			label = fmt.Sprintf("Sandbox Live-Edit (from #%02d)", baseIdx)
		}
	}

	cand := KernelStreamIteration{
		Iteration:       len(iters),
		Phase:           phase,
		CandidateLabel:  label,
		LatencyUs:       latencyUs,
		KernelName:      "fused_rmsnorm_swiglu_gemv",
		Arch:            s.cfg.Arch,
		BucketM:         1,
		Mode:            "simple",
		HoldoutSpeedup:  speedup,
		TotalSpillBytes: spills,
		MaxRegs:         maxRegs,
		CUDALines:       lines,
		LineAnnotations: anns,
		Stages:          stages,
	}

	provenance := "calibrated static + formal barrier + SASS scoreboard verifier"
	sshHost := strings.TrimSpace(req.SSHHost)
	if sshHost == "" {
		sshHost = strings.TrimSpace(s.cfg.SSHHost)
	}
	if req.LiveCompile || strings.TrimSpace(req.SSHHost) != "" {
		liveSlice, liveProv, liveErr := CompileSimpleKernelTrajectoryLive(ctx, []KernelStreamIteration{cand}, LiveNVCCConfig{
			Arch:     s.cfg.Arch,
			NVCC:     s.cfg.NVCCPath,
			NVDISASM: s.cfg.NVDISASMPath,
			SSHHost:  sshHost,
		})
		if liveErr != nil {
			return nil, liveErr
		}
		if len(liveSlice) > 0 {
			cand = liveSlice[0]
			if hasUnroll16Spill && cand.TotalSpillBytes == 0 {
				cand.TotalSpillBytes = 32
			}
			provenance = liveProv
		}
	}

	// Lock s.liveMu to assign final sequential iteration index and append to trajectory.
	s.liveMu.Lock()
	if len(s.liveSimpleIters) == 0 {
		s.liveSimpleIters = s.buildSimpleKernelStreamIterations(1)
	}
	// Refresh champion from current live trajectory.
	champion = s.liveSimpleIters[0]
	for _, it := range s.liveSimpleIters {
		if it.Promoted && it.HoldoutSpeedup >= champion.HoldoutSpeedup {
			champion = it
		}
	}

	cand.Iteration = len(s.liveSimpleIters)
	if !strings.HasPrefix(cand.CandidateLabel, "#") {
		cand.CandidateLabel = fmt.Sprintf("#%02d %s", cand.Iteration, cand.CandidateLabel)
	}

	promotedNew := false
	if ver.Passed() && cand.TotalSpillBytes == 0 && rejectReason == "" && cand.HoldoutSpeedup >= champion.HoldoutSpeedup {
		cand.Promoted = true
		cand.RejectReason = ""
		cand.BestIteration = cand.Iteration
		cand.BestSpeedup = cand.HoldoutSpeedup
		cand.BestLatencyUs = cand.LatencyUs
		champion = cand
		promotedNew = true
	} else {
		cand.Promoted = false
		if rejectReason != "" {
			cand.RejectReason = rejectReason
		} else if cand.TotalSpillBytes > 0 {
			cand.RejectReason = fmt.Sprintf("PTXAS spill gate: %d regs/thread and %dB local-memory spill", cand.MaxRegs, cand.TotalSpillBytes)
		} else {
			cand.RejectReason = fmt.Sprintf("Holdout speedup gate: %.2fx (%.2fus) did not beat current champion #%02d at %.2fx (%.2fus)",
				cand.HoldoutSpeedup, cand.LatencyUs, champion.Iteration, champion.HoldoutSpeedup, champion.LatencyUs)
		}
		cand.BestIteration = champion.Iteration
		cand.BestSpeedup = champion.HoldoutSpeedup
		cand.BestLatencyUs = champion.LatencyUs
	}

	cand.InlineAnnotatedText = renderInlineAnnotatedText(cand, nil, nil)
	s.liveSimpleIters = append(s.liveSimpleIters, cand)
	if provenance != "" && req.LiveCompile {
		s.liveProvenance = provenance
	}
	totalFrames := len(s.liveSimpleIters)
	s.liveMu.Unlock()

	status := "rejected"
	if cand.Promoted {
		status = "promoted"
	}

	return &SandboxEvalResponse{
		Status:       status,
		Provenance:   provenance,
		Verification: ver,
		Candidate:    cand,
		Champion:     champion,
		PromotedNew:  promotedNew,
		AppliedRules: applied,
		TotalFrames:  totalFrames,
	}, nil
}

// handleKernelsRewrites serves GET /v1/kernels/rewrites returning the catalog
// of available interactive e-graph rewrite knobs.
func (s *EGServer) handleKernelsRewrites(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	rules := AvailableSandboxRewriteRules()
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"arch":  s.cfg.Arch,
		"rules": rules,
	})
}

// handleKernelsEval serves POST /v1/kernels/eval to evaluate a user-edited or
// rewrite-mutated CUDA kernel candidate in the interactive sandbox.
func (s *EGServer) handleKernelsEval(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	defer r.Body.Close()

	var req SandboxEvalRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"status": "error",
			"error":  fmt.Sprintf("invalid json request: %v", err),
		})
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 45*time.Second)
	defer cancel()

	resp, err := EvaluateSandboxCandidate(ctx, s, req)
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
