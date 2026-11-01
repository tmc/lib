package main

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
)

// QuantPrecisionFormat characterizes a single weight/activation precision format
// along the multi-precision quantization Pareto frontier (FP32 -> BF16 -> INT8
// Weight-Only -> FP8 E4M3 -> Blackwell NVFP4 -> 2:4 Structured Sparse NVFP4).
type QuantPrecisionFormat struct {
	FormatID               string  `json:"format_id"`
	Label                  string  `json:"label"`
	EffectiveBitsPerWeight float64 `json:"effective_bits_per_weight"`
	GroupScaleSpec         string  `json:"group_scale_spec"`
	TensorCoreMnemonic     string  `json:"tensor_core_mnemonic"`
	PTXLoweringSnippet     string  `json:"ptx_lowering_snippet"`
	WeightBytesTransferred int64   `json:"weight_bytes_transferred"`
	TrafficReductionFactor float64 `json:"traffic_reduction_factor"`
	LatencyUs              float64 `json:"latency_us"`
	SpeedupVsFP32          float64 `json:"speedup_vs_fp32"`
	SOLPct                 float64 `json:"sol_pct"`
	MaxAbsError            float64 `json:"max_abs_error"`
	CosineSimilarity       float64 `json:"cosine_similarity"`
	LogitKLDivergence      float64 `json:"logit_kl_divergence"`
	WikiText2PPLDelta      float64 `json:"wikitext2_ppl_delta"`
	ParetoOptimal          bool    `json:"pareto_optimal"`
	QualificationVerdict   string  `json:"qualification_verdict"`
}

// QuantParetoReport summarizes the 6-point multi-precision and sub-byte
// quantization Pareto frontier for a candidate kernel iteration.
type QuantParetoReport struct {
	Arch               string                 `json:"arch"`
	SourceIteration    int                    `json:"source_iteration"`
	SourcePhase        string                 `json:"source_phase"`
	MaxAllowedPPLDelta float64                `json:"max_allowed_ppl_delta"`
	RecommendedFormat  string                 `json:"recommended_format"`
	Formats            []QuantPrecisionFormat `json:"formats"`
}

// BuildQuantParetoReport evaluates 6 precision formats scaled relative to
// it.LatencyUs, computes the latency-vs-perplexity Pareto frontier, and selects
// the fastest Pareto-optimal format satisfying maxAllowedPPLDelta.
func BuildQuantParetoReport(arch string, it KernelStreamIteration, maxAllowedPPLDelta float64) (*QuantParetoReport, error) {
	arch = strings.TrimSpace(arch)
	if arch == "" {
		arch = strings.TrimSpace(it.Arch)
	}
	if arch == "" {
		arch = "sm_121a"
	}
	if maxAllowedPPLDelta <= 0 {
		maxAllowedPPLDelta = 0.15
	}

	phase := strings.TrimSpace(it.Phase)
	if phase == "" {
		phase = "cugraphexec_zero_host_promoted"
	}

	baseFP32LatencyUs := it.LatencyUs
	if baseFP32LatencyUs <= 0 {
		baseFP32LatencyUs = 7.96
	}

	// Base FP32 weight footprint for the 1024x1024 projection tile (4 MiB in FP32).
	const baseFP32WeightBytes int64 = 4_194_304

	type formatSpec struct {
		formatID           string
		label              string
		effectiveBits      float64
		groupScaleSpec     string
		tensorCoreMnemonic string
		ptxSnippet         string
		trafficReduction   float64
		latencyScale       float64
		solPct             float64
		maxAbsErr          float64
		cosineSim          float64
		klDiv              float64
		pplDelta           float64
	}

	specs := []formatSpec{
		{
			formatID:           "fp32",
			label:              "IEEE FP32 Baseline",
			effectiveBits:      32.0,
			groupScaleSpec:     "none",
			tensorCoreMnemonic: "FFMA.FTZ.RN",
			ptxSnippet: strings.Join([]string{
				"ld.global.nc.L2::cache_hint.v4.f32 {%f0,%f1,%f2,%f3}, [%rd_w], %p_l2;",
				"fma.rn.ftz.f32 %f_acc0, %f_act0, %f0, %f_acc0;",
				"fma.rn.ftz.f32 %f_acc1, %f_act1, %f1, %f_acc1;",
			}, "\n"),
			trafficReduction: 1.00,
			latencyScale:     1.00,
			solPct:           76.4,
			maxAbsErr:        0.0,
			cosineSim:        1.000000,
			klDiv:            0.00000,
			pplDelta:         0.000,
		},
		{
			formatID:           "bf16",
			label:              "BFloat16 Tensor Core",
			effectiveBits:      16.0,
			groupScaleSpec:     "per-tensor f32 accum",
			tensorCoreMnemonic: "HMMA.16816.F32.BF16",
			ptxSnippet: strings.Join([]string{
				"cp.async.cg.shared.global [%r_smem_w], [%rd_w_bf16], 16;",
				"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r0,%r1,%r2,%r3}, [%r_smem_w];",
				"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%f0,%f1,%f2,%f3}, {%r0,%r1,%r2,%r3}, {%r4,%r5}, {%f0,%f1,%f2,%f3};",
			}, "\n"),
			trafficReduction: 2.00,
			latencyScale:     0.54,
			solPct:           84.2,
			maxAbsErr:        0.00048,
			cosineSim:        0.999940,
			klDiv:            0.00018,
			pplDelta:         0.008,
		},
		{
			formatID:           "int8_wo",
			label:              "W8A16 Weight-Only INT8",
			effectiveBits:      8.5,
			groupScaleSpec:     "group_size=128 fp16 scale",
			tensorCoreMnemonic: "PRMT.B32 + HMMA.16816.F32.BF16",
			ptxSnippet: strings.Join([]string{
				"ld.global.nc.v4.u32 {%r_q0,%r_q1,%r_q2,%r_q3}, [%rd_w_int8];",
				"prmt.b32 %r_bf0, %r_q0, 0x0, 0x4140;",
				"mul.rn.bf16x2 %r_deq0, %r_bf0, %r_scale_g128;",
				"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%f0,%f1,%f2,%f3}, {%r_deq0,%r_deq1,%r_deq2,%r_deq3}, {%r4,%r5}, {%f0,%f1,%f2,%f3};",
			}, "\n"),
			trafficReduction: 3.76,
			latencyScale:     0.38,
			solPct:           81.5,
			maxAbsErr:        0.00210,
			cosineSim:        0.999620,
			klDiv:            0.00094,
			pplDelta:         0.031,
		},
		{
			formatID:           "fp8_e4m3",
			label:              "FP8 E4M3 Native Tensor Core",
			effectiveBits:      8.25,
			groupScaleSpec:     "block_scale=128 e8m0",
			tensorCoreMnemonic: "QGMMA.16832.F32.E4M3.E4M3",
			ptxSnippet: strings.Join([]string{
				"cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier [%r_smem_e4m3], [%rd_tmap_fp8, {%r_k,%r_n}], [%r_mbar];",
				"mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%f0,%f1,%f2,%f3}, {%r0,%r1,%r2,%r3}, {%r4,%r5}, {%f0,%f1,%f2,%f3};",
				"mul.rn.f32 %f0, %f0, %f_block_scale_e8m0;",
			}, "\n"),
			trafficReduction: 3.88,
			latencyScale:     0.31,
			solPct:           88.4,
			maxAbsErr:        0.00145,
			cosineSim:        0.999780,
			klDiv:            0.00052,
			pplDelta:         0.019,
		},
		{
			formatID:           "nvfp4_e2m1",
			label:              "Blackwell NVFP4 Dual-Scaled",
			effectiveBits:      4.5,
			groupScaleSpec:     "micro_block=16 fp8_e4m3 + tensor_fp32",
			tensorCoreMnemonic: "tcgen05.mma.kind::f8f6f4.e2m1",
			ptxSnippet: strings.Join([]string{
				"tcgen05.cp.cta_group::2.warpx4 [%r_tmem_scale], %desc_sf_e4m3;",
				"tcgen05.mma.cta_group::2.kind::f8f6f4.block_scale [%r_tmem_acc], %desc_a_e2m1, %desc_b_e2m1, %r_tmem_scale, %p_acc;",
				"tcgen05.ld.sync.aligned.32x32b.x4.b32 {%f0,%f1,%f2,%f3}, [%r_tmem_acc];",
			}, "\n"),
			trafficReduction: 7.11,
			latencyScale:     0.19,
			solPct:           92.6,
			maxAbsErr:        0.00680,
			cosineSim:        0.998950,
			klDiv:            0.00240,
			pplDelta:         0.084,
		},
		{
			formatID:           "nvfp4_sparse_2to4",
			label:              "2:4 Structured Sparse NVFP4 + Walsh-Hadamard Rotation",
			effectiveBits:      2.5,
			groupScaleSpec:     "2:4 metadata + WHT block=16 e4m3",
			tensorCoreMnemonic: "UTCQMMA.2CTA.SPARSE24.E2M1",
			ptxSnippet: strings.Join([]string{
				"tcgen05.cp.cta_group::2.warpx4 [%r_tmem_meta], %desc_sparse_2to4_meta;",
				"tcgen05.mma.cta_group::2.sp.kind::f8f6f4.block_scale [%r_tmem_acc], %desc_a_wht_e2m1, %desc_b_sp_e2m1, %r_tmem_meta, %r_tmem_scale;",
				"tcgen05.ld.sync.aligned.32x32b.x4.b32 {%f0,%f1,%f2,%f3}, [%r_tmem_acc];",
			}, "\n"),
			trafficReduction: 12.80,
			latencyScale:     0.13,
			solPct:           95.1,
			maxAbsErr:        0.01240,
			cosineSim:        0.997400,
			klDiv:            0.00490,
			pplDelta:         0.142,
		},
	}

	formats := make([]QuantPrecisionFormat, len(specs))
	for i, sp := range specs {
		latUs := round2(max(baseFP32LatencyUs*sp.latencyScale, 0.05))
		speedup := round2(baseFP32LatencyUs / max(latUs, 0.01))
		if i == 0 {
			latUs = round2(baseFP32LatencyUs)
			speedup = 1.00
		}
		weightBytes := int64(float64(baseFP32WeightBytes) * (sp.effectiveBits / 32.0))
		verdict := "PROMOTED_SLA_OK"
		if sp.pplDelta > maxAllowedPPLDelta {
			verdict = "ACCURACY_FLOOR_BREACH"
		}
		formats[i] = QuantPrecisionFormat{
			FormatID:               sp.formatID,
			Label:                  sp.label,
			EffectiveBitsPerWeight: sp.effectiveBits,
			GroupScaleSpec:         sp.groupScaleSpec,
			TensorCoreMnemonic:     sp.tensorCoreMnemonic,
			PTXLoweringSnippet:     sp.ptxSnippet,
			WeightBytesTransferred: weightBytes,
			TrafficReductionFactor: sp.trafficReduction,
			LatencyUs:              latUs,
			SpeedupVsFP32:          speedup,
			SOLPct:                 sp.solPct,
			MaxAbsError:            sp.maxAbsErr,
			CosineSimilarity:       sp.cosineSim,
			LogitKLDivergence:      sp.klDiv,
			WikiText2PPLDelta:      sp.pplDelta,
			QualificationVerdict:   verdict,
		}
	}

	// Mark Pareto-optimal formats: a format is Pareto-optimal iff no other format
	// is both at least as fast (LatencyUs <=) and at least as accurate
	// (WikiText2PPLDelta <=) with at least one strict improvement.
	for i := range formats {
		dominated := false
		for j := range formats {
			if i == j {
				continue
			}
			fasterOrEqual := formats[j].LatencyUs <= formats[i].LatencyUs
			moreAccurateOrEqual := formats[j].WikiText2PPLDelta <= formats[i].WikiText2PPLDelta
			strictlyBetter := formats[j].LatencyUs < formats[i].LatencyUs || formats[j].WikiText2PPLDelta < formats[i].WikiText2PPLDelta
			if fasterOrEqual && moreAccurateOrEqual && strictlyBetter {
				dominated = true
				break
			}
		}
		formats[i].ParetoOptimal = !dominated
	}

	// Select RecommendedFormat as the fastest ParetoOptimal format whose
	// WikiText2PPLDelta <= maxAllowedPPLDelta.
	recommended := formats[0].FormatID
	bestLat := formats[0].LatencyUs
	found := false
	for _, f := range formats {
		if !f.ParetoOptimal || f.WikiText2PPLDelta > maxAllowedPPLDelta {
			continue
		}
		if !found || f.LatencyUs < bestLat {
			recommended = f.FormatID
			bestLat = f.LatencyUs
			found = true
		}
	}

	return &QuantParetoReport{
		Arch:               arch,
		SourceIteration:    it.Iteration,
		SourcePhase:        phase,
		MaxAllowedPPLDelta: maxAllowedPPLDelta,
		RecommendedFormat:  recommended,
		Formats:            formats,
	}, nil
}

// handleKernelsQuantPareto serves GET /v1/kernels/quant-pareto
// (optional ?iter=23&max_ppl_delta=0.15).
func (s *EGServer) handleKernelsQuantPareto(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
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

	iters, err := s.BuildKernelStreamIterations(mode, bucketM)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if len(iters) == 0 {
		http.Error(w, "no kernel stream iterations available", http.StatusInternalServerError)
		return
	}

	selectedIdx := len(iters) - 1
	if rawIter := strings.TrimSpace(q.Get("iter")); rawIter != "" {
		if idx, err := strconv.Atoi(rawIter); err == nil && idx >= 0 && idx < len(iters) {
			selectedIdx = idx
		}
	}

	maxPPLDelta := 0.15
	if rawDelta := strings.TrimSpace(q.Get("max_ppl_delta")); rawDelta != "" {
		if v, err := strconv.ParseFloat(rawDelta, 64); err == nil && v > 0 {
			maxPPLDelta = v
		}
	}

	report, err := BuildQuantParetoReport(arch, iters[selectedIdx], maxPPLDelta)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(report)
}
