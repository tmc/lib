package advise

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

func TestAdvice_CUDAGraphCandidate(t *testing.T) {
	tests := []struct {
		name         string
		cap          *cudacapture.Capture
		wantWarning  bool
		wantCategory Category
		wantSeverity Severity
		wantFinding  string
		wantRec      string
	}{
		{
			// 26.5% duty cycle and every launch issued after the previous
			// kernel had already retired.
			name: "short_kernels_idle_device_host_behind",
			cap: &cudacapture.Capture{
				Files: map[string][]byte{
					"capture-status.json": apiRecordsFile(t, map[int]uint64{2: 10000, 3: 30000, 4: 50000, 5: 70000}),
				},
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{
						{Type: "kernel_time", Unit: "nanoseconds"},
					},
					Sample: []*profile.Sample{
						microKernelSample("k1", 1, 1000, 4000, 1),
						microKernelSample("k2", 1, 20000, 24000, 2),
						microKernelSample("k3", 1, 40000, 45000, 3),
						microKernelSample("k4", 1, 60000, 66000, 4),
						microKernelSample("k5", 1, 80000, 84000, 5),
					},
				},
			},
			wantWarning:  true,
			wantCategory: CategoryArchitectureDispatch,
			wantSeverity: SeverityCost,
			wantFinding:  "Driver submission overhead dominates short kernel execution",
			wantRec:      "Capture sequence into a CUDA Graph (cudaGraphLaunch) to eliminate host submission latency",
		},
		{
			// Same short kernels, but back to back: the device is 98% busy, so
			// per-launch host cost is costing it nothing.
			name: "short_kernels_high_duty_cycle",
			cap: &cudacapture.Capture{
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{
						{Type: "kernel_time", Unit: "nanoseconds"},
					},
					Sample: []*profile.Sample{
						microKernelSample("k1", 1, 1000, 4000, 1),
						microKernelSample("k2", 1, 4100, 8100, 2),
						microKernelSample("k3", 1, 8200, 13200, 3),
						microKernelSample("k4", 1, 13300, 19300, 4),
						microKernelSample("k5", 1, 19400, 23400, 5),
					},
				},
			},
			// No dispatch finding of any kind belongs on a busy device.
			wantWarning:  false,
			wantCategory: CategoryArchitectureDispatch,
		},
		{
			// Idle device, but every launch was already issued before the
			// preceding kernel retired: the host is not the cause.
			name: "short_kernels_idle_device_host_ahead",
			cap: &cudacapture.Capture{
				Files: map[string][]byte{
					"capture-status.json": apiRecordsFile(t, map[int]uint64{2: 1500, 3: 21000, 4: 41000, 5: 61000}),
				},
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{
						{Type: "kernel_time", Unit: "nanoseconds"},
					},
					Sample: []*profile.Sample{
						microKernelSample("k1", 1, 1000, 4000, 1),
						microKernelSample("k2", 1, 20000, 24000, 2),
						microKernelSample("k3", 1, 40000, 45000, 3),
						microKernelSample("k4", 1, 60000, 66000, 4),
						microKernelSample("k5", 1, 80000, 84000, 5),
					},
				},
			},
			wantWarning:  false,
			wantCategory: CategoryArchitectureDispatch,
			wantFinding:  "Driver submission overhead dominates short kernel execution",
		},
		{
			name: "broken_sequence_by_memcpy",
			cap: &cudacapture.Capture{
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{
						{Type: "kernel_time", Unit: "nanoseconds"},
					},
					Sample: []*profile.Sample{
						{Value: []int64{3000}, Label: map[string][]string{"name": {"k1"}, "subtype": {"kernel"}, "stream": {"1"}}},
						{Value: []int64{4000}, Label: map[string][]string{"name": {"k2"}, "subtype": {"kernel"}, "stream": {"1"}}},
						{Value: []int64{5000}, Label: map[string][]string{"name": {"cudaMemcpyAsync"}, "subtype": {"memcpy"}, "stream": {"1"}}},
						{Value: []int64{6000}, Label: map[string][]string{"name": {"k3"}, "subtype": {"kernel"}, "stream": {"1"}}},
						{Value: []int64{4000}, Label: map[string][]string{"name": {"k4"}, "subtype": {"kernel"}, "stream": {"1"}}},
						{Value: []int64{4000}, Label: map[string][]string{"name": {"k5"}, "subtype": {"kernel"}, "stream": {"1"}}},
					},
				},
			},
			wantWarning: false,
		},
		{
			name: "broken_sequence_by_synchronization",
			cap: &cudacapture.Capture{
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{
						{Type: "kernel_time", Unit: "nanoseconds"},
					},
					Sample: []*profile.Sample{
						{Value: []int64{3000}, Label: map[string][]string{"name": {"k1"}, "subtype": {"kernel"}, "stream": {"1"}}},
						{Value: []int64{4000}, Label: map[string][]string{"name": {"k2"}, "subtype": {"kernel"}, "stream": {"1"}}},
						{Value: []int64{0}, Label: map[string][]string{"name": {"cudaStreamSynchronize"}, "subtype": {"synchronization"}, "stream": {"1"}}},
						{Value: []int64{6000}, Label: map[string][]string{"name": {"k3"}, "subtype": {"kernel"}, "stream": {"1"}}},
						{Value: []int64{4000}, Label: map[string][]string{"name": {"k4"}, "subtype": {"kernel"}, "stream": {"1"}}},
					},
				},
			},
			wantWarning: false,
		},
		{
			name: "long_kernels_above_10us",
			cap: &cudacapture.Capture{
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{
						{Type: "kernel_time", Unit: "nanoseconds"},
					},
					Sample: []*profile.Sample{
						{Value: []int64{15000}, Label: map[string][]string{"name": {"k1"}, "subtype": {"kernel"}, "stream": {"1"}}},
						{Value: []int64{20000}, Label: map[string][]string{"name": {"k2"}, "subtype": {"kernel"}, "stream": {"1"}}},
						{Value: []int64{12000}, Label: map[string][]string{"name": {"k3"}, "subtype": {"kernel"}, "stream": {"1"}}},
						{Value: []int64{14000}, Label: map[string][]string{"name": {"k4"}, "subtype": {"kernel"}, "stream": {"1"}}},
						{Value: []int64{16000}, Label: map[string][]string{"name": {"k5"}, "subtype": {"kernel"}, "stream": {"1"}}},
					},
				},
			},
			wantWarning: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			report, err := Analyze(tt.cap)
			if err != nil {
				t.Fatalf("Analyze failed: %v", err)
			}

			var found bool
			for _, item := range report.Items {
				if item.Category == tt.wantCategory && (tt.wantSeverity == "" || item.Severity == tt.wantSeverity) &&
					strings.Contains(item.Finding, tt.wantFinding) {
					found = true
					if !strings.Contains(item.ActionableRecommendation, tt.wantRec) {
						t.Errorf("recommendation = %q, want containing %q", item.ActionableRecommendation, tt.wantRec)
					}
					break
				}
			}

			if tt.wantWarning && !found {
				t.Errorf("did not find expected cuda graph candidate warning; items: %+v", report.Items)
			} else if !tt.wantWarning && found {
				t.Errorf("unexpected cuda graph candidate warning found in %+v", report.Items)
			}
		})
	}
}

func TestAdvice_PointwiseFusion(t *testing.T) {
	tests := []struct {
		name         string
		cap          *cudacapture.Capture
		wantWarning  bool
		wantCategory Category
		wantSeverity Severity
		wantFinding  string
		wantRec      string
	}{
		{
			name: "swiglu_followed_by_quantization_matching_shape",
			cap: &cudacapture.Capture{
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{
						{Type: "kernel_time", Unit: "nanoseconds"},
					},
					Sample: []*profile.Sample{
						{
							Value: []int64{50000},
							Label: map[string][]string{
								"name":    {"swiglu_forward_kernel"},
								"subtype": {"kernel"},
								"stream":  {"1"},
								"shape":   {"[4096, 4096]"},
							},
						},
						{
							Value: []int64{30000},
							Label: map[string][]string{
								"name":    {"quantize_fp8_kernel"},
								"subtype": {"kernel"},
								"stream":  {"1"},
								"shape":   {"[4096, 4096]"},
							},
						},
					},
				},
			},
			wantWarning:  true,
			wantCategory: CategoryMemoryCompute,
			wantSeverity: SeverityCost,
			wantFinding:  "Redundant global memory round-trips for intermediate tensors",
			wantRec:      "Fuse into a single kernel (e.g. fused SwiGLU + quant)",
		},
		{
			name: "pointwise_matching_buffers",
			cap: &cudacapture.Capture{
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{
						{Type: "kernel_time", Unit: "nanoseconds"},
					},
					Sample: []*profile.Sample{
						{
							Value: []int64{20000},
							Label: map[string][]string{
								"name":    {"silu_activation"},
								"subtype": {"kernel"},
								"stream":  {"2"},
								"buffer":  {"tensor_act_0"},
							},
						},
						{
							Value: []int64{15000},
							Label: map[string][]string{
								"name":    {"elementwise_scale_bias"},
								"subtype": {"kernel"},
								"stream":  {"2"},
								"buffer":  {"tensor_act_0"},
							},
						},
					},
				},
			},
			wantWarning:  true,
			wantCategory: CategoryMemoryCompute,
			wantSeverity: SeverityCost,
			wantFinding:  "Redundant global memory round-trips for intermediate tensors",
			wantRec:      "Fuse into a single kernel",
		},
		{
			name: "different_streams_no_fusion",
			cap: &cudacapture.Capture{
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{
						{Type: "kernel_time", Unit: "nanoseconds"},
					},
					Sample: []*profile.Sample{
						{
							Value: []int64{50000},
							Label: map[string][]string{
								"name":    {"swiglu_forward_kernel"},
								"subtype": {"kernel"},
								"stream":  {"1"},
								"shape":   {"[4096, 4096]"},
							},
						},
						{
							Value: []int64{30000},
							Label: map[string][]string{
								"name":    {"quantize_fp8_kernel"},
								"subtype": {"kernel"},
								"stream":  {"2"},
								"shape":   {"[4096, 4096]"},
							},
						},
					},
				},
			},
			wantWarning: false,
		},
		{
			name: "gemm_followed_by_quant_no_fusion",
			cap: &cudacapture.Capture{
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{
						{Type: "kernel_time", Unit: "nanoseconds"},
					},
					Sample: []*profile.Sample{
						{
							Value: []int64{100000},
							Label: map[string][]string{
								"name":    {"gemm_cutlass_kernel"},
								"subtype": {"kernel"},
								"stream":  {"1"},
								"shape":   {"[4096, 4096]"},
							},
						},
						{
							Value: []int64{30000},
							Label: map[string][]string{
								"name":    {"quantize_fp8_kernel"},
								"subtype": {"kernel"},
								"stream":  {"1"},
								"shape":   {"[4096, 4096]"},
							},
						},
					},
				},
			},
			wantWarning: false,
		},
		{
			name: "intervening_memcpy_no_fusion",
			cap: &cudacapture.Capture{
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{
						{Type: "kernel_time", Unit: "nanoseconds"},
					},
					Sample: []*profile.Sample{
						{
							Value: []int64{50000},
							Label: map[string][]string{
								"name":    {"swiglu_forward_kernel"},
								"subtype": {"kernel"},
								"stream":  {"1"},
								"shape":   {"[4096, 4096]"},
							},
						},
						{
							Value: []int64{10000},
							Label: map[string][]string{
								"name":    {"cudaMemcpyAsync"},
								"subtype": {"memcpy"},
								"stream":  {"1"},
							},
						},
						{
							Value: []int64{30000},
							Label: map[string][]string{
								"name":    {"quantize_fp8_kernel"},
								"subtype": {"kernel"},
								"stream":  {"1"},
								"shape":   {"[4096, 4096]"},
							},
						},
					},
				},
			},
			wantWarning: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			report, err := Analyze(tt.cap)
			if err != nil {
				t.Fatalf("Analyze failed: %v", err)
			}

			var found bool
			for _, item := range report.Items {
				if item.Category == tt.wantCategory && (tt.wantSeverity == "" || item.Severity == tt.wantSeverity) &&
					strings.Contains(item.Finding, tt.wantFinding) {
					found = true
					if !strings.Contains(item.ActionableRecommendation, tt.wantRec) {
						t.Errorf("recommendation = %q, want containing %q", item.ActionableRecommendation, tt.wantRec)
					}
					break
				}
			}

			if tt.wantWarning && !found {
				t.Errorf("did not find expected pointwise fusion warning; items: %+v", report.Items)
			} else if !tt.wantWarning && found {
				t.Errorf("unexpected pointwise fusion warning found in %+v", report.Items)
			}
		})
	}
}

func TestAdvice_BlackwellDeepSeekV4DecodeProfiling(t *testing.T) {
	// Comprehensive test synthesizing all 5 heuristics in a realistic Blackwell sm_100a
	// DeepSeek-V4 decode profiling session.
	const sass = `
	.section	.text.deepseek_decode,"ax",@progbits
deepseek_decode:
.text.deepseek_decode:
        /*0000*/                   MOV R1, c[0x0][0x28] ;
        /*0010*/                   LDG.E.128 R0, [R2] ;
        /*0020*/                   EXIT ;
`

	cap := &cudacapture.Capture{
		Metadata: []byte(`{"arch": "sm_100a", "sm_count": 136, "device": "NVIDIA GB200"}`),
		Code: cudacapture.CodeFiles{
			SASS: "kernels.sass",
		},
		Files: map[string][]byte{
			"kernels.sass":        []byte(sass),
			"capture-status.json": apiRecordsFile(t, map[int]uint64{102: 10000, 103: 30000, 104: 50000, 105: 70000}),
		},
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{
				{Type: "kernel_time", Unit: "nanoseconds"},
				{Type: "stall_scoreboard", Unit: "count"},
				{Type: "stall_mem", Unit: "count"},
			},
			Location: []*profile.Location{
				{ID: 1, Address: 0x0010},
			},
			Sample: []*profile.Sample{
				// 1 & 2. Underpopulated thread block (32 threads) and SM Grid Starvation (64 blocks < 136 SMs)
				{
					Value: []int64{8000, 0, 0},
					Label: map[string][]string{
						"kernel":  {"deepseek_decode"},
						"subtype": {"kernel"},
						"stream":  {"0"},
						"block":   {"32,1,1"},
						"grid":    {"64,1,1"},
					},
				},
				// 3. Memory-dominated LDG.E at PC 0x0010, 45% of stalls in stall_mem.
				// The counter is stall_mem: this comment used to say
				// stall_memory_throttle, which the profile does not declare and no
				// producer emits, and the finding string said so too.
				{
					Location: []*profile.Location{{ID: 1, Address: 0x0010}},
					Value:    []int64{0, 55, 45}, // 45 / 100 = 45% > 35%
					Label: map[string][]string{
						"kernel": {"deepseek_decode"},
					},
				},
				// 4. CUDA Graph candidates: 5 consecutive short kernels on stream 1
				// (< 10 µs) on an idle device, each launched after the previous
				// kernel had retired.
				microKernelSample("rms_norm", 1, 1000, 3000, 101, 0, 0),
				microKernelSample("q_proj", 1, 20000, 24000, 102, 0, 0),
				microKernelSample("k_proj", 1, 40000, 43000, 103, 0, 0),
				microKernelSample("v_proj", 1, 60000, 63000, 104, 0, 0),
				microKernelSample("rope_embed", 1, 80000, 85000, 105, 0, 0),
				// 5. Pointwise fusion: SwiGLU followed by quantization on stream 2
				{
					Value: []int64{45000, 0, 0},
					Label: map[string][]string{
						"name":    {"swiglu_mlp_gate"},
						"subtype": {"kernel"},
						"stream":  {"2"},
						"shape":   {"[2048, 7168]"},
					},
				},
				{
					Value: []int64{35000, 0, 0},
					Label: map[string][]string{
						"name":    {"quantize_fp8_linear"},
						"subtype": {"kernel"},
						"stream":  {"2"},
						"shape":   {"[2048, 7168]"},
					},
				},
			},
		},
	}

	report, err := Analyze(cap)
	if err != nil {
		t.Fatalf("Analyze failed: %v", err)
	}

	expectedFindings := []struct {
		name     string
		category Category
		finding  string
		rec      string
	}{
		{
			name:     "heuristic_1_single_warp",
			category: CategoryArchitectureCompute,
			finding:  "Single-warp thread blocks",
			rec:      "Restructure thread block geometry to >=128 threads (4 warps)",
		},
		{
			name:     "heuristic_2_grid_starvation",
			category: CategoryArchitecture,
			finding:  "launched with fewer grid blocks than the GPU has SMs",
			rec:      "Compare achieved occupancy against these grids before changing any of them",
		},
		{
			name:     "heuristic_3_memory_dominated_load",
			category: CategoryMemory,
			finding:  "Memory stalls dominate this load: stall_mem is 45.0% of stalls",
			rec:      "Check whether adjacent lane IDs access contiguous bytes",
		},
		{
			name:     "heuristic_4_cuda_graph",
			category: CategoryArchitectureDispatch,
			finding:  "Driver submission overhead dominates short kernel execution",
			rec:      "Capture sequence into a CUDA Graph (cudaGraphLaunch) to eliminate host submission latency",
		},
		{
			name:     "heuristic_5_pointwise_fusion",
			category: CategoryMemoryCompute,
			finding:  "Redundant global memory round-trips for intermediate tensors",
			rec:      "Fuse into a single kernel (e.g. fused SwiGLU + quant)",
		},
	}

	for _, ef := range expectedFindings {
		t.Run(ef.name, func(t *testing.T) {
			var found bool
			for _, item := range findings(report) {
				if item.Category == ef.category && strings.Contains(item.Finding, ef.finding) {
					found = true
					if !strings.Contains(item.ActionableRecommendation, ef.rec) {
						t.Errorf("recommendation = %q, want containing %q", item.ActionableRecommendation, ef.rec)
					}
					break
				}
			}
			if !found {
				t.Errorf("expected finding %q in category %q not found; report: %+v", ef.finding, ef.category, report.Items)
			}
		})
	}
}

func TestAdvice_SequentialRMSNormQuantFusion(t *testing.T) {
	// Verifies detection of sequential RMSNorm followed immediately by quantization.
	cap := &cudacapture.Capture{
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{
				{Type: "kernel_time", Unit: "nanoseconds"},
			},
			Sample: []*profile.Sample{
				{
					Value: []int64{4000},
					Label: map[string][]string{
						"name":    {"ds4_rms"},
						"subtype": {"kernel"},
						"stream":  {"1"},
					},
				},
				{
					Value: []int64{3000},
					Label: map[string][]string{
						"name":    {"ds4_quant_q8_1"},
						"subtype": {"kernel"},
						"stream":  {"1"},
					},
				},
			},
		},
	}

	report, err := Analyze(cap)
	if err != nil {
		t.Fatalf("Analyze failed: %v", err)
	}

	var found bool
	for _, item := range report.Items {
		if item.Category == CategoryMemoryCompute && strings.Contains(item.Finding, "Redundant global memory round-trips for normalized activations") {
			found = true
			if !strings.Contains(item.ActionableRecommendation, "Fuse normalization and quantization") {
				t.Errorf("recommendation = %q, want containing fuse normalization and quantization", item.ActionableRecommendation)
			}
			break
		}
	}
	if !found {
		t.Fatalf("expected RMSNorm->Quant fusion advisory not found; items: %+v", report.Items)
	}
}

func TestAdvice_DualGEMVPairedProjections(t *testing.T) {
	// Verifies detection of paired GEMVs operating on identical activation vectors.
	cap := &cudacapture.Capture{
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{
				{Type: "kernel_time", Unit: "nanoseconds"},
			},
			Sample: []*profile.Sample{
				{
					Value: []int64{25000},
					Label: map[string][]string{
						"name":    {"ds4_attnQA_gemv"},
						"subtype": {"kernel"},
						"stream":  {"1"},
						"buffer":  {"attn_norm_out"},
					},
				},
				{
					Value: []int64{25000},
					Label: map[string][]string{
						"name":    {"ds4_attnKV_gemv"},
						"subtype": {"kernel"},
						"stream":  {"1"},
						"buffer":  {"attn_norm_out"},
					},
				},
			},
		},
	}

	report, err := Analyze(cap)
	if err != nil {
		t.Fatalf("Analyze failed: %v", err)
	}

	var found bool
	for _, item := range report.Items {
		if item.Category == CategoryMemoryCompute && strings.Contains(item.Finding, "Redundant activation DRAM traffic") {
			found = true
			if !strings.Contains(item.ActionableRecommendation, "paired dual-GEMV") {
				t.Errorf("recommendation = %q, want containing paired dual-GEMV", item.ActionableRecommendation)
			}
			break
		}
	}
	if !found {
		t.Fatalf("expected Dual-GEMV paired projection advisory not found; items: %+v", report.Items)
	}
}

func TestAdvice_CumulativeDriverSubmissionOverhead(t *testing.T) {
	// Verifies that a stream with multiple micro-kernels reports cumulative statistics and top contributors.
	// Kernels are spaced 20 µs apart on an idle stream, and every launch is
	// issued 5 µs after the previous kernel retired.
	names := []string{"ds4_rms", "ds4_rms", "ds4_quant_q8_1", "ds4_quant_q8_1", "ds4_router_biased", "ds4_rms", "ds4_quant_q8_1", "ds4_rms", "ds4_router_biased", "ds4_rms"}
	durations := []uint64{2000, 2000, 2500, 2500, 3000, 2000, 2500, 2000, 3000, 2000}
	samples := make([]*profile.Sample, 0, len(names))
	apiStarts := make(map[int]uint64, len(names))
	var prevEnd uint64
	for i, name := range names {
		start := uint64(1000 + i*20000)
		samples = append(samples, microKernelSample(name, 1, start, start+durations[i], i+1))
		if i > 0 {
			apiStarts[i+1] = prevEnd + 5000
		}
		prevEnd = start + durations[i]
	}

	cap := &cudacapture.Capture{
		Files: map[string][]byte{"capture-status.json": apiRecordsFile(t, apiStarts)},
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{
				{Type: "kernel_time", Unit: "nanoseconds"},
			},
			Sample: samples,
		},
	}

	report, err := Analyze(cap)
	if err != nil {
		t.Fatalf("Analyze failed: %v", err)
	}

	var found bool
	for _, item := range report.Items {
		if item.Category == CategoryArchitectureDispatch && strings.Contains(item.Finding, "Driver submission overhead dominates short kernel execution") {
			found = true
			if !strings.Contains(item.Finding, "stream totals 10 micro-kernels") {
				t.Errorf("finding = %q, want containing stream totals 10 micro-kernels", item.Finding)
			}
			if !strings.Contains(item.Finding, "top micro-kernels") {
				t.Errorf("finding = %q, want containing top micro-kernels", item.Finding)
			}
			break
		}
	}
	if !found {
		t.Fatalf("expected Cumulative driver submission advisory not found; items: %+v", report.Items)
	}
}

func TestAdvice_MicroKernelDeterministicTieBreaking(t *testing.T) {
	// Verifies that micro-kernels with identical counts sort deterministically (alphabetically by name).
	cap := &cudacapture.Capture{
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{
				{Type: "kernel_time", Unit: "nanoseconds"},
			},
			Sample: []*profile.Sample{
				{Value: []int64{2000}, Label: map[string][]string{"name": {"kernel_z"}, "subtype": {"kernel"}, "stream": {"1"}}},
				{Value: []int64{2000}, Label: map[string][]string{"name": {"kernel_a"}, "subtype": {"kernel"}, "stream": {"1"}}},
				{Value: []int64{2000}, Label: map[string][]string{"name": {"kernel_z"}, "subtype": {"kernel"}, "stream": {"1"}}},
				{Value: []int64{2000}, Label: map[string][]string{"name": {"kernel_a"}, "subtype": {"kernel"}, "stream": {"1"}}},
				{Value: []int64{2000}, Label: map[string][]string{"name": {"kernel_m"}, "subtype": {"kernel"}, "stream": {"1"}}},
				{Value: []int64{2000}, Label: map[string][]string{"name": {"kernel_m"}, "subtype": {"kernel"}, "stream": {"1"}}},
				{Value: []int64{2000}, Label: map[string][]string{"name": {"kernel_b"}, "subtype": {"kernel"}, "stream": {"1"}}},
				{Value: []int64{2000}, Label: map[string][]string{"name": {"kernel_b"}, "subtype": {"kernel"}, "stream": {"1"}}},
				{Value: []int64{2000}, Label: map[string][]string{"name": {"kernel_c"}, "subtype": {"kernel"}, "stream": {"1"}}},
				{Value: []int64{2000}, Label: map[string][]string{"name": {"kernel_c"}, "subtype": {"kernel"}, "stream": {"1"}}},
			},
		},
	}

	for repeat := 0; repeat < 5; repeat++ {
		report, err := Analyze(cap)
		if err != nil {
			t.Fatalf("Analyze failed: %v", err)
		}
		var found bool
		for _, item := range report.Items {
			if strings.Contains(item.Finding, "top micro-kernels") {
				found = true
				want := "[top micro-kernels: kernel_a (2x), kernel_b (2x), kernel_c (2x)]"
				if !strings.Contains(item.Finding, want) {
					t.Fatalf("run %d: finding = %q, want containing %q", repeat, item.Finding, want)
				}
				break
			}
		}
		if !found {
			t.Fatalf("run %d: expected finding with top micro-kernels not found", repeat)
		}
	}
}

func TestAdvice_ShortKernelSequenceWithoutTimestamps(t *testing.T) {
	// A capture with no activity timestamps cannot support any statement about
	// device idle time or host submission.
	cap := &cudacapture.Capture{
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{{Type: "kernel_time", Unit: "nanoseconds"}},
			Sample: []*profile.Sample{
				{Value: []int64{3000}, Label: map[string][]string{"name": {"k1"}, "subtype": {"kernel"}, "stream": {"1"}}},
				{Value: []int64{4000}, Label: map[string][]string{"name": {"k2"}, "subtype": {"kernel"}, "stream": {"1"}}},
				{Value: []int64{5000}, Label: map[string][]string{"name": {"k3"}, "subtype": {"kernel"}, "stream": {"1"}}},
				{Value: []int64{6000}, Label: map[string][]string{"name": {"k4"}, "subtype": {"kernel"}, "stream": {"1"}}},
				{Value: []int64{4000}, Label: map[string][]string{"name": {"k5"}, "subtype": {"kernel"}, "stream": {"1"}}},
			},
		},
	}

	report := &Report{}
	auditStreamKernelSequences(cap, report)

	var found bool
	for _, item := range report.Items {
		if strings.Contains(item.Finding, "Driver submission overhead") || strings.Contains(item.Finding, "Low GPU duty cycle") {
			t.Errorf("asserted an unmeasured cause: %+v", item)
		}
		if item.Severity == SeverityContext && strings.Contains(item.Finding, "Short-kernel dispatch sequence with unmeasured GPU duty cycle") {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected the unmeasured-duty-cycle Info finding; items: %+v", report.Items)
	}
}

func TestAdvice_RepeatedHostLoopDispatch(t *testing.T) {
	var samples []*profile.Sample
	for i := 0; i < 60; i++ {
		start := uint64(1000 + i*5000)
		samples = append(samples, microKernelSample("repeated_gemv", 1, start, start+2000, 100+i))
	}
	cap := &cudacapture.Capture{
		Profile: &profile.Profile{Sample: samples},
	}

	report, err := Analyze(cap)
	if err != nil {
		t.Fatalf("Analyze failed: %v", err)
	}

	var found bool
	for _, item := range report.Items {
		if strings.Contains(item.Finding, "Repeated fine-grained host loop dispatch") && strings.Contains(item.Finding, "60 consecutive times") {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected repeated host loop dispatch finding, got: %+v", report.Items)
	}
}

func TestAdvice_PCSamplingExcludesGhostSequences(t *testing.T) {
	// PC-sampled instructions without stream/timeline labels must not be treated as microkernel launch sequences.
	var samples []*profile.Sample
	for i := 0; i < 20; i++ {
		samples = append(samples, &profile.Sample{
			Location: []*profile.Location{
				{ID: uint64(i + 1), Address: uint64(0x1000 + i*16), Line: []profile.Line{
					{Function: &profile.Function{ID: 1, Name: "my_kernel"}, Line: int64(i + 1)},
				}},
			},
			Value: []int64{10},
		})
	}
	cap := &cudacapture.Capture{
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{{Type: "pc_samples", Unit: "count"}},
			Sample:     samples,
		},
	}

	report, err := Analyze(cap)
	if err != nil {
		t.Fatalf("Analyze failed: %v", err)
	}

	for _, item := range report.Items {
		if strings.Contains(item.Finding, "Short-kernel dispatch sequence") || strings.Contains(item.Finding, "Repeated fine-grained host loop dispatch") {
			t.Errorf("unexpected dispatch sequence finding on pure PC samples: %+v", item)
		}
	}
}

func TestAdvice_PointwiseKernelPairFusion(t *testing.T) {
	var samples []*profile.Sample
	// Simulate 25 back-to-back pairs of norm -> quant
	for i := 0; i < 25; i++ {
		samples = append(samples, &profile.Sample{
			Label: map[string][]string{
				"stream":  {"1"},
				"subtype": {"kernel"},
			},
			NumLabel: map[string][]int64{
				"correlation": {int64(i*2 + 1)},
			},
			Location: []*profile.Location{
				{ID: 1, Line: []profile.Line{{Function: &profile.Function{ID: 1, Name: "my_norm"}}}},
			},
		})
		samples = append(samples, &profile.Sample{
			Label: map[string][]string{
				"stream":  {"1"},
				"subtype": {"kernel"},
			},
			NumLabel: map[string][]int64{
				"correlation": {int64(i*2 + 2)},
			},
			Location: []*profile.Location{
				{ID: 2, Line: []profile.Line{{Function: &profile.Function{ID: 2, Name: "my_quant"}}}},
			},
		})
	}

	cap := &cudacapture.Capture{
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{{Type: "pc_samples", Unit: "count"}},
			Sample:     samples,
		},
	}

	report, err := Analyze(cap)
	if err != nil {
		t.Fatalf("Analyze failed: %v", err)
	}

	var foundFusion bool
	for _, it := range report.Items {
		if strings.Contains(it.Finding, "occurred 25 times") {
			foundFusion = true
			if !strings.Contains(it.ActionableRecommendation, "eliminate 25 DRAM round-trips") {
				t.Errorf("expected recommendation to cite 25 round trips, got: %s", it.ActionableRecommendation)
			}
		}
	}
	if !foundFusion {
		t.Errorf("expected fusion finding with recurrence count 25, items: %+v", report.Items)
	}
}

// TestFusionPairFindingCarriesItsDeviceTime pins that a pair finding publishes
// the device time it already names in prose.
//
// The pair findings set Location.Kernel to "A -> B", which is no kernel in the
// profile, so the per-kernel fill-in in finalizeAndRankFindingsWithTiming
// looks it up and finds nothing. Both numeric fields then stayed zero and,
// being omitempty, left the JSON entirely: a consumer iterating items and
// reading device_time_nanos got a missing key, and the item could not be
// ranked or sorted against the per-kernel findings it competes with. The
// accumulated duration was in the finding text the whole time.
func TestFusionPairFindingCarriesItsDeviceTime(t *testing.T) {
	var samples []*profile.Sample
	for i := 0; i < 4; i++ {
		for _, name := range []string{"swiglu_fwd", "quant_block_q8"} {
			samples = append(samples, &profile.Sample{
				Location: []*profile.Location{{ID: uint64(len(samples) + 1), Line: []profile.Line{
					{Function: &profile.Function{ID: 1, Name: name}},
				}}},
				Value:    []int64{100000},
				Label:    map[string][]string{"name": {name}, "buffer": {"hidden"}},
				NumLabel: map[string][]int64{"stream": {7}},
			})
		}
	}
	cap := &cudacapture.Capture{Profile: &profile.Profile{
		SampleType: []*profile.ValueType{{Type: "kernel_time", Unit: "nanoseconds"}},
		Sample:     samples,
	}}

	report, err := Analyze(cap)
	if err != nil {
		t.Fatalf("Analyze: %v", err)
	}
	var pair *Item
	for i := range report.Items {
		if strings.Contains(report.Items[i].Location.Kernel, "->") {
			pair = &report.Items[i]
			break
		}
	}
	if pair == nil {
		t.Fatalf("no pair finding in %d items; the fixture no longer reaches the fusion path", len(report.Items))
	}
	// The invariant is that the field says what the prose says. Hardcoding a
	// number here would pin the fixture's adjacency arithmetic instead, and
	// that is not the property that broke.
	said, ok := accumulatedFromFinding(pair.Finding)
	if !ok {
		t.Fatalf("finding names no accumulation, so there is nothing to agree with: %s", pair.Finding)
	}
	if pair.DeviceTimeNanos != int64(said) {
		t.Errorf("device_time_nanos = %d, finding says %v: the field must carry what the prose already names", pair.DeviceTimeNanos, said)
	}
	if pair.ShareOfProfile <= 0 {
		t.Errorf("share_of_profile = %v, want the share derived from the device time", pair.ShareOfProfile)
	}

	// Asserted on the bytes: omitempty means a zero field is an ABSENT key,
	// which is what a consumer hit as a missing-key error, and a struct decode
	// would silently read it back as zero.
	b, err := json.Marshal(pair)
	if err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"device_time_nanos", "share_of_profile"} {
		if !bytes.Contains(b, []byte(`"`+key+`"`)) {
			t.Errorf("%s absent from the published item: %s", key, b)
		}
	}
}

// accumulatedFromFinding returns the duration a pair finding names in its
// "accumulating <d>" clause.
func accumulatedFromFinding(finding string) (time.Duration, bool) {
	const marker = "accumulating "
	i := strings.Index(finding, marker)
	if i < 0 {
		return 0, false
	}
	rest := finding[i+len(marker):]
	if j := strings.IndexAny(rest, ")"); j >= 0 {
		rest = rest[:j]
	}
	d, err := time.ParseDuration(rest)
	if err != nil {
		return 0, false
	}
	return d, true
}

// TestNoRecaptureRecommendedWhenTheJournalIsAttached pins the recommendation
// itself, which is the expensive half: a re-capture costs a run under GPU lock.
func TestNoRecaptureRecommendedWhenTheJournalIsAttached(t *testing.T) {
	withJournal := &cudacapture.Capture{Files: map[string][]byte{"capture.journal": []byte("x")}}
	if got := recaptureOrRead(withJournal); strings.Contains(strings.ToLower(got), "re-capture") {
		t.Errorf("recommends a re-capture for a capture holding its journal: %q", got)
	} else if !strings.Contains(got, "cudaprof gaps") {
		t.Errorf("does not point at the reader that decodes the journal: %q", got)
	}
	if got := journalRemedy(withJournal); !strings.Contains(got, "per-record journal") {
		t.Errorf("finding does not say the journal is present: %q", got)
	}

	// A capture that genuinely has no journal must still be told to re-capture:
	// the point is to stop the false claim, not to stop every claim.
	none := &cudacapture.Capture{}
	if got := recaptureOrRead(none); !strings.Contains(strings.ToLower(got), "re-capture") {
		t.Errorf("no journal, and no re-capture recommended: %q", got)
	}
	if got := journalRemedy(none); got != "" {
		t.Errorf("journalRemedy = %q for a capture with no journal", got)
	}
}

// TestUncorrelatedLaunchesAreNotAbsentOnes pins the distinction the wording
// erased. A capture holding host launch records whose correlation ids do not
// reach a sequence is not missing data, and telling someone to capture what
// they already have sends them back to the GPU for nothing.
func TestUncorrelatedLaunchesAreNotAbsentOnes(t *testing.T) {
	why, rec := unmeasuredLaunchReason(124834)
	if !strings.Contains(why, "124834") || !strings.Contains(why, "none correlates") {
		t.Errorf("why = %q, want it to say the records are present and uncorrelated", why)
	}
	if strings.Contains(strings.ToLower(rec), "capture host launch") {
		t.Errorf("recommends capturing records the capture already holds: %q", rec)
	}

	why, rec = unmeasuredLaunchReason(0)
	if !strings.Contains(why, "absent from this capture") {
		t.Errorf("why = %q, want the absence stated plainly", why)
	}
	if !strings.Contains(strings.ToLower(rec), "capture host launch") {
		t.Errorf("no records and no recommendation to capture them: %q", rec)
	}
}
