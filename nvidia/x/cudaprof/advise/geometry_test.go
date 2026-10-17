package advise

import (
	"fmt"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

func TestAdvice_UnderpopulatedThreadBlock(t *testing.T) {
	tests := []struct {
		name         string
		cap          *cudacapture.Capture
		wantWarning  bool
		wantKernel   string
		wantCategory Category
		wantSeverity Severity
		wantFinding  string
		wantRec      string
	}{
		{
			name: "single_warp_32_threads_profile",
			cap: &cudacapture.Capture{
				Metadata: []byte(`{"arch": "sm_121a"}`),
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{{Type: "samples", Unit: "count"}},
					Sample: []*profile.Sample{
						{
							Value: []int64{100},
							Label: map[string][]string{
								"kernel": {"decode_gemv"},
								"block":  {"32,1,1"},
								"grid":   {"128,1,1"},
							},
						},
					},
				},
			},
			wantWarning:  true,
			wantKernel:   "decode_gemv",
			wantCategory: CategoryArchitectureCompute,
			wantSeverity: SeverityCost,
			wantFinding:  "Single-warp thread blocks",
			wantRec:      "Restructure thread block geometry to >=128 threads (4 warps)",
		},
		{
			name: "underpopulated_16_threads_json",
			cap: &cudacapture.Capture{
				Metadata: []byte(`{"arch": "sm_121a"}`),
				Files: map[string][]byte{
					"activities.json": []byte(`[
						{"name": "tiny_kernel", "subtype": "kernel", "threads": 16, "grid": 64}
					]`),
				},
			},
			wantWarning:  true,
			wantKernel:   "tiny_kernel",
			wantCategory: CategoryArchitectureCompute,
			wantSeverity: SeverityCost,
			wantFinding:  "Single-warp thread blocks",
			wantRec:      "Restructure thread block geometry to >=128 threads (4 warps)",
		},
		{
			name: "ptx_reqntid_single_warp",
			cap: &cudacapture.Capture{
				Code: cudacapture.CodeFiles{PTX: "test.ptx"},
				Files: map[string][]byte{
					"test.ptx": []byte(`.version 9.0
.target sm_121a
.address_size 64
.visible .entry ptx_warp_kernel()
.reqntid 32, 1, 1
{
	ret;
}`),
				},
			},
			wantWarning:  true,
			wantKernel:   "ptx_warp_kernel",
			wantCategory: CategoryArchitectureCompute,
			wantSeverity: SeverityCost,
			wantFinding:  "Single-warp thread blocks",
			wantRec:      "Restructure thread block geometry to >=128 threads (4 warps)",
		},
		{
			name: "single_warp_unspecified_arch_suppressed",
			cap: &cudacapture.Capture{
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{{Type: "samples", Unit: "count"}},
					Sample: []*profile.Sample{
						{
							Value: []int64{100},
							Label: map[string][]string{
								"kernel": {"decode_gemv"},
								"block":  {"32,1,1"},
								"grid":   {"128,1,1"},
							},
						},
					},
				},
			},
			wantWarning: false,
		},
		{
			name: "single_warp_grid_1_suppressed",
			cap: &cudacapture.Capture{
				Metadata: []byte(`{"arch": "sm_121a"}`),
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{{Type: "samples", Unit: "count"}},
					Sample: []*profile.Sample{
						{
							Value: []int64{100},
							Label: map[string][]string{
								"kernel": {"setup_kernel"},
								"block":  {"32,1,1"},
								"grid":   {"1,1,1"},
							},
						},
					},
				},
			},
			wantWarning: false,
		},
		{
			name: "rtx_4090_sm_89_limits",
			cap: &cudacapture.Capture{
				Metadata: []byte(`{"arch": "sm_89", "sm_count": 128, "device": "NVIDIA GeForce RTX 4090"}`),
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{{Type: "samples", Unit: "count"}},
					Sample: []*profile.Sample{
						{
							Value: []int64{100},
							Label: map[string][]string{
								"kernel": {"ada_gemv"},
								"block":  {"32,1,1"},
								"grid":   {"256,1,1"},
							},
						},
					},
				},
			},
			wantWarning:  true,
			wantKernel:   "ada_gemv",
			wantCategory: CategoryArchitectureCompute,
			wantSeverity: SeverityCost,
			wantFinding:  "reaches 50% theoretical occupancy with 24/48 resident warps/SM across 24 blocks/SM",
			wantRec:      "Restructure thread block geometry to >=128 threads (4 warps)",
		},
		{
			name: "well_populated_128_threads",
			cap: &cudacapture.Capture{
				Metadata: []byte(`{"arch": "sm_121a"}`),
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{{Type: "samples", Unit: "count"}},
					Sample: []*profile.Sample{
						{
							Value: []int64{100},
							Label: map[string][]string{
								"kernel": {"good_kernel"},
								"block":  {"128,1,1"},
								"grid":   {"256,1,1"},
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
				if item.Category == tt.wantCategory && item.Severity == tt.wantSeverity &&
					strings.Contains(item.Finding, tt.wantFinding) {
					found = true
					if tt.wantKernel != "" && !strings.Contains(item.Finding, tt.wantKernel) && item.Location.Kernel != tt.wantKernel {
						t.Errorf("expected finding to mention kernel %q, got: %s", tt.wantKernel, item.Finding)
					}
					if !strings.Contains(item.ActionableRecommendation, tt.wantRec) {
						t.Errorf("recommendation = %q, want containing %q", item.ActionableRecommendation, tt.wantRec)
					}
					break
				}
			}

			if tt.wantWarning && !found {
				t.Errorf("did not find expected underpopulated thread block warning; items: %+v", report.Items)
			} else if !tt.wantWarning && found {
				t.Errorf("unexpected underpopulated thread block warning found in %+v", report.Items)
			}
		})
	}
}

func TestAdvice_SMGridStarvation(t *testing.T) {
	tests := []struct {
		name         string
		cap          *cudacapture.Capture
		wantWarning  bool
		wantKernel   string
		wantCategory Category
		wantSeverity Severity
		wantFinding  string
		wantRec      string
	}{
		{
			name: "grid_64_on_136_sm_blackwell",
			cap: &cudacapture.Capture{
				Metadata: []byte(`{"arch": "sm_121a", "sm_count": 136}`),
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{{Type: "samples", Unit: "count"}},
					Sample: []*profile.Sample{
						{
							Value: []int64{100},
							Label: map[string][]string{
								"kernel": {"decode_attention"},
								"grid":   {"64,1,1"},
								"block":  {"128,1,1"},
							},
						},
					},
				},
			},
			wantWarning:  true,
			wantKernel:   "decode_attention",
			wantCategory: CategoryArchitecture,
			wantSeverity: SeverityContext,
			wantFinding:  "launched with fewer grid blocks than the GPU has SMs",
			wantRec:      "Compare achieved occupancy against these grids before changing any of them",
		},
		{
			name: "grid_starvation_json_activities",
			cap: &cudacapture.Capture{
				Metadata: []byte(`{"device": "NVIDIA B200"}`), // 132 SMs
				Files: map[string][]byte{
					"activities.json": []byte(`[
						{"name": "starved_kernel", "subtype": "kernel", "grid_blocks": 32, "block_threads": 256}
					]`),
				},
			},
			wantWarning:  true,
			wantKernel:   "starved_kernel",
			wantCategory: CategoryArchitecture,
			wantSeverity: SeverityContext,
			wantFinding:  "launched with fewer grid blocks than the GPU has SMs",
			wantRec:      "Compare achieved occupancy against these grids before changing any of them",
		},
		{
			name: "sufficient_grid_256_blocks",
			cap: &cudacapture.Capture{
				Metadata: []byte(`{"sm_count": 136}`),
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{{Type: "samples", Unit: "count"}},
					Sample: []*profile.Sample{
						{
							Value: []int64{100},
							Label: map[string][]string{
								"kernel": {"fully_saturated_kernel"},
								"grid":   {"256,1,1"},
								"block":  {"128,1,1"},
							},
						},
					},
				},
			},
			wantWarning: false,
		},
		{
			name: "grid_starvation_unspecified_sm_count_suppressed",
			cap: &cudacapture.Capture{
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{{Type: "samples", Unit: "count"}},
					Sample: []*profile.Sample{
						{
							Value: []int64{100},
							Label: map[string][]string{
								"kernel": {"some_kernel"},
								"grid":   {"64,1,1"},
								"block":  {"128,1,1"},
							},
						},
					},
				},
			},
			wantWarning: false,
		},
		{
			name: "gb10_60_blocks_fills_48_sms_no_starvation",
			cap: &cudacapture.Capture{
				Metadata: []byte(`{"arch": "sm_121a"}`), // GB10 has 48 SMs; 60 blocks >= 48 SMs
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{{Type: "samples", Unit: "count"}},
					Sample: []*profile.Sample{
						{
							Value: []int64{100},
							Label: map[string][]string{
								"kernel": {"gb10_batch_kernel"},
								"grid":   {"60,1,1"},
								"block":  {"128,1,1"},
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
				if item.Category == tt.wantCategory && item.Severity == tt.wantSeverity &&
					strings.Contains(item.Finding, tt.wantFinding) {
					found = true
					if tt.wantKernel != "" && !strings.Contains(item.Finding, tt.wantKernel) && item.Location.Kernel != tt.wantKernel {
						t.Errorf("expected finding to mention kernel %q, got: %s", tt.wantKernel, item.Finding)
					}
					if !strings.Contains(item.ActionableRecommendation, tt.wantRec) {
						t.Errorf("recommendation = %q, want containing %q", item.ActionableRecommendation, tt.wantRec)
					}
					break
				}
			}

			if tt.wantWarning && !found {
				t.Errorf("did not find expected grid starvation warning; items: %+v", report.Items)
			} else if !tt.wantWarning && found {
				t.Errorf("unexpected grid starvation warning found in %+v", report.Items)
			}
		})
	}
}

func TestAdvice_MultiWarpReductionStarvation(t *testing.T) {
	// Verifies detection of multi-warp thread blocks executing sequential chunk reductions with idle warps.
	cap := &cudacapture.Capture{
		Metadata: []byte(`{"arch": "sm_121a"}`),
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{
				{Type: "samples", Unit: "count"},
			},
			Sample: []*profile.Sample{
				{
					Value: []int64{100},
					Label: map[string][]string{
						"kernel": {"ds4_f16_pair_ordered_chunks"},
						"block":  {"128,1,1"},
						"grid":   {"256,1,1"},
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
		if item.Category == CategoryArchitectureCompute && strings.Contains(item.Finding, "Multi-warp thread block underutilization in row reduction") {
			found = true
			if !strings.Contains(item.ActionableRecommendation, "Group multiple rows per thread block") {
				t.Errorf("recommendation = %q, want containing Group multiple rows per thread block", item.ActionableRecommendation)
			}
			// This advisory names ds4_f16_pair_ordered_chunks, the one kernel where
			// the transformation it recommends was measured 2.84% slower. It must
			// not issue that as an instruction.
			if !strings.Contains(item.ActionableRecommendation, occupancyCaveat) {
				t.Errorf("recommendation = %q, want occupancyCaveat appended", item.ActionableRecommendation)
			}
			break
		}
	}
	if !found {
		t.Fatalf("expected Multi-Warp reduction starvation advisory not found; items: %+v", report.Items)
	}
}

// geometrySample builds one kernel activity sample with launch geometry labels.
func geometrySample(name, grid, block string, durationNs int64) *profile.Sample {
	return &profile.Sample{
		Value: []int64{durationNs},
		Label: map[string][]string{
			"name":    {name},
			"subtype": {"kernel"},
			"stream":  {"1"},
			"grid":    {grid},
			"block":   {block},
		},
	}
}

func TestAdvice_UnderSubscribedLaunchGeometry(t *testing.T) {
	const gb10Metadata = `{"arch":"sm_121a","sm_count":48,"device":"NVIDIA GB10"}`
	// 48 SMs x 32 threads per warp.
	const warpPerSM = 1536

	tests := []struct {
		name         string
		metadata     string
		samples      []*profile.Sample
		wantSeverity Severity
		wantFindings []string
		wantRec      string
		wantAbsent   []string
		// wantCaveat is false for every row here on purpose: grid starvation
		// leaves SMs literally idle, so its advice is not a capacity figure being
		// read as throughput. This pins those sites bare against a later sweep
		// that appends occupancyCaveat to everything mentioning occupancy.
		wantCaveat bool
	}{
		{
			name:     "single_threaded_launches",
			metadata: gb10Metadata,
			samples: []*profile.Sample{
				geometrySample("ds4_compressor_attention", "1,1,1", "1,1,1", 361300),
				geometrySample("ds4_compressor_attention", "1,1,1", "1,1,1", 361300),
				geometrySample("ds4_hc_cached_split", "1,1,1", "1,1,1", 15200),
				geometrySample("ds4_hc_cached_split", "1,1,1", "1,1,1", 15200),
				geometrySample("ds4_hc_cached_split", "1,1,1", "1,1,1", 15200),
				geometrySample("ds4_moe_gemm", "96,1,1", "256,1,1", 100000),
			},
			wantSeverity: SeverityReadFirst,
			wantFindings: []string{
				"Single-threaded kernel launches leave the device idle",
				"5 of 6 captured launches (83.3%) ran one thread (grid 1 x block 1)",
				"consuming 0.77 ms of device time",
				"ds4_compressor_attention (1 thread, 2 launches, 361.3 µs mean, 0.72 ms total)",
			},
			wantRec: "Parallelize these kernels across threads and blocks",
		},
		{
			name:     "cannot_fill_one_warp_per_sm",
			metadata: gb10Metadata,
			samples: []*profile.Sample{
				geometrySample("ds4_compressor_indexer", "4,1,1", "32,1,1", 78800),
				geometrySample("ds4_compressor_indexer", "4,1,1", "32,1,1", 78800),
				geometrySample("ds4_moe_gemm", "96,1,1", "256,1,1", 100000),
			},
			wantSeverity: SeverityCost,
			wantFindings: []string{
				"Launch geometry cannot fill one warp per SM",
				"2 of 3 captured launches (66.7%) ran fewer than 1536 total threads (48 SMs x 32 threads per warp)",
				"ds4_compressor_indexer (128 threads, 2 launches, 78.8 µs mean, 0.16 ms total)",
			},
			wantRec:    "Grow the grid so at least one warp is resident per SM (>= 1536 total threads)",
			wantAbsent: []string{"Single-threaded kernel launches leave the device idle"},
		},
		{
			name:     "fully_subscribed_launches",
			metadata: gb10Metadata,
			samples: []*profile.Sample{
				geometrySample("ds4_moe_gemm", "96,1,1", "256,1,1", 100000),
				geometrySample("ds4_attention", "48,2,1", "128,1,1", 200000),
			},
			wantAbsent: []string{
				"Single-threaded kernel launches leave the device idle",
				"Launch geometry cannot fill one warp per SM",
			},
		},
	}

	if warpPerSM != 48*32 {
		t.Fatalf("warpPerSM constant %d does not match 48 SMs x 32 threads", warpPerSM)
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cap := &cudacapture.Capture{
				Metadata: []byte(tt.metadata),
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{{Type: "kernel_time", Unit: "nanoseconds"}},
					Sample:     tt.samples,
				},
			}
			report := &Report{}
			auditUnderSubscribedLaunches(cap, report)

			for _, want := range tt.wantFindings {
				var found bool
				for _, item := range report.Items {
					if strings.Contains(item.Finding, want) {
						found = true
						if item.Severity != tt.wantSeverity {
							t.Errorf("severity = %q, want %q", item.Severity, tt.wantSeverity)
						}
						if item.Category != CategoryArchitectureCompute {
							t.Errorf("category = %q, want %q", item.Category, CategoryArchitectureCompute)
						}
						if tt.wantRec != "" && !strings.Contains(item.ActionableRecommendation, tt.wantRec) {
							t.Errorf("recommendation = %q, want containing %q", item.ActionableRecommendation, tt.wantRec)
						}
						if got := strings.Contains(item.ActionableRecommendation, occupancyCaveat); got != tt.wantCaveat {
							t.Errorf("occupancyCaveat present = %v, want %v in %q", got, tt.wantCaveat, item.ActionableRecommendation)
						}
						break
					}
				}
				if !found {
					t.Errorf("finding containing %q not found; items: %+v", want, report.Items)
				}
			}
			for _, absent := range tt.wantAbsent {
				for _, item := range report.Items {
					if strings.Contains(item.Finding, absent) {
						t.Errorf("unexpected finding containing %q: %+v", absent, item)
					}
				}
			}
		})
	}
}

func TestAdvice_UnderSubscribedWithoutSMCount(t *testing.T) {
	// Without a recorded SM count the one-warp-per-SM rule cannot run, and the
	// single-thread finding has to say so.
	cap := &cudacapture.Capture{
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{{Type: "kernel_time", Unit: "nanoseconds"}},
			Sample: []*profile.Sample{
				geometrySample("ds4_compressor_attention", "1,1,1", "1,1,1", 361300),
				geometrySample("ds4_compressor_indexer", "4,1,1", "32,1,1", 78800),
			},
		},
	}

	report := &Report{}
	auditUnderSubscribedLaunches(cap, report)

	var found bool
	for _, item := range report.Items {
		if strings.Contains(item.Finding, "Single-threaded kernel launches leave the device idle") {
			found = true
			if !strings.Contains(item.Finding, "one-warp-per-SM rule was skipped") {
				t.Errorf("finding = %q, want it to disclose the skipped one-warp-per-SM rule", item.Finding)
			}
		}
		if strings.Contains(item.Finding, "Launch geometry cannot fill one warp per SM") {
			t.Errorf("warp-per-SM finding emitted without an SM count: %+v", item)
		}
	}
	if !found {
		t.Fatalf("expected single-threaded launch finding; items: %+v", report.Items)
	}
}

// TestAdvice_SkippedGeometryAuditsAreDisclosed checks that each geometry
// audit needing a device fact the capture lacks is disclosed on its own, and
// that an architecture outside the occupancy table is not reported as absent.
//
// The gb10 and gb200 rows name architectures that are in the occupancy table
// but that smCountForArch does not map to an SM count: the occupancy audit
// runs, and only the starvation audit is skipped unless sm_count is recorded.
func TestAdvice_SkippedGeometryAuditsAreDisclosed(t *testing.T) {
	const (
		archAbsent  = "Device architecture is not recorded in this capture"
		archUnknown = `Device architecture "sm_61" is recorded in this capture but is not in the occupancy table`
		occupancy   = "single-warp theoretical SM occupancy audit was skipped"
		smAbsent    = "Device SM count is not recorded in this capture"
		starvation  = "SM grid starvation audit was skipped"
	)
	tests := []struct {
		name     string
		metadata string
		want     [][]string // substrings of each wanted skip finding
		ran      []string   // audits that must produce their finding
	}{
		{"neither recorded", ``, [][]string{{archAbsent, occupancy}, {smAbsent, starvation}}, nil},
		{"sm_count only", `{"sm_count":48}`, [][]string{{archAbsent, occupancy}}, []string{starvation}},
		{"arch unsupported", `{"arch":"sm_61"}`, [][]string{{archUnknown, occupancy}, {smAbsent, starvation}}, nil},
		{"arch unsupported with sm_count", `{"arch":"sm_61","sm_count":28}`, [][]string{{archUnknown, occupancy}}, []string{starvation}},
		{"both supported", `{"arch":"sm_121a","sm_count":48}`, nil, []string{occupancy, starvation}},
		{"gb10 in table, sm_count not derivable", `{"arch":"gb10"}`, [][]string{{smAbsent, starvation}}, []string{occupancy}},
		{"gb10 in table with sm_count", `{"arch":"gb10","sm_count":48}`, nil, []string{occupancy, starvation}},
		{"gb200 in table, sm_count not derivable", `{"arch":"gb200"}`, [][]string{{smAbsent, starvation}}, []string{occupancy}},
		{"gb200 in table with sm_count", `{"arch":"gb200","sm_count":132}`, nil, []string{occupancy, starvation}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cap := &cudacapture.Capture{
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{{Type: "kernel_time", Unit: "nanoseconds"}},
					Sample: []*profile.Sample{{
						Value: []int64{100000},
						Label: map[string][]string{
							"name":    {"my_kernel"},
							"subtype": {"kernel"},
							"grid":    {"16,1,1"},
							"block":   {"32,1,1"},
						},
					}},
				},
			}
			if tt.metadata != "" {
				cap.Metadata = []byte(tt.metadata)
			}
			report := &Report{}
			auditLaunchGeometry(cap, report)

			var got []Item
			ran := map[string]bool{}
			for _, item := range report.Items {
				switch {
				case strings.Contains(item.Finding, "was skipped"):
					got = append(got, item)
				case strings.Contains(item.Finding, "theoretical occupancy"):
					ran[occupancy] = true
				case strings.Contains(item.Finding, "SMs)"):
					ran[starvation] = true
				}
			}
			for _, subs := range tt.want {
				if ran[subs[1]] {
					t.Errorf("audit disclosed as skipped also ran: %q", subs[1])
				}
			}
			for _, audit := range tt.ran {
				if !ran[audit] {
					t.Errorf("audit %q did not produce its finding", audit)
				}
			}
			if len(got) != len(tt.want) {
				t.Fatalf("got %d skip disclosures, want %d: %+v", len(got), len(tt.want), got)
			}
			for i, subs := range tt.want {
				item := got[i]
				if item.Severity != SeverityContext {
					t.Errorf("disclosure %d severity = %v, want %v", i, item.Severity, SeverityContext)
				}
				for _, sub := range append(subs, "does not query the local host's device") {
					if !strings.Contains(item.Finding, sub) {
						t.Errorf("disclosure %d = %q, want it to contain %q", i, item.Finding, sub)
					}
				}
			}
		})
	}
}

// TestGridStarvationIsOneFindingWithoutAPrescription pins both halves of the
// collapse: many starved kernels produce one finding, and that finding stops
// at the observation.
//
// Per-kernel it was 23 of 66 findings on a ds4 capture, all the same sentence,
// each recommending 2D tiling -- a remedy for a tuning decision that was never
// made. The grids were literals mirroring a .cu define and they were correct.
// A capture cannot tell a tuned grid from a fixed one, so the tool must not
// name a cause it cannot see.
func TestGridStarvationIsOneFindingWithoutAPrescription(t *testing.T) {
	var acts []string
	for i := 0; i < 12; i++ {
		acts = append(acts, fmt.Sprintf(`{"name":"k%02d","subtype":"kernel","grid_blocks":%d,"block_threads":256}`, i, 8+i))
	}
	cap := &cudacapture.Capture{
		Metadata: []byte(`{"device": "NVIDIA B200"}`), // 132 SMs
		Files:    map[string][]byte{"activities.json": []byte("[" + strings.Join(acts, ",") + "]")},
	}
	report, err := Analyze(cap)
	if err != nil {
		t.Fatalf("Analyze: %v", err)
	}
	var found []Item
	for _, it := range report.Items {
		if strings.Contains(it.Finding, "fewer grid blocks than the GPU has SMs") {
			found = append(found, it)
		}
	}
	if len(found) != 1 {
		t.Fatalf("12 starved kernels produced %d findings, want 1", len(found))
	}
	got := found[0]
	// Worst first, and the tail named as a count rather than dropped.
	if !strings.Contains(got.Finding, "12 kernel(s)") {
		t.Errorf("finding does not state how many kernels it covers: %s", got.Finding)
	}
	if !strings.Contains(got.Finding, "k00 (8 blocks") {
		t.Errorf("finding does not lead with the worst offender: %s", got.Finding)
	}
	if !strings.Contains(got.Finding, "and 4 more") {
		t.Errorf("finding does not account for the kernels it did not name: %s", got.Finding)
	}
	// The prescription is the part that was wrong on the merits.
	for _, gone := range []string{"2D tiling", "split-K", "griddepcontrol", "Decompose"} {
		if strings.Contains(got.ActionableRecommendation, gone) {
			t.Errorf("recommendation still prescribes %q for a grid whose origin the capture cannot see: %s", gone, got.ActionableRecommendation)
		}
	}
}
