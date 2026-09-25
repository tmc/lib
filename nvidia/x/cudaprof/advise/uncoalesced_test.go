package advise

import (
	"slices"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

func TestAdvice_UncoalescedMemoryAccess(t *testing.T) {
	const sampleLDGSASS = `
	.section	.text.uncoalesced_load,"ax",@progbits
uncoalesced_load:
.text.uncoalesced_load:
        /*0000*/                   MOV R1, c[0x0][0x28] ;
        /*0010*/                   LDG.E.128 R0, [R2] ;
        /*0020*/                   EXIT ;
`

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
			name: "uncoalesced_ldg_mem_throttle_gt_35pct",
			cap: &cudacapture.Capture{
				Code: cudacapture.CodeFiles{SASS: "kernels.sass"},
				Files: map[string][]byte{
					"kernels.sass": []byte(sampleLDGSASS),
				},
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{
						{Type: "stall_scoreboard", Unit: "count"},
						{Type: "stall_mem", Unit: "count"},
					},
					Function: []*profile.Function{
						{ID: 1, Name: "uncoalesced_load", Filename: "uncoalesced.cu"},
					},
					Location: []*profile.Location{
						{ID: 1, Address: 0x0010, Line: []profile.Line{{Function: &profile.Function{ID: 1, Name: "uncoalesced_load", Filename: "uncoalesced.cu"}, Line: 42}}},
					},
					Sample: []*profile.Sample{
						{
							Location: []*profile.Location{
								{ID: 1, Address: 0x0010, Line: []profile.Line{{Function: &profile.Function{ID: 1, Name: "uncoalesced_load", Filename: "uncoalesced.cu"}, Line: 42}}},
							},
							Value: []int64{55, 45}, // 45 / 100 = 45% > 35%
							Label: map[string][]string{
								"kernel": {"uncoalesced_load"},
							},
						},
					},
				},
			},
			wantWarning:  true,
			wantCategory: CategoryMemory,
			wantSeverity: SeverityCost,
			wantFinding:  "Memory stalls dominate this load: stall_mem is 45.0% of stalls",
			wantRec:      "Check whether adjacent lane IDs access contiguous bytes",
		},
		{
			name: "uncoalesced_critical_high_mem_stalls_60pct",
			cap: &cudacapture.Capture{
				Code: cudacapture.CodeFiles{SASS: "kernels.sass"},
				Files: map[string][]byte{
					"kernels.sass": []byte(sampleLDGSASS),
				},
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{
						{Type: "stall_math", Unit: "count"},
						{Type: "stall_mem", Unit: "count"},
					},
					Function: []*profile.Function{
						{ID: 1, Name: "uncoalesced_load", Filename: "uncoalesced.cu"},
					},
					Location: []*profile.Location{
						{ID: 1, Address: 0x0010, Line: []profile.Line{{Function: &profile.Function{ID: 1, Name: "uncoalesced_load", Filename: "uncoalesced.cu"}, Line: 42}}},
					},
					Sample: []*profile.Sample{
						{
							Location: []*profile.Location{
								{ID: 1, Address: 0x0010, Line: []profile.Line{{Function: &profile.Function{ID: 1, Name: "uncoalesced_load", Filename: "uncoalesced.cu"}, Line: 42}}},
							},
							Value: []int64{40, 60}, // 60 / 100 = 60% >= 50% -> Critical
							Label: map[string][]string{
								"kernel": {"uncoalesced_load"},
							},
						},
					},
				},
			},
			wantWarning:  true,
			wantCategory: CategoryMemory,
			wantSeverity: SeverityCost,
			wantFinding:  "Memory stalls dominate this load: stall_mem is 60.0% of stalls",
			wantRec:      "Check whether adjacent lane IDs access contiguous bytes",
		},
		{
			name: "coalesced_ldg_low_mem_stalls_15pct",
			cap: &cudacapture.Capture{
				Code: cudacapture.CodeFiles{SASS: "kernels.sass"},
				Files: map[string][]byte{
					"kernels.sass": []byte(sampleLDGSASS),
				},
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{
						{Type: "stall_scoreboard", Unit: "count"},
						{Type: "stall_mem", Unit: "count"},
					},
					Location: []*profile.Location{
						{ID: 1, Address: 0x0010},
					},
					Sample: []*profile.Sample{
						{
							Location: []*profile.Location{{ID: 1, Address: 0x0010}},
							Value:    []int64{85, 15}, // 15% <= 35%
							Label: map[string][]string{
								"kernel": {"uncoalesced_load"},
							},
						},
					},
				},
			},
			wantWarning: false,
		},
		{
			name: "address_arithmetic_non_contiguous_stride_label",
			cap: &cudacapture.Capture{
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{{Type: "samples", Unit: "count"}},
					Sample: []*profile.Sample{
						{
							Value: []int64{100},
							Label: map[string][]string{
								"kernel": {"strided_access_kernel"},
								"stride": {"32"},
							},
						},
					},
				},
			},
			wantWarning:  true,
			wantCategory: CategoryMemory,
			wantSeverity: SeverityCost,
			wantFinding:  "Threads within the warp generate strided or disparate 128-byte cache line requests",
			wantRec:      "Align and coalesce thread access so adjacent lane IDs access contiguous bytes within a single 128B cache line",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			report, err := Analyze(tt.cap)
			if err != nil {
				t.Fatalf("Analyze failed: %v", err)
			}

			var found bool
			for _, item := range findings(report) {
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
				t.Errorf("did not find expected uncoalesced memory warning; items: %+v", report.Items)
			} else if !tt.wantWarning && found {
				t.Errorf("unexpected uncoalesced memory warning found in %+v", report.Items)
			}
		})
	}
}

func TestAdvice_ScoreboardAndBranchStalls(t *testing.T) {
	cap := &cudacapture.Capture{
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{
				{Type: "stall_scoreboard", Unit: "count"},
				{Type: "stall_branch", Unit: "count"},
				{Type: "stall_math", Unit: "count"},
			},
			Sample: []*profile.Sample{
				{
					Location: []*profile.Location{
						{ID: 1, Address: 0x100, Line: []profile.Line{
							{Function: &profile.Function{ID: 1, Name: "k_stalls"}, Line: 10},
						}},
					},
					Value: []int64{80, 0, 20}, // Scoreboard: 80% >= 25% and 80 >= 50
					Label: map[string][]string{"opcode": {"FADD"}},
				},
				{
					Location: []*profile.Location{
						{ID: 2, Address: 0x200, Line: []profile.Line{
							{Function: &profile.Function{ID: 1, Name: "k_stalls"}, Line: 20},
						}},
					},
					Value: []int64{0, 70, 30}, // Branch: 70% >= 20% and 70 >= 50
					Label: map[string][]string{"opcode": {"BRA"}},
				},
			},
		},
	}

	report, err := Analyze(cap)
	if err != nil {
		t.Fatalf("Analyze failed: %v", err)
	}

	var foundScoreboard, foundBranch bool
	for _, it := range findings(report) {
		if strings.Contains(it.Finding, "Register scoreboard latency stall") {
			foundScoreboard = true
			if !strings.Contains(it.ActionableRecommendation, "software pipelining") {
				t.Errorf("expected software pipelining recommendation, got: %s", it.ActionableRecommendation)
			}
		}
		if strings.Contains(it.Finding, "Warp branch divergence") {
			foundBranch = true
			if !strings.Contains(it.ActionableRecommendation, "branchless predicated selects") {
				t.Errorf("expected branchless recommendation, got: %s", it.ActionableRecommendation)
			}
		}
	}
	if !foundScoreboard {
		t.Errorf("did not find expected register scoreboard finding")
	}
	if !foundBranch {
		t.Errorf("did not find expected warp branch divergence finding")
	}
}

func TestAdvice_StallAttributionAndDefUseDistance(t *testing.T) {
	cap := &cudacapture.Capture{
		Files: map[string][]byte{"kernel.sass": []byte(`
Function : ds4_q2_k_selected_q8_1_gather
        /*0000*/ LDG.E R1, [R2] ;
        /*0010*/ BRA 0x30 ;
        /*0020*/ FADD R3, R1, R4 ;
        /*0030*/ EXIT ;
`)},
		Code: cudacapture.CodeFiles{SASS: "kernel.sass"},
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{
				{Type: "stall_scoreboard", Unit: "count"},
				{Type: "stall_branch", Unit: "count"},
				{Type: "stall_math", Unit: "count"},
			},
			Sample: []*profile.Sample{
				{Location: []*profile.Location{{Address: 0x20, Line: []profile.Line{{Function: &profile.Function{Name: "ds4_q2_k_selected_q8_1_gather"}}}}}, Value: []int64{80, 0, 20}},
				{Location: []*profile.Location{{Address: 0x20, Line: []profile.Line{{Function: &profile.Function{Name: "ds4_q2_k_selected_q8_1_gather"}}}}}, Value: []int64{0, 80, 20}},
			},
		},
	}
	report, err := Analyze(cap)
	if err != nil {
		t.Fatal(err)
	}
	var scoreboard, branch bool
	for _, item := range findings(report) {
		if strings.Contains(item.Finding, "Register scoreboard latency stall") {
			scoreboard = true
			if !strings.Contains(item.Finding, "instructions later") {
				t.Errorf("scoreboard finding lacks measured def-use distance: %s", item.Finding)
			}
		}
		if strings.Contains(item.Finding, "Warp branch divergence") {
			branch = true
			if item.Location.PC != 0x10 || !strings.Contains(item.Finding, "preceding BRA") {
				t.Errorf("branch stall was not attributed to preceding branch: %+v", item)
			}
		}
	}
	if !scoreboard || !branch {
		t.Fatalf("missing scoreboard=%v branch=%v findings: %+v", scoreboard, branch, report.Items)
	}
}

func TestPrecedingControlFlowBranchIncludesZeroAndWarpSync(t *testing.T) {
	fn := &cudasass.Function{Instructions: []cudasass.Instruction{
		{Address: 0x00, Opcode: "BRA"},
		{Address: 0x10, Opcode: "WARPSYNC"},
		{Address: 0x20, Opcode: "FADD"},
	}}
	if pc, opcode := precedingControlFlowBranch(fn, 0x10); pc != 0 || opcode != "BRA" {
		t.Fatalf("branch at zero = (0x%x, %q), want (0x0, BRA)", pc, opcode)
	}
	if pc, opcode := precedingControlFlowBranch(fn, 0x20); pc != 0x10 || opcode != "WARPSYNC" {
		t.Fatalf("WARPSYNC = (0x%x, %q), want (0x10, WARPSYNC)", pc, opcode)
	}
}

// TestTiedStallsOrderByPC checks that loads with equal stall counts are
// reported in PC order, not in map iteration order.
func TestTiedStallsOrderByPC(t *testing.T) {
	var samples []*profile.Sample
	for _, pc := range []uint64{0x30, 0x20, 0x10} {
		samples = append(samples, &profile.Sample{
			Location: []*profile.Location{{Address: pc}},
			Value:    []int64{100},
			Label:    map[string][]string{"kernel": {"tied"}, "opcode": {"LDG.E"}},
		})
	}
	cap := &cudacapture.Capture{
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{{Type: "stall_mem", Unit: "count"}},
			Sample:     samples,
		},
	}
	want := []uint64{0x10, 0x20, 0x30}
	for range 20 {
		report, err := Analyze(cap)
		if err != nil {
			t.Fatal(err)
		}
		var got []uint64
		for _, item := range findings(report) {
			if strings.HasPrefix(item.Finding, "Memory stalls dominate this load") {
				got = append(got, item.Location.PC)
			}
		}
		if !slices.Equal(got, want) {
			t.Fatalf("stall PCs = %#x, want %#x", got, want)
		}
	}
}
