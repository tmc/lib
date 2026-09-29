package advise

import (
	"fmt"
	"strings"
	"testing"
)

func TestGroupRules(t *testing.T) {
	item := func(rule, kernel string, ms int64) Item {
		return Item{Severity: SeverityCost, Category: CategoryArchitecture, rule: rule, DeviceTimeNanos: ms * 1e6,
			Location: Location{Kernel: kernel}, Finding: fmt.Sprintf("%s %s", rule, kernel)}
	}
	tests := []struct {
		name     string
		items    []Item
		want     []string
		wantTime int64 // device time of the first item, in ms
		wantMag  string
		wantHeld int
	}{
		{"none", []Item{item("", "a", 9), item("", "b", 8)}, []string{" a", " b"}, 9, "", 0},
		{"one kept", []Item{item("imad-wide", "a", 4), item("fusion", "b", 3)}, []string{"imad-wide a", "fusion b"}, 4, "", 0},
		{"kernels summed", []Item{item("single-warp", "a", 5), item("fusion", "x", 9), item("single-warp", "b", 4), item("single-warp", "c", 3)},
			[]string{"3 kernels, 12.00 ms total: single-warp occupancy", "fusion x"}, 12, "", 3},
		{"pairs not summed", []Item{item("round-trip", "a -> b", 5), item("round-trip", "b -> c", 4)},
			[]string{"2 findings: intermediate tensor round-trip"}, 0, "2 findings, the largest 5.000 ms", 2},
		{"same kernel not summed", []Item{item("imad-wide", "a", 5), item("imad-wide", "a", 5)},
			[]string{"2 findings: scalar address arithmetic (IMAD.WIDE)"}, 0, "2 findings, the largest 5.000 ms", 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := groupRules(tt.items)
			rankItems(got)
			var findings []string
			for _, item := range got {
				findings = append(findings, item.Finding)
			}
			if strings.Join(findings, "|") != strings.Join(tt.want, "|") {
				t.Fatalf("findings = %q, want %q", findings, tt.want)
			}
			first := got[0]
			if first.DeviceTimeNanos != tt.wantTime*1e6 || first.magnitude != tt.wantMag || len(first.Patterns) != tt.wantHeld {
				t.Errorf("first = %+v", first)
			}
		})
	}
}

// TestMarkdownGroupedOnce checks that a group prints its recommendation once
// and lists each finding by location.
func TestMarkdownGroupedOnce(t *testing.T) {
	var items []Item
	for i := range 4 {
		items = append(items, Item{Severity: SeverityCost, Category: CategoryArchitecture, rule: "single-warp",
			Location: Location{Kernel: fmt.Sprintf("triton_per_%d", i)}, DeviceTimeNanos: int64(i+1) * 1e6,
			Finding: fmt.Sprintf("Single-warp thread blocks limit occupancy in triton_per_%d", i), ActionableRecommendation: "Restructure thread block geometry."})
	}
	md := (&Report{Items: groupRules(items)}).MarkdownWithLimit(0)
	if n := strings.Count(md, "Restructure thread block geometry."); n != 1 {
		t.Errorf("recommendation printed %d times, want 1:\n%s", n, md)
	}
	if strings.Contains(md, "Single-warp thread blocks") || !strings.Contains(md, "4 kernels, 10.00 ms total: single-warp occupancy") || !strings.Contains(md, "triton_per_3 (4.00 ms)") {
		t.Errorf("group not rendered as one pattern with a location list:\n%s", md)
	}
}

func TestMarkdownPatternLimit(t *testing.T) {
	var held []Item
	for i := range maxPatternLines + 2 {
		held = append(held, Item{Category: CategoryArchitecture, Finding: fmt.Sprintf("pattern %d", i)})
	}
	r := &Report{Items: []Item{{Severity: SeverityContext, Category: CategoryArchitecture, Finding: "held", Patterns: held}}}
	md := r.MarkdownWithLimit(0)
	if strings.Contains(md, fmt.Sprintf("pattern %d", maxPatternLines)) || !strings.Contains(md, "... 2 more (use --format=json for all).") {
		t.Errorf("markdown does not stop after %d patterns:\n%s", maxPatternLines, md)
	}
}

func TestDemangleText(t *testing.T) {
	tests := []struct {
		name, in, want string
	}{
		{"quoted", `kernel "_ZN7cutlass7Kernel2I52cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4EEvNT_6ParamsE" was launched`,
			`kernel "cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4>" was launched`},
		{"location", "kernel _ZN3mlx4core2cu9binary_vvINS1_8MultiplyEffjLi4EEEvPKT0_S6_PT1_T2_\n",
			"kernel mlx::core::cu::binary_vv<mlx::core::cu::Multiply, float, float, unsigned int, 4>\n"},
		{"extern C", "kernel nvjet_sm121_sss_tf32_mma_256x128x32_2_64x64x32_tmaAB_alignCD4_bz_TNNN", "kernel nvjet_sm121_sss_tf32_mma_256x128x32_2_64x64x32_tmaAB_alignCD4_bz_TNNN"},
		{"not mangled", "kernel _Zfoo", "kernel _Zfoo"},
		{"inside a word", "a_ZN3foo3barEv", "a_ZN3foo3barEv"},
	}
	for _, tt := range tests {
		if got := demangleText(tt.in); got != tt.want {
			t.Errorf("%s: demangleText(%q) = %q, want %q", tt.name, tt.in, got, tt.want)
		}
	}
}
