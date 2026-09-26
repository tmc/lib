package advise

import (
	"fmt"
	"strings"
	"testing"
)

func TestCollapseRules(t *testing.T) {
	item := func(rule string, ms int64) Item {
		return Item{Severity: SeverityCost, Category: CategoryArchitecture, rule: rule, DeviceTimeNanos: ms * 1e6, Finding: fmt.Sprintf("%s %d", rule, ms)}
	}
	tests := []struct {
		name  string
		items []Item
		want  []string
	}{
		{"none", []Item{item("", 9), item("", 8)}, []string{" 9", " 8"}},
		{"four kept", []Item{item("imad-wide", 4), item("imad-wide", 3), item("imad-wide", 2), item("imad-wide", 1)},
			[]string{"imad-wide 4", "imad-wide 3", "imad-wide 2", "imad-wide 1"}},
		{"five collapsed", []Item{item("imad-wide", 5), item("fusion", 9), item("imad-wide", 4), item("imad-wide", 3), item("imad-wide", 2), item("imad-wide", 1)},
			[]string{"imad-wide 5", "fusion 9", "imad-wide 4", "imad-wide 3", "2 more scalar address arithmetic (IMAD.WIDE) findings rank below the 3 listed above"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := collapseRules(tt.items)
			var findings []string
			for _, item := range got {
				findings = append(findings, item.Finding)
			}
			if strings.Join(findings, "|") != strings.Join(tt.want, "|") {
				t.Fatalf("findings = %q, want %q", findings, tt.want)
			}
			if tt.name != "five collapsed" {
				return
			}
			tail := got[len(got)-1]
			if tail.Severity != SeverityContext || len(tail.Patterns) != 2 || tail.magnitude != "2 findings, the largest 2.000 ms" || tail.DeviceTimeNanos != 0 {
				t.Errorf("tail = %+v", tail)
			}
		})
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
