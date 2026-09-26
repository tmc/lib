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
