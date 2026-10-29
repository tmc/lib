package egraph

import (
	"fmt"
	"strings"
	"testing"
)

func TestSaturationBudgets(t *testing.T) {
	tests := []struct {
		name     string
		opts     SaturationOptions
		reason   SaturationStopReason
		stopRule string
		applied  map[string]int // committed matches by rule, rematches included
		capped   string
	}{
		{
			name:    "unlimited",
			reason:  SaturationFixedPoint,
			applied: map[string]int{"fuse": 10, "grow": 10},
		},
		{
			name:     "fuse cap",
			opts:     SaturationOptions{RuleMatches: map[string]int{"fuse": 2}},
			reason:   SaturationRuleMatchLimit,
			stopRule: "fuse",
			applied:  map[string]int{"fuse": 2, "grow": 4},
			capped:   "fuse",
		},
		{
			name:     "grow cap",
			opts:     SaturationOptions{RuleMatches: map[string]int{"grow": 3}},
			reason:   SaturationRuleMatchLimit,
			stopRule: "grow",
			applied:  map[string]int{"fuse": 10, "grow": 3},
			capped:   "grow",
		},
		{
			name:    "unknown rule cap",
			opts:    SaturationOptions{RuleMatches: map[string]int{"absent": 1}},
			reason:  SaturationFixedPoint,
			applied: map[string]int{"fuse": 10, "grow": 10},
		},
		{
			name:    "iterations",
			opts:    SaturationOptions{MaxIterations: 1},
			reason:  SaturationIterationLimit,
			applied: map[string]int{"fuse": 5, "grow": 5},
		},
		{
			name:     "nodes",
			opts:     SaturationOptions{MaxNodes: 12},
			reason:   SaturationNodeLimit,
			stopRule: "fuse",
		},
		{
			name:     "match work",
			opts:     SaturationOptions{MaxMatchWork: 12},
			reason:   SaturationMatchWorkLimit,
			stopRule: "fuse",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := New()
			for i := range 5 {
				g.Add(NewNode("f", g.Add(NewLeaf(fmt.Sprintf("a%d", i)))))
			}
			rules := []*Rewrite{
				MustRewrite("fuse", "(f ?x)", "(g ?x)"),
				MustRewrite("grow", "(g ?x)", "(h ?x)"),
			}
			res, err := SaturateWithReport(g, rules, tt.opts)
			if err != nil {
				t.Fatal(err)
			}
			if res.StopReason != tt.reason || res.StopRule != tt.stopRule {
				t.Fatalf("stop = %v at %q, want %v at %q", res.StopReason, res.StopRule, tt.reason, tt.stopRule)
			}
			if tt.stopRule != "" && tt.reason != SaturationRuleMatchLimit && !strings.Contains(res.StopRun, "run(main)") {
				t.Errorf("StopRun = %q, want the main run", res.StopRun)
			}
			applied := make(map[string]int)
			capped := ""
			for _, st := range res.Rules {
				applied[st.Name] += st.Applied
				if st.Capped {
					capped = st.Name
				}
				if st.Calls > 0 && st.Duration <= 0 {
					t.Errorf("rule %s: %d calls in %v", st.Name, st.Calls, st.Duration)
				}
			}
			for rule, n := range tt.applied {
				if applied[rule] != n {
					t.Errorf("rule %s applied %d matches, want %d", rule, applied[rule], n)
				}
			}
			if capped != tt.capped {
				t.Errorf("capped rule %q, want %q", capped, tt.capped)
			}
		})
	}
}

func TestSaturateCheckedNamesRule(t *testing.T) {
	g := New()
	g.Add(NewNode("f", g.Add(NewLeaf("a"))))
	g.Add(NewNode("f", g.Add(NewLeaf("b"))))
	rules := []*Rewrite{MustRewrite("fuse", "(f ?x)", "(g ?x)")}
	_, err := SaturateChecked(g, rules, SaturationOptions{RuleMatches: map[string]int{"fuse": 1}})
	want := `rule match limit at rule "fuse"`
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Fatalf("SaturateChecked error = %v, want %q", err, want)
	}
}
