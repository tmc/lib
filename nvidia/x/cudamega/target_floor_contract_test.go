package cudamega_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

func TestEmitterTargetFloorContract(t *testing.T) {
	grid := cudamega.Grid{CTAs: 1, ThreadsPerCTA: 32}
	operations := map[string]cudamega.Operation{"x": {Kind: cudamega.Add, Scalar: 1}}
	for _, tc := range []struct {
		name  string
		scope cudataskgraph.Scope
	}{
		{name: "device", scope: cudataskgraph.ScopeDevice},
		{name: "system", scope: cudataskgraph.ScopeSystem},
	} {
		graph, err := cudataskgraph.New([]cudataskgraph.Task{{ID: "x", Rank: 0, Scope: tc.scope}})
		if err != nil {
			t.Fatal(err)
		}
		t.Run(tc.name, func(t *testing.T) {
			for _, target := range []cudamega.Target{cudamega.SM80, cudamega.SM90, "sm_100", cudamega.SM121a, "sm_130"} {
				if _, err := cudamega.NewProgram(graph, operations, grid, target); err != nil {
					t.Errorf("target %s: %v", target, err)
				}
			}
			for _, tc := range []struct {
				target cudamega.Target
				want   string
			}{
				{target: "sm_70", want: "predates emitter floor sm_80"},
				{target: "turing", want: "not spelled sm_"},
				{target: "sm_90A", want: "unrecognized suffix"},
			} {
				_, err := cudamega.NewProgram(graph, operations, grid, tc.target)
				if err == nil || !strings.Contains(err.Error(), tc.want) {
					t.Errorf("target %s: error %v, want %q", tc.target, err, tc.want)
				}
			}
		})
	}
}
