package cudamega_test

import (
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

func TestCTAProducerWiderConsumerRefused(t *testing.T) {
	golden, err := os.ReadFile("testdata/golden/cta-device-refusal.txt")
	if err != nil {
		t.Fatal(err)
	}
	for _, target := range []cudamega.Target{cudamega.SM80, cudamega.SM90, cudamega.SM121a} {
		for _, consumer := range []cudataskgraph.Scope{cudataskgraph.ScopeDevice, cudataskgraph.ScopeSystem} {
			for _, policy := range []cudamega.Policy{cudamega.WavePolicy, cudamega.CountedPolicy} {
				t.Run(fmt.Sprintf("%s/%s/%v", target, scopeLabel(consumer), policy), func(t *testing.T) {
					g, ops := scopePair(t, cudataskgraph.ScopeCTA, consumer)
					grid := cudamega.Grid{CTAs: 1, ThreadsPerCTA: 128, Policy: policy}
					want := fmt.Sprintf("task %q at CTA scope has non-CTA consumer %q; an mbarrier cannot publish at %s scope", "produce", "consume", scopeLabel(consumer))
					if consumer == cudataskgraph.ScopeDevice && want != strings.TrimSpace(string(golden)) {
						t.Fatalf("device refusal expectation differs from golden: %q", want)
					}
					p, err := cudamega.NewProgram(g, ops, grid, target)
					if p != nil || err == nil || err.Error() != want {
						t.Fatalf("CTA publication admitted: program=%v err=%v; want %s", p, err, want)
					}
					code, err := cudamega.ProgramSource(g, ops, grid, target)
					if code != "" || err == nil || err.Error() != want {
						t.Fatalf("CTA publication source admitted: err=%v; want %s", err, want)
					}
				})
			}
		}
	}
}
