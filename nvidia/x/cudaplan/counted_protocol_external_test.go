package cudaplan_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/cudaplan"
)

func TestCountedWorkerSharesScalarAbortWireValues(t *testing.T) {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{513}, cudair.Float32)
	y := g.AddUnary("neg", x)
	if err := g.SetOutputs(y); err != nil {
		t.Fatal(err)
	}
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	p, err := cudaplan.Compile(snap, cudaplan.WithCountedMega(2))
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	source := string(p.Steps()[0].Artifact.Raw)
	for _, code := range []int{cudamega.AbortBadLaunch, cudamega.AbortSpinExhausted, cudamega.AbortBadCompletion} {
		if !strings.Contains(source, fmt.Sprintf("atom.global.cas.b32 old, [state_ptr], 0, %d", code)) {
			t.Fatalf("counted worker changed abort wire value %d", code)
		}
	}
	if !strings.Contains(source, fmt.Sprintf("setp.ge.u32 pred, spins, %d", cudamega.DefaultSpinLimit)) {
		t.Fatal("counted worker changed poll bound")
	}
}
