package cudacheck_test

import (
	"context"
	"encoding/binary"
	"math"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacheck"
	"github.com/tmc/lib/nvidia/x/cudair"
)

func TestTraceNodeIdentity(t *testing.T) {
	for _, names := range [][3]string{{"first", "second", "out"}, {"same", "same", "out"}, {"", "", ""}, {"not an identifier", "not an identifier", "out"}} {
		t.Run(names[0]+"/"+names[1], func(t *testing.T) {
			g := cudair.NewGraph()
			x := g.AddInput("x", []int{1}, cudair.Float32)
			a := g.AddUnary("neg", x)
			b := g.AddBinary("mul", a, a)
			c := g.AddUnary("neg", b)
			a.Name, b.Name, c.Name = names[0], names[1], names[2]
			if err := g.SetOutputs(a, b, b, c); err != nil {
				t.Fatal(err)
			}
			before := g.Outputs()
			acc, err := cudacheck.EvaluateCPUGraph(g, map[string][]float32{"x": {-2}}, cudacheck.PolicyRawBits, true)
			if err != nil {
				t.Fatalf("EvaluateCPUGraph: %v", err)
			}
			var want uint64 = 14695981039346656037
			for _, v := range []float32{2, 4, -4} {
				var bits [4]byte
				binary.LittleEndian.PutUint32(bits[:], math.Float32bits(v))
				for _, b := range bits {
					want = (want ^ uint64(b)) * 1099511628211
				}
			}
			if acc.Hash() != want {
				t.Fatalf("hash = %x, want independent folds 2,4,-4 = %x", acc.Hash(), want)
			}
			events := acc.Events()
			if len(events) != 3 {
				t.Fatalf("events = %d, want 3", len(events))
			}
			for i, v := range []float64{2, 4, -4} {
				if events[i].Index != i || events[i].Value != v || events[i].NodeName != names[i] {
					t.Fatalf("event[%d] = %+v", i, events[i])
				}
			}
			if !reflect.DeepEqual(before, g.Outputs()) {
				t.Fatal("CPU evaluation changed explicit outputs")
			}
			result, err := cudacheck.Compare(context.Background(), g, cudacheck.WithInputs(map[string][]float32{"x": {-2}}))
			if err != nil || result.Status != cudacheck.StatusUnmeasured || result.CPUHash != want {
				t.Fatalf("Compare without device = %+v, %v", result, err)
			}
		})
	}
}
