package cudaautograd_test

import (
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaautograd"
	"github.com/tmc/lib/nvidia/x/cudair"
)

func savedNameNear(t *testing.T, what string, got float32, want, tolerance float64) {
	t.Helper()
	if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) || math.Abs(float64(got)-want) > tolerance*(1+math.Abs(want)) {
		t.Fatalf("%s = %g, want %g", what, got, want)
	}
}

func TestSavedActivationNameCollision(t *testing.T) {
	for _, overlap := range []bool{false, true} {
		name := "unique"
		if overlap {
			name = "input intermediate overlap"
		}
		t.Run(name, func(t *testing.T) {
			g := cudair.NewGraph()
			x := g.AddInput("x", []int{1}, cudair.Float32)
			y := g.AddUnary("exp", x)
			if overlap {
				y.Name = x.Name
			}
			loss := g.AddBinary("mul", y, x)
			if err := g.SetOutputs(loss); err != nil {
				t.Fatal(err)
			}
			snapshot, err := g.Snapshot()
			if err != nil {
				t.Fatal(err)
			}
			res, err := cudaautograd.Differentiate(snapshot, loss, []*cudair.Node{x})
			if err != nil {
				t.Fatal(err)
			}
			inputs := map[string][]float32{"x": {2}}
			values, acts, err := cudaautograd.Forward(res, g, inputs)
			if err != nil || acts == nil {
				t.Fatalf("Forward = %v, %v, %v", values, acts, err)
			}
			// L=x*exp(x), so dL/dx=exp(x)*(1+x). This oracle does
			// not read the backward graph or its saved-value bindings.
			wantLoss := 2 * math.Exp(2)
			wantGradient := 3 * math.Exp(2)
			if len(values[loss.Name]) != 1 {
				t.Fatalf("missing scalar loss: %v", values)
			}
			savedNameNear(t, "loss", values[loss.Name][0], wantLoss, 1e-5)
			gradients, err := cudaautograd.Backward(res, acts)
			if err != nil || len(gradients[x.ID]) != 1 {
				t.Fatalf("Backward = %v, %v", gradients, err)
			}
			savedNameNear(t, "analytic gradient", gradients[x.ID][0], wantGradient, 1e-5)
			if len(res.SavedActivations) != 2 || res.SavedActivations[x.ID] == nil || res.SavedActivations[y.ID] == nil || res.SavedActivations[x.ID].Name == res.SavedActivations[y.ID].Name {
				t.Fatal("distinct saved values share a backward input name")
			}
			numeric, err := cudaautograd.FiniteDifferenceGrad(g, loss, "x", inputs, 1e-3)
			if err != nil || len(numeric) != 1 {
				t.Fatalf("finite difference = %v, %v", numeric, err)
			}
			savedNameNear(t, "finite-difference gradient", numeric[0], wantGradient, 1e-3)
			if err := cudaautograd.CheckGradients(g, loss, res, inputs, 1e-3, 1e-3); err != nil {
				t.Fatalf("CheckGradients rejected the saved roles: %v", err)
			}
			if inputs["x"][0] != 2 {
				t.Fatal("gradient checks changed the input")
			}
			opt, err := cudaautograd.NewSGDOptimizer(.01, 0)
			if err != nil {
				t.Fatal(err)
			}
			params := map[string][]float32{"x": {2}}
			gotLoss, updated, err := cudaautograd.TrainingStep(g, loss, res, opt, params, nil)
			if err != nil || len(updated["x"]) != 1 {
				t.Fatalf("TrainingStep = %g, %v, %v", gotLoss, updated, err)
			}
			savedNameNear(t, "training loss", gotLoss, wantLoss, 1e-5)
			savedNameNear(t, "updated parameter", updated["x"][0], 2-.01*wantGradient, 1e-5)
			if len(g.Nodes()) != 3 || x.Name != "x" || y.Name != snapshot.NodeByID(y.ID).Name || loss.Name != snapshot.NodeByID(loss.ID).Name {
				t.Fatal("backward binding changed forward graph names")
			}
		})
	}
}
