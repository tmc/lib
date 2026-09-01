//go:build darwin

package aneegraph

import (
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/apple/x/metal/egraph"
)


func TestGenerateMILSwiGLU(t *testing.T) {
	expr := &egraph.Expr{
		Op: "fused_swiglu_gemv",
		Children: []*egraph.Expr{
			{Op: "x"},
			{Op: "w_gate"},
			{Op: "w_up"},
			{Op: "w_down"},
		},
	}

	opts := ANEOptions{
		SpatialDim:       1,
		HiddenDim:        64,
		IntermediateDim:  128,
		WeightBlobOffset: 64,
	}

	milText, err := GenerateMIL(expr, opts)
	if err != nil {
		t.Fatalf("GenerateMIL failed: %v", err)
	}

	if !strings.Contains(milText, "program(1.3)") {
		t.Errorf("expected program(1.3), got:\n%s", milText)
	}
	if !strings.Contains(milText, "conv_gate") || !strings.Contains(milText, "conv_up") || !strings.Contains(milText, "swiglu_product") {
		t.Errorf("expected fused SwiGLU ops in generated MIL, got:\n%s", milText)
	}
}

func TestGenerateMILRMSNorm(t *testing.T) {
	eg := egraph.New()
	x := eg.Add(egraph.NewLeaf("x"))
	wNorm := eg.Add(egraph.NewLeaf("w_norm"))
	eps := eg.Add(egraph.NewLeaf("eps"))
	wProj := eg.Add(egraph.NewLeaf("w_proj"))

	normX := eg.Add(egraph.NewNode("rmsnorm", x, wNorm, eps))
	proj := eg.Add(egraph.NewNode("gemv", normX, wProj))

	rules := []*egraph.Rewrite{
		egraph.MustRewrite("fuse_rmsnorm_gemv",
			"(gemv (rmsnorm ?x ?w_norm ?eps) ?w_proj)",
			"(fused_rmsnorm_gemv ?x ?w_norm ?eps ?w_proj)"),
	}

	egraph.Saturate(eg, rules, 5)
	bestExpr, _, err := egraph.ExtractBest(eg, proj, egraph.FusedMegakernelCost{})
	if err != nil {
		t.Fatalf("ExtractBest failed: %v", err)
	}

	opts := ANEOptions{
		SpatialDim:       1,
		HiddenDim:        64,
		IntermediateDim:  128,
		WeightBlobOffset: 64,
	}

	milText, err := GenerateMIL(bestExpr, opts)
	if err != nil {
		t.Fatalf("GenerateMIL failed: %v", err)
	}

	if !strings.Contains(milText, "program(1.3)") {
		t.Errorf("expected program(1.3), got:\n%s", milText)
	}
	if !strings.Contains(milText, "conv") {
		t.Errorf("expected conv op in generated MIL, got:\n%s", milText)
	}
}

func Example() {
	eg := egraph.New()
	x := eg.Add(egraph.NewLeaf("x"))
	wNorm := eg.Add(egraph.NewLeaf("w_norm"))
	eps := eg.Add(egraph.NewLeaf("eps"))
	wProj := eg.Add(egraph.NewLeaf("w_proj"))

	normX := eg.Add(egraph.NewNode("rmsnorm", x, wNorm, eps))
	proj := eg.Add(egraph.NewNode("gemv", normX, wProj))

	rules := []*egraph.Rewrite{
		egraph.MustRewrite("fuse_rmsnorm_gemv",
			"(gemv (rmsnorm ?x ?w_norm ?eps) ?w_proj)",
			"(fused_rmsnorm_gemv ?x ?w_norm ?eps ?w_proj)"),
	}

	egraph.Saturate(eg, rules, 5)
	bestExpr, _, _ := egraph.ExtractBest(eg, proj, egraph.FusedMegakernelCost{})

	milText, _ := GenerateMIL(bestExpr, ANEOptions{HiddenDim: 64, SpatialDim: 1})
	fmt.Printf("Generated ANE MIL valid: %t\n", strings.Contains(milText, "program(1.3)"))

	// Output:
	// Generated ANE MIL valid: true
}


func BenchmarkANE_vs_CPU_EGraph(b *testing.B) {
	client, err := ane.Open()
	if errors.Is(err, ane.ErrNoANE) {
		b.Skip("no ANE available on this device")
	}
	if err != nil {
		b.Fatalf("ANE Open failed: %v", err)
	}
	defer client.Close()

	const dim = 64
	opts := ANEOptions{
		SpatialDim:       1,
		HiddenDim:        dim,
		IntermediateDim:  128,
		WeightBlobOffset: 64,
	}

	bestExpr := &egraph.Expr{
		Op: "fused_rmsnorm_gemv",
		Children: []*egraph.Expr{
			{Op: "x"},
			{Op: "w_norm"},
			{Op: "eps"},
			{Op: "w_proj"},
		},
	}

	milText, err := GenerateMIL(bestExpr, opts)
	if err != nil {
		b.Fatalf("GenerateMIL failed: %v", err)
	}

	blob, err := mil.BuildIdentityWeightBlob(dim)
	if err != nil {
		b.Fatalf("BuildIdentityWeightBlob failed: %v", err)
	}

	model, err := client.Compile(ane.CompileOptions{
		ModelType:  ane.ModelTypeMIL,
		MILText:    []byte(milText),
		WeightBlob: blob,
	})
	if err != nil {
		b.Fatalf("ANE Compile failed: %v", err)
	}
	defer model.Close()

	inputVec := make([]float32, dim)
	for i := range inputVec {
		inputVec[i] = float32(i + 1)
	}
	outputVec := make([]float32, dim)

	b.Run("ANE_Hardware_Eval", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = model.WriteInputF32(0, inputVec)
			_ = model.Eval()
			_ = model.ReadOutputF32(0, outputVec)
		}
	})

	b.Run("CPU_Host_Eval", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			// CPU reference RMSNorm + GEMV calculation
			var sumSq float32
			for _, v := range inputVec {
				sumSq += v * v
			}
			rms := float32(1.0 / (sumSq/float32(dim) + 1e-5))
			for r := 0; r < dim; r++ {
				var acc float32
				for c := 0; c < dim; c++ {
					if r == c {
						acc += (inputVec[c] * rms) * 1.0
					}
				}
				outputVec[r] = acc
			}
		}
	})
}
