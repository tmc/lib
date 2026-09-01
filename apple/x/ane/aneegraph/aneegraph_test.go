//go:build darwin

package aneegraph

import (
	"errors"
	"strings"
	"testing"

	"github.com/tmc/apple/x/ane"
	"github.com/tmc/apple/x/ane/mil"
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

func TestCompileAndExecuteEGraphOnANE(t *testing.T) {
	client, err := ane.Open()
	if errors.Is(err, ane.ErrNoANE) {
		t.Skip("no ANE available on this device")
	}
	if err != nil {
		t.Fatalf("ANE Open failed: %v", err)
	}
	defer client.Close()

	// 1. Build an expression DAG and saturate through E-Graph
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
		t.Fatalf("E-Graph ExtractBest failed: %v", err)
	}

	// 2. Generate ANE MIL program
	const dim = 16
	opts := ANEOptions{
		SpatialDim:       1,
		HiddenDim:        dim,
		IntermediateDim:  32,
		WeightBlobOffset: 64,
	}

	milText, err := GenerateMIL(bestExpr, opts)
	if err != nil {
		t.Fatalf("GenerateMIL failed: %v", err)
	}

	// 3. Build weight blob using mil package
	blob, err := mil.BuildIdentityWeightBlob(dim)
	if err != nil {
		t.Fatalf("BuildIdentityWeightBlob failed: %v", err)
	}

	// 4. Compile onto Apple Neural Engine
	model, err := client.Compile(ane.CompileOptions{
		ModelType:  ane.ModelTypeMIL,
		MILText:    []byte(milText),
		WeightBlob: blob,
	})
	if err != nil {
		t.Fatalf("ANE Compile failed: %v", err)
	}
	defer model.Close()

	// 5. Run inference on Apple Neural Engine hardware
	inputVec := make([]float32, dim)
	for i := range inputVec {
		inputVec[i] = float32(i + 1)
	}

	if err := model.WriteInputF32(0, inputVec); err != nil {
		t.Fatalf("WriteInputF32 failed: %v", err)
	}

	if err := model.Eval(); err != nil {
		t.Fatalf("ANE Eval failed: %v", err)
	}

	outputVec := make([]float32, dim)
	if err := model.ReadOutputF32(0, outputVec); err != nil {
		t.Fatalf("ReadOutputF32 failed: %v", err)
	}

	t.Logf("=== Apple Neural Engine (ANE) E-Graph Hardware Evaluation ===")
	t.Logf("Model Arch: Fused RMSNorm-GEMV via E-Graph Rewrite")
	t.Logf("First 4 inputs:  %v", inputVec[:4])
	t.Logf("First 4 outputs: %v", outputVec[:4])

	diff := inputVec[0] - outputVec[0]
	if diff < -0.1 || diff > 0.1 {
		t.Errorf("expected parity with input on identity weights, got: %v", outputVec[:4])
	}
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

	const dim = 1024
	opts := ANEOptions{
		SpatialDim:       1,
		HiddenDim:        dim,
		IntermediateDim:  2048,
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
