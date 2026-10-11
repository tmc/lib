package egraph_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/tileir"
)

// TestGenerateTileIR_RMSNormGEMVSwiGLU verifies lowering of a complete fused transformer block:
// RMSNorm -> GEMV Projection -> SwiGLU FFN.
func TestGenerateTileIR_RMSNormGEMVSwiGLU(t *testing.T) {
	// 1. Build an E-Graph with an unfused transformer block subgraph
	g := egraph.New()
	x := g.Add(egraph.NewLeaf("x"))
	wNorm := g.Add(egraph.NewLeaf("w_norm"))
	eps := g.Add(egraph.NewLeaf("eps"))
	wProj := g.Add(egraph.NewLeaf("w_proj"))
	wg := g.Add(egraph.NewLeaf("w_gate"))
	wu := g.Add(egraph.NewLeaf("w_up"))
	wd := g.Add(egraph.NewLeaf("w_down"))

	norm := g.Add(egraph.NewNode("rmsnorm", x, wNorm, eps))
	proj := g.Add(egraph.NewNode("gemv", norm, wProj))
	gate := g.Add(egraph.NewNode("gemv", proj, wg))
	up := g.Add(egraph.NewNode("gemv", proj, wu))
	silu := g.Add(egraph.NewNode("silu", gate))
	act := g.Add(egraph.NewNode("mul", silu, up))
	ffnOut := g.Add(egraph.NewNode("gemv", act, wd))

	// 2. Saturate using transformer tile rules
	rules := egraph.TransformerTileRules()
	iters, merges := egraph.Saturate(g, rules, 20)
	if iters == 0 || merges == 0 {
		t.Fatalf("expected saturation to apply rules: iters=%d, merges=%d", iters, merges)
	}

	// 3. Extract optimal fused Tile IR expression
	best, cost, err := egraph.ExtractBest(g, ffnOut, egraph.FusedMegakernelCost{})
	if err != nil {
		t.Fatalf("ExtractBest failed: %v", err)
	}
	if best.Op != "tile_rmsnorm_gemv_swiglu" {
		t.Fatalf("extracted best.Op = %q, want 'tile_rmsnorm_gemv_swiglu'", best.Op)
	}
	t.Logf("Extracted optimal Tile IR plan: %s (cost: %.2f)", best, cost)

	// 4. Lower to Tile IR textual assembly
	opts := egraph.CodegenOptions{
		KernelName: "fused_rmsnorm_gemv_swiglu_kernel",
		Precision:  "f16",
	}
	textIR, err := egraph.GenerateTileIR(best, opts)
	if err != nil {
		t.Fatalf("GenerateTileIR failed: %v", err)
	}

	// 5. Verify generated Tile IR contains required MLIR constructs and tile math primitives
	checks := []string{
		"cuda_tile.module @kernels",
		"cuda_tile.entry @fused_rmsnorm_gemv_swiglu_kernel",
		"%tok0 = cuda_tile.make_token : !cuda_tile.token",
		"cuda_tile.load_view_tko",
		"cuda_tile.mulf",
		"cuda_tile.rsqrt",
		"cuda_tile.mmaf",
		"cuda_tile.exp",
		"cuda_tile.return",
	}
	for _, check := range checks {
		if !strings.Contains(textIR, check) {
			t.Errorf("GenerateTileIR missing expected construct %q in output:\n%s", check, textIR)
		}
	}

	// 6. Verify round-trip parsing of the generated Tile IR
	parsedMod, err := tileir.ParseString(textIR)
	if err != nil {
		t.Fatalf("tileir.ParseString failed on generated Tile IR: %v\n%s", err, textIR)
	}
	if len(parsedMod.Functions) != 1 {
		t.Fatalf("parsed %d functions, want 1", len(parsedMod.Functions))
	}
	fn := parsedMod.Functions[0]
	if fn.Name != opts.KernelName {
		t.Errorf("parsed function name = %q, want %q", fn.Name, opts.KernelName)
	}
	if !fn.Entry {
		t.Errorf("parsed function Entry = false, want true")
	}
	if len(fn.Params) != 7 {
		t.Errorf("len(Params) = %d, want 7 (%%X, %%W_norm, %%W_proj, %%W_gate, %%W_up, %%W_down, %%Out)", len(fn.Params))
	}

	if bc, err := parsedMod.EncodeBytecode(); bc != nil || !errors.Is(err, tileir.ErrUnsupportedASTEncoding) {
		t.Fatalf("incomplete AST encoding = %x, %v", bc, err)
	}
	if cubin, err := egraph.AssembleTileIR(context.Background(), best, opts); cubin != nil || !errors.Is(err, tileir.ErrUnsupportedASTEncoding) {
		t.Fatalf("incomplete AST assembly = %x, %v", cubin, err)
	}
}

// TestGenerateTileIR_AllPatterns tests lowering across all supported fused tile operators.
func TestGenerateTileIR_AllPatterns(t *testing.T) {
	tests := []struct {
		name       string
		op         string
		precision  string
		wantSubstr []string
	}{
		{
			name:      "tile_rmsnorm_qkv",
			op:        "tile_rmsnorm_qkv",
			precision: "f16",
			wantSubstr: []string{
				"cuda_tile.entry @rmsnorm_qkv_kernel",
				"cuda_tile.rsqrt",
				"cuda_tile.mmaf",
			},
		},
		{
			name:      "tile_rmsnorm_qkv_split",
			op:        "tile_rmsnorm_qkv_split",
			precision: "f16",
			wantSubstr: []string{
				"cuda_tile.entry @rmsnorm_qkv_split_kernel",
				"cuda_tile.rsqrt",
				"cuda_tile.mmaf",
				"cuda_tile.addf",
			},
		},
		{
			name:      "tile_rmsnorm_proj",
			op:        "tile_rmsnorm_proj",
			precision: "f32",
			wantSubstr: []string{
				"cuda_tile.entry @rmsnorm_proj_kernel",
				"cuda_tile.rsqrt",
				"cuda_tile.mmaf",
			},
		},
		{
			name:      "tile_swiglu",
			op:        "tile_swiglu",
			precision: "f16",
			wantSubstr: []string{
				"cuda_tile.entry @swiglu_kernel",
				"cuda_tile.mmaf",
				"cuda_tile.exp",
				"cuda_tile.mulf",
			},
		},
		{
			name:      "tile_geglu",
			op:        "tile_geglu",
			precision: "f16",
			wantSubstr: []string{
				"cuda_tile.entry @geglu_kernel",
				"cuda_tile.mmaf",
				"cuda_tile.mulf",
			},
		},
		{
			name:      "tile_gelu",
			op:        "tile_gelu",
			precision: "f32",
			wantSubstr: []string{
				"cuda_tile.entry @gelu_kernel",
				"cuda_tile.mmaf",
				"cuda_tile.exp",
			},
		},
		{
			name:      "tile_fp8_gemm",
			op:        "tile_fp8_gemm",
			precision: "fp8",
			wantSubstr: []string{
				"cuda_tile.entry @fp8_gemm_kernel",
				"f8E4M3FN",
				"cuda_tile.mmaf",
			},
		},
		{
			name:      "tile_fp4_gemm",
			op:        "tile_fp4_gemm",
			precision: "fp4",
			wantSubstr: []string{
				"cuda_tile.entry @fp4_gemm_kernel",
				"f4E2M1FN",
				"cuda_tile.mmaf",
			},
		},
		{
			name:      "tile_mma",
			op:        "tile_mma",
			precision: "bf16",
			wantSubstr: []string{
				"cuda_tile.entry @mma_kernel",
				"cuda_tile.mmaf",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node := &egraph.Node{Op: tt.op}
			opts := egraph.CodegenOptions{
				KernelName: strings.TrimPrefix(tt.op, "tile_") + "_kernel",
				Precision:  tt.precision,
			}

			// Test lowering with Lowerer and GenerateTileIR
			lowerer := egraph.NewLowerer(opts)
			mod, err := lowerer.Lower(node)
			if err != nil {
				t.Fatalf("Lowerer.Lower(%s): %v", tt.op, err)
			}

			textIR, err := egraph.GenerateTileIR(node, opts)
			if err != nil {
				t.Fatalf("GenerateTileIR(%s): %v", tt.op, err)
			}

			for _, sub := range tt.wantSubstr {
				if !strings.Contains(textIR, sub) {
					t.Errorf("GenerateTileIR(%s) missing %q in output:\n%s", tt.op, sub, textIR)
				}
			}

			// Verify parse roundtrip
			parsed, err := tileir.ParseString(textIR)
			if err != nil {
				t.Fatalf("tileir.ParseString failed for %s: %v", tt.op, err)
			}
			if len(parsed.Functions) != 1 {
				t.Errorf("parsed %d functions for %s, want 1", len(parsed.Functions), tt.op)
			}

			if bc, err := mod.EncodeBytecode(); bc != nil || !errors.Is(err, tileir.ErrUnsupportedASTEncoding) {
				t.Fatalf("incomplete AST encoding = %x, %v", bc, err)
			}
			if cubin, err := egraph.AssembleTileIR(context.Background(), node, opts); cubin != nil || !errors.Is(err, tileir.ErrUnsupportedASTEncoding) {
				t.Fatalf("incomplete AST assembly = %x, %v", cubin, err)
			}
		})
	}
}

// TestGenerateTileIR_AutoFusion verifies that passing an unfused AST expression directly
// to Lowerer automatically fuses the pattern before lowering.
func TestGenerateTileIR_AutoFusion(t *testing.T) {
	// Build an unfused composite AST expression: (gemv (rmsnorm x w_norm eps) w_qkv)
	x := egraph.NewLeafNode("x")
	wNorm := egraph.NewLeafNode("w_norm")
	eps := egraph.NewLeafNode("eps")
	wQKV := egraph.NewLeafNode("w_qkv")

	norm := egraph.NewNodeExpr("rmsnorm", x, wNorm, eps)
	gemv := egraph.NewNodeExpr("gemv", norm, wQKV)

	textIR, err := egraph.GenerateTileIR(gemv, egraph.CodegenOptions{
		KernelName: "auto_fused_rmsnorm_qkv",
		Precision:  "f16",
	})
	if err != nil {
		t.Fatalf("GenerateTileIR auto-fusion failed: %v", err)
	}

	if !strings.Contains(textIR, "cuda_tile.entry @auto_fused_rmsnorm_qkv") {
		t.Errorf("missing kernel entry name in auto-fused IR:\n%s", textIR)
	}
	if !strings.Contains(textIR, "cuda_tile.rsqrt") {
		t.Errorf("missing RMSNorm rsqrt in auto-fused IR:\n%s", textIR)
	}
	if !strings.Contains(textIR, "cuda_tile.mmaf") {
		t.Errorf("missing MMA in auto-fused IR:\n%s", textIR)
	}
}

// TestGenerateTileIR_NilNode verifies nil expression handling.
func TestGenerateTileIR_NilNode(t *testing.T) {
	_, err := egraph.GenerateTileIR(nil)
	if err == nil {
		t.Fatal("expected error for nil node, got nil")
	}
	if !strings.Contains(err.Error(), "nil node") {
		t.Errorf("error = %q, want 'nil node'", err.Error())
	}
}

// TestGenerateTileIR_Precisions tests precision mapping.
func TestGenerateTileIR_Precisions(t *testing.T) {
	tests := []struct {
		prec     string
		wantType string
	}{
		{"f16", "f16"},
		{"half", "f16"},
		{"f32", "f32"},
		{"float", "f32"},
		{"bf16", "bf16"},
		{"fp8", "f8E4M3FN"},
		{"fp4", "f4E2M1FN"},
	}

	for _, tt := range tests {
		t.Run(tt.prec, func(t *testing.T) {
			node := &egraph.Node{Op: "tile_mma"}
			textIR, err := egraph.GenerateTileIR(node, egraph.CodegenOptions{
				KernelName: "prec_test",
				Precision:  tt.prec,
			})
			if err != nil {
				t.Fatalf("GenerateTileIR(%s): %v", tt.prec, err)
			}
			if !strings.Contains(textIR, tt.wantType) {
				t.Errorf("textIR for precision %s does not contain %s:\n%s", tt.prec, tt.wantType, textIR)
			}
		})
	}
}
