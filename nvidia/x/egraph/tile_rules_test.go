package egraph_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/tileir"
	"github.com/tmc/lib/nvidia/x/tileiras"
)

func TestTransformerTileRulesTable(t *testing.T) {
	tests := []struct {
		name       string
		setupExpr  func(g *egraph.EGraph) egraph.Id
		wantOp     string
		wantCostLt float64
	}{
		{
			name: "RMSNorm -> QKV Projection GEMV",
			setupExpr: func(g *egraph.EGraph) egraph.Id {
				x := g.Add(egraph.NewLeaf("x"))
				wNorm := g.Add(egraph.NewLeaf("w_norm"))
				eps := g.Add(egraph.NewLeaf("eps"))
				wQKV := g.Add(egraph.NewLeaf("w_qkv"))

				norm := g.Add(egraph.NewNode("rmsnorm", x, wNorm, eps))
				return g.Add(egraph.NewNode("gemv", norm, wQKV))
			},
			wantOp:     "tile_rmsnorm_qkv",
			wantCostLt: 5.0,
		},
		{
			name: "RMSNorm -> QKV Projection Matmul",
			setupExpr: func(g *egraph.EGraph) egraph.Id {
				x := g.Add(egraph.NewLeaf("x"))
				wNorm := g.Add(egraph.NewLeaf("w_norm"))
				eps := g.Add(egraph.NewLeaf("eps"))
				wQKV := g.Add(egraph.NewLeaf("w_qkv"))

				norm := g.Add(egraph.NewNode("rmsnorm", x, wNorm, eps))
				return g.Add(egraph.NewNode("matmul", norm, wQKV))
			},
			wantOp:     "tile_rmsnorm_qkv",
			wantCostLt: 5.0,
		},
		{
			name: "RMSNorm -> Split QKV Projections",
			setupExpr: func(g *egraph.EGraph) egraph.Id {
				x := g.Add(egraph.NewLeaf("x"))
				wNorm := g.Add(egraph.NewLeaf("w_norm"))
				eps := g.Add(egraph.NewLeaf("eps"))
				wq := g.Add(egraph.NewLeaf("w_q"))
				wk := g.Add(egraph.NewLeaf("w_k"))
				wv := g.Add(egraph.NewLeaf("w_v"))

				norm := g.Add(egraph.NewNode("rmsnorm", x, wNorm, eps))
				qProj := g.Add(egraph.NewNode("gemv", norm, wq))
				kProj := g.Add(egraph.NewNode("gemv", norm, wk))
				vProj := g.Add(egraph.NewNode("gemv", norm, wv))

				kvSum := g.Add(egraph.NewNode("add", kProj, vProj))
				return g.Add(egraph.NewNode("add", qProj, kvSum))
			},
			wantOp:     "tile_rmsnorm_qkv_split",
			wantCostLt: 7.0,
		},
		{
			name: "Matmul -> SwiGLU FFN",
			setupExpr: func(g *egraph.EGraph) egraph.Id {
				x := g.Add(egraph.NewLeaf("x"))
				wg := g.Add(egraph.NewLeaf("w_g"))
				wu := g.Add(egraph.NewLeaf("w_u"))
				wd := g.Add(egraph.NewLeaf("w_d"))

				gate := g.Add(egraph.NewNode("matmul", x, wg))
				up := g.Add(egraph.NewNode("matmul", x, wu))
				silu := g.Add(egraph.NewNode("silu", gate))
				act := g.Add(egraph.NewNode("mul", silu, up))
				return g.Add(egraph.NewNode("matmul", act, wd))
			},
			wantOp:     "tile_swiglu",
			wantCostLt: 5.0,
		},
		{
			name: "GEMV -> GeLU FFN",
			setupExpr: func(g *egraph.EGraph) egraph.Id {
				x := g.Add(egraph.NewLeaf("x"))
				wi := g.Add(egraph.NewLeaf("w_in"))
				wo := g.Add(egraph.NewLeaf("w_out"))

				inProj := g.Add(egraph.NewNode("gemv", x, wi))
				gelu := g.Add(egraph.NewNode("gelu", inProj))
				return g.Add(egraph.NewNode("gemv", gelu, wo))
			},
			wantOp:     "tile_gelu",
			wantCostLt: 4.0,
		},
		{
			name: "Matmul -> Gated GeLU (GeGLU)",
			setupExpr: func(g *egraph.EGraph) egraph.Id {
				x := g.Add(egraph.NewLeaf("x"))
				wg := g.Add(egraph.NewLeaf("w_g"))
				wu := g.Add(egraph.NewLeaf("w_u"))
				wd := g.Add(egraph.NewLeaf("w_d"))

				gate := g.Add(egraph.NewNode("matmul", x, wg))
				up := g.Add(egraph.NewNode("matmul", x, wu))
				gelu := g.Add(egraph.NewNode("gelu", gate))
				act := g.Add(egraph.NewNode("mul", gelu, up))
				return g.Add(egraph.NewNode("matmul", act, wd))
			},
			wantOp:     "tile_geglu",
			wantCostLt: 5.0,
		},
	}

	rules := egraph.TransformerTileRules()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := egraph.New()
			root := tt.setupExpr(g)

			iters, merges := egraph.Saturate(g, rules, 10)
			if iters == 0 || merges == 0 {
				t.Fatalf("expected rewrites to apply: iters=%d, merges=%d", iters, merges)
			}

			best, cost, err := egraph.ExtractBest(g, root, egraph.FusedMegakernelCost{})
			if err != nil {
				t.Fatalf("ExtractBest failed: %v", err)
			}

			if best.Op != tt.wantOp {
				t.Errorf("best.Op = %q, want %q (expr: %s)", best.Op, tt.wantOp, best)
			}
			if cost >= tt.wantCostLt {
				t.Errorf("cost = %f, want < %f", cost, tt.wantCostLt)
			}

			// Verify declarative C++ Tile IR emission with __tile_global__
			src, err := egraph.GenerateDeclarativeTile(best, egraph.CodegenOptions{
				KernelName: "TestKernel_" + tt.wantOp,
			})
			if err != nil {
				t.Fatalf("GenerateDeclarativeTile failed: %v", err)
			}
			if !strings.Contains(src, "__tile_global__") {
				t.Errorf("expected source to contain '__tile_global__':\n%s", src)
			}
			if !strings.Contains(src, "using namespace cuda::tiles;") {
				t.Errorf("expected source to use cuda::tiles namespace:\n%s", src)
			}
			if !strings.Contains(src, "assume_aligned<16>") {
				t.Errorf("expected source to enforce 16-byte TMA alignment:\n%s", src)
			}
		})
	}
}

func TestTMALayoutRulesTable(t *testing.T) {
	tests := []struct {
		name      string
		setupExpr func(g *egraph.EGraph) egraph.Id
		wantSub   string
	}{
		{
			name: "TMA 16-byte load alignment",
			setupExpr: func(g *egraph.EGraph) egraph.Id {
				p := g.Add(egraph.NewLeaf("p"))
				shape := g.Add(egraph.NewLeaf("shape_16_16"))
				return g.Add(egraph.NewNode("tile_load", p, shape))
			},
			wantSub: "(tile_tma_load (assume_aligned p 16) (tma_box shape_16_16 16))",
		},
		{
			name: "TMA non-contiguous stride elimination",
			setupExpr: func(g *egraph.EGraph) egraph.Id {
				p := g.Add(egraph.NewLeaf("p"))
				stride := g.Add(egraph.NewLeaf("stride_dim1"))
				return g.Add(egraph.NewNode("load_strided", p, stride))
			},
			wantSub: "(tile_tma_load (coalesce_contiguous p) (tma_stride_1 stride_dim1))",
		},
		{
			name: "Tensor Core MMA alignment preventing FMA fallback (Shape [1,1,1] -> [16,16,16])",
			setupExpr: func(g *egraph.EGraph) egraph.Id {
				shape1 := g.Add(egraph.NewNode("shape",
					g.Add(egraph.NewLeaf("1")),
					g.Add(egraph.NewLeaf("1")),
					g.Add(egraph.NewLeaf("1")),
				))
				a := g.Add(egraph.NewLeaf("a"))
				b := g.Add(egraph.NewLeaf("b"))
				c := g.Add(egraph.NewLeaf("c"))
				return g.Add(egraph.NewNode("tile_mma", shape1, a, b, c))
			},
			wantSub: "(tile_mma_aligned (shape 16 16 16) a b c)",
		},
		{
			name: "TMA partition view alignment",
			setupExpr: func(g *egraph.EGraph) egraph.Id {
				span := g.Add(egraph.NewLeaf("span"))
				shape := g.Add(egraph.NewLeaf("shape_32_32"))
				return g.Add(egraph.NewNode("partition_view", span, shape))
			},
			wantSub: "(tma_partition_view (assume_aligned span 16) (tma_box shape_32_32 16))",
		},
	}

	rules := egraph.TMALayoutRules()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := egraph.New()
			root := tt.setupExpr(g)

			iters, merges := egraph.Saturate(g, rules, 5)
			if iters == 0 || merges == 0 {
				t.Fatalf("expected TMA layout rewrites to apply: iters=%d, merges=%d", iters, merges)
			}

			best, _, err := egraph.ExtractBest(g, root, egraph.FusedMegakernelCost{})
			if err != nil {
				t.Fatalf("ExtractBest failed: %v", err)
			}

			if best.String() != tt.wantSub {
				t.Errorf("got %s, want %s", best.String(), tt.wantSub)
			}
		})
	}
}

func TestSubByteFormatCoverageTable(t *testing.T) {
	tests := []struct {
		name         string
		setupExpr    func(g *egraph.EGraph) egraph.Id
		wantTileOp   string
		wantUnswizOp string
	}{
		{
			name: "Q8_0 quantized GEMV with half scale",
			setupExpr: func(g *egraph.EGraph) egraph.Id {
				x := g.Add(egraph.NewLeaf("x"))
				w := g.Add(egraph.NewLeaf("w"))
				scale := g.Add(egraph.NewLeaf("scale"))
				dequant := g.Add(egraph.NewNode("dequantize_q8_0", w, scale))
				return g.Add(egraph.NewNode("matmul", x, dequant))
			},
			wantTileOp:   "tile_mma",
			wantUnswizOp: "ptx_unpack_q8_0",
		},
		{
			name: "IQ2_XXS sub-byte 2-bit quantization (DeepSeek-V4)",
			setupExpr: func(g *egraph.EGraph) egraph.Id {
				x := g.Add(egraph.NewLeaf("x"))
				w := g.Add(egraph.NewLeaf("w"))
				scales := g.Add(egraph.NewLeaf("scales"))
				grid := g.Add(egraph.NewLeaf("grid"))
				dequant := g.Add(egraph.NewNode("dequantize_iq2_xxs", w, scales, grid))
				return g.Add(egraph.NewNode("matmul", x, dequant))
			},
			wantTileOp:   "tile_mma",
			wantUnswizOp: "ptx_unswizzle_iq2_xxs",
		},
		{
			name: "FP8 E4M3 quantization",
			setupExpr: func(g *egraph.EGraph) egraph.Id {
				x := g.Add(egraph.NewLeaf("x"))
				w := g.Add(egraph.NewLeaf("w"))
				scale := g.Add(egraph.NewLeaf("scale"))
				dequant := g.Add(egraph.NewNode("dequantize_fp8_e4m3", w, scale))
				return g.Add(egraph.NewNode("matmul", x, dequant))
			},
			wantTileOp:   "tile_mma_fp8",
			wantUnswizOp: "ptx_unpack_fp8_e4m3",
		},
		{
			name: "FP8 E5M2 quantization",
			setupExpr: func(g *egraph.EGraph) egraph.Id {
				x := g.Add(egraph.NewLeaf("x"))
				w := g.Add(egraph.NewLeaf("w"))
				scale := g.Add(egraph.NewLeaf("scale"))
				dequant := g.Add(egraph.NewNode("dequantize_fp8_e5m2", w, scale))
				return g.Add(egraph.NewNode("matmul", x, dequant))
			},
			wantTileOp:   "tile_mma_fp8",
			wantUnswizOp: "ptx_unpack_fp8_e5m2",
		},
		{
			name: "FP4 E2M1 4-bit quantization",
			setupExpr: func(g *egraph.EGraph) egraph.Id {
				x := g.Add(egraph.NewLeaf("x"))
				w := g.Add(egraph.NewLeaf("w"))
				scale := g.Add(egraph.NewLeaf("scale"))
				dequant := g.Add(egraph.NewNode("dequantize_fp4_e2m1", w, scale))
				return g.Add(egraph.NewNode("matmul", x, dequant))
			},
			wantTileOp:   "mmaf_scaled",
			wantUnswizOp: "ptx_unswizzle_fp4_e2m1",
		},
	}

	rules := egraph.SubByteRules()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := egraph.New()
			root := tt.setupExpr(g)

			iters, merges := egraph.Saturate(g, rules, 5)
			if iters == 0 || merges == 0 {
				t.Fatalf("expected sub-byte rewrites to apply: iters=%d, merges=%d", iters, merges)
			}

			best, _, err := egraph.ExtractBest(g, root, egraph.FusedMegakernelCost{})
			if err != nil {
				t.Fatalf("ExtractBest failed: %v", err)
			}

			if best.Op != tt.wantTileOp {
				t.Errorf("best.Op = %q, want %q (full expr: %s)", best.Op, tt.wantTileOp, best)
			}

			// Verify that the PTX bit-unswizzling node is cleanly isolated in child arguments
			foundUnswiz := false
			var checkChildren func(e *egraph.Expr)
			checkChildren = func(e *egraph.Expr) {
				if e == nil {
					return
				}
				if strings.Contains(e.Op, tt.wantUnswizOp) {
					foundUnswiz = true
				}
				for _, child := range e.Children {
					checkChildren(child)
				}
			}
			checkChildren(best)

			if !foundUnswiz {
				t.Errorf("expected to find isolated PTX unswizzling op %q in expr: %s", tt.wantUnswizOp, best)
			}
		})
	}
}

func TestEndToEndTileIRLowering(t *testing.T) {
	// 1. Build an unoptimized transformer feed-forward expression (GEMV -> SiLU -> GEMV -> GEMV)
	g := egraph.New()
	x := g.Add(egraph.NewLeaf("x"))
	wg := g.Add(egraph.NewLeaf("w_gate"))
	wu := g.Add(egraph.NewLeaf("w_up"))
	wd := g.Add(egraph.NewLeaf("w_down"))

	gate := g.Add(egraph.NewNode("gemv", x, wg))
	up := g.Add(egraph.NewNode("gemv", x, wu))
	silu := g.Add(egraph.NewNode("silu", gate))
	act := g.Add(egraph.NewNode("mul", silu, up))
	ffnOut := g.Add(egraph.NewNode("gemv", act, wd))

	// 2. Saturate with transformer tile rules
	rules := egraph.AllTileRules()
	iters, merges := egraph.Saturate(g, rules, 15)
	if iters == 0 || merges == 0 {
		t.Fatalf("expected saturation to apply rules: iters=%d, merges=%d", iters, merges)
	}

	// 3. Extract optimal declarative Tile IR expression
	best, cost, err := egraph.ExtractBest(g, ffnOut, egraph.FusedMegakernelCost{})
	if err != nil {
		t.Fatalf("ExtractBest failed: %v", err)
	}
	if best.Op != "tile_swiglu" {
		t.Fatalf("extracted best.Op = %q, want tile_swiglu", best.Op)
	}
	t.Logf("Extracted optimal Tile IR plan: %s (cost: %.2f)", best, cost)

	// 4. Lower AST expression to Tile IR Module (tileir.Module)
	mod, err := egraph.ToTileIR(best, egraph.CodegenOptions{
		KernelName: "fused_swiglu_tile",
		Precision:  "f16",
	})
	if err != nil {
		t.Fatalf("ToTileIR failed: %v", err)
	}

	// 5. Verify Tile IR textual MLIR emission and parse round-trip
	textIR := mod.String()
	if !strings.Contains(textIR, "cuda_tile.entry @fused_swiglu_tile") {
		t.Fatalf("textual Tile IR missing entry function header:\n%s", textIR)
	}
	if !strings.Contains(textIR, "cuda_tile.mma_f") {
		t.Fatalf("textual Tile IR missing mma_f operation:\n%s", textIR)
	}

	parsedMod, err := tileir.ParseString(textIR)
	if err != nil {
		t.Fatalf("tileir.ParseString failed on generated Tile IR text: %v\n%s", err, textIR)
	}
	if len(parsedMod.Functions) != 1 {
		t.Fatalf("parsed %d functions, want 1", len(parsedMod.Functions))
	}
	if parsedMod.Functions[0].Name != "fused_swiglu_tile" {
		t.Errorf("parsed function name = %q, want 'fused_swiglu_tile'", parsedMod.Functions[0].Name)
	}

	// 6. Verify Tile IR bytecode serialization
	bytecode, err := mod.EncodeBytecode()
	if err != nil {
		t.Fatalf("EncodeBytecode failed: %v", err)
	}

	if !tileiras.IsBytecode(bytecode) {
		t.Fatalf("generated bytecode not recognized by tileiras.IsBytecode")
	}

	decodedMod, err := tileir.DecodeBytecodeBytes(bytecode)
	if err != nil {
		t.Fatalf("DecodeBytecodeBytes failed: %v", err)
	}

	if len(decodedMod.Functions) != 1 {
		t.Fatalf("decoded %d functions, want 1", len(decodedMod.Functions))
	}
	decodedFn := decodedMod.Functions[0]
	if decodedFn.Name != "fused_swiglu_tile" {
		t.Errorf("decoded function name = %q, want 'fused_swiglu_tile'", decodedFn.Name)
	}
	if !decodedFn.IsEntry {
		t.Errorf("decoded function IsEntry = false, want true")
	}
}
