package egraph_test

import (
	"bytes"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/tileir"
)

func normalizedFixture(op string, shape egraph.TileNormalizedProjection) *egraph.Expr {
	names := []string{"x", "norm", "eps", "projection"}
	if strings.HasSuffix(op, "qkv_split") {
		names = []string{"x", "norm", "eps", "query", "key", "value"}
	}
	expr := &egraph.Expr{Op: op, Data: shape}
	for _, n := range names {
		expr.Children = append(expr.Children, egraph.NewLeafNode(n))
	}
	return expr
}

func projectionInputs(s egraph.TileNormalizedProjection, split bool, zero bool) [][]float32 {
	x := make([]float32, s.M*s.K)
	norm := make([]float32, s.K)
	for j := range x {
		if !zero {
			x[j] = float32((j*13+5)%31-15)/7 + float32(j/s.K)/11
		}
	}
	for j := range norm {
		norm[j] = float32(j%9-4)/5 + 0.3
	}
	buffers := [][]float32{x, norm}
	count := 1
	if split {
		count = 3
	}
	for p := 0; p < count; p++ {
		w := make([]float32, s.K*s.N)
		for j := range w {
			w[j] = float32((j*7+p*11+3)%37-18) / 17
		}
		buffers = append(buffers, w)
	}
	out := make([]float32, s.M*s.N)
	for j := range out {
		out[j] = float32(math.NaN())
	}
	return append(buffers, out)
}

func projectionMatches(buffers [][]float32, want []float64) bool {
	got := buffers[len(buffers)-1]
	for j, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.Abs(float64(v)-want[j]) > 2e-5+2e-5*math.Abs(want[j]) {
			return false
		}
	}
	return true
}

func TestNormalizedProjectionEncodedSemantics(t *testing.T) {
	for _, op := range []string{"tile_rmsnorm_proj", "fused_rmsnorm_proj", "tile_rmsnorm_qkv", "fused_rmsnorm_qkv", "tile_rmsnorm_qkv_split", "fused_rmsnorm_qkv_split"} {
		for _, s := range []egraph.TileNormalizedProjection{{1, 1, 1}, {2, 17, 15}, {3, 65, 17}, {17, 19, 65}, {2, 33, 128}} {
			for _, eps := range []float32{1e-5, 0.25, 3} {
				for _, zero := range []bool{false, true} {
					expr := normalizedFixture(op, s)
					mod, err := egraph.ToTileIR(expr, egraph.CodegenOptions{})
					if err != nil {
						t.Fatal(err)
					}
					wire, err := mod.EncodeBytecode()
					if err != nil {
						t.Fatal(err)
					}
					decoded, err := tileir.DecodeBytecodeBytes(wire)
					if err != nil {
						t.Fatal(err)
					}
					round, err := decoded.ToModule().EncodeBytecode()
					if err != nil || !bytes.Equal(wire, round) {
						t.Fatalf("roundtrip: %v", err)
					}
					split := strings.HasSuffix(op, "qkv_split")
					buffers := projectionInputs(s, split, zero)
					want := projectionOracle(buffers[0], buffers[1], buffers[2:len(buffers)-1], s.M, s.N, s.K, eps)
					grid, _ := s.Grid()
					if err := executeProjection(decoded.ToModule(), grid, buffers, eps); err != nil {
						t.Fatal(err)
					}
					if !projectionMatches(buffers, want) {
						t.Fatalf("%s shape%+v eps%g zero%v disagrees with scalar float64 oracle", op, s, eps, zero)
					}
				}
			}
		}
	}
}

func TestNormalizedProjectionAdmission(t *testing.T) {
	s := egraph.TileNormalizedProjection{2, 17, 65}
	for _, tc := range []struct {
		name   string
		change func(*egraph.Expr, *egraph.CodegenOptions)
	}{
		{"no shape", func(e *egraph.Expr, _ *egraph.CodegenOptions) { e.Data = nil }},
		{"zero K", func(e *egraph.Expr, _ *egraph.CodegenOptions) { e.Data = egraph.TileNormalizedProjection{1, 1, 0} }},
		{"excessive K", func(e *egraph.Expr, _ *egraph.CodegenOptions) {
			e.Data = egraph.TileNormalizedProjection{1, 1, 1 << 20}
		}},
		{"grid y limit", func(e *egraph.Expr, _ *egraph.CodegenOptions) {
			e.Data = egraph.TileNormalizedProjection{M: 65536, N: 1, K: 1}
		}},
		{"negative M", func(e *egraph.Expr, _ *egraph.CodegenOptions) { e.Data = egraph.TileNormalizedProjection{-1, 1, 1} }},
		{"excessive size", func(e *egraph.Expr, _ *egraph.CodegenOptions) {
			e.Data = egraph.TileNormalizedProjection{1 << 20, 2, 1}
		}},
		{"dtype", func(_ *egraph.Expr, o *egraph.CodegenOptions) { o.Precision = "f16" }},
		{"missing operand", func(e *egraph.Expr, _ *egraph.CodegenOptions) { e.Children = e.Children[:3] }},
		{"composite operand", func(e *egraph.Expr, _ *egraph.CodegenOptions) {
			e.Children[0].Children = []*egraph.Expr{{Op: "nested"}}
		}},
		{"literal epsilon", func(e *egraph.Expr, _ *egraph.CodegenOptions) { e.Children[2] = egraph.NewLeafNode(0.1) }},
		{"identifier", func(_ *egraph.Expr, o *egraph.CodegenOptions) { o.KernelName = "bad-name" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e := normalizedFixture("tile_rmsnorm_proj", s)
			var opts egraph.CodegenOptions
			tc.change(e, &opts)
			if _, err := egraph.ToTileIR(e, opts); err == nil {
				t.Fatal("invalid typed contract accepted")
			}
			if _, err := egraph.GenerateDeclarativeTile(e, opts); err == nil {
				t.Fatal("invalid source contract accepted")
			}
		})
	}
	for _, op := range []string{"tile_rmsnorm_gemv_swiglu", "fused_rmsnorm_swiglu", "tile_rmsnorm_swiglu", "fused_rmsnorm_gemv_swiglu"} {
		e := &egraph.Expr{Op: op}
		if egraph.IsLowerableTileIROp(op) {
			t.Fatalf("incomplete composite %s advertised", op)
		}
		if _, err := egraph.ToTileIR(e, egraph.CodegenOptions{}); err == nil || !strings.Contains(err.Error(), "full-row RMS") {
			t.Fatalf("composite diagnostic: %v", err)
		}
		if _, err := egraph.GenerateDeclarativeTile(e, egraph.CodegenOptions{}); err == nil {
			t.Fatal("composite source accepted")
		}
	}
}

func TestNormalizedProjectionABISource(t *testing.T) {
	s := egraph.TileNormalizedProjection{3, 65, 17}
	expr := normalizedFixture("tile_rmsnorm_qkv_split", s)
	mod, err := egraph.ToTileIR(expr, egraph.CodegenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	fn := mod.Functions[0]
	want := []string{"%X", "%W_norm", "%W_q", "%W_k", "%W_v", "%Out", "%eps"}
	if len(fn.Params) != len(want) {
		t.Fatal("ABI arity")
	}
	for j, p := range fn.Params {
		if p.Name != want[j] {
			t.Fatalf("param%d %s", j, p.Name)
		}
	}
	source, err := egraph.GenerateDeclarativeTile(expr, egraph.CodegenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	for _, token := range []string{"ct::rsqrt(", "ct::store_masked(", "ct::bid().y", "ct::select(", "float eps", "W_norm +", "W_q +", "W_k +", "W_v +"} {
		if !strings.Contains(source, token) {
			t.Fatalf("source missing %s", token)
		}
	}
	grid, err := s.Grid()
	if err != nil || grid != [3]uint32{5, 3, 1} {
		t.Fatalf("grid %v %v", grid, err)
	}
}

func TestNormalizedProjectionCausalOmissions(t *testing.T) {
	s := egraph.TileNormalizedProjection{M: 3, N: 65, K: 65}
	for _, mutant := range []string{"normalization", "elementwise rsqrt", "epsilon", "normalization weights", "projection weights", "wrong axis", "first reduction tile", "first output tile", "missing row", "missing store", "layout stride", "split only Q"} {
		t.Run(mutant, func(t *testing.T) {
			mod, err := egraph.ToTileIR(normalizedFixture("tile_rmsnorm_qkv_split", s), egraph.CodegenOptions{})
			if err != nil {
				t.Fatal(err)
			}
			fn := mod.Functions[0]
			defs := make(map[string]*tileir.Operation)
			for _, op := range fn.Operations {
				for _, r := range op.Results {
					defs[r] = op
				}
			}
			var root func(string) string
			root = func(v string) string {
				op := defs[v]
				if op == nil {
					return v
				}
				switch op.OpCode {
				case "cuda_tile.offset", "cuda_tile.reshape", "cuda_tile.broadcast":
					return root(op.Operands[0])
				}
				return ""
			}
			zero := ""
			inv := ""
			loads := make(map[string]string)
			for _, op := range fn.Operations {
				if op.OpCode == "cuda_tile.constant" && op.Attributes == "{value = 0}" && op.TypeVal.(*tileir.TileType).ElementType.Kind() == tileir.KindFloat32 {
					zero = op.Results[0]
				}
				if op.OpCode == "cuda_tile.rsqrt" {
					inv = op.Results[0]
				}
				if op.OpCode == "cuda_tile.load_ptr_tko" {
					loads[op.Results[0]] = root(op.Operands[0])
				}
			}
			changed := false
			grid, _ := s.Grid()
			switch mutant {
			case "normalization":
				op := defs[inv]
				op.OpCode = "cuda_tile.constant"
				op.Operands = nil
				op.Attributes = "{value = 1.0}"
				changed = true
			case "elementwise rsqrt":
				squares := make(map[string]string)
				for _, op := range fn.Operations {
					if op.OpCode == "cuda_tile.mulf" && op.Operands[0] == op.Operands[1] {
						squares[op.Operands[0]] = op.Results[0]
					}
				}
				var edited []*tileir.Operation
				for _, op := range fn.Operations {
					if op.OpCode == "cuda_tile.mulf" && op.Operands[1] == inv {
						name := op.Results[0] + "_elementwise_inverse"
						edited = append(edited, &tileir.Operation{OpCode: "cuda_tile.rsqrt", Results: []string{name}, Operands: []string{squares[op.Operands[0]]}, TypeVal: op.TypeVal})
						op.Operands[1] = name
						changed = true
					}
					edited = append(edited, op)
				}
				fn.Operations = edited
			case "epsilon":
				denom := defs[defs[inv].Operands[0]]
				denom.Operands[1] = zero
				changed = true
			case "normalization weights", "projection weights":
				for _, op := range fn.Operations {
					if op.OpCode == "cuda_tile.mulf" {
						base := loads[op.Operands[1]]
						if mutant == "normalization weights" && base == "%W_norm" || mutant == "projection weights" && (base == "%W_q" || base == "%W_k" || base == "%W_v") {
							op.OpCode = "cuda_tile.addf"
							op.Operands[1] = zero
							changed = true
						}
					}
				}
			case "first reduction tile":
				mean := defs[defs[defs[inv].Operands[0]].Operands[0]]
				sum := mean.Operands[0]
				for k := s.K; k > 16; k-- {
					sum = defs[sum].Operands[0]
				}
				mean.Operands[0] = sum
				changed = true
			case "wrong axis":
				// Change X's row base to column*inputWidth. The pointer-formation
				// check must reject even inactive addresses before dereference masks.
				var col string
				for _, op := range fn.Operations {
					if op.OpCode == "cuda_tile.cmpi" {
						col = op.Operands[0]
						break
					}
				}
				for _, op := range fn.Operations {
					if op.OpCode == "cuda_tile.load_ptr_tko" && root(op.Operands[0]) == "%X" {
						offset := defs[op.Operands[0]]
						safe := defs[offset.Operands[1]]
						add := defs[safe.Operands[1]]
						rowBase := defs[add.Operands[0]]
						rowBase.Operands[0] = col
						changed = true
						break
					}
				}
			case "first output tile":
				grid[0] = 1
				changed = true
			case "missing row":
				grid[1] = 1
				changed = true
			case "missing store":
				for j, op := range fn.Operations {
					if op.OpCode == "cuda_tile.store_ptr_tko" {
						fn.Operations = append(fn.Operations[:j], fn.Operations[j+1:]...)
						changed = true
						break
					}
				}
			case "layout stride":
				seen := make(map[*tileir.Operation]bool)
				for _, op := range fn.Operations {
					if op.OpCode == "cuda_tile.load_ptr_tko" && (root(op.Operands[0]) == "%W_q" || root(op.Operands[0]) == "%W_k" || root(op.Operands[0]) == "%W_v") {
						offset := defs[op.Operands[0]]
						safe := defs[offset.Operands[1]]
						add := defs[safe.Operands[1]]
						step := defs[add.Operands[0]]
						if seen[step] {
							continue
						}
						seen[step] = true
						var k int
						fmt.Sscanf(step.Attributes, "{value = %d}", &k)
						step.Attributes = fmt.Sprintf("{value = %d}", k/s.N)
						changed = true
					}
				}
			case "split only Q":
				for _, op := range fn.Operations {
					if op.OpCode == "cuda_tile.store_ptr_tko" {
						sum := defs[op.Operands[1]]
						sum.Operands[1] = zero
						changed = true
					}
				}
			}
			if !changed {
				t.Fatal("mutation did not edit emitted representation")
			}
			wire, err := mod.EncodeBytecode()
			if err != nil {
				t.Logf("rejected by real encoder: %v", err)
				return
			}
			decoded, err := tileir.DecodeBytecodeBytes(wire)
			if err != nil {
				t.Fatal(err)
			}
			buffers := projectionInputs(s, true, false)
			eps := float32(0.25)
			want := projectionOracle(buffers[0], buffers[1], buffers[2:len(buffers)-1], s.M, s.N, s.K, eps)
			if err := executeProjection(decoded.ToModule(), grid, buffers, eps); err != nil {
				t.Logf("rejected by pointer/store model: %v", err)
				return
			}
			if projectionMatches(buffers, want) {
				t.Fatal("causal omission survived independent oracle")
			}
		})
	}
}

func TestNormalizedProjectionExtractionBinding(t *testing.T) {
	g := egraph.New()
	x := g.Add(egraph.NewLeaf("x"))
	wn := g.Add(egraph.NewLeaf("norm"))
	eps := g.Add(egraph.NewLeaf("eps"))
	w := g.Add(egraph.NewLeaf("weights"))
	norm := g.Add(egraph.NewNode("rmsnorm", x, wn, eps))
	projection := g.Add(egraph.NewNode("matmul", norm, w))
	egraph.Saturate(g, egraph.TransformerTileRules(), 10)
	best, _, err := egraph.ExtractBest(g, projection, egraph.FusedMegakernelCost{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := egraph.ToTileIR(best, egraph.CodegenOptions{}); err == nil || !strings.Contains(err.Error(), "shape metadata") {
		t.Fatalf("unbound extracted shape: %v", err)
	}
	original := best.Fingerprint()
	best.Data = egraph.TileNormalizedProjection{M: 2, N: 17, K: 65}
	if best.Fingerprint() == original {
		t.Fatal("shape missing from expression fingerprint")
	}
	mod, err := egraph.ToTileIR(best, egraph.CodegenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	wire, err := mod.EncodeBytecode()
	if err != nil {
		t.Fatal(err)
	}
	best.Data = egraph.TileNormalizedProjection{M: 2, N: 19, K: 65}
	other, err := egraph.ToTileIR(best, egraph.CodegenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	changed, err := other.EncodeBytecode()
	if err != nil || bytes.Equal(wire, changed) {
		t.Fatalf("shape not reflected in bytecode: %v", err)
	}
	// Literal epsilon must survive auto-fusion and be rejected, not silently
	// rebound to the symbolic runtime eps ABI.
	unfused := egraph.NewNodeExpr("matmul", egraph.NewNodeExpr("rmsnorm", egraph.NewLeafNode("x"), egraph.NewLeafNode("norm"), egraph.NewLeafNode(0.1)), egraph.NewLeafNode("weights"))
	unfused.Data = egraph.TileNormalizedProjection{M: 2, N: 17, K: 65}
	if _, err := egraph.ToTileIR(unfused, egraph.CodegenOptions{}); err == nil {
		t.Fatal("auto-fused literal epsilon rebound")
	}
}

func TestNormalizedProjectionCompositeRefusal(t *testing.T) {
	for _, op := range []string{"tile_swiglu", "fused_swiglu"} {
		norm := egraph.NewNodeExpr("rmsnorm", egraph.NewLeafNode("x"), egraph.NewLeafNode("norm"), egraph.NewLeafNode("eps"))
		for _, input := range []*egraph.Expr{norm, egraph.NewNodeExpr("gemv", norm, egraph.NewLeafNode("projection")), egraph.NewNodeExpr("matmul", norm, egraph.NewLeafNode("projection")), normalizedFixture("tile_rmsnorm_proj", egraph.TileNormalizedProjection{M: 2, N: 17, K: 65})} {
			composite := egraph.NewNodeExpr(op, input, egraph.NewLeafNode("gate"), egraph.NewLeafNode("up"), egraph.NewLeafNode("down"))
			if _, err := egraph.ToTileIR(composite, egraph.CodegenOptions{}); err == nil || !strings.Contains(err.Error(), "full-row RMS") {
				t.Fatalf("composite admission: %v", err)
			}
			if _, err := egraph.GenerateDeclarativeTile(composite, egraph.CodegenOptions{}); err == nil {
				t.Fatal("composite source admitted")
			}
		}
	}
}

func TestNormalizedProjectionConstructionBudget(t *testing.T) {
	// M/N do not expand the unrolled source; a large buffer is not an
	// artificial rejection reason. This checks metadata, not allocations.
	if _, err := (egraph.TileNormalizedProjection{M: 65535, N: 4096, K: 8192}).Grid(); err != nil {
		t.Fatal(err)
	}
	// Exercise actual emitted/encoded support beyond the old arbitrary K128
	// limit without a large tensor allocation or any compiler/device work.
	expr := normalizedFixture("tile_rmsnorm_proj", egraph.TileNormalizedProjection{M: 1, N: 1, K: 4096})
	mod, err := egraph.ToTileIR(expr, egraph.CodegenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := mod.EncodeBytecode(); err != nil {
		t.Fatal(err)
	}
}
