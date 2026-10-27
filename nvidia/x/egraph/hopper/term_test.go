package hopper

import (
	"fmt"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
)

// runPlan runs every kernel of p emulated, in order, and returns the output.
func runPlan(t *testing.T, p *Plan, env map[string]Value) []float64 {
	t.Helper()
	var src strings.Builder
	var launches []string
	wsZero := make(map[string]int)
	for _, k := range p.Kernels {
		if k.GEMM != nil {
			s, err := k.GEMM.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			src.WriteString(s)
			grid, block, _ := k.GEMM.Grid()
			launches = append(launches, fmt.Sprintf("eg_emu_launch(dim3(%d, %d), dim3(%d), 0, %s<%d>, %s)",
				grid[0], grid[1], block[0], k.GEMM.Name, k.M, strings.Join(k.Args, ", ")))
			continue
		}
		if k.Tile != nil {
			s, err := k.Tile.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			src.WriteString(s)
			if k.Tile.Grouped {
				grid, block := k.Tile.Grid(k.MaxRows, k.Experts)
				launches = append(launches, fmt.Sprintf("eg_emu_launch(dim3(%d, %d, %d), dim3(%d), %d, %s, %d, %s)",
					grid[0], grid[1], grid[2], block[0], k.Tile.SharedBytes(), k.Tile.Name, k.M, strings.Join(k.Args, ", ")))
			} else {
				grid, block := k.Tile.Grid(k.M, 1)
				launches = append(launches, fmt.Sprintf("eg_emu_launch(dim3(%d, %d), dim3(%d), %d, %s, %d, %s)",
					grid[0], grid[1], block[0], k.Tile.SharedBytes(), k.Tile.Name, k.M, strings.Join(k.Args, ", ")))
			}
			continue
		}
		if k.Attention != nil {
			s, err := k.Attention.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			l, err := k.Attention.Launcher()
			if err != nil {
				t.Fatal(err)
			}
			src.WriteString(s + "\n" + l)
			args := append([]string(nil), k.Args[:4]...)
			args = append(args, fmt.Sprint(k.MaxPages))
			args = append(args, k.Args[4:]...)
			launches = append(launches, fmt.Sprintf("eg_emu_sms = 16; eg_emu_blocks_per_sm = 1; %s_launch(%d, %d, %s, nullptr)",
				k.Attention.Name, k.Batch, k.MaxKeys, strings.Join(args, ", ")))
			wsName := k.Args[len(k.Args)-1]
			wsZero[wsName] = attnCounterBytes(*k.Attention, k.Batch, k.Attention.Splits(k.MaxKeys))
			continue
		}
		if k.TopK != nil {
			s, err := k.TopK.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			src.WriteString(s)
			grid, block := k.TopK.Grid(k.Rows)
			launches = append(launches, fmt.Sprintf("eg_emu_launch(dim3(%d), dim3(%d), 0, %s, %d, %s)",
				grid[0], block[0], k.TopK.Name, k.Rows, strings.Join(k.Args, ", ")))
			continue
		}
		if k.Align != nil {
			s, err := k.Align.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			src.WriteString(s)
			al, _ := k.Align.norm()
			launches = append(launches, fmt.Sprintf("eg_emu_launch(dim3(1), dim3(%d), 0, %s, %d, %s)",
				32*al.Warps, k.Align.Name, k.Rows, strings.Join(k.Args, ", ")))
			continue
		}
		if k.ExpertSum != nil {
			s, err := k.ExpertSum.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			src.WriteString(s)
			grid, block := k.ExpertSum.Grid(k.Rows)
			launches = append(launches, fmt.Sprintf("eg_emu_launch(dim3(%d), dim3(%d), 0, %s, %d, %s)",
				grid[0], block[0], k.ExpertSum.Name, k.Rows, strings.Join(k.Args, ", ")))
			continue
		}
		s, err := k.Row.CUDA()
		if err != nil {
			t.Fatal(err)
		}
		src.WriteString(s)
		if k.Row.Cluster {
			l, err := k.Row.Launcher()
			if err != nil {
				t.Fatal(err)
			}
			src.WriteString("\n" + l)
			launches = append(launches, fmt.Sprintf("%s_launch(%d, %s, nullptr)",
				k.Row.Name, k.Rows, strings.Join(k.Args, ", ")))
			continue
		}
		grid, block := k.Row.Grid(k.Rows)
		launches = append(launches, fmt.Sprintf("eg_emu_launch(dim3(%d), dim3(%d), 0, %s, %d, %s)",
			grid[0], block[0], k.Row.Name, k.Rows, strings.Join(k.Args, ", ")))
	}
	var bufs []emuBuf
	for _, in := range p.Inputs {
		bufs = append(bufs, emuBuf{in.Name, in.DType.cType(), typed{in.DType, env[in.Name].Data}.bytes()})
	}
	for _, tmp := range append(p.Temps, p.Output) {
		raw := poison(size(tmp.Shape) * tmp.DType.Bits() / 8)
		if z, ok := wsZero[tmp.Name]; ok && z <= len(raw) {
			clear(raw[:z])
		}
		bufs = append(bufs, emuBuf{tmp.Name, tmp.DType.cType(), raw})
	}
	res := runEmu(t, src.String(), strings.Join(launches, ";\n    "), bufs...)
	return decode(p.Output.DType, res[p.Output.Name])
}

func randEnv(r *rand.Rand, ts ...Tensor) map[string]Value {
	env := make(map[string]Value)
	for _, t := range ts {
		v := randVals(r, size(t.Shape), 1)
		for i := range v {
			v[i] = float64(t.DType.Round(float32(v[i])))
		}
		env[t.Name] = Value{Shape: t.Shape, Data: v}
	}
	return env
}

func TestOptimizeFusesEpilogue(t *testing.T) {
	b := NewBuilder()
	x := b.Input("x", BF16, 3, 256)
	w := b.Input("w", BF16, 8, 256)
	scale := b.Input("scale", F32, 8)
	resid := b.Input("resid", BF16, 3, 8)
	mm := b.MatMul(x, w)
	y := b.Map(MustLambda("(add (mul $0 $1) $2)"), mm, scale, resid)
	root := b.Map(MustLambda("(silu $0)"), y)
	expr, err := Optimize(b.G, root)
	if err != nil {
		t.Fatal(err)
	}
	if expr.Op != OpGEMM {
		t.Fatalf("optimized to %s, want one gemm", expr)
	}
	if got := expr.Children[0].Op; got != "fn:(silu (add (mul $0 $1) $2))" {
		t.Errorf("fused epilogue %s", got)
	}
	r := rand.New(rand.NewPCG(11, 12))
	env := randEnv(r,
		Tensor{"x", BF16, []int{3, 256}}, Tensor{"w", BF16, []int{8, 256}},
		Tensor{"scale", F32, []int{8}}, Tensor{"resid", BF16, []int{3, 8}})
	orig, _, err := egraph.ExtractBest(b.G, root, egraph.AstSizeCost{})
	if err != nil {
		t.Fatal(err)
	}
	// AstSizeCost may also pick the fused form; build the unfused value directly.
	_ = orig
	want, err := Eval(expr, env)
	if err != nil {
		t.Fatal(err)
	}
	mmv, err := matmul(env["x"], env["w"])
	if err != nil {
		t.Fatal(err)
	}
	unfused, err := broadcastMap(MustLambda("(silu (add (mul $0 $1) $2))"), []Value{mmv, env["scale"], env["resid"]})
	if err != nil {
		t.Fatal(err)
	}
	for i := range want.Data {
		if want.Data[i] != unfused.Data[i] {
			t.Fatalf("fused eval differs at %d: %g vs %g", i, want.Data[i], unfused.Data[i])
		}
	}
	plan, err := Compile(expr, "proj", BF16)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Kernels) != 1 || plan.Kernels[0].GEMM == nil {
		t.Fatalf("plan has %d kernels", len(plan.Kernels))
	}
	got := runPlan(t, plan, env)
	agree(t, "proj_out", BF16, got, roundAll(BF16, want.Data))
}

func roundAll(d DType, v []float64) []float64 {
	out := make([]float64, len(v))
	for i, f := range v {
		out[i] = float64(d.Round(float32(f)))
	}
	return out
}

// TestCompileHeadNorm lowers per-head q RMSNorm from the fused qkv row
// (moe8_attention_decode/reference.py:266-268, 303) to one row kernel.
func TestCompileHeadNorm(t *testing.T) {
	const tokens = 3
	b := NewBuilder()
	qkv := b.Input("qkv", BF16, tokens, 2304)
	q := b.Reshape(b.Slice(qkv, 0, 1024), tokens, 16, 64)
	ss := b.ReduceSum(b.Map(MustLambda("(mul $0 $0)"), q))
	n := b.Map(MustLambda("(mul $0 (rsqrt (add (div $1 64) 1e-5)))"), q, ss)
	gate := b.Reshape(b.Slice(qkv, 1280, 1024), tokens, 16, 64)
	root := b.Map(MustLambda("(mul $0 (sigmoid $1))"), n, gate)
	expr, err := Optimize(b.G, root)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := Compile(expr, "headnorm", BF16)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Kernels) != 1 || plan.Kernels[0].Row == nil || plan.Kernels[0].Rows != tokens*16 {
		t.Fatalf("plan: %+v", plan.Kernels)
	}
	env := randEnv(rand.New(rand.NewPCG(13, 14)), Tensor{"qkv", BF16, []int{tokens, 2304}})
	want, err := Eval(expr, env)
	if err != nil {
		t.Fatal(err)
	}
	agree(t, "headnorm", BF16, runPlan(t, plan, env), roundAll(BF16, want.Data))
}

// TestCompileGEMMThenNorm materializes a projection and normalizes its rows.
func TestCompileGEMMThenNorm(t *testing.T) {
	b := NewBuilder()
	x := b.Input("x", BF16, 2, 256)
	w := b.Input("w", BF16, 64, 256)
	pos := b.Input("pos", I64, 2)
	y := b.MatMul(x, w)
	ss := b.ReduceSum(b.Map(MustLambda("(mul $0 $0)"), y))
	n := b.Map(MustLambda("(mul $0 (rsqrt (add (div $1 64) 1e-5)))"), y, ss)
	// Interleaved RoPE with a per-row position and a per-column frequency.
	ang := b.Map(MustLambda("(mul $0 (exp (mul (sub $1 (mul 0.5 (add 1 $2))) -0.41)))"),
		b.Reshape(pos, 2, 1), b.Col(n), b.Sgn(n))
	root := b.Map(MustLambda("(add (mul $0 (cos $1)) (mul (mul $2 $3) (sin $1)))"), n, ang, b.Sgn(n), b.Pair(n))
	expr, err := Optimize(b.G, root)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := Compile(expr, "chain", F32)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Kernels) != 2 || plan.Kernels[0].GEMM == nil || plan.Kernels[1].Row == nil {
		t.Fatalf("plan: %d kernels", len(plan.Kernels))
	}
	env := randEnv(rand.New(rand.NewPCG(15, 16)), Tensor{"x", BF16, []int{2, 256}}, Tensor{"w", BF16, []int{64, 256}})
	env["pos"] = Value{Shape: []int{2}, Data: []float64{3, 1000}}
	want, err := Eval(expr, env)
	if err != nil {
		t.Fatal(err)
	}
	agree(t, "chain", F32, runPlan(t, plan, env), want.Data)
}

// TestCompileCastInputGEMM rounds an F32 input to BF16 for a projection: the
// cast of a bare input is its own row kernel.
func TestCompileCastInputGEMM(t *testing.T) {
	b := NewBuilder()
	x := b.Input("x", F32, 2, 256)
	w := b.Input("w", BF16, 64, 256)
	expr, err := Optimize(b.G, b.MatMul(b.Cast(BF16, x), w))
	if err != nil {
		t.Fatal(err)
	}
	plan, err := Compile(expr, "castproj", F32)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Kernels) != 2 || plan.Kernels[0].Row == nil || plan.Kernels[1].GEMM == nil {
		t.Fatalf("plan: %+v", plan.Kernels)
	}
	env := randEnv(rand.New(rand.NewPCG(17, 18)), Tensor{"x", F32, []int{2, 256}}, Tensor{"w", BF16, []int{64, 256}})
	want, err := Eval(expr, env)
	if err != nil {
		t.Fatal(err)
	}
	agree(t, "castproj", F32, runPlan(t, plan, env), want.Data)
}

// TestFuseEpilogueGEMMOperand folds a residual that is itself a projection's
// result into the next projection's epilogue: out = (cast(h) · w2ᵀ) + h with
// h = x · w1ᵀ is two gemms and the cast, with no row kernel for the add.
func TestFuseEpilogueGEMMOperand(t *testing.T) {
	b := NewBuilder()
	x := b.Input("x", BF16, 2, 256)
	h := b.MatMul(x, b.Input("w1", BF16, 256, 256))
	y := b.MatMul(b.Cast(BF16, h), b.Input("w2", BF16, 256, 256))
	expr, err := Optimize(b.G, b.Map(MustLambda("(add $0 $1)"), y, h))
	if err != nil {
		t.Fatal(err)
	}
	plan, err := Compile(expr, "resid", F32)
	if err != nil {
		t.Fatal(err)
	}
	var gemms, rows int
	for _, k := range plan.Kernels {
		switch {
		case k.GEMM != nil:
			gemms++
		case k.Row != nil:
			rows++
		}
	}
	last := plan.Kernels[len(plan.Kernels)-1]
	if gemms != 2 || rows != 1 || last.GEMM == nil || len(last.GEMM.Extras) != 1 {
		t.Fatalf("plan: %d gemms, %d row kernels, last %+v; want the add in the last gemm's epilogue", gemms, rows, last)
	}
	env := randEnv(rand.New(rand.NewPCG(19, 20)), Tensor{"x", BF16, []int{2, 256}}, Tensor{"w1", BF16, []int{256, 256}}, Tensor{"w2", BF16, []int{256, 256}})
	want, err := Eval(expr, env)
	if err != nil {
		t.Fatal(err)
	}
	agree(t, "resid", F32, runPlan(t, plan, env), want.Data)
}

// TestCompilePrefillGEMM lowers a many-row projection with a fused
// residual add to one tensor-core TileGEMM.
func TestCompilePrefillGEMM(t *testing.T) {
	b := NewBuilder()
	x := b.Input("x", BF16, 70, 128)
	w := b.Input("w", BF16, 64, 128)
	resid := b.Input("resid", F32, 70, 64)
	root := b.Map(MustLambda("(add $0 $1)"), b.MatMul(x, w), resid)
	expr, err := Optimize(b.G, root)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := Compile(expr, "prefill", F32)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Kernels) != 1 || plan.Kernels[0].Tile == nil {
		t.Fatalf("plan: %+v", plan.Kernels)
	}
	if _, err := plan.Source(); err != nil {
		t.Fatal(err)
	}
	env := randEnv(rand.New(rand.NewPCG(17, 18)),
		Tensor{"x", BF16, []int{70, 128}}, Tensor{"w", BF16, []int{64, 128}}, Tensor{"resid", F32, []int{70, 64}})
	want, err := Eval(expr, env)
	if err != nil {
		t.Fatal(err)
	}
	agree(t, "prefill", F32, runPlan(t, plan, env), want.Data)
}

func ExampleOptimize() {
	b := NewBuilder()
	x := b.Input("x", BF16, 8, 8192)
	w := b.Input("w", BF16, 2304, 8192)
	y := b.Map(MustLambda("(mul $0 2)"), b.MatMul(x, w))
	expr, _ := Optimize(b.G, y)
	plan, _ := Compile(expr, "qkv", BF16)
	fmt.Println(expr)
	fmt.Println(len(plan.Kernels), plan.Kernels[0].GEMM.Epilogue)
	// Output:
	// (gemm fn:(mul $0 2) t:x{{x bf16 [8 8192]}} t:w{{w bf16 [2304 8192]}})
	// 1 (mul $0 2)
}

func TestFuseEpiloguesRenumbers(t *testing.T) {
	tests := []struct {
		inner, outer string
		want         string
	}{
		{"$0", "(exp $0)", "fn:(exp $0)"},
		{"(add $0 $1)", "(mul $0 $1)", "fn:(mul (add $0 $1) $2)"},
		{"(add (mul $0 $1) $2)", "(sub $1 $0)", "fn:(sub $3 (add (mul $0 $1) $2))"},
		{"(mul $0 $1)", "(mul $0 2)", "fn:(mul (mul $0 $1) 2)"},
	}
	for _, tt := range tests {
		b := NewBuilder()
		x, w := b.Input("x", BF16, 1, 256), b.Input("w", BF16, 4, 256)
		inner := MustLambda(tt.inner)
		args := []egraph.Id{b.G.Add(egraph.ENode{Op: "fn:" + inner.String()}), x, w}
		for i := 1; i < inner.Arity(); i++ {
			args = append(args, b.Input(fmt.Sprintf("a%d", i), F32, 4))
		}
		gemm := b.G.Add(egraph.NewNode(OpGEMM, args...))
		outer := MustLambda(tt.outer)
		mapArgs := []egraph.Id{gemm}
		for i := 1; i < outer.Arity(); i++ {
			mapArgs = append(mapArgs, b.Input(fmt.Sprintf("b%d", i), F32, 4))
		}
		root := b.Map(outer, mapArgs...)
		if _, err := FuseEpilogues(b.G); err != nil {
			t.Fatal(err)
		}
		expr, _, err := egraph.ExtractBest(b.G, root, Cost{})
		if err != nil {
			t.Fatal(err)
		}
		if expr.Op != OpGEMM || expr.Children[0].Op != tt.want {
			t.Errorf("%s ∘ %s: got %s, want %s", tt.outer, tt.inner, expr.Children[0].Op, tt.want)
		}
		if want := 3 + inner.Arity() - 1 + outer.Arity() - 1; expr.Op == OpGEMM && len(expr.Children) != want {
			t.Errorf("%s ∘ %s: %d operands, want %d", tt.outer, tt.inner, len(expr.Children), want)
		}
	}
}
