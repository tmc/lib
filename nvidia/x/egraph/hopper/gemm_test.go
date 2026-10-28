package hopper

import (
	"fmt"
	"math/rand/v2"
	"strings"
	"testing"
)

func randBF16(r *rand.Rand, n int, scale float64) []uint16 {
	v := make([]uint16, n)
	for i := range v {
		v[i] = BF16Bits(float32(r.NormFloat64() * scale))
	}
	return v
}

func randNVFP4(r *rand.Rand, n, k int) GEMMWeights {
	w := GEMMWeights{Packed: make([]uint8, n*k/2), Scales: make([]uint8, n*k/16), GlobalScal: 0.0123}
	for i := range w.Packed {
		w.Packed[i] = uint8(r.IntN(256))
	}
	for i := range w.Scales {
		w.Scales[i] = E4M3Bits(float32(8 + r.Float64()*400))
	}
	return w
}

func TestSkinnyGEMMEmulated(t *testing.T) {
	tests := []struct {
		name string
		g    SkinnyGEMM
		m    int
	}{
		{"bf16", SkinnyGEMM{N: 8, K: 512, MaxM: 4, Weight: WeightBF16, Out: BF16}, 3},
		{"bf16_r2_u1", SkinnyGEMM{N: 8, K: 512, MaxM: 2, Weight: WeightBF16, Out: F32, RowsPerBlock: 2, Warps: 2, Unroll: 1}, 2},
		{"bf16_split", SkinnyGEMM{N: 4, K: 1024, MaxM: 8, Weight: WeightBF16, Out: BF16, SplitK: 2, RowsPerBlock: 2}, 8},
		{"hilo", SkinnyGEMM{N: 4, K: 256, MaxM: 2, Weight: WeightBF16HiLo, Out: F32}, 2},
		{"nvfp4", SkinnyGEMM{N: 4, K: 2048, MaxM: 2, Weight: WeightNVFP4, Out: F32, RowsPerBlock: 2}, 2},
		{"nvfp4_split", SkinnyGEMM{N: 2, K: 2048, MaxM: 1, Weight: WeightNVFP4, Out: BF16, SplitK: 2, Warps: 1}, 1},
		{"epilogue", SkinnyGEMM{N: 8, K: 256, MaxM: 2, Weight: WeightBF16, Out: BF16,
			Epilogue: MustLambda("(add (mul $0 $1) (silu $2))"),
			Extras:   []Operand{{"scale", BF16, PerCol}, {"resid", F32, Full}}}, 2},
		{"bf16_prefetch2", SkinnyGEMM{N: 8, K: 1024, MaxM: 4, Weight: WeightBF16, Out: BF16, RowsPerBlock: 2, Warps: 1, Prefetch: 2}, 3},
		{"bf16_prefetch4_full", SkinnyGEMM{N: 8, K: 1024, MaxM: 2, Weight: WeightBF16, Out: F32, RowsPerBlock: 2, Warps: 1, Prefetch: 4}, 2},
		{"hilo_prefetch2", SkinnyGEMM{N: 4, K: 1024, MaxM: 2, Weight: WeightBF16HiLo, Out: F32, Warps: 1, Prefetch: 2}, 2},
		{"nvfp4_prefetch2", SkinnyGEMM{N: 4, K: 4096, MaxM: 2, Weight: WeightNVFP4, Out: F32, RowsPerBlock: 2, Warps: 1, Prefetch: 2}, 2},
		{"nvfp4_blackwell", SkinnyGEMM{N: 4, K: 2048, MaxM: 2, Weight: WeightNVFP4, Out: F32, RowsPerBlock: 2, BlackwellFP4: true, Prefetch: 2}, 2},
		{"bf16_split_prefetch4", SkinnyGEMM{N: 4, K: 2048, MaxM: 4, Weight: WeightBF16, Out: BF16, SplitK: 2, RowsPerBlock: 2, Warps: 1, Prefetch: 4}, 4},
	}
	r := rand.New(rand.NewPCG(1, 2))
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := tt.g
			g.Name = "gemm_" + tt.name
			src, err := g.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			n, k, m := g.N, g.K, tt.m
			x := randBF16(r, m*k, 1)
			var w GEMMWeights
			bufs := []emuBuf{{"x", "uint16_t", u16Bytes(x)}}
			args := []string{"x"}
			switch g.Weight {
			case WeightBF16:
				w.BF16 = randBF16(r, n*k, 0.05)
				bufs = append(bufs, emuBuf{"w", "uint16_t", u16Bytes(w.BF16)})
				args = append(args, "w")
			case WeightBF16HiLo:
				w.BF16, w.Lo = make([]uint16, n*k), make([]uint16, n*k)
				for i := range w.BF16 {
					w.BF16[i], w.Lo[i] = SplitBF16(float32(r.NormFloat64() * 0.05))
				}
				bufs = append(bufs, emuBuf{"w_hi", "uint16_t", u16Bytes(w.BF16)}, emuBuf{"w_lo", "uint16_t", u16Bytes(w.Lo)})
				args = append(args, "w_hi", "w_lo")
			case WeightNVFP4:
				w = randNVFP4(r, n, k)
				bufs = append(bufs, emuBuf{"w", "uint8_t", w.Packed}, emuBuf{"w_sf", "uint8_t", w.Scales},
					emuBuf{"w_gs", "float", f32Bytes([]float32{w.GlobalScal})})
				args = append(args, "w", "w_sf", "w_gs")
			}
			var extras [][]float32
			for _, e := range g.Extras {
				size := n
				if e.Broadcast == Full {
					size = m * n
				}
				v := make([]float32, size)
				for i := range v {
					v[i] = e.DType.Round(float32(r.NormFloat64()))
				}
				extras = append(extras, v)
				if e.DType == BF16 {
					b := make([]uint16, size)
					for i, f := range v {
						b[i] = BF16Bits(f)
					}
					bufs = append(bufs, emuBuf{e.Name, "uint16_t", u16Bytes(b)})
				} else {
					bufs = append(bufs, emuBuf{e.Name, "float", f32Bytes(v)})
				}
				args = append(args, e.Name)
			}
			outBytes := m * n * g.Out.Bits() / 8
			bufs = append(bufs, emuBuf{"out", g.Out.cType(), poison(outBytes)})
			args = append(args, "out")
			if ws := g.WorkspaceBytes(); ws > 0 {
				bufs = append(bufs, emuBuf{"ws", "float", make([]byte, ws)})
				args = append(args, "ws")
			}
			grid, block, err := g.Grid()
			if err != nil {
				t.Fatal(err)
			}
			launch := fmt.Sprintf("eg_emu_launch(dim3(%d, %d), dim3(%d), 0, %s<%d>, %s)", grid[0], grid[1], block[0], g.Name, m, strings.Join(args, ", "))
			if g.SplitK > 1 { // a second launch checks the counters were reset
				launch += "; " + launch
			}
			res := runEmu(t, src, launch, bufs...)
			want, err := g.Reference(m, x, w, extras)
			if err != nil {
				t.Fatal(err)
			}
			var got []float32
			if g.Out == BF16 {
				for _, b := range bytesU16(res["out"]) {
					got = append(got, FromBF16Bits(b))
				}
			} else {
				got = bytesF32(res["out"])
			}
			rel := 1e-5
			if g.Out == BF16 {
				rel = 1.0 / 128
			}
			if i, ok := closeTo(got, want, rel, 1e-6); !ok {
				t.Fatalf("out[%d] = %g, want %g (relL2 %g)", i, got[i], want[i], relL2(got, want))
			}
			if g.SplitK > 1 {
				counters := res["ws"][4*g.SplitK*g.MaxM*g.N:]
				for i, c := range counters {
					if c != 0 {
						t.Fatalf("counter byte %d = %d after launch, want 0", i, c)
					}
				}
			}
		})
	}
}

func TestSkinnyGEMMErrors(t *testing.T) {
	ok := SkinnyGEMM{Name: "g", N: 8, K: 256, MaxM: 1, Weight: WeightBF16, Out: BF16}
	for name, mutate := range map[string]func(*SkinnyGEMM){
		"name":     func(g *SkinnyGEMM) { g.Name = "eg_x" },
		"out":      func(g *SkinnyGEMM) { g.Out = E4M3 },
		"maxm":     func(g *SkinnyGEMM) { g.MaxM = 17 },
		"ragged_k": func(g *SkinnyGEMM) { g.K = 300 },
		"ragged_n": func(g *SkinnyGEMM) { g.RowsPerBlock = 3 },
		"epilogue": func(g *SkinnyGEMM) { g.Epilogue = MustLambda("(add $0 $1)") },
		"split":    func(g *SkinnyGEMM) { g.SplitK = 2 },
		"prefetch": func(g *SkinnyGEMM) { g.Prefetch = -1 },
		"bw_fp4":   func(g *SkinnyGEMM) { g.BlackwellFP4 = true },
		"red_bf16": func(g *SkinnyGEMM) { g.Store = StoreReduceAdd },
		"store":    func(g *SkinnyGEMM) { g.Store = 9 },
	} {
		g := ok
		mutate(&g)
		if _, err := g.CUDA(); err == nil {
			t.Errorf("%s: CUDA succeeded", name)
		}
	}
	if _, err := ok.CUDA(); err != nil {
		t.Fatal(err)
	}
}

// TestSkinnyGEMMReduceAddEmulated runs a reducing-store gemm once per
// rank, each rank with its own K slice of x and W, into one output that
// starts at the residual: the output must be the residual plus the full
// product, the row-parallel projection with its all-reduce fused into the
// epilogue.
func TestSkinnyGEMMReduceAddEmulated(t *testing.T) {
	for _, store := range []GEMMStore{StoreReduceAdd, StoreMultimemReduceAdd} {
		t.Run(store.String(), func(t *testing.T) {
			const ranks, m, n, k = 2, 2, 8, 512
			g := SkinnyGEMM{Name: "gemm_red", N: n, K: k / ranks, MaxM: m, Weight: WeightBF16, Out: F32, Store: store}
			src, err := g.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			r := rand.New(rand.NewPCG(3, 4))
			x := randBF16(r, m*k, 1)
			w := randBF16(r, n*k, 0.05)
			resid := make([]float32, m*n)
			for i := range resid {
				resid[i] = float32(r.NormFloat64())
			}
			full := SkinnyGEMM{Name: "gemm_full", N: n, K: k, MaxM: m, Weight: WeightBF16, Out: F32}
			want, err := full.Reference(m, x, GEMMWeights{BF16: w}, nil)
			if err != nil {
				t.Fatal(err)
			}
			for i := range want {
				want[i] += resid[i]
			}
			bufs := []emuBuf{{"out", "float", f32Bytes(resid)}}
			grid, block, _ := g.Grid()
			var launches []string
			for rk := range ranks {
				xs, ws := make([]uint16, 0, m*k/ranks), make([]uint16, 0, n*k/ranks)
				for i := range m {
					xs = append(xs, x[i*k+rk*k/ranks:i*k+(rk+1)*k/ranks]...)
				}
				for i := range n {
					ws = append(ws, w[i*k+rk*k/ranks:i*k+(rk+1)*k/ranks]...)
				}
				xn, wn := fmt.Sprintf("x%d", rk), fmt.Sprintf("w%d", rk)
				bufs = append(bufs, emuBuf{xn, "uint16_t", u16Bytes(xs)}, emuBuf{wn, "uint16_t", u16Bytes(ws)})
				launches = append(launches, fmt.Sprintf("eg_emu_launch(dim3(%d, %d), dim3(%d), 0, gemm_red<%d>, %s, %s, out)", grid[0], grid[1], block[0], m, xn, wn))
			}
			res := runEmu(t, src, strings.Join(launches, "; "), bufs...)
			got := bytesF32(res["out"])
			if i, ok := closeTo(got, want, 1e-5, 1e-6); !ok {
				t.Fatalf("out[%d] = %g, want %g", i, got[i], want[i])
			}
		})
	}
}

// TestSkinnyGEMMMultimemSource checks the multimem store's PTX and its
// guard against targets without multimem.
func TestSkinnyGEMMMultimemSource(t *testing.T) {
	g := SkinnyGEMM{Name: "g", N: 8, K: 256, MaxM: 1, Weight: WeightBF16, Out: F32, Store: StoreMultimemReduceAdd}
	src, err := g.CUDA()
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"multimem.red.relaxed.sys.global.add.f32 [%0], %1;", "__CUDA_ARCH__ < 900", "#error"} {
		if !strings.Contains(src, want) {
			t.Errorf("source lacks %q", want)
		}
	}
	g.Store = StoreWrite
	src, _ = g.CUDA()
	if strings.Contains(src, "multimem") || strings.Contains(src, "atomicAdd") {
		t.Error("write store emits a reduction")
	}
}

// TestSkinnyGEMMGolden pins the Moe8 projection kernels' source.
func TestSkinnyGEMMGolden(t *testing.T) {
	for _, g := range []SkinnyGEMM{
		{Name: "moe8_qkv_gate", N: 2304, K: 8192, MaxM: 8, Weight: WeightBF16, Out: BF16},
		{Name: "moe8_router_hilo", N: 256, K: 8192, MaxM: 8, Weight: WeightBF16HiLo, Out: F32, SplitK: 4},
		{Name: "moe8_expert_down_nvfp4", N: 4096, K: 4096, MaxM: 4, Weight: WeightNVFP4, Out: F32},
	} {
		src, err := g.Source()
		if err != nil {
			t.Fatal(err)
		}
		checkGolden(t, g.Name+".cu", src)
	}
}

func ExampleSkinnyGEMM() {
	// moe8_bf16_proj's "o" layer: [M, 1024] x [8192, 1024]^T, BF16 out.
	g := SkinnyGEMM{Name: "o_proj", N: 8192, K: 1024, MaxM: 8, Weight: WeightBF16, Out: BF16}
	grid, block, _ := g.Grid()
	src, _ := g.Source()
	fmt.Println(grid, block, strings.Contains(src, "o_proj_launch(int m"))
	// Output:
	// [2048 1 1] [256 1 1] true
}

// TestSkinnyGEMMPrefetchMutations verifies that corrupting the prefetched
// weight buffer index or dropping the last prefetched step fails the emulator.
func TestSkinnyGEMMPrefetchMutations(t *testing.T) {
	g := SkinnyGEMM{Name: "gemm_pf", N: 8, K: 1024, MaxM: 2, Weight: WeightBF16, Out: F32, RowsPerBlock: 2, Warps: 1, Prefetch: 4}
	src, err := g.CUDA()
	if err != nil {
		t.Fatal(err)
	}
	r := rand.New(rand.NewPCG(7, 8))
	const m = 2
	x := randBF16(r, m*g.K, 1)
	w := GEMMWeights{BF16: randBF16(r, g.N*g.K, 0.05)}
	want, err := g.Reference(m, x, w, nil)
	if err != nil {
		t.Fatal(err)
	}
	grid, block, err := g.Grid()
	if err != nil {
		t.Fatal(err)
	}
	launch := fmt.Sprintf("eg_emu_launch(dim3(%d, %d), dim3(%d), 0, %s<%d>, x, w, out)", grid[0], grid[1], block[0], g.Name, m)
	mutants := []struct {
		name string
		old  string
		repl string
	}{
		{
			name: "off_by_one_step",
			old:  "const uint32_t ww[4] = {wv[u][r][0].x, wv[u][r][0].y, wv[u][r][0].z, wv[u][r][0].w};",
			repl: "const uint32_t ww[4] = {wv[(u + 1) % U][r][0].x, wv[(u + 1) % U][r][0].y, wv[(u + 1) % U][r][0].z, wv[(u + 1) % U][r][0].w};",
		},
		{
			name: "drop_tail_prefetch",
			old:  "for (int u = 0; u < U; ++u) {\n            const int s = base + u * WARPS;",
			repl: "for (int u = 0; u < U; ++u) {\n            const int s = u + 1 < U ? base + u * WARPS : last;",
		},
	}
	for _, mut := range mutants {
		if !strings.Contains(src, mut.old) {
			t.Fatalf("%s: target not in source", mut.name)
		}
		msrc := strings.Replace(src, mut.old, mut.repl, 1)
		res := runEmu(t, msrc, launch,
			emuBuf{"x", "uint16_t", u16Bytes(x)},
			emuBuf{"w", "uint16_t", u16Bytes(w.BF16)},
			emuBuf{"out", "float", poison(4 * m * g.N)})
		got := bytesF32(res["out"])
		if _, ok := closeTo(got, want, 1e-5, 1e-6); ok {
			t.Errorf("mutant %s survived", mut.name)
		}
	}
}
