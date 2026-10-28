package hopper

import (
	"crypto/sha256"
	"fmt"
	"math"
	"strings"
	"testing"
)

func TestNVFP4TMASourceOrigin(t *testing.T) {
	const want = "1e315c85905306580bdbebbd3a007490f7af892568637ccc6dfea1f52e615e50"
	if got := fmt.Sprintf("%x", sha256.Sum256([]byte(nvfp4TMADevice))); got != want {
		t.Fatalf("retained device port changed: got %s, want %s", got, want)
	}
	g := NVFP4TMAGEMM{Name: "w13_tma", N: 8192, K: 4096, FastScales: true, GLU: true, Producers: 4}
	src, err := g.Source()
	if err != nil {
		t.Fatal(err)
	}
	for _, token := range []string{
		"namespace w13_tma_detail",
		"cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes",
		"for (int t = blockIdx.x; t < total; t += gridDim.x)",
		"sc[u] = lds64(st + WBYTES + 8 * (ct / 4))",
		"if (sn == STAGES) sn = 0, pn ^= 1",
		"w13_tma_detail::gemm_kernel<128, 1, 10, true, true, 0>",
		"cuuint32_t box[2] = {64, 16}",
		"out, 4096, 64, 1, 4)",
		"inline cudaError_t w13_tma_configure()",
		"return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 216224)",
	} {
		if !strings.Contains(src, token) {
			t.Errorf("source missing %q", token)
		}
	}
	for _, forbidden := range []string{"torch::", "ATen/", "CUDAGuard", "cudaGetDriverEntryPoint", "namespace pm"} {
		if strings.Contains(src, forbidden) {
			t.Errorf("source contains unwanted runtime dependency %q", forbidden)
		}
	}
	if strings.Contains(src, "%!") {
		t.Fatal("source has an unexpanded formatting operand")
	}
}

func TestNVFP4TMAConfig(t *testing.T) {
	for _, tt := range []struct {
		bm, stages, smem, box int
	}{
		{64, 16, 214272, 16},
		{128, 10, 216224, 16},
		{256, 6, 228448, 32},
	} {
		g := NVFP4TMAGEMM{Name: "g", N: 128, K: 128, BM: tt.bm, FastScales: true}
		n, err := g.norm()
		if err != nil {
			t.Fatal(err)
		}
		if n.Stages != tt.stages || g.SharedBytes() != tt.smem {
			t.Fatalf("BM%d stages/shared = %d/%d, want %d/%d", tt.bm, n.Stages, g.SharedBytes(), tt.stages, tt.smem)
		}
		src, err := g.Launcher()
		if err != nil || !strings.Contains(src, fmt.Sprintf("box[2] = {64, %d}", tt.box)) {
			t.Fatalf("BM%d descriptor box: %v", tt.bm, err)
		}
		g.FastScales = false
		src, err = g.Launcher()
		if err != nil || !strings.Contains(src, fmt.Sprintf("box[2] = {64, %d}", tt.bm)) {
			t.Fatalf("BM%d general descriptor box: %v", tt.bm, err)
		}
	}
	base := NVFP4TMAGEMM{Name: "g", N: 128, K: 128}
	for _, tt := range []struct {
		name string
		edit func(*NVFP4TMAGEMM)
	}{
		{"name", func(g *NVFP4TMAGEMM) { g.Name = "bad name" }},
		{"N", func(g *NVFP4TMAGEMM) { g.N = 192 }},
		{"odd_k_steps", func(g *NVFP4TMAGEMM) { g.K = 192 }},
		{"BM", func(g *NVFP4TMAGEMM) { g.BM = 32 }},
		{"stages", func(g *NVFP4TMAGEMM) { g.Stages = 1 }},
		{"two_stage_cycle", func(g *NVFP4TMAGEMM) { g.Stages, g.K = 2, 256 }},
		{"two_stage_short_k", func(g *NVFP4TMAGEMM) { g.Stages, g.K = 2, 128 }},
		{"shared_limit", func(g *NVFP4TMAGEMM) { g.BM, g.Stages = 256, 7 }},
		{"GroupM", func(g *NVFP4TMAGEMM) { g.GroupM = -1 }},
		{"GroupM_overflow", func(g *NVFP4TMAGEMM) { g.N, g.GroupM = 8192, math.MaxInt32 }},
		{"Producers", func(g *NVFP4TMAGEMM) { g.Producers = 5 }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			g := base
			tt.edit(&g)
			if _, err := g.Source(); err == nil || g.SharedBytes() != 0 {
				t.Fatalf("invalid configuration accepted: %+v", g)
			}
		})
	}
}

func TestNVFP4TMAPersistentTilePermutation(t *testing.T) {
	for _, mb := range []int{0, 1, 3, 17} {
		for _, nt := range []int{1, 2, 64} {
			for _, group := range []int{1, 2, 4, 32} {
				for _, grid := range []int{1, 7, 132} {
					seen := make(map[[2]int]bool)
					for cta := 0; cta < grid; cta++ {
						for work := cta; work < mb*nt; work += grid {
							m, n := nvfp4TMATile(work, mb, nt, group)
							key := [2]int{m, n}
							if m < 0 || m >= mb || n < 0 || n >= nt || seen[key] {
								t.Fatalf("invalid/duplicate tile %v for mb%d nt%d group%d grid%d", key, mb, nt, group, grid)
							}
							seen[key] = true
						}
					}
					if len(seen) != mb*nt {
						t.Fatalf("lost tiles: got %d, want %d", len(seen), mb*nt)
					}
				}
			}
		}
	}
}

func TestNVFP4TMAProducerPartition(t *testing.T) {
	for nprod := 1; nprod <= 4; nprod++ {
		for _, kb := range []int{2, 6, 32, 64} {
			it := 0
			for tile := 0; tile < 9; tile++ {
				seen := make([]bool, kb)
				for pid := 0; pid < nprod; pid++ {
					first := (pid + nprod - it%nprod) % nprod
					for k := first; k < kb; k += nprod {
						if seen[k] || (it+k)%nprod != pid {
							t.Fatal("producer partition overlaps or changes ownership")
						}
						seen[k] = true
					}
				}
				for _, ok := range seen {
					if !ok {
						t.Fatal("producer partition dropped a k-step")
					}
				}
				it += kb
			}
		}
	}
}

func TestNVFP4TMAVariableN(t *testing.T) {
	for _, tt := range []struct{ bm, valid, want int }{
		{64, 1, 16}, {64, 17, 32}, {64, 64, 64},
		{128, 1, 16}, {128, 96, 96}, {128, 97, 128}, {128, 128, 128},
		{256, 1, 160}, {256, 160, 160}, {256, 161, 192}, {256, 193, 256},
	} {
		if got := nvfp4TMAVariableN(tt.bm, tt.valid, true); got != tt.want {
			t.Errorf("BM%d valid%d: got %d, want %d", tt.bm, tt.valid, got, tt.want)
		}
		if got := nvfp4TMAVariableN(tt.bm, tt.valid, false); got != tt.bm {
			t.Errorf("general scale path N = %d, want BM%d", got, tt.bm)
		}
	}
}

func nvfp4TMAConstantWeights(n, k int) GEMMWeights {
	w := GEMMWeights{Packed: make([]byte, n*k/2), Scales: make([]byte, n*k/16), GlobalScal: 1.0 / 128}
	for i := range w.Packed {
		w.Packed[i] = 0x22 // E2M1 code2 = 1
		if i >= len(w.Packed)/2 {
			w.Packed[i] = 0x44 // E2M1 code4 = 2
		}
	}
	for i := range w.Scales {
		w.Scales[i] = 0x38 // E4M3 1
	}
	return w
}

func TestNVFP4TMAReference(t *testing.T) {
	g := NVFP4TMAGEMM{Name: "ref", N: 128, K: 128, BM: 64, FastScales: true, GLU: true}
	w := nvfp4TMAConstantWeights(g.N, g.K)
	x := make([]uint16, 2*g.BM*g.K)
	for i := range x {
		x[i] = BF16Bits(1)
	}
	weights := []GEMMWeights{w, w}
	weights[1].GlobalScal *= 2
	out, err := g.Reference(x, []int32{0, 1}, []int32{3, 1}, weights)
	if err != nil {
		t.Fatal(err)
	}
	for tile, gate := range []float32{1, 2} {
		valid := []int{3, 1}[tile]
		want := BF16.Round(gate / (1 + float32(math.Exp(float64(-gate)))) * (2 * gate))
		for row := 0; row < g.BM; row++ {
			for col := 0; col < g.N/2; col++ {
				got := out[(tile*g.BM+row)*(g.N/2)+col]
				if row < valid && got != want {
					t.Fatalf("tile%d row%d col%d = %g, want %g", tile, row, col, got, want)
				}
				if row >= valid && !math.IsNaN(float64(got)) {
					t.Fatal("padding was presented as a semantic output")
				}
			}
		}
	}
	for _, tt := range []struct {
		name        string
		x           []uint16
		eids, valid []int32
	}{
		{"missing_tile", x, []int32{0}, []int32{1}},
		{"bad_expert", x[:g.BM*g.K], []int32{2}, []int32{1}},
		{"zero_valid", x[:g.BM*g.K], []int32{0}, []int32{0}},
		{"too_many_valid", x[:g.BM*g.K], []int32{0}, []int32{65}},
	} {
		if _, err := g.Reference(tt.x, tt.eids, tt.valid, weights); err == nil {
			t.Errorf("accepted %s", tt.name)
		}
	}
	for _, bad := range []GEMMWeights{
		{Packed: w.Packed[:len(w.Packed)-1], Scales: w.Scales, GlobalScal: w.GlobalScal},
		{Packed: w.Packed, Scales: w.Scales[:len(w.Scales)-1], GlobalScal: w.GlobalScal},
		{Packed: w.Packed, Scales: w.Scales, GlobalScal: math.MaxFloat32},
	} {
		if _, err := g.Reference(x[:g.BM*g.K], []int32{0}, []int32{1}, []GEMMWeights{bad}); err == nil {
			t.Fatal("accepted truncated weights or overflowing global scale")
		}
	}
	weights[0].Scales[0] = 3 // subnormal needs the general scale path
	if _, err := g.Reference(x, []int32{0, 1}, []int32{3, 1}, weights); err == nil {
		t.Fatal("FastScales accepted a subnormal block scale")
	}
	g.FastScales = false
	if _, err := g.Reference(x, []int32{0, 1}, []int32{3, 1}, weights); err != nil {
		t.Fatalf("general scale reference rejected a subnormal: %v", err)
	}
}

func ExampleNVFP4TMAGEMM() {
	g := NVFP4TMAGEMM{Name: "w13", N: 8192, K: 4096, FastScales: true, GLU: true, Producers: 4}
	source, err := g.Source()
	fmt.Println(len(source) > 0, err, g.SharedBytes())
	// Output: true <nil> 216224
}

func ExampleNVFP4TMAGEMM_CUDA() {
	g := NVFP4TMAGEMM{Name: "w13", N: 8192, K: 4096}
	source, err := g.CUDA()
	fmt.Println(strings.Contains(source, "namespace w13_detail"), err)
	// Output: true <nil>
}

func ExampleNVFP4TMAGEMM_Launcher() {
	g := NVFP4TMAGEMM{Name: "w13", N: 8192, K: 4096}
	source, err := g.Launcher()
	fmt.Println(strings.Contains(source, "w13_make_map"), strings.Contains(source, "w13_launch"), err)
	// Output: true true <nil>
}

func ExampleNVFP4TMAGEMM_Reference() {
	g := NVFP4TMAGEMM{Name: "empty", N: 128, K: 128}
	y, err := g.Reference(nil, nil, nil, nil)
	fmt.Println(len(y), err)
	// Output: 0 <nil>
}

func ExampleNVFP4TMAGEMM_SharedBytes() {
	g := NVFP4TMAGEMM{Name: "w13", N: 8192, K: 4096, BM: 256}
	fmt.Println(g.SharedBytes())
	// Output: 228448
}

func ExampleNVFP4TMAGEMM_Source() {
	g := NVFP4TMAGEMM{Name: "w13", N: 8192, K: 4096}
	source, err := g.Source()
	fmt.Println(strings.Contains(source, "__expf"), err)
	// Output: true <nil>
}
