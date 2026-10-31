package hopper

import (
	"bytes"
	"errors"
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
)

func vectorActivationConfig() TileGEMM {
	return TileGEMM{Name: "vector_activation", N: 128, K: 192, BM: 64, BN: 128,
		Weight: WeightNVFP4, Out: F32, WGMMA: true, Swizzle128B: true, Repacked: true,
		DequantWords: 2, WarpSpecialized: true, VectorActivationLoads: true, Stages: 2}
}

// This checks the emitted synchronization contract, not GPU memory ordering.
func checkVectorActivationContract(source string) error {
	steps := []string{
		"*reinterpret_cast<eg_u32x4*>(sa + eg_swizzle_128b(r * 128 + ch * 16)) = value;",
		"eg_fence_proxy_async();",
		"eg_emu_bar_sync(1, 128);",
		`asm volatile("barrier.cta.sync 1, 128;" ::: "memory");`,
		"if (tid == 0) eg_mbar_arrive_expect_tx(&full[s], WBYTES + SBYTES);",
		"eg_bulk_copy(sa + XBYTES, we + (size_t)kt * 256 * 4, WBYTES, &full[s]);",
	}
	position := 0
	for _, step := range steps {
		i := strings.Index(source[position:], step)
		if i < 0 {
			return fmt.Errorf("missing or reordered step %q", step)
		}
		position += i + len(step)
	}
	if strings.Contains(source, "eg_mbar_arrive_expect_tx(&full[s], TX)") || strings.Contains(source, "bar.sync 1, 128") {
		return fmt.Errorf("unexpected activation byte accounting or aligned subgroup barrier")
	}
	return nil
}

func TestVectorActivationLoadsSource(t *testing.T) {
	for _, tt := range []struct {
		name                     string
		n, k, bm, stages         int
		gather, grouped, compact bool
	}{
		{"dense_odd_k", 128, 192, 64, 2, false, false, false},
		{"grouped_gather", 256, 384, 64, 2, true, true, true},
		{"dense_wide", 256, 640, 128, 3, false, false, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			g := vectorActivationConfig()
			g.N, g.K, g.BM, g.Stages = tt.n, tt.k, tt.bm, tt.stages
			g.Gather, g.Grouped, g.CompactScales = tt.gather, tt.grouped, tt.compact
			source, err := g.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			if err := checkVectorActivationContract(source); err != nil {
				t.Fatal(err)
			}
			if tt.gather && !strings.Contains(source, "(size_t)srow[r] * K") {
				t.Fatal("missing gather indexing")
			}
			if !tt.gather && !strings.Contains(source, "(size_t)min(bm0 + r, row1 - 1) * K") {
				t.Fatal("missing dense indexing")
			}
			g.VectorActivationLoads = false
			bulk, err := g.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			if strings.Contains(bulk, "barrier.cta.sync 1, 128") || !strings.Contains(bulk, "eg_mbar_arrive_expect_tx(&full[s], TX)") {
				t.Fatal("changed default loader")
			}
			start := "    // Consumer warpgroups."
			if bulk[strings.Index(bulk, start):] != source[strings.Index(source, start):] {
				t.Fatal("vector loader changed consumer math or output")
			}
			bulkBytes := g.SharedBytes()
			g.VectorActivationLoads = true
			if g.SharedBytes() != bulkBytes {
				t.Fatal("vector loader changed ring allocation")
			}
		})
	}
}

func TestVectorActivationLoadsDefaultGolden(t *testing.T) {
	g := TileGEMM{Name: "moe8_moe_w13_ws_2w_compact", N: 8192, K: 4096, Weight: WeightNVFP4, Out: BF16, BN: 128, Stages: 4,
		Grouped: true, Gather: true, GLU: true, Epilogue: MustLambda("(mul (silu $0) $1)"), WGMMA: true, Swizzle128B: true,
		Repacked: true, DequantWords: 2, CompactScales: true, WarpSpecialized: true}
	got, err := g.Source()
	if err != nil {
		t.Fatal(err)
	}
	want, err := os.ReadFile(filepath.Join("testdata", "golden", g.Name+".cu"))
	if err != nil {
		t.Fatal(err)
	}
	if got != string(want) {
		t.Fatal("default source differs from retained golden")
	}
}

func TestVectorActivationLoadsContractMutants(t *testing.T) {
	g := vectorActivationConfig()
	source, err := g.CUDA()
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct{ old, replacement string }{
		{"eg_fence_proxy_async();", ""},
		{"barrier.cta.sync 1, 128;", "barrier.cta.sync 1, 32;"},
		{"WBYTES + SBYTES);", "TX);"},
	} {
		if err := checkVectorActivationContract(strings.Replace(source, tt.old, tt.replacement, 1)); err == nil {
			t.Errorf("accepted contract mutant %q", tt.old)
		}
	}
}

func TestVectorActivationLoadsErrors(t *testing.T) {
	for _, tt := range []struct {
		name string
		edit func(*TileGEMM)
	}{
		{"no_warp_specialization", func(g *TileGEMM) { g.WarpSpecialized = false }},
		{"bf16", func(g *TileGEMM) { g.Weight = WeightBF16; g.Repacked = false; g.DequantWords = 0 }},
		{"no_repack", func(g *TileGEMM) { g.Repacked = false }},
		{"raw_decode", func(g *TileGEMM) { g.DequantWords = 0 }},
		{"one_word", func(g *TileGEMM) { g.DequantWords = 1 }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			g := vectorActivationConfig()
			tt.edit(&g)
			if _, err := g.CUDA(); err == nil {
				t.Fatal("accepted unsupported vector activation configuration")
			}
		})
	}
}

func vectorActivationFixture(t *testing.T, g TileGEMM, m, tokens int, offsets []int32) (string, []emuBuf, []float32) {
	t.Helper()
	r := rand.New(rand.NewPCG(728, 729))
	xrows := m
	var rows []int32
	if g.Gather {
		xrows = tokens
		for range m {
			rows = append(rows, int32(r.IntN(tokens)))
		}
	}
	x := randBF16(r, xrows*g.K, 0.5)
	experts := 1
	if g.Grouped {
		experts = len(offsets) - 1
	}
	var weights []GEMMWeights
	var allWords []uint32
	var allScales []byte
	var scales []float32
	for range experts {
		w := randNVFP4(r, g.N, g.K)
		w.GlobalScal = 1.0 / 64
		w.Scales[0] = 0x03
		words, sf, err := RepackNVFP4Dequant2Word(w.Packed, w.Scales, g.N, g.K, g.GLU)
		if err != nil {
			t.Fatal(err)
		}
		if g.CompactScales {
			sf, err = CompactNVFP4Scales(sf)
			if err != nil {
				t.Fatal(err)
			}
		}
		allWords = append(allWords, words...)
		allScales = append(allScales, sf...)
		scales = append(scales, w.GlobalScal)
		weights = append(weights, w)
	}
	want, err := g.Reference(m, x, rows, offsets, weights, nil)
	if err != nil {
		t.Fatal(err)
	}
	grid, block := g.Grid(m, experts)
	args := "x, w, w_sf, w_gs"
	bufs := []emuBuf{{"x", "uint16_t", u16Bytes(x)}}
	if g.Gather {
		args = "x, rows, w, w_sf, w_gs"
		bufs = append(bufs, emuBuf{"rows", "int32_t", i32Bytes(rows)})
	}
	bufs = append(bufs, emuBuf{"w", "uint32_t", u32Bytes(allWords)}, emuBuf{"w_sf", "uint8_t", allScales}, emuBuf{"w_gs", "float", f32Bytes(scales)})
	if g.Grouped {
		args += ", offsets"
		bufs = append(bufs, emuBuf{"offsets", "int32_t", i32Bytes(offsets)})
	}
	outType, outBytes := "float", 4*len(want)
	if g.Out == BF16 {
		outType, outBytes = "uint16_t", 2*len(want)
	}
	bufs = append(bufs, emuBuf{"out", outType, poison(outBytes)})
	launch := fmt.Sprintf("eg_emu_launch(dim3(%d,%d,%d),dim3(%d),%d,%s,%d,%s,out)", grid[0], grid[1], grid[2], block[0], g.SharedBytes(), g.Name, m, args)
	return launch, bufs, want
}

func TestVectorActivationLoadsEmulated(t *testing.T) {
	for _, tt := range []struct {
		name      string
		g         TileGEMM
		m, tokens int
		offsets   []int32
	}{
		{"odd_k", vectorActivationConfig(), 17, 0, nil},
		{"grouped_gather_glu", TileGEMM{Name: "vector_glu", N: 256, K: 384, BM: 64, BN: 128, Stages: 2, Weight: WeightNVFP4, Out: BF16,
			WGMMA: true, Swizzle128B: true, Repacked: true, CompactScales: true, DequantWords: 2, WarpSpecialized: true, VectorActivationLoads: true,
			Grouped: true, Gather: true, GLU: true, Epilogue: MustLambda("(mul (silu $0) $1)")}, 70, 25, []int32{0, 19, 19, 70}},
		{"ring_wrap", TileGEMM{Name: "vector_wrap", N: 128, K: 640, BM: 128, BN: 128, Stages: 3, Weight: WeightNVFP4, Out: F32,
			WGMMA: true, Swizzle128B: true, Repacked: true, CompactScales: true, DequantWords: 2, WarpSpecialized: true, VectorActivationLoads: true}, 35, 0, nil},
	} {
		t.Run(tt.name, func(t *testing.T) {
			g := tt.g
			launch, bufs, want := vectorActivationFixture(t, g, tt.m, tt.tokens, tt.offsets)
			src, err := g.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			vector := runEmu(t, src, launch, bufs...)
			got := decode(g.Out, vector["out"])
			rel := 1e-5
			if g.Out == BF16 {
				rel = 1.0 / 128
			}
			if i, ok := closeToFloat64(got, want, rel); !ok {
				t.Fatalf("reference mismatch at %d: got %g want %g", i, got[i], want[i])
			}
			g.VectorActivationLoads = false
			src, err = g.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			bulk := runEmu(t, src, launch, bufs...)
			if !bytes.Equal(vector["out"], bulk["out"]) {
				t.Fatal("software vector and bulk loaders differ")
			}
		})
	}
}

func closeToFloat64(got []float64, want []float32, rel float64) (int, bool) {
	g := make([]float32, len(got))
	for i := range got {
		g[i] = float32(got[i])
	}
	return closeTo(g, want, rel, 1e-4)
}

func TestVectorActivationLoadsNamedBarrierNegativeControl(t *testing.T) {
	g := vectorActivationConfig()
	launch, bufs, _ := vectorActivationFixture(t, g, 17, 0, nil)
	src, err := g.CUDA()
	if err != nil {
		t.Fatal(err)
	}
	src = strings.Replace(src, "eg_emu_bar_sync(1, 128);", "eg_emu_bar_sync(1, 384);", 1)
	var buffers []cudahostemu.Buffer
	for _, b := range bufs {
		buffers = append(buffers, cudahostemu.Buffer{Name: b.name, CType: b.ctype, Data: b.data})
	}
	_, err = cudahostemu.Run(t.Context(), t.TempDir(), cudahostemu.Program{Source: Prelude + src, Launch: launch, Buffers: buffers})
	if errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip("host C++ compiler unavailable; negative control NOT_RUN")
	}
	if err == nil || (!strings.Contains(err.Error(), "deadlock") && !strings.Contains(err.Error(), "incomplete")) {
		t.Fatalf("wrong-participant control error=%v, want a barrier failure", err)
	}
}

func TestVectorActivationLoadsLauncherErrorPropagation(t *testing.T) {
	g := vectorActivationConfig()
	src, err := g.CUDA()
	if err != nil {
		t.Fatal(err)
	}
	launcher, err := g.Launcher()
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name, overrides, call, edit string
		want                        uint32
	}{
		{"attribute", "#define cudaFuncSetAttribute(...) cudaErrorInvalidValue\n", "17", "", 1},
		{"launch", "#undef EG_LAUNCH\n#define EG_LAUNCH(...) do { eg_emu_last_error = cudaErrorInvalidConfiguration; } while(0)\n", "17", "", 9},
		{"negative_rows", "", "-1", "", 1},
		{"zero_rows", "#define cudaFuncSetAttribute(...) cudaErrorInvalidValue\n", "0", "", 0},
		{"attribute_return_mutant", "#define cudaFuncSetAttribute(...) cudaErrorInvalidValue\n#undef EG_LAUNCH\n#define EG_LAUNCH(...) do { eg_emu_last_error = cudaErrorInvalidConfiguration; } while(0)\n", "17", "if (attr != cudaSuccess) return attr;", 9},
	} {
		t.Run(tt.name, func(t *testing.T) {
			l := launcher
			if tt.edit != "" {
				l = strings.Replace(l, tt.edit, "(void)attr;", 1)
			}
			res, err := cudahostemu.Run(t.Context(), t.TempDir(), cudahostemu.Program{
				Source:  Prelude + src + "\n" + tt.overrides + l,
				Launch:  fmt.Sprintf("status[0] = vector_activation_launch(%s, (const uint16_t*)nullptr, (const uint32_t*)nullptr, (const uint8_t*)nullptr, (const float*)nullptr, (float*)nullptr, nullptr)", tt.call),
				Buffers: []cudahostemu.Buffer{{Name: "status", CType: "uint32_t", Data: make([]byte, 4)}},
			})
			if errors.Is(err, cudahostemu.ErrNoCompiler) {
				t.Skip("host C++ compiler unavailable; API propagation control NOT_RUN")
			}
			if err != nil {
				t.Fatal(err)
			}
			if got := bytesU32(res["status"])[0]; got != tt.want {
				t.Fatalf("returned %d want %d", got, tt.want)
			}
		})
	}
}

func ExampleTileGEMM_vectorActivationLoads() {
	g := TileGEMM{Name: "expert", N: 256, K: 384, Weight: WeightNVFP4, Out: BF16,
		BM: 64, BN: 128, WGMMA: true, Swizzle128B: true, Repacked: true,
		DequantWords: 2, WarpSpecialized: true, VectorActivationLoads: true}
	source, err := g.CUDA()
	fmt.Println(err == nil, strings.Contains(source, "barrier.cta.sync 1, 128"))
	// Output: true true
}
