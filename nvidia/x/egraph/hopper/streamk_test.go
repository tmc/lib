package hopper

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math/rand/v2"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// streamKTests are Stream-K TileGEMMs whose total_tiles × KT iterations
// divide unevenly among the blocks, so that tiles are split two, three and
// five ways and some blocks start or end mid-tile.
var streamKTests = []struct {
	name   string
	g      TileGEMM
	m      int // rows of out
	tokens int // rows of x with Gather
}{
	// 6 tiles × 4 k-steps = 24 iterations over 5 blocks: 4.8 each.
	{"mma_bf16", TileGEMM{N: 192, K: 256, Weight: WeightBF16, Out: F32, BM: 64, BN: 64, Stages: 2, StreamK: 5}, 100, 0},
	// 9 tiles × 4 = 36 over 5, GroupM order, epilogue and row sums of squares.
	{"mma_bf16_groupm_sumsq", TileGEMM{N: 192, K: 256, Weight: WeightBF16, Out: F32, BM: 64, BN: 64, Stages: 2, GroupM: 2, StreamK: 5, SumSq: true,
		Epilogue: MustLambda("(add (mul $0 $1) $2)"), Extras: []Operand{{"scale", BF16, PerCol}, {"bias", F32, PerRow}}}, 140, 0},
	// 2 tiles × 7 = 14 over 5: each tile split three ways.
	{"mma_nvfp4_glu_gather", TileGEMM{N: 128, K: 448, Weight: WeightNVFP4, Out: BF16, BM: 64, BN: 64, Stages: 3, StreamK: 5,
		GLU: true, Gather: true, SumSq: true, Epilogue: MustLambda("(mul (silu $0) $1)")}, 50, 20},
	// 6 tiles × 3 = 18 over 7.
	{"wgmma_bf16_swizzle_sumsq", TileGEMM{N: 192, K: 192, Weight: WeightBF16, Out: F32, BN: 64, WGMMA: true, Swizzle128B: true, GroupM: 3, StreamK: 7, Stages: 2, SumSq: true,
		Epilogue: MustLambda("(add (mul $0 $1) $2)"), Extras: []Operand{{"scale", F32, PerCol}, {"resid", BF16, Full}}}, 150, 0},
	// 1 tile × 5 = 5 over 7: one tile split five ways, two idle blocks.
	{"wgmma_nvfp4_repacked_glu", TileGEMM{N: 128, K: 320, Weight: WeightNVFP4, Out: F32, BN: 128, WGMMA: true, Repacked: true, Swizzle128B: true, StreamK: 7,
		GLU: true, Gather: true, SumSq: true, Epilogue: MustLambda("(mul (silu $0) $1)")}, 90, 25},
	// 4 tiles × 3 = 12 over 5, FP8 weights.
	{"wgmma_e4m3", TileGEMM{N: 128, K: 192, Weight: WeightE4M3, Out: F32, BN: 64, WGMMA: true, Stages: 2, StreamK: 5}, 135, 0},
}

// tileInputs are the operands of one TileGEMM launch.
type tileInputs struct {
	m      int
	x      []uint16 // BF16 bits, or E4M3 bits widened
	rows   []int32
	w      GEMMWeights
	extras [][]float32
}

// newTileInputs returns random operands for g. With exact, x and W hold
// small integers (and NVFP4 its E2M1 codes and E4M3 scales), so that every
// FP32 partial sum is exact and any summation order gives the same bits.
func newTileInputs(r *rand.Rand, g TileGEMM, m, tokens int, exact bool) tileInputs {
	in := tileInputs{m: m}
	xrows := m
	if g.Gather {
		xrows = tokens
		for range m {
			in.rows = append(in.rows, int32(r.IntN(tokens)))
		}
	}
	small := func() float32 { return float32(r.IntN(9) - 4) }
	in.x = make([]uint16, xrows*g.K)
	for i := range in.x {
		switch {
		case g.Weight == WeightE4M3 && exact:
			in.x[i] = uint16(E4M3Bits(small()))
		case g.Weight == WeightE4M3:
			in.x[i] = uint16(E4M3Bits(float32(r.NormFloat64() * 0.5)))
		case exact:
			in.x[i] = BF16Bits(small())
		default:
			in.x[i] = BF16Bits(float32(r.NormFloat64()))
		}
	}
	switch g.Weight {
	case WeightNVFP4:
		in.w = randNVFP4(r, g.N, g.K)
		in.w.GlobalScal = 1.0 / 64
	case WeightE4M3:
		in.w.E4M3 = make([]uint8, g.N*g.K)
		for i := range in.w.E4M3 {
			if exact {
				in.w.E4M3[i] = E4M3Bits(small())
			} else {
				in.w.E4M3[i] = E4M3Bits(float32(r.NormFloat64() * 0.125))
			}
		}
	default:
		in.w.BF16 = make([]uint16, g.N*g.K)
		for i := range in.w.BF16 {
			if exact {
				in.w.BF16[i] = BF16Bits(small())
			} else {
				in.w.BF16[i] = BF16Bits(float32(r.NormFloat64() * 0.05))
			}
		}
	}
	nout := g.N
	if g.GLU {
		nout /= 2
	}
	for _, e := range g.Extras {
		v := make([]float32, map[Broadcast]int{PerCol: nout, PerRow: m, Full: m * nout}[e.Broadcast])
		for i := range v {
			v[i] = e.DType.Round(float32(r.NormFloat64()))
		}
		in.extras = append(in.extras, v)
	}
	return in
}

// emuTileGEMM runs g's kernel src on in, launched launches times in a row,
// with blocks numbered in reverse if reverse is set (the emulator runs
// blocks in order, so this reverses which block finishes a split tile).
// It returns the final buffers.
func emuTileGEMM(t *testing.T, g TileGEMM, src string, in tileInputs, launches int, reverse bool) map[string][]byte {
	t.Helper()
	var bufs []emuBuf
	if g.Weight == WeightE4M3 {
		xb := make([]byte, len(in.x))
		for i, v := range in.x {
			xb[i] = uint8(v)
		}
		bufs = append(bufs, emuBuf{"x", "uint8_t", xb})
	} else {
		bufs = append(bufs, emuBuf{"x", "uint16_t", u16Bytes(in.x)})
	}
	args := []string{fmt.Sprint(in.m), "x"}
	if g.Gather {
		bufs = append(bufs, emuBuf{"rows", "int32_t", i32Bytes(in.rows)})
		args = append(args, "rows")
	}
	switch {
	case g.Weight == WeightNVFP4 && g.Repacked:
		q, sf, err := RepackNVFP4WGMMA(in.w.Packed, in.w.Scales, g.N, g.K, g.GLU)
		if err != nil {
			t.Fatal(err)
		}
		bufs = append(bufs, emuBuf{"w", "uint32_t", u32Bytes(q)}, emuBuf{"w_sf", "uint8_t", sf},
			emuBuf{"w_gs", "float", f32Bytes([]float32{in.w.GlobalScal})})
		args = append(args, "w", "w_sf", "w_gs")
	case g.Weight == WeightNVFP4:
		bufs = append(bufs, emuBuf{"w", "uint8_t", in.w.Packed}, emuBuf{"w_sf", "uint8_t", in.w.Scales},
			emuBuf{"w_gs", "float", f32Bytes([]float32{in.w.GlobalScal})})
		args = append(args, "w", "w_sf", "w_gs")
	case g.Weight == WeightE4M3:
		bufs = append(bufs, emuBuf{"w", "uint8_t", in.w.E4M3})
		args = append(args, "w")
	default:
		bufs = append(bufs, emuBuf{"w", "uint16_t", u16Bytes(in.w.BF16)})
		args = append(args, "w")
	}
	for i, e := range g.Extras {
		bufs = append(bufs, emuBuf{e.Name, e.DType.cType(), typed{e.DType, f64s(in.extras[i])}.bytes()})
		args = append(args, e.Name)
	}
	nout := g.N
	if g.GLU {
		nout /= 2
	}
	bufs = append(bufs, emuBuf{"out", g.Out.cType(), poison(in.m * nout * g.Out.Bits() / 8)})
	args = append(args, "out")
	if g.SumSq {
		bufs = append(bufs, emuBuf{"ss", "float", poison(4 * in.m * nout / 32)})
		args = append(args, "ss")
	}
	if g.StreamK > 0 {
		// Partials start as NaN; only the counters must start at zero.
		ws := poison(g.WorkspaceBytes())
		clear(ws[len(ws)-4*g.StreamK:])
		bufs = append(bufs, emuBuf{"ws", "float", ws})
		args = append(args, "ws")
	}
	grid, block := g.Grid(in.m, 1)
	kernel := g.Name
	if reverse {
		kernel = fmt.Sprintf("[](auto... a) { blockIdx.x = gridDim.x - 1 - blockIdx.x; %s(a...); }", g.Name)
	}
	launch := fmt.Sprintf("eg_emu_launch(dim3(%d, %d, %d), dim3(%d), %d, %s, %s)",
		grid[0], grid[1], grid[2], block[0], g.SharedBytes(), kernel, strings.Join(args, ", "))
	return runEmu(t, src, strings.Repeat(launch+"; ", launches), bufs...)
}

// TestTileGEMMStreamK checks a Stream-K TileGEMM bit for bit: on exact
// data against the data-parallel kernel (StreamK 0) and, without an
// epilogue, against Reference; on random data against itself with the
// blocks in reverse order, which changes which block finishes each split
// tile. Each Stream-K run is two launches, to check that the kernel leaves
// the workspace counters zero for the next (as a CUDA graph replays it).
func TestTileGEMMStreamK(t *testing.T) {
	r := rand.New(rand.NewPCG(301, 302))
	for _, tt := range streamKTests {
		t.Run(tt.name, func(t *testing.T) {
			g := tt.g
			g.Name = "tile_" + tt.name
			src, err := g.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			dp := g
			dp.StreamK = 0
			dpSrc, err := dp.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			if grid, _ := g.Grid(tt.m, 1); grid != [3]int{g.StreamK, 1, 1} {
				t.Fatalf("Grid = %v, want [%d 1 1]", grid, g.StreamK)
			}

			exact := newTileInputs(r, g, tt.m, tt.tokens, true)
			sk := emuTileGEMM(t, g, src, exact, 2, false)
			if ws := sk["ws"]; !bytes.Equal(ws[len(ws)-4*g.StreamK:], make([]byte, 4*g.StreamK)) {
				t.Fatalf("workspace counters after launch = %x, want zero", ws[len(ws)-4*g.StreamK:])
			}
			want := emuTileGEMM(t, dp, dpSrc, exact, 1, false)
			for _, buf := range []string{"out", "ss"} {
				if !bytes.Equal(sk[buf], want[buf]) {
					t.Errorf("exact data: %s differs from StreamK 0 at %s", buf, firstDiff(g.Out, sk[buf], want[buf]))
				}
			}
			if g.Epilogue.IsIdentity() && !g.GLU && g.Out == F32 {
				ref, err := g.Reference(tt.m, exact.x, exact.rows, nil, []GEMMWeights{exact.w}, exact.extras)
				if err != nil {
					t.Fatal(err)
				}
				if got := sk["out"]; !bytes.Equal(got, f32Bytes(ref)) {
					t.Errorf("exact data: out differs from Reference at %s", firstDiff(F32, got, f32Bytes(ref)))
				}
			}

			random := newTileInputs(r, g, tt.m, tt.tokens, false)
			fwd := emuTileGEMM(t, g, src, random, 1, false)
			rev := emuTileGEMM(t, g, src, random, 1, true)
			for _, buf := range []string{"out", "ss"} {
				if !bytes.Equal(fwd[buf], rev[buf]) {
					t.Errorf("random data: %s depends on block order at %s", buf, firstDiff(g.Out, fwd[buf], rev[buf]))
				}
			}
			ref, err := g.Reference(tt.m, random.x, random.rows, nil, []GEMMWeights{random.w}, random.extras)
			if err != nil {
				t.Fatal(err)
			}
			got := make([]float32, len(ref))
			for i, v := range decode(g.Out, fwd["out"]) {
				got[i] = float32(v)
			}
			tol := 1e-6
			if g.Out == BF16 {
				tol = 4e-3
			}
			if e := relL2(got, ref); e > tol {
				t.Errorf("random data: relative L2 error %g against Reference, want <= %g", e, tol)
			}
		})
	}
}

// firstDiff describes the first element at which a and b differ.
func firstDiff(d DType, a, b []byte) string {
	if len(a) != len(b) {
		return fmt.Sprintf("length %d vs %d", len(a), len(b))
	}
	if d != F32 && d != BF16 {
		d = F32
	}
	va, vb := decode(d, a), decode(d, b)
	for i := range va {
		if !bytes.Equal(a[i*d.Bits()/8:(i+1)*d.Bits()/8], b[i*d.Bits()/8:(i+1)*d.Bits()/8]) {
			return fmt.Sprintf("element %d: %g vs %g", i, va[i], vb[i])
		}
	}
	return "no element"
}

// TestTileGEMMStreamKMutations checks that the tests above catch a wrong
// split-tile combine: a dropped partial, a partial read from the wrong
// workspace slot, a wrong segment length, counters left set for the next
// launch, and a combine whose summation order depends on which block
// finishes the tile.
func TestTileGEMMStreamKMutations(t *testing.T) {
	type mutant struct{ name, old, new string }
	r := rand.New(rand.NewPCG(401, 402))
	for _, wgmma := range []bool{false, true} {
		// 6 tiles × 8 k-steps over 13 blocks, or 4 × 8 over 13: tiles
		// split three ways, so that the order of a sum shows in its bits.
		g := TileGEMM{Name: "tile_sk_mut", N: 128, K: 512, Weight: WeightBF16, Out: F32, BM: 64, BN: 64, Stages: 2, StreamK: 13}
		zero := "acc[mt][nt][i] = 0.0f;"
		if wgmma {
			g.BM, g.WGMMA = 128, true
			zero = "d[i] = 0.0f;\n        for (int c"
		}
		mutants := []mutant{
			{"drop last partial", "c <= sk_ctrl[3]; ++c", "c < sk_ctrl[3]; ++c"},
			{"drop first partial", "for (int c = sk_ctrl[2];", "for (int c = sk_ctrl[2] + 1;"},
			{"wrong slot", "(c0 <= tile_idx * KT ? 1 : 0)", "(c0 <= tile_idx * KT ? 0 : 1)"},
			{"short split segment", "kt < sk_k1; ++kt", "kt < sk_k1 - is_partial; ++kt"},
			{"no counter reset", "if (tid == 0) counters[sk_ctrl[2]] = 0;", ""},
			// The finishing block keeps its own partial in registers and
			// adds the others' to it, so the order of the sum depends on
			// which block finishes.
			{"arrival-order combine", zero, strings.Replace(zero, "= 0.0f", "+= 0.0f", 1)},
		}
		src, err := g.CUDA()
		if err != nil {
			t.Fatal(err)
		}
		const m = 130
		exact := newTileInputs(r, g, m, 0, true)
		random := newTileInputs(r, g, m, 0, false)
		good := emuTileGEMM(t, g, src, exact, 2, false)
		for _, mu := range mutants {
			t.Run(fmt.Sprintf("wgmma=%t/%s", wgmma, mu.name), func(t *testing.T) {
				if !strings.Contains(src, mu.old) {
					t.Fatalf("%q not in source", mu.old)
				}
				msrc := strings.Replace(src, mu.old, mu.new, 1)
				if mu.name == "arrival-order combine" {
					msrc = strings.Replace(msrc, "if (c0 >= c1) continue;", "if (c0 >= c1 || c == (int)blockIdx.x) continue;", 1)
					fwd := emuTileGEMM(t, g, msrc, random, 1, false)
					rev := emuTileGEMM(t, g, msrc, random, 1, true)
					if bytes.Equal(fwd["out"], rev["out"]) {
						t.Errorf("mutant survived: out does not depend on block order")
					}
					return
				}
				got := emuTileGEMM(t, g, msrc, exact, 2, false)
				ws := got["ws"]
				if bytes.Equal(got["out"], good["out"]) && bytes.Equal(ws[len(ws)-4*g.StreamK:], make([]byte, 4*g.StreamK)) {
					t.Errorf("mutant survived: out and counters match the correct kernel")
				}
			})
		}
	}
}

// TestTileGEMMStreamKCompile compiles each Stream-K TileGEMM for sm_90a
// through cudacompilecheck -no-wgmma-serial, set up as for
// TestGoldensCompile, and checks that ptxas serializes no wgmma
// instructions; and that it does serialize them (C7519) in a variant
// whose wgmma issue depends on the thread, so that the gate is seen to
// fire. It skips unless CUDACOMPILECHECK is set.
func TestTileGEMMStreamKCompile(t *testing.T) {
	check := os.Getenv("CUDACOMPILECHECK")
	if check == "" {
		t.Skip("CUDACOMPILECHECK not set; Stream-K kernels not compiled")
	}
	// With -exec the sources must be visible to the compiling host, which
	// mounts the home directory but not the system temporary directory.
	dir, err := os.MkdirTemp(".", ".streamk-compile-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	args := []string{"-arch", "sm_90a", "-std", "c++20", "-flags", "-O3", "-no-wgmma-serial", "-json"}
	if x := os.Getenv("CUDACOMPILECHECK_EXEC"); x != "" {
		args = append(args, "-exec", x)
	}
	if nvcc := os.Getenv("CUDACOMPILECHECK_NVCC"); nvcc != "" {
		args = append(args, "-nvcc", nvcc)
	}
	compile := func(t *testing.T, name, src string) (codes []string, err error) {
		path, err := filepath.Abs(filepath.Join(dir, name+".cu"))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(Prelude+src), 0o644); err != nil {
			t.Fatal(err)
		}
		out, err := exec.Command(check, append(args, path)...).Output()
		var reports []struct {
			Functions []struct {
				Entry       bool
				Diagnostics []struct{ Code string }
			}
		}
		if jerr := json.Unmarshal(out, &reports); jerr != nil {
			var stderr []byte
			if ee, ok := err.(*exec.ExitError); ok {
				stderr = ee.Stderr
			}
			t.Fatalf("cudacompilecheck %s: %v %v\n%s%s", name, err, jerr, out, stderr)
		}
		entries := 0
		for _, r := range reports {
			for _, f := range r.Functions {
				if f.Entry {
					entries++
				}
				for _, d := range f.Diagnostics {
					codes = append(codes, d.Code)
				}
			}
		}
		if entries != 1 {
			t.Fatalf("%s: %d entry functions, want 1", name, entries)
		}
		return codes, err
	}
	for _, tt := range streamKTests {
		t.Run(tt.name, func(t *testing.T) {
			g := tt.g
			g.Name = "tile_" + tt.name
			src, err := g.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			if codes, err := compile(t, tt.name, src); err != nil || len(codes) > 0 {
				t.Errorf("ptxas diagnostics %q, gate error %v; want none", codes, err)
			}
			if !g.WGMMA || g.Weight != WeightBF16 {
				return
			}
			call := g.Name + "_wgmma(d, "
			if !strings.Contains(src, call) {
				t.Fatalf("%q not in source", call)
			}
			div := strings.Replace(src, call, "if (tid < m_total) "+call, 1)
			codes, err := compile(t, tt.name+"_divergent", div)
			if err == nil || !slices.Contains(codes, "C7519") {
				t.Errorf("divergent variant: ptxas diagnostics %q, gate error %v; want C7519 and a failed gate", codes, err)
			}
		})
	}
}
