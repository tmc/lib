package tcgen05

import (
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatmem"
)

// best2of4Energy returns the largest x[i]^2 + x[j]^2 over all 6 pairs i < j.
func best2of4Energy(v [4]float32) float64 {
	best := 0.0
	for i := range 4 {
		for j := i + 1; j < 4; j++ {
			best = max(best, float64(v[i])*float64(v[i])+float64(v[j])*float64(v[j]))
		}
	}
	return best
}

func randSwiGLU(r *rand.Rand, n int) (gate, up, h []float32) {
	gate = make([]float32, n)
	up = make([]float32, n)
	for i := range gate {
		gate[i] = float32(r.NormFloat64())
		up[i] = float32(r.NormFloat64())
	}
	return gate, up, SwiGLU(gate, up)
}

func TestDynamicPrune2To4Optimal(t *testing.T) {
	r := rand.New(rand.NewPCG(27, 1))
	tests := []struct {
		name string
		gen  func(i int) float32
	}{
		{"normal", func(int) float32 { return float32(r.NormFloat64()) }},
		{"ties", func(int) float32 { return float32(r.IntN(5)-2) * 0.5 }},
		{"zeros", func(int) float32 { return 0 }},
		{"heavy-tail", func(int) float32 { return float32(math.Pow(r.Float64(), 4) * float64(1-2*r.IntN(2))) }},
	}
	const rows, cols = 8, 64
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := make([]float32, rows*cols)
			for i := range x {
				x[i] = tt.gen(i)
			}
			tile, err := DynamicPrune2To4(DynSparseKVCachePages, x, rows, cols, false)
			if err != nil {
				t.Fatal(err)
			}
			dense, err := tile.Dense()
			if err != nil {
				t.Fatal(err)
			}
			var kept, total float64
			for g := range rows * cols / 4 {
				v := [4]float32(x[4*g : 4*g+4])
				nz := 0
				var groupKept float64
				for i := range 4 {
					total += float64(v[i]) * float64(v[i])
					d := dense[4*g+i]
					groupKept += float64(d) * float64(d)
					if d != 0 && d != v[i] {
						t.Fatalf("group %d slot %d = %v, want 0 or %v", g, i, d, v[i])
					}
					if d != 0 {
						nz++
					}
				}
				if nz > 2 {
					t.Fatalf("group %d keeps %d nonzeros, want <= 2", g, nz)
				}
				if best := best2of4Energy(v); groupKept != best {
					t.Fatalf("group %d %v keeps energy %v, best 2:4 pair has %v", g, v, groupKept, best)
				}
				kept += groupKept
			}
			want := 1.0
			if total > 0 {
				want = kept / total
			}
			if math.Abs(tile.RetainedL2Energy-want) > 1e-12 {
				t.Fatalf("RetainedL2Energy = %v, want %v", tile.RetainedL2Energy, want)
			}
		})
	}
}

// TestDynamicPrune2To4SwiGLUEnergy checks the retained energy on
// h = SiLU(g) * u with g, u ~ N(0, 1) and that a selector that keeps the
// first two elements of each group (the mutant) falls far below it.
func TestDynamicPrune2To4SwiGLUEnergy(t *testing.T) {
	const rows, cols = 64, 256
	_, _, h := randSwiGLU(rand.New(rand.NewPCG(27, 2)), rows*cols)
	tile, err := DynamicPrune2To4(DynSparsePostSwiGLU, h, rows, cols, false)
	if err != nil {
		t.Fatal(err)
	}
	// Top-2-of-4 keeps about 0.97 of the energy of this distribution.
	if tile.RetainedL2Energy < 0.94 {
		t.Fatalf("RetainedL2Energy = %.4f, want >= 0.94", tile.RetainedL2Energy)
	}
	var first2, total float64
	for g := range rows * cols / 4 {
		for i := range 4 {
			v := float64(h[4*g+i])
			total += v * v
			if i < 2 {
				first2 += v * v
			}
		}
	}
	if mut := first2 / total; mut > 0.6 || tile.RetainedL2Energy-mut < 0.3 {
		t.Fatalf("first-two mutant energy %.4f vs top-2 %.4f: oracle does not separate them", mut, tile.RetainedL2Energy)
	}
	t.Logf("retained energy top-2=%.4f first-2=%.4f", tile.RetainedL2Energy, first2/total)
}

func TestDynamicPrune2To4Quantized(t *testing.T) {
	const rows, cols = 4, 128
	_, _, h := randSwiGLU(rand.New(rand.NewPCG(27, 3)), rows*cols)
	exact, err := DynamicPrune2To4(DynSparsePostSwiGLU, h, rows, cols, false)
	if err != nil {
		t.Fatal(err)
	}
	q, err := DynamicPrune2To4(DynSparsePostSwiGLU, h, rows, cols, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(q.PackedE2M1) != rows*cols/4 || len(q.ScalesUE8M0) != rows*cols/64 {
		t.Fatalf("quantized lengths packed=%d scales=%d", len(q.PackedE2M1), len(q.ScalesUE8M0))
	}
	for i := range q.Meta {
		if q.Meta[i] != exact.Meta[i] {
			t.Fatalf("meta[%d] = %#x, want %#x", i, q.Meta[i], exact.Meta[i])
		}
	}
	var num, den float64
	for i := range q.Values {
		e := float64(q.Values[i] - exact.Values[i])
		num += e * e
		den += float64(exact.Values[i]) * float64(exact.Values[i])
	}
	if sqnr := 10 * math.Log10(den/num); sqnr < 10 {
		t.Fatalf("MXFP4 SQNR on kept values = %.2f dB, want >= 10", sqnr)
	}
	if _, err := DynamicPrune2To4(DynSparsePostSwiGLU, h[:4*96], 4, 96, true); err == nil {
		t.Fatal("quantize with cols=96 succeeded, want error")
	}
}

func TestDynSparseTileStorageBytes(t *testing.T) {
	const tokens, headDim = 64, 128
	x := make([]float32, tokens*headDim)
	for i := range x {
		x[i] = float32(i%7) - 3
	}
	tile, err := DynamicPrune2To4(DynSparseKVCachePages, x, tokens, headDim, false)
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		bits     int
		num, den int64
	}{
		{16, 9, 16},
		{8, 5, 8},
	}
	for _, tt := range tests {
		dense := int64(tokens * headDim * tt.bits / 8)
		got := tile.StorageBytes(tt.bits)
		if got*tt.den != dense*tt.num {
			t.Errorf("StorageBytes(%d) = %d, dense %d, want ratio %d/%d", tt.bits, got, dense, tt.num, tt.den)
		}
	}
}

func TestSparseMMAActivationDenseWeight(t *testing.T) {
	const m, k, n = 4, 64, 8
	r := rand.New(rand.NewPCG(27, 4))
	x := make([]float32, m*k)
	w := make([]float32, n*k)
	for i := range x {
		x[i] = float32(r.NormFloat64())
	}
	for i := range w {
		w[i] = float32(r.NormFloat64())
	}
	tile, err := DynamicPrune2To4(DynSparsePostSwiGLU, x, m, k, false)
	if err != nil {
		t.Fatal(err)
	}
	got, err := SparseMMAActivationDenseWeight(tile, w, n)
	if err != nil {
		t.Fatal(err)
	}
	dense, _ := tile.Dense()
	for i := range m {
		for j := range n {
			var want float64
			for kk := range k {
				want += float64(dense[i*k+kk]) * float64(w[j*k+kk])
			}
			if d := math.Abs(float64(got[i*n+j]) - want); d > 1e-4 {
				t.Fatalf("D[%d,%d] = %v, want %v", i, j, got[i*n+j], want)
			}
		}
	}
	bad := tile
	bad.Meta = append([]uint32(nil), tile.Meta...)
	bad.Meta[0] = bad.Meta[0]&^0xf | 0x5 // idx0 == idx1
	if _, err := SparseMMAActivationDenseWeight(bad, w, n); err == nil {
		t.Fatal("invalid metadata accepted")
	}
}

func TestTCGen05DynSparseSwiGLUValidate(t *testing.T) {
	tests := []struct {
		name string
		k    TCGen05DynSparseSwiGLU
		ok   bool
	}{
		{"default", TCGen05DynSparseSwiGLU{}, true},
		{"sm100a", TCGen05DynSparseSwiGLU{Target: "sm_100a"}, true},
		{"i32n32", TCGen05DynSparseSwiGLU{I: 32, N: 32}, true},
		{"sm121a", TCGen05DynSparseSwiGLU{Target: "sm_121a"}, false},
		{"m64", TCGen05DynSparseSwiGLU{M: 64}, false},
		{"n>2i", TCGen05DynSparseSwiGLU{I: 32, N: 128}, false},
		{"smem", TCGen05DynSparseSwiGLU{I: 128, N: 128}, false},
		{"name", TCGen05DynSparseSwiGLU{Name: "eg_bad"}, false},
	}
	for _, tt := range tests {
		if err := tt.k.Validate(); (err == nil) != tt.ok {
			t.Errorf("%s: Validate() = %v, want ok=%v", tt.name, err, tt.ok)
		}
	}
}

func TestTCGen05DynSparseSwiGLUSource(t *testing.T) {
	k := TCGen05DynSparseSwiGLU{Name: "dynsparse_src"}
	src, err := k.CUDA()
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"tcgen05.ld.red.spcompress.sync.aligned.32x32b.x4.max.sp::2:4.abs.f32.b2",
		"tcgen05.mma.sp.cta_group::1.kind::tf32",
		"tcgen05.cp.cta_group::1.32x128b.warpx4",
		"__CUDA_ARCH__ == 1070",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("CUDA() missing %q", want)
		}
	}
	idesc, err := k.InstDescriptor()
	if err != nil {
		t.Fatal(err)
	}
	if u, err := DecodeInstDesc(KindTF32, idesc); err != nil || !u.Sparse {
		t.Fatalf("DecodeInstDesc(tf32, %#x) = %+v, %v; want sparse", idesc, u, err)
	}
	if !strings.Contains(src, fmt.Sprintf("0x%08xu", idesc)) {
		t.Fatalf("CUDA() does not embed idesc %#x", idesc)
	}

	ev := k.TraceEvents()
	if _, err := cudatmem.VerifyTrace(cudatmem.SM107A, ev); err != nil {
		t.Fatalf("VerifyTrace: %v", err)
	}
	// Mutant: drop the tcgen05.wait after the MMA; the accumulator is
	// still busy at dealloc.
	var noWait []cudatmem.TraceEvent
	for _, e := range ev {
		if e.Kind != cudatmem.EventWait {
			noWait = append(noWait, e)
		}
	}
	if _, err := cudatmem.VerifyTrace(cudatmem.SM107A, noWait); err == nil {
		t.Fatal("VerifyTrace accepted a trace without tcgen05.wait")
	}
}

// TestTCGen05DynSparseSwiGLUEmu runs the emitted kernel under cudahostemu and
// compares it with Reference, then checks that a reference built with the
// first-two selector disagrees with the kernel.
func TestTCGen05DynSparseSwiGLUEmu(t *testing.T) {
	for _, tc := range []struct{ i, n int }{{64, 64}, {32, 32}} {
		t.Run(fmt.Sprintf("I%d_N%d", tc.i, tc.n), func(t *testing.T) {
			k := TCGen05DynSparseSwiGLU{Name: "dynsparse_emu", I: tc.i, N: tc.n}
			r := rand.New(rand.NewPCG(27, uint64(tc.i)))
			gate, up, _ := randSwiGLU(r, 128*tc.i)
			w := make([]float32, tc.n*tc.i)
			for i := range w {
				w[i] = float32(r.NormFloat64() * 0.1)
			}
			want, err := k.Reference(gate, up, w)
			if err != nil {
				t.Fatal(err)
			}
			src, err := k.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			res := runEmu(t, src,
				"eg_emu_launch(dim3(1), dim3(128), 0, dynsparse_emu, gate, up, w_down, out_h_comp, out_meta, out_row_absmax, out)",
				emuBuf{"gate", "float", f32Bytes(gate)},
				emuBuf{"up", "float", f32Bytes(up)},
				emuBuf{"w_down", "float", f32Bytes(w)},
				emuBuf{"out_h_comp", "float", poison(4 * 128 * tc.i / 2)},
				emuBuf{"out_meta", "uint32_t", poison(4 * 128 * tc.i / 32)},
				emuBuf{"out_row_absmax", "float", poison(4 * 128)},
				emuBuf{"out", "float", poison(4 * 128 * tc.n)},
			)
			gotMeta := bytesU32(res["out_meta"])
			for i, m := range want.H.Meta {
				if gotMeta[i] != m {
					t.Fatalf("out_meta[%d] = %#08x, want %#08x", i, gotMeta[i], m)
				}
			}
			closeF32(t, "out_h_comp", bytesF32(res["out_h_comp"]), want.H.Values, 1e-6)
			closeF32(t, "out_row_absmax", bytesF32(res["out_row_absmax"]), want.RowAbsMax, 1e-6)
			got := bytesF32(res["out"])
			closeF32(t, "out", got, want.Out, 1e-4)

			// Mutant: keep columns 0 and 1 of every group.
			mut := want.H
			mut.Values = make([]float32, len(want.H.Values))
			mut.Meta = make([]uint32, len(want.H.Meta))
			h := SwiGLU(gate, up)
			for g := range 128 * tc.i / 4 {
				mut.Values[2*g], mut.Values[2*g+1] = h[4*g], h[4*g+1]
				mut.Meta[g/8] |= 0x4 << (4 * (g % 8))
			}
			mutOut, err := SparseMMAActivationDenseWeight(mut, w, tc.n)
			if err != nil {
				t.Fatal(err)
			}
			var maxDiff float64
			for i := range got {
				maxDiff = max(maxDiff, math.Abs(float64(got[i]-mutOut[i])))
			}
			if maxDiff < 1e-2 {
				t.Fatalf("first-two mutant matches kernel output (max diff %g)", maxDiff)
			}
		})
	}
}

// closeF32 fails t if got and want differ by more than tol relative to
// max(1, |want|).
func closeF32(t *testing.T, name string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d, want %d", name, len(got), len(want))
	}
	for i := range want {
		if d := math.Abs(float64(got[i] - want[i])); d > tol*max(1, math.Abs(float64(want[i]))) || math.IsNaN(float64(got[i])) {
			t.Fatalf("%s[%d] = %v, want %v", name, i, got[i], want[i])
		}
	}
}

func ExampleDynamicPrune2To4() {
	x := []float32{
		0.1, -3, 2, 0.5, 1, 1, -1, 0, 4, 0, 0, -4, 0.2, 0.3, 0.1, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	}
	tile, err := DynamicPrune2To4(DynSparsePostSwiGLU, x, 1, 32, false)
	if err != nil {
		panic(err)
	}
	fmt.Println(tile.Values[:8])
	fmt.Printf("meta %#08x\n", tile.Meta[0])
	fmt.Printf("energy %.3f\n", tile.RetainedL2Energy)
	// Output:
	// [-3 2 1 1 4 -4 0.2 0.3]
	// meta 0x44444c49
	// energy 0.974
}
