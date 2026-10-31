package tcgen05

import (
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatmem"
)

// recallData returns W_0 and a key-value recall sequence: the same L
// key-value pairs in every chunk, values LN(A k) for a fixed random A.
func recallData(r *rand.Rand, s TTTSchedule) (w0, q, k, v [][]float32) {
	d, l := s.HeadDim, s.ChunkLen
	w0 = randMat(r, d, d, 0.5/math.Sqrt(float64(d)))
	a := randMat(r, d, d, 1/math.Sqrt(float64(d)))
	keys := randMat(r, l, d, 1/math.Sqrt(float64(d)))
	vals := make([][]float32, l)
	for i, key := range keys {
		z := make([]float64, d)
		for j := range z {
			for p := range key {
				z[j] += float64(a[j][p]) * float64(key[p])
			}
		}
		var mu, vr float64
		for _, x := range z {
			mu += x
		}
		mu /= float64(d)
		for _, x := range z {
			vr += (x - mu) * (x - mu)
		}
		sig := math.Sqrt(vr / float64(d))
		vals[i] = make([]float32, d)
		for j, x := range z {
			vals[i][j] = float32((x - mu) / sig)
		}
	}
	for range s.NumChunks {
		k = append(k, keys...)
		v = append(v, vals...)
		q = append(q, keys...)
	}
	return w0, q, k, v
}

func TestTTTValidate(t *testing.T) {
	ok := TTTSchedule{HeadDim: 128, ChunkLen: 64, NumChunks: 64, LR: 0.01}
	if err := ok.Validate(); err != nil {
		t.Fatalf("Validate(%+v) = %v", ok, err)
	}
	for _, tt := range []struct {
		name string
		edit func(*TTTSchedule)
	}{
		{"head dim", func(s *TTTSchedule) { s.HeadDim = 40 }},
		{"chunk len", func(s *TTTSchedule) { s.ChunkLen = 128 }},
		{"no chunks", func(s *TTTSchedule) { s.NumChunks = 0 }},
		{"lr", func(s *TTTSchedule) { s.LR = 0 }},
		{"decay", func(s *TTTSchedule) { s.WeightDecay = 1 }},
		{"momentum", func(s *TTTSchedule) { s.Momentum = -0.1 }},
		{"target", func(s *TTTSchedule) { s.Target = "sm_90a" }},
		{"variant", func(s *TTTSchedule) { s.Variant = 9 }},
	} {
		s := ok
		tt.edit(&s)
		if err := s.Validate(); err == nil {
			t.Errorf("%s: Validate(%+v) = nil, want error", tt.name, s)
		}
	}
	titans := ok
	titans.Variant = TTTTitansMomentum
	if c := titans.TMEMColumns(); c != 512 {
		t.Errorf("Titans d=128 L=64 TMEMColumns = %d, want 512", c)
	}
}

// TestLayerNormL2VJP compares the analytic VJP with central differences,
// and checks that the same comparison rejects the VJP without its
// variance-Jacobian term ẑ mean(r ⊙ ẑ).
func TestLayerNormL2VJP(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 5))
	const d, eps, h = 64, 1e-5, 1e-6
	z, v := make([]float64, d), make([]float64, d)
	for i := range z {
		z[i] = r.NormFloat64()*2 + 0.5
		v[i] = r.NormFloat64()
	}
	_, g := LayerNormL2VJP(z, v, eps)
	fd := make([]float64, d)
	for i := range z {
		zp := append([]float64(nil), z...)
		zm := append([]float64(nil), z...)
		zp[i] += h
		zm[i] -= h
		lp, _ := LayerNormL2VJP(zp, v, eps)
		lm, _ := LayerNormL2VJP(zm, v, eps)
		fd[i] = (lp - lm) / (2 * h)
	}
	relErr := func(a, b []float64) float64 {
		var num, den float64
		for i := range a {
			num += (a[i] - b[i]) * (a[i] - b[i])
			den += b[i] * b[i]
		}
		return math.Sqrt(num / den)
	}
	if e := relErr(g, fd); e > 1e-6 {
		t.Fatalf("VJP vs central differences: rel err %g", e)
	}

	// The VJP without the variance term must fail the same check.
	var mu, vr float64
	for _, x := range z {
		mu += x
	}
	mu /= d
	for _, x := range z {
		vr += (x - mu) * (x - mu)
	}
	sigma := math.Sqrt(vr/d + eps)
	var mrz float64
	for i, x := range z {
		zh := (x - mu) / sigma
		mrz += (zh - v[i]) * zh / d
	}
	bad := make([]float64, d)
	for i, x := range z {
		bad[i] = g[i] + (x-mu)/sigma*mrz/sigma
	}
	if e := relErr(bad, fd); e < 1e-3 {
		t.Fatalf("mutant VJP passed: rel err %g", e)
	}
}

// TestTTTLossDecreases checks on an associative-recall sequence of 8
// chunks of 64 tokens that the fast weights learn the key-value map.
func TestTTTLossDecreases(t *testing.T) {
	for _, v := range []TTTVariant{TTTLinearSGD, TTTTitansMomentum} {
		s := TTTSchedule{Variant: v, HeadDim: 64, ChunkLen: 64, NumChunks: 8, LR: 0.003, Momentum: 0.5}
		w0, q, k, vv := recallData(rand.New(rand.NewPCG(1, 1)), s)
		_, _, loss, err := s.Reference(w0, q, k, vv)
		if err != nil {
			t.Fatal(err)
		}
		for c := 1; c < len(loss); c++ {
			if !(loss[c] < loss[c-1]) {
				t.Errorf("%v: loss rose at chunk %d: %v", v, c, loss)
			}
		}
		if !(loss[7] < 0.25*loss[0]) {
			t.Errorf("%v: loss[7] = %g, want < 0.25 * %g", v, loss[7], loss[0])
		}
	}
}

func TestTTTTraceVerifies(t *testing.T) {
	for _, v := range []TTTVariant{TTTLinearSGD, TTTTitansMomentum} {
		s := TTTSchedule{Variant: v, HeadDim: 128, ChunkLen: 64, NumChunks: 4, LR: 0.01}
		for _, target := range []cudatmem.Target{cudatmem.SM100A, cudatmem.SM103A, cudatmem.SM107A} {
			if _, err := cudatmem.VerifyTrace(target, s.TraceEvents()); err != nil {
				t.Fatalf("%v %s: VerifyTrace: %v", v, target, err)
			}
		}
	}
}

func runTTTKernel(t *testing.T, s TTTSchedule, src string, w0s, qs, ks, vs [][][]float32) (out, w, loss []float32) {
	t.Helper()
	heads := len(w0s)
	var w0, q, k, v []float32
	for h := range heads {
		w0 = append(w0, flat(w0s[h])...)
		q = append(q, flat(qs[h])...)
		k = append(k, flat(ks[h])...)
		v = append(v, flat(vs[h])...)
	}
	grid, block := s.Grid(heads, 0)
	launch := fmt.Sprintf("eg_emu_launch(dim3(%d, 1, 1), dim3(%d, 1, 1), %d, %s, %d, w0, q, k, v, out, w, loss)",
		grid[0], block[0], s.SharedBytes(), s.Name, heads)
	res := runEmu(t, src, launch,
		emuBuf{"w0", "float", f32Bytes(w0)},
		emuBuf{"q", "float", f32Bytes(q)},
		emuBuf{"k", "float", f32Bytes(k)},
		emuBuf{"v", "float", f32Bytes(v)},
		emuBuf{"out", "float", poison(4 * len(q))},
		emuBuf{"w", "float", poison(4 * len(w0))},
		emuBuf{"loss", "float", poison(4 * heads * s.SeqLen())},
	)
	return bytesF32(res["out"]), bytesF32(res["w"]), bytesF32(res["loss"])
}

type tttCase struct {
	w0s, qs, ks, vs [][][]float32
	out, w, loss    []float32
}

func newTTTCase(t *testing.T, s TTTSchedule, heads int, seed uint64) tttCase {
	t.Helper()
	var c tttCase
	r := rand.New(rand.NewPCG(seed, seed))
	for range heads {
		w0, q, k, v := recallData(r, s)
		q = randMat(r, len(q), s.HeadDim, 1/math.Sqrt(float64(s.HeadDim)))
		out, w, cl, err := s.Reference(w0, q, k, v)
		if err != nil {
			t.Fatal(err)
		}
		c.w0s, c.qs, c.ks, c.vs = append(c.w0s, w0), append(c.qs, q), append(c.ks, k), append(c.vs, v)
		c.out = append(c.out, flat(out)...)
		c.w = append(c.w, flat(w)...)
		// Per-token losses: the chunk mean must match.
		for _, l := range cl {
			c.loss = append(c.loss, float32(l))
		}
	}
	return c
}

func chunkMeans(loss []float32, l int) []float32 {
	var m []float32
	for i := 0; i < len(loss); i += l {
		var s float64
		for _, x := range loss[i : i+l] {
			s += float64(x)
		}
		m = append(m, float32(s/float64(l)))
	}
	return m
}

// TestTTTEmulated runs the emitted kernel for both variants on the host
// emulator and compares outputs, final weights and chunk losses with the
// float64 reference.
func TestTTTEmulated(t *testing.T) {
	if testing.Short() {
		t.Skip("emulated kernel run")
	}
	for _, v := range []TTTVariant{TTTLinearSGD, TTTTitansMomentum} {
		t.Run(v.String(), func(t *testing.T) {
			s := TTTSchedule{Name: "ttt_emu_" + v.String(), Variant: v, HeadDim: 64, ChunkLen: 64, NumChunks: 3, LR: 0.003, WeightDecay: 0.01, Momentum: 0.5}
			c := newTTTCase(t, s, 2, uint64(10+v))
			src, err := s.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			out, w, loss := runTTTKernel(t, s, src, c.w0s, c.qs, c.ks, c.vs)
			t.Logf("rel L2: out %.3g, w %.3g, loss %.3g", relL2(out, c.out), relL2(w, c.w), relL2(chunkMeans(loss, s.ChunkLen), c.loss))
			if e := relL2(out, c.out); !(e <= 1e-4) {
				t.Errorf("out rel L2 = %g", e)
			}
			if e := relL2(w, c.w); !(e <= 1e-4) {
				t.Errorf("w rel L2 = %g", e)
			}
			if e := relL2(chunkMeans(loss, s.ChunkLen), c.loss); !(e <= 1e-4) {
				t.Errorf("chunk loss rel L2 = %g", e)
			}
		})
	}
}

// TestTTTEmulatedMutants checks that the emulated oracle rejects a kernel
// whose VJP omits the variance-Jacobian term, and one that overwrites W
// with the gradient step instead of accumulating onto it.
func TestTTTEmulatedMutants(t *testing.T) {
	if testing.Short() {
		t.Skip("emulated kernel run")
	}
	s := TTTSchedule{Name: "ttt_mutant", HeadDim: 32, ChunkLen: 32, NumChunks: 3, LR: 0.003}
	c := newTTTCase(t, s, 1, 3)
	src, err := s.CUDA()
	if err != nil {
		t.Fatal(err)
	}
	out, _, _ := runTTTKernel(t, s, src, c.w0s, c.qs, c.ks, c.vs)
	if e := relL2(out, c.out); !(e <= 1e-4) {
		t.Fatalf("unmutated kernel rel L2 = %g", e)
	}
	for _, m := range []struct{ name, old, new string }{
		{"no variance term", "- mr - zh * mrz) * rsig", "- mr) * rsig"},
		{"W enable_input_d=0", "eg_tm_mma_ss(tm, base + C_W, gt, kt, D, L, 1)", "eg_tm_mma_ss(tm, base + C_W, gt, kt, D, L, 0)"},
	} {
		t.Run(m.name, func(t *testing.T) {
			if !strings.Contains(src, m.old) {
				t.Fatalf("source lacks %q", m.old)
			}
			out, _, _ := runTTTKernel(t, s, strings.Replace(src, m.old, m.new, 1), c.w0s, c.qs, c.ks, c.vs)
			if e := relL2(out, c.out); e <= 1e-2 {
				t.Errorf("mutant passed the oracle: rel L2 = %g", e)
			}
		})
	}
}
