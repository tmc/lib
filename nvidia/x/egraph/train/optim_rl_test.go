package train

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudagradcheck"
)

func TestGRPOTokenLossGrad(t *testing.T) {
	cfg := GRPOConfig{ClipEpsLow: 0.2, ClipEpsHigh: 0.28, BetaKL: 0.04}
	tests := []struct {
		name string
		logp float64
		m    GRPOTokenMeta
	}{
		{"unclipped_pos", -1.0, GRPOTokenMeta{OldLogProb: -1.05, RefLogProb: -1.2, Advantage: 0.7}},
		{"unclipped_neg", -2.0, GRPOTokenMeta{OldLogProb: -1.9, RefLogProb: -1.5, Advantage: -1.3}},
		{"clipped_high_pos", -0.5, GRPOTokenMeta{OldLogProb: -1.5, RefLogProb: -0.7, Advantage: 1.1}},
		{"clipped_high_neg", -0.5, GRPOTokenMeta{OldLogProb: -1.5, RefLogProb: -0.7, Advantage: -1.1}},
		{"clipped_low_neg", -3.0, GRPOTokenMeta{OldLogProb: -1.0, RefLogProb: -2.0, Advantage: -0.4}},
		{"clipped_low_pos", -3.0, GRPOTokenMeta{OldLogProb: -1.0, RefLogProb: -2.0, Advantage: 0.4}},
	}
	for _, tc := range tests {
		_, got := GRPOTokenLoss(tc.logp, tc.m, cfg)
		const h = 1e-6
		lp, _ := GRPOTokenLoss(tc.logp+h, tc.m, cfg)
		lm, _ := GRPOTokenLoss(tc.logp-h, tc.m, cfg)
		want := (lp - lm) / (2 * h)
		if math.Abs(got-want) > 1e-6*max(1, math.Abs(want)) {
			t.Errorf("%s: dloss/dlogp = %v, finite difference %v", tc.name, got, want)
		}
	}
}

type grpoCase struct {
	h, w [][]float32
	meta []GRPOTokenMeta
	cfg  GRPOConfig
}

func newGRPOCase(seed uint64, M, V, D int, cfg GRPOConfig) grpoCase {
	r := rand.New(rand.NewPCG(seed, 20))
	c := grpoCase{cfg: cfg}
	c.cfg.VocabSize, c.cfg.HiddenDim = V, D
	c.h = make([][]float32, M)
	for i := range c.h {
		c.h[i] = make([]float32, D)
		for k := range c.h[i] {
			c.h[i][k] = float32(r.NormFloat64())
		}
	}
	c.w = make([][]float32, V)
	for i := range c.w {
		c.w[i] = make([]float32, D)
		for k := range c.w[i] {
			c.w[i][k] = float32(r.NormFloat64() / math.Sqrt(float64(D)))
		}
	}
	c.meta = make([]GRPOTokenMeta, M)
	for i := range c.meta {
		// Old and reference log-probs near the current one so both the
		// clipped and unclipped branches occur.
		c.meta[i] = GRPOTokenMeta{
			TargetTokenID: int32(r.IntN(V)),
			OldLogProb:    float32(-math.Log(float64(V)) + r.NormFloat64()*0.4),
			RefLogProb:    float32(-math.Log(float64(V)) + r.NormFloat64()*0.4),
			Advantage:     float32(r.NormFloat64()),
		}
	}
	return c
}

// grpoMaterialized is the reference: it materializes all M×V logits in
// float64 and returns the loss and the analytic dH and dW.
func grpoMaterialized(h, w [][]float64, meta []GRPOTokenMeta, cfg GRPOConfig) (loss float64, dH, dW [][]float64) {
	M, V, D := len(h), len(w), len(h[0])
	T := float64(cfg.Temperature)
	if T == 0 {
		T = 1
	}
	dH = make([][]float64, M)
	dW = make([][]float64, V)
	for i := range dW {
		dW[i] = make([]float64, D)
	}
	for i := range M {
		z := make([]float64, V)
		mx := math.Inf(-1)
		for v := range V {
			for k := range D {
				z[v] += h[i][k] * w[v][k]
			}
			z[v] /= T
			mx = max(mx, z[v])
		}
		var se float64
		for _, x := range z {
			se += math.Exp(x - mx)
		}
		lse := mx + math.Log(se)
		y := int(meta[i].TargetTokenID)
		l, g := GRPOTokenLoss(z[y]-lse, meta[i], cfg)
		loss += l / float64(M)
		g /= float64(M)
		dH[i] = make([]float64, D)
		for v := range V {
			dz := -g * math.Exp(z[v]-lse) / T
			if v == y {
				dz += g / T
			}
			for k := range D {
				dH[i][k] += dz * w[v][k]
				dW[v][k] += dz * h[i][k]
			}
		}
	}
	return loss, dH, dW
}

func to64(a [][]float32) [][]float64 {
	out := make([][]float64, len(a))
	for i, r := range a {
		out[i] = make([]float64, len(r))
		for j, v := range r {
			out[i][j] = float64(v)
		}
	}
	return out
}

func flat64(a [][]float64) []float64 {
	var out []float64
	for _, r := range a {
		out = append(out, r...)
	}
	return out
}

func flat32(a [][]float32) []float32 {
	var out []float32
	for _, r := range a {
		out = append(out, r...)
	}
	return out
}

// maxRelErr returns max|got-want| / max|want|.
func maxRelErr(got []float32, want []float64) float64 {
	var num, den float64
	for i := range want {
		num = max(num, math.Abs(float64(got[i])-want[i]))
		den = max(den, math.Abs(want[i]))
	}
	if den == 0 {
		return num
	}
	return num / den
}

func TestComputeFusedGRPOChunk(t *testing.T) {
	base := GRPOConfig{ClipEpsLow: 0.2, ClipEpsHigh: 0.28, BetaKL: 0.04}
	tests := []struct {
		name    string
		M, V, D int
		tileM   int
		tileV   int
		temp    float32
	}{
		{"single_tile", 4, 48, 16, 8, 64, 1},
		{"ragged_vocab", 7, 100, 12, 3, 32, 1},
		{"temperature", 5, 70, 8, 2, 16, 0.7},
		{"tile_1", 3, 9, 4, 1, 1, 1.3},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := base
			cfg.TileM, cfg.TileV, cfg.Temperature = tc.tileM, tc.tileV, tc.temp
			c := newGRPOCase(uint64(tc.M*1000+tc.V), tc.M, tc.V, tc.D, cfg)
			loss, dH, dW, err := ComputeFusedGRPOChunk(c.h, c.w, c.meta, c.cfg)
			if err != nil {
				t.Fatal(err)
			}
			h64, w64 := to64(c.h), to64(c.w)
			wantLoss, wantDH, wantDW := grpoMaterialized(h64, w64, c.meta, c.cfg)
			if e := math.Abs(float64(loss)-wantLoss) / max(1e-12, math.Abs(wantLoss)); e > 1e-5 {
				t.Errorf("loss = %v, want %v (rel %g)", loss, wantLoss, e)
			}
			if e := maxRelErr(flat32(dH), flat64(wantDH)); e > 1e-5 {
				t.Errorf("dH rel err %g", e)
			}
			if e := maxRelErr(flat32(dW), flat64(wantDW)); e > 1e-5 {
				t.Errorf("dW rel err %g", e)
			}

			// The materialized analytic gradient must match finite
			// differences of the loss, so the oracle is not just the
			// same algebra twice.
			fh := func(x []float64) float64 {
				hh := make([][]float64, tc.M)
				for i := range hh {
					hh[i] = x[i*tc.D : (i+1)*tc.D]
				}
				l, _, _ := grpoMaterialized(hh, w64, c.meta, c.cfg)
				return l
			}
			if rep, err := cudagradcheck.CheckGrad(fh, flat64(h64), flat64(wantDH), cudagradcheck.GradOptions{Tolerance: cudagradcheck.Tolerance{Atol: 1e-7, Rtol: 1e-4}}); err != nil {
				t.Errorf("dH gradcheck: %v (%v)", err, rep)
			}
			fw := func(x []float64) float64 {
				ww := make([][]float64, tc.V)
				for i := range ww {
					ww[i] = x[i*tc.D : (i+1)*tc.D]
				}
				l, _, _ := grpoMaterialized(h64, ww, c.meta, c.cfg)
				return l
			}
			if rep, err := cudagradcheck.CheckGrad(fw, flat64(w64), flat64(wantDW), cudagradcheck.GradOptions{Tolerance: cudagradcheck.Tolerance{Atol: 1e-7, Rtol: 1e-4}}); err != nil {
				t.Errorf("dW gradcheck: %v (%v)", err, rep)
			}
		})
	}
}

func TestComputeFusedGRPOChunkInvalid(t *testing.T) {
	c := newGRPOCase(1, 2, 8, 4, GRPOConfig{})
	bad := c.meta[:1]
	if _, _, _, err := ComputeFusedGRPOChunk(c.h, c.w, bad, c.cfg); err == nil {
		t.Error("short meta: no error")
	}
	c.meta[0].TargetTokenID = 8
	if _, _, _, err := ComputeFusedGRPOChunk(c.h, c.w, c.meta, c.cfg); err == nil {
		t.Error("target out of range: no error")
	}
	c.meta[0].TargetTokenID = 0
	cfg := c.cfg
	cfg.ClipEpsLow = 1
	if _, _, _, err := ComputeFusedGRPOChunk(c.h, c.w, c.meta, cfg); err == nil {
		t.Error("ClipEpsLow 1: no error")
	}
}

// grpoKernelFixture is the GRPO case shared by the emulated and native
// kernel tests.
type grpoKernelFixture struct {
	grpoCase
	target        []int32
	old, ref, adv []float32
}

func newGRPOKernelFixture() grpoKernelFixture {
	const M, V, D = 6, 90, 24
	cfg := GRPOConfig{TileV: 32, ClipEpsLow: 0.2, ClipEpsHigh: 0.28, BetaKL: 0.04, Temperature: 0.9}
	f := grpoKernelFixture{grpoCase: newGRPOCase(77, M, V, D, cfg)}
	f.target = make([]int32, M)
	f.old, f.ref, f.adv = make([]float32, M), make([]float32, M), make([]float32, M)
	for i, m := range f.meta {
		f.target[i], f.old[i], f.ref[i], f.adv[i] = m.TargetTokenID, m.OldLogProb, m.RefLogProb, m.Advantage
	}
	return f
}

func (f grpoKernelFixture) kernel(name string) FusedGRPOChunk {
	return FusedGRPOChunk{Name: name, Config: f.cfg, Threads: 64}
}

// check compares kernel outputs (per-token loss, dH, dW) with the
// materialized float64 reference at 1e-4 relative error.
func (f grpoKernelFixture) check(loss, dH, dW []float32) error {
	M := len(f.h)
	wantLoss, wantDH, wantDW := grpoMaterialized(to64(f.h), to64(f.w), f.meta, f.cfg)
	var gotLoss float64
	for _, l := range loss {
		gotLoss += float64(l) / float64(M)
	}
	var errs []error
	if e := math.Abs(gotLoss-wantLoss) / math.Abs(wantLoss); !(e <= 1e-4) {
		errs = append(errs, fmt.Errorf("loss = %v, want %v", gotLoss, wantLoss))
	}
	if e := maxRelErr(dH, flat64(wantDH)); !(e <= 1e-4) {
		errs = append(errs, fmt.Errorf("dH rel err %g", e))
	}
	if e := maxRelErr(dW, flat64(wantDW)); !(e <= 1e-4) {
		errs = append(errs, fmt.Errorf("dW rel err %g", e))
	}
	return errors.Join(errs...)
}

// runGRPOEmu runs src, the emitted translation unit of f.kernel(name), under
// cudahostemu.
func (f grpoKernelFixture) runEmu(t *testing.T, name, src string) (loss, dH, dW []float32) {
	t.Helper()
	M, V, D := len(f.h), f.cfg.VocabSize, f.cfg.HiddenDim
	launch := fmt.Sprintf("%s_launch(loss, dH, dW, H, W, target, old_logp, ref_logp, adv, %d, %v, 0)", name, M, floatLiteral(1/float32(M)))
	res := runEmu(t, src, launch,
		emuBuf{"loss", "float", poison(4 * M)},
		emuBuf{"dH", "float", poison(4 * M * D)},
		emuBuf{"dW", "float", make([]byte, 4*V*D)},
		emuBuf{"H", "float", f32Bytes(flat32(f.h))},
		emuBuf{"W", "float", f32Bytes(flat32(f.w))},
		emuBuf{"target", "int", i32Bytes(f.target)},
		emuBuf{"old_logp", "float", f32Bytes(f.old)},
		emuBuf{"ref_logp", "float", f32Bytes(f.ref)},
		emuBuf{"adv", "float", f32Bytes(f.adv)},
	)
	return bytesF32(res["loss"]), bytesF32(res["dH"]), bytesF32(res["dW"])
}

func TestFusedGRPOChunkEmu(t *testing.T) {
	f := newGRPOKernelFixture()
	src, err := f.kernel("grpo_test").Emit()
	if err != nil {
		t.Fatal(err)
	}
	if err := f.check(f.runEmu(t, "grpo_test", src)); err != nil {
		t.Error(err)
	}
}

// randOrthonormal returns n orthonormal vectors of length m (n <= m).
func randOrthonormal(r *rand.Rand, m, n int) [][]float64 {
	q := make([][]float64, 0, n)
	for len(q) < n {
		v := make([]float64, m)
		for i := range v {
			v[i] = r.NormFloat64()
		}
		for range 2 {
			for _, u := range q {
				var d float64
				for i := range v {
					d += v[i] * u[i]
				}
				for i := range v {
					v[i] -= d * u[i]
				}
			}
		}
		var nrm float64
		for _, x := range v {
			nrm += x * x
		}
		nrm = math.Sqrt(nrm)
		for i := range v {
			v[i] /= nrm
		}
		q = append(q, v)
	}
	return q
}

// symEigen returns the eigenvalues of the symmetric matrix a (n×n, row
// major) by cyclic Jacobi rotations.
func symEigen(a []float64, n int) []float64 {
	a = append([]float64(nil), a...)
	for sweep := 0; sweep < 50; sweep++ {
		var off float64
		for i := range n {
			for j := i + 1; j < n; j++ {
				off += a[i*n+j] * a[i*n+j]
			}
		}
		if off < 1e-22 {
			break
		}
		for p := range n {
			for q := p + 1; q < n; q++ {
				apq := a[p*n+q]
				if math.Abs(apq) < 1e-300 {
					continue
				}
				theta := (a[q*n+q] - a[p*n+p]) / (2 * apq)
				tt := math.Copysign(1, theta) / (math.Abs(theta) + math.Sqrt(theta*theta+1))
				c := 1 / math.Sqrt(tt*tt+1)
				s := tt * c
				for k := range n {
					akp, akq := a[k*n+p], a[k*n+q]
					a[k*n+p], a[k*n+q] = c*akp-s*akq, s*akp+c*akq
				}
				for k := range n {
					apk, aqk := a[p*n+k], a[q*n+k]
					a[p*n+k], a[q*n+k] = c*apk-s*aqk, s*apk+c*aqk
				}
			}
		}
	}
	ev := make([]float64, n)
	for i := range n {
		ev[i] = a[i*n+i]
	}
	return ev
}

// muonSpectrum returns the singular values of x (rows <= cols) and
// ‖X·Xᵀ − I‖_F / sqrt(rows).
func muonSpectrum(x [][]float32) (sv []float64, orthErr float64) {
	m, n := len(x), len(x[0])
	gram := make([]float64, m*m)
	for i := range m {
		for j := range m {
			var s float64
			for k := range n {
				s += float64(x[i][k]) * float64(x[j][k])
			}
			gram[i*m+j] = s
			d := s
			if i == j {
				d -= 1
			}
			orthErr += d * d
		}
	}
	for _, e := range symEigen(gram, m) {
		sv = append(sv, math.Sqrt(max(e, 0)))
	}
	return sv, math.Sqrt(orthErr) / math.Sqrt(float64(m))
}

// illConditioned returns a rows×cols matrix with singular values spaced
// logarithmically from 1 to kappa.
func illConditioned(seed uint64, rows, cols int, kappa float64) [][]float32 {
	r := rand.New(rand.NewPCG(seed, 3))
	u := randOrthonormal(r, rows, rows)
	v := randOrthonormal(r, cols, rows)
	g := make([][]float32, rows)
	for i := range rows {
		g[i] = make([]float32, cols)
		for j := range cols {
			var s float64
			for k := range rows {
				sk := math.Pow(kappa, float64(k)/float64(rows-1))
				s += u[k][i] * sk * v[k][j]
			}
			g[i][j] = float32(s)
		}
	}
	return g
}

func TestMuonOrthogonalize(t *testing.T) {
	g := illConditioned(1, 128, 256, 60)
	sv0, _ := muonSpectrum(g)
	lo, hi := math.Inf(1), 0.0
	for _, s := range sv0 {
		lo, hi = min(lo, s), max(hi, s)
	}
	if hi/lo <= 50 {
		t.Fatalf("condition number %v, want > 50", hi/lo)
	}
	check := func(cfg MuonConfig) error {
		x, err := MuonOrthogonalize(g, cfg)
		if err != nil {
			return err
		}
		sv, oe := muonSpectrum(x)
		for _, s := range sv {
			if s < 0.68 || s > 1.32 {
				return fmt.Errorf("singular value %v outside [0.68, 1.32]", s)
			}
		}
		// The quintic leaves singular values spread over about
		// [0.68, 1.2], so ‖XXᵀ−I‖_F/√m stays near 0.35 with the
		// reference coefficients; 0.22 is not reachable.
		if oe >= 0.4 {
			return fmt.Errorf("‖XXᵀ−I‖_F/√m = %v, want < 0.4", oe)
		}
		return nil
	}
	if err := check(MuonConfig{}); err != nil {
		t.Fatalf("default coefficients: %v", err)
	}
	// Each perturbed coefficient set must fail the spectral oracle.
	for _, c := range [][3]float32{
		{3.0, -4.7750, 2.0315},
		{3.4445, -4.0, 2.0315},
		{3.4445, -4.7750, 2.6},
		{1.5, -0.5, 0}, // cubic Newton–Schulz, too slow in 5 steps
	} {
		if err := check(MuonConfig{Coeffs: c}); err == nil {
			t.Errorf("coefficients %v passed the spectral oracle", c)
		}
	}
	if err := check(MuonConfig{Steps: 3}); err == nil {
		t.Error("3 steps passed the spectral oracle")
	}
}

func TestMuonOrthogonalizeTall(t *testing.T) {
	g := illConditioned(2, 24, 40, 20)
	gt := make([][]float32, 40)
	for j := range gt {
		gt[j] = make([]float32, 24)
		for i := range 24 {
			gt[j][i] = g[i][j]
		}
	}
	x, err := MuonOrthogonalize(g, MuonConfig{})
	if err != nil {
		t.Fatal(err)
	}
	xt, err := MuonOrthogonalize(gt, MuonConfig{})
	if err != nil {
		t.Fatal(err)
	}
	for i := range 24 {
		for j := range 40 {
			if math.Abs(float64(x[i][j]-xt[j][i])) > 1e-6 {
				t.Fatalf("tall result is not the transpose of the wide result at (%d,%d)", i, j)
			}
		}
	}
}

// checkMuonOutput compares a kernel's orthogonalized matrix with
// MuonOrthogonalize at 1e-4 relative error.
func checkMuonOutput(got []float32, g [][]float32, cfg MuonConfig) error {
	want, err := MuonOrthogonalize(g, cfg)
	if err != nil {
		return err
	}
	if e := maxRelErr(got, flat64(to64(want))); !(e <= 1e-4) {
		return fmt.Errorf("muon rel err %g", e)
	}
	return nil
}

func TestMuonNewtonSchulzEmu(t *testing.T) {
	for _, steps := range []int{4, 5} {
		t.Run(fmt.Sprint(steps), func(t *testing.T) {
			const R, C = 12, 20
			g := illConditioned(3, R, C, 10)
			k := MuonNewtonSchulz{Name: "muon_test", Rows: R, Cols: C, Config: MuonConfig{Steps: steps}}
			src, err := k.Emit()
			if err != nil {
				t.Fatal(err)
			}
			res := runEmu(t, src, "muon_test_launch(X, Xtmp, A, B, G, 0)",
				emuBuf{"X", "float", poison(4 * R * C)},
				emuBuf{"Xtmp", "float", poison(4 * R * C)},
				emuBuf{"A", "float", poison(4 * R * R)},
				emuBuf{"B", "float", poison(4 * R * R)},
				emuBuf{"G", "float", f32Bytes(flat32(g))},
			)
			if err := checkMuonOutput(bytesF32(res["X"]), g, k.Config); err != nil {
				t.Error(err)
			}
		})
	}
	if _, err := (MuonNewtonSchulz{Rows: 8, Cols: 4}).Emit(); err == nil {
		t.Error("Rows > Cols: no error")
	}
}

func ExampleGRPOTokenLoss() {
	cfg := GRPOConfig{ClipEpsLow: 0.2, ClipEpsHigh: 0.28, BetaKL: 0.04}
	m := GRPOTokenMeta{OldLogProb: -1.5, RefLogProb: -1.0, Advantage: 1}
	for _, logp := range []float64{-1.5, -1.0} {
		loss, d := GRPOTokenLoss(logp, m, cfg)
		fmt.Printf("logp %.1f: loss %.4f dloss/dlogp %.4f\n", logp, loss, d)
	}
	// Output:
	// logp -1.5: loss -0.9941 dloss/dlogp -1.0259
	// logp -1.0: loss -1.2800 dloss/dlogp 0.0000
}
