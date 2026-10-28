package nvl72

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
)

// moeLayer is a float64 MoE layer for one token: Y = Ws x + sum_k g_k(u)
// tanh(W_k x), g = sigmoid(u) / sum sigmoid(u).
type moeLayer struct {
	D, K int
	Ws   [][]float64   // [D][D]
	W    [][][]float64 // [K][D][D]
}

func newMoELayer(rng *rand.Rand, D, K int) moeLayer {
	mat := func() [][]float64 {
		m := make([][]float64, D)
		for i := range m {
			m[i] = make([]float64, D)
			for j := range m[i] {
				m[i][j] = rng.NormFloat64() / math.Sqrt(float64(D))
			}
		}
		return m
	}
	l := moeLayer{D: D, K: K, Ws: mat()}
	for range K {
		l.W = append(l.W, mat())
	}
	return l
}

func matVec(m [][]float64, x []float64) []float64 {
	y := make([]float64, len(m))
	for i, row := range m {
		for j, v := range row {
			y[i] += v * x[j]
		}
	}
	return y
}

func matTVec(m [][]float64, x []float64) []float64 {
	y := make([]float64, len(m[0]))
	for i, row := range m {
		for j, v := range row {
			y[j] += v * x[i]
		}
	}
	return y
}

func (l moeLayer) expert(k int, x []float64) []float64 {
	y := matVec(l.W[k], x)
	for i := range y {
		y[i] = math.Tanh(y[i])
	}
	return y
}

// loss is <dY, Y(x, u)>.
func (l moeLayer) loss(x, u, dY []float64) float64 {
	y := matVec(l.Ws, x)
	var S float64
	s := make([]float64, l.K)
	for k := range s {
		s[k] = 1 / (1 + math.Exp(-u[k]))
		S += s[k]
	}
	for k := range l.K {
		e := l.expert(k, x)
		for i := range y {
			y[i] += s[k] / S * e[i]
		}
	}
	var L float64
	for i := range y {
		L += dY[i] * y[i]
	}
	return L
}

func f32s(x []float64) []float32 {
	y := make([]float32, len(x))
	for i, v := range x {
		y[i] = float32(v)
	}
	return y
}

type moeBwdCase struct {
	dY, dXShared, logits [][]float32
	gradX, fwdY          [][][]float32
	fdX, fdU             [][]float64
}

// newMoEBwdCase builds per-token inputs to ComputeMoEBackwardCombine from
// the layer's analytic per-expert VJPs, and central finite differences of
// the layer's loss for the oracle.
func newMoEBwdCase(rng *rand.Rand, T, D, K int) moeBwdCase {
	var c moeBwdCase
	for range T {
		l := newMoELayer(rng, D, K)
		x, u, dY := make([]float64, D), make([]float64, K), make([]float64, D)
		for i := range x {
			x[i], dY[i] = rng.NormFloat64(), rng.NormFloat64()
		}
		for k := range u {
			u[k] = rng.NormFloat64()
		}
		var gx, fy [][]float32
		for k := range K {
			pre := matVec(l.W[k], x)
			back := make([]float64, D)
			for i := range back {
				th := math.Tanh(pre[i])
				back[i] = dY[i] * (1 - th*th)
			}
			gx = append(gx, f32s(matTVec(l.W[k], back)))
			fy = append(fy, f32s(l.expert(k, x)))
		}
		const h = 1e-6
		fdX := make([]float64, D)
		for i := range x {
			xp, xm := append([]float64(nil), x...), append([]float64(nil), x...)
			xp[i] += h
			xm[i] -= h
			fdX[i] = (l.loss(xp, u, dY) - l.loss(xm, u, dY)) / (2 * h)
		}
		fdU := make([]float64, K)
		for k := range u {
			up, um := append([]float64(nil), u...), append([]float64(nil), u...)
			up[k] += h
			um[k] -= h
			fdU[k] = (l.loss(x, up, dY) - l.loss(x, um, dY)) / (2 * h)
		}
		c.dY = append(c.dY, f32s(dY))
		c.dXShared = append(c.dXShared, f32s(matTVec(l.Ws, dY)))
		c.logits = append(c.logits, f32s(u))
		c.gradX = append(c.gradX, gx)
		c.fwdY = append(c.fwdY, fy)
		c.fdX = append(c.fdX, fdX)
		c.fdU = append(c.fdU, fdU)
	}
	return c
}

func maxAbsDiff(got [][]float32, want [][]float64) float64 {
	worst := 0.0
	for t := range got {
		for i := range got[t] {
			worst = max(worst, math.Abs(float64(got[t][i])-want[t][i]))
		}
	}
	return worst
}

// TestMoEBackwardFiniteDifference is the autograd oracle: GradX and the
// pre-sigmoid GradGateLogits match central differences of the forward MoE
// layer, in both combine modes. The two classic mistakes, dropping the
// -sum_j g_j dg_j term of the normalization pullback and summing dX
// without the routing weights, are rejected by the same oracle.
func TestMoEBackwardFiniteDifference(t *testing.T) {
	const T, D, K = 6, 16, 4
	c := newMoEBwdCase(rand.New(rand.NewPCG(41, 42)), T, D, K)
	for _, sw := range []bool{false, true} {
		t.Run(fmt.Sprintf("switch=%v", sw), func(t *testing.T) {
			cfg := MoEBackwardCombineConfig{NumRanks: 4, NumExperts: 8, TopK: K, HiddenDim: D, HasSharedExp: true, UseSwitchRed: sw}
			r, err := ComputeMoEBackwardCombine(c.dY, c.dXShared, c.gradX, c.fwdY, c.logits, cfg)
			if err != nil {
				t.Fatal(err)
			}
			if d := maxAbsDiff(r.GradX, c.fdX); !(d <= 1e-5) {
				t.Fatalf("GradX differs from finite differences by %g", d)
			}
			if d := maxAbsDiff(r.GradGateLogits, c.fdU); !(d <= 1e-5) {
				t.Fatalf("GradGateLogits differs from finite differences by %g", d)
			}
		})
	}

	cfg := MoEBackwardCombineConfig{NumRanks: 4, NumExperts: 8, TopK: K, HiddenDim: D, HasSharedExp: true}
	r, err := ComputeMoEBackwardCombine(c.dY, c.dXShared, c.gradX, c.fwdY, c.logits, cfg)
	if err != nil {
		t.Fatal(err)
	}
	noQuotient := make([][]float32, T)
	unscaled := make([][]float32, T)
	for tok := range T {
		s, _ := gateWeights(c.logits[tok])
		var S float32
		for _, x := range s {
			S += x
		}
		for k := range K {
			noQuotient[tok] = append(noQuotient[tok], s[k]*(1-s[k])/S*r.GradGate[tok][k])
		}
		unscaled[tok] = append([]float32(nil), c.dXShared[tok]...)
		for k := range K {
			for d := range D {
				unscaled[tok][d] += c.gradX[tok][k][d]
			}
		}
	}
	if d := maxAbsDiff(noQuotient, c.fdU); d < 1e-3 {
		t.Fatalf("pullback without the normalization term is within %g of the oracle", d)
	}
	if d := maxAbsDiff(unscaled, c.fdX); d < 1e-3 {
		t.Fatalf("combine without pre-scaling is within %g of the oracle", d)
	}
}

func TestMoEBackwardSwitchMatchesHomeCombine(t *testing.T) {
	const T, D, K = 5, 32, 8
	c := newMoEBwdCase(rand.New(rand.NewPCG(43, 44)), T, D, K)
	cfg := MoEBackwardCombineConfig{NumRanks: 8, NumExperts: 64, TopK: K, HiddenDim: D, HasSharedExp: true}
	home, err := ComputeMoEBackwardCombine(c.dY, c.dXShared, c.gradX, c.fwdY, c.logits, cfg)
	if err != nil {
		t.Fatal(err)
	}
	cfg.UseSwitchRed = true
	sw, err := ComputeMoEBackwardCombine(c.dY, c.dXShared, c.gradX, c.fwdY, c.logits, cfg)
	if err != nil {
		t.Fatal(err)
	}
	for tok := range T {
		for d := range D {
			if math.Float32bits(home.GradX[tok][d]) != math.Float32bits(sw.GradX[tok][d]) {
				t.Fatalf("token %d dim %d: switch %v, home %v", tok, d, sw.GradX[tok][d], home.GradX[tok][d])
			}
		}
	}
}

func TestMoEBackwardTraffic(t *testing.T) {
	// One layer of 4096 tokens at D = 7168 in BF16, top-8 plus a shared expert.
	const T, D, K = 4096, 7168, 8
	row := int64(T * D * 2)
	tests := []struct {
		switchRed             bool
		reads, ingress, swRed int64
	}{
		{false, 9 * row, 8 * row, 0},
		{true, 0, row, 8 * row},
	}
	for _, tc := range tests {
		cfg := MoEBackwardCombineConfig{NumRanks: 72, NumExperts: 256, TopK: K, HiddenDim: 1, HasSharedExp: true, UseSwitchRed: tc.switchRed}
		// Compute on a 1-wide slice and scale: traffic is linear in D.
		c := newMoEBwdCase(rand.New(rand.NewPCG(45, 46)), 1, 1, K)
		r, err := ComputeMoEBackwardCombine(c.dY, c.dXShared, c.gradX, c.fwdY, c.logits, cfg)
		if err != nil {
			t.Fatal(err)
		}
		scale := int64(T * D)
		if got := r.HomeRankHBMCombineReads * scale; got != tc.reads {
			t.Fatalf("switch=%v: combine reads %d, want %d", tc.switchRed, got, tc.reads)
		}
		if got := r.HomeRankIngressBytes * scale; got != tc.ingress {
			t.Fatalf("switch=%v: ingress %d, want %d", tc.switchRed, got, tc.ingress)
		}
		if got := r.SwitchReducedBytes * scale; got != tc.swRed {
			t.Fatalf("switch=%v: switch bytes %d, want %d", tc.switchRed, got, tc.swRed)
		}
	}
	t.Logf("home-rank combine reads avoided per layer: %d bytes (%.0f MiB)", 9*row, float64(9*row)/(1<<20))
}

func TestMoEBackwardErrors(t *testing.T) {
	c := newMoEBwdCase(rand.New(rand.NewPCG(47, 48)), 2, 4, 2)
	good := MoEBackwardCombineConfig{NumRanks: 2, NumExperts: 4, TopK: 2, HiddenDim: 4, HasSharedExp: true}
	for name, cfg := range map[string]MoEBackwardCombineConfig{
		"wide_hidden": {NumRanks: 2, NumExperts: 4, TopK: 2, HiddenDim: 5, HasSharedExp: true},
		"topk_gt_e":   {NumRanks: 2, NumExperts: 1, TopK: 2, HiddenDim: 4},
		"bad_target":  {Target: "sm_90a", NumRanks: 2, NumExperts: 4, TopK: 2, HiddenDim: 4},
	} {
		if _, err := ComputeMoEBackwardCombine(c.dY, c.dXShared, c.gradX, c.fwdY, c.logits, cfg); err == nil {
			t.Fatalf("%s: succeeded, want error", name)
		}
	}
	if _, err := ComputeMoEBackwardCombine(c.dY, nil, c.gradX, c.fwdY, c.logits, good); err == nil {
		t.Fatal("missing shared gradients accepted")
	}
}

func TestEmitSwitchMoEBackwardEpiloguePTX(t *testing.T) {
	for _, tc := range []struct {
		target string
		sw     bool
		want   []string
		absent []string
	}{
		{"sm_107a", true, []string{"fabric.try_red.async.tensor.2d.multimem", "mul.f32 %f_dx0, %f_dx0, %f_g", "shfl.sync.bfly.b32"}, []string{"tcgen05.ld.red"}},
		{"sm_100a", true, []string{"cp.reduce.async.bulk.tensor.2d.global.shared::cta.add", "mul.f32 %f_dx0"}, []string{"fabric.try_red"}},
		{"sm_100a", false, []string{"cp.async.bulk.tensor.2d.global.shared::cta.tile"}, []string{"mul.f32 %f_dx0", "cp.reduce"}},
	} {
		ptx, err := EmitSwitchMoEBackwardEpiloguePTX(MoEBackwardCombineConfig{Target: tc.target, NumRanks: 72, NumExperts: 256, TopK: 8, HiddenDim: 7168, UseSwitchRed: tc.sw}, "moe_bwd_epi")
		if err != nil {
			t.Fatal(err)
		}
		for _, w := range tc.want {
			if !strings.Contains(ptx, w) {
				t.Fatalf("%s switch=%v: missing %q", tc.target, tc.sw, w)
			}
		}
		for _, a := range tc.absent {
			if strings.Contains(ptx, a) {
				t.Fatalf("%s switch=%v: unexpected %q", tc.target, tc.sw, a)
			}
		}
		// Scaling must precede the reduction it feeds.
		if tc.sw && strings.Index(ptx, "mul.f32 %f_dx0") > strings.Index(ptx, "fence.proxy.async") {
			t.Fatalf("%s: dX scaled after staging", tc.target)
		}
	}
}

// TestSwitchMoEBackwardKernelHostEmu runs the emitted combine on 4 emulated
// ranks and checks every home rank's GradX and dg against the reference.
func TestSwitchMoEBackwardKernelHostEmu(t *testing.T) {
	const P, T, K, D = 4, 8, 3, 40
	c := newMoEBwdCase(rand.New(rand.NewPCG(49, 50)), T, D, K)
	cfg := MoEBackwardCombineConfig{NumRanks: P, NumExperts: 12, TopK: K, HiddenDim: D, HasSharedExp: true, UseSwitchRed: true}
	want, err := ComputeMoEBackwardCombine(c.dY, c.dXShared, c.gradX, c.fwdY, c.logits, cfg)
	if err != nil {
		t.Fatal(err)
	}
	k := SwitchMoEBackwardKernel{Name: "moe_bwd_combine", Config: cfg, Tokens: T}
	src, err := k.Source()
	if err != nil {
		t.Fatal(err)
	}
	var dy, y, dx, gate, gradX []float32
	var route []int32
	for range P {
		for tok := range T {
			dy = append(dy, c.dY[tok]...)
			gradX = append(gradX, c.dXShared[tok]...)
			_, g := gateWeights(c.logits[tok])
			for kk := range K {
				y = append(y, c.fwdY[tok][kk]...)
				dx = append(dx, c.gradX[tok][kk]...)
				gate = append(gate, g[kk])
				route = append(route, int32((tok+kk+1)%P))
			}
		}
	}
	gxBytes := f32Bytes(gradX)
	dgBytes := cudahostemu.Poison(4 * P * T * K)
	grid, block := k.Grid()
	err = cudahostemu.RunMultiRank(context.Background(), src, P, cudahostemu.Launch{
		Kernel: k.Name,
		Grid:   cudahostemu.Dim3{X: grid[0], Y: grid[1], Z: grid[2]},
		Block:  cudahostemu.Dim3{X: block[0], Y: block[1], Z: block[2]},
	},
		cudahostemu.BufferArg("dy", "float", f32Bytes(dy)),
		cudahostemu.BufferArg("y", "float", f32Bytes(y)),
		cudahostemu.BufferArg("dx", "float", f32Bytes(dx)),
		cudahostemu.BufferArg("gate", "float", f32Bytes(gate)),
		cudahostemu.BufferArg("route", "int32_t", i32Bytes(route)),
		cudahostemu.BufferArg("grad_x", "float", gxBytes),
		cudahostemu.BufferArg("dg", "float", dgBytes),
	)
	if errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip(err)
	}
	if err != nil {
		t.Fatalf("RunMultiRank: %v", err)
	}
	gotX, gotDG := bytesF32(gxBytes), bytesF32(dgBytes)
	for tok := range T {
		home := tok % P
		for d := range D {
			got := gotX[(home*T+tok)*D+d]
			if !(math.Abs(float64(got-want.GradX[tok][d])) <= 1e-5) {
				t.Fatalf("token %d dim %d on home rank %d: %v, want %v", tok, d, home, got, want.GradX[tok][d])
			}
		}
		for kk := range K {
			got := gotDG[(home*T+tok)*K+kk]
			if !(math.Abs(float64(got-want.GradGate[tok][kk])) <= 1e-5) {
				t.Fatalf("token %d slot %d dg %v, want %v", tok, kk, got, want.GradGate[tok][kk])
			}
		}
	}
}

func ExampleComputeMoEBackwardCombine() {
	dY := [][]float32{{1, 0}}
	shared := [][]float32{{0.5, 0.5}}
	dX := [][][]float32{{{1, 1}, {2, 2}}} // unscaled expert input gradients
	Y := [][][]float32{{{3, 0}, {1, 0}}}  // expert outputs
	u := [][]float32{{0, 0}}              // equal gates: g = 0.5, 0.5
	r, err := ComputeMoEBackwardCombine(dY, shared, dX, Y, u, MoEBackwardCombineConfig{
		NumRanks: 2, NumExperts: 2, TopK: 2, HiddenDim: 2, HasSharedExp: true, UseSwitchRed: true,
	})
	if err != nil {
		panic(err)
	}
	fmt.Println(r.GradX[0], r.GradGate[0], r.GradGateLogits[0], r.HomeRankHBMCombineReads)
	// Output:
	// [2 2] [3 1] [0.25 -0.25] 0
}
