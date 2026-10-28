package tcgen05

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatmem"
)

func ones32() [32]float32 {
	var h [32]float32
	for i := range h {
		h[i] = 1
	}
	return h
}

func sqnrDB(w, wHat []float32) float64 {
	var s, e float64
	for i := range w {
		s += float64(w[i]) * float64(w[i])
		d := float64(w[i] - wHat[i])
		e += d * d
	}
	return 10 * math.Log10(s/e)
}

func TestE4M3GridRoundTrip(t *testing.T) {
	finite := 0
	for c := range 256 {
		code := uint8(c)
		if code&0x7f == 0x7f {
			if v := FromE4M3Bits(code); !math.IsNaN(float64(v)) {
				t.Fatalf("FromE4M3Bits(%#02x) = %v, want NaN", code, v)
			}
			continue
		}
		finite++
		v := FromE4M3Bits(code)
		if got := E4M3Bits(v); got != code && !(v == 0 && got&0x7f == 0) {
			t.Fatalf("E4M3Bits(FromE4M3Bits(%#02x)=%v) = %#02x", code, v, got)
		}
	}
	if finite != 254 {
		t.Fatalf("finite E4M3 codes = %d, want 254", finite)
	}
	tests := []struct {
		v    float32
		want uint8
	}{
		{448, 0x7e},
		{1000, 0x7e},
		{-448, 0xfe},
		{1.0 / 512, 0x01}, // smallest subnormal
		{1.0 / 64, 0x08},  // smallest normal
		{1, 0x38},
	}
	for _, tt := range tests {
		if got := E4M3Bits(tt.v); got != tt.want {
			t.Errorf("E4M3Bits(%v) = %#02x, want %#02x", tt.v, got, tt.want)
		}
	}
}

func lutTestBlocks() []struct {
	name string
	w    [32]float32
} {
	r := rand.New(rand.NewPCG(28, 1))
	var outlier, normal, skewed, bimodal [32]float32
	for i := range 32 {
		outlier[i] = float32(r.NormFloat64())
		normal[i] = float32(r.NormFloat64())
		skewed[i] = float32(r.ExpFloat64())
		bimodal[i] = float32(2*float64(1-2*(i%2)) + 0.3*r.NormFloat64())
	}
	outlier[0] = 12
	return []struct {
		name string
		w    [32]float32
	}{
		{"outlier", outlier},
		{"normal", normal},
		{"skewed", skewed},
		{"bimodal", bimodal},
	}
}

// TestFitBlock32LearnedLUTSQNR compares the learned codebook with the fixed
// uniform symmetric codebook on blocks where a single codebook fits poorly.
func TestFitBlock32LearnedLUTSQNR(t *testing.T) {
	minGain := map[string]float64{"outlier": 5, "normal": 2, "skewed": 5, "bimodal": 5}
	for _, tt := range lutTestBlocks() {
		t.Run(tt.name, func(t *testing.T) {
			b, err := FitBlock32LearnedLUT(tt.w, ones32(), 100)
			if err != nil {
				t.Fatal(err)
			}
			u := UniformLUTBlock32(tt.w)
			got, base := b.Dequantize(), u.Dequantize()
			gain := sqnrDB(tt.w[:], got[:]) - sqnrDB(tt.w[:], base[:])
			t.Logf("learned %.2f dB, uniform %.2f dB, gain %.2f dB, mse ratio %.2f, iters %d",
				sqnrDB(tt.w[:], got[:]), sqnrDB(tt.w[:], base[:]), gain, u.WeightedMSE/b.WeightedMSE, b.Iterations)
			if gain < minGain[tt.name] {
				t.Fatalf("SQNR gain %.2f dB, want >= %.1f", gain, minGain[tt.name])
			}
			if tt.name == "outlier" && u.WeightedMSE < 3.5*b.WeightedMSE {
				t.Fatalf("uniform/learned MSE ratio %.2f, want >= 3.5", u.WeightedMSE/b.WeightedMSE)
			}
			if b.Iterations >= 100 {
				t.Fatalf("Lloyd-Max did not converge in 100 iterations")
			}
		})
	}
}

func TestFitLUT8Monotone(t *testing.T) {
	r := rand.New(rand.NewPCG(28, 2))
	for trial := range 50 {
		vals := make([]float64, 32+trial)
		wts := make([]float64, len(vals))
		for i := range vals {
			vals[i] = r.NormFloat64() * math.Exp(2*r.NormFloat64())
			wts[i] = 0.1 + r.ExpFloat64()
		}
		fit := fitLUT8(vals, wts, 200)
		for i := 1; i < len(fit.history); i++ {
			if fit.history[i] > fit.history[i-1]*(1+1e-12) {
				t.Fatalf("trial %d: objective rose at iteration %d: %v -> %v", trial, i, fit.history[i-1], fit.history[i])
			}
		}
		if fit.iters >= 200 {
			t.Fatalf("trial %d: no convergence in 200 iterations", trial)
		}
	}
}

func TestFitBlock32LearnedLUTRepresentable(t *testing.T) {
	for _, tt := range lutTestBlocks() {
		b, err := FitBlock32LearnedLUT(tt.w, ones32(), 100)
		if err != nil {
			t.Fatal(err)
		}
		s := DecodeUE8M0(b.ScaleUE8M0)
		got := b.Dequantize()
		var mse float64
		for i, v := range got {
			c := FromE4M3Bits(b.Codebook[b.Indices[i]])
			if v != c*s {
				t.Fatalf("%s: wHat[%d] = %v, want %v * %v", tt.name, i, v, c, s)
			}
			if q := v / s; FromE4M3Bits(E4M3Bits(q)) != q {
				t.Fatalf("%s: wHat[%d]/scale = %v is not on the E4M3 grid", tt.name, i, q)
			}
			d := float64(tt.w[i] - v)
			mse += d * d
		}
		if math.Abs(mse/32-b.WeightedMSE) > 1e-9*max(1, mse) {
			t.Fatalf("%s: WeightedMSE = %v, recomputed %v", tt.name, b.WeightedMSE, mse/32)
		}
		var amax float32
		for _, v := range tt.w {
			amax = max(amax, float32(math.Abs(float64(v))))
		}
		if amax/s > 448 || amax/s <= 224 {
			t.Fatalf("%s: max|w|/scale = %v, want in (224, 448]", tt.name, amax/s)
		}
	}
}

// TestFitBlock32LearnedLUTHessian checks that the Hessian weights change the
// fit: weighting a few channels heavily must lower the weighted error
// compared with the unweighted fit.
func TestFitBlock32LearnedLUTHessian(t *testing.T) {
	r := rand.New(rand.NewPCG(28, 3))
	var w, h [32]float32
	for i := range w {
		w[i] = float32(r.NormFloat64())
		h[i] = 1
	}
	for _, i := range []int{3, 11, 19, 27} {
		h[i] = 1000
	}
	weighted, err := FitBlock32LearnedLUT(w, h, 100)
	if err != nil {
		t.Fatal(err)
	}
	plain, err := FitBlock32LearnedLUT(w, ones32(), 100)
	if err != nil {
		t.Fatal(err)
	}
	werr := func(b LearnedLUTBlock) float64 {
		got := b.Dequantize()
		var e, s float64
		for i := range w {
			d := float64(w[i] - got[i])
			e += float64(h[i]) * d * d
			s += float64(h[i])
		}
		return e / s
	}
	if math.Abs(werr(weighted)-weighted.WeightedMSE) > 1e-9 {
		t.Fatalf("WeightedMSE = %v, recomputed %v", weighted.WeightedMSE, werr(weighted))
	}
	if !(werr(weighted) < werr(plain)) {
		t.Fatalf("Hessian-weighted error %v not below unweighted fit's %v", werr(weighted), werr(plain))
	}
}

func TestFitBlock32LearnedLUTErrors(t *testing.T) {
	var w [32]float32
	h := ones32()
	h[5] = 0
	if _, err := FitBlock32LearnedLUT(w, h, 10); err == nil {
		t.Error("zero Hessian entry accepted")
	}
	w[0] = float32(math.Inf(1))
	if _, err := FitBlock32LearnedLUT(w, ones32(), 10); err == nil {
		t.Error("infinite weight accepted")
	}
	if _, err := FitBlock32LearnedLUT([32]float32{}, ones32(), 0); err == nil {
		t.Error("maxIters=0 accepted")
	}
	b, err := FitBlock32LearnedLUT([32]float32{}, ones32(), 10)
	if err != nil || b.WeightedMSE != 0 {
		t.Errorf("all-zero block: %v, mse %v", err, b.WeightedMSE)
	}
}

func TestLearnedLUTPacking(t *testing.T) {
	var b LearnedLUTBlock
	for i := range b.Indices {
		b.Indices[i] = uint8((i*5 + 3) % 8)
	}
	p := b.PackIndices()
	if got := UnpackIndices3Bit(p); got != b.Indices {
		t.Fatalf("unpack(pack) = %v, want %v", got, b.Indices)
	}
	var one LearnedLUTBlock
	one.Indices[1] = 7  // bits 3..5
	one.Indices[2] = 5  // bits 6..8, straddles bytes 0 and 1
	one.Indices[31] = 6 // bits 93..95
	p = one.PackIndices()
	if p[0] != 0x78 || p[1] != 0x01 || p[11] != 0xc0 {
		t.Fatalf("packed bytes = % x, want 78 01 ... c0", p)
	}
	b.Codebook = [8]uint8{0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08}
	if w := b.CodebookWords(); w != [2]uint32{0x04030201, 0x08070605} {
		t.Fatalf("CodebookWords = %#x", w)
	}
}

// TestLUTColumnAlignment checks the 2-column TMEM LUT rule both in
// ValidateLUTColumn and in cudatmem.VerifyTrace for a decompress::lut::b MMA.
func TestLUTColumnAlignment(t *testing.T) {
	for _, tt := range []struct {
		col uint32
		ok  bool
	}{{0, true}, {128, true}, {510, true}, {129, false}, {511, false}, {512, false}} {
		if err := ValidateLUTColumn(tt.col); (err == nil) != tt.ok {
			t.Errorf("ValidateLUTColumn(%d) = %v, want ok=%v", tt.col, err, tt.ok)
		}
	}
	trace := func(lut uint32) []cudatmem.TraceEvent {
		return []cudatmem.TraceEvent{
			{Kind: cudatmem.EventAlloc, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128},
			{Kind: cudatmem.EventAlloc, ActiveThreads: 32, CTAGroup: 1, Offset: 128, Columns: 32, Rows: 128},
			{Kind: cudatmem.EventCp, ActiveThreads: 1, CTAGroup: 1, Offset: 128, Columns: 32, Rows: 128},
			{Kind: cudatmem.EventMMA, ActiveThreads: 1, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128, DecompressLUTB: true, LUTMetadataOffset: lut},
			{Kind: cudatmem.EventCommit, ActiveThreads: 1, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128},
			{Kind: cudatmem.EventWait, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128},
			{Kind: cudatmem.EventDealloc, ActiveThreads: 32, CTAGroup: 1, Offset: 0, Columns: 128, Rows: 128},
			{Kind: cudatmem.EventDealloc, ActiveThreads: 32, CTAGroup: 1, Offset: 128, Columns: 32, Rows: 128},
		}
	}
	if _, err := cudatmem.VerifyTrace(cudatmem.SM107A, trace(128)); err != nil {
		t.Fatalf("VerifyTrace(lut col 128): %v", err)
	}
	if _, err := cudatmem.VerifyTrace(cudatmem.SM107A, trace(129)); err == nil {
		t.Fatal("VerifyTrace accepted odd LUT column 129")
	}
}

func lutTestMatrix(K, N int, seed uint64) (b, h []float32) {
	r := rand.New(rand.NewPCG(28, seed))
	b = make([]float32, K*N)
	for i := range b {
		b[i] = float32(r.NormFloat64() * math.Exp(0.5*r.NormFloat64()))
	}
	for k := 0; k < K; k += 37 {
		b[k*N+(k%N)] = 9 // sparse outliers
	}
	h = make([]float32, K)
	for k := range h {
		h[k] = float32(0.05 + r.ExpFloat64())
	}
	return b, h
}

func weightedErr(b, bHat, h []float32, K, N int) float64 {
	var e float64
	for k := range K {
		for n := range N {
			d := float64(b[k*N+n] - bHat[k*N+n])
			e += float64(h[k]) * d * d
		}
	}
	return e
}

// TestQuantizeLearnedLUTB checks the learned slice LUTs against
// Quantize3BitLUTB on the same operand layout and runs the Rubin LUT GEMM
// kernel under cudahostemu with the learned operands.
func TestQuantizeLearnedLUTB(t *testing.T) {
	const K, N = 64, 128
	b, h := lutTestMatrix(K, N, 4)
	idx, lut, sc, deq, err := QuantizeLearnedLUTB(b, h, K, N, 100)
	if err != nil {
		t.Fatal(err)
	}
	_, _, baseSc, baseDeq, err := Quantize3BitLUTB(b, K, N)
	if err != nil {
		t.Fatal(err)
	}
	for i := range sc {
		if sc[i] != baseSc[i] {
			t.Fatalf("scale[%d] = %d, Quantize3BitLUTB has %d", i, sc[i], baseSc[i])
		}
	}
	learned, base := weightedErr(b, deq, h, K, N), weightedErr(b, baseDeq, h, K, N)
	t.Logf("Hessian-weighted error learned %.4g, Quantize3BitLUTB %.4g (ratio %.2f)", learned, base, base/learned)
	if !(learned < base) {
		t.Fatalf("learned weighted error %v not below Quantize3BitLUTB %v", learned, base)
	}

	k := TCGen05LUTDecompressGEMM{Name: "learned_lut_gemm", K: K, Target: "sm_107a"}
	r := rand.New(rand.NewPCG(28, 5))
	a := make([]float32, 128*K)
	for i := range a {
		a[i] = float32(r.NormFloat64())
	}
	aE4M3, aScales := QuantizeMXFP8E4M3(a)
	want, err := k.Reference(aE4M3, aScales, idx, lut, sc)
	if err != nil {
		t.Fatal(err)
	}
	src, err := k.CUDA()
	if err != nil {
		t.Fatal(err)
	}
	res := runEmu(t, src,
		"eg_emu_launch(dim3(1), dim3(128), 0, learned_lut_gemm, a_e4m3, a_scales, b_indices, b_lut_e4m3, b_scales, out)",
		emuBuf{"a_e4m3", "uint8_t", aE4M3},
		emuBuf{"a_scales", "uint8_t", aScales},
		emuBuf{"b_indices", "uint8_t", idx},
		emuBuf{"b_lut_e4m3", "uint8_t", lut},
		emuBuf{"b_scales", "uint8_t", sc},
		emuBuf{"out", "float", poison(4 * 128 * 128)},
	)
	closeF32(t, "out", bytesF32(res["out"]), want, 1e-4)

	if _, _, _, _, err := QuantizeLearnedLUTB(b, h[:K-1], K, N, 10); err == nil {
		t.Error("short hessianDiag accepted")
	}
}

func ExampleFitBlock32LearnedLUT() {
	var w, h [32]float32
	for i := range w {
		w[i] = float32(i%8) - 3.5 // eight distinct levels
		h[i] = 1
	}
	w[0] = 12 // outlier
	b, err := FitBlock32LearnedLUT(w, h, 50)
	if err != nil {
		panic(err)
	}
	u := UniformLUTBlock32(w)
	fmt.Printf("learned mse %.4f, uniform mse %.4f\n", b.WeightedMSE, u.WeightedMSE)
	p := b.PackIndices()
	fmt.Printf("scale 2^%d, packed % x\n", int(b.ScaleUE8M0)-UE8M0Bias, p[:3])
	// Output:
	// learned mse 0.0625, uniform mse 1.0391
	// scale 2^-5, packed 8f c6 da
}
