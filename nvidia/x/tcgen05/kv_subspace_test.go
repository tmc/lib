package tcgen05

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatmem"
)

// kvSubspace is a fixed random rank-r0 mixing matrix [r0, d].
type kvSubspace struct {
	r0, d int
	a     []float64
}

func newKVSubspace(r *rand.Rand, r0, d int) kvSubspace {
	a := make([]float64, r0*d)
	for i := range a {
		a[i] = r.NormFloat64() / math.Sqrt(float64(r0))
	}
	return kvSubspace{r0, d, a}
}

// lowRankKV returns tokens rows of width d drawn from a fresh rank-r0
// subspace plus isotropic noise of standard deviation noise.
func lowRankKV(r *rand.Rand, tokens, d, r0 int, noise float64) []float32 {
	return newKVSubspace(r, r0, d).rows(r, tokens, noise)
}

// rows returns tokens rows from s plus isotropic noise.
func (s kvSubspace) rows(r *rand.Rand, tokens int, noise float64) []float32 {
	r0, d, a := s.r0, s.d, s.a
	x := make([]float32, tokens*d)
	z := make([]float64, r0)
	for t := range tokens {
		for j := range z {
			// Decaying spectrum across the r0 directions.
			z[j] = r.NormFloat64() * math.Exp(-float64(j)/float64(r0))
		}
		for i := range d {
			s := noise * r.NormFloat64()
			for j := range r0 {
				s += z[j] * a[j*d+i]
			}
			x[t*d+i] = float32(s)
		}
	}
	return x
}

func TestRowOrthogonalize(t *testing.T) {
	r := rand.New(rand.NewPCG(30, 1))
	for _, sh := range []struct{ rows, cols int }{{6, 10}, {10, 10}, {12, 5}} {
		m := make([]float64, sh.rows*sh.cols)
		for i := range m {
			m[i] = r.NormFloat64()
		}
		gram := func(m []float64) []float64 {
			g := make([]float64, sh.cols*sh.cols)
			for k := range sh.rows {
				for i := range sh.cols {
					for j := range sh.cols {
						g[i*sh.cols+j] += m[k*sh.cols+i] * m[k*sh.cols+j]
					}
				}
			}
			return g
		}
		before := gram(m)
		rowOrthogonalize(m, sh.rows, sh.cols)
		after := gram(m)
		for i := range before {
			if math.Abs(before[i]-after[i]) > 1e-9 {
				t.Fatalf("%dx%d: M^T M changed at %d: %v -> %v", sh.rows, sh.cols, i, before[i], after[i])
			}
		}
		for i := range sh.rows {
			for j := i + 1; j < sh.rows; j++ {
				ri, rj := m[i*sh.cols:(i+1)*sh.cols], m[j*sh.cols:(j+1)*sh.cols]
				if g := dotF64(ri, rj); math.Abs(g) > 1e-9*math.Sqrt(dotF64(ri, ri)*dotF64(rj, rj))+1e-12 {
					t.Fatalf("%dx%d: rows %d and %d not orthogonal: %g", sh.rows, sh.cols, i, j, g)
				}
			}
		}
	}
}

func TestCalibrateSubspaceBasis(t *testing.T) {
	tests := []struct {
		name                    string
		tokens, heads, dim, rnk int
	}{
		{"covariance", 256, 2, 32, 16}, // D=64 <= tokens: D x D Jacobi
		{"gram", 48, 4, 32, 32},        // D=128 > tokens: Gram path
		{"complete", 8, 2, 16, 16},     // rank > sample span: completion
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := rand.New(rand.NewPCG(30, uint64(tt.tokens)))
			d := tt.heads * tt.dim
			k := lowRankKV(r, tt.tokens, d, min(tt.rnk, tt.tokens), 0.01)
			v := lowRankKV(r, tt.tokens, d, min(tt.rnk, tt.tokens), 0.01)
			p, err := CalibrateSubspaceBasis(k, v, tt.tokens, tt.heads, tt.dim, tt.rnk)
			if err != nil {
				t.Fatal(err)
			}
			if err := p.Validate(); err != nil {
				t.Fatal(err)
			}
			if p.ExplainedVarK < 0.99 || p.ExplainedVarV < 0.99 || p.ExplainedVarK > 1+1e-9 {
				t.Fatalf("explained variance K=%.5f V=%.5f, want >= 0.99", p.ExplainedVarK, p.ExplainedVarV)
			}
		})
	}
}

// TestCalibrateMatchesCovariancePath checks that the Gram-matrix path and
// the covariance path span the same subspace.
func TestCalibrateMatchesCovariancePath(t *testing.T) {
	r := rand.New(rand.NewPCG(30, 2))
	const tokens, d, rank = 40, 48, 8
	x := lowRankKV(r, tokens, d, rank, 0.001)
	gram, _ := principalBasis(x, tokens, d, rank)
	// Repeat rows so the covariance path (d <= t) is taken.
	x2 := slices.Concat(x, x)
	cov, _ := principalBasis(x2, 2*tokens, d, rank)
	// ||Pg - Pc||_F for the projectors U U^T.
	var e float64
	for i := range d {
		for j := range d {
			var pg, pc float64
			for k := range rank {
				pg += float64(gram[i*rank+k]) * float64(gram[j*rank+k])
				pc += float64(cov[i*rank+k]) * float64(cov[j*rank+k])
			}
			e += (pg - pc) * (pg - pc)
		}
	}
	if math.Sqrt(e) > 1e-3 {
		t.Fatalf("Gram and covariance subspaces differ: ||Pg - Pc||_F = %g", math.Sqrt(e))
	}
}

// testProjector calibrates a projector on tokens rows from fixed K and V
// subspaces of rank 3/4 of rank and returns those subspaces for drawing
// page rows.
func testProjector(t *testing.T, seed uint64, tokens, heads, hd, rank int) (p SubspaceKVProjector, r *rand.Rand, ks, vs kvSubspace) {
	t.Helper()
	r = rand.New(rand.NewPCG(30, seed))
	d := heads * hd
	ks, vs = newKVSubspace(r, rank*3/4, d), newKVSubspace(r, rank*3/4, d)
	p, err := CalibrateSubspaceBasis(ks.rows(r, tokens, 0.02), vs.rows(r, tokens, 0.02), tokens, heads, hd, rank)
	if err != nil {
		t.Fatal(err)
	}
	return p, r, ks, vs
}

// TestAbsorbedLatentAttentionIdentity checks that attention in the latent
// space equals attention over the explicitly reconstructed K and V, and that
// a non-orthonormal basis or swapped head slices break the identity.
func TestAbsorbedLatentAttentionIdentity(t *testing.T) {
	const heads, hd, rank, qHeads, tokens = 4, 32, 64, 8, 64
	p, r, ks, vs := testProjector(t, 3, 200, heads, hd, rank)
	k := ks.rows(r, tokens, 0.02)
	v := vs.rows(r, tokens, 0.02)
	q := make([]float32, qHeads*hd)
	for i := range q {
		q[i] = float32(r.NormFloat64())
	}
	page, err := p.CompressKVPage(k, v, tokens, 7, false)
	if err != nil {
		t.Fatal(err)
	}
	ls, lo, err := p.AbsorbedLatentAttention(q, qHeads, page)
	if err != nil {
		t.Fatal(err)
	}
	kHat, vHat := p.DecompressKVPage(page)
	es, eo, err := GQAAttention(q, kHat, vHat, tokens, qHeads, heads, hd)
	if err != nil {
		t.Fatal(err)
	}
	maxDiff := func(a, b []float32) float64 {
		var m float64
		for i := range a {
			m = max(m, math.Abs(float64(a[i]-b[i])))
		}
		return m
	}
	if d := max(maxDiff(ls, es), maxDiff(lo, eo)); d > 1e-5 {
		t.Fatalf("absorbed vs explicit max diff %g, want < 1e-5", d)
	}

	// Mutant 1: perturb U_K so it is no longer orthonormal. Validate must
	// reject it.
	bad := p
	bad.BasisK = slices.Clone(p.BasisK)
	bad.BasisK[0] += 0.05
	if err := bad.Validate(); err == nil {
		t.Fatal("Validate accepted a non-orthonormal BasisK")
	}
	if _, err := bad.CompressKVPage(k, v, tokens, 0, false); err == nil {
		t.Fatal("CompressKVPage accepted a non-orthonormal BasisK")
	}

	// Mutant 2: swap the row blocks of KV heads 0 and 1 in U_K after
	// compression. The basis is still orthonormal, but the absorbed query
	// uses the wrong head slice and no longer matches attention over the
	// reconstructed keys.
	swapped := p
	swapped.BasisK = slices.Clone(p.BasisK)
	for i := range hd * rank {
		swapped.BasisK[i], swapped.BasisK[hd*rank+i] = p.BasisK[hd*rank+i], p.BasisK[i]
	}
	if err := swapped.Validate(); err != nil {
		t.Fatalf("head-swapped basis should stay orthonormal: %v", err)
	}
	ms, _, err := swapped.AbsorbedLatentAttention(q, qHeads, page)
	if err != nil {
		t.Fatal(err)
	}
	if d := maxDiff(ms, es); d < 1e-2 {
		t.Fatalf("head-swapped absorbed scores match explicit scores (max diff %g)", d)
	}
}

// TestCompressKVPageLUT checks the compound compression ratio and the
// attention output SQNR against uncompressed attention on low-rank-plus-noise
// KV rows with D=512 and rank 128. Rank projection alone (BF16 latents, 4x)
// keeps the output above 26 dB; adding 3-bit LUT latents (12.2x) costs about
// 12 dB, the quantization noise of 8 levels per 32-value block.
func TestCompressKVPageLUT(t *testing.T) {
	const heads, hd, rank, qHeads, tokens = 4, 128, 128, 16, 64
	p, r, ks, vs := testProjector(t, 4, 256, heads, hd, rank)
	d := heads * hd
	k := ks.rows(r, tokens, 0.02)
	v := vs.rows(r, tokens, 0.02)
	q := make([]float32, qHeads*hd)
	for i := range q {
		q[i] = float32(r.NormFloat64())
	}
	_, want, err := GQAAttention(q, k, v, tokens, qHeads, heads, hd)
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		lut       bool
		wantRatio float64
		minSQNR   float64
	}{
		{false, 4, 26},
		{true, 12.19, 14},
	}
	for _, tt := range tests {
		page, err := p.CompressKVPage(k, v, tokens, 0, tt.lut)
		if err != nil {
			t.Fatal(err)
		}
		ratio := p.CompressionRatio(page)
		if math.Abs(ratio-tt.wantRatio) > 0.01 {
			t.Errorf("lut=%v: compression ratio %.2f, want %.2f", tt.lut, ratio, tt.wantRatio)
		}
		_, got, err := p.AbsorbedLatentAttention(q, qHeads, page)
		if err != nil {
			t.Fatal(err)
		}
		sqnr := sqnrDB(want, got)
		t.Logf("lut=%v: compression %.2fx (%d bytes vs %d BF16), attention output SQNR %.1f dB", tt.lut, ratio, page.Bytes(), 2*tokens*d*2, sqnr)
		if sqnr < tt.minSQNR {
			t.Errorf("lut=%v: attention output SQNR %.1f dB, want >= %.0f", tt.lut, sqnr, tt.minSQNR)
		}
	}
	if _, err := p.CompressKVPage(k[:48*d], v[:48*d], 48, 0, true); err == nil {
		t.Error("LUT compression of a 48-token page accepted")
	}
}

func TestKVCompressSchedule(t *testing.T) {
	s := KVCompressSchedule{
		PageTokens:      64,
		HotWindowTokens: 256,
		Partition:       KVCompressorPartition{TotalSMs: 148, BackgroundSMs: 8, SMGranularity: 2},
	}
	tests := []struct {
		seqLen, done int
		want         []int
	}{
		{100, 0, nil},
		{256 + 63, 0, nil},
		{256 + 64, 0, []int{0}},
		{256 + 64*3 + 10, 0, []int{0, 1, 2}},
		{256 + 64*3 + 10, 2, []int{2}},
		{256 + 64*12, 0, []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}},
	}
	for _, tt := range tests {
		tasks, err := s.Plan(tt.seqLen, tt.done)
		if err != nil {
			t.Fatal(err)
		}
		var ids []int
		for _, task := range tasks {
			ids = append(ids, task.PageID)
			if task.FirstToken+s.PageTokens > tt.seqLen-s.HotWindowTokens {
				t.Fatalf("seqLen %d: page %d overlaps the hot window", tt.seqLen, task.PageID)
			}
			if task.SM != task.PageID%8 {
				t.Fatalf("page %d on SM %d, want %d", task.PageID, task.SM, task.PageID%8)
			}
		}
		if !slices.Equal(ids, tt.want) {
			t.Errorf("Plan(%d, %d) pages = %v, want %v", tt.seqLen, tt.done, ids, tt.want)
		}
	}
	if s.Partition.DecodeSMs() != 140 {
		t.Errorf("DecodeSMs = %d, want 140", s.Partition.DecodeSMs())
	}
	for _, bad := range []KVCompressorPartition{
		{TotalSMs: 148, BackgroundSMs: 0, SMGranularity: 2},
		{TotalSMs: 148, BackgroundSMs: 148, SMGranularity: 2},
		{TotalSMs: 148, BackgroundSMs: 7, SMGranularity: 2},
		{TotalSMs: 148, BackgroundSMs: 8, SMGranularity: 0},
	} {
		if err := bad.Validate(); err == nil {
			t.Errorf("Validate(%+v) = nil, want error", bad)
		}
	}
}

func TestTCGen05KVSubspaceProject(t *testing.T) {
	if err := (TCGen05KVSubspaceProject{Dim: 48}).Validate(); err == nil {
		t.Error("Dim=48 accepted")
	}
	if err := (TCGen05KVSubspaceProject{Dim: 64, Rank: 256}).Validate(); err == nil {
		t.Error("Rank=256 accepted")
	}
	k := TCGen05KVSubspaceProject{Name: "kv_subspace_emu", Dim: 96, Rank: 64}
	ev := k.TraceEvents()
	if _, err := cudatmem.VerifyTrace(cudatmem.SM100A, ev); err != nil {
		t.Fatalf("VerifyTrace: %v", err)
	}
	if _, err := cudatmem.VerifyTrace(cudatmem.SM100A, ev[:len(ev)-1]); err == nil {
		t.Fatal("VerifyTrace accepted a trace without dealloc")
	}

	p, r, ks, _ := testProjector(t, 5, 256, 3, 32, 64)
	x := ks.rows(r, 128, 0.02)
	want, err := k.Reference(x, p.BasisK)
	if err != nil {
		t.Fatal(err)
	}
	lat := projectRows(x, p.BasisK, 128, 96, 64)
	closeF32(t, "Reference vs projectRows", want, lat, 1e-4)
	src, err := k.CUDA()
	if err != nil {
		t.Fatal(err)
	}
	res := runEmu(t, src,
		"eg_emu_launch(dim3(1), dim3(128), 0, kv_subspace_emu, x, basis, out)",
		emuBuf{"x", "float", f32Bytes(x)},
		emuBuf{"basis", "float", f32Bytes(p.BasisK)},
		emuBuf{"out", "float", poison(4 * 128 * 64)},
	)
	got := bytesF32(res["out"])
	closeF32(t, "out", got, want, 1e-4)

	// Mutant: a transposed basis (reading U as [Rank, Dim]) gives a different projection.
	bt := make([]float32, len(p.BasisK))
	for i := range 96 {
		for j := range 64 {
			bt[j*96+i] = p.BasisK[i*64+j]
		}
	}
	mut, _ := k.Reference(x, bt)
	var maxDiff float64
	for i := range got {
		maxDiff = max(maxDiff, math.Abs(float64(got[i]-mut[i])))
	}
	if maxDiff < 1e-2 {
		t.Fatalf("transposed-basis mutant matches kernel output")
	}
}

func ExampleKVCompressSchedule_Plan() {
	s := KVCompressSchedule{
		PageTokens:      64,
		HotWindowTokens: 256,
		Partition:       KVCompressorPartition{TotalSMs: 148, BackgroundSMs: 8, SMGranularity: 2},
	}
	tasks, err := s.Plan(600, 1)
	if err != nil {
		panic(err)
	}
	for _, task := range tasks {
		fmt.Printf("page %d tokens [%d, %d) on background SM %d\n", task.PageID, task.FirstToken, task.FirstToken+64, task.SM)
	}
	// Output:
	// page 1 tokens [64, 128) on background SM 1
	// page 2 tokens [128, 192) on background SM 2
	// page 3 tokens [192, 256) on background SM 3
	// page 4 tokens [256, 320) on background SM 4
}
