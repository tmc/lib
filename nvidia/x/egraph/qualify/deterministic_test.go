package qualify

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func TestPhilox4x32KnownAnswers(t *testing.T) {
	// Random123 kat_vectors, philox4x32 with 10 rounds.
	tests := []struct {
		ctr  [4]uint32
		key  [2]uint32
		want [4]uint32
	}{
		{[4]uint32{}, [2]uint32{}, [4]uint32{0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8}},
		{[4]uint32{^uint32(0), ^uint32(0), ^uint32(0), ^uint32(0)}, [2]uint32{^uint32(0), ^uint32(0)}, [4]uint32{0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd}},
		{[4]uint32{0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344}, [2]uint32{0xa4093822, 0x299f31d0}, [4]uint32{0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1}},
	}
	for _, tt := range tests {
		if got := Philox4x32(tt.ctr, tt.key, 10); got != tt.want {
			t.Errorf("Philox4x32(%#x, %#x, 10) = %#x, want %#x", tt.ctr, tt.key, got, tt.want)
		}
	}
}

func TestStochasticRoundE2M1(t *testing.T) {
	tests := []struct {
		v    float32
		r    uint8
		want uint8
	}{
		{0, 255, 0},
		{0.5, 255, 1},
		{6, 255, 7},
		{3, 200, 5},
		{0.75, 127, 1}, // discarded 0x80 + 0x7f does not carry
		{0.75, 128, 2}, // carries: rounds up to 1.0
		{-0.75, 128, 8 | 2},
		{5, 127, 6}, // 5 is halfway between 4 and 6
		{5, 128, 7},
		{7, 0, 7}, // satfinite
		{-100, 0, 15},
		{0.1, 0, 0},
		{0.1, 255, 1},
		{float32(math.NaN()), 0, 7},
	}
	for _, tt := range tests {
		if got := stochasticRoundE2M1(tt.v, tt.r); got != tt.want {
			t.Errorf("stochasticRoundE2M1(%v, %d) = %#x, want %#x", tt.v, tt.r, got, tt.want)
		}
	}
	// Over all 256 random-bit values the mean of the rounded result is the
	// input, for inputs whose discarded fraction fits in 8 bits.
	for _, v := range []float32{0.25, 0.75, 1.125, 2.5, 3.75, 4.5, 5.25, -2.25} {
		var sum float64
		for r := range 256 {
			sum += float64(DecodeE2M1(stochasticRoundE2M1(v, uint8(r))))
		}
		if mean := sum / 256; mean != float64(v) {
			t.Errorf("mean rounding of %v = %v, want unbiased", v, mean)
		}
	}
}

func TestStochasticRoundE2M1x4Packing(t *testing.T) {
	// Each operand takes its own byte of rbits; a value with discarded
	// fraction 0x80 rounds up only when its byte is >= 0x80.
	v := [4]float32{0.75, 0.75, 0.75, 0.75}
	tests := []struct {
		rbits uint32
		want  uint16
	}{
		{0x00000000, 0x1111},
		{0x80000000, 0x2111}, // a
		{0x00008000, 0x1211}, // b
		{0x00800000, 0x1121}, // e
		{0x00000080, 0x1112}, // f
	}
	for _, tt := range tests {
		if got := StochasticRoundE2M1x4(v, tt.rbits); got != tt.want {
			t.Errorf("StochasticRoundE2M1x4(0.75s, %#08x) = %#04x, want %#04x", tt.rbits, got, tt.want)
		}
	}
}

func TestAuditPTXDeterminism(t *testing.T) {
	cfg := DeterministicConfig{GlobalSeed: 0x1234_5678_9abc_def0}
	ptx, err := EmitDeterministicStochasticRoundPTX(cfg, "qd_sr")
	if err != nil {
		t.Fatal(err)
	}
	cvt := "cvt.rs.satfinite.e2m1x4.f32 %h, {%f0, %f1, %f2, %f3}, %r13;"
	if !strings.Contains(ptx, cvt) {
		t.Fatalf("emitted PTX lacks %q", cvt)
	}
	tests := []struct {
		name string
		ptx  string
		want []string // rules, in order
	}{
		{"emitted", ptx, nil},
		{"fp atomic add", ptx + "\natom.global.add.f32 %f0, [%rd0], %f1;", []string{RuleNoFPAtomics}},
		{"fp red bf16x2", ptx + "\nred.global.add.noftz.bf16x2 [%rd0], %r1;", []string{RuleNoFPAtomics}},
		{"multimem ld_reduce", ptx + "\nmultimem.ld_reduce.relaxed.sys.global.add.f32 %f0, [%rd0];", []string{RuleNoFPAtomics}},
		{"bulk reduce", ptx + "\ncp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%rd0], [%rd1], 128;", []string{RuleNoFPAtomics}},
		{"integer atomic", ptx + "\natom.global.add.u32 %r1, [%rd0], 1;", nil},
		{"fp max reduce", ptx + "\nmultimem.ld_reduce.relaxed.sys.global.max.f16x2 %r1, [%rd0];", nil},
		{"commented atomic", ptx + "\n// atom.global.add.f32 %f0, [%rd0], %f1;\n/* red.global.add.f32 [%rd0], %f1; */", nil},
		{"timer rbits", strings.Replace(ptx, cvt, "mov.u32 %r30, %globaltimer_lo;\n\txor.b32 %r31, %r30, %r13;\n\t"+strings.Replace(cvt, "%r13;", "%r31;", 1), 1), []string{RulePhiloxSRSeed}},
		{"clock rbits", strings.Replace(ptx, cvt, strings.Replace(cvt, "%r13;", "%clock;", 1), 1), []string{RulePhiloxSRSeed}},
		{"smid into data", ptx + "\nmov.u32 %r1, %smid;\ncvt.rn.f32.u32 %f0, %r1;\nadd.f32 %f1, %f1, %f0;", []string{RuleNondetEntropy, RuleNondetEntropy}},
		{"timeout loop", ptx + "\nmov.u64 %rd6, %globaltimer;\nsub.u64 %rd7, %rd6, %rd5;\nsetp.gt.u64 %p, %rd7, 1000;\n@%p bra DONE;\nst.global.u64 [%rd1], %rd7;", nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			vs, err := AuditPTXDeterminism(tt.ptx, cfg)
			if err != nil {
				t.Fatal(err)
			}
			var got []string
			for _, v := range vs {
				got = append(got, v.Rule)
			}
			if fmt.Sprint(got) != fmt.Sprint(tt.want) {
				t.Errorf("rules = %v, want %v\n%v", got, tt.want, vs)
			}
		})
	}
}

// randomGEMV returns a K-element row and n weight rows with values in [-1, 1).
func randomGEMV(seed uint64, k, n int) ([]float32, [][]float32) {
	r := rand.New(rand.NewPCG(seed, 1))
	x := make([]float32, k)
	for i := range x {
		x[i] = r.Float32()*2 - 1
	}
	w := make([][]float32, n)
	for c := range w {
		w[c] = make([]float32, k)
		for i := range w[c] {
			w[c][i] = r.Float32()*2 - 1
		}
	}
	return x, w
}

func TestBatchInvarianceCheck(t *testing.T) {
	x, w := randomGEMV(12, 1024, 32)
	for i := range x {
		x[i] /= 16 // logits near 1, where softmax rounding is sensitive
	}
	batches := []int{1, 2, 7, 19, 37, 64, 128}
	tests := []struct {
		name  string
		s     Schedule
		agree bool
	}{
		{"fixed tree", FixedSchedule(ReductionPlan{KChunk: 64, SoftmaxChunk: 16}), true},
		{"fixed tree, split varies with batch", ScheduleFunc(func(b int) ReductionPlan {
			return ReductionPlan{KChunk: 64, SplitK: 1 + b%5, SoftmaxChunk: 16}
		}), true},
		{"naive split-K below batch 8", ScheduleFunc(func(b int) ReductionPlan {
			p := ReductionPlan{KChunk: 64, SplitK: 1, Combine: CombineAtomic, SoftmaxChunk: 16}
			if b < 8 {
				p.SplitK = 8
			}
			return p
		}), false},
		{"atomic split, fixed count", FixedSchedule(ReductionPlan{KChunk: 64, SplitK: 4, Combine: CombineAtomic, SoftmaxChunk: 16}), false},
		{"K chunk depends on batch", ScheduleFunc(func(b int) ReductionPlan {
			p := ReductionPlan{KChunk: 128, SoftmaxChunk: 16}
			if b >= 64 {
				p.KChunk = 64
			}
			return p
		}), false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := BatchInvarianceCheck(tt.s, x, w, batches)
			if tt.agree && err != nil {
				t.Fatalf("BatchInvarianceCheck: %v", err)
			}
			if !tt.agree && err == nil {
				t.Fatal("BatchInvarianceCheck passed a batch-dependent schedule")
			}
			if err != nil {
				t.Log(err)
			}
			if vs := AuditSchedule(tt.s, batches); (len(vs) == 0) != tt.agree {
				t.Errorf("AuditSchedule = %v, want violations=%v", vs, !tt.agree)
			}
		})
	}
}

func TestSoftmaxChunkBatchDependence(t *testing.T) {
	// Online softmax is chunk-independent while the running max never
	// changes, so the dependence shows only when larger logits come later.
	z := make([]float32, 32)
	for i := range z {
		z[i] = float32(i)*0.173 + float32(i%3)*0.01
	}
	a, b := onlineSoftmax(z, 32), onlineSoftmax(z, 4)
	differs := false
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			differs = true
		}
	}
	if !differs {
		t.Error("online softmax with chunks of 32 and 4 agreed bit for bit on rising logits")
	}
	s := ScheduleFunc(func(b int) ReductionPlan { return ReductionPlan{KChunk: 64, SoftmaxChunk: 4 + 4*(b%3)} })
	vs := AuditSchedule(s, []int{1, 2, 3})
	if len(vs) != 2 || vs[0].Rule != RuleBatchInvariantTile {
		t.Errorf("AuditSchedule = %v, want 2 %s violations", vs, RuleBatchInvariantTile)
	}
}

func TestFixedTreeSum(t *testing.T) {
	const big = 1 << 24
	tests := []struct {
		in   []float32
		want float32
	}{
		{nil, 0},
		{[]float32{3}, 3},
		{[]float32{1, 2, 3, 4, 5}, 15},
		// ((1+1) + (2^24+1)) = 2^24+2; left to right gives 2^24+4.
		{[]float32{1, 1, big, 1}, big + 2},
	}
	for _, tt := range tests {
		if got := FixedTreeSum(tt.in); got != tt.want {
			t.Errorf("FixedTreeSum(%v) = %v, want %v", tt.in, got, tt.want)
		}
	}
}

func TestFixedTreeSumSplitInvariant(t *testing.T) {
	// The fixed tree depends only on the chunk partials, so every split
	// count gives the same bits; summing splits sequentially does not.
	x, w := randomGEMV(5, 4096, 1)
	parts := chunkSums(x, w[0], 64)
	want := FixedTreeSum(parts)
	differs := false
	for split := 1; split <= 16; split++ {
		p := ReductionPlan{KChunk: 64, SplitK: split}
		if got := planDot(x, w[0], p, 0); math.Float32bits(got) != math.Float32bits(want) {
			t.Errorf("fixed tree, split %d = %#08x, want %#08x", split, math.Float32bits(got), math.Float32bits(want))
		}
		p.Combine = CombineAtomic
		if got := planDot(x, w[0], p, uint64(split)); math.Float32bits(got) != math.Float32bits(want) {
			differs = true
		}
	}
	if !differs {
		t.Error("atomic split-K matched the fixed tree for every split count")
	}
}

func TestFilterDeterministicCandidates(t *testing.T) {
	leaf := &egraph.Expr{Op: "x"}
	cands := []egraph.CandidateAST{
		{Expr: &egraph.Expr{Op: "tile_mma", Children: []*egraph.Expr{leaf}}, Cost: 2},
		{Expr: &egraph.Expr{Op: "add", Children: []*egraph.Expr{{Op: "splitk_atomic", Children: []*egraph.Expr{leaf}}}}, Cost: 1},
		{Expr: &egraph.Expr{Op: "multimem_ld_reduce"}, Cost: 0.5},
	}
	got := FilterDeterministicCandidates(cands)
	if len(got) != 1 || got[0].Expr.Op != "tile_mma" {
		t.Fatalf("FilterDeterministicCandidates kept %v, want only tile_mma", got)
	}
}

func TestEmitDeterministicStochasticRoundPTXMatchesReference(t *testing.T) {
	const tokens, cols = 3, 16
	in := make([]float32, tokens*cols)
	for i := range in {
		in[i] = float32(i%13)*0.37 - 2.1
	}
	for _, rounds := range []int{7, 10} {
		cfg := DeterministicConfig{GlobalSeed: 0xfeed_beef_0bad_cafe, PhiloxRounds: rounds}
		ptx, err := EmitDeterministicStochasticRoundPTX(cfg, "qd_sr")
		if err != nil {
			t.Fatal(err)
		}
		if n := strings.Count(ptx, "mul.hi.u32"); n != 2*rounds {
			t.Errorf("rounds=%d: %d mul.hi.u32, want %d", rounds, n, 2*rounds)
		}
		const layer, token0 = 9, 1000
		// Run every thread of two launch shapes: 1x12 and 4x4 (with spares).
		for _, shape := range [][2]uint64{{1, 12}, {4, 4}} {
			sim := &ptxSim{
				params: map[string]uint64{"p_in": 0x1000, "p_out": 0x8000, "p_layer": layer, "p_token0": token0, "p_tokens": tokens, "p_cols": cols},
				mem:    map[uint64]uint32{},
				out:    map[uint64]uint16{},
			}
			for i, v := range in {
				sim.mem[0x1000+4*uint64(i)] = math.Float32bits(v)
			}
			for cta := range shape[0] {
				for tid := range shape[1] {
					sim.regs = map[string]uint64{"%ctaid.x": cta, "%ntid.x": shape[1], "%tid.x": tid}
					sim.run(t, ptx)
				}
			}
			for i := range tokens * cols / 4 {
				token, col := uint32(i/(cols/4)), uint32(i%(cols/4)*4)
				rb := StochasticRoundBits(cfg, layer, token0+token, col)
				want := StochasticRoundE2M1x4([4]float32(in[int(token)*cols+int(col):]), rb)
				got, ok := sim.out[0x8000+2*uint64(i)]
				if !ok || got != want {
					t.Errorf("rounds=%d shape=%v group %d: got %#04x (written %v), want %#04x", rounds, shape, i, got, ok, want)
				}
			}
		}
	}
}

func f32Bytes(v []float32) []byte {
	b := make([]byte, 4*len(v))
	for i, f := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(f))
	}
	return b
}

func bytesF32(b []byte) []float32 {
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return v
}

func TestDeterministicCUDAHostEmu(t *testing.T) {
	if _, err := cudahostemu.Compiler(); err != nil {
		t.Skip(err)
	}
	cfg := DeterministicConfig{GlobalSeed: 42}
	src, err := DeterministicCUDA(cfg)
	if err != nil {
		t.Fatal(err)
	}

	const tokens, cols = 4, 32
	in := make([]float32, tokens*cols)
	for i := range in {
		in[i] = float32(i%23)*0.29 - 3.3
	}
	want := make([]uint16, tokens*cols/4)
	for i := range want {
		tok, col := i/(cols/4), i%(cols/4)*4
		want[i] = StochasticRoundE2M1x4([4]float32(in[tok*cols+col:]), StochasticRoundBits(cfg, 3, uint32(7+tok), uint32(col)))
	}
	for _, block := range []int{32, 5} {
		grid := (len(want) + block - 1) / block
		out, err := cudahostemu.Run(t.Context(), t.TempDir(), cudahostemu.Program{
			Source: src,
			Launch: fmt.Sprintf("eg_emu_launch(dim3(%d), dim3(%d), 0, qd_stochastic_round, in, out, 3u, 7u, %du, %du)", grid, block, tokens, cols),
			Buffers: []cudahostemu.Buffer{
				{Name: "in", CType: "float", Data: f32Bytes(in)},
				{Name: "out", CType: "uint16_t", Data: cudahostemu.Poison(2 * len(want))},
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		for i, w := range want {
			if got := binary.LittleEndian.Uint16(out["out"][2*i:]); got != w {
				t.Errorf("block %d: group %d = %#04x, want %#04x", block, i, got, w)
			}
		}
	}

	const rows, n, k, kchunk = 2, 8, 1024, 64
	x, w := randomGEMV(77, rows*k, n)
	wFlat := make([]float32, 0, n*rows*k)
	for _, r := range w {
		wFlat = append(wFlat, r[:k]...)
	}
	ref := make([]float32, rows*n)
	for r := range rows {
		for c := range n {
			ref[r*n+c] = FixedTreeSum(chunkSums(x[r*k:(r+1)*k], wFlat[c*k:(c+1)*k], kchunk))
		}
	}
	run := func(kernel string, split int) []float32 {
		out, err := cudahostemu.Run(t.Context(), t.TempDir(), cudahostemu.Program{
			Source: src,
			Launch: fmt.Sprintf("eg_emu_launch(dim3(%d, %d), dim3(%d), 0, %s, x, w, y, %d, %d, %d)", n, rows, split, kernel, k, n, kchunk),
			Buffers: []cudahostemu.Buffer{
				{Name: "x", CType: "float", Data: f32Bytes(x)},
				{Name: "w", CType: "float", Data: f32Bytes(wFlat)},
				{Name: "y", CType: "float", Data: make([]byte, 4*rows*n)},
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		return bytesF32(out["y"])
	}
	for _, split := range []int{1, 3, 16} {
		got := run("qd_det_gemv", split)
		for i := range ref {
			if math.Float32bits(got[i]) != math.Float32bits(ref[i]) {
				t.Errorf("qd_det_gemv split %d: y[%d] = %#08x, want %#08x", split, i, math.Float32bits(got[i]), math.Float32bits(ref[i]))
			}
		}
	}
	naive := run("qd_naive_splitk_gemv", 8)
	same := true
	for i := range ref {
		if math.Float32bits(naive[i]) != math.Float32bits(ref[i]) {
			same = false
		}
		if d := math.Abs(float64(naive[i] - ref[i])); d > 1e-3 {
			t.Errorf("qd_naive_splitk_gemv y[%d] = %v, far from %v", i, naive[i], ref[i])
		}
	}
	if same {
		t.Error("naive split-K reduction matched the fixed tree bit for bit in every output")
	}
}

func ExamplePhilox4x32() {
	fmt.Printf("%#x\n", Philox4x32([4]uint32{}, [2]uint32{}, 10))
	cfg := DeterministicConfig{GlobalSeed: 1}
	fmt.Printf("%#04x\n", StochasticRoundE2M1x4([4]float32{0.75, 1, -2.5, 7}, StochasticRoundBits(cfg, 0, 0, 0)))
	// Output:
	// [0x6627e8d5 0xe169c58d 0xbc57ac4c 0x9b00dbd8]
	// 0x22c7
}
