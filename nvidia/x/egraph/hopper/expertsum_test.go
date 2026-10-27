package hopper

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

// sumSquares returns y's sums of squares per 32 columns, as TileGEMM's SumSq.
func sumSquares(y []float32) []float32 {
	ss := make([]float32, len(y)/32)
	for i, v := range y {
		ss[i/32] += v * v
	}
	return ss
}

func i32s(raw []byte) []int32 {
	v := make([]int32, len(raw)/4)
	for i := range v {
		v[i] = int32(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	return v
}

func TestExpertSumEmulated(t *testing.T) {
	tests := []struct {
		name   string
		s      ExpertSum
		tokens int
	}{
		{"bf16_k8", ExpertSum{Name: "esum_bf16", Width: 4096, K: 8, Y: BF16, Out: BF16}, 5},
		{"f32_k3", ExpertSum{Name: "esum_f32", Width: 96, K: 3, Y: F32, Out: F32}, 4},
		{"k12_two_rounds", ExpertSum{Name: "esum_k12", Width: 2304, K: 12, Y: BF16, Out: F32}, 3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := rand.New(rand.NewPCG(5, uint64(tt.s.K)))
			s, n := tt.s, tt.tokens
			// Routes take distinct rows in shuffled order; about a third are
			// absent and token 0 has none.
			rows := n * s.K
			perm := r.Perm(rows)
			slots := make([]int32, rows)
			weights := make([]float32, rows)
			for i := range slots {
				slots[i] = int32(perm[i])
				if i < s.K || r.IntN(3) == 0 {
					slots[i] = -1
				}
				weights[i] = float32(r.Float64())
			}
			y := make([]float32, rows*s.Width)
			for i := range y {
				y[i] = s.Y.Round(float32(r.NormFloat64() * 3))
			}
			ss := sumSquares(y)
			src, err := s.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			grid, block := s.Grid(n)
			res := runEmu(t, src, fmt.Sprintf("eg_emu_launch(dim3(%d), dim3(%d), 0, %s, %d, y, ss, slots, weights, out)", grid[0], block[0], s.Name, n),
				emuBuf{"y", s.Y.cType(), typed{s.Y, f64s(y)}.bytes()},
				emuBuf{"ss", "float", f32Bytes(ss)},
				emuBuf{"slots", "int32_t", i32Bytes(slots)},
				emuBuf{"weights", "float", f32Bytes(weights)},
				emuBuf{"out", s.Out.cType(), poison(n * s.Width * s.Out.Bits() / 8)})
			want, err := s.Reference(n, y, ss, slots, weights)
			if err != nil {
				t.Fatal(err)
			}
			got := make([]float32, len(want))
			for i, v := range decode(s.Out, res["out"]) {
				got[i] = float32(v)
			}
			rel := 1e-5
			if s.Out == BF16 {
				rel = 1.0 / 128
			}
			if i, ok := closeTo(got, want, rel, 1e-6); !ok {
				t.Fatalf("out[%d] = %g, want %g", i, got[i], want[i])
			}
			for i, v := range got[:s.Width] {
				if v != 0 {
					t.Fatalf("token 0 (no local route) out[%d] = %g, want 0", i, v)
				}
			}
		})
	}
}

// TestExpertSumFormula checks Reference against reference.py's
// post_expert_norm and weighted sum computed directly in float64.
func TestExpertSumFormula(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	s := ExpertSum{Name: "e", Width: 64, K: 4, Y: F32, Out: F32}
	const tokens = 3
	slots := []int32{5, -1, 0, 9, -1, -1, -1, -1, 2, 7, 1, -1}
	weights := make([]float32, len(slots))
	for i := range weights {
		weights[i] = float32(r.Float64())
	}
	y := make([]float32, 10*s.Width)
	for i := range y {
		y[i] = float32(r.NormFloat64())
	}
	got, err := s.Reference(tokens, y, sumSquares(y), slots, weights)
	if err != nil {
		t.Fatal(err)
	}
	want := make([]float32, tokens*s.Width)
	for i, row := range slots {
		if row < 0 {
			continue
		}
		v := y[int(row)*s.Width:][:s.Width]
		var ms float64
		for _, x := range v {
			ms += float64(x) * float64(x) / float64(s.Width)
		}
		inv := 1 / math.Sqrt(ms+1e-8)
		for c, x := range v {
			want[(i/s.K)*s.Width+c] += float32(float64(x) * inv * float64(weights[i]))
		}
	}
	if i, ok := closeTo(got, want, 1e-5, 1e-6); !ok {
		t.Fatalf("out[%d] = %g, want %g", i, got[i], want[i])
	}
}

func TestExpertSumErrors(t *testing.T) {
	ok := ExpertSum{Name: "e", Width: 64, K: 8, Y: BF16, Out: BF16}
	tests := []struct {
		name string
		edit func(*ExpertSum)
	}{
		{"name", func(s *ExpertSum) { s.Name = "" }},
		{"width", func(s *ExpertSum) { s.Width = 48 }},
		{"k", func(s *ExpertSum) { s.K = 33 }},
		{"y", func(s *ExpertSum) { s.Y = E4M3 }},
		{"out", func(s *ExpertSum) { s.Out = 0 }},
	}
	if _, err := ok.CUDA(); err != nil {
		t.Fatal(err)
	}
	for _, tt := range tests {
		s := ok
		tt.edit(&s)
		if _, err := s.CUDA(); err == nil {
			t.Errorf("%s: no error", tt.name)
		}
	}
}

// TestMoEDownChainEmulated runs the generated routing and MoE down
// projection chain, RowTopK → ExpertAlign → TileGEMM (grouped NVFP4,
// (1 + post_w) epilogue, SumSq) → ExpertSum, and checks it against
// reference.py's loop over local experts in float64
// (moe8_moe_w4a16/reference.py:88-90, 98-106).
func TestMoEDownChainEmulated(t *testing.T) {
	const (
		tokens, experts, local, start = 6, 32, 8, 8
		width, dExpert                = 256, 128
	)
	r := rand.New(rand.NewPCG(3, 4))
	route := RowTopK{Name: "route", D: experts, K: 8, X: F32, Bias: F32,
		Score: MustLambda("(sigmoid (mul $0 0.1767766952966369))"), Key: MustLambda("(add $0 $1)"),
		Normalize: true, Eps: 1e-15, Scale: 1}
	align := ExpertAlign{Name: "align", K: 8, Experts: experts, Local: local, Warps: 4}
	down := TileGEMM{Name: "down", N: width, K: dExpert, Weight: WeightNVFP4, Out: BF16, BM: 64, BN: 64,
		Grouped: true, SumSq: true, Epilogue: MustLambda("(mul $0 (add 1 $1))"), Extras: []Operand{{"post_w", BF16, PerCol}}}
	sum := ExpertSum{Name: "esum", Width: width, K: 8, Y: BF16, Out: BF16}

	logits := make([]float32, tokens*experts)
	for i := range logits {
		logits[i] = float32(r.NormFloat64() * 4)
		if i < experts && i >= start && i < start+local {
			logits[i] = -40 // token 0 routes to no local expert
		}
	}
	bias := make([]float32, experts)
	for i := range bias {
		bias[i] = float32(r.NormFloat64() * 0.1)
	}

	// Routing and alignment.
	var src string
	for _, k := range []interface{ CUDA() (string, error) }{route, align, down, sum} {
		s, err := k.CUDA()
		if err != nil {
			t.Fatal(err)
		}
		src += s
	}
	rgrid, rblock := route.Grid(tokens)
	res := runEmu(t, src, fmt.Sprintf("eg_emu_launch(dim3(%d), dim3(%d), 0, route, %d, x, bias, ids, weights); "+
		"eg_emu_launch(dim3(1), dim3(%d), 0, align, %d, ids, local_start, offsets, rows, slots)",
		rgrid[0], rblock[0], tokens, 32*align.Warps, tokens),
		emuBuf{"x", "float", f32Bytes(logits)},
		emuBuf{"bias", "float", f32Bytes(bias)},
		emuBuf{"ids", "int32_t", poison(4 * tokens * 8)},
		emuBuf{"weights", "float", poison(4 * tokens * 8)},
		emuBuf{"local_start", "int32_t", i32Bytes([]int32{start})},
		emuBuf{"offsets", "int32_t", poison(4 * (local + 1))},
		emuBuf{"rows", "int32_t", poison(4 * tokens * 8)},
		emuBuf{"slots", "int32_t", poison(4 * tokens * 8)})
	ids, slots, offsets := i32s(res["ids"]), i32s(res["slots"]), i32s(res["offsets"])
	weights := make([]float32, tokens*8)
	for i, v := range decode(F32, res["weights"]) {
		weights[i] = float32(v)
	}
	wantIDs, wantW, err := route.Reference(tokens, logits, bias)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(ids, wantIDs) {
		t.Fatalf("ids = %v, want %v", ids, wantIDs)
	}
	if i, ok := closeTo(weights, wantW, 1e-6, 1e-7); !ok {
		t.Fatalf("weights[%d] = %g, want %g", i, weights[i], wantW[i])
	}

	// The SwiGLU output each route feeds the down projection, as GEMM1
	// writes it: route i's row at slots[i].
	nrows := int(offsets[local])
	inter := make([][]uint16, tokens*8)
	x := make([]uint16, nrows*dExpert)
	for i, row := range slots {
		inter[i] = randBF16(r, dExpert, 1)
		if row >= 0 {
			copy(x[int(row)*dExpert:], inter[i])
		}
	}
	w := make([]GEMMWeights, local)
	var all GEMMWeights
	var gs []float32
	for e := range w {
		w[e] = randNVFP4(r, width, dExpert)
		w[e].GlobalScal = float32(0.001 + r.Float64()*0.001)
		all.Packed = append(all.Packed, w[e].Packed...)
		all.Scales = append(all.Scales, w[e].Scales...)
		gs = append(gs, w[e].GlobalScal)
	}
	postW := make([]float32, width)
	for i := range postW {
		postW[i] = BF16.Round(float32(r.NormFloat64() * 0.1))
	}
	maxRows := 1
	for e := range local {
		maxRows = max(maxRows, int(offsets[e+1]-offsets[e]))
	}
	dgrid, dblock := down.Grid(maxRows, local)
	res = runEmu(t, src, fmt.Sprintf("eg_emu_launch(dim3(%d, %d, %d), dim3(%d), %d, down, %d, inter, w, w_sf, w_gs, offsets, post_w, y, ss); "+
		"eg_emu_launch(dim3(%d), dim3(256), 0, esum, %d, y, ss, slots, weights, out)",
		dgrid[0], dgrid[1], dgrid[2], dblock[0], down.SharedBytes(), nrows, tokens, tokens),
		emuBuf{"inter", "uint16_t", u16Bytes(x)},
		emuBuf{"w", "uint8_t", all.Packed}, emuBuf{"w_sf", "uint8_t", all.Scales}, emuBuf{"w_gs", "float", f32Bytes(gs)},
		emuBuf{"offsets", "int32_t", i32Bytes(offsets)},
		emuBuf{"post_w", "uint16_t", typed{BF16, f64s(postW)}.bytes()},
		emuBuf{"y", "uint16_t", poison(2 * max(nrows, 1) * width)},
		emuBuf{"ss", "float", poison(4 * max(nrows, 1) * width / 32)},
		emuBuf{"slots", "int32_t", i32Bytes(slots)},
		emuBuf{"weights", "float", f32Bytes(weights)},
		emuBuf{"out", "uint16_t", poison(2 * tokens * width)})
	got := make([]float32, tokens*width)
	for i, v := range decode(BF16, res["out"]) {
		got[i] = float32(v)
	}

	// reference.py: for each local expert e in increasing order, the routes
	// to it add post_expert_norm(x @ W2[e]ᵀ, post_w) · weight.
	want := make([]float64, tokens*width)
	plain := TileGEMM{Name: "p", N: width, K: dExpert, Weight: WeightNVFP4, Out: F32}
	for e := range local {
		for i, id := range ids {
			if int(id) != start+e {
				continue
			}
			y, err := plain.Reference(1, inter[i], nil, nil, []GEMMWeights{w[e]}, nil)
			if err != nil {
				t.Fatal(err)
			}
			var ms float64
			for c, v := range y {
				g := float64(v) * (1 + float64(postW[c]))
				ms += g * g / width
			}
			inv := 1 / math.Sqrt(ms+1e-8)
			for c, v := range y {
				want[(i/8)*width+c] += float64(v) * (1 + float64(postW[c])) * inv * float64(weights[i])
			}
		}
	}
	routed := 0
	for _, s := range slots {
		if s >= 0 {
			routed++
		}
	}
	if routed == 0 || routed == len(slots) {
		t.Fatalf("%d of %d routes local; the case should mix local and remote", routed, len(slots))
	}
	if e := relErr(f64s(got), want); e > 5e-3 || math.IsNaN(e) {
		t.Errorf("relative L2 error %.2e over %d local routes", e, routed)
	} else {
		t.Logf("relative L2 error %.2e over %d local routes", e, routed)
	}
	for c := range width {
		if got[c] != 0 {
			t.Fatalf("token 0 out[%d] = %g, want 0", c, got[c])
		}
	}
}

func TestExpertSumGolden(t *testing.T) {
	src, err := ExpertSum{Name: "moe8_expert_sum", Width: 4096, K: 8, Y: BF16, Out: BF16}.Source()
	if err != nil {
		t.Fatal(err)
	}
	checkGolden(t, "moe8_expert_sum.cu", src)
}

func ExampleExpertSum() {
	// Moe8's post-expert norm and top-8 sum over the 4096 hidden columns.
	s := ExpertSum{Name: "moe_sum", Width: 4096, K: 8, Y: BF16, Out: BF16}
	grid, block := s.Grid(16)
	src, _ := s.Source()
	fmt.Println(grid, block, len(src) > 0)
	// Output: [16 1 1] [256 1 1] true
}
