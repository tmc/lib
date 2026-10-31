package tcgen05

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatmem"
)

func TestTubeletPatchValidate(t *testing.T) {
	ok := TubeletPatchSpec{Frames: 16, Height: 224, Width: 224, Channels: 4, PatchT: 2, PatchH: 14, PatchW: 14, EmbedDim: 1024}
	if err := ok.Validate(); err != nil {
		t.Fatalf("Validate(%+v) = %v", ok, err)
	}
	if got := ok.Patches(); got != 8*16*16 {
		t.Fatalf("Patches = %d", got)
	}
	for _, tt := range []struct {
		name string
		edit func(*TubeletPatchSpec)
	}{
		{"rgb", func(s *TubeletPatchSpec) { s.Channels = 3 }},
		{"channels", func(s *TubeletPatchSpec) { s.Channels = 36 }},
		{"patch", func(s *TubeletPatchSpec) { s.PatchT = 17 }},
		{"embed", func(s *TubeletPatchSpec) { s.EmbedDim = 200 }},
		{"embed tile", func(s *TubeletPatchSpec) { s.EmbedDim = 144 }},
		{"target", func(s *TubeletPatchSpec) { s.Target = "sm_121a" }},
	} {
		s := ok
		tt.edit(&s)
		if err := s.Validate(); err == nil {
			t.Errorf("%s: Validate(%+v) = nil, want error", tt.name, s)
		}
	}
}

func TestBuildTubeletTMADescriptor(t *testing.T) {
	s := TubeletPatchSpec{Frames: 16, Height: 224, Width: 224, Channels: 4, PatchT: 2, PatchH: 14, PatchW: 14, EmbedDim: 1024}
	d, err := BuildTubeletTMADescriptor(s, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	if d.Rank != 4 || d.Dims[0] != 4 || d.Dims[3] != 16 || d.Box != [5]uint32{4, 14, 14, 2} {
		t.Fatalf("descriptor %+v", d)
	}
	if want := [5]uint64{16, 16 * 224, 16 * 224 * 224}; d.Strides != want {
		t.Fatalf("Strides = %v, want %v", d.Strides, want)
	}
	if _, err := BuildTubeletTMADescriptor(s, 1<<20+4); err == nil {
		t.Fatal("unaligned base accepted")
	}
}

func TestTubeletReference(t *testing.T) {
	// A 1x1x2 patch over a width-2 video with identity weights returns
	// the video itself.
	s := TubeletPatchSpec{Frames: 2, Height: 3, Width: 2, Channels: 4, PatchT: 1, PatchH: 1, PatchW: 2, EmbedDim: 16}
	video := make([]float32, 2*3*2*4)
	for i := range video {
		video[i] = float32(i)
	}
	w := make([][]float32, 16)
	for e := range w {
		w[e] = make([]float32, 8)
		if e < 8 {
			w[e][e] = 1
		}
	}
	got, err := s.Reference(video, w)
	if err != nil {
		t.Fatal(err)
	}
	for p, row := range got {
		if !slices.Equal(row[:8], video[8*p:8*p+8]) || slices.ContainsFunc(row[8:], func(x float32) bool { return x != 0 }) {
			t.Fatalf("patch %d = %v, want %v", p, row, video[8*p:8*p+8])
		}
	}
}

func TestTubeletTraceVerifies(t *testing.T) {
	s := TubeletPatchSpec{Frames: 16, Height: 224, Width: 224, Channels: 4, PatchT: 2, PatchH: 14, PatchW: 14, EmbedDim: 1024}
	sum, err := cudatmem.VerifyTrace(cudatmem.SM100A, s.TraceEvents())
	if err != nil {
		t.Fatal(err)
	}
	if sum.MMAs != s.PatchLen()/32 {
		t.Fatalf("summary %+v, want %d MMAs", sum, s.PatchLen()/32)
	}
}

// TestTubeletEmulated runs a video whose size is not a multiple of the
// patch, so the far tubelets are zero padded, over two embed tiles.
func TestTubeletEmulated(t *testing.T) {
	if testing.Short() {
		t.Skip("emulated kernel run")
	}
	s := TubeletPatchSpec{Name: "tubelet_emu", Frames: 3, Height: 9, Width: 10, Channels: 4, PatchT: 2, PatchH: 4, PatchW: 4, EmbedDim: 256}
	r := rand.New(rand.NewPCG(17, 1))
	video := flat(randMat(r, s.Frames*s.Height*s.Width, s.Channels, 1))
	w := randMat(r, s.EmbedDim, s.PatchLen(), 0.1)
	want, err := s.Reference(video, w)
	if err != nil {
		t.Fatal(err)
	}
	src, err := s.CUDA()
	if err != nil {
		t.Fatal(err)
	}
	run := func(src string) []float32 {
		grid, block := s.Grid()
		launch := fmt.Sprintf("eg_emu_launch(dim3(%d, %d, 1), dim3(%d, 1, 1), %d, %s, video, w, out)", grid[0], grid[1], block[0], s.SharedBytes(), s.Name)
		res := runEmu(t, src, launch,
			emuBuf{"video", "float", f32Bytes(video)},
			emuBuf{"w", "float", f32Bytes(flat(w))},
			emuBuf{"out", "float", poison(4 * s.Patches() * s.EmbedDim)},
		)
		return bytesF32(res["out"])
	}
	e := relL2(run(src), flat(want))
	t.Logf("rel L2 %.6g over %d patches", e, s.Patches())
	if !(e <= 1e-5) {
		t.Fatalf("rel L2 = %g", e)
	}
	// Dropping the edge test reads past the frame and width edges.
	const old = "p < P && t < T && h < H && x < W ?"
	if !strings.Contains(src, old) {
		t.Fatalf("source lacks %q", old)
	}
	mut := strings.Replace(src, old, "p < P && t < T && h < H ?", 1)
	if e := relL2(run(mut), flat(want)); e <= 1e-3 {
		t.Fatalf("mutant passed: rel L2 = %g", e)
	}
}

func TestMRoPEValidate(t *testing.T) {
	for _, tt := range []struct {
		s  MRoPESpec
		ok bool
	}{
		{MRoPESpec{HeadDim: 128, PairsT: 16, PairsH: 24, PairsW: 24, BaseTheta: 1e6}, true},
		{MRoPESpec{HeadDim: 64, PairsT: 8, PairsH: 8, PairsW: 8, BaseTheta: 1e4}, true},
		{MRoPESpec{HeadDim: 64, PairsT: 16, PairsH: 8, PairsW: 9, BaseTheta: 1e4}, false},
		{MRoPESpec{HeadDim: 64, BaseTheta: 1e4}, false},
		{MRoPESpec{HeadDim: 64, PairsT: 4, BaseTheta: 1}, false},
	} {
		if err := tt.s.Validate(); (err == nil) != tt.ok {
			t.Errorf("Validate(%+v) = %v, want ok %v", tt.s, err, tt.ok)
		}
	}
}

// TestMRoPEDegeneratesToRoPE checks that equal (t, h, w) positions give
// the 1D RoPE bits exactly, and distinct positions do not.
func TestMRoPEDegeneratesToRoPE(t *testing.T) {
	spec := MRoPESpec{HeadDim: 64, PairsT: 8, PairsH: 12, PairsW: 12, BaseTheta: 1e6}
	r := rand.New(rand.NewPCG(17, 2))
	x := randMat(r, 40, spec.HeadDim, 1)
	pos := make([]int32, len(x))
	thw := make([][3]int32, len(x))
	for i := range pos {
		pos[i] = int32(r.IntN(4096))
		thw[i] = [3]int32{pos[i], pos[i], pos[i]}
	}
	a, err := ApplyMRoPEReference(x, thw, spec)
	if err != nil {
		t.Fatal(err)
	}
	b, err := ApplyRoPEReference(x, pos, spec)
	if err != nil {
		t.Fatal(err)
	}
	for i := range a {
		for c := range a[i] {
			if math.Float32bits(a[i][c]) != math.Float32bits(b[i][c]) {
				t.Fatalf("row %d channel %d: mrope %g, rope %g", i, c, a[i][c], b[i][c])
			}
		}
	}
	thw[3][2]++
	a, _ = ApplyMRoPEReference(x, thw, spec)
	if slices.Equal(a[3], b[3]) {
		t.Fatal("width position ignored")
	}
}

func TestBuildJaggedBatchSchedule(t *testing.T) {
	s, err := BuildJaggedBatchSchedule([]int32{0, 73, 73, 585, 775}, 128)
	if err != nil {
		t.Fatal(err)
	}
	want := [][2]int{{0, 0}, {2, 0}, {2, 1}, {2, 2}, {2, 3}, {3, 0}, {3, 1}}
	if !slices.Equal(s.TileMap, want) {
		t.Fatalf("TileMap = %v, want %v", s.TileMap, want)
	}
	for _, cu := range [][]int32{{0}, {1, 5}, {0, 5, 3}} {
		if _, err := BuildJaggedBatchSchedule(cu, 128); err == nil {
			t.Errorf("BuildJaggedBatchSchedule(%v) = nil error", cu)
		}
	}
}

func TestJaggedAttentionTraceVerifies(t *testing.T) {
	a := JaggedAttention{MRoPE: MRoPESpec{HeadDim: 128, PairsT: 16, PairsH: 24, PairsW: 24, BaseTheta: 1e6}}
	sum, err := cudatmem.VerifyTrace(cudatmem.SM100A, a.TraceEvents(8))
	if err != nil {
		t.Fatal(err)
	}
	if sum.MaxAllocatedColumns != 256 || sum.MMAs != 16 {
		t.Fatalf("summary %+v", sum)
	}
}

// jaggedInputs builds one head of q, k, v and (t, h, w) positions laid out
// as a video-like grid per sequence.
func jaggedInputs(r *rand.Rand, lens []int, d int) (cu []int32, q, k, v [][]float32, pos [][3]int32) {
	cu = []int32{0}
	for _, n := range lens {
		cu = append(cu, cu[len(cu)-1]+int32(n))
		for i := range n {
			pos = append(pos, [3]int32{int32(i / 64), int32(i / 8 % 8), int32(i % 8)})
		}
	}
	n := int(cu[len(cu)-1])
	return cu, randMat(r, n, d, 1), randMat(r, n, d, 1), randMat(r, n, d, 1), pos
}

func runJagged(t *testing.T, a JaggedAttention, src string, cu []int32, q, k, v [][]float32, pos [][3]int32) []float32 {
	t.Helper()
	sched, err := BuildJaggedBatchSchedule(cu, 128)
	if err != nil {
		t.Fatal(err)
	}
	var tseq, tidx []int32
	for _, m := range sched.TileMap {
		tseq = append(tseq, int32(m[0]))
		tidx = append(tidx, int32(m[1]))
	}
	var p []int32
	for _, x := range pos {
		p = append(p, x[:]...)
	}
	grid, block := a.Grid(sched, 1)
	launch := fmt.Sprintf("eg_emu_launch(dim3(%d, %d, 1), dim3(%d, 1, 1), %d, %s, %d, tile_seq, tile_idx, cu_seqlens, inv_freq, pos, q, k, v, out)",
		grid[0], grid[1], block[0], a.SharedBytes(), a.Name, len(q))
	res := runEmu(t, src, launch,
		emuBuf{"tile_seq", "int", i32Bytes(tseq)},
		emuBuf{"tile_idx", "int", i32Bytes(tidx)},
		emuBuf{"cu_seqlens", "int", i32Bytes(cu)},
		emuBuf{"inv_freq", "float", f32Bytes(a.MRoPE.InvFreq())},
		emuBuf{"pos", "int", i32Bytes(p)},
		emuBuf{"q", "float", f32Bytes(flat(q))},
		emuBuf{"k", "float", f32Bytes(flat(k))},
		emuBuf{"v", "float", f32Bytes(flat(v))},
		emuBuf{"out", "float", poison(4 * len(q) * a.MRoPE.HeadDim)},
	)
	return bytesF32(res["out"])
}

// TestJaggedAttentionEmulated runs sequences of 73, 300 and 190 tokens
// in one jagged batch, checks them against the float64 reference, and
// checks that each sequence's output is bit-identical to running it alone
// (batch isolation). A mutant whose key range runs to the end of the batch
// (non-causal) or starts at the first token of the batch (causal) must fail.
func TestJaggedAttentionEmulated(t *testing.T) {
	if testing.Short() {
		t.Skip("emulated kernel run")
	}
	for _, causal := range []bool{false, true} {
		t.Run(fmt.Sprintf("causal=%v", causal), func(t *testing.T) {
			a := JaggedAttention{Name: "jagged_emu", Causal: causal, MRoPE: MRoPESpec{HeadDim: 32, PairsT: 4, PairsH: 6, PairsW: 6, BaseTheta: 1e4}}
			lens := []int{73, 300, 190}
			cu, q, k, v, pos := jaggedInputs(rand.New(rand.NewPCG(17, 3)), lens, a.MRoPE.HeadDim)
			sched, _ := BuildJaggedBatchSchedule(cu, 128)
			want, err := a.Reference(sched, q, k, v, pos)
			if err != nil {
				t.Fatal(err)
			}
			src, err := a.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			got := runJagged(t, a, src, cu, q, k, v, pos)
			e := relL2(got, flat(want))
			t.Logf("rel L2 %.6g over %d tokens", e, len(q))
			if !(e <= 1e-5) {
				t.Fatalf("rel L2 = %g", e)
			}
			d := a.MRoPE.HeadDim
			lo, hi := int(cu[1]), int(cu[2])
			alone := runJagged(t, a, src, []int32{0, int32(hi - lo)}, q[lo:hi], k[lo:hi], v[lo:hi], pos[lo:hi])
			for i, x := range alone {
				if math.Float32bits(x) != math.Float32bits(got[lo*d+i]) {
					t.Fatalf("sequence 1 alone differs at %d: %g vs %g", i, x, got[lo*d+i])
				}
			}
			// Under causal masking kv_end only bounds work, so the causal
			// mutant starts the key loop at token 0 instead, attending
			// into earlier sequences.
			old, new := "const int kv_end = CAUSAL ? (q0 + qrows < seq1 ? q0 + qrows : seq1) : seq1;", "const int kv_end = tokens;"
			if causal {
				old, new = "for (int kv0 = seq0; kv0 < kv_end; kv0 += KT)", "for (int kv0 = 0; kv0 < kv_end; kv0 += KT)"
			}
			if !strings.Contains(src, old) {
				t.Fatalf("source lacks %q", old)
			}
			mut := strings.Replace(src, old, new, 1)
			e = relL2(runJagged(t, a, mut, cu, q, k, v, pos), flat(want))
			t.Logf("mutant rel L2 %.3g", e)
			if e <= 1e-3 {
				t.Fatalf("mutant passed: rel L2 = %g", e)
			}
		})
	}
}

func ExampleBuildJaggedBatchSchedule() {
	// Three images of 73, 300 and 190 patches in one batch.
	s, err := BuildJaggedBatchSchedule([]int32{0, 73, 373, 563}, 128)
	if err != nil {
		panic(err)
	}
	fmt.Println(s.Tokens(), s.TileMap)
	// Output: 563 [[0 0] [1 0] [1 1] [1 2] [2 0] [2 1]]
}
