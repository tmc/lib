package tcgen05

import (
	"fmt"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatmem"
)

// loraBatch is a mixed batch: eight adapters of ranks 8..64 plus
// base-only tokens, assigned to tokens at random.
type loraBatch struct {
	x, w, aPool, bPool [][]float32
	ids                []int
	adapters           map[int]LoRAAdapterSlot
}

func newLoRABatch(r *rand.Rand, tokens, dimIn, dimOut int) loraBatch {
	b := loraBatch{adapters: make(map[int]LoRAAdapterSlot)}
	ranks := []int{8, 16, 32, 64, 8, 16, 32, 64}
	aoff, boff := 0, 0
	for i, rank := range ranks {
		id := 100 + 7*i
		b.adapters[id] = LoRAAdapterSlot{AdapterID: id, Rank: rank, Scale: float32(16) / float32(rank), AOffset: aoff, BOffset: boff + 3}
		aoff += rank
		boff += rank
	}
	b.aPool = randMat(r, aoff, dimIn, 0.2)
	b.bPool = randMat(r, boff+3, dimOut, 0.2)
	b.x = randMat(r, tokens, dimIn, 1)
	b.w = randMat(r, dimOut, dimIn, 0.1)
	for range tokens {
		if r.IntN(9) == 0 {
			b.ids = append(b.ids, -1)
			continue
		}
		b.ids = append(b.ids, 100+7*r.IntN(len(ranks)))
	}
	return b
}

func TestPlanLoRAGather(t *testing.T) {
	adapters := map[int]LoRAAdapterSlot{
		3: {AdapterID: 3, Rank: 8, Scale: 2, AOffset: 16},
		1: {AdapterID: 1, Rank: 16, Scale: 1, AOffset: 0},
	}
	ids := []int{3, -1, 1, 3, 1, 1, -1, 3, 1, 1, 1}
	p, err := PlanLoRAGather(ids, adapters)
	if err != nil {
		t.Fatal(err)
	}
	if want := []int32{2, 4, 5, 8, 9, 10, 0, 3, 7, 1, 6}; !slices.Equal(p.Order, want) {
		t.Fatalf("Order = %v, want %v", p.Order, want)
	}
	if len(p.Segments) != 2 || p.Segments[0].Slot.AdapterID != 1 || p.Segments[0].End != 6 || p.Segments[1].Start != 6 || p.Segments[1].End != 9 {
		t.Fatalf("Segments = %+v", p.Segments)
	}
	if want := [][4]int32{{2, 4, 5, 8}, {9, 10, 10, 10}, {0, 3, 7, 7}}; !slices.Equal(p.TokenQuads, want) {
		t.Fatalf("TokenQuads = %v, want %v", p.TokenQuads, want)
	}
	if want := [][4]int32{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}; !slices.Equal(p.ARowQuads, want) {
		t.Fatalf("ARowQuads = %v, want %v", p.ARowQuads, want)
	}
	if s := p.Segments[1]; s.TokenQuads != [2]int{2, 3} || s.ARowQuads != [2]int{4, 6} {
		t.Fatalf("segment 1 quad ranges = %v, %v", s.TokenQuads, s.ARowQuads)
	}
	specs, err := p.Gather4Specs("sm_100a", 64, 4)
	if err != nil {
		t.Fatal(err)
	}
	if len(specs) != 9 || specs[2].RowIndices != [4]int32{0, 3, 7, 7} {
		t.Fatalf("Gather4Specs = %+v", specs)
	}
	if ptx, err := specs[0].FormatPTX(); err != nil || !strings.Contains(ptx, ".gather4") {
		t.Fatalf("FormatPTX = %q, %v", ptx, err)
	}
	if _, err := p.Gather4Specs("sm_100a", 2, 4); err == nil {
		t.Fatal("Gather4Specs accepted 8-byte rows")
	}
	if _, err := PlanLoRAGather([]int{5}, adapters); err == nil {
		t.Fatal("unknown adapter accepted")
	}
	if _, err := PlanLoRAGather([]int{1}, map[int]LoRAAdapterSlot{1: {Rank: 12}}); err == nil {
		t.Fatal("rank 12 accepted")
	}
}

func TestSegmentedLoRAValidate(t *testing.T) {
	ok := SegmentedLoRASchedule{DimIn: 4096, DimOut: 4096, TileN: 256, BK: 64, MaxRank: 64}
	if err := ok.Validate(); err != nil {
		t.Fatalf("Validate(%+v) = %v", ok, err)
	}
	if c := ok.TMEMColumns(); c != 512 {
		t.Fatalf("TMEMColumns = %d, want 512", c)
	}
	for _, tt := range []struct {
		name string
		edit func(*SegmentedLoRASchedule)
	}{
		{"tile", func(s *SegmentedLoRASchedule) { s.TileN = 24 }},
		{"tile over 256", func(s *SegmentedLoRASchedule) { s.TileN = 512 }},
		{"tile divides N", func(s *SegmentedLoRASchedule) { s.DimOut = 4000 }},
		{"bk", func(s *SegmentedLoRASchedule) { s.BK = 48 }},
		{"k", func(s *SegmentedLoRASchedule) { s.DimIn = 100 }},
		{"rank", func(s *SegmentedLoRASchedule) { s.MaxRank = 128 }},
		{"target", func(s *SegmentedLoRASchedule) { s.Target = "sm_120a" }},
	} {
		s := ok
		tt.edit(&s)
		if err := s.Validate(); err == nil {
			t.Errorf("%s: Validate(%+v) = nil, want error", tt.name, s)
		}
	}
}

func TestSegmentedLoRATraceVerifies(t *testing.T) {
	s := SegmentedLoRASchedule{DimIn: 256, DimOut: 512, TileN: 256, BK: 64, MaxRank: 64}
	sum, err := cudatmem.VerifyTrace(cudatmem.SM100A, s.TraceEvents(9))
	if err != nil {
		t.Fatal(err)
	}
	if sum.MaxAllocatedColumns != 512 || sum.MMAs != 4+9*5 {
		t.Fatalf("summary %+v", sum)
	}
}

func runLoRAKernel(t *testing.T, s SegmentedLoRASchedule, src string, b loraBatch) []float32 {
	t.Helper()
	plan, err := PlanLoRAGather(b.ids, b.adapters)
	if err != nil {
		t.Fatal(err)
	}
	order, start, end, rank, aoff, boff, scale := LoRAKernelArgs(plan)
	grid, block := s.Grid(len(b.x))
	launch := fmt.Sprintf("eg_emu_launch(dim3(%d, %d, 1), dim3(%d, 1, 1), %d, %s, %d, %d, order, seg_start, seg_end, seg_rank, seg_aoff, seg_boff, seg_scale, x, w, a_pool, b_pool, y)",
		grid[0], grid[1], block[0], s.SharedBytes(), s.Name, len(b.x), len(start))
	res := runEmu(t, src, launch,
		emuBuf{"order", "int", i32Bytes(append(order, 0))},
		emuBuf{"seg_start", "int", i32Bytes(append(start, 0))},
		emuBuf{"seg_end", "int", i32Bytes(append(end, 0))},
		emuBuf{"seg_rank", "int", i32Bytes(append(rank, 0))},
		emuBuf{"seg_aoff", "int", i32Bytes(append(aoff, 0))},
		emuBuf{"seg_boff", "int", i32Bytes(append(boff, 0))},
		emuBuf{"seg_scale", "float", f32Bytes(append(scale, 0))},
		emuBuf{"x", "float", f32Bytes(flat(b.x))},
		emuBuf{"w", "float", f32Bytes(flat(b.w))},
		emuBuf{"a_pool", "float", f32Bytes(flat(b.aPool))},
		emuBuf{"b_pool", "float", f32Bytes(flat(b.bPool))},
		emuBuf{"y", "float", poison(4 * len(b.x) * s.DimOut)},
	)
	return bytesF32(res["y"])
}

// TestSegmentedLoRAEmulated runs mixed batches of eight adapters (ranks 8
// to 64) and base-only tokens on the host emulator: 128 tokens in one tile,
// and 200 tokens over two token tiles and two output tiles.
func TestSegmentedLoRAEmulated(t *testing.T) {
	if testing.Short() {
		t.Skip("emulated kernel run")
	}
	for _, tt := range []struct {
		name          string
		tokens, n, tn int
	}{
		{"128x64", 128, 64, 64},
		{"200x128", 200, 128, 64},
	} {
		t.Run(tt.name, func(t *testing.T) {
			s := SegmentedLoRASchedule{Name: "lora_emu", DimIn: 64, DimOut: tt.n, TileN: tt.tn, BK: 32, MaxRank: 64}
			b := newLoRABatch(rand.New(rand.NewPCG(19, uint64(tt.tokens))), tt.tokens, s.DimIn, s.DimOut)
			want, err := ReferenceFusedLoRA(b.x, b.w, b.ids, b.aPool, b.bPool, b.adapters)
			if err != nil {
				t.Fatal(err)
			}
			src, err := s.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			got := runLoRAKernel(t, s, src, b)
			e := relL2(got, flat(want))
			t.Logf("rel L2 %.6g over %d outputs", e, len(got))
			if !(e <= 1e-4) {
				t.Fatalf("rel L2 = %g, want <= 1e-4", e)
			}
		})
	}
}

// TestSegmentedLoRAEmulatedMutant checks that overwriting the base
// accumulator with the LoRA expand (enable_input_d = 0) fails.
func TestSegmentedLoRAEmulatedMutant(t *testing.T) {
	if testing.Short() {
		t.Skip("emulated kernel run")
	}
	s := SegmentedLoRASchedule{Name: "lora_mutant", DimIn: 32, DimOut: 32, TileN: 32, BK: 32, MaxRank: 16}
	b := newLoRABatch(rand.New(rand.NewPCG(20, 20)), 64, s.DimIn, s.DimOut)
	for id, a := range b.adapters {
		if a.Rank > 16 {
			a.Rank = 16
			b.adapters[id] = a
		}
	}
	want, err := ReferenceFusedLoRA(b.x, b.w, b.ids, b.aPool, b.bPool, b.adapters)
	if err != nil {
		t.Fatal(err)
	}
	src, err := s.CUDA()
	if err != nil {
		t.Fatal(err)
	}
	if e := relL2(runLoRAKernel(t, s, src, b), flat(want)); !(e <= 1e-4) {
		t.Fatalf("unmutated kernel rel L2 = %g", e)
	}
	const old = "eg_tm_mma_ss(tm, base + C_B, us, bs, TN, R, 1);"
	if !strings.Contains(src, old) {
		t.Fatalf("source lacks %q", old)
	}
	mut := strings.Replace(src, old, "eg_tm_mma_ss(tm, base + C_B, us, bs, TN, R, 0);", 1)
	if e := relL2(runLoRAKernel(t, s, mut, b), flat(want)); e <= 1e-2 {
		t.Fatalf("mutant passed: rel L2 = %g", e)
	}
}
