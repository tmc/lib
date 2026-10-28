package tcgen05

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatmem"
)

// TestPhilox4x32_10 checks the Random123 known-answer vectors.
func TestPhilox4x32_10(t *testing.T) {
	tests := []struct {
		ctr  [4]uint32
		key  [2]uint32
		want [4]uint32
	}{
		{[4]uint32{0, 0, 0, 0}, [2]uint32{0, 0}, [4]uint32{0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8}},
		{[4]uint32{0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff}, [2]uint32{0xffffffff, 0xffffffff}, [4]uint32{0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd}},
		{[4]uint32{0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344}, [2]uint32{0xa4093822, 0x299f31d0}, [4]uint32{0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1}},
	}
	for _, tt := range tests {
		if got := Philox4x32_10(tt.ctr, tt.key); got != tt.want {
			t.Errorf("Philox4x32_10(%08x, %08x) = %08x, want %08x", tt.ctr, tt.key, got, tt.want)
		}
	}
}

type flashCase struct {
	h, w  []float32
	draft []int32
}

func newFlashCase(r *rand.Rand, rows, v, d int) flashCase {
	c := flashCase{h: make([]float32, rows*d), w: make([]float32, v*d), draft: make([]int32, rows)}
	for i := range c.h {
		c.h[i] = float32(r.NormFloat64())
	}
	for i := range c.w {
		c.w[i] = float32(r.NormFloat64() / math.Sqrt(float64(d)))
	}
	for i := range c.draft {
		c.draft[i] = int32(r.IntN(v))
	}
	c.draft[0] = -1
	return c
}

// materialized computes the outputs from the full [rows, V] logit matrix:
// float32 logits from flashLogit (the same values the stream sees) for the
// Gumbel argmax, and float64 logits for LSE and probabilities.
func materialized(s FlashLogitsSchedule, c flashCase) (tok []int32, logProb, lse, draftProb []float64) {
	d, v := s.HiddenDim, s.VocabSize
	rows := len(c.h) / d
	z32 := make([]float32, rows*v)
	z64 := make([]float64, rows*v)
	for r := range rows {
		for j := range v {
			z32[r*v+j] = flashLogit(c.h[r*d:(r+1)*d], c.w[j*d:(j+1)*d])
			var acc float64
			for i := range d {
				acc += float64(c.h[r*d+i]) * float64(c.w[j*d+i])
			}
			z64[r*v+j] = acc
		}
	}
	tok = make([]int32, rows)
	logProb = make([]float64, rows)
	lse = make([]float64, rows)
	draftProb = make([]float64, rows)
	temp := float64(s.Temperature)
	for r := range rows {
		bestKey, best := math.Inf(-1), -1
		m := math.Inf(-1)
		for j := range v {
			if key := float64(z32[r*v+j]/s.Temperature) + FlashLogitsGumbel(s.Seed, r, j); key > bestKey {
				bestKey, best = key, j
			}
			m = max(m, z64[r*v+j]/temp)
		}
		var sum float64
		for j := range v {
			sum += math.Exp(z64[r*v+j]/temp - m)
		}
		lse[r] = m + math.Log(sum)
		tok[r] = int32(best)
		logProb[r] = z64[r*v+best]/temp - lse[r]
		if dt := c.draft[r]; dt >= 0 && int(dt) < v {
			draftProb[r] = math.Exp(z64[r*v+int(dt)]/temp - lse[r])
		}
	}
	return tok, logProb, lse, draftProb
}

// TestFlashLogitsReferenceMatchesMaterialized checks the single-pass stream
// against the fully materialized logits: identical sampled tokens and LSE,
// log-probabilities and draft probabilities within 1e-5.
func TestFlashLogitsReferenceMatchesMaterialized(t *testing.T) {
	tests := []struct {
		v, d, tileV int
		temp        float32
	}{
		{4096, 64, 64, 1},
		{5000, 32, 128, 0.7}, // partial last tile
		{32768, 32, 128, 1.3},
		{100, 64, 32, 0.25},
	}
	for _, tt := range tests {
		t.Run(fmt.Sprintf("V%d_D%d_T%v", tt.v, tt.d, tt.temp), func(t *testing.T) {
			s := FlashLogitsSchedule{TileV: tt.tileV, VocabSize: tt.v, HiddenDim: tt.d, Temperature: tt.temp, Seed: 0x5eed0000 + uint64(tt.v), DoubleBuffer: true}
			c := newFlashCase(rand.New(rand.NewPCG(32, uint64(tt.v))), 8, tt.v, tt.d)
			got, err := s.ExecuteReference(c.h, c.w, c.draft)
			if err != nil {
				t.Fatal(err)
			}
			tok, lp, lse, dp := materialized(s, c)
			for r := range tok {
				if got.SampledTokenID[r] != tok[r] {
					t.Fatalf("row %d: streamed token %d, materialized %d", r, got.SampledTokenID[r], tok[r])
				}
				for _, f := range []struct {
					name      string
					got, want float64
				}{
					{"LSE", float64(got.LSE[r]), lse[r]},
					{"SampledLogProb", float64(got.SampledLogProb[r]), lp[r]},
					{"DraftTokenProb", float64(got.DraftTokenProb[r]), dp[r]},
				} {
					if math.Abs(f.got-f.want) > 1e-5*max(1, math.Abs(f.want)) {
						t.Fatalf("row %d: %s = %v, want %v", r, f.name, f.got, f.want)
					}
				}
			}
			if got.DraftTokenProb[0] != 0 {
				t.Fatalf("row 0 has no draft token but DraftTokenProb = %v", got.DraftTokenProb[0])
			}
		})
	}
}

// TestFlashLogitsRescale feeds logits that rise from tile to tile, so every
// tile raises the running max and the partition sum must be rescaled.
func TestFlashLogitsRescale(t *testing.T) {
	const v, d = 512, 32
	s := FlashLogitsSchedule{TileV: 32, VocabSize: v, HiddenDim: d, Temperature: 1}
	c := flashCase{h: make([]float32, d), w: make([]float32, v*d), draft: []int32{v - 3}}
	c.h[0] = 1
	for j := range v {
		c.w[j*d] = float32(j) * 0.05 // z_j = 0.05 j, increasing across tiles
	}
	got, err := s.ExecuteReference(c.h, c.w, c.draft)
	if err != nil {
		t.Fatal(err)
	}
	_, _, lse, dp := materialized(s, c)
	if math.Abs(float64(got.LSE[0])-lse[0]) > 1e-5*lse[0] || math.Abs(float64(got.DraftTokenProb[0])-dp[0]) > 1e-5 {
		t.Fatalf("LSE %v draft prob %v, want %v and %v", got.LSE[0], got.DraftTokenProb[0], lse[0], dp[0])
	}
}

func TestFlashLogitsGumbelDistribution(t *testing.T) {
	// The Gumbel-max sample of uniform logits is uniform; check the mean
	// and that tokens spread over the vocabulary.
	const n = 20000
	var sum float64
	for i := range n {
		sum += FlashLogitsGumbel(7, i/100, i%100)
	}
	if mean := sum / n; math.Abs(mean-0.5772) > 0.03 {
		t.Fatalf("Gumbel mean %.4f, want Euler-Mascheroni 0.5772", mean)
	}
	s := FlashLogitsSchedule{TileV: 32, VocabSize: 40, HiddenDim: 32, Temperature: 1, Seed: 9}
	h := make([]float32, 400*32) // zero hidden states: all logits 0
	got, err := s.ExecuteReference(h, make([]float32, 40*32), nil)
	if err != nil {
		t.Fatal(err)
	}
	var counts [40]int
	for _, tok := range got.SampledTokenID {
		counts[tok]++
	}
	for tok, c := range counts {
		if c == 0 || c > 30 {
			t.Fatalf("token %d sampled %d times of 400 from a uniform distribution", tok, c)
		}
	}
}

func TestFlashLogitsScheduleValidate(t *testing.T) {
	ok := FlashLogitsSchedule{VocabSize: 100, HiddenDim: 64, Temperature: 1}
	if err := ok.Validate(); err != nil {
		t.Fatal(err)
	}
	for name, bad := range map[string]FlashLogitsSchedule{
		"temp":   {VocabSize: 100, HiddenDim: 64},
		"dim":    {VocabSize: 100, HiddenDim: 48, Temperature: 1},
		"tileV":  {VocabSize: 100, HiddenDim: 64, Temperature: 1, TileV: 256},
		"tileM":  {VocabSize: 100, HiddenDim: 64, Temperature: 1, TileM: 64},
		"target": {VocabSize: 100, HiddenDim: 64, Temperature: 1, Target: "sm_121a"},
		"vocab":  {HiddenDim: 64, Temperature: 1},
	} {
		if err := bad.Validate(); err == nil {
			t.Errorf("%s: Validate() = nil, want error", name)
		}
	}
}

func TestFlashLogitsTrace(t *testing.T) {
	for _, dbl := range []bool{false, true} {
		s := FlashLogitsSchedule{TileV: 64, VocabSize: 300, HiddenDim: 64, Temperature: 1, DoubleBuffer: dbl}
		ev := s.TraceEvents()
		sum, err := cudatmem.VerifyTrace(cudatmem.SM100A, ev)
		if err != nil {
			t.Fatalf("double=%v: VerifyTrace: %v", dbl, err)
		}
		if want := 5 * 2; sum.MMAs != want {
			t.Fatalf("double=%v: %d MMAs, want %d", dbl, sum.MMAs, want)
		}
		if want := uint32(64 * s.buffers()); sum.MaxAllocatedColumns != want {
			t.Fatalf("double=%v: %d TMEM columns, want %d", dbl, sum.MaxAllocatedColumns, want)
		}
		// Mutant: a ping-pong MMA into tmem[TileV..2*TileV) without the
		// second buffer allocated.
		if !dbl {
			bad := append([]cudatmem.TraceEvent(nil), ev...)
			for i := range bad {
				if bad[i].Kind == cudatmem.EventMMA {
					bad[i].Offset = 64
					break
				}
			}
			if _, err := cudatmem.VerifyTrace(cudatmem.SM100A, bad); err == nil {
				t.Fatal("VerifyTrace accepted an MMA into an unallocated buffer")
			}
		}
	}
}

func TestFlashLogitsSource(t *testing.T) {
	s := FlashLogitsSchedule{Name: "flash_src", VocabSize: 1000, HiddenDim: 64, Temperature: 0.5, DoubleBuffer: true}
	src, err := s.CUDA()
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"tcgen05.mma.cta_group::1.kind::tf32",
		"mbarrier.try_wait.parity.shared::cta.b64",
		"constexpr float TEMP = 5e-01f;",
		"NBUF = 2",
		"l *= expf(m - m_new);",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("CUDA() missing %q", want)
		}
	}
	// The only global outputs are the four per-row arrays.
	sig := src[strings.Index(src, "flash_src("):]
	sig = sig[:strings.Index(sig, ")")]
	if n := strings.Count(sig, "__restrict__"); n != 7 {
		t.Fatalf("kernel has %d pointer parameters, want 7 (3 inputs, 4 per-row outputs)", n)
	}
}

// TestFlashLogitsEmu runs the emitted kernel under cudahostemu and compares
// it with ExecuteReference.
func TestFlashLogitsEmu(t *testing.T) {
	for _, tc := range []struct {
		v, d, tileV int
		dbl         bool
	}{
		{300, 64, 64, true},
		{97, 32, 32, false},
	} {
		t.Run(fmt.Sprintf("V%d_D%d", tc.v, tc.d), func(t *testing.T) {
			s := FlashLogitsSchedule{Name: "flash_emu", TileV: tc.tileV, VocabSize: tc.v, HiddenDim: tc.d, Temperature: 0.8, Seed: 1234567, DoubleBuffer: tc.dbl}
			c := newFlashCase(rand.New(rand.NewPCG(32, uint64(tc.v))), 128, tc.v, tc.d)
			want, err := s.ExecuteReference(c.h, c.w, c.draft)
			if err != nil {
				t.Fatal(err)
			}
			src, err := s.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			draft := make([]byte, 4*len(c.draft))
			for i, v := range c.draft {
				binary.LittleEndian.PutUint32(draft[4*i:], uint32(v))
			}
			res := runEmu(t, src,
				"eg_emu_launch(dim3(1), dim3(128), 0, flash_emu, h, w_vocab, draft, out_token, out_logprob, out_lse, out_draft_prob)",
				emuBuf{"h", "float", f32Bytes(c.h)},
				emuBuf{"w_vocab", "float", f32Bytes(c.w)},
				emuBuf{"draft", "int", draft},
				emuBuf{"out_token", "int", poison(4 * 128)},
				emuBuf{"out_logprob", "float", poison(4 * 128)},
				emuBuf{"out_lse", "float", poison(4 * 128)},
				emuBuf{"out_draft_prob", "float", poison(4 * 128)},
			)
			tok := bytesU32(res["out_token"])
			for r := range 128 {
				if int32(tok[r]) != want.SampledTokenID[r] {
					t.Fatalf("row %d: kernel token %d, reference %d", r, int32(tok[r]), want.SampledTokenID[r])
				}
			}
			closeF32(t, "out_lse", bytesF32(res["out_lse"]), want.LSE, 1e-5)
			closeF32(t, "out_logprob", bytesF32(res["out_logprob"]), want.SampledLogProb, 1e-4)
			closeF32(t, "out_draft_prob", bytesF32(res["out_draft_prob"]), want.DraftTokenProb, 1e-4)
		})
	}
}

func ExampleFlashLogitsSchedule_ExecuteReference() {
	s := FlashLogitsSchedule{TileV: 32, VocabSize: 100, HiddenDim: 32, Temperature: 1, Seed: 42}
	h := make([]float32, 32)
	w := make([]float32, 100*32)
	h[0] = 1
	w[17*32] = 20 // token 17 has logit 20, all others 0
	out, err := s.ExecuteReference(h, w, []int32{17})
	if err != nil {
		panic(err)
	}
	fmt.Printf("token %d, log p %.4f, draft p %.6f\n", out.SampledTokenID[0], out.SampledLogProb[0], out.DraftTokenProb[0])
	// Output:
	// token 17, log p 0.0000, draft p 1.000000
}
