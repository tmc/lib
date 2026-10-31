package hopper

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
)

func TestGrammarBitmask(t *testing.T) {
	m := NewGrammarBitmask(70)
	if len(m.Words) != 3 {
		t.Fatalf("len(Words) = %d, want 3", len(m.Words))
	}
	for _, v := range []int{0, 31, 32, 69} {
		m.Allow(v)
	}
	for v := range 70 {
		want := v == 0 || v == 31 || v == 32 || v == 69
		if m.Allowed(v) != want {
			t.Errorf("Allowed(%d) = %v, want %v", v, m.Allowed(v), want)
		}
	}
	all := AllowAllGrammar(70)
	for v := range 70 {
		if !all.Allowed(v) {
			t.Fatalf("AllowAllGrammar(70) disallows %d", v)
		}
	}
	if all.Words[2] != 0x3f {
		t.Errorf("AllowAllGrammar(70).Words[2] = %#x, want only the 6 valid bits", all.Words[2])
	}
}

func TestSamplingParamsProbs(t *testing.T) {
	logits := []float32{1, 3, 2, 0, 3}
	allow := AllowAllGrammar(5)
	no1 := AllowAllGrammar(5)
	no1.Words[0] &^= 1 << 1
	softmax := func(xs ...float64) []float64 {
		var z float64
		for _, x := range xs {
			z += math.Exp(x)
		}
		out := make([]float64, len(xs))
		for i, x := range xs {
			out[i] = math.Exp(x) / z
		}
		return out
	}
	full := softmax(1, 3, 2, 0, 3)
	tests := []struct {
		name string
		p    SamplingParams
		mask GrammarBitmask
		seen []uint32
		want []float64
	}{
		{"softmax", SamplingParams{Temperature: 1}, allow, nil, full},
		{"greedy tie to lower id", SamplingParams{}, allow, nil, []float64{0, 1, 0, 0, 0}},
		{"greedy masked", SamplingParams{}, no1, nil, []float64{0, 0, 0, 0, 1}},
		{"masked", SamplingParams{Temperature: 1}, no1, nil, func() []float64 {
			p := softmax(1, 2, 0, 3)
			return []float64{p[0], 0, p[1], p[2], p[3]}
		}()},
		{"temperature", SamplingParams{Temperature: 2}, allow, nil, softmax(0.5, 1.5, 1, 0, 1.5)},
		{"top-k 2", SamplingParams{Temperature: 1, TopK: 2}, allow, nil, []float64{0, 0.5, 0, 0, 0.5}},
		{"top-k 3", SamplingParams{Temperature: 1, TopK: 3}, allow, nil, func() []float64 {
			p := softmax(3, 2, 3)
			return []float64{0, p[0], p[1], 0, p[2]}
		}()},
		// Ranked 1, 4, 2, 0, 3; mass before rank 2 is full[1]+full[4] < 0.9.
		{"top-p", SamplingParams{Temperature: 1, TopP: 0.9}, allow, nil, func() []float64 {
			p := softmax(3, 2, 3)
			return []float64{0, p[0], p[1], 0, p[2]}
		}()},
		{"top-p small keeps top-1", SamplingParams{Temperature: 1, TopP: 0.01}, allow, nil, []float64{0, 1, 0, 0, 0}},
		// p_v >= 0.3 p_max: e^(s-3) >= 0.3 keeps logits 3, 3, 2.
		{"min-p", SamplingParams{Temperature: 1, MinP: 0.3}, allow, nil, func() []float64 {
			p := softmax(3, 2, 3)
			return []float64{0, p[0], p[1], 0, p[2]}
		}()},
		{"penalty", SamplingParams{Temperature: 1, RepetitionPenalty: 3}, allow, []uint32{1<<1 | 1<<3}, softmax(1, 1, 2, 0, 3)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.p.VocabSize = 5
			got, err := tt.p.Probs(logits, tt.mask, tt.seen)
			if err != nil {
				t.Fatal(err)
			}
			for v := range got {
				if math.Abs(got[v]-tt.want[v]) > 1e-12 {
					t.Fatalf("Probs = %v, want %v", got, tt.want)
				}
			}
		})
	}
	if _, err := (SamplingParams{VocabSize: 5}).Probs(logits, NewGrammarBitmask(5), nil); err == nil {
		t.Error("Probs with an empty grammar succeeded")
	}
}

func randomSamplerRow(r *rand.Rand, v int) ([]float32, GrammarBitmask, []uint32) {
	logits := make([]float32, v)
	for i := range logits {
		logits[i] = float32(r.NormFloat64() * 2)
	}
	mask := NewGrammarBitmask(v)
	for i := range v {
		if r.IntN(3) > 0 {
			mask.Allow(i)
		}
	}
	mask.Allow(r.IntN(v))
	seen := make([]uint32, (v+31)/32)
	for i := range seen {
		seen[i] = r.Uint32() & r.Uint32()
	}
	return logits, mask, seen
}

func randomSamplingParams(r *rand.Rand, v int) SamplingParams {
	p := SamplingParams{VocabSize: v, Seed: r.Uint64()}
	if r.IntN(5) > 0 {
		p.Temperature = 0.3 + 1.5*r.Float32()
	}
	if r.IntN(2) == 0 {
		p.TopK = 1 + r.IntN(v)
	}
	if r.IntN(2) == 0 {
		p.TopP = 0.2 + 0.75*r.Float32()
	}
	if r.IntN(2) == 0 {
		p.MinP = 0.3 * r.Float32()
	}
	if r.IntN(2) == 0 {
		p.RepetitionPenalty = 0.8 + r.Float32()
	}
	return p
}

// TestSampleTokenFusedGrammar draws 10⁵ tokens under random grammars and
// parameters: none may be disallowed or outside the reference support.
func TestSampleTokenFusedGrammar(t *testing.T) {
	r := rand.New(rand.NewPCG(18, 1))
	const draws = 100000
	for i := range draws {
		v := 1 + r.IntN(80)
		logits, mask, seen := randomSamplerRow(r, v)
		p := randomSamplingParams(r, v)
		tok, _, err := SampleTokenFused(logits, mask, seen, p, uint64(i))
		if err != nil {
			t.Fatal(err)
		}
		if tok < 0 || int(tok) >= v || !mask.Allowed(int(tok)) {
			t.Fatalf("draw %d: token %d not allowed by the grammar", i, tok)
		}
		if i%10 == 0 {
			probs, err := p.Probs(logits, mask, seen)
			if err != nil {
				t.Fatal(err)
			}
			if probs[tok] == 0 {
				t.Fatalf("draw %d: token %d outside the support of %+v", i, tok, p)
			}
		}
	}
}

// TestSampleTokenFusedDistribution compares 50 000 draws per configuration
// with the exact distribution of the pipeline.
func TestSampleTokenFusedDistribution(t *testing.T) {
	const (
		v     = 16
		draws = 50000
	)
	r := rand.New(rand.NewPCG(18, 2))
	logits, mask, seen := randomSamplerRow(r, v)
	tests := []SamplingParams{
		{Temperature: 1},
		{Temperature: 0.6},
		{Temperature: 1.4, TopK: 4},
		{Temperature: 1, TopP: 0.7},
		{Temperature: 1, MinP: 0.1},
		{Temperature: 0.8, TopK: 6, TopP: 0.8, MinP: 0.05, RepetitionPenalty: 1.3},
		{Temperature: 1, TopK: 1},
	}
	for _, p := range tests {
		p.VocabSize, p.Seed = v, 99
		t.Run(fmt.Sprintf("%+v", p), func(t *testing.T) {
			want, err := p.Probs(logits, mask, seen)
			if err != nil {
				t.Fatal(err)
			}
			got := make([]float64, v)
			for step := range uint64(draws) {
				tok, lp, err := SampleTokenFused(logits, mask, seen, p, step+1)
				if err != nil {
					t.Fatal(err)
				}
				got[tok]++
				if step == 0 {
					all := p
					all.TopK, all.TopP = 0, 0
					pre, err := all.Probs(logits, mask, seen)
					if err != nil {
						t.Fatal(err)
					}
					if d := math.Abs(math.Exp(float64(lp)) - pre[tok]); d > 1e-5 {
						t.Errorf("logProb %v of token %d, want log %v", lp, tok, pre[tok])
					}
				}
			}
			var tvd float64
			for i := range got {
				tvd += math.Abs(got[i]/draws - want[i])
			}
			tvd /= 2
			if tvd > 0.01 {
				t.Fatalf("TVD %.4f, want < 0.01\ngot  %v\nwant %v", tvd, got, want)
			}
		})
	}
}

func TestGrammarSamplerErrors(t *testing.T) {
	tests := []struct {
		name string
		g    GrammarSampler
		want string
	}{
		{"name", GrammarSampler{Name: "x y", Params: SamplingParams{VocabSize: 8}}, "bad name"},
		{"threads", GrammarSampler{Threads: 48, Params: SamplingParams{VocabSize: 8}}, "power of two"},
		{"vocab", GrammarSampler{}, "VocabSize"},
		{"temperature", GrammarSampler{Params: SamplingParams{VocabSize: 8, Temperature: -1}}, "Temperature"},
		{"min-p", GrammarSampler{Params: SamplingParams{VocabSize: 8, MinP: 2}}, "MinP"},
	}
	for _, tt := range tests {
		if _, err := tt.g.CUDA(); err == nil || !strings.Contains(err.Error(), tt.want) {
			t.Errorf("%s: CUDA() error = %v, want %q", tt.name, err, tt.want)
		}
	}
	g := GrammarSampler{Params: SamplingParams{VocabSize: 8}}
	if _, _, err := g.Sample(make([]float32, 7), AllowAllGrammar(8), nil, 0); err == nil {
		t.Error("Sample with 7 logits for vocab 8 succeeded")
	}
}

// samplerEmuRows runs g on the emulator for each row and returns the tokens
// and log-probabilities.
func samplerEmuRows(t *testing.T, g GrammarSampler, src string, logits []float32, mask, seen []uint32, steps []uint64) ([]int32, []float32) {
	t.Helper()
	rows := len(steps)
	stepBytes := make([]byte, 8*rows)
	for i, s := range steps {
		for k := range 8 {
			stepBytes[8*i+k] = byte(s >> (8 * k))
		}
	}
	res := runEmu(t, src, fmt.Sprintf("%s_launch(%d, logits, mask, seen, steps, tokens, logprobs, 0)", g.Name, rows),
		emuBuf{"logits", "float", f32Bytes(logits)},
		emuBuf{"mask", "uint32_t", u32le(mask)},
		emuBuf{"seen", "uint32_t", u32le(seen)},
		emuBuf{"steps", "uint64_t", stepBytes},
		emuBuf{"tokens", "int32_t", poison(4 * rows)},
		emuBuf{"logprobs", "float", poison(4 * rows)},
	)
	return specBytesI32(res["tokens"]), bytesF32(res["logprobs"])
}

func TestGrammarSamplerEmulated(t *testing.T) {
	if _, err := cudahostemu.Compiler(); errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip(err)
	}
	const (
		v    = 300 // not a multiple of the block or of 32
		rows = 24
	)
	r := rand.New(rand.NewPCG(18, 3))
	var logits []float32
	var mask, seen []uint32
	var masks []GrammarBitmask
	var seens [][]uint32
	var steps []uint64
	for i := range rows {
		l, m, s := randomSamplerRow(r, v)
		logits = append(logits, l...)
		mask = append(mask, m.Words...)
		seen = append(seen, s...)
		masks = append(masks, m)
		seens = append(seens, s)
		steps = append(steps, uint64(i)<<33|uint64(i))
	}
	configs := []SamplingParams{
		{},
		{Temperature: 0.7},
		{Temperature: 1, TopK: 5},
		{Temperature: 1.2, TopP: 0.6, MinP: 0.02},
		{Temperature: 0.9, TopK: 20, TopP: 0.5, RepetitionPenalty: 1.5},
	}
	for ci, p := range configs {
		p.VocabSize, p.Seed = v, 0xfeedface0000+uint64(ci)
		g := GrammarSampler{Name: fmt.Sprintf("sampler_%d", ci), Params: p, Threads: 64}
		t.Run(fmt.Sprintf("%+v", p), func(t *testing.T) {
			src, err := g.Source()
			if err != nil {
				t.Fatal(err)
			}
			want := make([]int32, rows)
			wantLP := make([]float32, rows)
			for i := range rows {
				want[i], wantLP[i], err = g.Sample(logits[i*v:(i+1)*v], masks[i], seens[i], steps[i])
				if err != nil {
					t.Fatal(err)
				}
			}
			got, gotLP := samplerEmuRows(t, g, src, logits, mask, seen, steps)
			for i := range rows {
				if got[i] != want[i] || math.Abs(float64(gotLP[i]-wantLP[i])) > 1e-4 {
					t.Fatalf("row %d: emulated token %d logprob %v, host %d %v", i, got[i], gotLP[i], want[i], wantLP[i])
				}
			}
			if p.greedy() {
				return
			}
			mutants := []struct{ name, old, new string }{
				{"grammar mask dropped", "if (((mk[v >> 5] >> (v & 31)) & 1u) == 0u) return NEG_INF;", ""},
				{"natural log as log2", "return log2f(x) * 0.6931472f;", "return log2f(x);"},
			}
			for _, m := range mutants {
				t.Run(m.name, func(t *testing.T) {
					if !strings.Contains(src, m.old) {
						t.Fatalf("mutant site %q not in source", m.old)
					}
					got, _ := samplerEmuRows(t, g, strings.Replace(src, m.old, m.new, 1), logits, mask, seen, steps)
					for i := range rows {
						if got[i] != want[i] || got[i] < 0 || !masks[i].Allowed(int(got[i])) {
							return
						}
					}
					t.Fatal("mutant survived")
				})
			}
		})
	}
}

func TestGrammarSamplerEmulatedEmptyGrammar(t *testing.T) {
	if _, err := cudahostemu.Compiler(); errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip(err)
	}
	g := GrammarSampler{Name: "sampler_empty", Params: SamplingParams{VocabSize: 40, Temperature: 1}, Threads: 32}
	src, err := g.Source()
	if err != nil {
		t.Fatal(err)
	}
	got, lp := samplerEmuRows(t, g, src, make([]float32, 40), make([]uint32, 2), nil, []uint64{0})
	if got[0] != -1 || !math.IsInf(float64(lp[0]), -1) {
		t.Fatalf("empty grammar: token %d logprob %v, want -1 -Inf", got[0], lp[0])
	}
}

func ExampleSampleTokenFused() {
	// The grammar allows only tokens 2 and 5; greedy decoding picks the
	// larger of their logits even though token 0's is larger still.
	logits := []float32{9, 1, 3, 0, 0, 4, 0, 0}
	mask := NewGrammarBitmask(8)
	mask.Allow(2)
	mask.Allow(5)
	tok, _, err := SampleTokenFused(logits, mask, nil, SamplingParams{VocabSize: 8}, 0)
	if err != nil {
		panic(err)
	}
	fmt.Println(tok)
	// Output: 5
}
