package specdecode

import (
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// A GrammarBitmask is the set of tokens a structured-output grammar allows
// in its current state, packed as XGrammar and llguidance pack it: bit v%32
// of Words[v/32] is set iff token v is allowed.
type GrammarBitmask struct {
	VocabSize int
	Words     []uint32
}

// NewGrammarBitmask returns a bitmask over vocabSize tokens that allows none.
func NewGrammarBitmask(vocabSize int) GrammarBitmask {
	return GrammarBitmask{VocabSize: vocabSize, Words: make([]uint32, (vocabSize+31)/32)}
}

// AllowAllGrammar returns a bitmask over vocabSize tokens that allows all.
func AllowAllGrammar(vocabSize int) GrammarBitmask {
	m := NewGrammarBitmask(vocabSize)
	for v := range vocabSize {
		m.Allow(v)
	}
	return m
}

// Allow adds token v to m.
func (m GrammarBitmask) Allow(v int) { m.Words[v/32] |= 1 << (v % 32) }

// Allowed reports whether m allows token v.
func (m GrammarBitmask) Allowed(v int) bool { return bitSet(m.Words, v) }

func bitSet(words []uint32, v int) bool { return words[v/32]>>(v%32)&1 != 0 }

// SamplingParams configures token sampling. The pipeline, applied to one
// row of logits, is
//
//  1. drop tokens the grammar bitmask disallows;
//  2. divide positive logits of already-seen tokens by RepetitionPenalty and
//     multiply negative ones by it (the CTRL / Hugging Face rule);
//  3. divide by Temperature (0 means greedy: emit the largest logit);
//  4. drop tokens whose probability is below MinP times the largest;
//  5. rank the rest by logit, ties to the lower token ID, and keep a token
//     iff its rank is below TopK and the probability of the tokens ranked
//     above it is below TopP (the joint top-k/top-p rule; both use the
//     distribution after step 4);
//  6. sample from the kept tokens in proportion to their probability.
//
// A zero field disables its step, as do TopP >= 1 and RepetitionPenalty 1.
type SamplingParams struct {
	VocabSize         int
	Temperature       float32
	TopK              int
	TopP              float32
	MinP              float32
	RepetitionPenalty float32
	Seed              uint64 // Philox4x32 key for the Gumbel noise
}

func (p SamplingParams) validate() error {
	switch {
	case p.VocabSize < 1 || p.VocabSize > 1<<24:
		return fmt.Errorf("sampling params: VocabSize %d outside 1..%d", p.VocabSize, 1<<24)
	case !(p.Temperature >= 0) || math.IsInf(float64(p.Temperature), 0):
		return fmt.Errorf("sampling params: Temperature %v", p.Temperature)
	case p.TopK < 0:
		return fmt.Errorf("sampling params: TopK %d < 0", p.TopK)
	case !(p.TopP >= 0):
		return fmt.Errorf("sampling params: TopP %v", p.TopP)
	case !(p.MinP >= 0 && p.MinP <= 1):
		return fmt.Errorf("sampling params: MinP %v outside [0, 1]", p.MinP)
	case !(p.RepetitionPenalty >= 0) || math.IsInf(float64(p.RepetitionPenalty), 0):
		return fmt.Errorf("sampling params: RepetitionPenalty %v", p.RepetitionPenalty)
	}
	return nil
}

func (p SamplingParams) greedy() bool  { return p.Temperature == 0 }
func (p SamplingParams) topP() bool    { return p.TopP > 0 && p.TopP < 1 }
func (p SamplingParams) penalty() bool { return p.RepetitionPenalty > 0 && p.RepetitionPenalty != 1 }

// temp and pen are the divisors the kernel embeds, 1 when unused.
func (p SamplingParams) temp() float32 {
	if p.greedy() {
		return 1
	}
	return p.Temperature
}

func (p SamplingParams) pen() float32 {
	if !p.penalty() {
		return 1
	}
	return p.RepetitionPenalty
}

// logMinP is ln(MinP) rounded to float32, the constant the kernel embeds,
// or 0 when min-p is off.
func (p SamplingParams) logMinP() float32 {
	if p.MinP == 0 {
		return 0
	}
	return float32(math.Log(float64(p.MinP)))
}

func (p SamplingParams) checkRow(logits []float32, mask GrammarBitmask, seen []uint32) error {
	if err := p.validate(); err != nil {
		return err
	}
	w := (p.VocabSize + 31) / 32
	switch {
	case len(logits) != p.VocabSize:
		return fmt.Errorf("sample: %d logits, want %d", len(logits), p.VocabSize)
	case mask.VocabSize != p.VocabSize || len(mask.Words) != w:
		return fmt.Errorf("sample: bitmask over %d tokens in %d words, want %d in %d", mask.VocabSize, len(mask.Words), p.VocabSize, w)
	case seen != nil && len(seen) != w:
		return fmt.Errorf("sample: %d seen words, want %d", len(seen), w)
	}
	return nil
}

// penalized returns logit v after step 2 of the pipeline, in float32.
func (p SamplingParams) penalized(logits []float32, seen []uint32, v int) float32 {
	z := logits[v]
	if p.penalty() && seen != nil && bitSet(seen, v) {
		if z > 0 {
			z /= p.RepetitionPenalty
		} else {
			z *= p.RepetitionPenalty
		}
	}
	return z
}

// Probs returns the exact distribution of the sampling pipeline over the
// vocabulary, computed in float64 by sorting. It is the reference for
// [SampleTokenFused] and [GrammarSampler]. seen is a packed bitmask of
// tokens the penalty applies to, or nil.
func (p SamplingParams) Probs(logits []float32, mask GrammarBitmask, seen []uint32) ([]float64, error) {
	if err := p.checkRow(logits, mask, seen); err != nil {
		return nil, err
	}
	type tok struct {
		v int
		s float64
	}
	var toks []tok
	for v := range p.VocabSize {
		if mask.Allowed(v) {
			s := float64(p.penalized(logits, seen, v))
			if !p.greedy() {
				s /= float64(p.Temperature)
			}
			toks = append(toks, tok{v, s})
		}
	}
	if len(toks) == 0 {
		return nil, fmt.Errorf("sample: grammar allows no token")
	}
	slices.SortStableFunc(toks, func(a, b tok) int {
		switch {
		case a.s > b.s:
			return -1
		case a.s < b.s:
			return 1
		}
		return a.v - b.v
	})
	probs := make([]float64, p.VocabSize)
	if p.greedy() {
		probs[toks[0].v] = 1
		return probs, nil
	}
	m := toks[0].s
	if p.MinP > 0 {
		cut := math.Log(float64(p.MinP))
		toks = slices.DeleteFunc(toks, func(t tok) bool { return t.s-m < cut })
	}
	var z float64
	for _, t := range toks {
		z += math.Exp(t.s - m)
	}
	var above, kept float64
	for r, t := range toks {
		w := math.Exp(t.s-m) / z
		if (p.TopK == 0 || r < p.TopK) && (!p.topP() || above < float64(p.TopP)) {
			probs[t.v] = w
			kept += w
		}
		above += w
	}
	for v := range probs {
		probs[v] /= kept
	}
	return probs, nil
}

// SampleTokenFused samples one token from logits under mask and params, as
// [GrammarSampler] with 256 threads does on the device. step selects the
// noise: equal (Seed, step) pairs give equal tokens. logProb is the token's
// log-probability after steps 1-4 of the pipeline, before top-k/top-p
// truncation; it is 0 when Temperature is 0.
func SampleTokenFused(logits []float32, mask GrammarBitmask, seen []uint32, params SamplingParams, step uint64) (tokenID int32, logProb float32, err error) {
	return GrammarSampler{Params: params}.Sample(logits, mask, seen, step)
}

// GrammarSampler configures a kernel that samples one token per row of
// logits under a grammar bitmask without writing any vocabulary-sized
// buffer: each block reads its row of logits, bitmask and seen-token
// bitmask a few times and writes only the token and its log-probability.
//
// Sampling uses the Gumbel-max trick: the token maximizing
// logit/T + G_v, with G_v = -ln(-ln U_v) and U_v from Philox4x32, is a
// sample from softmax(logit/T). Disallowed tokens never enter the
// maximum. Top-k and top-p are applied by rejection, as FlashInfer's
// samplers do: a candidate whose rank or preceding probability mass puts it
// outside the kept set is rejected, and the next round, with fresh noise,
// samples only among tokens ranked above it. Every round shrinks the
// candidate set and the top-ranked token is always kept, so the loop ends,
// and conditioning on acceptance leaves exactly the kept set's distribution.
// A round costs two passes over the row.
//
// The logarithms use lg2.approx.ftz.f32 scaled by ln 2. Each block reduces
// in a fixed order, which [GrammarSampler.Sample] reproduces on the host, so
// the two agree except where float rounding of exp and log differs at a
// near tie.
//
// The kernel reads logits from global memory; fusing it into the epilogue
// of the LM-head GEMM, so that logits never reach memory at all, is not
// done here. It uses no sm_90+ instructions and was run on the host
// emulator; the CUDA_EGRAPH_HOPPER_NATIVE tests run it on the GPU.
type GrammarSampler struct {
	Name    string
	Params  SamplingParams
	Threads int // threads per block, a power of two in 32..1024 (0 means 256)
}

func (g GrammarSampler) norm() (GrammarSampler, error) {
	if g.Name == "" {
		g.Name = "grammar_sampler"
	}
	if !cIdent(g.Name) {
		return g, fmt.Errorf("grammar sampler: bad name %q", g.Name)
	}
	if g.Threads == 0 {
		g.Threads = 256
	}
	if g.Threads < 32 || g.Threads > 1024 || g.Threads&(g.Threads-1) != 0 {
		return g, fmt.Errorf("grammar sampler %s: Threads %d not a power of two in 32..1024", g.Name, g.Threads)
	}
	if err := g.Params.validate(); err != nil {
		return g, fmt.Errorf("grammar sampler %s: %w", g.Name, err)
	}
	return g, nil
}

// Grid returns the launch shape for rows rows: one block per row.
func (g GrammarSampler) Grid(rows int) (grid, block [3]int) {
	g, _ = g.norm()
	return [3]int{rows, 1, 1}, [3]int{g.Threads, 1, 1}
}

func (g GrammarSampler) params() []string {
	return []string{
		"const float* __restrict__ logits",
		"const uint32_t* __restrict__ mask",
		"const uint32_t* __restrict__ seen",
		"const uint64_t* __restrict__ steps",
		"int32_t* __restrict__ tokens",
		"float* __restrict__ logprobs",
	}
}

// CUDA returns the kernel (without [hopper.Prelude]). Its parameters are
// row-major logits [rows][VocabSize], mask and seen [rows][(VocabSize+31)/32]
// (seen may be null), steps [rows], and the outputs tokens and logprobs
// [rows]. A row whose grammar allows no token gets token -1.
func (g GrammarSampler) CUDA() (string, error) {
	g, err := g.norm()
	if err != nil {
		return "", err
	}
	p := g.Params
	var b strings.Builder
	b.WriteString(philoxDevice)
	b.WriteString(`#ifndef EG_SAMPLER_LN_DEFINED
#define EG_SAMPLER_LN_DEFINED 1
// eg_sampler_ln is ln x as lg2.approx.ftz.f32 times ln 2.
EG_DEV float eg_sampler_ln(float x) {
#if defined(EG_HOST_EMU)
    return log2f(x) * 0.6931472f;
#else
    float r;
    asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r * 0.6931472f;
#endif
}
#endif
`)
	fmt.Fprintf(&b, "// %s: grammar-masked sampler, vocab %d, T=%v top_k=%d top_p=%v min_p=%v penalty=%v.\n",
		g.Name, p.VocabSize, p.Temperature, p.TopK, p.TopP, p.MinP, p.RepetitionPenalty)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(%d) %s(\n    %s) {\n", g.Threads, g.Name, strings.Join(g.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int V = %d, W = (V + 31) / 32, T = %d, TOPK = %d;\n", p.VocabSize, g.Threads, p.TopK)
	fmt.Fprintf(&b, "    constexpr bool GREEDY = %v, PENALTY = %v, MINP = %v, TOPP = %v, FILTER = %v;\n",
		p.greedy(), p.penalty(), p.MinP > 0, p.topP(), !p.greedy() && (p.TopK > 0 || p.topP()))
	fmt.Fprintf(&b, "    constexpr float TEMP = %s, PEN = %s, LOG_MINP = %s, TOP_P = %s;\n",
		floatLiteral(p.temp()), floatLiteral(p.pen()), floatLiteral(p.logMinP()), floatLiteral(p.TopP))
	fmt.Fprintf(&b, "    constexpr uint32_t SEED_LO = %#xu, SEED_HI = %#xu;\n", uint32(p.Seed), uint32(p.Seed>>32))
	b.WriteString(`    const float NEG_INF = -__uint_as_float(0x7f800000u);
    __shared__ float red_f[T];
    __shared__ int32_t red_i[T];
    __shared__ float bcast_f[2];
    __shared__ int32_t bcast_i;
    const int row = blockIdx.x, tid = threadIdx.x;
    const float* z = logits + (size_t)row * V;
    const uint32_t* mk = mask + (size_t)row * W;
    const uint32_t* sn = seen ? seen + (size_t)row * W : nullptr;
    const uint64_t step = steps[row];

    // s returns token v's scaled logit, or -inf when the grammar forbids it.
    auto s_of = [&](int v) -> float {
        if (((mk[v >> 5] >> (v & 31)) & 1u) == 0u) return NEG_INF;
        float x = z[v];
        if (PENALTY && sn && ((sn[v >> 5] >> (v & 31)) & 1u)) x = x > 0.0f ? x / PEN : x * PEN;
        return GREEDY ? x : x / TEMP;
    };

    // Pass 1: the largest allowed scaled logit.
    float m = NEG_INF;
    for (int v = tid; v < V; v += T) m = fmaxf(m, s_of(v));
    red_f[tid] = m;
    __syncthreads();
    for (int h = T / 2; h > 0; h >>= 1) {
        if (tid < h) red_f[tid] = fmaxf(red_f[tid], red_f[tid + h]);
        __syncthreads();
    }
    const float M = red_f[0];
    __syncthreads();
    if (M == NEG_INF) {
        if (tid == 0) {
            tokens[row] = -1;
            logprobs[row] = NEG_INF;
        }
        return;
    }
    const float THETA = MINP ? M + LOG_MINP : NEG_INF;

    // Pass 2: the normalizer of the min-p-filtered distribution.
    float zsum = 1.0f;
    if (!GREEDY) {
        float acc = 0.0f;
        for (int v = tid; v < V; v += T) {
            const float s = s_of(v);
            if (s != NEG_INF && s >= THETA) acc += expf(s - M);
        }
        red_f[tid] = acc;
        __syncthreads();
        for (int h = T / 2; h > 0; h >>= 1) {
            if (tid < h) red_f[tid] += red_f[tid + h];
            __syncthreads();
        }
        zsum = red_f[0];
        __syncthreads();
    }

    // Rounds: Gumbel-max among tokens ranked above the last rejected one.
    bool has_lb = false;
    float lb_s = 0.0f, c_s = 0.0f;
    int32_t lb_v = 0, c = -1;
    for (uint32_t round = 0;; ++round) {
        float best = NEG_INF;
        int32_t best_v = -1;
        for (int v = tid; v < V; v += T) {
            const float s = s_of(v);
            if (s == NEG_INF || s < THETA) continue;
            if (has_lb && !(s > lb_s || (s == lb_s && v < lb_v))) continue;
            float sc = s;
            if (!GREEDY) {
                uint32_t w[4];
                eg_philox4x32((uint32_t)v, round, (uint32_t)step, (uint32_t)(step >> 32), SEED_LO, SEED_HI, w);
                const float u = eg_philox_open_uniform(w[0]);
                sc = s - eg_sampler_ln(-eg_sampler_ln(u));
            }
            if (best_v < 0 || sc > best) {
                best = sc;
                best_v = v;
            }
        }
        red_f[tid] = best;
        red_i[tid] = best_v;
        __syncthreads();
        for (int h = T / 2; h > 0; h >>= 1) {
            if (tid < h) {
                const float o = red_f[tid + h];
                const int32_t ov = red_i[tid + h];
                if (ov >= 0 && (red_i[tid] < 0 || o > red_f[tid] || (o == red_f[tid] && ov < red_i[tid]))) {
                    red_f[tid] = o;
                    red_i[tid] = ov;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            bcast_i = red_i[0];
            bcast_f[0] = red_i[0] >= 0 ? s_of(red_i[0]) : NEG_INF;
        }
        __syncthreads();
        c = bcast_i;
        c_s = bcast_f[0];
        __syncthreads();
        if (!FILTER || c < 0) break;

        // Count and weigh the kept tokens ranked above c.
        int cnt = 0;
        float mass = 0.0f;
        for (int v = tid; v < V; v += T) {
            const float s = s_of(v);
            if (s == NEG_INF || s < THETA) continue;
            if (s > c_s || (s == c_s && v < c)) {
                ++cnt;
                mass += expf(s - M);
            }
        }
        red_f[tid] = mass;
        red_i[tid] = cnt;
        __syncthreads();
        for (int h = T / 2; h > 0; h >>= 1) {
            if (tid < h) {
                red_f[tid] += red_f[tid + h];
                red_i[tid] += red_i[tid + h];
            }
            __syncthreads();
        }
        const bool accept = (TOPK == 0 || red_i[0] < TOPK) && (!TOPP || red_f[0] < TOP_P * zsum);
        __syncthreads();
        if (accept) break;
        has_lb = true;
        lb_s = c_s;
        lb_v = c;
    }
    if (tid == 0) {
        tokens[row] = c;
        logprobs[row] = GREEDY ? 0.0f : (c_s - M) - logf(zsum);
    }
}
`)
	return b.String(), nil
}

// Launcher returns a host function
//
//	void Name_launch(int rows, <kernel params...>, cudaStream_t stream)
func (g GrammarSampler) Launcher() (string, error) {
	g, err := g.norm()
	if err != nil {
		return "", err
	}
	params := g.params()
	var b strings.Builder
	fmt.Fprintf(&b, "inline void %s_launch(int rows, %s, cudaStream_t stream) {\n", g.Name, strings.Join(params, ", "))
	fmt.Fprintf(&b, "    const dim3 grid(rows, 1, 1), block(%d, 1, 1);\n", g.Threads)
	fmt.Fprintf(&b, "    EG_LAUNCH(%s, grid, block, 0, stream, %s);\n}\n", g.Name, strings.Join(paramNames(params), ", "))
	return b.String(), nil
}

// Source returns [hopper.Prelude], the kernel and its launcher.
func (g GrammarSampler) Source() (string, error) {
	src, err := g.CUDA()
	if err != nil {
		return "", err
	}
	l, err := g.Launcher()
	if err != nil {
		return "", err
	}
	return hopper.Prelude + "\n" + src + "\n" + l, nil
}

// Sample runs the kernel's algorithm for one row on the host, in float32,
// with the kernel's per-thread partition and reduction order. It returns
// an error, not token -1, when the grammar allows no token.
func (g GrammarSampler) Sample(logits []float32, mask GrammarBitmask, seen []uint32, step uint64) (int32, float32, error) {
	g, err := g.norm()
	if err != nil {
		return 0, 0, err
	}
	p := g.Params
	if err := p.checkRow(logits, mask, seen); err != nil {
		return 0, 0, err
	}
	negInf := float32(math.Inf(-1))
	n := p.VocabSize
	s := make([]float32, n)
	m := negInf
	for v := range n {
		s[v] = negInf
		if mask.Allowed(v) {
			x := p.penalized(logits, seen, v)
			if !p.greedy() {
				x /= p.Temperature
			}
			s[v] = x
		}
		m = max(m, s[v])
	}
	if m == negInf {
		return 0, 0, fmt.Errorf("sample: grammar allows no token")
	}
	theta := negInf
	if p.MinP > 0 {
		theta = m + p.logMinP()
	}
	kept := func(v int) bool { return s[v] != negInf && s[v] >= theta }
	expm := func(x float32) float32 { return float32(math.Exp(float64(x - m))) }
	zsum := float32(1)
	if !p.greedy() {
		zsum = g.blockSum(func(v int) float32 {
			if kept(v) {
				return expm(s[v])
			}
			return 0
		})
	}
	filter := !p.greedy() && (p.TopK > 0 || p.topP())
	ranksAbove := func(v, c int) bool { return s[v] > s[c] || (s[v] == s[c] && v < c) }
	lb := -1
	for round := uint32(0); ; round++ {
		c := -1
		var best float32
		for v := range n {
			if !kept(v) || (lb >= 0 && !ranksAbove(v, lb)) {
				continue
			}
			sc := s[v]
			if !p.greedy() {
				w := Philox4x32([4]uint32{uint32(v), round, uint32(step), uint32(step >> 32)}, [2]uint32{uint32(p.Seed), uint32(p.Seed >> 32)})
				sc = s[v] - samplerLn(-samplerLn(philoxOpenUniform(w[0])))
			}
			if c < 0 || sc > best {
				best, c = sc, v
			}
		}
		if !filter {
			return g.result(c, s, m, zsum)
		}
		cnt := 0
		for v := range n {
			if kept(v) && ranksAbove(v, c) {
				cnt++
			}
		}
		mass := g.blockSum(func(v int) float32 {
			if kept(v) && ranksAbove(v, c) {
				return expm(s[v])
			}
			return 0
		})
		if (p.TopK == 0 || cnt < p.TopK) && (!p.topP() || mass < p.TopP*zsum) {
			return g.result(c, s, m, zsum)
		}
		lb = c
	}
}

func (g GrammarSampler) result(c int, s []float32, m, zsum float32) (int32, float32, error) {
	if g.Params.greedy() {
		return int32(c), 0, nil
	}
	return int32(c), (s[c] - m) - float32(math.Log(float64(zsum))), nil
}

// blockSum sums f over the vocabulary as the kernel does: thread t adds
// tokens t, t+T, ... in order, then the block adds pairs in a halving tree.
func (g GrammarSampler) blockSum(f func(v int) float32) float32 {
	red := make([]float32, g.Threads)
	for t := range red {
		for v := t; v < g.Params.VocabSize; v += g.Threads {
			red[t] += f(v)
		}
	}
	for h := g.Threads / 2; h > 0; h >>= 1 {
		for t := range h {
			red[t] += red[t+h]
		}
	}
	return red[0]
}

// samplerLn mirrors eg_sampler_ln: log2 rounded to float32, times ln 2.
func samplerLn(x float32) float32 {
	// The outer conversion keeps the compiler from fusing the product
	// into a caller's subtraction, which the emulator does not do.
	return float32(float32(math.Log2(float64(x))) * float32(math.Ln2))
}
