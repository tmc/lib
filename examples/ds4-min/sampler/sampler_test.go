package sampler

import (
	"math"
	"sort"
	"testing"
)

type referenceCandidate struct {
	id    int
	logit float32
}

func referenceBetter(a, b referenceCandidate) bool {
	return a.logit > b.logit || (a.logit == b.logit && a.id < b.id)
}

// referenceSample is deliberately independent of Sampler's scratch buffers
// and candidate insertion logic. It is the host oracle for the seeded replay
// gate, not a second implementation to use in production.
func referenceSample(logits []float32, p Params, r *RNG) int {
	if p.Temperature <= 0 || r == nil {
		return Argmax(logits)
	}
	topP, minP := p.TopP, p.MinP
	if topP <= 0 || topP > 1 {
		topP = 1
	}
	if minP < 0 {
		minP = 0
	}
	if p.TopK > 0 {
		k := p.TopK
		if k > 1024 {
			k = 1024
		}
		if k > len(logits) {
			k = len(logits)
		}
		cs := make([]referenceCandidate, 0, len(logits))
		for id, v := range logits {
			if isCandidate(v) {
				cs = append(cs, referenceCandidate{id: id, logit: v})
			}
		}
		sort.Slice(cs, func(i, j int) bool { return referenceBetter(cs[i], cs[j]) })
		if len(cs) > k {
			cs = cs[:k]
		}
		return referenceRanked(cs, p.Temperature, topP, minP, r)
	}
	best := Argmax(logits)
	if minP > 1 {
		return best
	}
	if topP < 1 {
		cs := make([]referenceCandidate, 0, len(logits))
		for id, v := range logits {
			if isCandidate(v) {
				cs = append(cs, referenceCandidate{id: id, logit: v})
			}
		}
		sort.Slice(cs, func(i, j int) bool { return referenceBetter(cs[i], cs[j]) })
		return referenceRanked(cs, p.Temperature, topP, minP, r)
	}
	max := negInf
	for _, v := range logits {
		if isCandidate(v) && v > max {
			max = v
		}
	}
	if max == negInf {
		return best
	}
	probs := make([]float32, len(logits))
	var sum float32
	for i, v := range logits {
		if isCandidate(v) {
			q := float32(math.Exp(float64(float32((v - max) / p.Temperature))))
			if q >= minP {
				probs[i] = q
				sum = float32(sum + q)
			}
		}
	}
	if !(sum > 0) || !isFinite(sum) {
		return best
	}
	roll := float32(r.Float32() * sum)
	for i, q := range probs {
		if q == 0 {
			continue
		}
		roll = float32(roll - q)
		if roll <= 0 {
			return i
		}
	}
	return best
}

func referenceRanked(cs []referenceCandidate, temperature, topP, minP float32, r *RNG) int {
	if len(cs) == 0 {
		return 0
	}
	probs := make([]float32, len(cs))
	var sum float32
	for i, c := range cs {
		q := float32(math.Exp(float64(float32((c.logit - cs[0].logit) / temperature))))
		probs[i] = q
		sum = float32(sum + q)
	}
	if !(sum > 0) || !isFinite(sum) {
		return cs[0].id
	}
	minProb := float32((probs[0] / sum) * minP)
	end := len(probs)
	var filtered float32
	for i, q := range probs {
		if i > 0 && q/sum < minProb {
			end = i
			break
		}
		filtered = float32(filtered + q)
		if filtered/sum >= topP {
			end = i + 1
			break
		}
	}
	roll := float32(r.Float32() * filtered)
	for i := 0; i < end; i++ {
		roll = float32(roll - probs[i])
		if roll <= 0 {
			return cs[i].id
		}
	}
	return cs[end-1].id
}

func TestRNGStreams(t *testing.T) {
	tests := []struct {
		name string
		seed uint64
	}{
		{"zero", 0},
		{"one", 1},
		{"large", 12345},
		{"golden", zeroState},
		{"max", ^uint64(0)},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r := RNG{State: tc.seed}
			seen := make(map[uint64]bool, 10)
			for i := 0; i < 10; i++ {
				v := r.Uint64()
				if seen[v] {
					t.Fatalf("duplicate output at %d: %x", i, v)
				}
				seen[v] = true
			}
			if r.State == 0 {
				t.Fatal("state returned to zero")
			}
		})
	}
	var zero, golden RNG
	for i := 0; i < 10; i++ {
		if got, want := zero.Uint64(), golden.Uint64(); got != want {
			t.Fatalf("zero/golden stream differs at %d: %x != %x", i, got, want)
		}
	}
	for _, seed := range []uint64{0, 1, zeroState, ^uint64(0)} {
		r := RNG{State: seed}
		for i := 0; i < 10; i++ {
			v := r.Float32()
			if v < 0 || v >= 1 || math.IsNaN(float64(v)) {
				t.Fatalf("seed %x output %d = %g outside [0,1)", seed, i, v)
			}
		}
	}
	t.Logf("RNG gate: cases=%d ran=%d skipped=0", len(tests), len(tests))
}

func TestArgmax(t *testing.T) {
	tests := []struct {
		name string
		in   []float32
		want int
	}{
		{"empty", nil, 0},
		{"tie-lower-id", []float32{4, 8, 8, 1}, 1},
		{"nan-first", []float32{float32(math.NaN()), 2, 1}, 1},
		{"all-sentinel", []float32{-1e30, float32(math.NaN()), -2e30}, 0},
		{"largest", []float32{-2, -1, -3}, 1},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := Argmax(tc.in); got != tc.want {
				t.Fatalf("Argmax(%v) = %d, want %d", tc.in, got, tc.want)
			}
		})
	}
	t.Logf("T=0 argmax gate: cases=%d ran=%d skipped=0", len(tests), len(tests))
}

func TestTemperatureZeroDoesNotAdvanceRNG(t *testing.T) {
	logits := []float32{1, 4, 4, 2}
	r := RNG{State: 12345}
	want := r.State
	if got := New(len(logits)).Sample(logits, Params{Temperature: 0}, &r); got != 1 {
		t.Fatalf("sample = %d, want tie's lowest id 1", got)
	}
	if r.State != want {
		t.Fatalf("temperature-zero sample advanced RNG from %x to %x", want, r.State)
	}
}

func TestSeededReplay(t *testing.T) {
	logits := []float32{3, 2, 1, 0, -1}
	params := []Params{
		{Temperature: 0.6, TopP: 1, MinP: 0},
		{Temperature: 1, TopP: 0.95, MinP: 0.05},
		{Temperature: 1.3, TopP: 0.5, MinP: 0.2},
	}
	seeds := []uint64{0, 1, 12345, zeroState, ^uint64(0)}
	rows, moved := 0, 0
	for _, p := range params {
		for _, seed := range seeds {
			a, b := RNG{State: seed}, RNG{State: seed}
			for step := 0; step < 8; step++ {
				before := a.State
				got, again := New(len(logits)).Sample(logits, p, &a), New(len(logits)).Sample(logits, p, &b)
				if got != again || a.State != b.State {
					t.Fatalf("seed %x params %+v step %d: (%d,%x) != (%d,%x)", seed, p, step, got, a.State, again, b.State)
				}
				if p.Temperature > 0 && a.State != before {
					moved++
				}
				rows++
			}
		}
	}
	if moved == 0 {
		t.Fatal("positive-temperature replay never advanced RNG")
	}
	t.Logf("seeded replay gate: cases=%d ran=%d skipped=0 positive-temperature RNG rows=%d", len(params)*len(seeds), rows, moved)
}

func TestSeededReplayReferenceGrid(t *testing.T) {
	logits := make([]float32, 257)
	for i := range logits {
		logits[i] = float32((i*37%101)-50) / 17
	}
	logits[0] = float32(math.NaN())
	logits[17] = negInf
	logits[83], logits[84] = 2, 2
	params := []Params{
		{Temperature: 0},
		{Temperature: .6, TopP: 1, MinP: 0},
		{Temperature: 1, TopP: 1, MinP: .05},
		{Temperature: 1.3, TopP: .95, MinP: 0},
		{Temperature: 1, TopP: .9, MinP: .2},
		{Temperature: 1, TopP: .5, MinP: .05, TopK: 40},
		{Temperature: 1, TopP: .95, MinP: .05, TopK: 2000},
	}
	rows, moved := 0, 0
	for pi, p := range params {
		for seed := uint64(0); seed < 256; seed++ {
			gotR, wantR := RNG{State: seed}, RNG{State: seed}
			gotS := New(len(logits))
			for step := 0; step < 4; step++ {
				before := gotR.State
				got := gotS.Sample(logits, p, &gotR)
				want := referenceSample(logits, p, &wantR)
				if got != want || gotR.State != wantR.State {
					t.Fatalf("grid row p=%d seed=%d step=%d: got (%d,%016x), want (%d,%016x)", pi, seed, step, got, gotR.State, want, wantR.State)
				}
				if p.Temperature > 0 && gotR.State != before {
					moved++
				}
				rows++
			}
		}
	}
	if rows != len(params)*256*4 || moved == 0 {
		t.Fatalf("replay grid rows=%d moved=%d", rows, moved)
	}
	t.Logf("seeded reference gate: parameter_sets=%d seeds=256 steps=4 rows=%d positive_temperature_rng_rows=%d", len(params), rows, moved)
}

// TestSamplingDistribution is the host statistical oracle for the sampler's
// documented full-vocabulary path.  The logits are fixed, the RNG stream is
// fixed, and the expected probabilities are the mathematical softmax of those
// logits.  The six-sigma count bound is deliberately wider than the nominal
// 99.9% binomial interval because the xorshift stream is deterministic rather
// than a source of independent random variables.
func TestSamplingDistribution(t *testing.T) {
	logits := []float32{0, -0.5108256, -0.9162907}
	want := []float64{0.5, 0.3, 0.2}
	const draws = 200000
	counts := make([]int, len(logits))
	s := New(len(logits))
	for i := 0; i < draws; i++ {
		id := s.Sample(logits, Params{Temperature: 1, TopP: 1}, &RNG{State: uint64(i + 1)})
		if id < 0 || id >= len(counts) {
			t.Fatalf("draw %d selected invalid token %d", i, id)
		}
		counts[id]++
	}
	for id, p := range want {
		mean := float64(draws) * p
		margin := 6*math.Sqrt(float64(draws)*p*(1-p)) + 8
		if math.Abs(float64(counts[id])-mean) > margin {
			t.Fatalf("token %d count=%d, want %.0f +/- %.0f (p=%.3f, draws=%d)", id, counts[id], mean, margin, p, draws)
		}
	}
	t.Logf("distribution gate: draws=%d counts=%v expected=%v bound=six-sigma+8", draws, counts, want)
}

// TestSamplingDistributionRejectsMutation keeps the statistical gate honest:
// an always-argmax sampler must not be mistaken for a sampler that preserves
// the registered distribution.  This is a host-only mutation control; it does
// not claim anything about an upstream or device runtime.
func TestSamplingDistributionRejectsMutation(t *testing.T) {
	logits := []float32{0, -0.5108256, -0.9162907}
	want := []float64{0.5, 0.3, 0.2}
	const draws = 256
	counts := make([]int, len(logits))
	for i := 0; i < draws; i++ {
		counts[Argmax(logits)]++ // deliberate always-argmax mutation
	}
	for id, p := range want {
		mean := float64(draws) * p
		margin := 6*math.Sqrt(float64(draws)*p*(1-p)) + 8
		if math.Abs(float64(counts[id])-mean) <= margin {
			t.Fatalf("mutation was not rejected for token %d: count=%d mean=%.1f bound=%.1f", id, counts[id], mean, margin)
		}
	}
}

func TestRNGReferenceState(t *testing.T) {
	for _, seed := range []uint64{0, 1, 12345, zeroState, ^uint64(0)} {
		r := RNG{State: seed}
		for i := 0; i < 1024; i++ {
			x := r.State
			if x == 0 {
				x = zeroState
			}
			x ^= x >> 12
			x ^= x << 25
			x ^= x >> 27
			want := x * 0x2545f4914f6cdd1d
			if got := r.Uint64(); got != want || r.State != x {
				t.Fatalf("seed=%x step=%d: got (%x,%x), want (%x,%x)", seed, i, got, r.State, want, x)
			}
		}
	}
}

func TestSamplingDegenerateLogits(t *testing.T) {
	tests := []struct {
		name string
		in   []float32
	}{
		{"nan", []float32{float32(math.NaN()), float32(math.NaN())}},
		{"sentinel", []float32{-1e30, -2e30}},
		{"one-finite", []float32{float32(math.NaN()), -1, float32(math.NaN())}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r := RNG{State: 7}
			before := r.State
			got := New(len(tc.in)).Sample(tc.in, Default, &r)
			if tc.name != "one-finite" && r.State != before {
				t.Fatalf("degenerate case advanced RNG: %x -> %x", before, r.State)
			}
			if tc.name == "one-finite" && got != 1 {
				t.Fatalf("one finite token = %d, want 1", got)
			}
		})
	}
	t.Logf("expf/sampling edge gate: cases=%d ran=%d skipped=0", len(tests), len(tests))
}

func TestTopKDegenerateLogitsDoNotDraw(t *testing.T) {
	logits := []float32{float32(math.NaN()), negInf, -2e30}
	r := RNG{State: 7}
	if got := New(len(logits)).Sample(logits, Params{Temperature: 1, TopK: 40, TopP: 1}, &r); got != 0 {
		t.Fatalf("all-invalid top-k sample = %d, want 0", got)
	}
	if r.State != 7 {
		t.Fatalf("all-invalid top-k advanced RNG: %x", r.State)
	}
}

func TestSamplingReferenceOrdering(t *testing.T) {
	tests := []struct {
		name   string
		logits []float32
		params Params
		seed   uint64
		want   int
	}{
		// top-p=1 deliberately walks vocabulary order, not logit order.
		{"full-vocabulary-order", []float32{1, 3, 2}, Params{Temperature: 1, TopP: 1}, 16, 0},
		// In the full path min-p is compared against the unnormalized exp
		// value. A normalized comparison would discard token 1 here.
		{"relative-min-p", []float32{3, 1.5}, Params{Temperature: 1, TopP: 1, MinP: 0.2}, 7, 1},
		{"top-k-tie-lower-id", []float32{2, 4, 4, 1}, Params{Temperature: 1, TopK: 2, TopP: 1}, 1, 1},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := New(len(tc.logits)).Sample(tc.logits, tc.params, &RNG{State: tc.seed})
			if got != tc.want {
				t.Fatalf("sample = %d, want %d", got, tc.want)
			}
		})
	}
	t.Logf("reference ordering gate: cases=%d ran=%d skipped=0", len(tests), len(tests))
}

func TestLogprobs(t *testing.T) {
	got := TopLogprobs([]float32{1, 3, 3, -1}, 3)
	if len(got) != 3 || got[0].ID != 1 || got[1].ID != 2 {
		t.Fatalf("top scores = %+v, want ids [1 2 ...]", got)
	}
	score, err := TokenLogprob([]float32{0, 1}, 1)
	if err != nil || score.ID != 1 || !(score.Logprob < 0) {
		t.Fatalf("token score = %+v, err=%v", score, err)
	}
}

func TestTopLogprobsRejectsNonFinite(t *testing.T) {
	scores := TopLogprobs([]float32{float32(math.Inf(1)), 3, float32(math.NaN()), 2}, 4)
	if len(scores) != 2 {
		t.Fatalf("scores = %d, want only finite logits", len(scores))
	}
	for _, score := range scores {
		if !isFinite(score.Logit) || !isFinite(score.Logprob) {
			t.Fatalf("non-finite score = %+v", score)
		}
	}
	if scores[0].ID != 1 || scores[1].ID != 3 {
		t.Fatalf("scores = %+v, want ids 1,3", scores)
	}
}
