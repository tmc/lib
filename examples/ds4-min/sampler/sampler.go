// Package sampler implements the host-side token sampler used by ds4.
package sampler

import (
	"errors"
	"math"
	"sort"
)

const zeroState = uint64(0x9e3779b97f4a7c15)
const negInf = float32(-1.0e30)

// Params controls one sampling decision. Temperature <= 0 selects Argmax.
type Params struct {
	Temperature float32
	TopK        int
	TopP        float32
	MinP        float32
}

// Default matches the upstream ds4 CLI defaults.
var Default = Params{Temperature: 1, TopP: 1, MinP: 0.05}

// RNG is the upstream xorshift64* pseudo-random stream. Its zero value is
// usable and is equivalent to a stream seeded with zeroState.
type RNG struct{ State uint64 }

// Uint64 returns the next value in the stream.
func (r *RNG) Uint64() uint64 {
	x := r.State
	if x == 0 {
		x = zeroState
	}
	x ^= x >> 12
	x ^= x << 25
	x ^= x >> 27
	r.State = x
	return x * 0x2545f4914f6cdd1d
}

// Float32 returns a uniformly distributed 24-bit value in [0, 1).
func (r *RNG) Float32() float32 {
	return float32((r.Uint64()>>40)&0xffffff) / float32(1<<24)
}

// Sampler owns scratch space for a sampler and is not safe for concurrent use.
type Sampler struct {
	candidates []candidate
	probs      []float32
}

type candidate struct {
	id    int
	logit float32
}

// New returns a sampler with capacity for vocab candidates.
func New(vocab int) *Sampler {
	if vocab < 0 {
		vocab = 0
	}
	return &Sampler{candidates: make([]candidate, 0, vocab), probs: make([]float32, 0, vocab)}
}

// Sample returns one token. It draws at most once and never draws when the
// temperature is non-positive or the distribution is degenerate.
func (s *Sampler) Sample(logits []float32, p Params, r *RNG) int {
	if p.Temperature <= 0 {
		return Argmax(logits)
	}
	if r == nil {
		return Argmax(logits)
	}
	topP := p.TopP
	if topP <= 0 || topP > 1 {
		topP = 1
	}
	minP := p.MinP
	if minP < 0 {
		minP = 0
	}
	if p.TopK > 0 {
		return s.sampleTopK(logits, p.Temperature, p.TopK, topP, minP, r)
	}
	return s.sampleFull(logits, p.Temperature, topP, minP, r)
}

// sampleTopK is kept separate because upstream's top-k path normalizes its
// min-p threshold, unlike the full-vocabulary top-p=1 fast path.
func (s *Sampler) sampleTopK(logits []float32, temperature float32, topK int, topP float32, minP float32, r *RNG) int {
	if topK > 1024 {
		topK = 1024
	}
	if topK > len(logits) {
		topK = len(logits)
	}
	if topK <= 0 {
		return Argmax(logits)
	}
	s.candidates = s.candidates[:0]
	for id, logit := range logits {
		if !isCandidate(logit) {
			continue
		}
		c := candidate{id: id, logit: logit}
		at := sort.Search(len(s.candidates), func(i int) bool { return better(c, s.candidates[i]) })
		if len(s.candidates) < topK {
			s.candidates = append(s.candidates, candidate{})
			copy(s.candidates[at+1:], s.candidates[at:])
			s.candidates[at] = c
		} else if at < topK {
			copy(s.candidates[at+1:], s.candidates[at:topK-1])
			s.candidates[at] = c
		}
	}
	if len(s.candidates) == 0 {
		return Argmax(logits)
	}
	return s.sampleRanked(temperature, topP, minP, r)
}

func (s *Sampler) sampleFull(logits []float32, temperature, topP, minP float32, r *RNG) int {
	best := Argmax(logits)
	if minP > 1 {
		return best
	}
	if topP < 1 {
		s.candidates = s.candidates[:0]
		for id, logit := range logits {
			if isCandidate(logit) {
				s.candidates = append(s.candidates, candidate{id: id, logit: logit})
			}
		}
		sort.Slice(s.candidates, func(i, j int) bool { return better(s.candidates[i], s.candidates[j]) })
		if len(s.candidates) == 0 {
			return best
		}
		return s.sampleRanked(temperature, topP, minP, r)
	}

	// The upstream top-p=1 path preserves vocabulary order and compares the
	// unnormalized exp value directly with min_p.
	s.probs = s.probs[:0]
	max := float32(-math.MaxFloat32)
	for _, v := range logits {
		if isCandidate(v) && v > max {
			max = v
		}
	}
	if max == float32(-math.MaxFloat32) || minP > 1 {
		return best
	}
	var sum float32
	scale := newTemperatureScale(temperature)
	for _, v := range logits {
		var q float32
		if isCandidate(v) {
			q = Exp(scale.apply(float32(v - max)))
			if q < minP {
				q = 0
			}
		}
		s.probs = append(s.probs, q)
		sum = float32(sum + q)
	}
	if !(sum > 0) || !isFinite(sum) {
		return best
	}
	return walk(logits, s.probs, sum, r)
}

func (s *Sampler) sampleRanked(temperature, topP, minP float32, r *RNG) int {
	s.probs = s.probs[:0]
	var sum float32
	scale := newTemperatureScale(temperature)
	for _, c := range s.candidates {
		q := Exp(scale.apply(float32(c.logit - s.candidates[0].logit)))
		s.probs = append(s.probs, q)
		sum = float32(sum + q)
	}
	if !(sum > 0) || !isFinite(sum) {
		return s.candidates[0].id
	}
	// Ranked paths apply min-p relative to the top normalized probability.
	min := float32((s.probs[0] / sum) * minP)
	end := len(s.candidates)
	var filtered float32
	for i, q := range s.probs {
		if i > 0 && q/sum < min {
			end = i
			break
		}
		filtered = float32(filtered + q)
		if filtered/sum >= topP {
			end = i + 1
			break
		}
	}
	if end == 0 {
		return s.candidates[0].id
	}
	roll := float32(r.Float32() * filtered)
	for i := 0; i < end; i++ {
		roll = float32(roll - s.probs[i])
		if roll <= 0 {
			return s.candidates[i].id
		}
	}
	return s.candidates[end-1].id
}

func walk(logits, probs []float32, sum float32, r *RNG) int {
	filtered := sum
	roll := float32(r.Float32() * filtered)
	for id, q := range probs {
		if q == 0 {
			continue
		}
		roll = float32(roll - q)
		if roll <= 0 {
			return id
		}
	}
	return Argmax(logits)
}

func better(a, b candidate) bool {
	return a.logit > b.logit || (a.logit == b.logit && a.id < b.id)
}

func isCandidate(v float32) bool { return !math.IsNaN(float64(v)) && v > negInf }
func isFinite(v float32) bool    { return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) }

// Argmax returns the lowest index of the maximum candidate. NaNs and values
// at or below the upstream sentinel are ignored; 0 is returned if none win.
func Argmax(logits []float32) int {
	best, bestV := 0, negInf
	for i, v := range logits {
		if isCandidate(v) && v > bestV {
			best, bestV = i, v
		}
	}
	return best
}

// TokenScore is one logprob result.
type TokenScore struct {
	ID      int     `json:"token"`
	Logit   float32 `json:"logit"`
	Logprob float32 `json:"logprob"`
}

// TopLogprobs returns up to k finite scores in logit-descending order.
func TopLogprobs(logits []float32, k int) []TokenScore {
	if k <= 0 {
		return nil
	}
	all := make([]candidate, 0, len(logits))
	for id, v := range logits {
		// The logprob API is defined over finite logits. Sampling keeps the
		// upstream sentinel rules separately, but an infinity here would make
		// the float64 log-sum-exp produce NaN rather than a useful score.
		if isFinite(v) {
			all = append(all, candidate{id: id, logit: v})
		}
	}
	sort.Slice(all, func(i, j int) bool { return better(all[i], all[j]) })
	if k > len(all) {
		k = len(all)
	}
	if len(all) == 0 {
		return nil
	}
	max := float64(all[0].logit)
	var sum float64
	for _, c := range all {
		sum += math.Exp(float64(c.logit) - max)
	}
	logsum := max + math.Log(sum)
	out := make([]TokenScore, k)
	for i := range out {
		out[i] = TokenScore{ID: all[i].id, Logit: all[i].logit, Logprob: float32(float64(all[i].logit) - logsum)}
	}
	return out
}

// TokenLogprob returns one token's score, or an error for an invalid id.
func TokenLogprob(logits []float32, id int) (TokenScore, error) {
	if id < 0 || id >= len(logits) {
		return TokenScore{}, errors.New("token id out of range")
	}
	all := TopLogprobs(logits, len(logits))
	for _, score := range all {
		if score.ID == id {
			return score, nil
		}
	}
	return TokenScore{ID: id, Logit: logits[id], Logprob: negInf}, nil
}
