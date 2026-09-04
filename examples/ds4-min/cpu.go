package main

import (
	"errors"
	"fmt"
	"math"
	"sort"

	"github.com/tmc/lib/lm/moe"
)

var errShape = errors.New("invalid tensor shape")

func rmsNorm(x, weight []float32, eps float32) ([]float32, error) {
	if len(x) == 0 || len(x) != len(weight) || eps <= 0 {
		return nil, errShape
	}
	var sum float64
	for _, v := range x {
		sum += float64(v * v)
	}
	scale := float32(1 / math.Sqrt(sum/float64(len(x))+float64(eps)))
	out := make([]float32, len(x))
	for i := range x {
		out[i] = x[i] * scale * weight[i]
	}
	return out, nil
}

func matvec(matrix, x []float32, rows, cols int) ([]float32, error) {
	if rows <= 0 || cols <= 0 || len(matrix) != rows*cols || len(x) != cols {
		return nil, errShape
	}
	out := make([]float32, rows)
	for row := range rows {
		for col := range cols {
			out[row] += matrix[row*cols+col] * x[col]
		}
	}
	return out, nil
}

func rotary(x []float32, position int, base float64) ([]float32, error) {
	if len(x) == 0 || len(x)%2 != 0 || base <= 0 {
		return nil, errShape
	}
	out := append([]float32(nil), x...)
	for i := 0; i < len(x); i += 2 {
		angle := float64(position) / math.Pow(base, float64(i)/float64(len(x)))
		c, s := float32(math.Cos(angle)), float32(math.Sin(angle))
		out[i] = x[i]*c - x[i+1]*s
		out[i+1] = x[i]*s + x[i+1]*c
	}
	return out, nil
}

type attentionRow struct {
	key, value []float32
	index      []float32
	end        int
}

type compressorWeights struct {
	kv, gate []float32
	ape      []float32
	norm     []float32
}

type compressorState struct {
	kv, score []float32
}

type compressedAttention struct {
	dim, hidden, window, ratio int
	attention, indexer         compressorWeights
	attentionState, indexState compressorState
	indexQuery                 []float32
	indexWeight                []float32
	sink                       float32
	raw                        []attentionRow
	compressed                 []attentionRow
}

func newCompressedAttention(dim, hidden, window, ratio int, attention, indexer compressorWeights, indexQuery, indexWeight []float32, sink float32) (*compressedAttention, error) {
	width := dim
	if ratio == 4 {
		width *= 2
	}
	rows := ratio
	if ratio == 4 {
		rows *= 2
	}
	valid := func(w compressorWeights) bool {
		return len(w.kv) == width*hidden && len(w.gate) == width*hidden && len(w.ape) == ratio*width && len(w.norm) == dim
	}
	if dim <= 0 || hidden <= 0 || window <= 0 || ratio <= 0 || !valid(attention) || !valid(indexer) || len(indexQuery) != dim*hidden || len(indexWeight) != hidden {
		return nil, errShape
	}
	attentionState := compressorState{make([]float32, rows*width), make([]float32, rows*width)}
	indexState := compressorState{make([]float32, rows*width), make([]float32, rows*width)}
	for i := range attentionState.score {
		attentionState.score[i] = -math.MaxFloat32
		indexState.score[i] = -math.MaxFloat32
	}
	return &compressedAttention{dim: dim, hidden: hidden, window: window, ratio: ratio,
		attention: attention, indexer: indexer, indexQuery: indexQuery, indexWeight: indexWeight, sink: sink,
		attentionState: attentionState, indexState: indexState}, nil
}

func (s *compressedAttention) Add(x, kv []float32, position int) error {
	if len(x) != s.hidden || len(kv) != s.dim || position < 0 {
		return errShape
	}
	row := attentionRow{key: append([]float32(nil), kv...), value: append([]float32(nil), kv...), end: position}
	s.raw = append(s.raw, row)
	if len(s.raw) > s.window {
		s.raw = s.raw[len(s.raw)-s.window:]
	}
	compressed, emit, err := compressorUpdate(x, position, s.dim, s.hidden, s.ratio, s.attention, &s.attentionState)
	if err != nil {
		return err
	}
	indexed, indexEmit, err := compressorUpdate(x, position, s.dim, s.hidden, s.ratio, s.indexer, &s.indexState)
	if err != nil {
		return fmt.Errorf("update index compressor: %w", err)
	}
	if emit != indexEmit {
		return errors.New("compressor lanes disagree")
	}
	if emit {
		s.compressed = append(s.compressed, attentionRow{compressed, append([]float32(nil), compressed...), indexed, position})
	}
	return nil
}

func (s *compressedAttention) Attend(query, cur, qrNorm []float32, position, topK int) ([]float32, []int, error) {
	if len(query) != s.dim || len(cur) != s.hidden || len(qrNorm) != s.hidden || position < 0 || topK < 0 {
		return nil, nil, errShape
	}
	if len(s.raw) == 0 {
		return nil, nil, errors.New("empty attention state")
	}
	indexQuery, err := matvec(s.indexQuery, qrNorm, s.dim, s.hidden)
	if err != nil {
		return nil, nil, err
	}
	indexQuery, err = rotary(indexQuery, position, 10000)
	if err != nil {
		return nil, nil, err
	}
	indexWeight, err := matvec(s.indexWeight, cur, 1, s.hidden)
	if err != nil {
		return nil, nil, err
	}
	indexWeight[0] /= float32(math.Sqrt(float64(s.dim)))
	oldestRaw := s.raw[0].end
	type scored struct {
		row   attentionRow
		index int
		score float32
	}
	var candidates []scored
	for i, row := range s.compressed {
		if row.end < oldestRaw {
			score := max(float32(0), dot(indexQuery, row.index)) * indexWeight[0]
			candidates = append(candidates, scored{row, i, score})
		}
	}
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].score != candidates[j].score {
			return candidates[i].score > candidates[j].score
		}
		return candidates[i].index < candidates[j].index
	})
	if topK < len(candidates) {
		candidates = candidates[:topK]
	}
	rows := make([]attentionRow, 0, len(candidates)+len(s.raw))
	selected := make([]int, len(candidates))
	for i, candidate := range candidates {
		rows = append(rows, candidate.row)
		selected[i] = candidate.index
	}
	rows = append(rows, s.raw...)
	if len(rows) == 0 {
		return nil, nil, errors.New("empty attention state")
	}
	scores := make([]float32, len(rows))
	maxScore := s.sink
	for i, row := range rows {
		scores[i] = dot(query, row.key) / float32(math.Sqrt(float64(s.dim)))
		if scores[i] > maxScore {
			maxScore = scores[i]
		}
	}
	denom := float32(math.Exp(float64(s.sink - maxScore)))
	for i := range scores {
		scores[i] = float32(math.Exp(float64(scores[i] - maxScore)))
		denom += scores[i]
	}
	out := make([]float32, s.dim)
	for i, row := range rows {
		for d := range s.dim {
			out[d] += scores[i] / denom * row.value[d]
		}
	}
	return out, selected, nil
}

func compressorUpdate(x []float32, position, dim, hidden, ratio int, weights compressorWeights, state *compressorState) ([]float32, bool, error) {
	width := dim
	rows := ratio
	if ratio == 4 {
		width *= 2
		rows *= 2
	}
	if len(x) != hidden || len(state.kv) != rows*width || len(state.score) != rows*width {
		return nil, false, errShape
	}
	kv, err := matvec(weights.kv, x, width, hidden)
	if err != nil {
		return nil, false, err
	}
	score, _ := matvec(weights.gate, x, width, hidden)
	mod := position % ratio
	for j := range width {
		score[j] += weights.ape[mod*width+j]
	}
	row := mod
	if ratio == 4 {
		row += ratio
	}
	copy(state.kv[row*width:], kv)
	copy(state.score[row*width:], score)
	if (position+1)%ratio != 0 {
		return nil, false, nil
	}
	pooled := make([]float32, dim)
	for j := range dim {
		maxScore := float32(-math.MaxFloat32)
		for r := range ratio {
			maxScore = max(maxScore, state.score[r*width+j])
			if ratio == 4 {
				maxScore = max(maxScore, state.score[(ratio+r)*width+dim+j])
			}
		}
		var denom, sum float32
		for r := range ratio {
			weight := float32(math.Exp(float64(state.score[r*width+j] - maxScore)))
			denom += weight
			sum += weight * state.kv[r*width+j]
			if ratio == 4 {
				weight = float32(math.Exp(float64(state.score[(ratio+r)*width+dim+j] - maxScore)))
				denom += weight
				sum += weight * state.kv[(ratio+r)*width+dim+j]
			}
		}
		pooled[j] = sum / denom
	}
	pooled, err = rmsNorm(pooled, weights.norm, 1e-5)
	if err != nil {
		return nil, false, err
	}
	pooled, err = rotary(pooled, position+1-ratio, 10000)
	if ratio == 4 {
		// Seed both lanes with the completed current window. Later tokens replace
		// rows in the upper lane one at a time, matching compressor_decode_one.
		copy(state.kv[:ratio*width], state.kv[ratio*width:])
		copy(state.score[:ratio*width], state.score[ratio*width:])
		copy(state.kv[ratio*width:], state.kv[:ratio*width])
		copy(state.score[ratio*width:], state.score[:ratio*width])
	}
	return pooled, true, err
}

func swiglu(gate, up []float32) ([]float32, error) {
	if len(gate) == 0 || len(gate) != len(up) {
		return nil, errShape
	}
	out := make([]float32, len(gate))
	for i := range gate {
		out[i] = gate[i] / (1 + float32(math.Exp(float64(-gate[i])))) * up[i]
	}
	return out, nil
}

func routeExperts(logits, bias []float32, topK int) (moe.RoutingDecision, error) {
	if len(logits) == 0 || len(logits) != len(bias) || topK <= 0 || topK > len(logits) {
		return moe.RoutingDecision{}, errShape
	}
	router := moe.NewGroupedTopKRouter(moe.MoEConfig{NumRoutedExperts: len(logits), TopK: topK})
	if err := router.SetBiases(bias); err != nil {
		return moe.RoutingDecision{}, err
	}
	return router.Route(logits), nil
}

func expertSum(input []float32, experts []tinyExpert, decision moe.RoutingDecision) ([]float32, error) {
	out := make([]float32, len(input))
	for slot, id := range decision.SelectedExperts {
		if id < 0 || id >= len(experts) || slot >= len(decision.Weights) {
			return nil, errShape
		}
		expert := experts[id]
		gate, err := matvec(expert.gate, input, expert.intermediate, len(input))
		if err != nil {
			return nil, err
		}
		up, _ := matvec(expert.up, input, expert.intermediate, len(input))
		hidden, _ := swiglu(gate, up)
		value, _ := matvec(expert.down, hidden, len(input), expert.intermediate)
		for i := range out {
			out[i] += decision.Weights[slot] * value[i]
		}
	}
	return out, nil
}

func hcSplit(mix, scale, base []float32, streams, iterations int, eps float32) (pre, post, combine []float32, err error) {
	want := streams * (streams + 2)
	if streams <= 0 || iterations <= 0 || len(mix) != want || len(base) != want || len(scale) != 3 {
		return nil, nil, nil, errShape
	}
	pre, post = make([]float32, streams), make([]float32, streams)
	combine = make([]float32, streams*streams)
	for i := range streams {
		pre[i] = sigmoid(mix[i]*scale[0]+base[i]) + eps
		post[i] = 2 * sigmoid(mix[streams+i]*scale[1]+base[streams+i])
	}
	for dst := range streams {
		maxValue := float32(-math.MaxFloat32)
		for src := range streams {
			i := src + dst*streams
			combine[i] = mix[2*streams+i]*scale[2] + base[2*streams+i]
			maxValue = max(maxValue, combine[i])
		}
		var sum float32
		for src := range streams {
			i := src + dst*streams
			combine[i] = float32(math.Exp(float64(combine[i] - maxValue)))
			sum += combine[i]
		}
		for src := range streams {
			i := src + dst*streams
			combine[i] = combine[i]/sum + eps
		}
	}
	for iteration := 0; iteration < iterations; iteration++ {
		for src := range streams {
			var sum float32
			for dst := range streams {
				sum += combine[src+dst*streams]
			}
			for dst := range streams {
				combine[src+dst*streams] /= sum + eps
			}
		}
		if iteration+1 == iterations {
			break
		}
		for dst := range streams {
			var sum float32
			for src := range streams {
				sum += combine[src+dst*streams]
			}
			for src := range streams {
				combine[src+dst*streams] /= sum + eps
			}
		}
	}
	return pre, post, combine, nil
}

func hcWeightedSum(state, weights []float32, dim int) ([]float32, error) {
	if dim <= 0 || len(weights) == 0 || len(state) != dim*len(weights) {
		return nil, errShape
	}
	out := make([]float32, dim)
	for stream, weight := range weights {
		for d := range dim {
			out[d] += state[stream*dim+d] * weight
		}
	}
	return out, nil
}

func hcPost(block, residual, post, combine []float32) ([]float32, error) {
	streams := len(post)
	if streams == 0 || len(combine) != streams*streams || len(residual)%streams != 0 {
		return nil, errShape
	}
	dim := len(residual) / streams
	if len(block) != dim {
		return nil, errShape
	}
	out := make([]float32, len(residual))
	for dst := range streams {
		for d := range dim {
			out[dst*dim+d] = block[d] * post[dst]
			for src := range streams {
				out[dst*dim+d] += combine[dst+src*streams] * residual[src*dim+d]
			}
		}
	}
	return out, nil
}

func dot(a, b []float32) float32 {
	var out float32
	for i := range a {
		out += a[i] * b[i]
	}
	return out
}

func sigmoid(x float32) float32 { return 1 / (1 + float32(math.Exp(float64(-x)))) }

func allFinite(values ...[]float32) bool {
	for _, vector := range values {
		for _, value := range vector {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return false
			}
		}
	}
	return true
}

type tinyExpert struct {
	intermediate int
	gate, up     []float32
	down         []float32
}

func requireFinite(name string, values ...[]float32) error {
	if !allFinite(values...) {
		return fmt.Errorf("%s produced non-finite values", name)
	}
	return nil
}
