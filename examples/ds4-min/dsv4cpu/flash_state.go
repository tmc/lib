package dsv4cpu

import (
	"errors"
	"fmt"
	"math"
	"runtime"
)

// Request ownership and the complete scalar computation are kept private
// until end-to-end source controls and the dsv4 State codec are qualified.
type flashState struct {
	model                    *flashModel
	context, position        int
	failed                   bool
	closed                   bool
	layers                   [43]flashLayerState
	logits                   []float32
	steering                 []float32
	attentionScale, ffnScale float32
	support                  *flashSupport
}

type flashLayerState struct {
	raw                   *Cache
	attention, index      *Compressor
	compressed, indexRows []float32
}

func (m *flashModel) newState(context int) (*flashState, error) {
	if m == nil || m.catalog == nil || context <= 0 || context > 1<<20 {
		return nil, errors.New("invalid Flash context")
	}
	s := &flashState{model: m, context: context, logits: make([]float32, 129280)}
	for il, l := range m.layers {
		ls := &s.layers[il]
		capacity := 128
		if context < capacity {
			capacity = context
		}
		var err error
		if ls.raw, err = NewCache(512, capacity); err != nil {
			return nil, err
		}
		if l.ratio != 0 {
			if ls.attention, err = NewCompressor(512, l.ratio); err != nil {
				return nil, err
			}
			ls.compressed = make([]float32, 0, (context/l.ratio+2)*512)
			if l.ratio == 4 {
				if ls.index, err = NewCompressor(128, 4); err != nil {
					return nil, err
				}
				ls.indexRows = make([]float32, 0, (context/4+2)*128)
			}
		}
	}
	return s, nil
}

func (s *flashState) reset() error {
	if s == nil || s.model == nil || s.closed {
		return errors.New("invalid Flash state")
	}
	for i := range s.layers {
		ls := &s.layers[i]
		ls.raw.Reset()
		if ls.attention != nil {
			if err := ls.attention.Reset(); err != nil {
				return err
			}
		}
		if ls.index != nil {
			if err := ls.index.Reset(); err != nil {
				return err
			}
		}
		clear(ls.compressed)
		clear(ls.indexRows)
		ls.compressed = ls.compressed[:0]
		ls.indexRows = ls.indexRows[:0]
	}
	clear(s.logits)
	if s.support != nil {
		if err := s.support.reset(); err != nil {
			s.failed = true
			return err
		}
	}
	s.position = 0
	s.failed = false
	return nil
}

func cpuDense(dst, x []float32, m *Matrix) error {
	if m == nil {
		return errors.New("missing dense tensor")
	}
	if m.encoding == Q80 {
		a := make([]Q80Activation, len(x)/32)
		if err := QuantizeQ80(a, x); err != nil {
			return err
		}
		return cpuDensePrequant(dst, a, m)
	}
	if m.encoding == F16 && runtime.GOARCH == "arm64" {
		return nativeF16Mul(dst, x, m)
	}
	return m.Mul(dst, x)
}

func cpuDensePrequant(dst []float32, a []Q80Activation, m *Matrix) error {
	if runtime.GOARCH == "arm64" {
		return nativeQ80Mul(dst, a, m)
	}
	return m.MulQ80(dst, a)
}

func (h *flashHC) pre(plain, post, comb, residual []float32) error {
	flat := make([]float32, 16384)
	if err := RMSNorm(flat, residual, nil, 1e-6); err != nil {
		return err
	}
	mix, split := make([]float32, 24), make([]float32, 24)
	var err error
	if h.nativeF16 {
		err = nativeF16Mul(mix, flat, h.projection)
	} else {
		err = h.projection.Mul(mix, flat)
	}
	if err != nil {
		return err
	}
	if err := Split(split, mix, h.scale, h.base, 4, 20, 1e-6); err != nil {
		return err
	}
	if err := Collapse(plain, residual, split[:4]); err != nil {
		return err
	}
	copy(post, split[4:8])
	copy(comb, split[8:])
	return nil
}

func flashRoPE(ratio int) RoPE {
	if ratio == 0 {
		return RoPE{Base: 10000, Scale: 1}
	}
	return RoPE{Base: 160000, Scale: 1.0 / 16, OriginalContext: 65536, BetaFast: 32, BetaSlow: 1}
}

func pushCompression(dst *[]float32, state *Compressor, w *flashCompression, x []float32, position, ratio, dim int) error {
	width := state.projectionWidth
	kv, score, bias := make([]float32, width), make([]float32, width), make([]float32, width)
	if err := cpuDense(kv, x, w.kv); err != nil {
		return err
	}
	if err := cpuDense(score, x, w.score); err != nil {
		return err
	}
	if err := w.bias.Row(bias, position%ratio); err != nil {
		return err
	}
	for i := range score {
		score[i] += bias[i]
	}
	pooled := make([]float32, dim)
	emitted, err := state.Push(pooled, kv, score, position)
	if err != nil {
		return err
	}
	if !emitted {
		return nil
	}
	if err := RMSNorm(pooled, pooled, w.norm, 1e-6); err != nil {
		return err
	}
	if err := flashRoPE(ratio).Rotate(pooled, dim, 64, position+1-ratio, false); err != nil {
		return err
	}
	if dim == 512 {
		err = QuantizeKV(pooled, 64)
	} else {
		err = QuantizeIndexer(pooled)
	}
	if err != nil {
		return err
	}
	if len(*dst)+dim > cap(*dst) {
		return errors.New("compressed cache capacity exceeded")
	}
	for _, v := range pooled {
		*dst = append(*dst, half(toHalf(v)))
	}
	return nil
}

func flashSelectRows(l *flashLayer, ls *flashLayerState, cur, qr []float32, pos int) ([]bool, error) {
	n := len(ls.indexRows) / 128
	if n != len(ls.compressed)/512 {
		return nil, errors.New("indexer and attention cache counts differ")
	}
	if n <= 512 {
		return nil, nil
	}
	q, weight := make([]float32, 8192), make([]float32, 64)
	if err := cpuDense(q, qr, l.indexQ); err != nil {
		return nil, err
	}
	if err := flashRoPE(l.ratio).Rotate(q, 128, 64, pos, false); err != nil {
		return nil, err
	}
	if err := QuantizeIndexer(q); err != nil {
		return nil, err
	}
	if err := cpuDense(weight, cur, l.indexProjection); err != nil {
		return nil, err
	}
	scale := float32(1) / float32(math.Sqrt(8192))
	for i := range weight {
		weight[i] *= scale
	}
	return selectCompressedRows(q, weight, ls.indexRows, 512)
}

func selectCompressedRows(query, weights, rows []float32, topK int) ([]bool, error) {
	if topK <= 0 || len(rows) == 0 || len(rows)%128 != 0 || len(query) != 8192 || len(weights) != 64 {
		return nil, errors.New("invalid indexer selection shape")
	}
	n := len(rows) / 128
	allowed := make([]bool, n)
	if n <= topK {
		for i := range allowed {
			allowed[i] = true
		}
		return allowed, nil
	}
	scores := make([]float32, n)
	for row := range scores {
		var score float32
		for h, w := range weights {
			var dot float32
			for d := 0; d < 128; d++ {
				dot += float32(query[h*128+d] * rows[row*128+d])
			}
			if dot < 0 {
				dot = 0
			}
			score += float32(dot * w)
		}
		if math.IsNaN(float64(score)) || math.IsInf(float64(score), 0) {
			return nil, errors.New("nonfinite indexer score")
		}
		scores[row] = score
	}
	for k := 0; k < topK; k++ {
		best := -1
		score := negativeInfinity
		for row, v := range scores {
			if !allowed[row] && v > score {
				best = row
				score = v
			}
		}
		if best < 0 {
			return nil, errors.New("indexer score below supported floor")
		}
		allowed[best] = true
	}
	return allowed, nil
}

func groupedOutput(dst, heads []float32, l *flashLayer) error {
	low := make([]float32, 8192)
	for group := 0; group < 8; group++ {
		m := l.outA
		slice, err := NewMatrix(m.source, m.offset+int64(group*1024*m.rowBytes), int64(1024*m.rowBytes), 1024, 4096, m.encoding)
		if err != nil {
			return err
		}
		if err := cpuDense(low[group*1024:(group+1)*1024], heads[group*4096:(group+1)*4096], slice); err != nil {
			return err
		}
	}
	return cpuDense(dst, low, l.outB)
}

func (m *flashModel) routed(dst, x []float32, l *flashLayer, il, token int) error {
	logits, probs := make([]float32, 256), make([]float32, 256)
	if err := l.router.Mul(logits, x); err != nil {
		return err
	}
	if err := RouterProbabilities(probs, logits); err != nil {
		return err
	}
	ids, weights := make([]int, 6), make([]float32, 6)
	if il < 3 {
		if err := m.selectedHash(ids, il, token); err != nil {
			return err
		}
		if err := ExpertWeights(weights, probs, ids, 1.5); err != nil {
			return err
		}
	} else {
		if err := SelectExperts(ids, weights, probs, l.expertBias, 1.5); err != nil {
			return err
		}
	}
	xq := make([]Q8K, 16)
	if err := QuantizeQ8K(xq, x); err != nil {
		return err
	}
	gate, up, mid, down := make([]float32, 2048), make([]float32, 2048), make([]float32, 2048), make([]float32, 4096)
	midq := make([]Q8K, 8)
	clear(dst)
	for slot, id := range ids {
		g, err := m.catalog.Matrix(l.gateName, id)
		if err != nil {
			return err
		}
		u, err := m.catalog.Matrix(l.upName, id)
		if err != nil {
			return err
		}
		d, err := m.catalog.Matrix(l.downName, id)
		if err != nil {
			return err
		}
		if err := g.MulQ8K(gate, xq); err != nil {
			return err
		}
		if err := u.MulQ8K(up, xq); err != nil {
			return err
		}
		if err := SwiGLU(mid, gate, up, 10); err != nil {
			return err
		}
		for i := range mid {
			mid[i] *= weights[slot]
		}
		if err := QuantizeQ8K(midq, mid); err != nil {
			return err
		}
		if err := d.MulQ8K(down, midq); err != nil {
			return err
		}
		for i, v := range down {
			dst[i] += v
		}
	}
	return nil
}

func sharedFFN(dst, x []float32, l *flashLayer) error {
	gate, up, mid := make([]float32, 2048), make([]float32, 2048), make([]float32, 2048)
	a := make([]Q80Activation, 128)
	if err := QuantizeQ80(a, x); err != nil {
		return err
	}
	if err := cpuDensePrequant(gate, a, l.sharedGate); err != nil {
		return err
	}
	if err := cpuDensePrequant(up, a, l.sharedUp); err != nil {
		return err
	}
	if err := SwiGLU(mid, gate, up, 10); err != nil {
		return err
	}
	return cpuDense(dst, mid, l.sharedDown)
}

func (m *flashModel) eval(s *flashState, token, pos int) (err error) {
	if m == nil || m.catalog == nil || s == nil || s.model != m || s.failed || s.closed || token < 0 || token >= 129280 || pos != s.position || pos >= s.context {
		return errors.New("invalid Flash evaluation")
	}
	defer func() {
		if err != nil {
			s.failed = true
			clear(s.logits)
		}
	}()
	embed := make([]float32, 4096)
	if err = m.embedding.Row(embed, token); err != nil {
		return err
	}
	cur, next := make([]float32, 16384), make([]float32, 16384)
	for h := 0; h < 4; h++ {
		copy(cur[h*4096:], embed)
	}
	for il := range m.layers {
		l, ls := &m.layers[il], &s.layers[il]
		if err = m.evalLayer(s, next, cur, l, ls, il, pos, token); err != nil {
			return fmt.Errorf("layer %d: %w", il, err)
		}
		cur, next = next, cur
	}
	flat, control, weights := make([]float32, 16384), make([]float32, 4), make([]float32, 4)
	if err = RMSNorm(flat, cur, nil, 1e-6); err != nil {
		return err
	}
	if err = cpuDense(control, flat, m.outputHC); err != nil {
		return err
	}
	for i, v := range control {
		z := float32(v*m.outputScale[0]) + m.outputBase[i]
		if z >= 0 {
			weights[i] = 1 / (1 + float32(math.Exp(float64(-z))))
		} else {
			e := float32(math.Exp(float64(z)))
			weights[i] = e / (1 + e)
		}
		weights[i] += 1e-6
	}
	if err = Collapse(embed, cur, weights); err != nil {
		return err
	}
	if err = RMSNorm(embed, embed, m.outputNorm, 1e-6); err != nil {
		return err
	}
	if err = cpuDense(s.logits, embed, m.output); err != nil {
		return err
	}
	for _, v := range s.logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return errors.New("nonfinite Flash logits")
		}
	}
	if s.support != nil {
		if err = s.support.publish(s.position, token); err != nil {
			return err
		}
	}
	s.position++
	return nil
}

func (m *flashModel) evalLayer(s *flashState, dst, cur []float32, l *flashLayer, ls *flashLayerState, il, pos, token int) error {
	plain, norm := make([]float32, 4096), make([]float32, 4096)
	post, comb := make([]float32, 4), make([]float32, 16)
	if err := l.attentionHC.pre(plain, post, comb, cur); err != nil {
		return err
	}
	if err := RMSNorm(norm, plain, l.attnNorm, 1e-6); err != nil {
		return err
	}
	qr, q, kv := make([]float32, 1024), make([]float32, 32768), make([]float32, 512)
	if err := cpuDense(qr, norm, l.qA); err != nil {
		return err
	}
	if err := RMSNorm(qr, qr, l.qNorm, 1e-6); err != nil {
		return err
	}
	if err := cpuDense(q, qr, l.qB); err != nil {
		return err
	}
	for h := 0; h < 64; h++ {
		row := q[h*512 : (h+1)*512]
		if err := RMSNorm(row, row, nil, 1e-6); err != nil {
			return err
		}
	}
	if err := cpuDense(kv, norm, l.kv); err != nil {
		return err
	}
	if err := RMSNorm(kv, kv, l.kvNorm, 1e-6); err != nil {
		return err
	}
	rope := flashRoPE(l.ratio)
	if err := rope.Rotate(q, 512, 64, pos, false); err != nil {
		return err
	}
	if err := rope.Rotate(kv, 512, 64, pos, false); err != nil {
		return err
	}
	if err := QuantizeKV(kv, 64); err != nil {
		return err
	}
	if err := ls.raw.Push(kv); err != nil {
		return err
	}
	if l.ratio != 0 {
		if err := pushCompression(&ls.compressed, ls.attention, &l.attentionCompression, norm, pos, l.ratio, 512); err != nil {
			return err
		}
	}
	var allowed []bool
	if l.ratio == 4 {
		if err := pushCompression(&ls.indexRows, ls.index, &l.indexCompression, norm, pos, 4, 128); err != nil {
			return err
		}
		var err error
		allowed, err = flashSelectRows(l, ls, norm, qr, pos)
		if err != nil {
			return err
		}
	}
	rows := append([]float32(nil), ls.raw.rows[:ls.raw.count*512]...)
	for row := 0; row < len(ls.compressed)/512; row++ {
		if allowed == nil || allowed[row] {
			rows = append(rows, ls.compressed[row*512:(row+1)*512]...)
		}
	}
	heads := make([]float32, 32768)
	if err := Attend(heads, q, rows, l.sinks, 512); err != nil {
		return err
	}
	if err := rope.Rotate(heads, 512, 64, pos, true); err != nil {
		return err
	}
	block, after := make([]float32, 4096), make([]float32, 16384)
	if err := groupedOutput(block, heads, l); err != nil {
		return err
	}
	if err := s.projectSteering(block, il, s.attentionScale); err != nil {
		return err
	}
	if err := Combine(after, block, cur, post, comb); err != nil {
		return err
	}
	if err := l.ffnHC.pre(plain, post, comb, after); err != nil {
		return err
	}
	if err := RMSNorm(norm, plain, l.ffnNorm, 1e-6); err != nil {
		return err
	}
	moe, shared := make([]float32, 4096), make([]float32, 4096)
	if err := m.routed(moe, norm, l, il, token); err != nil {
		return err
	}
	if err := sharedFFN(shared, norm, l); err != nil {
		return err
	}
	for i := range block {
		block[i] = moe[i] + shared[i]
	}
	if err := s.projectSteering(block, il, s.ffnScale); err != nil {
		return err
	}
	return s.combineLayer(dst, block, after, post, comb, il)
}

// setSteering copies the caller's directions; subsequent caller edits cannot
// change this request. Directions are not normalized by the pinned CPU path.
func (s *flashState) setSteering(directions []float32, attention, ffn float32) error {
	if s == nil || s.closed || s.position != 0 || len(directions) != 43*4096 || math.IsNaN(float64(attention)) || math.IsInf(float64(attention), 0) || math.IsNaN(float64(ffn)) || math.IsInf(float64(ffn), 0) {
		return errors.New("invalid CPU steering")
	}
	for _, v := range directions {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return errors.New("nonfinite CPU steering direction")
		}
	}
	s.steering = append([]float32(nil), directions...)
	s.attentionScale, s.ffnScale = attention, ffn
	return nil
}

func (s *flashState) projectSteering(x []float32, layer int, scale float32) error {
	if scale == 0 || len(s.steering) == 0 {
		return nil
	}
	if layer < 0 || layer >= 43 || len(x) != 4096 {
		return errors.New("invalid CPU steering row")
	}
	dir := s.steering[layer*4096 : (layer+1)*4096]
	var dot float32
	for i, v := range x {
		dot += float32(v * dir[i])
	}
	coefficient := float32(scale * dot)
	for i := range x {
		x[i] -= float32(coefficient * dir[i])
	}
	return nil
}
