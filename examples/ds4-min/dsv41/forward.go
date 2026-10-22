package dsv41

import (
	"fmt"

	"github.com/tmc/lib/examples/ds4-min/dsv4cpu"
)

// State owns one scalar V4.1 request's Engram, raw/compressed KV, pooling and
// selection history. It executes the CPU reference basis, not a CUDA or Metal
// backend. It is not safe for concurrent use. After an execution error Reset
// is required: partially published KV cannot be retried as a completed token.
type State struct {
	model             *Model
	engram            *EngramState
	kv                *KVState
	pool              [4]Pool
	candidates        Candidates
	position, context int
	valid             bool
	tokens            []int32
	logits            []float32
}

// NewState allocates request-local buffers over borrowed immutable model
// resources. No model-sized device or host copy is made.
func (m *Model) NewState(context int) (*State, error) {
	if m == nil || context <= 0 || context > 1048576 {
		return nil, fmt.Errorf("invalid V4.1 model or context")
	}
	engram, err := m.NewEngram()
	if err != nil {
		return nil, err
	}
	kv, err := NewKVState(context)
	if err != nil {
		return nil, err
	}
	return &State{model: m, engram: engram, kv: kv, context: context, valid: true}, nil
}

// Reset clears request history after a completed run or an execution error.
func (s *State) Reset() {
	if s == nil || s.model == nil {
		return
	}
	s.engram.Reset()
	s.kv.Reset()
	s.candidates.Reset()
	for i := range s.pool {
		s.pool[i].Reset()
	}
	s.position = 0
	s.valid = true
	s.tokens = s.tokens[:0]
	s.logits = nil
}

// Eval consumes one token and returns its next-token logits. It follows the
// scalar C graph's lagged HC pre-weight schedule, all forty layer/owner steps,
// Engram sites, candidate-block mask and released BF16/FP4/FP8 boundaries.
// The signed scalar CPU matvec basis is not proof of CUDA GEMM equivalence;
// full V2/V3 model-oracle qualification remains required before parity claims.
func (s *State) Eval(token int32) (logits []float32, err error) {
	if s == nil || s.model == nil || !s.valid || s.position >= s.context || token < 0 || token >= 129280 {
		return nil, fmt.Errorf("invalid V4.1 state or token")
	}
	defer func() {
		if err != nil {
			s.valid = false
		}
	}()
	if err = s.engram.Prepare(token, true); err != nil {
		return nil, err
	}
	embed, err := s.model.catalog.Matrix("token_embd.weight", 0)
	if err != nil {
		return nil, err
	}
	x := make([]float32, 5120)
	if err = embed.Row(x, int(token)); err != nil {
		return nil, err
	}
	residual := make([]float32, 4*5120)
	for h := 0; h < 4; h++ {
		copy(residual[h*5120:], x)
	}
	pre := []float32{1, 0, 0, 0}
	for layer := 0; layer < 40; layer++ {
		if err = s.engram.Apply(layer, residual); err != nil {
			return nil, fmt.Errorf("layer %d engram: %w", layer, err)
		}
		norm, attnSplit, e := s.model.hcInput(layer, false, residual, pre)
		if e != nil {
			return nil, fmt.Errorf("layer %d attention HC: %w", layer, e)
		}
		block, e := s.attention(layer, norm)
		if e != nil {
			return nil, fmt.Errorf("layer %d attention: %w", layer, e)
		}
		after := make([]float32, len(residual))
		if err = hcExpand(after, block, residual, attnSplit); err != nil {
			return nil, err
		}
		norm, ffnSplit, e := s.model.hcInput(layer, true, after, attnSplit[:4])
		if e != nil {
			return nil, e
		}
		block, e = s.model.moe(layer, norm)
		if e != nil {
			return nil, fmt.Errorf("layer %d MoE: %w", layer, e)
		}
		if err = hcExpand(residual, block, after, ffnSplit); err != nil {
			return nil, err
		}
		copy(pre, ffnSplit[:4])
	}
	if err = dsv4cpu.Collapse(x, residual, pre); err != nil {
		return nil, err
	}
	if err = Quantize(x, 5120, BFloat16); err != nil {
		return nil, err
	}
	if err = s.model.norm(x, x, "output_norm.weight"); err != nil {
		return nil, err
	}
	logits = make([]float32, 129280)
	if err = s.model.mul(logits, x, "output.weight", 0, false); err != nil {
		return nil, err
	}
	s.commitResult(token, logits)
	return logits, nil
}

func (s *State) commitResult(token int32, logits []float32) {
	s.position++
	s.tokens = append(s.tokens, token)
	s.logits = append(s.logits[:0], logits...)
}

func (m *Model) mul(dst, x []float32, name string, expert int, round bool) error {
	matrix, err := m.catalog.Matrix(name, expert)
	if err != nil {
		return err
	}
	switch m.encodings[name] {
	case 8:
		q := make([]dsv4cpu.Q80Activation, len(x)/32)
		if err = dsv4cpu.QuantizeQ80(q, x); err == nil {
			err = matrix.MulQ80(dst, q)
		}
	case 10, 16:
		q := make([]dsv4cpu.Q8K, len(x)/256)
		if err = dsv4cpu.QuantizeQ8K(q, x); err == nil {
			err = matrix.MulQ8K(dst, q)
		}
	default:
		err = matrix.Mul(dst, x)
	}
	if err != nil {
		return fmt.Errorf("multiply %s: %w", name, err)
	}
	if round {
		return Quantize(dst, len(dst), BFloat16)
	}
	return nil
}

func (m *Model) norm(dst, x []float32, name string) error {
	w, err := m.vectorRows(name, 1, len(x))
	if err != nil {
		return err
	}
	if err = dsv4cpu.RMSNorm(dst, x, w, 1e-20); err != nil {
		return err
	}
	return Quantize(dst, len(dst), BFloat16)
}

func (s *State) attention(layer int, norm []float32) ([]float32, error) {
	m := s.model
	p := fmt.Sprintf("blk.%d.", layer)
	l, _ := Schedule(layer)
	qr, q, raw := make([]float32, 1280), make([]float32, 64*512), make([]float32, 512)
	if err := m.mul(qr, norm, p+"attn_q_a.weight", 0, true); err != nil {
		return nil, err
	}
	if err := m.norm(qr, qr, p+"attn_q_a_norm.weight"); err != nil {
		return nil, err
	}
	if err := m.mul(q, qr, p+"attn_q_b.weight", 0, true); err != nil {
		return nil, err
	}
	if err := m.mul(raw, norm, p+"attn_kv.weight", 0, true); err != nil {
		return nil, err
	}
	if err := m.norm(raw, raw, p+"attn_kv_a_norm.weight"); err != nil {
		return nil, err
	}
	if err := rotary(q, 512, s.position, l.Ratio != 0, false); err != nil {
		return nil, err
	}
	if err := rotary(raw, 512, s.position, l.Ratio != 0, false); err != nil {
		return nil, err
	}
	if err := Quantize(raw, 512, FP8E8M0); err != nil {
		return nil, err
	}
	var compressed, index []float32
	if l.PublishKV {
		poolKV := make([]float32, 512)
		if err := m.mul(poolKV, norm, p+"attn_compressor_kv.weight", 0, l.Ratio == 1); err != nil {
			return nil, err
		}
		if l.Ratio == 2 {
			scores := make([]float32, 512)
			if err := m.mul(scores, norm, p+"attn_compressor_gate.weight", 0, false); err != nil {
				return nil, err
			}
			if (s.position+1)%2 == 0 {
				compressed = make([]float32, 512)
			}
			if err := s.pool[l.Owner].Append(compressed, poolKV, scores, 512, s.position); err != nil {
				return nil, err
			}
		} else {
			compressed = poolKV
		}
		if len(compressed) != 0 {
			if err := m.norm(compressed, compressed, p+"attn_compressor_norm.weight"); err != nil {
				return nil, err
			}
			index = make([]float32, 128)
			if err := m.mul(index, compressed, p+"indexer.attn_k.weight", 0, true); err != nil {
				return nil, err
			}
			if err := m.norm(index, index, p+"indexer.k_norm.weight"); err != nil {
				return nil, err
			}
			ropePos := s.position + 1 - l.Ratio
			if err := rotary(index, 128, ropePos, true, false); err != nil {
				return nil, err
			}
			if err := Quantize(index, 128, FP4E8M0); err != nil {
				return nil, err
			}
			if err := rotary(compressed, 512, ropePos, true, false); err != nil {
				return nil, err
			}
			if err := Quantize(compressed, 512, FP4E4M3); err != nil {
				return nil, err
			}
		}
	}
	if err := s.kv.Commit(layer, s.position, raw, compressed, index); err != nil {
		return nil, err
	}
	rows, cache, keys, err := s.kv.Rows(layer, s.position)
	if err != nil {
		return nil, err
	}
	var scores []float32
	if l.RefreshIndex && len(cache) > 0 {
		query, weights := make([]float32, 32*128), make([]float32, 32)
		if err := m.mul(query, qr, p+"indexer.attn_q_b.weight", 0, true); err != nil {
			return nil, err
		}
		if err := rotary(query, 128, s.position, true, false); err != nil {
			return nil, err
		}
		if err := Quantize(query, 128, FP4E8M0); err != nil {
			return nil, err
		}
		if err := m.mul(weights, norm, p+"indexer.proj.weight", 0, true); err != nil {
			return nil, err
		}
		scores, err = indexScores(query, weights, keys)
		if err != nil {
			return nil, err
		}
	}
	ids, err := s.candidates.Pick(layer, s.position, scores)
	if err != nil {
		return nil, err
	}
	for _, id := range ids {
		if id < 0 || id >= len(cache)/512 {
			return nil, fmt.Errorf("invalid compressed selection")
		}
		rows = append(rows, cache[id*512:(id+1)*512]...)
	}
	sinks, err := m.vectorRows(p+"attn_sinks.weight", 1, 64)
	if err != nil {
		return nil, err
	}
	heads := make([]float32, len(q))
	if err = dsv4cpu.Attend(heads, q, rows, sinks, 512); err != nil {
		return nil, err
	}
	if err = Quantize(heads, 512, BFloat16); err != nil {
		return nil, err
	}
	if err = rotary(heads, 512, s.position, l.Ratio != 0, true); err != nil {
		return nil, err
	}
	low := make([]float32, 8192)
	matrix, err := m.catalog.Matrix(p+"attn_output_a.weight", 0)
	if err != nil {
		return nil, err
	}
	row := make([]float32, 4096)
	for group := 0; group < 8; group++ {
		for out := 0; out < 1024; out++ {
			if err = matrix.Row(row, group*1024+out); err != nil {
				return nil, err
			}
			var sum float32
			for i, v := range row {
				sum += float32(v * heads[group*4096+i])
			}
			low[group*1024+out] = BF16(sum)
		}
	}
	block := make([]float32, 5120)
	if err = m.mul(block, low, p+"attn_output_b.weight", 0, false); err != nil {
		return nil, err
	}
	if err = Quantize(block, 5120, BFloat16); err != nil {
		return nil, err
	}
	return block, nil
}

func (m *Model) moe(layer int, x []float32) ([]float32, error) {
	p := fmt.Sprintf("blk.%d.", layer)
	logits, probs := make([]float32, 384), make([]float32, 384)
	if err := m.mul(logits, x, p+"ffn_gate_inp.weight", 0, false); err != nil {
		return nil, err
	}
	if err := dsv4cpu.RouterProbabilities(probs, logits); err != nil {
		return nil, err
	}
	bias, err := m.vectorRows(p+"exp_probs_b.bias", 1, 384)
	if err != nil {
		return nil, err
	}
	ids, weights := make([]int, 6), make([]float32, 6)
	if err = dsv4cpu.SelectExperts(ids, weights, probs, bias, 1.5); err != nil {
		return nil, err
	}
	block := make([]float32, 5120)
	gate, up, mid := make([]float32, 2304), make([]float32, 2304), make([]float32, 2304)
	down := make([]float32, 5120)
	for slot, id := range ids {
		if err = m.mul(gate, x, p+"ffn_gate_exps.weight", id, false); err != nil {
			return nil, err
		}
		if err = m.mul(up, x, p+"ffn_up_exps.weight", id, false); err != nil {
			return nil, err
		}
		if err = routedMid(mid, gate, up, weights[slot]); err != nil {
			return nil, err
		}
		if err = m.mul(down, mid, p+"ffn_down_exps.weight", id, false); err != nil {
			return nil, err
		}
		for i := range block {
			block[i] += down[i]
		}
	}
	if err = m.mul(gate, x, p+"ffn_gate_shexp.weight", 0, true); err != nil {
		return nil, err
	}
	if err = m.mul(up, x, p+"ffn_up_shexp.weight", 0, true); err != nil {
		return nil, err
	}
	if err = dsv4cpu.SwiGLU(mid, gate, up, 10); err != nil {
		return nil, err
	}
	if err = Quantize(mid, 2304, BFloat16); err != nil {
		return nil, err
	}
	if err = m.mul(down, mid, p+"ffn_down_shexp.weight", 0, true); err != nil {
		return nil, err
	}
	for i := range block {
		block[i] += down[i]
	}
	if err = Quantize(block, 5120, BFloat16); err != nil {
		return nil, err
	}
	return block, nil
}

// routedMid applies the selected expert's weight before the down activation
// quantization, as in the scalar C routed-MoE kernel. Moving it past MulQ8K
// changes the quantized operand and is not the same execution contract.
func routedMid(dst, gate, up []float32, weight float32) error {
	if err := dsv4cpu.SwiGLU(dst, gate, up, 10); err != nil {
		return err
	}
	for i := range dst {
		dst[i] *= weight
	}
	return nil
}
