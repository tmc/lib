package main

import (
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/tmc/lib/gguf"
)

// dsparkModel owns a copied tensor directory, not device state or weight
// buffers. Admission follows ds4.c:5615-5875 and 7774-7868 at 11e8cad.
// Successful admission does not imply a support forward is implemented.
type dsparkModel struct {
	reader                     io.ReaderAt
	dataOffset                 uint64
	sha256                     string
	block, rank, noise, stages int
	targets                    []int
	tensors                    map[string]gguf.Tensor
}

// tensor returns a detached descriptor so a resource builder cannot change
// the admitted directory while binding device weights.
func (m *dsparkModel) tensor(name string) (gguf.Tensor, bool) {
	if m == nil {
		return gguf.Tensor{}, false
	}
	t, ok := m.tensors[name]
	t.Dimensions = append([]uint64(nil), t.Dimensions...)
	return t, ok
}

func readDSparkModel(r io.ReaderAt, size int64, sha string) (*dsparkModel, error) {
	if len(sha) != 64 {
		return nil, fmt.Errorf("dspark: full support sha256 required")
	}
	if err := gguf.CheckSHA256(r, size, sha); err != nil {
		return nil, fmt.Errorf("dspark support: %w", err)
	}
	g, err := gguf.Read(r, size)
	if err != nil {
		return nil, fmt.Errorf("dspark support: %w", err)
	}
	m, err := admitDSparkModel(g)
	if err != nil {
		return nil, err
	}
	m.sha256 = sha
	m.reader = r
	return m, nil
}

func admitDSparkModel(g *gguf.File) (*dsparkModel, error) {
	if g == nil {
		return nil, fmt.Errorf("dspark: missing container")
	}
	m := &dsparkModel{dataOffset: g.DataOffset, tensors: make(map[string]gguf.Tensor)}
	integer := func(key string) (int, error) {
		for _, prefix := range []string{"deepseek4.dspark.", "deepseek4.dspark_", "dspark."} {
			if value, ok := g.Metadata[prefix+key]; ok {
				n, ok := value.Int()
				if !ok || n < 0 || n > 4294967295 {
					return 0, fmt.Errorf("dspark: invalid %s", key)
				}
				return int(n), nil
			}
		}
		return 0, fmt.Errorf("dspark: missing %s", key)
	}
	var err error
	if m.block, err = integer("block_size"); err != nil {
		return nil, err
	}
	if m.rank, err = integer("markov_rank"); err != nil {
		return nil, err
	}
	if m.noise, err = integer("noise_token_id"); err != nil {
		return nil, err
	}
	if m.block < 1 || m.block > 16 || m.rank < 1 || m.noise >= Flash.Vocab {
		return nil, fmt.Errorf("dspark: metadata outside Flash limits")
	}
	for _, prefix := range []string{"deepseek4.dspark.", "deepseek4.dspark_", "dspark."} {
		value, ok := g.Metadata[prefix+"target_layer_ids"]
		if !ok {
			continue
		}
		a, ok := value.Value.([]any)
		if !ok || len(a) < 1 || len(a) > 8 {
			return nil, fmt.Errorf("dspark: invalid target layer list")
		}
		for _, v := range a {
			n, ok := (gguf.Value{Value: v}).Int()
			if !ok || n < 0 || n >= int64(Flash.Layers) || len(m.targets) > 0 && int(n) <= m.targets[len(m.targets)-1] {
				return nil, fmt.Errorf("dspark: invalid target layer order")
			}
			m.targets = append(m.targets, int(n))
		}
		break
	}
	if len(m.targets) == 0 {
		return nil, fmt.Errorf("dspark: missing target layer list")
	}
	for _, t := range g.Tensors {
		if _, ok := m.tensors[t.Name]; ok {
			return nil, fmt.Errorf("dspark: duplicate tensor %q", t.Name)
		}
		parts := strings.Split(t.Name, ".")
		if len(parts) < 3 || parts[0] != "mtp" {
			continue
		}
		stage, err := strconv.Atoi(parts[1])
		if err != nil || stage < 0 || stage >= 8 {
			return nil, fmt.Errorf("dspark: invalid stage %q", t.Name)
		}
		if stage+1 > m.stages {
			m.stages = stage + 1
		}
		t.Dimensions = append([]uint64(nil), t.Dimensions...)
		m.tensors[t.Name] = t
	}
	if m.stages < 3 {
		return nil, fmt.Errorf("dspark: require three to eight support stages")
	}
	for stage := 0; stage < m.stages; stage++ {
		for _, rule := range dsparkRules(stage, m) {
			name := fmt.Sprintf("mtp.%d.%s", stage, rule.name)
			t, ok := m.tensors[name]
			if !ok {
				return nil, fmt.Errorf("dspark: missing %s", name)
			}
			if !rule.accepts(t.TypeID) || len(t.Dimensions) != len(rule.dims) {
				return nil, fmt.Errorf("dspark: invalid layout for %s", name)
			}
			for i, n := range rule.dims {
				if t.Dimensions[i] != n {
					return nil, fmt.Errorf("dspark: invalid dimension %d for %s", i, name)
				}
			}
			bytes, err := t.Bytes()
			if err != nil {
				return nil, err
			}
			if g.DataOffset > g.Size || t.Offset > g.Size-g.DataOffset || bytes > g.Size-g.DataOffset-t.Offset {
				return nil, fmt.Errorf("dspark: tensor %s outside file", name)
			}
		}
		gate := m.tensors[fmt.Sprintf("mtp.%d.ffn_gate_exps.weight", stage)]
		up := m.tensors[fmt.Sprintf("mtp.%d.ffn_up_exps.weight", stage)]
		if gate.TypeID != up.TypeID {
			return nil, fmt.Errorf("dspark: gate/up quant types differ at stage %d", stage)
		}
	}
	return m, nil
}

type dsparkRule struct {
	name string
	kind int
	dims []uint64
}

func (r dsparkRule) accepts(t uint32) bool {
	switch r.kind {
	case 0:
		return t == 0
	case 1:
		return t == 0 || t == 1
	case 2:
		return t == 0 || t == 1 || t == 8
	case 3:
		return t == 8 || t == 16 || t == 10 || t == 12 || t == 13 || t == 14 || t == 39
	}
	return false
}
func dsparkRules(stage int, m *dsparkModel) []dsparkRule {
	e, h, q, o, x, n := uint64(Flash.Embed), uint64(Flash.HC), uint64(Flash.QLoRA), uint64(Flash.OutGroups*Flash.OutLoRA), uint64(Flash.ExpertFF), uint64(Flash.Experts)
	d, heads := uint64(Flash.HeadDim), uint64(Flash.Heads)
	rules := []dsparkRule{
		{"hc_attn_fn.weight", 1, []uint64{e * h, 2*h + h*h}}, {"hc_attn_scale.weight", 0, []uint64{3}}, {"hc_attn_base.weight", 0, []uint64{2*h + h*h}},
		{"attn_norm.weight", 0, []uint64{e}}, {"attn_q_a.weight", 2, []uint64{e, q}}, {"attn_q_a_norm.weight", 0, []uint64{q}}, {"attn_q_b.weight", 2, []uint64{q, heads * d}},
		{"attn_kv.weight", 2, []uint64{e, d}}, {"attn_kv_a_norm.weight", 0, []uint64{d}}, {"attn_sinks.weight", 0, []uint64{heads}},
		{"attn_output_a.weight", 2, []uint64{d * heads / uint64(Flash.OutGroups), o}}, {"attn_output_b.weight", 2, []uint64{o, e}},
		{"hc_ffn_fn.weight", 1, []uint64{e * h, 2*h + h*h}}, {"hc_ffn_scale.weight", 0, []uint64{3}}, {"hc_ffn_base.weight", 0, []uint64{2*h + h*h}},
		{"ffn_norm.weight", 0, []uint64{e}}, {"ffn_gate_inp.weight", 2, []uint64{e, n}}, {"exp_probs_b.bias", 0, []uint64{n}},
		{"ffn_gate_exps.weight", 3, []uint64{e, x, n}}, {"ffn_up_exps.weight", 3, []uint64{e, x, n}}, {"ffn_down_exps.weight", 3, []uint64{x, e, n}},
		{"ffn_gate_shexp.weight", 2, []uint64{e, x}}, {"ffn_up_shexp.weight", 2, []uint64{e, x}}, {"ffn_down_shexp.weight", 2, []uint64{x, e}},
	}
	if stage == 0 {
		rules = append(rules, dsparkRule{"main_proj.weight", 2, []uint64{uint64(len(m.targets)) * e, e}}, dsparkRule{"main_norm.weight", 0, []uint64{e}})
	}
	if stage == m.stages-1 {
		rank, vocab := uint64(m.rank), uint64(Flash.Vocab)
		rules = append(rules, dsparkRule{"norm.weight", 0, []uint64{e}}, dsparkRule{"hc_head_base.weight", 0, []uint64{h}}, dsparkRule{"hc_head_fn.weight", 1, []uint64{e * h, h}}, dsparkRule{"hc_head_scale.weight", 0, []uint64{1}}, dsparkRule{"markov_head.markov_w1.weight", 2, []uint64{rank, vocab}}, dsparkRule{"markov_head.markov_w2.weight", 2, []uint64{rank, vocab}}, dsparkRule{"confidence_head.proj.weight", 2, []uint64{e + rank, 1}})
	}
	return rules
}
