package dsv4shape

import (
	"fmt"
	"slices"

	"github.com/tmc/lib/gguf"
)

// TensorSpec describes one required weight, in GGUF dimension order.
// Optional weights are checked when present. Types describe upstream layouts,
// not the narrower kernel registry of a particular backend.
type TensorSpec struct {
	Name       string
	Dimensions []uint64
	Types      []uint32
	Optional   bool
}

// Weights returns the full single-stage weight contract for c. Split pipeline
// files require a separate layer-range contract and are not admitted here.
func (c Config) Weights() ([]TensorSpec, error) {
	if err := c.Validate(); err != nil {
		return nil, err
	}
	e, h, v := uint64(c.Embed), uint64(c.HC), uint64(c.Vocab)
	mix := h * (h + 2)
	q := uint64(c.Heads * c.HeadDim)
	low := uint64(c.OutGroups * c.OutLoRA)
	var out []TensorSpec
	add := func(name string, types []uint32, dims ...uint64) {
		out = append(out, TensorSpec{Name: name, Dimensions: dims, Types: append([]uint32(nil), types...)})
	}
	f32, f16, i32 := []uint32{0}, []uint32{1}, []uint32{26}
	dense, routed := []uint32{8, 12, 2}, []uint32{8, 16, 10, 12, 13, 14, 39}
	add("token_embd.weight", f16, e, v)
	add("output_hc_base.weight", f32, h)
	add("output_hc_fn.weight", f16, e*h, h)
	add("output_hc_scale.weight", f32, 1)
	add("output_norm.weight", f32, e)
	add("output.weight", dense, e, v)
	for i := 0; i < c.Layers; i++ {
		l, _ := c.Layer(i)
		prefix := fmt.Sprintf("blk.%d.", i)
		layer := func(n string, types []uint32, dims ...uint64) { add(prefix+n, types, dims...) }
		for _, phase := range []string{"attn", "ffn"} {
			layer("hc_"+phase+"_fn.weight", f16, e*h, mix)
			layer("hc_"+phase+"_base.weight", f32, mix)
			layer("hc_"+phase+"_scale.weight", f32, 3)
		}
		layer("attn_norm.weight", f32, e)
		layer("attn_q_a.weight", dense, e, uint64(c.QLoRA))
		layer("attn_q_a_norm.weight", f32, uint64(c.QLoRA))
		layer("attn_q_b.weight", dense, uint64(c.QLoRA), q)
		layer("attn_kv.weight", dense, e, uint64(c.HeadDim))
		layer("attn_kv_a_norm.weight", f32, uint64(c.HeadDim))
		layer("attn_sinks.weight", f32, uint64(c.Heads))
		layer("attn_output_a.weight", dense, uint64(c.HeadDim*(c.Heads/c.OutGroups)), low)
		layer("attn_output_b.weight", dense, low, e)
		if l.CompressRatio != 0 {
			width := uint64(c.HeadDim)
			if l.CompressRatio == 4 {
				width *= 2
			}
			layer("attn_compressor_ape.weight", f16, width, uint64(l.CompressRatio))
			layer("attn_compressor_kv.weight", f16, e, width)
			layer("attn_compressor_gate.weight", f16, e, width)
			layer("attn_compressor_norm.weight", f32, uint64(c.HeadDim))
		}
		if l.Indexer {
			width := uint64(2 * c.IndexerHeadDim)
			layer("indexer.attn_q_b.weight", []uint32{1, 8}, uint64(c.QLoRA), uint64(c.IndexerHeads*c.IndexerHeadDim))
			layer("indexer.proj.weight", f16, e, uint64(c.IndexerHeads))
			layer("indexer_compressor_ape.weight", f16, width, uint64(l.CompressRatio))
			layer("indexer_compressor_kv.weight", f16, e, width)
			layer("indexer_compressor_gate.weight", f16, e, width)
			layer("indexer_compressor_norm.weight", f32, uint64(c.IndexerHeadDim))
		}
		layer("ffn_norm.weight", f32, e)
		layer("ffn_gate_inp.weight", f16, e, uint64(c.Experts))
		layer("exp_probs_b.bias", f32, uint64(c.Experts))
		out[len(out)-1].Optional = true
		layer("ffn_gate_exps.weight", routed, e, uint64(c.ExpertFF), uint64(c.Experts))
		layer("ffn_up_exps.weight", routed, e, uint64(c.ExpertFF), uint64(c.Experts))
		layer("ffn_down_exps.weight", routed, uint64(c.ExpertFF), e, uint64(c.Experts))
		layer("ffn_gate_shexp.weight", dense, e, uint64(c.ExpertFF))
		layer("ffn_up_shexp.weight", dense, e, uint64(c.ExpertFF))
		layer("ffn_down_shexp.weight", dense, uint64(c.ExpertFF), e)
		if l.HashRouter {
			layer("ffn_gate_tid2eid.weight", i32, uint64(c.ExpertsUsed), v)
		}
	}
	return out, nil
}

// WeightView is a checked file range. RowBytes is the quantized row stride;
// ExpertBytes is zero except for routed tensors, where it is the expert stride.
type WeightView struct {
	Name                                 string
	Type                                 uint32
	Offset, Bytes, RowBytes, ExpertBytes uint64
}

// Binding is the validated full-model weight plan. It owns no host/device memory.
type Binding struct {
	Weights                    []WeightView
	ResidentBytes, RoutedBytes uint64
}

func addBytes(a, b uint64) (uint64, error) {
	if b > ^uint64(0)-a {
		return 0, fmt.Errorf("shape: byte count overflow")
	}
	return a + b, nil
}

// Bind checks required tensors, quantized row/expert strides and file extents.
// Tensor bytes are never dereferenced. A binding is not a kernel-support receipt.
func (c Config) Bind(g *gguf.File) (*Binding, error) {
	specs, err := c.Weights()
	if err != nil {
		return nil, err
	}
	if g == nil || g.DataOffset > g.Size {
		return nil, fmt.Errorf("shape: invalid GGUF file extent")
	}
	byName := make(map[string]gguf.Tensor)
	for _, t := range g.Tensors {
		if _, ok := byName[t.Name]; ok {
			return nil, fmt.Errorf("shape: duplicate tensor %q", t.Name)
		}
		byName[t.Name] = t
	}
	b := new(Binding)
	for _, s := range specs {
		t, ok := byName[s.Name]
		if !ok {
			if s.Optional {
				continue
			}
			return nil, fmt.Errorf("shape: missing tensor %q", s.Name)
		}
		if !slices.Equal(t.Dimensions, s.Dimensions) || !slices.Contains(s.Types, t.TypeID) {
			return nil, fmt.Errorf("shape: tensor %q dimensions/type %v/%d", s.Name, t.Dimensions, t.TypeID)
		}
		n, err := t.Bytes()
		if err != nil {
			return nil, err
		}
		off, err := addBytes(g.DataOffset, t.Offset)
		if err != nil {
			return nil, err
		}
		if off > g.Size || n > g.Size-off {
			return nil, fmt.Errorf("shape: tensor %q exceeds file", s.Name)
		}
		row := t
		row.Dimensions = t.Dimensions[:1]
		stride, err := row.Bytes()
		if err != nil {
			return nil, err
		}
		view := WeightView{Name: t.Name, Type: t.TypeID, Offset: off, Bytes: n, RowBytes: stride}
		if len(t.Dimensions) == 3 {
			view.ExpertBytes = n / t.Dimensions[2]
			b.RoutedBytes, err = addBytes(b.RoutedBytes, n)
		} else {
			b.ResidentBytes, err = addBytes(b.ResidentBytes, n)
		}
		if err != nil {
			return nil, err
		}
		b.Weights = append(b.Weights, view)
	}
	for i := 0; i < c.Layers; i++ {
		prefix := fmt.Sprintf("blk.%d.", i)
		if byName[prefix+"ffn_gate_exps.weight"].TypeID != byName[prefix+"ffn_up_exps.weight"].TypeID {
			return nil, fmt.Errorf("shape: layer %d gate/up types differ", i)
		}
	}
	return b, nil
}

// AdmitResident checks the full single-copy weight budget plus an independently
// qualified backend State and reserve budget. It does not model SSD streaming,
// mapped-page residency, temporary repacking or allocator fragmentation.
func (b *Binding) AdmitResident(available, state, reserve uint64) error {
	if b == nil {
		return fmt.Errorf("shape: missing binding")
	}
	n, err := addBytes(b.ResidentBytes, b.RoutedBytes)
	if err != nil {
		return err
	}
	n, err = addBytes(n, state)
	if err != nil {
		return err
	}
	n, err = addBytes(n, reserve)
	if err != nil {
		return err
	}
	if n > available {
		return fmt.Errorf("shape: resident bytes %d exceed available %d", n, available)
	}
	return nil
}
