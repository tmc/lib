package dsv4cpu

import (
	"fmt"
	"testing"
)

type virtualFlashSource struct{ readBytes int }

func (s *virtualFlashSource) ReadAt(p []byte, off int64) (int, error) {
	s.readBytes += len(p)
	clear(p)
	return len(p), nil
}

// All serialized matrices retain real Flash dimensions and byte extents.
// The ReaderAt supplies zero values without allocating synthetic model pages.
func virtualFlashDescriptors() ([]Tensor, int64) {
	var tensors []Tensor
	var offset int64
	add := func(name string, e Encoding, dims ...int) {
		columns := dims[0]
		rowBytes := int64(0)
		switch e {
		case F32, 26:
			rowBytes = int64(columns) * 4
		case F16:
			rowBytes = int64(columns) * 2
		case Q80:
			rowBytes = int64(columns/32) * 34
		case IQ2XXS:
			rowBytes = int64(columns/256) * 66
		case Q2K:
			rowBytes = int64(columns/256) * 84
		}
		bytes := rowBytes
		for _, d := range dims[1:] {
			bytes *= int64(d)
		}
		tensors = append(tensors, Tensor{name, dims, e, offset, bytes})
		offset += bytes
	}
	add("token_embd.weight", F16, 4096, 129280)
	add("output.weight", Q80, 4096, 129280)
	add("output_hc_fn.weight", F16, 16384, 4)
	add("output_hc_scale.weight", F32, 1)
	add("output_hc_base.weight", F32, 4)
	add("output_norm.weight", F32, 4096)
	for il := 0; il < 43; il++ {
		p := fmt.Sprintf("blk.%d.", il)
		for _, hc := range []string{"hc_attn", "hc_ffn"} {
			add(p+hc+"_fn.weight", F16, 16384, 24)
			add(p+hc+"_scale.weight", F32, 3)
			add(p+hc+"_base.weight", F32, 24)
		}
		for _, v := range []struct {
			name string
			n    int
		}{{"attn_norm", 4096}, {"attn_q_a_norm", 1024}, {"attn_kv_a_norm", 512}, {"attn_sinks", 64}, {"ffn_norm", 4096}} {
			add(p+v.name+".weight", F32, v.n)
		}
		for _, m := range []struct {
			name       string
			cols, rows int
			e          Encoding
		}{{"attn_q_a", 4096, 1024, Q80}, {"attn_q_b", 1024, 32768, Q80}, {"attn_kv", 4096, 512, Q80}, {"attn_output_a", 4096, 8192, Q80}, {"attn_output_b", 8192, 4096, Q80}, {"ffn_gate_inp", 4096, 256, F16}, {"ffn_gate_shexp", 4096, 2048, Q80}, {"ffn_up_shexp", 4096, 2048, Q80}, {"ffn_down_shexp", 2048, 4096, Q80}} {
			add(p+m.name+".weight", m.e, m.cols, m.rows)
		}
		add(p+"ffn_gate_exps.weight", IQ2XXS, 4096, 2048, 256)
		add(p+"ffn_up_exps.weight", IQ2XXS, 4096, 2048, 256)
		add(p+"ffn_down_exps.weight", Q2K, 2048, 4096, 256)
		if il < 3 {
			add(p+"ffn_gate_tid2eid.weight", 26, 6, 129280)
		}
		if il >= 2 {
			ratio, lanes := 128, 1
			if il%2 == 0 {
				ratio, lanes = 4, 2
			}
			for _, c := range []struct {
				name  string
				width int
			}{{"attn_compressor", 512}, {"indexer_compressor", 128}} {
				if c.width == 128 && ratio != 4 {
					continue
				}
				add(p+c.name+"_kv.weight", F16, 4096, lanes*c.width)
				add(p+c.name+"_gate.weight", F16, 4096, lanes*c.width)
				add(p+c.name+"_ape.weight", F16, lanes*c.width, ratio)
				add(p+c.name+"_norm.weight", F32, c.width)
			}
			if ratio == 4 {
				add(p+"indexer.attn_q_b.weight", F16, 1024, 8192)
				add(p+"indexer.proj.weight", F16, 4096, 64)
			}
		}
	}
	return tensors, offset
}

func TestFlashImmutableBinding(t *testing.T) {
	tensors, size := virtualFlashDescriptors()
	source := &virtualFlashSource{}
	m, err := newFlashModel(source, size, tensors)
	if err != nil {
		t.Fatal(err)
	}
	if source.readBytes > 2<<20 {
		t.Fatalf("binding expanded model weights: %d bytes", source.readBytes)
	}
	if m.layers[0].ratio != 0 || m.layers[2].ratio != 4 || m.layers[41].ratio != 128 {
		t.Fatal("wrong Flash schedule")
	}
	var ids [6]int
	if err := m.selectedHash(ids[:], 0, 129279); err != nil {
		t.Fatal(err)
	}
	t.Logf("bound %d real-dimensional descriptors, virtual serialized bytes %d; only %d small-vector/hash bytes read", len(tensors), size, source.readBytes)
	tensors[0].Dimensions[0] = 1
	if m.embedding.cols != 4096 {
		t.Fatal("caller metadata changed binding")
	}
}

func TestFlashBindingRejectsMissingRole(t *testing.T) {
	tensors, size := virtualFlashDescriptors()
	for _, name := range []string{"output.weight", "blk.0.hc_attn_fn.weight", "blk.42.ffn_down_exps.weight", "blk.2.indexer.proj.weight", "blk.0.ffn_gate_tid2eid.weight"} {
		var copy []Tensor
		for _, v := range tensors {
			if v.Name != name {
				copy = append(copy, v)
			}
		}
		if _, err := newFlashModel(&virtualFlashSource{}, size, copy); err == nil {
			t.Fatal("missing role accepted", name)
		}
	}
}

func TestFlashRequestOwnership(t *testing.T) {
	tensors, size := virtualFlashDescriptors()
	m, err := newFlashModel(&virtualFlashSource{}, size, tensors)
	if err != nil {
		t.Fatal(err)
	}
	a, err := m.newState(256)
	if err != nil {
		t.Fatal(err)
	}
	b, err := m.newState(256)
	if err != nil {
		t.Fatal(err)
	}
	row := make([]float32, 512)
	row[0] = 1
	for il := range a.layers {
		if err := a.layers[il].raw.Push(row); err != nil {
			t.Fatal(err)
		}
		if b.layers[il].raw.count != 0 {
			t.Fatal("raw state shared", il)
		}
		if a.layers[il].attention != nil {
			a.layers[il].attention.kv[0] = 3
			if b.layers[il].attention.kv[0] != 0 {
				t.Fatal("compressor state shared", il)
			}
		}
	}
	if err := a.reset(); err != nil {
		t.Fatal(err)
	}
	for il := range a.layers {
		if a.layers[il].raw.count != 0 {
			t.Fatal("reset retained raw state", il)
		}
	}
	other := &flashModel{catalog: m.catalog}
	if err := other.eval(a, 0, 0); err == nil {
		t.Fatal("foreign model state accepted")
	}
	if err := m.eval(a, 0, 1); err == nil {
		t.Fatal("skipped token position accepted")
	}
	if a.failed {
		t.Fatal("admission rejection damaged state")
	}
}
