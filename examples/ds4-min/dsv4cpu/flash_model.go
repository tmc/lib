package dsv4cpu

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"runtime"
)

// This binding is private until the complete Flash State and its literal
// forward controls are implemented. It does not admit a CLI/backend path.
type flashModel struct {
	digest                              [32]byte // Set by the owning loader after hashing the actual model.
	catalog                             *Catalog
	embedding, output, outputHC         *Matrix
	outputNorm, outputBase, outputScale []float32
	layers                              [43]flashLayer
	hash                                [3]Tensor
}

type flashHC struct {
	projection  *Matrix
	scale, base []float32
	nativeF16   bool
}

type flashCompression struct {
	kv, score, bias *Matrix
	norm            []float32
}

type flashLayer struct {
	attentionHC, ffnHC                                               flashHC
	qA, qB, kv, outA, outB, router, sharedGate, sharedUp, sharedDown *Matrix
	attnNorm, qNorm, kvNorm, sinks, ffnNorm, expertBias              []float32
	attentionCompression, indexCompression                           flashCompression
	indexQ, indexProjection                                          *Matrix
	ratio                                                            int
	gateName, upName, downName                                       string
}

func newFlashModel(source io.ReaderAt, size int64, tensors []Tensor) (*flashModel, error) {
	var ordinary []Tensor
	hashes := make(map[string]Tensor)
	for _, t := range tensors {
		if t.Encoding == 26 {
			if _, exists := hashes[t.Name]; exists {
				return nil, fmt.Errorf("duplicate integer tensor %q", t.Name)
			}
			hashes[t.Name] = t
		} else {
			ordinary = append(ordinary, t)
		}
	}
	c, err := NewCatalog(source, size, ordinary)
	if err != nil {
		return nil, err
	}
	m := &flashModel{catalog: c}
	matrix := func(name string, columns, rows int, encoding Encoding) (*Matrix, error) {
		t, ok := c.tensors[name]
		if !ok || len(t.Dimensions) != 2 || t.Dimensions[0] != columns || t.Dimensions[1] != rows || t.Encoding != encoding {
			return nil, fmt.Errorf("tensor %q does not match Flash layout", name)
		}
		return c.Matrix(name, 0)
	}
	vector := func(name string, n int) ([]float32, error) {
		t, ok := c.tensors[name]
		if !ok || len(t.Dimensions) != 1 || t.Dimensions[0] != n || t.Encoding != F32 {
			return nil, fmt.Errorf("tensor %q does not match Flash vector", name)
		}
		mat, err := c.Matrix(name, 0)
		if err != nil {
			return nil, err
		}
		out := make([]float32, n)
		if err := mat.Row(out, 0); err != nil {
			return nil, err
		}
		for _, v := range out {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return nil, fmt.Errorf("tensor %q contains nonfinite weights", name)
			}
		}
		return out, nil
	}
	if m.embedding, err = matrix("token_embd.weight", 4096, 129280, F16); err != nil {
		return nil, err
	}
	if m.output, err = matrix("output.weight", 4096, 129280, Q80); err != nil {
		return nil, err
	}
	if m.outputHC, err = matrix("output_hc_fn.weight", 16384, 4, F16); err != nil {
		return nil, err
	}
	if m.outputNorm, err = vector("output_norm.weight", 4096); err != nil {
		return nil, err
	}
	if m.outputBase, err = vector("output_hc_base.weight", 4); err != nil {
		return nil, err
	}
	if m.outputScale, err = vector("output_hc_scale.weight", 1); err != nil {
		return nil, err
	}
	for il := range m.layers {
		l := &m.layers[il]
		prefix := fmt.Sprintf("blk.%d.", il)
		bind := func(dst **Matrix, role string, columns, rows int, encoding Encoding) error {
			var err error
			*dst, err = matrix(prefix+role+".weight", columns, rows, encoding)
			return err
		}
		vec := func(dst *[]float32, role string, n int) error {
			var err error
			*dst, err = vector(prefix+role+".weight", n)
			return err
		}
		for _, hc := range []struct {
			name string
			dst  *flashHC
		}{{"hc_attn", &l.attentionHC}, {"hc_ffn", &l.ffnHC}} {
			hc.dst.nativeF16 = runtime.GOARCH == "arm64"
			if err := bind(&hc.dst.projection, hc.name+"_fn", 16384, 24, F16); err != nil {
				return nil, err
			}
			if err := vec(&hc.dst.scale, hc.name+"_scale", 3); err != nil {
				return nil, err
			}
			if err := vec(&hc.dst.base, hc.name+"_base", 24); err != nil {
				return nil, err
			}
		}
		for _, p := range []struct {
			dst        **Matrix
			name       string
			cols, rows int
			encoding   Encoding
		}{
			{&l.qA, "attn_q_a", 4096, 1024, Q80}, {&l.qB, "attn_q_b", 1024, 32768, Q80}, {&l.kv, "attn_kv", 4096, 512, Q80},
			{&l.outA, "attn_output_a", 4096, 8192, Q80}, {&l.outB, "attn_output_b", 8192, 4096, Q80},
			{&l.router, "ffn_gate_inp", 4096, 256, F16}, {&l.sharedGate, "ffn_gate_shexp", 4096, 2048, Q80}, {&l.sharedUp, "ffn_up_shexp", 4096, 2048, Q80}, {&l.sharedDown, "ffn_down_shexp", 2048, 4096, Q80},
		} {
			if err := bind(p.dst, p.name, p.cols, p.rows, p.encoding); err != nil {
				return nil, err
			}
		}
		for _, v := range []struct {
			dst  *[]float32
			name string
			n    int
		}{{&l.attnNorm, "attn_norm", 4096}, {&l.qNorm, "attn_q_a_norm", 1024}, {&l.kvNorm, "attn_kv_a_norm", 512}, {&l.sinks, "attn_sinks", 64}, {&l.ffnNorm, "ffn_norm", 4096}} {
			if err := vec(v.dst, v.name, v.n); err != nil {
				return nil, err
			}
		}
		biasName := prefix + "exp_probs_b.bias"
		if _, ok := c.tensors[biasName]; ok {
			if l.expertBias, err = vector(biasName, 256); err != nil {
				return nil, err
			}
		}
		l.gateName, l.upName, l.downName = prefix+"ffn_gate_exps.weight", prefix+"ffn_up_exps.weight", prefix+"ffn_down_exps.weight"
		for _, p := range []struct {
			name          string
			input, output int
			encoding      Encoding
		}{{l.gateName, 4096, 2048, IQ2XXS}, {l.upName, 4096, 2048, IQ2XXS}, {l.downName, 2048, 4096, Q2K}} {
			t, ok := c.tensors[p.name]
			if !ok || len(t.Dimensions) != 3 || t.Dimensions[0] != p.input || t.Dimensions[1] != p.output || t.Dimensions[2] != 256 || t.Encoding != p.encoding {
				return nil, fmt.Errorf("tensor %q does not match Flash expert layout", p.name)
			}
		}
		if il < 3 {
			name := prefix + "ffn_gate_tid2eid.weight"
			t, ok := hashes[name]
			if !ok || len(t.Dimensions) != 2 || t.Dimensions[0] != 6 || t.Dimensions[1] != 129280 || t.Bytes != 6*129280*4 || t.Offset < 0 || t.Offset > size-t.Bytes {
				return nil, fmt.Errorf("tensor %q does not match Flash hash layout", name)
			}
			m.hash[il] = t
			delete(hashes, name)
		}
		if il >= 2 {
			l.ratio = 128
			if il%2 == 0 {
				l.ratio = 4
			}
			lanes := 1
			if l.ratio == 4 {
				lanes = 2
			}
			compression := func(dst *flashCompression, name string, width int) error {
				if err := bind(&dst.kv, name+"_kv", 4096, lanes*width, F16); err != nil {
					return err
				}
				if err := bind(&dst.score, name+"_gate", 4096, lanes*width, F16); err != nil {
					return err
				}
				if err := bind(&dst.bias, name+"_ape", lanes*width, l.ratio, F16); err != nil {
					return err
				}
				return vec(&dst.norm, name+"_norm", width)
			}
			if err := compression(&l.attentionCompression, "attn_compressor", 512); err != nil {
				return nil, err
			}
			if l.ratio == 4 {
				if err := compression(&l.indexCompression, "indexer_compressor", 128); err != nil {
					return nil, err
				}
				t, ok := c.tensors[prefix+"indexer.attn_q_b.weight"]
				if !ok || (t.Encoding != F16 && t.Encoding != Q80) {
					return nil, fmt.Errorf("invalid indexer query encoding at layer %d", il)
				}
				if err := bind(&l.indexQ, "indexer.attn_q_b", 1024, 8192, t.Encoding); err != nil {
					return nil, err
				}
				if err := bind(&l.indexProjection, "indexer.proj", 4096, 64, F16); err != nil {
					return nil, err
				}
			}
		}
	}
	if len(hashes) != 0 {
		return nil, fmt.Errorf("unexpected integer tensors in Flash model")
	}
	return m, nil
}

func (m *flashModel) selectedHash(dst []int, layer, token int) error {
	if m == nil || layer < 0 || layer >= 3 || token < 0 || token >= 129280 || len(dst) != 6 {
		return fmt.Errorf("invalid Flash hash request")
	}
	t := m.hash[layer]
	var row [24]byte
	n, err := m.catalog.source.ReadAt(row[:], t.Offset+int64(token)*24)
	if err == nil && n != len(row) {
		err = io.ErrUnexpectedEOF
	}
	if err != nil {
		return fmt.Errorf("read hash layer %d token %d: %w", layer, token, err)
	}
	var selected [6]int
	for i := range selected {
		selected[i] = int(int32(binary.LittleEndian.Uint32(row[i*4:])))
		if selected[i] < 0 || selected[i] >= 256 {
			return fmt.Errorf("hash expert %d out of range", selected[i])
		}
	}
	copy(dst, selected[:])
	return nil
}
