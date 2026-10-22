package dsv41

import (
	"fmt"
	"io"
	"math"

	"github.com/tmc/lib/examples/ds4-min/dsv41engram"
	"github.com/tmc/lib/examples/ds4-min/dsv4cpu"
	"github.com/tmc/lib/gguf"
)

// Model binds the V4.1 Flash tensor directory and Engram SSD tables to one
// immutable file source. It never owns or closes the ReaderAt. All states must
// finish before the source is changed or closed. Bind reads no tensor payloads.
type Model struct {
	catalog     *dsv4cpu.Catalog
	layout      *dsv41engram.Layout
	tables      [2]*dsv41engram.Table
	encodings   map[string]uint32
	assetSHA    [32]byte
	tensors     map[string]TensorDescriptor
	fileSize    uint64
	boundSource io.ReaderAt
	boundSize   int64
}

// Bind validates the V4.1 metadata and every required scalar text tensor. All
// matrix payloads remain serialized ReaderAt views; the two Engram tables are
// bounded separate views, not expanded or included in the matrix catalog.
// Unsupported CPU encodings return an error rather than silently falling back.
func Bind(source io.ReaderAt, size int64, g *gguf.File) (*Model, error) {
	if source == nil || g == nil || size < 0 || uint64(size) != g.Size {
		return nil, fmt.Errorf("invalid V4.1 file source")
	}
	layout, err := ValidateMetadata(g)
	if err != nil {
		return nil, err
	}
	byName := make(map[string]gguf.Tensor, len(g.Tensors))
	for _, t := range g.Tensors {
		if _, ok := byName[t.Name]; ok {
			return nil, fmt.Errorf("duplicate tensor %q", t.Name)
		}
		byName[t.Name] = t
	}
	var tensors []dsv4cpu.Tensor
	var tables [2]*dsv41engram.Table
	descriptors := make(map[string]TensorDescriptor)
	needs := tensorRequirements()
	// Optional image-routing bias uses the same validated descriptor path.
	for layer := 0; layer < 40; layer++ {
		name := fmt.Sprintf("blk.%d.exp_probs_b_vl.bias", layer)
		if _, ok := byName[name]; ok {
			needs = append(needs, requirement{name: name, dims: []int{384}, types: 1 << 0, table: -1})
		}
	}
	for _, need := range needs {
		t, ok := byName[need.name]
		if !ok {
			return nil, fmt.Errorf("missing tensor %q", need.name)
		}
		if len(t.Dimensions) != len(need.dims) {
			return nil, fmt.Errorf("tensor %q dimension count", need.name)
		}
		for i, d := range need.dims {
			if t.Dimensions[i] != uint64(d) {
				return nil, fmt.Errorf("tensor %q dimension %d", need.name, i)
			}
		}
		if t.TypeID >= 64 || need.types&(uint64(1)<<t.TypeID) == 0 {
			return nil, fmt.Errorf("tensor %q encoding %d", need.name, t.TypeID)
		}
		bytes, err := t.Bytes()
		if err != nil {
			return nil, fmt.Errorf("tensor %q: %w", need.name, err)
		}
		if t.Offset > math.MaxUint64-g.DataOffset || g.DataOffset+t.Offset > math.MaxInt64 || bytes > math.MaxInt64 {
			return nil, fmt.Errorf("tensor %q extent overflow", need.name)
		}
		offset := int64(g.DataOffset + t.Offset)
		if offset > size-int64(bytes) {
			return nil, fmt.Errorf("tensor %q outside file", need.name)
		}
		descriptors[t.Name] = TensorDescriptor{
			Name: t.Name, Encoding: t.TypeID,
			Dimensions: append([]uint64(nil), t.Dimensions...),
			Offset:     offset, Bytes: int64(bytes),
		}
		if need.table >= 0 {
			table, err := dsv41engram.NewTable(source, size, offset, uint32(need.dims[1]))
			if err != nil {
				return nil, err
			}
			tables[need.table] = table
			continue
		}
		tensors = append(tensors, dsv4cpu.Tensor{Name: t.Name, Dimensions: append([]int(nil), need.dims...), Encoding: dsv4cpu.Encoding(t.TypeID), Offset: offset, Bytes: int64(bytes)})
	}
	for layer := 0; layer < 40; layer++ {
		gate, up := byName[fmt.Sprintf("blk.%d.ffn_gate_exps.weight", layer)], byName[fmt.Sprintf("blk.%d.ffn_up_exps.weight", layer)]
		if gate.TypeID != up.TypeID {
			return nil, fmt.Errorf("layer %d expert gate/up encoding mismatch", layer)
		}
	}
	catalog, err := dsv4cpu.NewCatalog(source, size, tensors)
	if err != nil {
		return nil, err
	}
	encodings := make(map[string]uint32, len(tensors))
	for _, t := range tensors {
		encodings[t.Name] = uint32(t.Encoding)
	}
	return &Model{catalog: catalog, layout: layout, tables: tables, encodings: encodings, tensors: descriptors, fileSize: uint64(size), boundSource: source, boundSize: size}, nil
}

// NewEngram creates request-local execution state from this bound model. Only
// the two small gate-normalization tensors are copied; the projection matrices
// and SSD tables remain borrowed serialized views.
func (m *Model) NewEngram() (*EngramState, error) {
	if m == nil || m.catalog == nil {
		return nil, fmt.Errorf("invalid V4.1 model")
	}
	var weights [2]EngramWeights
	for i, layer := range []int{1, 14} {
		prefix := fmt.Sprintf("blk.%d.engram_", layer)
		projection, err := m.catalog.Matrix(prefix+"kv.weight", 0)
		if err != nil {
			return nil, err
		}
		query, err := m.vectorRows(prefix+"q_norm.weight", 4, 5120)
		if err != nil {
			return nil, err
		}
		key, err := m.vectorRows(prefix+"k_norm.weight", 4, 5120)
		if err != nil {
			return nil, err
		}
		weights[i] = EngramWeights{Table: m.tables[i], Projection: projection, Query: query, Key: key}
	}
	return NewEngramState(m.layout, weights)
}

func (m *Model) vectorRows(name string, rows, width int) ([]float32, error) {
	matrix, err := m.catalog.Matrix(name, 0)
	if err != nil {
		return nil, err
	}
	out := make([]float32, rows*width)
	for i := 0; i < rows; i++ {
		if err := matrix.Row(out[i*width:(i+1)*width], i); err != nil {
			return nil, fmt.Errorf("read %s: %w", name, err)
		}
	}
	return out, nil
}

type requirement struct {
	name  string
	dims  []int
	types uint64
	table int
}

func tensorRequirements() []requirement {
	var all []requirement
	add := func(name string, types uint64, dims ...int) { all = append(all, requirement{name, dims, types, -1}) }
	const dense = uint64(1<<8 | 1<<12 | 1<<2) // Actual upstream dense set; Catalog qualifies supported types.
	const routed = uint64(1<<8 | 1<<16 | 1<<10 | 1<<12 | 1<<13 | 1<<14 | 1<<39)
	add("token_embd.weight", uint64(1<<1), 5120, 129280)
	add("output_norm.weight", uint64(1<<0), 5120)
	add("output.weight", dense, 5120, 129280)
	for layer := 0; layer < 40; layer++ {
		prefix := fmt.Sprintf("blk.%d.", layer)
		l, _ := Schedule(layer)
		for _, sub := range []string{"hc_attn", "hc_ffn"} {
			add(prefix+sub+"_fn.weight", uint64(1<<1), 20480, 24)
			add(prefix+sub+"_scale.weight", uint64(1<<0), 3)
			add(prefix+sub+"_base.weight", uint64(1<<0), 24)
		}
		for _, name := range []string{"attn_norm", "ffn_norm"} {
			add(prefix+name+".weight", uint64(1<<0), 5120)
		}
		add(prefix+"attn_q_a.weight", dense, 5120, 1280)
		add(prefix+"attn_q_a_norm.weight", uint64(1<<0), 1280)
		add(prefix+"attn_q_b.weight", dense, 1280, 32768)
		add(prefix+"attn_kv.weight", dense, 5120, 512)
		add(prefix+"attn_kv_a_norm.weight", uint64(1<<0), 512)
		add(prefix+"attn_sinks.weight", uint64(1<<0), 64)
		add(prefix+"attn_output_a.weight", uint64(1<<8), 4096, 8192)
		add(prefix+"attn_output_b.weight", uint64(1<<8), 8192, 5120)
		if l.PublishKV {
			add(prefix+"attn_compressor_kv.weight", uint64(1<<1), 5120, 512)
			add(prefix+"attn_compressor_norm.weight", uint64(1<<0), 512)
			if l.Ratio == 2 {
				add(prefix+"attn_compressor_gate.weight", uint64(1<<1), 5120, 512)
			}
			add(prefix+"indexer.attn_k.weight", uint64(1<<1), 512, 128)
			add(prefix+"indexer.k_norm.weight", uint64(1<<0), 128)
		}
		if l.RefreshIndex {
			add(prefix+"indexer.attn_q_b.weight", uint64(1<<1), 1280, 4096)
			add(prefix+"indexer.proj.weight", uint64(1<<1), 5120, 32)
		}
		if l.Engram >= 0 {
			add(prefix+"engram_kv.weight", uint64(1<<1), 6144, 25600)
			add(prefix+"engram_q_norm.weight", uint64(1<<0), 5120, 4)
			add(prefix+"engram_k_norm.weight", uint64(1<<0), 5120, 4)
			rows := []int{384006168, 384016682}[l.Engram]
			all = append(all, requirement{prefix + "engram_embd.weight", []int{264, rows}, uint64(1 << 24), l.Engram})
		}
		add(prefix+"ffn_gate_inp.weight", uint64(1<<0), 5120, 384)
		add(prefix+"exp_probs_b.bias", uint64(1<<0), 384)
		for _, name := range []string{"ffn_gate_exps", "ffn_up_exps"} {
			add(prefix+name+".weight", routed, 5120, 2304, 384)
		}
		add(prefix+"ffn_down_exps.weight", routed, 2304, 5120, 384)
		for _, name := range []string{"ffn_gate_shexp", "ffn_up_shexp"} {
			add(prefix+name+".weight", dense, 5120, 2304)
		}
		add(prefix+"ffn_down_shexp.weight", dense, 2304, 5120)
	}
	return all
}
