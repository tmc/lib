package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/examples/ds4-min/dsv41"
	"github.com/tmc/lib/gguf"
)

// This catalog fixture copies the released V4.1 tensor and metadata contracts.
// Payload reads return bounded zeros; no forward evaluation is performed.
var v41FixtureIntegers = map[string]uint32{
	"vocab_size": 129280, "hidden_size": 5120, "moe_intermediate_size": 2304, "num_hidden_layers": 40,
	"num_attention_heads": 64, "num_key_value_heads": 1, "head_dim": 512, "qk_rope_head_dim": 64,
	"q_lora_rank": 1280, "o_lora_rank": 1024, "o_groups": 8, "n_routed_experts": 384, "n_shared_experts": 1,
	"num_experts_per_tok": 6, "max_position_embeddings": 1048576, "sliding_window": 128,
	"index_n_heads": 32, "index_head_dim": 128, "index_topk": 512, "candidate_source_layer_id": 20,
	"candidate_topk_blocks": 2048, "candidate_block_size": 8, "hc_mult": 4, "hc_sinkhorn_iters": 20,
	"rope_theta": 10000, "compress_rope_theta": 160000, "engram.compressed_vocab_size": 99092,
}

var v41FixtureFloats = map[string]float32{
	"rms_norm_eps": 1e-20, "hc_eps": 1e-6, "swiglu_limit": 10, "routed_scaling_factor": 1.5,
	"rope_scaling.factor": 16, "rope_scaling.beta_fast": 32, "rope_scaling.beta_slow": 1,
	"rope_scaling.original_max_position_embeddings": 65536,
}

type v41FixtureRequirement struct {
	name  string
	dims  []int
	types uint64
	table int
}

func v41FixtureRequirements() []v41FixtureRequirement {
	var all []v41FixtureRequirement
	add := func(name string, types uint64, dims ...int) {
		all = append(all, v41FixtureRequirement{name, dims, types, -1})
	}
	const dense = uint64(1<<8 | 1<<12 | 1<<2) // Actual upstream dense set; Catalog qualifies supported types.
	const routed = uint64(1<<8 | 1<<16 | 1<<10 | 1<<12 | 1<<13 | 1<<14 | 1<<39)
	add("token_embd.weight", uint64(1<<1), 5120, 129280)
	add("output_norm.weight", uint64(1<<0), 5120)
	add("output.weight", dense, 5120, 129280)
	for layer := 0; layer < 40; layer++ {
		prefix := fmt.Sprintf("blk.%d.", layer)
		l, _ := dsv41.Schedule(layer)
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
			all = append(all, v41FixtureRequirement{prefix + "engram_embd.weight", []int{264, rows}, uint64(1 << 24), l.Engram})
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
func v41MetadataFixture() *gguf.File {
	g := &gguf.File{Metadata: map[string]gguf.Value{"general.architecture": {TypeID: gguf.TypeString, Value: "deepseek41"}}}
	for key, value := range v41FixtureIntegers {
		g.Metadata["deepseek41."+key] = gguf.Value{TypeID: gguf.TypeUint32, Value: value}
	}
	for key, value := range v41FixtureFloats {
		g.Metadata["deepseek41."+key] = gguf.Value{TypeID: gguf.TypeFloat32, Value: value}
	}
	for key, value := range map[string]string{"scoring_func": "sqrtsoftplus", "hidden_act": "silu", "topk_method": "noaux_tc", "engram.encoding": "e4m3_e8m0_32_row264"} {
		g.Metadata["deepseek41."+key] = gguf.Value{TypeID: gguf.TypeString, Value: value}
	}
	g.Metadata["deepseek41.norm_topk_prob"] = gguf.Value{TypeID: gguf.TypeBool, Value: true}
	arrays := map[string][]uint32{"kv_source_layer_ids": {2, 8, 14, 20}, "index_source_layer_ids": {2, 8, 14, 20, 24, 28, 32, 36}, "engram.layer_ids": {1, 14}, "engram.rows": {384006168, 384016682}, "engram.pad_id": nil}
	delete(arrays, "engram.pad_id")
	ratios := make([]uint32, 40)
	for i := range ratios {
		l, _ := dsv41.Schedule(i)
		ratios[i] = uint32(l.Ratio)
	}
	arrays["compress_ratios"] = ratios
	arrays["engram.token_map"] = make([]uint32, 129280)
	primes := make([]uint32, 48)
	for i := range primes {
		primes[i] = 2
	}
	primes[23] = 384006168 - 46
	primes[47] = 384016682 - 46
	arrays["engram.primes"] = primes
	for key, values := range arrays {
		a := make([]any, len(values))
		for i, v := range values {
			a[i] = v
		}
		g.Metadata["deepseek41."+key] = gguf.Value{TypeID: gguf.TypeArray, Value: a}
	}
	g.Metadata["deepseek41.engram.pad_id"] = gguf.Value{TypeID: gguf.TypeUint32, Value: uint32(0)}
	multipliers := make([]any, 8)
	for i := range multipliers {
		multipliers[i] = uint64(1)
	}
	g.Metadata["deepseek41.engram.multipliers"] = gguf.Value{TypeID: gguf.TypeArray, Value: multipliers}
	return g
}

type v41ZeroReader struct {
	reads int
	size  int64
}

func (r *v41ZeroReader) ReadAt(dst []byte, off int64) (int, error) {
	if off < 0 || off > r.size-int64(len(dst)) {
		return 0, io.ErrUnexpectedEOF
	}
	r.reads++
	clear(dst)
	return len(dst), nil
}

func v41ModelFixture(t *testing.T) *gguf.File {
	t.Helper()
	g := v41MetadataFixture()
	g.DataOffset = 4096
	var offset uint64
	for _, need := range v41FixtureRequirements() {
		var kind uint32
		for need.types&(uint64(1)<<kind) == 0 {
			kind++
		}
		if need.types&(1<<8) != 0 {
			kind = 8
		}
		if len(need.dims) == 3 {
			kind = 16
			if need.dims[0] == 2304 {
				kind = 10
			}
		}
		tensor := gguf.Tensor{Name: need.name, TypeID: kind, Offset: offset}
		for _, dim := range need.dims {
			tensor.Dimensions = append(tensor.Dimensions, uint64(dim))
		}
		n, err := tensor.Bytes()
		if err != nil {
			t.Fatal(err)
		}
		offset += (n + 31) &^ 31
		g.Tensors = append(g.Tensors, tensor)
	}
	g.Size = g.DataOffset + offset
	return g
}

func TestV41BoundSessionHost(t *testing.T) {
	g := v41ModelFixture(t)
	if len(g.Tensors) != 1006 {
		t.Fatalf("tensor count = %d", len(g.Tensors))
	}
	r := &v41ZeroReader{size: int64(g.Size)}
	cfg := config{family: "auto", backend: "cpu", context: 4}
	if err := validateFamilyProfile(&cfg, g); err != nil || cfg.family != "v41" {
		t.Fatalf("auto family = %s, %v", cfg.family, err)
	}
	m, err := dsv41.Bind(r, int64(g.Size), g)
	if err != nil {
		t.Fatal(err)
	}
	if r.reads != 0 {
		t.Fatal("Bind read payloads")
	}
	b, err := dsv41.NewBackend(m)
	if err != nil {
		t.Fatal(err)
	}
	for _, context := range []int{0, 1048577} {
		if _, _, err := newStateSessionBackend(b, context); err == nil {
			t.Fatal("invalid context admitted")
		}
	}
	s, closer, err := newStateSessionBackend(b, 4)
	if err != nil {
		t.Fatal(err)
	}
	if r.reads != 16 {
		t.Fatalf("normalization reads = %d", r.reads)
	}
	if err := s.Reset(); err != nil {
		t.Fatal(err)
	}
	if err := closer.Close(); err != nil {
		t.Fatal(err)
	}
	if !errors.Is(s.Reset(), dsv4.ErrClosed) {
		t.Fatal("closed session usable")
	}
	if err := closer.Close(); err != nil {
		t.Fatal(err)
	}
}

var _ io.ReaderAt = (*v41ZeroReader)(nil)

// v41FixtureHeader encodes the literal metadata and tensor directory so CLI
// gates exercise the actual GGUF reader. Payloads remain bounded zero views.
func v41FixtureHeader(t *testing.T, g *gguf.File) []byte {
	t.Helper()
	var b bytes.Buffer
	put := func(v any) {
		if err := binary.Write(&b, binary.LittleEndian, v); err != nil {
			t.Fatal(err)
		}
	}
	text := func(s string) { put(uint64(len(s))); b.WriteString(s) }
	b.WriteString("GGUF")
	put(uint32(3))
	put(uint64(len(g.Tensors)))
	put(uint64(len(g.Metadata)))
	keys := make([]string, 0, len(g.Metadata))
	for key := range g.Metadata {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		v := g.Metadata[key]
		text(key)
		put(v.TypeID)
		switch value := v.Value.(type) {
		case string:
			text(value)
		case uint32:
			put(value)
		case float32:
			put(math.Float32bits(value))
		case bool:
			if value {
				put(uint8(1))
			} else {
				put(uint8(0))
			}
		case []any:
			if len(value) == 0 {
				t.Fatal("empty fixture array")
			}
			switch value[0].(type) {
			case uint32:
				put(uint32(gguf.TypeUint32))
			case uint64:
				put(uint32(gguf.TypeUint64))
			default:
				t.Fatalf("unsupported fixture array %s", key)
			}
			put(uint64(len(value)))
			for _, item := range value {
				put(item)
			}
		default:
			t.Fatalf("unsupported fixture metadata %s", key)
		}
	}
	for _, tensor := range g.Tensors {
		text(tensor.Name)
		put(uint32(len(tensor.Dimensions)))
		for _, dim := range tensor.Dimensions {
			put(dim)
		}
		put(tensor.TypeID)
		put(tensor.Offset)
	}
	for b.Len()%32 != 0 {
		b.WriteByte(0)
	}
	return b.Bytes()
}

func TestV41CLITokenizerAdmission(t *testing.T) {
	header := v41FixtureHeader(t, v41MetadataFixture())
	path := filepath.Join(t.TempDir(), "metadata.gguf")
	if err := os.WriteFile(path, header, 0600); err != nil {
		t.Fatal(err)
	}
	args := []string{"-model", path, "-backend", "cpu", "-prompt", "x", "-ctx", "4", "-tokens", "1"}
	err := run(args, strings.NewReader(""), &bytes.Buffer{}, &bytes.Buffer{})
	if !errors.Is(err, errInvalidTokenizer) {
		t.Fatalf("CLI did not reach tokenizer admission: %v", err)
	}
	args = append(args, "-family", "flash")
	err = run(args, strings.NewReader(""), &bytes.Buffer{}, &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "does not match") {
		t.Fatalf("CLI family mismatch = %v", err)
	}
}

type v41CatalogFile struct {
	header        []byte
	size          int64
	reads, closes int
}

func (f *v41CatalogFile) ReadAt(dst []byte, offset int64) (int, error) {
	if offset < 0 || offset > f.size-int64(len(dst)) {
		return 0, io.ErrUnexpectedEOF
	}
	f.reads++
	clear(dst)
	if offset < int64(len(f.header)) {
		copy(dst, f.header[offset:])
	}
	return len(dst), nil
}
func (f *v41CatalogFile) Close() error               { f.closes++; return nil }
func (f *v41CatalogFile) Stat() (os.FileInfo, error) { return v41CatalogInfo{f.size}, nil }

type v41CatalogInfo struct{ size int64 }

func (f v41CatalogInfo) Name() string       { return "catalog.gguf" }
func (f v41CatalogInfo) Size() int64        { return f.size }
func (f v41CatalogInfo) Mode() os.FileMode  { return 0600 }
func (f v41CatalogInfo) ModTime() time.Time { return time.Time{} }
func (f v41CatalogInfo) IsDir() bool        { return false }
func (f v41CatalogInfo) Sys() any           { return nil }

func TestV41CatalogConstructorSession(t *testing.T) {
	g := v41ModelFixture(t)
	header := v41FixtureHeader(t, g)
	f := &v41CatalogFile{header: header, size: int64(g.Size-g.DataOffset) + int64(len(header))}
	b, err := openV41CPUBackend("catalog", v41CPUOptions{}, func(string) (v41ModelFile, error) { return f, nil })
	if err != nil {
		t.Fatal(err)
	}
	s, closer, err := newStateSessionBackend(b, 4)
	if err != nil {
		t.Fatal(err)
	}
	if f.closes != 0 {
		t.Fatal("model file closed before state")
	}
	if err := s.Reset(); err != nil {
		t.Fatal(err)
	}
	if err := closer.Close(); err != nil {
		t.Fatal(err)
	}
	if f.closes != 1 || !errors.Is(s.Reset(), dsv4.ErrClosed) {
		t.Fatal("session retirement lost owner")
	}
	if err := closer.Close(); err != nil || f.closes != 1 {
		t.Fatal("file retired twice", err)
	}
}
