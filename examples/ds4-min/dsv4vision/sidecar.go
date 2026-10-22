package dsv4vision

import (
	"fmt"
	"io"
	"math"

	"github.com/tmc/lib/gguf"
)

// EncoderRevision is the source checkpoint required by the Flash vision sidecar.
const EncoderRevision = "e46e16bf6035c6f317eb2ac7458eb0362926d402"

// Tensor describes a validated, bounded sidecar tensor. Offset is absolute.
// Dimensions are in GGUF order (input dimension first).
type Tensor struct {
	Name          string
	TypeID        uint32
	Dimensions    []uint64
	Offset, Bytes uint64
}

// Sidecar owns immutable tensor descriptors, not the underlying reader.
// Keep the reader open and its contents unchanged for the Sidecar's lifetime.
// A Sidecar does not claim a content hash or execute a vision encoder.
type Sidecar struct {
	r       io.ReaderAt
	tensors map[string]Tensor
}

type tensorContract struct {
	name   string
	typeID uint32
	dims   []uint64
}

func sidecarContracts() []tensorContract {
	var out []tensorContract
	add := func(name string, typ uint32, dims ...uint64) { out = append(out, tensorContract{name, typ, dims}) }
	for _, c := range []struct {
		name string
		dims []uint64
	}{
		{"vision.patch_embed.proj.weight", []uint64{588, 1024}},
		{"vision.patch_embed.proj.bias", []uint64{1024}},
		{"vision.norm.weight", []uint64{1024}},
		{"aligner.w1.weight", []uint64{9216, 4096}},
		{"aligner.w1.bias", []uint64{4096}},
		{"aligner.w2.weight", []uint64{4096, 4096}},
		{"aligner.w2.bias", []uint64{4096}},
		{"image_start", []uint64{4096}},
		{"image_pad", []uint64{4096}},
		{"image_newline", []uint64{4096}},
		{"image_end", []uint64{4096}},
	} {
		add(c.name, 30, c.dims...)
	}
	for layer := 0; layer < 32; layer++ {
		for _, c := range []struct {
			name string
			dims []uint64
		}{
			{"norm1.weight", []uint64{1024}},
			{"attn.wqkv.weight", []uint64{1024, 3072}},
			{"attn.wqkv.bias", []uint64{3072}},
			{"attn.wo.weight", []uint64{1024, 1024}},
			{"attn.wo.bias", []uint64{1024}},
			{"norm2.weight", []uint64{1024}},
			{"mlp.w1.weight", []uint64{1024, 5632}},
			{"mlp.w2.weight", []uint64{2816, 1024}},
		} {
			add(fmt.Sprintf("vision.blocks.%d.%s", layer, c.name), 30, c.dims...)
		}
	}
	for layer := 0; layer < 43; layer++ {
		add(fmt.Sprintf("layers.%d.ffn.gate.bias_vl", layer), 0, 256)
	}
	for layer := 0; layer < 3; layer++ {
		add(fmt.Sprintf("mtp.%d.ffn.gate.bias_vl", layer), 0, 256)
		add(fmt.Sprintf("layers.%d.ffn.gate.bias", layer), 0, 256)
	}
	return out
}

var sidecarDimensions = []struct {
	key   string
	value uint64
}{
	{"block_count", 32}, {"embedding_length", 1024}, {"feed_forward_length", 2816},
	{"attention.head_count", 16}, {"projection_length", 4096}, {"patch_size", 14},
	{"downsample_ratio", 3}, {"image.max_tokens", 384}, {"image.min_pixels", 147456},
	{"image.max_width_height_ratio", 8}, {"language.block_count", 43}, {"language.expert_count", 256},
}

// ReadSidecar parses and binds the exact Flash Vision encoder metadata and all
// 316 tensor descriptors. It reads headers only, not tensor contents. It rejects
// overlapping or truncated ranges before exposing any tensor reader.
func ReadSidecar(r io.ReaderAt, size int64) (*Sidecar, error) {
	if r == nil {
		return nil, fmt.Errorf("vision sidecar: nil reader")
	}
	g, err := gguf.Read(r, size)
	if err != nil {
		return nil, fmt.Errorf("vision sidecar: %w", err)
	}
	for _, want := range []struct{ key, value string }{
		{"general.architecture", "deepseek4-vision"},
		{"general.source.revision", EncoderRevision},
		{"deepseek4-vision.checkpoint_variant", "vision-exp"},
	} {
		v, ok := g.Metadata[want.key]
		s, sok := v.Value.(string)
		if !ok || !sok || v.TypeID != gguf.TypeString || s != want.value {
			return nil, fmt.Errorf("vision sidecar: %s mismatch", want.key)
		}
	}
	for _, want := range sidecarDimensions {
		key := "deepseek4-vision." + want.key
		v, ok := g.Metadata[key]
		n, nok := v.Uint()
		if !ok || v.TypeID != gguf.TypeUint32 || !nok || n != want.value {
			return nil, fmt.Errorf("vision sidecar: %s mismatch", key)
		}
	}
	eps, ok := g.Metadata["deepseek4-vision.attention.layer_norm_rms_epsilon"]
	var value float32
	switch v := eps.Value.(type) {
	case float32:
		value = v
	case float64:
		value = float32(v)
	default:
		return nil, fmt.Errorf("vision sidecar: invalid RMS epsilon type")
	}
	if !ok || value != float32(1e-6) || math.IsNaN(float64(value)) {
		return nil, fmt.Errorf("vision sidecar: RMS epsilon mismatch")
	}
	contracts := sidecarContracts()
	if len(g.Tensors) != len(contracts) {
		return nil, fmt.Errorf("vision sidecar: %d tensors, want %d", len(g.Tensors), len(contracts))
	}
	byName := make(map[string]gguf.Tensor, len(g.Tensors))
	for _, t := range g.Tensors {
		if _, ok := byName[t.Name]; ok {
			return nil, fmt.Errorf("vision sidecar: duplicate tensor %q", t.Name)
		}
		byName[t.Name] = t
	}
	s := &Sidecar{r: r, tensors: make(map[string]Tensor, len(contracts))}
	for _, c := range contracts {
		t, ok := byName[c.name]
		if !ok {
			return nil, fmt.Errorf("vision sidecar: missing tensor %q", c.name)
		}
		if t.TypeID != c.typeID || len(t.Dimensions) != len(c.dims) {
			return nil, fmt.Errorf("vision sidecar: tensor %q type or rank mismatch", c.name)
		}
		for j, d := range c.dims {
			if t.Dimensions[j] != d {
				return nil, fmt.Errorf("vision sidecar: tensor %q dimension %d mismatch", c.name, j)
			}
		}
		bytes, err := t.Bytes()
		if err != nil {
			return nil, fmt.Errorf("vision sidecar: %w", err)
		}
		// gguf.Read has already proved aligned canonical offsets and complete
		// ranges within the nonnegative int64 file size.
		off := g.DataOffset + t.Offset
		d := Tensor{c.name, c.typeID, append([]uint64(nil), c.dims...), off, bytes}
		s.tensors[c.name] = d
	}
	return s, nil
}

// Tensor returns a copy of a tensor descriptor. Changing its Dimensions cannot
// change the Sidecar's validated directory.
func (s *Sidecar) Tensor(name string) (Tensor, error) {
	if s == nil {
		return Tensor{}, fmt.Errorf("vision sidecar: nil sidecar")
	}
	t, ok := s.tensors[name]
	if !ok {
		return Tensor{}, fmt.Errorf("vision sidecar: unknown tensor %q", name)
	}
	t.Dimensions = append([]uint64(nil), t.Dimensions...)
	return t, nil
}

// Reader returns a bounded, independent read-only view of one tensor. It never
// reads adjacent tensors and does not close the underlying reader.
func (s *Sidecar) Reader(name string) (*io.SectionReader, error) {
	t, err := s.Tensor(name)
	if err != nil {
		return nil, err
	}
	return io.NewSectionReader(s.r, int64(t.Offset), int64(t.Bytes)), nil
}
