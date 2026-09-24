package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"path/filepath"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/safetensors"
)

// encLayer holds one ModernBERT encoder layer on the device. Weights keep the
// safetensors [out, in] row-major layout; the GEMM helper transposes in cuBLAS.
type encLayer struct {
	AttnNorm cudamem.Slice[float32] // zero for layer 0, where it is the identity
	Wqkv     tensor
	Wo       tensor
	MLPNorm  cudamem.Slice[float32]
	Wi       tensor
	Wo2      tensor
}

// headLayer holds one nn.TransformerEncoderLayer of the decision head.
type headLayer struct {
	Norm1W, Norm1B cudamem.Slice[float32]
	Norm2W, Norm2B cudamem.Slice[float32]
	InW, OutW      tensor
	L1W, L2W       tensor
	InB, OutB      cudamem.Slice[float32]
	L1B, L2B       cudamem.Slice[float32]
}

// weights is the full model resident in device memory, plus the small host-side
// tensors used by the scorer and action head. The matrices, which are the
// operands of a product, are tensors in the pass's precision; the norms,
// biases and type embedding are float32 in every precision.
type weights struct {
	Emb       tensor
	EmbNorm   cudamem.Slice[float32]
	FinalNorm cudamem.Slice[float32]
	Layers    []encLayer
	Head      []headLayer
	TypeEmb   cudamem.Slice[float32]

	// Scorer and action head run on the host: a few hundred kFLOP each.
	ScLNW, ScLNB []float32
	ScW1, ScB1   []float32
	ScW2, ScB2   []float32
	ActW1, ActB1 []float32
	ActW2, ActB2 []float32
}

// loadWeights reads model.safetensors from dir and uploads the encoder and
// head parameters to dev. Every tensor is widened to float32, except that with
// half the matrices are narrowed to half instead: those are the only copy on
// the device, so a half pass holds half the weight memory.
func loadWeights(dir string, dev *cuda.Device, c *config, half bool) (*weights, error) {
	st, err := safetensors.Open(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("open weights: %w", err)
	}

	// host reads a tensor as float32 and checks its element count.
	host := func(name string, want int) ([]float32, error) {
		t, err := st.Tensor(name)
		if err != nil {
			return nil, fmt.Errorf("tensor %s: %w", name, err)
		}
		v, err := toFloat32(t.Data, string(t.Dtype))
		if err != nil {
			return nil, fmt.Errorf("tensor %s: %w", name, err)
		}
		if want > 0 && len(v) != want {
			return nil, fmt.Errorf("tensor %s: have %d elements, want %d", name, len(v), want)
		}
		return v, nil
	}
	// dvc uploads a tensor to the device.
	dvc := func(name string, want int) (cudamem.Slice[float32], error) {
		v, err := host(name, want)
		if err != nil {
			return cudamem.Slice[float32]{}, err
		}
		return upload(dev, v)
	}
	// mat uploads a matrix in the pass's precision. A half checkpoint tensor
	// is uploaded as it is stored.
	mat := func(name string, want int) (tensor, error) {
		if !half {
			v, err := dvc(name, want)
			return tensor{f32: v}, err
		}
		t, err := st.Tensor(name)
		if err != nil {
			return tensor{}, fmt.Errorf("tensor %s: %w", name, err)
		}
		raw := t.Data
		var h []uint16
		if t.Dtype == safetensors.F16 || t.Dtype == "float16" {
			h = make([]uint16, len(raw)/2)
			for i := range h {
				h[i] = binary.LittleEndian.Uint16(raw[i*2:])
			}
		} else {
			v, err := toFloat32(raw, string(t.Dtype))
			if err != nil {
				return tensor{}, fmt.Errorf("tensor %s: %w", name, err)
			}
			h = make([]uint16, len(v))
			for i, x := range v {
				h[i] = float32to16(x)
			}
		}
		if want > 0 && len(h) != want {
			return tensor{}, fmt.Errorf("tensor %s: have %d elements, want %d", name, len(h), want)
		}
		s, err := cudamem.Alloc[uint16](dev, len(h))
		if err != nil {
			return tensor{}, err
		}
		if err := s.CopyFromHost(h); err != nil {
			s.Free()
			return tensor{}, err
		}
		return tensor{f16: s}, nil
	}

	// Encoder tensors appear under one of these prefixes depending on how the
	// checkpoint was exported.
	enc := ""
	for _, p := range []string{"encoder.", "model.", ""} {
		if _, ok := st.Info(p + "embeddings.tok_embeddings.weight"); ok {
			enc = p
			break
		}
	}

	d, dff := c.D, c.DFF
	w := &weights{}

	if w.Emb, err = mat(enc+"embeddings.tok_embeddings.weight", c.Vocab*d); err != nil {
		return nil, err
	}
	if w.EmbNorm, err = dvc(enc+"embeddings.norm.weight", d); err != nil {
		return nil, err
	}
	if w.FinalNorm, err = dvc(enc+"final_norm.weight", d); err != nil {
		return nil, err
	}

	w.Layers = make([]encLayer, c.NumLayers)
	for i := range w.Layers {
		p := fmt.Sprintf("%slayers.%d.", enc, i)
		L := &w.Layers[i]
		if i > 0 {
			if L.AttnNorm, err = dvc(p+"attn_norm.weight", d); err != nil {
				return nil, err
			}
		}
		if L.Wqkv, err = mat(p+"attn.Wqkv.weight", 3*d*d); err != nil {
			return nil, err
		}
		if L.Wo, err = mat(p+"attn.Wo.weight", d*d); err != nil {
			return nil, err
		}
		if L.MLPNorm, err = dvc(p+"mlp_norm.weight", d); err != nil {
			return nil, err
		}
		if L.Wi, err = mat(p+"mlp.Wi.weight", 2*dff*d); err != nil {
			return nil, err
		}
		if L.Wo2, err = mat(p+"mlp.Wo.weight", d*dff); err != nil {
			return nil, err
		}
	}

	// The head feed-forward width comes from the checkpoint, not the config.
	if c.HeadLayers > 0 {
		info, ok := st.Info("head.layers.0.linear1.weight")
		if !ok {
			return nil, fmt.Errorf("head.layers.0.linear1.weight not found (head_layers=%d)", c.HeadLayers)
		}
		c.HeadFF = info.Shape[0]
	} else {
		c.HeadFF = 4 * d
	}
	hff := c.HeadFF

	w.Head = make([]headLayer, c.HeadLayers)
	for i := range w.Head {
		p := fmt.Sprintf("head.layers.%d.", i)
		H := &w.Head[i]
		for _, e := range []struct {
			dst  *cudamem.Slice[float32]
			name string
			n    int
		}{
			{&H.Norm1W, "norm1.weight", d}, {&H.Norm1B, "norm1.bias", d},
			{&H.Norm2W, "norm2.weight", d}, {&H.Norm2B, "norm2.bias", d},
			{&H.InB, "self_attn.in_proj_bias", 3 * d},
			{&H.OutB, "self_attn.out_proj.bias", d},
			{&H.L1B, "linear1.bias", hff},
			{&H.L2B, "linear2.bias", d},
		} {
			if *e.dst, err = dvc(p+e.name, e.n); err != nil {
				return nil, err
			}
		}
		for _, e := range []struct {
			dst  *tensor
			name string
			n    int
		}{
			{&H.InW, "self_attn.in_proj_weight", 3 * d * d},
			{&H.OutW, "self_attn.out_proj.weight", d * d},
			{&H.L1W, "linear1.weight", hff * d},
			{&H.L2W, "linear2.weight", d * hff},
		} {
			if *e.dst, err = mat(p+e.name, e.n); err != nil {
				return nil, err
			}
		}
	}

	if w.TypeEmb, err = dvc("type_emb.weight", 3*d); err != nil {
		return nil, err
	}

	a0, ok := st.Info("act_head.0.weight")
	if !ok {
		return nil, fmt.Errorf("act_head.0.weight not found")
	}
	a2, ok := st.Info("act_head.2.weight")
	if !ok {
		return nil, fmt.Errorf("act_head.2.weight not found")
	}
	c.ActHidden, c.NumAct = a0.Shape[0], a2.Shape[0]
	if c.NumAct > 16 {
		return nil, fmt.Errorf("act_head has %d outputs (max 16)", c.NumAct)
	}

	for _, e := range []struct {
		dst  *[]float32
		name string
		n    int
	}{
		{&w.ScLNW, "scorer.0.weight", d}, {&w.ScLNB, "scorer.0.bias", d},
		{&w.ScW1, "scorer.1.weight", d * d}, {&w.ScB1, "scorer.1.bias", d},
		{&w.ScW2, "scorer.3.weight", d}, {&w.ScB2, "scorer.3.bias", 1},
		{&w.ActW1, "act_head.0.weight", c.ActHidden * (d + 4)}, {&w.ActB1, "act_head.0.bias", c.ActHidden},
		{&w.ActW2, "act_head.2.weight", c.NumAct * c.ActHidden}, {&w.ActB2, "act_head.2.bias", c.NumAct},
	} {
		if *e.dst, err = host(e.name, e.n); err != nil {
			return nil, err
		}
	}
	return w, nil
}

func upload(dev *cuda.Device, v []float32) (cudamem.Slice[float32], error) {
	s, err := cudamem.Alloc[float32](dev, len(v))
	if err != nil {
		return cudamem.Slice[float32]{}, err
	}
	if err := s.CopyFromHost(v); err != nil {
		s.Free()
		return cudamem.Slice[float32]{}, err
	}
	return s, nil
}

// toFloat32 widens a safetensors payload to float32.
func toFloat32(raw []byte, dtype string) ([]float32, error) {
	switch dtype {
	case "F32", "float32":
		out := make([]float32, len(raw)/4)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return out, nil
	case "F16", "float16":
		out := make([]float32, len(raw)/2)
		for i := range out {
			out[i] = float16to32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
		return out, nil
	case "BF16", "bfloat16":
		out := make([]float32, len(raw)/2)
		for i := range out {
			out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[i*2:])) << 16)
		}
		return out, nil
	}
	return nil, fmt.Errorf("unsupported dtype %q", dtype)
}

// float16to32 widens an IEEE half, including subnormals, infinities and NaNs.
func float16to32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1f
	man := uint32(h & 0x3ff)
	switch exp {
	case 0:
		if man == 0 {
			return math.Float32frombits(sign)
		}
		// Subnormal: renormalize into a float32 normal.
		e := uint32(0)
		for man&0x400 == 0 {
			man <<= 1
			e++
		}
		man &= 0x3ff
		return math.Float32frombits(sign | (127-14-e)<<23 | man<<13)
	case 0x1f:
		return math.Float32frombits(sign | 0xff<<23 | man<<13)
	}
	return math.Float32frombits(sign | (exp+127-15)<<23 | man<<13)
}

// float32to16 narrows x to the nearest IEEE half, ties to even, as
// cvt.rn.f16.f32 does: overflow goes to infinity, and a value below the
// smallest subnormal to a signed zero.
func float32to16(x float32) uint16 {
	b := math.Float32bits(x)
	sign := uint16(b>>16) & 0x8000
	exp := int(b>>23) & 0xff
	man := b & 0x7fffff
	switch {
	case exp == 0xff: // inf or NaN; keep NaN a NaN
		if man != 0 {
			return sign | 0x7e00
		}
		return sign | 0x7c00
	case exp-127+15 >= 0x1f:
		return sign | 0x7c00
	case exp-127+15 <= 0: // subnormal half, or zero
		shift := uint(-1 - (exp - 127))
		if exp == 0 || shift > 24 {
			return sign
		}
		m := man | 0x800000
		h := m >> shift
		rem := m & (1<<shift - 1)
		halfway := uint32(1) << (shift - 1)
		if rem > halfway || rem == halfway && h&1 == 1 {
			h++
		}
		return sign | uint16(h)
	}
	h := uint32(exp-127+15)<<10 | man>>13
	rem := man & 0x1fff
	if rem > 0x1000 || rem == 0x1000 && h&1 == 1 {
		h++ // a carry into the exponent is the correct rounding, up to inf
	}
	return sign | uint16(h)
}
