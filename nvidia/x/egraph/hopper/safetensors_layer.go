package hopper

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
	"github.com/tmc/lib/nvidia/x/safetensors"
)

// LayerSafetensorsSpec describes tensor naming conventions for extracting a
// single transformer decoder layer from a safetensors archive. Zero-valued
// fields fall back to standard HuggingFace tensor names under Prefix.
type LayerSafetensorsSpec struct {
	Prefix       string
	InputNorm    string
	QKVWeight    string
	QWeight      string
	KWeight      string
	VWeight      string
	OWeight      string
	PostAttnNorm string
	GateUpWeight string
	GateWeight   string
	UpWeight     string
	DownWeight   string
}

// RepackedLayerWeights holds extracted and hardware-repacked weight and scale
// buffers for a single transformer decoder layer loaded from a safetensors archive.
type RepackedLayerWeights struct {
	Prefix        string
	WeightDType   DType
	RepackedNVFP4 bool
	InputNorm     []uint16 // BF16 [HiddenSize]
	QKVWeight     []byte   // BF16, E4M3, or repacked NVFP4 [QKV_N, HiddenSize]
	QKVScale      []byte   // optional E4M3 block scales for NVFP4
	OWeight       []byte   // [HiddenSize, NumAttentionHeads * HeadDim]
	OScale        []byte
	PostAttnNorm  []uint16 // BF16 [HiddenSize]
	// Dense MLP weights:
	GateUpWeight []byte // [2 * IntermediateSize, HiddenSize]
	GateUpScale  []byte
	DownWeight   []byte // [HiddenSize, IntermediateSize]
	DownScale    []byte
}

// RepackNVFP4Weights losslessly repacks an [n, k] NVFP4 weight matrix (packed
// [n, k/2] E2M1 bytes) into Hopper WGMMA 2-word bitopt dequant order
// ([RepackNVFP4Dequant2Word] when n%128 == 0 and k%64 == 0, or 2-word
// [PackDequant2Words] blocks for smaller tile dimensions).
func RepackNVFP4Weights(packed []byte, n, k int) ([]byte, error) {
	if n <= 0 || k <= 0 || k%16 != 0 {
		return nil, fmt.Errorf("repack nvfp4 weights: invalid shape [%d, %d]", n, k)
	}
	if len(packed) != n*k/2 {
		return nil, fmt.Errorf("repack nvfp4 weights: shape [%d, %d] needs %d bytes, have %d", n, k, n*k/2, len(packed))
	}
	if n%128 == 0 && k%64 == 0 {
		dummyScales := make([]uint8, n*k/16)
		wq, _, err := RepackNVFP4Dequant2Word(packed, dummyScales, n, k, false)
		if err != nil {
			return nil, err
		}
		out := make([]byte, 4*len(wq))
		for i, w := range wq {
			binary.LittleEndian.PutUint32(out[4*i:], w)
		}
		return out, nil
	}
	if len(packed)%8 != 0 {
		return nil, fmt.Errorf("repack nvfp4 weights: byte length %d must be a multiple of 8", len(packed))
	}
	out := make([]byte, len(packed))
	for i := 0; i < len(packed); i += 8 {
		c0 := binary.LittleEndian.Uint32(packed[i : i+4])
		c1 := binary.LittleEndian.Uint32(packed[i+4 : i+8])
		q := PackDequant2Words(c0, c1)
		binary.LittleEndian.PutUint32(out[i:i+4], q[0])
		binary.LittleEndian.PutUint32(out[i+4:i+8], q[1])
	}
	return out, nil
}

// UnpackNVFP4Weights reverses [RepackNVFP4Weights], recovering the original
// [n, k/2] packed E2M1 bytes bit-for-bit.
func UnpackNVFP4Weights(repacked []byte, n, k int) ([]byte, error) {
	if n <= 0 || k <= 0 || k%16 != 0 {
		return nil, fmt.Errorf("unpack nvfp4 weights: invalid shape [%d, %d]", n, k)
	}
	if len(repacked) != n*k/2 {
		return nil, fmt.Errorf("unpack nvfp4 weights: shape [%d, %d] needs %d bytes, have %d", n, k, n*k/2, len(repacked))
	}
	if n%128 == 0 && k%64 == 0 {
		wq := make([]uint32, len(repacked)/4)
		for i := range wq {
			wq[i] = binary.LittleEndian.Uint32(repacked[4*i:])
		}
		dummyWS := make([]uint8, (n/128)*(k/64)*256*8)
		packed, _, err := UnpackNVFP4Dequant2Word(wq, dummyWS, n, k, false)
		if err != nil {
			return nil, err
		}
		return packed, nil
	}
	if len(repacked)%8 != 0 {
		return nil, fmt.Errorf("unpack nvfp4 weights: byte length %d must be a multiple of 8", len(repacked))
	}
	out := make([]byte, len(repacked))
	for i := 0; i < len(repacked); i += 8 {
		q0 := binary.LittleEndian.Uint32(repacked[i : i+4])
		q1 := binary.LittleEndian.Uint32(repacked[i+4 : i+8])
		c0, c1 := UnpackDequant2Words([2]uint32{q0, q1})
		binary.LittleEndian.PutUint32(out[i:i+4], c0)
		binary.LittleEndian.PutUint32(out[i+4:i+8], c1)
	}
	return out, nil
}

// RepackNVFP4Scales losslessly repacks an [n, k/16] E4M3 block-scale matrix
// into Hopper WGMMA per-thread scale layout ([RepackNVFP4Dequant2Word] when
// n%128 == 0 and k%64 == 0, or column-major k-block order for smaller tiles).
func RepackNVFP4Scales(scales []byte, n, k int) ([]byte, error) {
	if n <= 0 || k <= 0 || k%16 != 0 {
		return nil, fmt.Errorf("repack nvfp4 scales: invalid shape [%d, %d]", n, k)
	}
	if len(scales) != n*k/16 {
		return nil, fmt.Errorf("repack nvfp4 scales: shape [%d, %d] needs %d scale bytes, have %d", n, k, n*k/16, len(scales))
	}
	if n%128 == 0 && k%64 == 0 {
		dummyPacked := make([]uint8, n*k/2)
		_, ws, err := RepackNVFP4Dequant2Word(dummyPacked, scales, n, k, false)
		if err != nil {
			return nil, err
		}
		return ws, nil
	}
	cols := k / 16
	out := make([]byte, len(scales))
	for r := range n {
		for c := range cols {
			out[c*n+r] = scales[r*cols+c]
		}
	}
	return out, nil
}

// UnpackNVFP4Scales reverses [RepackNVFP4Scales], recovering the original
// [n, k/16] E4M3 block-scale bytes bit-for-bit.
func UnpackNVFP4Scales(repacked []byte, n, k int) ([]byte, error) {
	if n <= 0 || k <= 0 || k%16 != 0 {
		return nil, fmt.Errorf("unpack nvfp4 scales: invalid shape [%d, %d]", n, k)
	}
	if n%128 == 0 && k%64 == 0 {
		wantWS := (n / 128) * (k / 64) * 256 * 8
		if len(repacked) == wantWS {
			dummyWQ := make([]uint32, (n/128)*(k/64)*256*4)
			_, scales, err := UnpackNVFP4Dequant2Word(dummyWQ, repacked, n, k, false)
			if err != nil {
				return nil, err
			}
			return scales, nil
		}
	}
	if len(repacked) != n*k/16 {
		return nil, fmt.Errorf("unpack nvfp4 scales: shape [%d, %d] needs %d bytes, have %d", n, k, n*k/16, len(repacked))
	}
	cols := k / 16
	out := make([]byte, len(repacked))
	for r := range n {
		for c := range cols {
			out[r*cols+c] = repacked[c*n+r]
		}
	}
	return out, nil
}

// LoadLayerSafetensors parses stData using nvidia/x/safetensors, extracts the
// weights and scales of layerPrefix (supporting either fused qkv_proj /
// gate_up_proj or separate q_proj/k_proj/v_proj and gate_proj/up_proj tensors),
// and optionally repacks NVFP4 weights and E4M3 block scales for Hopper WGMMA.
func LoadLayerSafetensors(stData []byte, cfg ArchConfig, layerPrefix string, repackForWGMMA bool) (*RepackedLayerWeights, error) {
	var err error
	cfg, err = cfg.norm()
	if err != nil {
		return nil, err
	}
	f, err := safetensors.NewFile(bytes.NewReader(stData), int64(len(stData)))
	if err != nil {
		return nil, fmt.Errorf("load layer safetensors: %w", err)
	}

	prefix := strings.TrimSuffix(strings.TrimSpace(layerPrefix), ".")
	fullName := func(suffix string) string {
		if prefix == "" {
			return suffix
		}
		return prefix + "." + suffix
	}
	findTensor := func(suffixes ...string) (safetensors.Tensor, bool, error) {
		for _, s := range suffixes {
			name := fullName(s)
			if _, ok := f.Info(name); ok {
				t, err := f.Tensor(name)
				return t, true, err
			}
			if prefix != "" {
				if _, ok := f.Info(s); ok {
					t, err := f.Tensor(s)
					return t, true, err
				}
			}
		}
		return safetensors.Tensor{}, false, nil
	}

	H := cfg.HiddenSize
	I := cfg.IntermediateSize
	hq := cfg.NumAttentionHeads
	hkv := cfg.NumKeyValueHeads
	headDim := cfg.HeadDim
	if headDim <= 0 {
		headDim = H / hq
	}
	qDim := hq * headDim
	kvDim := hkv * headDim
	qkvDim := qDim + 2*kvDim

	loadNorm := func(label string, suffixes ...string) ([]uint16, error) {
		t, ok, err := findTensor(suffixes...)
		if err != nil {
			return nil, err
		}
		if !ok {
			return nil, fmt.Errorf("load layer safetensors: missing %s tensor (prefix %q)", label, prefix)
		}
		return decodeNormBF16(t, H)
	}

	inNorm, err := loadNorm("input_layernorm",
		"input_layernorm.weight", "attn_norm.weight", "attention_norm.weight", "attn_norm_w")
	if err != nil {
		return nil, err
	}
	postNorm, err := loadNorm("post_attention_layernorm",
		"post_attention_layernorm.weight", "ffn_norm.weight", "post_attn_norm.weight", "ffn_norm_w")
	if err != nil {
		return nil, err
	}

	alreadyRepacked := f.Metadata != nil && f.Metadata["repacked_nvfp4"] == "true"

	// Load QKV weight (fused or separate [Q; K; V]) and optional scale.
	var qkvWeight, qkvScale []byte
	qkvRows := qkvDim
	if tQKV, ok, err := findTensor(
		"self_attn.qkv_proj.weight", "attn.qkv_proj.weight", "qkv_proj.weight", "wqkv", "wq",
	); err != nil {
		return nil, err
	} else if ok {
		qkvWeight = tQKV.Data
		if len(tQKV.Shape) >= 1 && tQKV.Shape[0] > 0 {
			qkvRows = tQKV.Shape[0]
		}
		if cfg.WeightDType == NVFP4 {
			tScale, okS, err := findTensor(
				"self_attn.qkv_proj.weight_scale", "self_attn.qkv_proj.scale",
				"attn.qkv_proj.weight_scale", "qkv_proj.weight_scale", "qkv_proj.scale", "wq_sf",
			)
			if err != nil {
				return nil, err
			}
			if !okS {
				return nil, fmt.Errorf("load layer safetensors: missing qkv_proj scale for NVFP4")
			}
			qkvScale = tScale.Data
		}
	} else {
		tQ, okQ, err := findTensor("self_attn.q_proj.weight", "attn.q_proj.weight", "q_proj.weight")
		if err != nil {
			return nil, err
		}
		tK, okK, err := findTensor("self_attn.k_proj.weight", "attn.k_proj.weight", "k_proj.weight")
		if err != nil {
			return nil, err
		}
		tV, okV, err := findTensor("self_attn.v_proj.weight", "attn.v_proj.weight", "v_proj.weight")
		if err != nil {
			return nil, err
		}
		if !okQ || !okK || !okV {
			return nil, fmt.Errorf("load layer safetensors: missing qkv_proj or q_proj/k_proj/v_proj weights")
		}
		if len(tQ.Shape) == 2 && tQ.Shape[0] > 0 {
			qDim = tQ.Shape[0]
		}
		if len(tK.Shape) == 2 && tK.Shape[0] > 0 {
			kvDim = tK.Shape[0]
		}
		qkvDim = qDim + 2*kvDim
		if err := checkWeightBytes("q_proj", tQ.Data, qDim, H, cfg.WeightDType); err != nil {
			return nil, err
		}
		if err := checkWeightBytes("k_proj", tK.Data, kvDim, H, cfg.WeightDType); err != nil {
			return nil, err
		}
		if err := checkWeightBytes("v_proj", tV.Data, kvDim, H, cfg.WeightDType); err != nil {
			return nil, err
		}
		qkvWeight = slices.Concat(tQ.Data, tK.Data, tV.Data)
		qkvRows = qkvDim
		if cfg.WeightDType == NVFP4 {
			sQ, okSQ, err := findTensor("self_attn.q_proj.weight_scale", "self_attn.q_proj.scale", "q_proj.weight_scale", "q_proj.scale")
			if err != nil {
				return nil, err
			}
			sK, okSK, err := findTensor("self_attn.k_proj.weight_scale", "self_attn.k_proj.scale", "k_proj.weight_scale", "k_proj.scale")
			if err != nil {
				return nil, err
			}
			sV, okSV, err := findTensor("self_attn.v_proj.weight_scale", "self_attn.v_proj.scale", "v_proj.weight_scale", "v_proj.scale")
			if err != nil {
				return nil, err
			}
			if !okSQ || !okSK || !okSV {
				return nil, fmt.Errorf("load layer safetensors: missing q_proj/k_proj/v_proj scale tensors for NVFP4")
			}
			qkvScale = slices.Concat(sQ.Data, sK.Data, sV.Data)
		}
	}
	if err := checkWeightBytes("qkv_proj", qkvWeight, qkvRows, H, cfg.WeightDType); err != nil {
		return nil, err
	}

	// Load O projection weight and optional scale.
	tO, okO, err := findTensor("self_attn.o_proj.weight", "attn.o_proj.weight", "o_proj.weight", "wo", "wo_q")
	if err != nil {
		return nil, err
	}
	if !okO {
		return nil, fmt.Errorf("load layer safetensors: missing o_proj.weight")
	}
	if err := checkWeightBytes("o_proj", tO.Data, H, qDim, cfg.WeightDType); err != nil {
		return nil, err
	}
	oWeight := slices.Clone(tO.Data)
	var oScale []byte
	if cfg.WeightDType == NVFP4 {
		sO, okSO, err := findTensor("self_attn.o_proj.weight_scale", "self_attn.o_proj.scale", "o_proj.weight_scale", "o_proj.scale", "wo_sf")
		if err != nil {
			return nil, err
		}
		if !okSO {
			return nil, fmt.Errorf("load layer safetensors: missing o_proj scale for NVFP4")
		}
		oScale = slices.Clone(sO.Data)
	}

	// Load GateUp weight (fused or separate [Gate; Up]) and optional scale.
	var gateUpWeight, gateUpScale []byte
	if tGU, okGU, err := findTensor(
		"mlp.gate_up_proj.weight", "gate_up_proj.weight", "w_gate_up", "w_gate_up_q",
	); err != nil {
		return nil, err
	} else if okGU {
		gateUpWeight = slices.Clone(tGU.Data)
		if cfg.WeightDType == NVFP4 {
			sGU, okSGU, err := findTensor(
				"mlp.gate_up_proj.weight_scale", "mlp.gate_up_proj.scale",
				"gate_up_proj.weight_scale", "gate_up_proj.scale", "w_gate_up_sf",
			)
			if err != nil {
				return nil, err
			}
			if !okSGU {
				return nil, fmt.Errorf("load layer safetensors: missing gate_up_proj scale for NVFP4")
			}
			gateUpScale = slices.Clone(sGU.Data)
		}
	} else {
		tGate, okG, err := findTensor("mlp.gate_proj.weight", "gate_proj.weight")
		if err != nil {
			return nil, err
		}
		tUp, okU, err := findTensor("mlp.up_proj.weight", "up_proj.weight")
		if err != nil {
			return nil, err
		}
		if !okG || !okU {
			return nil, fmt.Errorf("load layer safetensors: missing gate_up_proj or gate_proj/up_proj weights")
		}
		if err := checkWeightBytes("gate_proj", tGate.Data, I, H, cfg.WeightDType); err != nil {
			return nil, err
		}
		if err := checkWeightBytes("up_proj", tUp.Data, I, H, cfg.WeightDType); err != nil {
			return nil, err
		}
		gateUpWeight = slices.Concat(tGate.Data, tUp.Data)
		if cfg.WeightDType == NVFP4 {
			sG, okSG, err := findTensor("mlp.gate_proj.weight_scale", "mlp.gate_proj.scale", "gate_proj.weight_scale", "gate_proj.scale")
			if err != nil {
				return nil, err
			}
			sU, okSU, err := findTensor("mlp.up_proj.weight_scale", "mlp.up_proj.scale", "up_proj.weight_scale", "up_proj.scale")
			if err != nil {
				return nil, err
			}
			if !okSG || !okSU {
				return nil, fmt.Errorf("load layer safetensors: missing gate_proj/up_proj scale tensors for NVFP4")
			}
			gateUpScale = slices.Concat(sG.Data, sU.Data)
		}
	}
	if err := checkWeightBytes("gate_up_proj", gateUpWeight, 2*I, H, cfg.WeightDType); err != nil {
		return nil, err
	}

	// Load Down projection weight and optional scale.
	tDown, okD, err := findTensor("mlp.down_proj.weight", "down_proj.weight", "w_down", "w_down_q")
	if err != nil {
		return nil, err
	}
	if !okD {
		return nil, fmt.Errorf("load layer safetensors: missing down_proj.weight")
	}
	if err := checkWeightBytes("down_proj", tDown.Data, H, I, cfg.WeightDType); err != nil {
		return nil, err
	}
	downWeight := slices.Clone(tDown.Data)
	var downScale []byte
	if cfg.WeightDType == NVFP4 {
		sD, okSD, err := findTensor("mlp.down_proj.weight_scale", "mlp.down_proj.scale", "down_proj.weight_scale", "down_proj.scale", "w_down_sf")
		if err != nil {
			return nil, err
		}
		if !okSD {
			return nil, fmt.Errorf("load layer safetensors: missing down_proj scale for NVFP4")
		}
		downScale = slices.Clone(sD.Data)
	}

	repacked := false
	if cfg.WeightDType == NVFP4 {
		switch {
		case repackForWGMMA && !alreadyRepacked:
			if qkvWeight, err = RepackNVFP4Weights(qkvWeight, qkvRows, H); err != nil {
				return nil, fmt.Errorf("repack qkv weight: %w", err)
			}
			if qkvScale, err = RepackNVFP4Scales(qkvScale, qkvRows, H); err != nil {
				return nil, fmt.Errorf("repack qkv scale: %w", err)
			}
			if oWeight, err = RepackNVFP4Weights(oWeight, H, qDim); err != nil {
				return nil, fmt.Errorf("repack o_proj weight: %w", err)
			}
			if oScale, err = RepackNVFP4Scales(oScale, H, qDim); err != nil {
				return nil, fmt.Errorf("repack o_proj scale: %w", err)
			}
			if gateUpWeight, err = RepackNVFP4Weights(gateUpWeight, 2*I, H); err != nil {
				return nil, fmt.Errorf("repack gate_up_proj weight: %w", err)
			}
			if gateUpScale, err = RepackNVFP4Scales(gateUpScale, 2*I, H); err != nil {
				return nil, fmt.Errorf("repack gate_up_proj scale: %w", err)
			}
			if downWeight, err = RepackNVFP4Weights(downWeight, H, I); err != nil {
				return nil, fmt.Errorf("repack down_proj weight: %w", err)
			}
			if downScale, err = RepackNVFP4Scales(downScale, H, I); err != nil {
				return nil, fmt.Errorf("repack down_proj scale: %w", err)
			}
			repacked = true
		case !repackForWGMMA && alreadyRepacked:
			if qkvWeight, err = UnpackNVFP4Weights(qkvWeight, qkvRows, H); err != nil {
				return nil, fmt.Errorf("unpack qkv weight: %w", err)
			}
			if qkvScale, err = UnpackNVFP4Scales(qkvScale, qkvRows, H); err != nil {
				return nil, fmt.Errorf("unpack qkv scale: %w", err)
			}
			if oWeight, err = UnpackNVFP4Weights(oWeight, H, qDim); err != nil {
				return nil, fmt.Errorf("unpack o_proj weight: %w", err)
			}
			if oScale, err = UnpackNVFP4Scales(oScale, H, qDim); err != nil {
				return nil, fmt.Errorf("unpack o_proj scale: %w", err)
			}
			if gateUpWeight, err = UnpackNVFP4Weights(gateUpWeight, 2*I, H); err != nil {
				return nil, fmt.Errorf("unpack gate_up_proj weight: %w", err)
			}
			if gateUpScale, err = UnpackNVFP4Scales(gateUpScale, 2*I, H); err != nil {
				return nil, fmt.Errorf("unpack gate_up_proj scale: %w", err)
			}
			if downWeight, err = UnpackNVFP4Weights(downWeight, H, I); err != nil {
				return nil, fmt.Errorf("unpack down_proj weight: %w", err)
			}
			if downScale, err = UnpackNVFP4Scales(downScale, H, I); err != nil {
				return nil, fmt.Errorf("unpack down_proj scale: %w", err)
			}
			repacked = false
		default:
			repacked = alreadyRepacked
		}
	}

	return &RepackedLayerWeights{
		Prefix:        prefix,
		WeightDType:   cfg.WeightDType,
		RepackedNVFP4: repacked,
		InputNorm:     inNorm,
		QKVWeight:     qkvWeight,
		QKVScale:      qkvScale,
		OWeight:       oWeight,
		OScale:        oScale,
		PostAttnNorm:  postNorm,
		GateUpWeight:  gateUpWeight,
		GateUpScale:   gateUpScale,
		DownWeight:    downWeight,
		DownScale:     downScale,
	}, nil
}

// EncodeLayerSafetensors serializes w into a valid .safetensors binary archive.
func EncodeLayerSafetensors(w *RepackedLayerWeights, cfg ArchConfig) ([]byte, error) {
	if w == nil {
		return nil, fmt.Errorf("encode layer safetensors: nil weights")
	}
	var err error
	cfg, err = cfg.norm()
	if err != nil {
		return nil, err
	}
	dtype := w.WeightDType
	if dtype == 0 {
		dtype = cfg.WeightDType
	}
	H := cfg.HiddenSize
	I := cfg.IntermediateSize
	hq := cfg.NumAttentionHeads
	hkv := cfg.NumKeyValueHeads
	headDim := cfg.HeadDim
	if headDim <= 0 {
		headDim = H / hq
	}
	qDim := hq * headDim
	qkvDim := (hq + 2*hkv) * headDim
	rowBytes := (H * dtype.Bits()) / 8
	if rowBytes > 0 && len(w.QKVWeight) > 0 && len(w.QKVWeight)%rowBytes == 0 {
		qkvDim = len(w.QKVWeight) / rowBytes
	}
	if rowBytes > 0 && len(w.OWeight) > 0 && len(w.OWeight)%rowBytes == 0 {
		qDim = len(w.OWeight) / rowBytes
	}

	prefix := strings.TrimSuffix(strings.TrimSpace(w.Prefix), ".")
	fullName := func(suffix string) string {
		if prefix == "" {
			return suffix
		}
		return prefix + "." + suffix
	}

	if len(w.InputNorm) != H {
		return nil, fmt.Errorf("encode layer safetensors: InputNorm len %d, want %d", len(w.InputNorm), H)
	}
	if len(w.PostAttnNorm) != H {
		return nil, fmt.Errorf("encode layer safetensors: PostAttnNorm len %d, want %d", len(w.PostAttnNorm), H)
	}

	var tensors []safetensors.Tensor
	tensors = append(tensors, safetensors.Tensor{
		Name:  fullName("input_layernorm.weight"),
		Dtype: safetensors.BF16,
		Shape: []int{H},
		Data:  encodeU16LE(w.InputNorm),
	})

	addWeight := func(base string, data, scale []byte, n, k int) error {
		if err := checkWeightBytes(base, data, n, k, dtype); err != nil {
			return err
		}
		switch dtype {
		case NVFP4:
			tensors = append(tensors, safetensors.Tensor{
				Name:  fullName(base + ".weight"),
				Dtype: safetensors.U8,
				Shape: []int{n, k / 2},
				Data:  data,
			})
			if len(scale) == 0 || len(scale)%n != 0 {
				return fmt.Errorf("encode layer safetensors: %s scale length %d invalid for %d rows", base, len(scale), n)
			}
			tensors = append(tensors, safetensors.Tensor{
				Name:  fullName(base + ".weight_scale"),
				Dtype: safetensors.F8E4M3,
				Shape: []int{n, len(scale) / n},
				Data:  scale,
			})
		case E4M3:
			tensors = append(tensors, safetensors.Tensor{
				Name:  fullName(base + ".weight"),
				Dtype: safetensors.F8E4M3,
				Shape: []int{n, k},
				Data:  data,
			})
		default:
			tensors = append(tensors, safetensors.Tensor{
				Name:  fullName(base + ".weight"),
				Dtype: safetensors.BF16,
				Shape: []int{n, k},
				Data:  data,
			})
		}
		return nil
	}

	if err := addWeight("self_attn.qkv_proj", w.QKVWeight, w.QKVScale, qkvDim, H); err != nil {
		return nil, err
	}
	if err := addWeight("self_attn.o_proj", w.OWeight, w.OScale, H, qDim); err != nil {
		return nil, err
	}
	tensors = append(tensors, safetensors.Tensor{
		Name:  fullName("post_attention_layernorm.weight"),
		Dtype: safetensors.BF16,
		Shape: []int{H},
		Data:  encodeU16LE(w.PostAttnNorm),
	})
	if err := addWeight("mlp.gate_up_proj", w.GateUpWeight, w.GateUpScale, 2*I, H); err != nil {
		return nil, err
	}
	if err := addWeight("mlp.down_proj", w.DownWeight, w.DownScale, H, I); err != nil {
		return nil, err
	}

	var meta map[string]string
	if w.RepackedNVFP4 {
		meta = map[string]string{"repacked_nvfp4": "true"}
	}
	var buf bytes.Buffer
	if err := safetensors.Write(&buf, tensors, meta); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// RunCompiledDenseLayer executes all steps of a compiled dense layer on the
// host CPU via [cudahostemu.Run] using weights and returns the final layer
// output [batchM * cfg.HiddenSize].
func RunCompiledDenseLayer(
	ctx context.Context,
	layer *CompiledLayer,
	weights *RepackedLayerWeights,
	hiddenIn []float32,
	kCache, vCache []float32,
	pageTable []int32,
	seqLens []int32,
) ([]float32, error) {
	if layer == nil || layer.Plan == nil {
		return nil, fmt.Errorf("run compiled dense layer: nil compiled layer or plan")
	}
	if weights == nil {
		return nil, fmt.Errorf("run compiled dense layer: nil weights")
	}
	cfg, err := layer.Config.norm()
	if err != nil {
		return nil, err
	}
	if cfg.IsMoE() {
		return nil, fmt.Errorf("run compiled dense layer: config is MoE, want dense layer")
	}

	batchM := layer.BatchM
	H := cfg.HiddenSize
	I := cfg.IntermediateSize
	hq := cfg.NumAttentionHeads
	hkv := cfg.NumKeyValueHeads
	qDim := hq * HeadDim

	if len(hiddenIn) != batchM*H {
		return nil, fmt.Errorf("run compiled dense layer: hiddenIn len %d, want %d (%d*%d)", len(hiddenIn), batchM*H, batchM, H)
	}
	if len(seqLens) != batchM {
		return nil, fmt.Errorf("run compiled dense layer: seqLens len %d, want %d", len(seqLens), batchM)
	}

	p := layer.Plan
	// Determine which NVFP4 weight buffers are consumed by Repacked WGMMA TileGEMMs.
	repackedConsumer := make(map[string]bool)
	for _, k := range p.Kernels {
		if k.Tile != nil && k.Tile.Weight == WeightNVFP4 && k.Tile.WGMMA && k.Tile.Repacked {
			for _, a := range k.Args {
				if strings.HasSuffix(a, "_q") {
					repackedConsumer[strings.TrimSuffix(a, "_q")] = true
				}
			}
		}
	}

	// Prepare Q weight slice [qDim, H] from weights.QKVWeight [qkvRows, H].
	wDType := weights.WeightDType
	if wDType == 0 {
		wDType = cfg.WeightDType
	}
	rowBytes := (H * wDType.Bits()) / 8
	qkvRows := (hq + 2*hkv) * HeadDim
	if rowBytes > 0 && len(weights.QKVWeight)%rowBytes == 0 {
		qkvRows = len(weights.QKVWeight) / rowBytes
	}

	resolveWeight := func(name string, wBytes, sBytes []byte, fullN, targetN, k int) (qBuf, sfBuf []byte, qCType string, err error) {
		switch wDType {
		case NVFP4:
			rawQ, rawSF := wBytes, sBytes
			if weights.RepackedNVFP4 {
				rawQ, err = UnpackNVFP4Weights(wBytes, fullN, k)
				if err != nil {
					return nil, nil, "", fmt.Errorf("%s: %w", name, err)
				}
				rawSF, err = UnpackNVFP4Scales(sBytes, fullN, k)
				if err != nil {
					return nil, nil, "", fmt.Errorf("%s: %w", name, err)
				}
			}
			rawQ = rawQ[:targetN*(k/2)]
			rawSF = rawSF[:targetN*(k/16)]
			if repackedConsumer[name] {
				qRep, err := RepackNVFP4Weights(rawQ, targetN, k)
				if err != nil {
					return nil, nil, "", fmt.Errorf("%s: %w", name, err)
				}
				sfRep, err := RepackNVFP4Scales(rawSF, targetN, k)
				if err != nil {
					return nil, nil, "", fmt.Errorf("%s: %w", name, err)
				}
				return qRep, sfRep, "uint32_t", nil
			}
			return rawQ, rawSF, "uint8_t", nil

		case E4M3:
			sliced := wBytes[:targetN*k]
			// BuildLayerProgram declares non-NVFP4 weight inputs as BF16 [N, K].
			bf := make([]byte, 2*len(sliced))
			for i, b := range sliced {
				binary.LittleEndian.PutUint16(bf[2*i:], BF16Bits(FromE4M3Bits(b)))
			}
			return bf, nil, "uint16_t", nil

		default:
			return wBytes[:targetN*k*2], nil, "uint16_t", nil
		}
	}

	wqBytes, wqSF, wqCType, err := resolveWeight("wq", weights.QKVWeight, weights.QKVScale, qkvRows, qDim, H)
	if err != nil {
		return nil, err
	}
	woBytes, woSF, woCType, err := resolveWeight("wo", weights.OWeight, weights.OScale, H, H, qDim)
	if err != nil {
		return nil, err
	}
	wguBytes, wguSF, wguCType, err := resolveWeight("w_gate_up", weights.GateUpWeight, weights.GateUpScale, 2*I, 2*I, H)
	if err != nil {
		return nil, err
	}
	wdBytes, wdSF, wdCType, err := resolveWeight("w_down", weights.DownWeight, weights.DownScale, H, H, I)
	if err != nil {
		return nil, err
	}

	encodeF32AsE4M3 := func(vals []float32, wantLen int) []byte {
		out := make([]byte, wantLen)
		n := min(len(vals), wantLen)
		for i := range n {
			out[i] = E4M3Bits(vals[i])
		}
		return out
	}

	var cbufs []cudahostemu.Buffer
	for _, in := range p.Inputs {
		var data []byte
		ctype := in.DType.cType()
		switch in.Name {
		case "x":
			data = encodeF32ToBF16LE(hiddenIn)
		case "attn_norm_w":
			data = encodeU16LE(weights.InputNorm)
		case "ffn_norm_w", "post_norm_w":
			data = encodeU16LE(weights.PostAttnNorm)
		case "qk_norm_w":
			unit := make([]uint16, size(in.Shape))
			for i := range unit {
				unit[i] = BF16Bits(1.0)
			}
			data = encodeU16LE(unit)
		case "wq", "wq_q":
			data, ctype = wqBytes, wqCType
		case "wq_sf":
			data = wqSF
		case "wo", "wo_q":
			data, ctype = woBytes, woCType
		case "wo_sf":
			data = woSF
		case "w_gate_up", "w_gate_up_q":
			data, ctype = wguBytes, wguCType
		case "w_gate_up_sf":
			data = wguSF
		case "w_down", "w_down_q":
			data, ctype = wdBytes, wdCType
		case "w_down_sf":
			data = wdSF
		case "wq_gs", "wo_gs", "w_gate_up_gs", "w_down_gs", "k_scale", "v_scale":
			data = encodeF32LE([]float32{1.0})
		case "k_cache":
			data = encodeF32AsE4M3(kCache, size(in.Shape))
		case "v_cache":
			data = encodeF32AsE4M3(vCache, size(in.Shape))
		case "page_table":
			pt := make([]int32, size(in.Shape))
			copy(pt, pageTable)
			data = encodeI32LE(pt)
		case "seqlens":
			data = encodeI32LE(seqLens)
		default:
			return nil, fmt.Errorf("run compiled dense layer: unexpected plan input %q", in.Name)
		}
		cbufs = append(cbufs, cudahostemu.Buffer{
			Name:  in.Name,
			CType: ctype,
			Data:  data,
		})
	}

	for _, tmp := range append(slices.Clone(p.Temps), p.Output) {
		nbytes := (size(tmp.Shape) * tmp.DType.Bits()) / 8
		cbufs = append(cbufs, cudahostemu.Buffer{
			Name:  tmp.Name,
			CType: tmp.DType.cType(),
			Data:  make([]byte, max(nbytes, 4)),
		})
	}

	var src strings.Builder
	src.WriteString(Prelude)
	var (
		sawNVFP4Helpers   bool
		sawSwizzleHelpers bool
		sawBWFP4Helpers   bool
	)
	dedupHelpers := func(s string) string {
		if strings.Contains(s, nvfp4WGMMAHelpers) {
			if sawNVFP4Helpers {
				s = strings.Replace(s, nvfp4WGMMAHelpers, "", 1)
			}
			sawNVFP4Helpers = true
		}
		if strings.Contains(s, wgmmaSwizzleHelpers) {
			if sawSwizzleHelpers {
				s = strings.Replace(s, wgmmaSwizzleHelpers, "", 1)
			}
			sawSwizzleHelpers = true
		}
		if strings.Contains(s, blackwellFP4Helpers) {
			if sawBWFP4Helpers {
				s = strings.Replace(s, blackwellFP4Helpers, "", 1)
			}
			sawBWFP4Helpers = true
		}
		return s
	}
	var launches []string
	for _, k := range p.Kernels {
		switch {
		case k.GEMM != nil:
			s, err := k.GEMM.CUDA()
			if err != nil {
				return nil, err
			}
			src.WriteString(dedupHelpers(s))
			grid, block, _ := k.GEMM.Grid()
			launches = append(launches, fmt.Sprintf("eg_emu_launch(dim3(%d, %d), dim3(%d), 0, %s<%d>, %s)",
				grid[0], grid[1], block[0], k.GEMM.Name, k.M, strings.Join(k.Args, ", ")))
		case k.Tile != nil:
			s, err := k.Tile.CUDA()
			if err != nil {
				return nil, err
			}
			src.WriteString(dedupHelpers(s))
			if k.Tile.Grouped {
				grid, block := k.Tile.Grid(k.MaxRows, k.Experts)
				launches = append(launches, fmt.Sprintf("eg_emu_launch(dim3(%d, %d, %d), dim3(%d), %d, %s, %d, %s)",
					grid[0], grid[1], grid[2], block[0], k.Tile.SharedBytes(), k.Tile.Name, k.M, strings.Join(k.Args, ", ")))
			} else {
				grid, block := k.Tile.Grid(k.M, 1)
				launches = append(launches, fmt.Sprintf("eg_emu_launch(dim3(%d, %d), dim3(%d), %d, %s, %d, %s)",
					grid[0], grid[1], block[0], k.Tile.SharedBytes(), k.Tile.Name, k.M, strings.Join(k.Args, ", ")))
			}
		case k.Attention != nil:
			s, err := k.Attention.CUDA()
			if err != nil {
				return nil, err
			}
			l, err := k.Attention.Launcher()
			if err != nil {
				return nil, err
			}
			src.WriteString(s + "\n" + l)
			args := append([]string(nil), k.Args[:4]...)
			args = append(args, fmt.Sprint(k.MaxPages))
			args = append(args, k.Args[4:]...)
			launches = append(launches, fmt.Sprintf("eg_emu_sms = 16; eg_emu_blocks_per_sm = 1; %s_launch(%d, %d, %s, nullptr)",
				k.Attention.Name, k.Batch, k.MaxKeys, strings.Join(args, ", ")))
		case k.Row != nil:
			s, err := k.Row.CUDA()
			if err != nil {
				return nil, err
			}
			src.WriteString(s)
			if k.Row.Cluster {
				l, err := k.Row.Launcher()
				if err != nil {
					return nil, err
				}
				src.WriteString("\n" + l)
				launches = append(launches, fmt.Sprintf("%s_launch(%d, %s, nullptr)",
					k.Row.Name, k.Rows, strings.Join(k.Args, ", ")))
			} else {
				grid, block := k.Row.Grid(k.Rows)
				launches = append(launches, fmt.Sprintf("eg_emu_launch(dim3(%d), dim3(%d), 0, %s, %d, %s)",
					grid[0], block[0], k.Row.Name, k.Rows, strings.Join(k.Args, ", ")))
			}
		default:
			return nil, fmt.Errorf("run compiled dense layer: unsupported kernel step in plan")
		}
	}

	tmpDir, err := os.MkdirTemp("", "hopper-safetensors-layer-*")
	if err != nil {
		return nil, fmt.Errorf("run compiled dense layer: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	res, err := cudahostemu.Run(ctx, tmpDir, cudahostemu.Program{
		Source:  src.String(),
		Launch:  strings.Join(launches, ";\n    "),
		Buffers: cbufs,
		Flags:   strings.Fields(os.Getenv("EG_EMU_FLAGS")),
	})
	if err != nil {
		return nil, fmt.Errorf("run compiled dense layer: %w", err)
	}

	outBytes := res[p.Output.Name]
	return decodeOutputF32(p.Output.DType, outBytes), nil
}

func decodeNormBF16(t safetensors.Tensor, wantLen int) ([]uint16, error) {
	if len(t.Shape) != 1 || t.Shape[0] != wantLen {
		return nil, fmt.Errorf("tensor %q: shape %v, want [%d]", t.Name, t.Shape, wantLen)
	}
	out := make([]uint16, wantLen)
	switch t.Dtype {
	case safetensors.BF16, safetensors.U16:
		for i := range out {
			out[i] = binary.LittleEndian.Uint16(t.Data[2*i:])
		}
	case safetensors.F32:
		for i := range out {
			f := math.Float32frombits(binary.LittleEndian.Uint32(t.Data[4*i:]))
			out[i] = BF16Bits(f)
		}
	default:
		return nil, fmt.Errorf("tensor %q: unsupported norm dtype %q", t.Name, t.Dtype)
	}
	return out, nil
}

func checkWeightBytes(name string, data []byte, n, k int, dtype DType) error {
	want := (n * k * dtype.Bits()) / 8
	if len(data) != want {
		return fmt.Errorf("%s: have %d bytes, want %d for %v [%d, %d]", name, len(data), want, dtype, n, k)
	}
	return nil
}

func encodeU16LE(v []uint16) []byte {
	b := make([]byte, 2*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint16(b[2*i:], x)
	}
	return b
}

func encodeF32ToBF16LE(v []float32) []byte {
	b := make([]byte, 2*len(v))
	for i, f := range v {
		binary.LittleEndian.PutUint16(b[2*i:], BF16Bits(f))
	}
	return b
}

func encodeF32LE(v []float32) []byte {
	b := make([]byte, 4*len(v))
	for i, f := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(f))
	}
	return b
}

func encodeI32LE(v []int32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], uint32(x))
	}
	return b
}

func decodeOutputF32(dt DType, b []byte) []float32 {
	switch dt {
	case BF16:
		out := make([]float32, len(b)/2)
		for i := range out {
			out[i] = FromBF16Bits(binary.LittleEndian.Uint16(b[2*i:]))
		}
		return out
	case F32:
		out := make([]float32, len(b)/4)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
		}
		return out
	case E4M3:
		out := make([]float32, len(b))
		for i, x := range b {
			out[i] = FromE4M3Bits(x)
		}
		return out
	default:
		return nil
	}
}
