package model

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
	"github.com/tmc/lib/nvidia/x/egraph/bitopt"
	"github.com/tmc/lib/nvidia/x/egraph/tilebridge"
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

// LayerWeightMap is an alias for [LayerSafetensorsSpec].
type LayerWeightMap = LayerSafetensorsSpec

// PackedLayerWeights holds extracted and hardware-repacked weight and scale
// buffers for a single transformer decoder layer loaded from a safetensors archive.
type PackedLayerWeights struct {
	Prefix        string
	WeightDType   DType
	RepackedNVFP4 bool
	InputNorm     []uint16 // BF16 [HiddenSize]
	QKVWeight     []byte   // BF16, E4M3, or repacked NVFP4 [QKV_N, HiddenSize]
	QKVScale      []byte   // optional E4M3 block scales for NVFP4
	OWeight       []byte   // [HiddenSize, NumAttentionHeads * HeadDim]
	OScale        []byte
	PostAttnNorm  []uint16 // BF16 [HiddenSize]
	GateUpWeight  []byte   // [2 * IntermediateSize, HiddenSize]
	GateUpScale   []byte
	DownWeight    []byte // [HiddenSize, IntermediateSize]
	DownScale     []byte
}

// RepackedLayerWeights is an alias for [PackedLayerWeights].
type RepackedLayerWeights = PackedLayerWeights

var nvfp4Prog2 = bitopt.NVFP4Program2Word()

func packDequant2Words(codes0, codes1 uint32) [2]uint32 {
	var cs [16]uint8
	for i := range 8 {
		cs[i] = uint8((codes0 >> (4 * i)) & 15)
		cs[8+i] = uint8((codes1 >> (4 * i)) & 15)
	}
	w := nvfp4Prog2.PackWords(cs[:])
	return [2]uint32{w[0], w[1]}
}

func unpackDequant2Words(q [2]uint32) (codes0, codes1 uint32) {
	cs := nvfp4Prog2.UnpackWords(q[:])
	for i := range 8 {
		codes0 |= uint32(cs[i]&15) << (4 * i)
		codes1 |= uint32(cs[8+i]&15) << (4 * i)
	}
	return codes0, codes1
}

func wgmmaTileRow(n, nt, r int, glu bool) int {
	if !glu {
		return nt*128 + r
	}
	col := nt*64 + (r/16)*8 + r%8
	if (r % 16) < 8 {
		return col
	}
	return n/2 + col
}

func repackNVFP4Dequant2Word(packed, scales []uint8, n, k int, glu bool) (wq []uint32, ws []uint8, err error) {
	if n <= 0 || n%128 != 0 || k <= 0 || k%64 != 0 {
		return nil, nil, fmt.Errorf("repack nvfp4 dequant2word: shape [%d, %d] must be a multiple of [128, 64]", n, k)
	}
	if len(packed) != n*k/2 || len(scales) != n*k/16 {
		return nil, nil, fmt.Errorf("repack nvfp4 dequant2word: [%d, %d] needs %d packed bytes and %d scales, have %d and %d",
			n, k, n*k/2, n*k/16, len(packed), len(scales))
	}
	ntiles := n / 128
	kblocks := k / 64
	wq = make([]uint32, ntiles*kblocks*256*4)
	ws = make([]uint8, ntiles*kblocks*256*8)
	codeAt := func(row, col int) uint8 {
		b := packed[row*(k/2)+col/2]
		return (b >> (4 * (col & 1))) & 15
	}
	koff := [4]int{0, 0, 8, 8}
	for nt := range ntiles {
		for kb := range kblocks {
			for th := range 256 {
				cw, warp, lane := th/128, (th%128)/32, th%32
				g, tq := lane/4, lane%4
				rg := 64*cw + 16*warp + g
				r0 := wgmmaTileRow(n, nt, rg, glu)
				r1 := wgmmaTileRow(n, nt, rg+8, glu)
				trows := [4]int{r0, r1, r0, r1}
				wIdx := ((nt*kblocks+kb)*256 + th) * 4
				sIdx := ((nt*kblocks+kb)*256 + th) * 8
				for p := range 2 {
					var codes [16]uint8
					for sub := range 2 {
						ks := 2*p + sub
						for j := range 4 {
							row := trows[j]
							col := kb*64 + ks*16 + koff[j] + 2*tq
							codes[8*sub+2*j+0] = codeAt(row, col)
							codes[8*sub+2*j+1] = codeAt(row, col+1)
						}
						grp := kb*4 + ks
						ws[sIdx+2*ks] = scales[r0*(k/16)+grp]
						ws[sIdx+2*ks+1] = scales[r1*(k/16)+grp]
					}
					pair := nvfp4Prog2.PackWords(codes[:])
					wq[wIdx+2*p+0] = pair[0]
					wq[wIdx+2*p+1] = pair[1]
				}
			}
		}
	}
	return wq, ws, nil
}

func unpackNVFP4Dequant2Word(wq []uint32, ws []uint8, n, k int, glu bool) (packed, scales []uint8, err error) {
	if n <= 0 || n%128 != 0 || k <= 0 || k%64 != 0 {
		return nil, nil, fmt.Errorf("unpack nvfp4 dequant2word: shape [%d, %d] must be a multiple of [128, 64]", n, k)
	}
	ntiles := n / 128
	kblocks := k / 64
	if len(wq) != ntiles*kblocks*256*4 || len(ws) != ntiles*kblocks*256*8 {
		return nil, nil, fmt.Errorf("unpack nvfp4 dequant2word: have %d words and %d scale bytes, want %d and %d",
			len(wq), len(ws), ntiles*kblocks*256*4, ntiles*kblocks*256*8)
	}
	packed = make([]uint8, n*k/2)
	scales = make([]uint8, n*k/16)
	setCode := func(row, col int, c uint8) {
		idx := row*(k/2) + col/2
		shift := 4 * (col & 1)
		packed[idx] = (packed[idx] &^ (0xf << shift)) | ((c & 15) << shift)
	}
	koff := [4]int{0, 0, 8, 8}
	for nt := range ntiles {
		for kb := range kblocks {
			for th := range 256 {
				cw, warp, lane := th/128, (th%128)/32, th%32
				g, tq := lane/4, lane%4
				rg := 64*cw + 16*warp + g
				r0 := wgmmaTileRow(n, nt, rg, glu)
				r1 := wgmmaTileRow(n, nt, rg+8, glu)
				trows := [4]int{r0, r1, r0, r1}
				wIdx := ((nt*kblocks+kb)*256 + th) * 4
				sIdx := ((nt*kblocks+kb)*256 + th) * 8
				for p := range 2 {
					codes := nvfp4Prog2.UnpackWords(wq[wIdx+2*p : wIdx+2*p+2])
					for sub := range 2 {
						ks := 2*p + sub
						for j := range 4 {
							row := trows[j]
							col := kb*64 + ks*16 + koff[j] + 2*tq
							setCode(row, col, codes[8*sub+2*j+0])
							setCode(row, col+1, codes[8*sub+2*j+1])
						}
						grp := kb*4 + ks
						if tq == 0 {
							scales[r0*(k/16)+grp] = ws[sIdx+2*ks]
							scales[r1*(k/16)+grp] = ws[sIdx+2*ks+1]
						}
					}
				}
			}
		}
	}
	return packed, scales, nil
}

// RepackNVFP4Weights losslessly repacks an [n, k] NVFP4 weight matrix (packed
// [n, k/2] E2M1 bytes) into Hopper WGMMA 2-word bitopt dequant order.
func RepackNVFP4Weights(packed []byte, n, k int) ([]byte, error) {
	if n <= 0 || k <= 0 || k%16 != 0 {
		return nil, fmt.Errorf("repack nvfp4 weights: invalid shape [%d, %d]", n, k)
	}
	if len(packed) != n*k/2 {
		return nil, fmt.Errorf("repack nvfp4 weights: shape [%d, %d] needs %d bytes, have %d", n, k, n*k/2, len(packed))
	}
	if n%128 == 0 && k%64 == 0 {
		dummyScales := make([]uint8, n*k/16)
		wq, _, err := repackNVFP4Dequant2Word(packed, dummyScales, n, k, false)
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
		q := packDequant2Words(c0, c1)
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
		packed, _, err := unpackNVFP4Dequant2Word(wq, dummyWS, n, k, false)
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
		c0, c1 := unpackDequant2Words([2]uint32{q0, q1})
		binary.LittleEndian.PutUint32(out[i:i+4], c0)
		binary.LittleEndian.PutUint32(out[i+4:i+8], c1)
	}
	return out, nil
}

// RepackNVFP4Scales losslessly repacks an [n, k/16] E4M3 block-scale matrix
// into Hopper WGMMA per-thread scale layout.
func RepackNVFP4Scales(scales []byte, n, k int) ([]byte, error) {
	if n <= 0 || k <= 0 || k%16 != 0 {
		return nil, fmt.Errorf("repack nvfp4 scales: invalid shape [%d, %d]", n, k)
	}
	if len(scales) != n*k/16 {
		return nil, fmt.Errorf("repack nvfp4 scales: shape [%d, %d] needs %d scale bytes, have %d", n, k, n*k/16, len(scales))
	}
	if n%128 == 0 && k%64 == 0 {
		dummyPacked := make([]uint8, n*k/2)
		_, ws, err := repackNVFP4Dequant2Word(dummyPacked, scales, n, k, false)
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
			_, scales, err := unpackNVFP4Dequant2Word(dummyWQ, repacked, n, k, false)
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

var e2m1Grid = [8]float32{0, 0.5, 1, 1.5, 2, 3, 4, 6}

// QuantizeTensorNVFP4 quantizes an [n, k] float32 matrix into packed E2M1 bytes
// [n, k/2] and E4M3 block scales [n, k/16].
func QuantizeTensorNVFP4(vals []float32, n, k int) (packed, scales []byte, err error) {
	if n <= 0 || k <= 0 || k%16 != 0 || len(vals) != n*k {
		return nil, nil, fmt.Errorf("quantize nvfp4: invalid shape [%d, %d] for %d values", n, k, len(vals))
	}
	packed = make([]byte, n*k/2)
	scales = make([]byte, n*k/16)
	for blk := 0; blk < len(scales); blk++ {
		base := blk * 16
		var maxAbs float32
		for i := range 16 {
			if a := float32(math.Abs(float64(vals[base+i]))); a > maxAbs {
				maxAbs = a
			}
		}
		sb := tilebridge.E4M3Bits(maxAbs / 6.0)
		if sb == 0 {
			sb = tilebridge.E4M3Bits(1.0)
		}
		scales[blk] = sb
		sc := tilebridge.FromE4M3Bits(sb)
		for i := range 16 {
			v := vals[base+i] / sc
			var sign uint8
			if v < 0 {
				sign = 8
				v = -v
			}
			bestIdx := uint8(0)
			bestDiff := float32(math.Inf(1))
			for idx, g := range e2m1Grid {
				if d := float32(math.Abs(float64(v - g))); d < bestDiff {
					bestDiff = d
					bestIdx = uint8(idx)
				}
			}
			code := sign | bestIdx
			pos := base + i
			packed[pos/2] |= code << (4 * (pos & 1))
		}
	}
	return packed, scales, nil
}

// DequantizeTensorNVFP4 decodes packed [n, k/2] E2M1 bytes and [n, k/16] E4M3
// block scales into [n * k] float32 values.
func DequantizeTensorNVFP4(packed, scales []byte, n, k int, globalScale float32) ([]float32, error) {
	if n <= 0 || k <= 0 || k%16 != 0 || len(packed) != n*k/2 || len(scales) != n*k/16 {
		return nil, fmt.Errorf("dequantize nvfp4: invalid shape [%d, %d]", n, k)
	}
	if globalScale == 0 {
		globalScale = 1.0
	}
	out := make([]float32, n*k)
	for i := range out {
		code := (packed[i/2] >> (4 * (i & 1))) & 15
		v := e2m1Grid[code&7]
		if code&8 != 0 {
			v = -v
		}
		out[i] = v * (tilebridge.FromE4M3Bits(scales[i/16]) * globalScale)
	}
	return out, nil
}

// LoadLayerSafetensors parses stData using nvidia/x/safetensors, extracts the
// weights and scales of layerPrefix (supporting either fused qkv_proj /
// gate_up_proj or separate q_proj/k_proj/v_proj and gate_proj/up_proj tensors),
// and optionally repacks NVFP4 weights and E4M3 block scales for Hopper WGMMA.
func LoadLayerSafetensors(stData []byte, cfg ArchConfig, layerPrefix string, repackForWGMMA bool) (*PackedLayerWeights, error) {
	var err error
	cfg, err = ValidateArchConfig(cfg)
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

	return &PackedLayerWeights{
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
func EncodeLayerSafetensors(w *PackedLayerWeights, cfg ArchConfig) ([]byte, error) {
	if w == nil {
		return nil, fmt.Errorf("encode layer safetensors: nil weights")
	}
	var err error
	cfg, err = ValidateArchConfig(cfg)
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
		Data:  EncodeU16LE(w.InputNorm),
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
		Data:  EncodeU16LE(w.PostAttnNorm),
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

// PlanTensorSpec describes a tensor input, temporary, or output in a compiled plan.
type PlanTensorSpec struct {
	Name  string
	DType DType
	Shape []int
}

func tensorSize(shape []int) int {
	n := 1
	for _, s := range shape {
		n *= s
	}
	return n
}

// BuildDenseLayerBuffers resolves and repacks layer weights and encodes all
// input, temporary, and output [cudahostemu.Buffer] arguments for host
// emulation of a compiled dense layer.
func BuildDenseLayerBuffers(
	cfg ArchConfig,
	batchM int,
	weights *PackedLayerWeights,
	repackedConsumers map[string]bool,
	inputs []PlanTensorSpec,
	tempsAndOut []PlanTensorSpec,
	hiddenIn, kCache, vCache []float32,
	pageTable, seqLens []int32,
) ([]cudahostemu.Buffer, error) {
	H := cfg.HiddenSize
	I := cfg.IntermediateSize
	hq := cfg.NumAttentionHeads
	hkv := cfg.NumKeyValueHeads
	qDim := hq * 64

	if len(hiddenIn) != batchM*H {
		return nil, fmt.Errorf("run compiled dense layer: hiddenIn len %d, want %d (%d*%d)", len(hiddenIn), batchM*H, batchM, H)
	}
	if len(seqLens) != batchM {
		return nil, fmt.Errorf("run compiled dense layer: seqLens len %d, want %d", len(seqLens), batchM)
	}

	wDType := weights.WeightDType
	if wDType == 0 {
		wDType = cfg.WeightDType
	}
	rowBytes := (H * wDType.Bits()) / 8
	qkvRows := (hq + 2*hkv) * 64
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
			if repackedConsumers[name] {
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
			bf := make([]byte, 2*len(sliced))
			for i, b := range sliced {
				binary.LittleEndian.PutUint16(bf[2*i:], tilebridge.BF16Bits(tilebridge.FromE4M3Bits(b)))
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
			out[i] = tilebridge.E4M3Bits(vals[i])
		}
		return out
	}

	var cbufs []cudahostemu.Buffer
	for _, in := range inputs {
		var data []byte
		ctype := in.DType.CType()
		switch in.Name {
		case "x":
			data = EncodeF32ToBF16LE(hiddenIn)
		case "attn_norm_w":
			data = EncodeU16LE(weights.InputNorm)
		case "ffn_norm_w", "post_norm_w":
			data = EncodeU16LE(weights.PostAttnNorm)
		case "qk_norm_w":
			unit := make([]uint16, tensorSize(in.Shape))
			for i := range unit {
				unit[i] = tilebridge.BF16Bits(1.0)
			}
			data = EncodeU16LE(unit)
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
			data = EncodeF32LE([]float32{1.0})
		case "k_cache":
			data = encodeF32AsE4M3(kCache, tensorSize(in.Shape))
		case "v_cache":
			data = encodeF32AsE4M3(vCache, tensorSize(in.Shape))
		case "page_table":
			pt := make([]int32, tensorSize(in.Shape))
			copy(pt, pageTable)
			data = EncodeI32LE(pt)
		case "seqlens":
			data = EncodeI32LE(seqLens)
		default:
			return nil, fmt.Errorf("run compiled dense layer: unexpected plan input %q", in.Name)
		}
		cbufs = append(cbufs, cudahostemu.Buffer{
			Name:  in.Name,
			CType: ctype,
			Data:  data,
		})
	}

	for _, tmp := range tempsAndOut {
		nbytes := (tensorSize(tmp.Shape) * tmp.DType.Bits()) / 8
		cbufs = append(cbufs, cudahostemu.Buffer{
			Name:  tmp.Name,
			CType: tmp.DType.CType(),
			Data:  make([]byte, max(nbytes, 4)),
		})
	}
	return cbufs, nil
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
			out[i] = tilebridge.BF16Bits(f)
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

// EncodeU16LE encodes a uint16 slice as little-endian bytes.
func EncodeU16LE(v []uint16) []byte {
	b := make([]byte, 2*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint16(b[2*i:], x)
	}
	return b
}

// EncodeF32ToBF16LE encodes a float32 slice as little-endian bfloat16 bytes.
func EncodeF32ToBF16LE(v []float32) []byte {
	b := make([]byte, 2*len(v))
	for i, f := range v {
		binary.LittleEndian.PutUint16(b[2*i:], tilebridge.BF16Bits(f))
	}
	return b
}

// EncodeF32LE encodes a float32 slice as little-endian bytes.
func EncodeF32LE(v []float32) []byte {
	b := make([]byte, 4*len(v))
	for i, f := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(f))
	}
	return b
}

// EncodeI32LE encodes an int32 slice as little-endian bytes.
func EncodeI32LE(v []int32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], uint32(x))
	}
	return b
}

// DecodeOutputF32 decodes raw output bytes of dtype dt into float32 values.
func DecodeOutputF32(dt DType, b []byte) []float32 {
	switch dt {
	case BF16:
		out := make([]float32, len(b)/2)
		for i := range out {
			out[i] = tilebridge.FromBF16Bits(binary.LittleEndian.Uint16(b[2*i:]))
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
			out[i] = tilebridge.FromE4M3Bits(x)
		}
		return out
	default:
		return nil
	}
}
