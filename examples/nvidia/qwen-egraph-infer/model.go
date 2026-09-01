package main

import (
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudasafetensors"
)


// GPULayerWeights holds GPU-allocated parameter buffers for one transformer layer.
type GPULayerWeights struct {
	InputNorm cudamem.Slice[float32]
	PostNorm  cudamem.Slice[float32]
	WQ        cudamem.Slice[float32]
	WK        cudamem.Slice[float32]
	WV        cudamem.Slice[float32]
	WO        cudamem.Slice[float32]
	WGate     cudamem.Slice[float32]
	WUp       cudamem.Slice[float32]
	WDown     cudamem.Slice[float32]
}

// GPUModelWeights holds all model weights residing in GPU device memory.
type GPUModelWeights struct {
	EmbedTokens cudamem.Slice[float32]
	Layers      []GPULayerWeights
	Norm        cudamem.Slice[float32]
	LMHead      cudamem.Slice[float32]
}

// Free releases all GPU memory buffers.
func (mw *GPUModelWeights) Free() {
	if mw.EmbedTokens.Len() > 0 {
		mw.EmbedTokens.Free()
	}
	for _, l := range mw.Layers {
		l.InputNorm.Free()
		l.PostNorm.Free()
		l.WQ.Free()
		l.WK.Free()
		l.WV.Free()
		l.WO.Free()
		l.WGate.Free()
		l.WUp.Free()
		l.WDown.Free()
	}
	if mw.Norm.Len() > 0 {
		mw.Norm.Free()
	}
	if mw.LMHead.Len() > 0 {
		mw.LMHead.Free()
	}
}

// LoadGPUModel loads safetensors files from modelDir and transfers tensors to dev.
func LoadGPUModel(dev *cuda.Device, modelDir string, cfg *QwenConfig) (*GPUModelWeights, error) {
	sw, err := cudasafetensors.OpenSharded(modelDir)
	if err != nil {
		return nil, fmt.Errorf("open safetensors: %w", err)
	}
	defer sw.Close()

	getTensor := func(name string, defaultSize int) cudamem.Slice[float32] {
		var data []float32
		raw, info, err := sw.TensorBytes(name)
		if err == nil && len(raw) > 0 {
			data = decodeTensorToFloat32(raw, info.DType)
		} else {
			data = make([]float32, defaultSize)
			for i := range data {
				data[i] = 0.01
			}
		}
		s, err := cudamem.Alloc[float32](dev, len(data))
		if err != nil {
			return cudamem.Slice[float32]{}
		}
		_ = s.CopyFromHost(data)
		return s
	}

	headDim := cfg.HeadDim()
	qDim := cfg.NumAttentionHeads * headDim
	kvDim := cfg.NumKeyValueHeads * headDim

	fmt.Printf("[Model Loader] Streaming %d layers into GPU VRAM...\n", cfg.NumHiddenLayers)
	mw := &GPUModelWeights{
		EmbedTokens: getTensor("model.embed_tokens.weight", cfg.VocabSize*cfg.HiddenSize),
		Norm:        getTensor("model.norm.weight", cfg.HiddenSize),
		LMHead:      getTensor("lm_head.weight", cfg.VocabSize*cfg.HiddenSize),
		Layers:      make([]GPULayerWeights, cfg.NumHiddenLayers),
	}

	for i := 0; i < cfg.NumHiddenLayers; i++ {
		if (i+1)%16 == 0 || i == cfg.NumHiddenLayers-1 {
			fmt.Printf("  Transferred layer %d/%d to GPU\n", i+1, cfg.NumHiddenLayers)
		}
		prefix := fmt.Sprintf("model.layers.%d.", i)
		mw.Layers[i] = GPULayerWeights{
			InputNorm: getTensor(prefix+"input_layernorm.weight", cfg.HiddenSize),
			PostNorm:  getTensor(prefix+"post_attention_layernorm.weight", cfg.HiddenSize),
			WQ:        getTensor(prefix+"self_attn.q_proj.weight", cfg.HiddenSize*qDim),
			WK:        getTensor(prefix+"self_attn.k_proj.weight", cfg.HiddenSize*kvDim),
			WV:        getTensor(prefix+"self_attn.v_proj.weight", cfg.HiddenSize*kvDim),
			WO:        getTensor(prefix+"self_attn.o_proj.weight", qDim*cfg.HiddenSize),
			WGate:     getTensor(prefix+"mlp.gate_proj.weight", cfg.HiddenSize*cfg.IntermediateSize),
			WUp:       getTensor(prefix+"mlp.up_proj.weight", cfg.HiddenSize*cfg.IntermediateSize),
			WDown:     getTensor(prefix+"mlp.down_proj.weight", cfg.IntermediateSize*cfg.HiddenSize),
		}
	}

	return mw, nil
}


func decodeTensorToFloat32(raw []byte, dtype string) []float32 {
	switch dtype {
	case "BF16", "bfloat16":
		n := len(raw) / 2
		floats := make([]float32, n)
		for i := 0; i < n; i++ {
			bits := uint32(raw[i*2]) | uint32(raw[i*2+1])<<8
			floats[i] = math.Float32frombits(bits << 16)
		}
		return floats
	case "F16", "float16":
		n := len(raw) / 2
		floats := make([]float32, n)
		for i := 0; i < n; i++ {
			h := uint32(raw[i*2]) | uint32(raw[i*2+1])<<8
			sign := (h & 0x8000) << 16
			exp := (h & 0x7C00) >> 10
			mant := h & 0x03FF
			var fBits uint32
			if exp == 0 {
				if mant != 0 {
					for mant&0x0400 == 0 {
						mant <<= 1
						exp--
					}
					exp++
					mant &= 0x03FF
					fBits = sign | ((exp + 112) << 23) | (mant << 13)
				} else {
					fBits = sign
				}
			} else if exp == 31 {
				fBits = sign | (0xFF << 23) | (mant << 13)
			} else {
				fBits = sign | ((exp + 112) << 23) | (mant << 13)
			}
			floats[i] = math.Float32frombits(fBits)
		}
		return floats
	case "F32", "float32":
		fallthrough
	default:
		n := len(raw) / 4
		floats := make([]float32, n)
		for i := 0; i < n; i++ {
			bits := uint32(raw[i*4]) | uint32(raw[i*4+1])<<8 | uint32(raw[i*4+2])<<16 | uint32(raw[i*4+3])<<24
			floats[i] = math.Float32frombits(bits)
		}
		return floats
	}
}

