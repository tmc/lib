package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/tmc/lib/nvidia/x/cudasafetensors"
)

// ModelConfig holds the architecture parameters for a LLaMA model.
type ModelConfig struct {
	HiddenSize         int     `json:"hidden_size"`
	IntermediateSize   int     `json:"intermediate_size"`
	NumHiddenLayers    int     `json:"num_hidden_layers"`
	NumAttentionHeads  int     `json:"num_attention_heads"`
	NumKeyValueHeads   int     `json:"num_key_value_heads"`
	VocabSize          int     `json:"vocab_size"`
	RmsNormEps         float32 `json:"rms_norm_eps"`
	RopeTheta          float32 `json:"rope_theta"`
	MaxSeqLen          int     `json:"max_position_embeddings"`
}

// DefaultModelConfig provides a compact configuration for testing.
func DefaultModelConfig() ModelConfig {
	return ModelConfig{
		HiddenSize:        64,
		IntermediateSize:  128,
		NumHiddenLayers:   2,
		NumAttentionHeads: 4,
		NumKeyValueHeads:  2,
		VocabSize:         256,
		RmsNormEps:        1e-5,
		RopeTheta:         10000.0,
		MaxSeqLen:         64,
	}
}

// HeadDim returns the dimension per attention head.
func (cfg *ModelConfig) HeadDim() int {
	if cfg.NumAttentionHeads == 0 {
		return 16
	}
	return cfg.HiddenSize / cfg.NumAttentionHeads
}

// LayerWeights holds the float32 parameter buffers for one transformer layer.
type LayerWeights struct {
	InputNorm []float32
	PostNorm  []float32
	WQ        []float32
	WK        []float32
	WV        []float32
	WO        []float32
	WGate     []float32
	WUp       []float32
	WDown     []float32
}

// ModelWeights holds all parameters for the LLaMA model.
type ModelWeights struct {
	EmbedTokens []float32
	Layers      []LayerWeights
	Norm        []float32
	LMHead      []float32
}

// KVCacheState holds the multi-layer KV memory cache for autoregressive inference.
type KVCacheState struct {
	K [][][]float32 // [Layer][SeqLen][NumKVHeads * HeadDim]
	V [][][]float32 // [Layer][SeqLen][NumKVHeads * HeadDim]
}

// NewKVCacheState allocates memory for a new KV cache.
func NewKVCacheState(numLayers, maxSeqLen, kvDim int) *KVCacheState {
	k := make([][][]float32, numLayers)
	v := make([][][]float32, numLayers)
	for i := 0; i < numLayers; i++ {
		k[i] = make([][]float32, maxSeqLen)
		v[i] = make([][]float32, maxSeqLen)
		for t := 0; t < maxSeqLen; t++ {
			k[i][t] = make([]float32, kvDim)
			v[i][t] = make([]float32, kvDim)
		}
	}
	return &KVCacheState{K: k, V: v}
}

// LoadModel loads model configuration and weights from a directory containing config.json and .safetensors.
func LoadModel(dir string) (*ModelWeights, *ModelConfig, error) {
	configPath := filepath.Join(dir, "config.json")
	cfgBytes, err := os.ReadFile(configPath)
	if err != nil {
		return nil, nil, fmt.Errorf("read config.json: %w", err)
	}

	var cfg ModelConfig
	if err := json.Unmarshal(cfgBytes, &cfg); err != nil {
		return nil, nil, fmt.Errorf("parse config.json: %w", err)
	}
	if cfg.RmsNormEps == 0 {
		cfg.RmsNormEps = 1e-5
	}
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 10000.0
	}
	if cfg.MaxSeqLen == 0 {
		cfg.MaxSeqLen = 2048
	}

	sw, err := cudasafetensors.OpenSharded(dir)
	if err != nil {
		return nil, nil, fmt.Errorf("open safetensors: %w", err)
	}
	defer sw.Close()

	getTensor := func(name string) ([]float32, error) {
		raw, _, err := sw.TensorBytes(name)
		if err != nil {
			return nil, err
		}
		// Convert byte slice to float32 slice
		floats := make([]float32, len(raw)/4)
		for i := range floats {
			bits := uint32(raw[i*4]) | uint32(raw[i*4+1])<<8 | uint32(raw[i*4+2])<<16 | uint32(raw[i*4+3])<<24
			floats[i] = math.Float32frombits(bits)
		}
		return floats, nil
	}

	embed, err := getTensor("model.embed_tokens.weight")
	if err != nil {
		return nil, nil, fmt.Errorf("load embed_tokens: %w", err)
	}

	norm, err := getTensor("model.norm.weight")
	if err != nil {
		return nil, nil, fmt.Errorf("load model.norm: %w", err)
	}

	lmHead, err := getTensor("lm_head.weight")
	if err != nil {
		// LM head might be tied to embed_tokens
		lmHead = embed
	}

	layers := make([]LayerWeights, cfg.NumHiddenLayers)
	for i := 0; i < cfg.NumHiddenLayers; i++ {
		prefix := fmt.Sprintf("model.layers.%d.", i)
		inNorm, _ := getTensor(prefix + "input_layernorm.weight")
		postNorm, _ := getTensor(prefix + "post_attention_layernorm.weight")
		wq, _ := getTensor(prefix + "self_attn.q_proj.weight")
		wk, _ := getTensor(prefix + "self_attn.k_proj.weight")
		wv, _ := getTensor(prefix + "self_attn.v_proj.weight")
		wo, _ := getTensor(prefix + "self_attn.o_proj.weight")
		wgate, _ := getTensor(prefix + "mlp.gate_proj.weight")
		wup, _ := getTensor(prefix + "mlp.up_proj.weight")
		wdown, _ := getTensor(prefix + "mlp.down_proj.weight")

		layers[i] = LayerWeights{
			InputNorm: inNorm,
			PostNorm:  postNorm,
			WQ:        wq,
			WK:        wk,
			WV:        wv,
			WO:        wo,
			WGate:     wgate,
			WUp:       wup,
			WDown:     wdown,
		}
	}

	return &ModelWeights{
		EmbedTokens: embed,
		Layers:      layers,
		Norm:        norm,
		LMHead:      lmHead,
	}, &cfg, nil
}

// GenerateDeterministicWeights generates initialized weights for self-contained testing.
func GenerateDeterministicWeights(cfg *ModelConfig) *ModelWeights {
	headDim := cfg.HeadDim()
	kvDim := cfg.NumKeyValueHeads * headDim

	initSlice := func(size int, scale float32) []float32 {
		s := make([]float32, size)
		for i := range s {
			s[i] = float32(math.Sin(float64(i+1)*0.01)) * scale
		}
		return s
	}

	layers := make([]LayerWeights, cfg.NumHiddenLayers)
	for i := 0; i < cfg.NumHiddenLayers; i++ {
		layers[i] = LayerWeights{
			InputNorm: initSlice(cfg.HiddenSize, 1.0),
			PostNorm:  initSlice(cfg.HiddenSize, 1.0),
			WQ:        initSlice(cfg.HiddenSize*cfg.HiddenSize, 0.02),
			WK:        initSlice(cfg.HiddenSize*kvDim, 0.02),
			WV:        initSlice(cfg.HiddenSize*kvDim, 0.02),
			WO:        initSlice(cfg.HiddenSize*cfg.HiddenSize, 0.02),
			WGate:     initSlice(cfg.HiddenSize*cfg.IntermediateSize, 0.02),
			WUp:       initSlice(cfg.HiddenSize*cfg.IntermediateSize, 0.02),
			WDown:     initSlice(cfg.IntermediateSize*cfg.HiddenSize, 0.02),
		}
	}

	return &ModelWeights{
		EmbedTokens: initSlice(cfg.VocabSize*cfg.HiddenSize, 0.1),
		Layers:      layers,
		Norm:        initSlice(cfg.HiddenSize, 1.0),
		LMHead:      initSlice(cfg.VocabSize*cfg.HiddenSize, 0.02),
	}
}

// Forward computes the step-for-step forward pass of the LLaMA model for a single token at position pos.
func Forward(weights *ModelWeights, cfg *ModelConfig, kv *KVCacheState, tokenID int32, pos int) []float32 {
	headDim := cfg.HeadDim()
	qHeads := cfg.NumAttentionHeads
	kvHeads := cfg.NumKeyValueHeads
	kvGroupSize := qHeads / kvHeads
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	// 1. Embedding lookup
	x := make([]float32, cfg.HiddenSize)
	embedOffset := int(tokenID) * cfg.HiddenSize
	if embedOffset >= 0 && embedOffset+cfg.HiddenSize <= len(weights.EmbedTokens) {
		copy(x, weights.EmbedTokens[embedOffset:embedOffset+cfg.HiddenSize])
	}

	// 2. Transformer Decoder Layers
	for layerIdx := 0; layerIdx < cfg.NumHiddenLayers; layerIdx++ {
		layer := &weights.Layers[layerIdx]

		// Attention Pre-Normalization (RMSNorm)
		normX := rmsNorm(x, layer.InputNorm, cfg.RmsNormEps)

		// Q, K, V Projections
		q := matmulVec(normX, layer.WQ, cfg.HiddenSize, cfg.HiddenSize)
		k := matmulVec(normX, layer.WK, cfg.HiddenSize, kvHeads*headDim)
		v := matmulVec(normX, layer.WV, cfg.HiddenSize, kvHeads*headDim)

		// Rotary Position Embedding (RoPE)
		applyRoPE(q, pos, qHeads, headDim, cfg.RopeTheta)
		applyRoPE(k, pos, kvHeads, headDim, cfg.RopeTheta)

		// Store in KV cache
		if pos < len(kv.K[layerIdx]) {
			copy(kv.K[layerIdx][pos], k)
			copy(kv.V[layerIdx][pos], v)
		}

		// Multi-Head / Grouped-Query Attention
		attnOut := make([]float32, cfg.HiddenSize)
		seqLen := pos + 1

		for h := 0; h < qHeads; h++ {
			kvH := h / kvGroupSize
			qPtr := q[h*headDim : (h+1)*headDim]

			// Step 2a: Dot products Q x K^T for 0..pos
			scores := make([]float32, seqLen)
			maxScore := float32(-1e20)
			for t := 0; t < seqLen; t++ {
				kPtr := kv.K[layerIdx][t][kvH*headDim : (kvH+1)*headDim]
				dot := float32(0)
				for d := 0; d < headDim; d++ {
					dot += qPtr[d] * kPtr[d]
				}
				dot *= scale
				scores[t] = dot
				if dot > maxScore {
					maxScore = dot
				}
			}

			// Step 2b: Softmax
			sumExp := float32(0)
			for t := 0; t < seqLen; t++ {
				e := float32(math.Exp(float64(scores[t] - maxScore)))
				scores[t] = e
				sumExp += e
			}

			// Step 2c: Weighted sum over V
			for d := 0; d < headDim; d++ {
				acc := float32(0)
				for t := 0; t < seqLen; t++ {
					vPtr := kv.V[layerIdx][t][kvH*headDim : (kvH+1)*headDim]
					acc += scores[t] * vPtr[d]
				}
				if sumExp > 0 {
					acc /= sumExp
				}
				attnOut[h*headDim+d] = acc
			}
		}

		// Output Projection & Residual 1
		attnProj := matmulVec(attnOut, layer.WO, cfg.HiddenSize, cfg.HiddenSize)
		for i := range x {
			x[i] += attnProj[i]
		}

		// SwiGLU Feed-Forward Network & Residual 2
		normH := rmsNorm(x, layer.PostNorm, cfg.RmsNormEps)
		gate := matmulVec(normH, layer.WGate, cfg.HiddenSize, cfg.IntermediateSize)
		up := matmulVec(normH, layer.WUp, cfg.HiddenSize, cfg.IntermediateSize)

		ffnAct := make([]float32, cfg.IntermediateSize)
		for i := range ffnAct {
			// silu(gate) * up
			silu := gate[i] / (1.0 + float32(math.Exp(float64(-gate[i]))))
			ffnAct[i] = silu * up[i]
		}

		ffnOut := matmulVec(ffnAct, layer.WDown, cfg.IntermediateSize, cfg.HiddenSize)
		for i := range x {
			x[i] += ffnOut[i]
		}
	}

	// 3. Final RMSNorm
	normedOut := rmsNorm(x, weights.Norm, cfg.RmsNormEps)

	// 4. LM Head Logits Projection
	logits := matmulVec(normedOut, weights.LMHead, cfg.HiddenSize, cfg.VocabSize)
	return logits
}

func rmsNorm(x, weight []float32, eps float32) []float32 {
	out := make([]float32, len(x))
	sumSq := float32(0)
	for _, v := range x {
		sumSq += v * v
	}
	meanSq := sumSq / float32(len(x))
	rsqrt := float32(1.0 / math.Sqrt(float64(meanSq+eps)))

	for i := range out {
		w := float32(1.0)
		if i < len(weight) {
			w = weight[i]
		}
		out[i] = x[i] * rsqrt * w
	}
	return out
}

func applyRoPE(vec []float32, pos int, numHeads, headDim int, theta float32) {
	for h := 0; h < numHeads; h++ {
		headVec := vec[h*headDim : (h+1)*headDim]
		for i := 0; i < headDim/2; i++ {
			freq := 1.0 / math.Pow(float64(theta), float64(2*i)/float64(headDim))
			val := float64(pos) * freq
			cos := float32(math.Cos(val))
			sin := float32(math.Sin(val))

			v0 := headVec[i]
			v1 := headVec[i+headDim/2]
			headVec[i] = v0*cos - v1*sin
			headVec[i+headDim/2] = v0*sin + v1*cos
		}
	}
}

func matmulVec(x []float32, w []float32, inDim, outDim int) []float32 {
	out := make([]float32, outDim)
	if len(w) < inDim*outDim {
		return out
	}
	for o := 0; o < outDim; o++ {
		sum := float32(0)
		for i := 0; i < inDim; i++ {
			sum += x[i] * w[o*inDim+i]
		}
		out[o] = sum
	}
	return out
}
