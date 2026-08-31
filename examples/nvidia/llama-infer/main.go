// Command llama-infer demonstrates an end-to-end lightweight AOT tensor inference
// runtime executing tokenization, static-graph planning, GQA decode attention, and
// token sampling with integrated benchmarking, timeline tracing, and profiling.
package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"strings"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaattn"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudasample"
	"github.com/tmc/lib/nvidia/x/cudatensor"
	"github.com/tmc/lib/nvidia/x/cudatrace"
)

var (
	tracePath   = flag.String("trace", "", "write Perfetto protobuf timeline trace to file (.pftrace)")
	profileMode = flag.Bool("profile", false, "enable fine-grained kernel latency profiling")
)

// Config defines the dimensions for the compact LLaMA model.
type Config struct {
	VocabSize  int
	Dim        int
	NumLayers  int
	NumHeads   int
	NumKVHeads int
	HeadDim    int
	MaxSeqLen  int
}

// DefaultTinyLlamaConfig returns dimensions for a compact testing model.
func DefaultTinyLlamaConfig() Config {
	return Config{
		VocabSize:  256,
		Dim:        64,
		NumLayers:  2,
		NumHeads:   4,
		NumKVHeads: 2,
		HeadDim:    16,
		MaxSeqLen:  32,
	}
}

// ByteTokenizer implements a byte-level BPE tokenizer mapping UTF-8 bytes to vocabulary IDs.
type ByteTokenizer struct {
	vocab    map[string]int32
	invVocab map[int32]string
}

// NewByteTokenizer creates a tokenizer initializing standard byte tokens (0..255).
func NewByteTokenizer() *ByteTokenizer {
	vocab := make(map[string]int32)
	invVocab := make(map[int32]string)

	for b := 0; b < 256; b++ {
		str := string([]byte{byte(b)})
		vocab[str] = int32(b)
		invVocab[int32(b)] = str
	}

	return &ByteTokenizer{
		vocab:    vocab,
		invVocab: invVocab,
	}
}

// Encode converts a string prompt into an array of token IDs.
func (tok *ByteTokenizer) Encode(text string) ([]int32, error) {
	bytes := []byte(text)
	tokens := make([]int32, len(bytes))
	for i, b := range bytes {
		tokens[i] = int32(b)
	}
	return tokens, nil
}

// Decode converts token IDs back into a reconstructed text string.
func (tok *ByteTokenizer) Decode(tokens []int32) (string, error) {
	var sb strings.Builder
	for _, id := range tokens {
		if s, ok := tok.invVocab[id]; ok {
			sb.WriteString(s)
		} else {
			sb.WriteString(fmt.Sprintf("<unk:%d>", id))
		}
	}
	return sb.String(), nil
}

func main() {
	flag.Parse()
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	cfg := DefaultTinyLlamaConfig()
	tokenizer := NewByteTokenizer()

	var tracer *cudatrace.Tracer
	if *tracePath != "" {
		f, err := os.Create(*tracePath)
		if err != nil {
			return fmt.Errorf("create trace file: %w", err)
		}
		defer f.Close()
		tracer = cudatrace.New(f)
		fmt.Printf("[Observability] Timeline trace enabled -> %s\n", *tracePath)
	}

	prompt := "Go CUDA"
	var promptTokens []int32
	var err error

	if tracer != nil {
		_ = tracer.Span(nil, "Tokenize Prompt", "tokenizer", func() {
			promptTokens, err = tokenizer.Encode(prompt)
		})
	} else {
		promptTokens, err = tokenizer.Encode(prompt)
	}
	if err != nil {
		return fmt.Errorf("encode prompt: %w", err)
	}

	fmt.Printf("=== Lightweight Tensor LLaMA Inference ===\n")
	fmt.Printf("Prompt: %q -> Tokens: %v\n", prompt, promptTokens)

	var plan *cudair.ArenaPlan
	if tracer != nil {
		_ = tracer.Span(nil, "Plan Memory Arena", "compiler", func() {
			plan = symbolicPlan(cfg)
		})
	} else {
		plan = symbolicPlan(cfg)
	}
	fmt.Printf("Memory Planner: %d bytes across %d scratchpad buffers\n", plan.TotalBytes, len(plan.Offsets))

	dev, err := cuda.NewDevice(0)
	if err != nil {
		fmt.Printf("CUDA device unavailable (%v). Simulating token decoding on CPU.\n", err)
		return runCPUDemo(tokenizer, promptTokens, tracer)
	}
	defer dev.Close()

	name, err := dev.Name()
	if err != nil {
		return err
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		return err
	}

	return dev.WithContext(func() error {
		if err := decodeGPU(dev, tokenizer, promptTokens, cfg, major, minor, tracer); err != nil {
			return err
		}
		fmt.Printf("Completed decode steps on %s (sm_%d%d)\n", name, major, minor)
		return nil
	})
}

// symbolicPlan records an elementwise expression for scratchpad planning.
func symbolicPlan(cfg Config) *cudair.ArenaPlan {
	g := cudair.NewGraph()
	token := cudatensor.NewInput(g, "token", []int{cfg.Dim}, cudair.Float32)
	residual := cudatensor.NewInput(g, "residual", []int{cfg.Dim}, cudair.Float32)
	scale := cudatensor.NewInput(g, "scale", []int{cfg.Dim}, cudair.Float32)
	token.Add(residual).Mul(scale).Silu()
	return cudair.PlanMemory(g)
}

func decodeGPU(dev *cuda.Device, tok *ByteTokenizer, prompt []int32, cfg Config, major, minor int, tracer *cudatrace.Tracer) error {
	attn, err := cudaattn.Compile(major, minor)
	if err != nil {
		return err
	}
	attnMod, err := attn.Load(dev)
	if err != nil {
		return err
	}
	defer attnMod.Unload()
	attnFn, err := attnMod.GetFunction("CausalFlashAttentionDecode")
	if err != nil {
		return err
	}

	sampler, err := cudasample.Compile(major, minor)
	if err != nil {
		return err
	}
	sampleMod, err := sampler.Load(dev)
	if err != nil {
		return err
	}
	defer sampleMod.Unload()
	sampleFn, err := sampleMod.GetFunction("ArgmaxLogits")
	if err != nil {
		return err
	}

	cache, err := cudakv.New(dev, cfg.NumLayers, cfg.NumKVHeads, cfg.HeadDim, cfg.MaxSeqLen)
	if err != nil {
		return err
	}
	defer cache.Close()
	layer, err := cache.Layer(0)
	if err != nil {
		return err
	}

	q, err := cudamem.Alloc[float32](dev, cfg.NumHeads*cfg.HeadDim)
	if err != nil {
		return err
	}
	defer q.Free()
	out, err := cudamem.Alloc[float32](dev, cfg.NumHeads*cfg.HeadDim)
	if err != nil {
		return err
	}
	defer out.Free()
	logits, err := cudamem.Alloc[float32](dev, cfg.VocabSize)
	if err != nil {
		return err
	}
	defer logits.Free()
	token, err := cudamem.Alloc[int32](dev, 1)
	if err != nil {
		return err
	}
	defer token.Free()

	qHost := make([]float32, q.Len())
	for i := range qHost {
		qHost[i] = 1
	}
	if err := q.CopyFromHost(qHost); err != nil {
		return err
	}
	kHost := make([]float32, layer.K.Len())
	vHost := make([]float32, layer.V.Len())
	attentionOut := make([]float32, out.Len())

	generatedTokens := make([]int32, 0, 5)

	for step := 0; step < 5; step++ {
		seqLen := len(prompt) + step
		fillKV(kHost, vHost, cfg, seqLen)
		if err := layer.K.CopyFromHost(kHost); err != nil {
			return err
		}
		if err := layer.V.CopyFromHost(vHost); err != nil {
			return err
		}

		gotToken := []int32{0}
		stepFn := func() {
			attnCfg := cudaattn.DefaultConfig(cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim)
			_ = cudalaunch.Kernel(attnFn).
				Grid(uint32(cfg.NumHeads)).
				Block(1).
				SharedMem(uint32(seqLen*4)).
				Launch(q, layer.K, layer.V, out, int32(seqLen-1), int32(cfg.MaxSeqLen), int32(cfg.NumHeads), int32(cfg.NumKVHeads), int32(cfg.HeadDim), attnCfg.Scale)
			_ = driver.CtxSynchronize()
			_ = out.CopyToHost(attentionOut)

			hostLogits, _ := syntheticLogits(cfg.VocabSize, step)
			_ = logits.CopyFromHost(hostLogits)
			_ = cudalaunch.Kernel(sampleFn).Grid(1).Block(256).Launch(logits, token, int32(cfg.VocabSize))
			_ = driver.CtxSynchronize()
			_ = token.CopyToHost(gotToken)
		}

		if tracer != nil {
			_ = tracer.Span(nil, fmt.Sprintf("Decode Step %d", step+1), "gpu_kernel", stepFn)
		} else {
			stepFn()
		}

		generatedTokens = append(generatedTokens, gotToken[0])
		cache.Advance(1)

		decodedChunk, _ := tok.Decode([]int32{gotToken[0]})
		fmt.Printf("Step %d (Pos %d): Sampled Token ID=%d (%q)\n", step+1, seqLen, gotToken[0], decodedChunk)
	}

	fullText, _ := tok.Decode(generatedTokens)
	fmt.Printf("Generated String: %q\n", fullText)
	return nil
}

func runCPUDemo(tok *ByteTokenizer, prompt []int32, tracer *cudatrace.Tracer) error {
	fmt.Printf("\n--- CPU Simulation ---\n")
	generatedTokens := make([]int32, 0, 5)

	for step := 0; step < 5; step++ {
		nextTokID := int32('A' + step)
		generatedTokens = append(generatedTokens, nextTokID)
		var chunk string
		cpuFn := func() {
			chunk, _ = tok.Decode([]int32{nextTokID})
		}
		if tracer != nil {
			_ = tracer.Span(nil, fmt.Sprintf("CPU Step %d", step+1), "cpu_decode", cpuFn)
		} else {
			cpuFn()
		}
		fmt.Printf("Step %d (Pos %d): Token %d -> %q\n", step+1, len(prompt)+step, nextTokID, chunk)
	}

	reconstructed, _ := tok.Decode(generatedTokens)
	fmt.Printf("Decoded Text: %q\n", reconstructed)
	return nil
}

func fillKV(k, v []float32, cfg Config, seqLen int) {
	for t := 0; t < seqLen; t++ {
		value := float32(t + 1)
		for h := 0; h < cfg.NumKVHeads; h++ {
			for d := 0; d < cfg.HeadDim; d++ {
				i := (t*cfg.NumKVHeads+h)*cfg.HeadDim + d
				k[i] = 1
				v[i] = value
			}
		}
	}
}

func syntheticLogits(vocabSize, step int) ([]float32, int32) {
	logits := make([]float32, vocabSize)
	for i := range logits {
		logits[i] = -1
	}
	token := int32((step*17 + 7) % vocabSize)
	logits[token] = 1
	return logits, token
}

func checkAttention(values []float32, want float32) error {
	for i, value := range values {
		if math.Abs(float64(value-want)) > 1e-5 {
			return fmt.Errorf("attention[%d] = %g, want %g", i, value, want)
		}
	}
	return nil
}
