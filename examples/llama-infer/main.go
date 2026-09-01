// Command llama-infer demonstrates an end-to-end lightweight AOT static-graph tensor
// inference runtime for LLaMA architectures, automatically selecting either the Apple
// Silicon Metal backend (on darwin/arm64) or the NVIDIA CUDA backend with CPU oracle fallback.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
)

// Config defines hyperparameters for a LLaMA model.
type Config struct {
	Name       string
	ModelPath  string
	VocabSize  int
	Dim        int
	HiddenDim  int
	NumLayers  int
	NumHeads   int
	NumKVHeads int
	HeadDim    int
	MaxSeqLen  int
	Eps        float32
	RopeTheta  float32
	FuseEGraph bool
	Prompt     string
	// HTTPAddr sets the optional address for the live telemetry HTTP server.
	HTTPAddr string
	// PeerAddr sets an optional peer node to join into the P2P inference mesh.
	PeerAddr string
	// NodeKeyPath stores the persistent Iroh node key. An empty path uses an
	// ephemeral identity, which is useful for tests.
	NodeKeyPath string
	// PeerStatePath stores registered mesh peers. An empty path keeps peer
	// registrations in memory only, which is useful for tests.
	PeerStatePath string
	// ServeMode keeps the server running indefinitely to accept OpenAI requests.
	ServeMode bool
	// BOSTokenID is prepended to the prompt when nonzero. LLaMA 3 conditions
	// on a beginning-of-text token, and its absence skews the first generated
	// token.
	BOSTokenID int32
	// GenTokens sets the number of autoregressive decode tokens to generate.
	// If <= 0, defaults to 10.
	GenTokens int
	// SearchGraphs is the number of graph search candidates explored.
	SearchGraphs int
	// SearchDepths is an optional list of search depths to sweep during benchmarks.
	SearchDepths []int
	// TestMode explicitly permits synthetic deterministic test weights for unit testing.
	TestMode bool
}

// DefaultTinyLlamaConfig returns configuration for a tiny testing model.
func DefaultTinyLlamaConfig() Config {
	return Config{
		TestMode:   true,
		Name:       "TinyLlama-Testing",
		VocabSize:  256,
		Dim:        64,
		HiddenDim:  128,
		NumLayers:  2,
		NumHeads:   4,
		NumKVHeads: 2,
		HeadDim:    16,
		MaxSeqLen:  32,
		Eps:        1e-5,
		RopeTheta:  10000.0,
	}
}

// Llama3_2_1BConfig returns configuration for official LLaMA-3.2-1B.
func Llama3_2_1BConfig(modelPath string) Config {
	return Config{
		Name:       "LLaMA-3.2-1B-Instruct",
		ModelPath:  modelPath,
		VocabSize:  128256,
		Dim:        2048,
		HiddenDim:  8192,
		NumLayers:  16,
		NumHeads:   32,
		NumKVHeads: 8,
		HeadDim:    64,
		MaxSeqLen:  8192,
		Eps:        1e-5,
		RopeTheta:  500000.0,
		BOSTokenID: 128000,
	}
}

// Llama3_8BConfig returns configuration for standard LLaMA-3-8B.
func Llama3_8BConfig() Config {
	return Config{
		Name:       "LLaMA-3-8B",
		VocabSize:  128256,
		Dim:        4096,
		HiddenDim:  14336,
		NumLayers:  32,
		NumHeads:   32,
		NumKVHeads: 8,
		HeadDim:    128,
		MaxSeqLen:  8192,
		Eps:        1e-5,
		RopeTheta:  500000.0,
	}
}

// TinyLlama_1_1BConfig returns configuration for TinyLlama-1.1B.
func TinyLlama_1_1BConfig() Config {
	return Config{
		Name:       "TinyLlama-1.1B",
		VocabSize:  32000,
		Dim:        2048,
		HiddenDim:  5632,
		NumLayers:  22,
		NumHeads:   32,
		NumKVHeads: 4,
		HeadDim:    64,
		MaxSeqLen:  2048,
		Eps:        1e-5,
		RopeTheta:  10000.0,
	}
}

func findDefaultLlama3_2_1BPath() string {
	home, _ := os.UserHomeDir()
	pattern := filepath.Join(home, ".cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/*")
	matches, _ := filepath.Glob(pattern)
	if len(matches) > 0 {
		return matches[0]
	}
	return ""
}

func main() {
	modelSize := flag.String("model", "llama3.2-1b", "model size (tiny, llama3.2-1b, 1b, 8b)")
	modelPath := flag.String("path", "", "path to model directory containing model.safetensors")
	prompt := flag.String("prompt", "The key to scalable AI architectures is", "prompt text for inference")
	genTokens := flag.Int("gen-tokens", 10, "number of tokens to generate during autoregressive decode")
	fuseEGraph := flag.Bool("fuse-egraph", false, "enable E-Graph equality saturation and megakernel fusion")
	httpAddr := flag.String("http", "", "HTTP address to serve live telemetry dashboard (e.g. :8080)")
	peerAddr := flag.String("peer", "", "peer HTTP address to join into P2P inference mesh (e.g. spark-31db:8080)")
	nodeKeyPath := flag.String("node-key", defaultNodeKeyPath(), "path to persistent Iroh node key")
	peerStatePath := flag.String("peer-state", defaultPeerStatePath(), "path to persistent mesh peer registrations")
	serve := flag.Bool("serve", false, "run in continuous OpenAI-compatible serving mode")
	benchMode := flag.Bool("bench", false, "run benchstat-compatible benchmark suite")
	benchCount := flag.Int("bench-count", 10, "number of benchmark iterations to run")
	searchGraphs := flag.Int("search-graphs", 100, "number of search graphs / depth for compiler optimization search")
	searchDepthsFlag := flag.String("search-depths", "", "comma-separated list of search depths to sweep (e.g. 0,10,50,100,200,500)")
	flag.Parse()

	var cfg Config
	if *modelPath != "" {
		if _, loadedCfg, err := LoadModel(*modelPath); err == nil {
			name := filepath.Base(*modelPath)
			if *modelSize == "llama3.2-1b" || *modelSize == "llama3.2" || *modelSize == "3.2-1b" {
				name = "LLaMA-3.2-1B-Instruct"
			}
			cfg = Config{
				Name:       name,
				ModelPath:  *modelPath,
				VocabSize:  loadedCfg.VocabSize,
				Dim:        loadedCfg.HiddenSize,
				HiddenDim:  loadedCfg.IntermediateSize,
				NumLayers:  loadedCfg.NumHiddenLayers,
				NumHeads:   loadedCfg.NumAttentionHeads,
				NumKVHeads: loadedCfg.NumKeyValueHeads,
				HeadDim:    loadedCfg.HeadDim(),
				MaxSeqLen:  loadedCfg.MaxSeqLen,
				Eps:        loadedCfg.RmsNormEps,
				RopeTheta:  loadedCfg.RopeTheta,
			}
		}
	}

	if cfg.Dim == 0 {
		switch *modelSize {
		case "llama3.2-1b", "llama3.2", "3.2-1b":
			path := *modelPath
			if path == "" {
				path = findDefaultLlama3_2_1BPath()
			}
			cfg = Llama3_2_1BConfig(path)
		case "1b", "tinyllama":
			cfg = TinyLlama_1_1BConfig()
		case "8b", "llama3":
			cfg = Llama3_8BConfig()
		default:
			cfg = DefaultTinyLlamaConfig()
		}
	}
	cfg.GenTokens = *genTokens
	cfg.FuseEGraph = *fuseEGraph
	cfg.Prompt = *prompt
	cfg.HTTPAddr = *httpAddr
	cfg.PeerAddr = *peerAddr
	cfg.NodeKeyPath = *nodeKeyPath
	cfg.PeerStatePath = *peerStatePath
	cfg.ServeMode = *serve
	cfg.SearchGraphs = *searchGraphs

	if *searchDepthsFlag != "" {
		parts := strings.Split(*searchDepthsFlag, ",")
		for _, p := range parts {
			p = strings.TrimSpace(p)
			if d, err := strconv.Atoi(p); err == nil {
				cfg.SearchDepths = append(cfg.SearchDepths, d)
			}
		}
	}
	if len(cfg.SearchDepths) == 0 {
		cfg.SearchDepths = []int{*searchGraphs}
	}

	if *benchMode {
		if err := RunBenchSuite(cfg, *benchCount); err != nil {
			log.Fatalf("llama-infer: %v", err)
		}
		return
	}

	if *serve && cfg.HTTPAddr == "" {
		cfg.HTTPAddr = ":8080"
	}

	fmt.Printf("=== Cross-Platform LLaMA Tensor Inference (%s on %s/%s) ===\n", cfg.Name, runtime.GOOS, runtime.GOARCH)
	fmt.Printf("Architecture: Dim=%d, HiddenDim=%d, Layers=%d, Heads=%d, KVHeads=%d, Vocab=%d\n",
		cfg.Dim, cfg.HiddenDim, cfg.NumLayers, cfg.NumHeads, cfg.NumKVHeads, cfg.VocabSize)
	if cfg.ModelPath != "" {
		fmt.Printf("Safetensors Path: %s\n", cfg.ModelPath)
	}
	fmt.Printf("E-Graph Optimization: %v\n", cfg.FuseEGraph)
	if *serve {
		fmt.Printf("Serving Mode: Active (HTTP: %s, OpenAI API: http://localhost%s/v1/chat/completions)\n\n", cfg.HTTPAddr, cfg.HTTPAddr)
	} else {
		fmt.Printf("Prompt: %q\n\n", *prompt)
	}

	if err := runPlatform(cfg); err != nil {
		log.Fatalf("llama-infer: %v", err)
	}

	if *serve {
		select {} // Keep server running indefinitely
	}
}
