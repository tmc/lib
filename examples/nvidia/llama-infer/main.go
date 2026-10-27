// Command llama-infer demonstrates an end-to-end tensor LLM inference runtime
// executing full Hugging Face BPE tokenization, 100% GPU-resident transformer decoder
// forward passes, KV-cache management, GQA causal attention, SwiGLU MLP, and token generation
// with integrated pprof GPU SASS profiling and Perfetto timeline tracing.
package main

import (
	"context"
	_ "embed"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/tmc/lib/lm/speculative"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaattn"
	"github.com/tmc/lib/nvidia/x/cudagrammar"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cudasample"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/cudatrace"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

var (
	modelPath         = flag.String("model", "", "path to model directory containing config.json, tokenizer.json, and model.safetensors")
	promptFlag        = flag.String("prompt", "The capital of France is", "text prompt for model generation")
	tokenIDsFlag      = flag.String("token-ids", "", "comma-separated int32 prompt token IDs (overrides -prompt tokenization when non-empty)")
	stepsFlag         = flag.Int("steps", 10, "number of autoregressive tokens to generate")
	tempFlag          = flag.Float64("temp", 0.0, "sampling temperature (0.0 = greedy argmax)")
	seedFlag          = flag.Uint64("seed", 0, "deterministic Philox-4x32-10 sequence seed for sampling")
	topKFlag          = flag.Int("top-k", 0, "top-k sampling cutoff (0 = disabled)")
	topPFlag          = flag.Float64("top-p", 1.0, "sorting-free cumulative probability mass top-p cutoff in [0, 1]")
	minPFlag          = flag.Float64("min-p", 0.0, "min-p relative probability threshold in [0, 1]")
	jsonSchemaFlag    = flag.String("json-schema", "", "optional JSON Schema constraining token sampling")
	syncGuardFlag     = flag.String("sync-guard", "", "host synchronization guard mode for decode graph execution: allow, warn, or reject")
	yocoProducersFlag = flag.Int("yoco-producers", 0, "number of KV producer layers for YoCo cross-layer KV sharing topology (0 = disabled)")
	swaWindowFlag     = flag.Int("swa-window", 0, "sliding-window attention (SWA) bounded replay window size (0 = disabled)")
	tracePath         = flag.String("trace", "", "write Perfetto protobuf timeline trace to file (.pftrace)")
	profilePath       = flag.String("profile", "", "write GPU SASS/PTX pprof profile to file (.pprof)")
	disasmFlag        = flag.String("disasm", "", "print JIT-compiled disassembly for kernel (e.g. LinearGEMV, RMSNorm, all)")
	disasmModeFlag    = flag.String("disasm-mode", "both", "disassembly format mode: both, ptx, or sass")
	fuseEGraphFlag    = flag.Bool("fuse-egraph", false, "enable pure-Go E-Graph megakernel fusion (fused RMSNorm+QKV and SwiGLU)")
	fuseDecodeFlag    = flag.Bool("fuse-decode", false, "Use 5-node/layer norm-folded and epilogue-fused decode kernels")
	specDraftFlag     = flag.Int("draft", 0, "speculative decoding draft budget (e.g. 3..5, 0 = disabled)")
	benchFlag         = flag.Bool("bench", false, "run benchstat-compatible benchmark suite")
	benchCountFlag    = flag.Int("bench-count", 10, "number of benchmark iterations to run")
	weightDTypeFlag   = flag.String("weight-dtype", "f32", "device storage of matrix weights: f32 or bf16 (bf16 rounds to nearest even; exact for BF16 checkpoints)")
	cudaGraphFlag     = flag.Bool("cuda-graph", false, "replay the single-token decode step as an explicit CUDA graph")
	stepTimesFlag     = flag.Bool("step-times", false, "time every forward pass with CUDA events and print each step's wall and event nanoseconds")
	egraphFlag        = flag.Bool("egraph", false, "run norms, projections, SwiGLU and residuals as nvidia/x/egraph/hopper generated kernels (needs -weight-dtype bf16)")
)

func main() {
	flag.Parse()
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() (rerr error) {
	if *disasmFlag != "" {
		return printKernelDisassembly(*disasmFlag, *disasmModeFlag)
	}

	var tracer *cudatrace.Tracer
	if *tracePath != "" {
		f, err := os.Create(*tracePath)
		if err != nil {
			return fmt.Errorf("create trace file: %w", err)
		}
		defer closeInto(&rerr, f, "close trace file")
		tracer = cudatrace.New(f)
		fmt.Printf("[Observability] Timeline trace enabled -> %s\n", *tracePath)
	}

	var profSession *cudaprof.Session
	if *profilePath != "" {
		f, err := os.Create(*profilePath)
		if err != nil {
			return fmt.Errorf("create profile file: %w", err)
		}
		defer closeInto(&rerr, f, "close profile file")
		profSession = cudaprof.NewSession(f)
		// Registered after the file's, so the session flushes into the
		// file before the file closes.
		defer closeInto(&rerr, profSession, "close profile session")
		fmt.Printf("[Profiling] GPU pprof SASS profiler enabled -> %s\n", *profilePath)
	}

	// 1. Load or initialize Tokenizer
	effectivePath := findDefaultModelPath()
	tokenizer, tokType, err := loadTokenizer(effectivePath)
	if err != nil {
		return err
	}

	dtype, err := parseWeightDType(*weightDTypeFlag)
	if err != nil {
		return err
	}
	if *cudaGraphFlag && *specDraftFlag > 0 {
		return errors.New("-cuda-graph cannot be combined with -draft")
	}

	// 2. Load or generate Model Weights
	weights, cfg, err := loadWeights(effectivePath)
	if err != nil {
		return err
	}
	if effectivePath != "" {
		fmt.Printf("[Model Loader] Loaded real weights from %s (layers: %d, hidden: %d)\n",
			effectivePath, cfg.NumHiddenLayers, cfg.HiddenSize)
	} else {
		fmt.Printf("[Model Loader] No model path given; initialized testing LLaMA model (layers: %d, hidden: %d, heads: %d, kv_heads: %d)\n",
			cfg.NumHiddenLayers, cfg.HiddenSize, cfg.NumAttentionHeads, cfg.NumKeyValueHeads)
	}

	// 3. Tokenize Prompt (or parse explicit -token-ids)
	var promptTokens []int32
	if *tokenIDsFlag != "" {
		promptTokens, err = parseTokenIDs(*tokenIDsFlag, cfg.VocabSize)
		if err != nil {
			return fmt.Errorf("parse -token-ids: %w", err)
		}
	} else {
		promptTokens, err = encodePrompt(tracer, tokenizer, *promptFlag)
		if err != nil {
			return fmt.Errorf("tokenize prompt: %w", err)
		}
		if promptTokens, err = effectivePrompt(promptTokens, cfg.VocabSize); err != nil {
			return err
		}
	}

	fmt.Printf("\n=== Tensor LLM Inference Engine ===\n")
	fmt.Printf("Tokenizer : %s\n", tokType)
	fmt.Printf("Prompt    : %q\n", *promptFlag)
	fmt.Printf("Tokens    : %v\n", promptTokens)
	fmt.Printf("Generating: %d tokens (temp=%.2f)\n\n", *stepsFlag, *tempFlag)

	if *yocoProducersFlag > 0 {
		topo, yerr := buildModelYOCOTopology(cfg.NumHiddenLayers, *yocoProducersFlag, *swaWindowFlag)
		if yerr != nil {
			return yerr
		}
		fmt.Printf("[KV Topology] YoCo cross-layer KV sharing enabled (%d/%d producer layers, %.1f%% memory savings)\n",
			*yocoProducersFlag, cfg.NumHiddenLayers, topo.MemorySavingsRatio()*100)
	}
	if *swaWindowFlag > 0 {
		win, serr := planLayerSWAReplay(len(promptTokens)+*stepsFlag, 1, *swaWindowFlag, *specDraftFlag, 64)
		if serr != nil {
			return serr
		}
		fmt.Printf("[KV Topology] SWA bounded replay active (window: %d, active: %d tokens, pages: %d)\n",
			*swaWindowFlag, win.ActiveTokens, win.RequiredPages)
	}

	// Check if CUDA device is available for 100% GPU-resident execution
	dev, err := cuda.NewDevice(0)
	if err == nil {
		defer func() {
			// Work on an undrained stream may still use the context.
			if errors.Is(rerr, errReleasesHeld) {
				return
			}
			if cerr := dev.Close(); cerr != nil {
				rerr = errors.Join(rerr, fmt.Errorf("close device: %w", cerr))
			}
		}()
		name, err := dev.Name()
		if err != nil {
			return fmt.Errorf("device name: %w", err)
		}
		major, minor, err := dev.ComputeCapability()
		if err != nil {
			return fmt.Errorf("device compute capability: %w", err)
		}
		fmt.Printf("[Execution Target] Running 100%% GPU-Resident on %s (sm_%d%d)\n", name, major, minor)

		return dev.WithContext(func() error {
			return runGPUInference(dev, weights, cfg, tokenizer, promptTokens, major, minor, tracer, profSession)
		})
	}

	if dtype != weightF32 {
		return fmt.Errorf("-weight-dtype %s needs a CUDA device (%v); the CPU reference runs float32 only", dtype, err)
	}
	if *cudaGraphFlag {
		return fmt.Errorf("-cuda-graph needs a CUDA device (%v); the CPU reference has no graph executor", err)
	}
	fmt.Printf("[Execution Target] CUDA device unavailable (%v). Running CPU reference executor.\n", err)
	return runCPUInference(weights, cfg, tokenizer, promptTokens, tracer, profSession)
}

func runGPUInference(
	dev *cuda.Device,
	w *ModelWeights,
	cfg *ModelConfig,
	tokenizer Tokenizer,
	promptTokens []int32,
	major, minor int,
	tracer *cudatrace.Tracer,
	prof *cudaprof.Session,
) (rerr error) {
	// The deferred releases below record their errors in td, which is
	// joined into the result last. Once the step timer reports that its
	// stream did not drain, the releases that run after it keep their
	// resources, which the undrained work may still use.
	td := new(teardown)
	defer func() {
		rerr = errors.Join(rerr, td.err())
	}()

	// Compile GPU kernel modules
	gm, err := CompileGPUModules(dev)
	if gm != nil {
		defer func() {
			if td.held() {
				return
			}
			if cerr := gm.Close(); cerr != nil {
				td.add(fmt.Errorf("close gpu modules: %w", cerr))
			}
		}()
	}
	if err != nil {
		return fmt.Errorf("compile gpu modules: %w", err)
	}

	attnModCompiled, err := cudaattn.Compile(major, minor)
	if err != nil {
		return fmt.Errorf("compile attn: %w", releaseErrorOwner(err))
	}
	attnMod, err := attnModCompiled.Load(dev)
	if err != nil {
		return releaseErrorOwner(err)
	}
	defer func() {
		if td.held() {
			return
		}
		if cerr := attnMod.Close(); cerr != nil {
			td.add(fmt.Errorf("close attention module: %w", cerr))
		}
	}()
	attnFn, err := attnMod.Function("CausalFlashAttentionDecode")
	if err != nil {
		return fmt.Errorf("lookup kernel CausalFlashAttentionDecode: %w", err)
	}

	sampleModCompiled, err := cudasample.Compile(major, minor)
	if err != nil {
		return fmt.Errorf("compile sampler: %w", releaseErrorOwner(err))
	}
	sampleMod, err := sampleModCompiled.Load(dev)
	if err != nil {
		return releaseErrorOwner(err)
	}
	defer func() {
		if td.held() {
			return
		}
		if cerr := sampleMod.Close(); cerr != nil {
			td.add(fmt.Errorf("close sampler module: %w", cerr))
		}
	}()
	sampleFn, err := sampleMod.Function("ArgmaxLogits")
	if err != nil {
		return fmt.Errorf("lookup kernel ArgmaxLogits: %w", err)
	}

	// Upload weights directly into GPU VRAM
	dtype, err := parseWeightDType(*weightDTypeFlag)
	if err != nil {
		return err
	}
	// uploadModel can return the weights it could not free along with its
	// error; release them on every path.
	gpuWeights, err := uploadModel(dev, w, cfg, dtype)
	if gpuWeights != nil {
		defer func() {
			if td.held() {
				return
			}
			if ferr := gpuWeights.Free(); ferr != nil {
				td.add(fmt.Errorf("free gpu weights: %w", ferr))
			}
		}()
	}
	if err != nil {
		return fmt.Errorf("upload weights to gpu: %w", err)
	}

	// Allocate GPU KV Cache & Scratchpad
	headDim := cfg.HeadDim()
	// cudakv.New can return a cache to close along with its error.
	kvCache, err := cudakv.New(dev, cfg.NumHiddenLayers, cfg.NumKeyValueHeads, headDim, cfg.MaxSeqLen)
	if kvCache != nil {
		defer func() {
			if td.held() {
				return
			}
			if cerr := kvCache.Close(); cerr != nil {
				td.add(fmt.Errorf("close kv cache: %w", cerr))
			}
		}()
	}
	if err != nil {
		return fmt.Errorf("alloc kv cache: %w", err)
	}

	scratch, err := NewGPUScratchpad(dev, cfg)
	if scratch != nil {
		defer func() {
			if td.held() {
				return
			}
			if ferr := scratch.Free(); ferr != nil {
				td.add(fmt.Errorf("free scratchpad: %w", ferr))
			}
		}()
	}
	if err != nil {
		return fmt.Errorf("alloc scratchpad: %w", err)
	}

	if *egraphFlag {
		eg, err := newEgraphDecoder(dev, w, cfg, gpuWeights, scratch)
		if err != nil {
			return fmt.Errorf("e-graph decoder: %w", err)
		}
		defer func() {
			if td.held() {
				return
			}
			if cerr := eg.Close(); cerr != nil {
				td.add(fmt.Errorf("close e-graph decoder: %w", cerr))
			}
		}()
		gm.eg = eg
		fmt.Printf("[E-Graph] %d generated launches per step: %s; llama-infer kernels: %s\n",
			eg.Kernels(), strings.Join(egraphGenerated, ", "), strings.Join(egraphFallback, ", "))
	}

	var dg *DecodeGraph
	if *cudaGraphFlag {
		if *specDraftFlag > 0 {
			return errors.New("-cuda-graph cannot be combined with -draft")
		}
		syncMode, serr := parseSyncGuardMode(*syncGuardFlag)
		if serr != nil {
			return serr
		}
		defer closeDecodeGraph(td, &dg)
		if dg, err = NewDecodeGraph(dev, gpuWeights, cfg, kvCache, scratch, gm, attnFn, sampleFn); err != nil {
			return fmt.Errorf("build decode graph: %w", err)
		}
		dg.SetSyncArbiter(newDecodeSyncArbiter(syncMode, nil))
		bucketRunner, berr := newDecodeBucketRunner(1, dg)
		if berr != nil {
			return fmt.Errorf("register decode bucket runner: %w", berr)
		}
		var bucketPad any
		dg, bucketPad, err = selectDecodeBucket(bucketRunner, 1)
		if err != nil {
			return fmt.Errorf("select decode bucket: %w (%v)", err, bucketPad)
		}
	}

	forwardCU := findForwardCUPath()

	var timer *stepTimer
	if prof != nil || tracer != nil || *stepTimesFlag {
		// newStepTimer can return a timer it could not release along with
		// its error; close it on every path.
		timer, err = newStepTimer(dev, prof, tracer, forwardCU)
		if timer != nil {
			defer func() {
				if cerr := timer.close(); cerr != nil {
					td.add(fmt.Errorf("close step timer: %w", cerr))
				}
			}()
		}
		if err != nil {
			return fmt.Errorf("start step timer: %w", err)
		}
	}

	if *benchFlag {
		forward := func(tok int32, pos int) (int32, error) {
			if dg != nil && pos >= len(promptTokens) {
				return dg.Step(pos, nil)
			}
			return GPUForwardPass(gpuWeights, cfg, kvCache, scratch, gm, attnFn, sampleFn, tok, pos)
		}
		return runBenchmark(os.Stdout, dtype, promptTokens, *stepsFlag, *benchCountFlag, cfg.MaxSeqLen, kvCache.Reset, forward)
	}

	fmt.Print("Streaming : ", *promptFlag)

	// Speculative decoding initialization
	var drafter *speculative.NgramDrafter
	if *specDraftFlag > 0 {
		drafter = speculative.NewNgramDrafter(3)
		fmt.Printf("[Speculative Decoding] N-gram drafter active (budget: %d tokens/step)\n", *specDraftFlag)
	}
	st := newDecodeState(promptTokens, *stepsFlag, drafter)
	if *specDraftFlag > 0 {
		sched, serr := newSpecVarlenScheduler(*specDraftFlag)
		if serr != nil {
			return serr
		}
		st.setScheduler(sched)
	}

	prefill := func(tokID int32, pos int) (int32, error) {
		var tok int32
		var err error
		labels := cudaprof.Labels("phase", "prefill", "pos", fmt.Sprintf("%d", pos), "token_id", fmt.Sprintf("%d", tokID))
		cudaprof.Do(context.Background(), labels, func(stepCtx context.Context) {
			if timer == nil {
				tok, err = GPUForwardPass(gpuWeights, cfg, kvCache, scratch, gm, attnFn, sampleFn, tokID, pos)
				return
			}
			err = timer.measure(stepCtx, fmt.Sprintf("Prefill Token %d", pos+1), stepAnnotations("prefill", pos, tokID, major, minor), func(stream *cuda.Stream) error {
				var ferr error
				_, ferr = enqueueGPUForwardPass(gpuWeights, cfg, kvCache, scratch, gm, attnFn, sampleFn, tokID, pos, stream)
				return ferr
			})
			if err == nil {
				tok = scratch.HostTokenOut[0]
			}
		})
		return tok, err
	}
	forward := func(tok int32, pos int) (int32, error) {
		return GPUForwardPass(gpuWeights, cfg, kvCache, scratch, gm, attnFn, sampleFn, tok, pos)
	}

	show := func(tok int32) error {
		return printToken(tokenizer, tok)
	}
	plain := func(step int, nextTokID int32, curPos int) (int32, error) {
		var tok int32
		var err error
		labels := cudaprof.Labels("phase", "decode", "step", fmt.Sprintf("%d", step+1), "pos", fmt.Sprintf("%d", curPos), "token_id", fmt.Sprintf("%d", nextTokID))
		cudaprof.Do(context.Background(), labels, func(stepCtx context.Context) {
			if timer == nil {
				if dg != nil {
					tok, err = dg.Step(curPos, nil)
					return
				}
				tok, err = GPUForwardPass(gpuWeights, cfg, kvCache, scratch, gm, attnFn, sampleFn, nextTokID, curPos)
				return
			}
			err = timer.measure(stepCtx, fmt.Sprintf("Decode Token %d", step+1), stepAnnotations("decode", curPos, nextTokID, major, minor), func(stream *cuda.Stream) error {
				var ferr error
				if dg != nil {
					_, ferr = dg.Enqueue(curPos, stream)
					return ferr
				}
				_, ferr = enqueueGPUForwardPass(gpuWeights, cfg, kvCache, scratch, gm, attnFn, sampleFn, nextTokID, curPos, stream)
				return ferr
			})
			if err == nil {
				tok = scratch.HostTokenOut[0]
			}
		})
		if err != nil {
			return 0, fmt.Errorf("gpu decode step %d: %w", step, err)
		}
		return tok, nil
	}

	prefillDuration, genDuration, err := generate(st, promptTokens, prefill, forward, plain, show, cfg.MaxSeqLen, *stepsFlag, *specDraftFlag)
	if err != nil {
		return err
	}

	fmt.Printf("\nGEN_IDS: %v\n", st.generated)
	if timer != nil && *stepTimesFlag {
		// Steps 0..len(promptTokens)-1 are prefill positions; the rest decode.
		for i, s := range timer.steps {
			fmt.Printf("STEP %d wall_ns=%d event_ns=%d\n", i, s.wall.Nanoseconds(), s.event.Nanoseconds())
		}
	}

	tokensPerSec := float64(len(st.generated)) / genDuration.Seconds()
	fmt.Printf("\n\n--- GPU Performance Metrics ---\n")
	fmt.Printf("Prefill Time : %v (%d prompt tokens)\n", prefillDuration, len(promptTokens))
	fmt.Printf("Decode Time  : %v (%d generated tokens)\n", genDuration, len(st.generated))
	fmt.Printf("Throughput   : %.2f tokens/sec\n", tokensPerSec)
	if len(st.generated) > 0 {
		tpot := genDuration / time.Duration(len(st.generated))
		if bw := formatBenchmarkBandwidth(gpuWeights.StreamedBytesPerToken(), tpot); bw != "" {
			fmt.Printf("Bandwidth    : %s\n", bw)
		}
	}

	return nil
}

func runCPUInference(
	weights *ModelWeights,
	cfg *ModelConfig,
	tokenizer Tokenizer,
	promptTokens []int32,
	tracer *cudatrace.Tracer,
	prof *cudaprof.Session,
) error {
	headDim := cfg.HeadDim()
	kvDim := cfg.NumKeyValueHeads * headDim
	kvCache := NewKVCacheState(cfg.NumHiddenLayers, cfg.MaxSeqLen, kvDim)

	fmt.Print("Streaming : ", *promptFlag)
	var lastLogits []float32

	tPrefill0 := time.Now()
	for pos, tokID := range promptTokens {
		lastLogits = Forward(weights, cfg, kvCache, tokID, pos)
	}

	prefillDuration := time.Since(tPrefill0)

	curPos := len(promptTokens)
	generatedTokens := make([]int32, 0, *stepsFlag)

	var grammarMatcher *cudagrammar.Matcher
	var grammarVocab *cudagrammar.Vocabulary
	if *jsonSchemaFlag != "" {
		g, gerr := cudagrammar.CompileString(*jsonSchemaFlag)
		if gerr != nil {
			return fmt.Errorf("compile -json-schema: %w", gerr)
		}
		grammarVocab, gerr = buildGrammarVocabulary(tokenizer, cfg.VocabSize)
		if gerr != nil {
			return gerr
		}
		grammarMatcher = g.Match()
	}

	tGen0 := time.Now()
	for step := 0; step < *stepsFlag; step++ {
		if curPos >= cfg.MaxSeqLen {
			break
		}

		var allowedWords []uint32
		if grammarMatcher != nil {
			bm, berr := grammarMatcher.Bitmask(grammarVocab)
			if berr != nil {
				return fmt.Errorf("compute grammar bitmask at step %d: %w", step, berr)
			}
			allowedWords = bm.Words
		}

		var nextTokID int32
		if *tempFlag <= 0 && len(allowedWords) == 0 {
			nextTokID = sampleArgmax(lastLogits)
		} else {
			var serr error
			nextTokID, serr = sampleStepLogits(lastLogits, step, sampleConfig{
				seed:           *seedFlag,
				temp:           float32(*tempFlag),
				topK:           *topKFlag,
				topP:           float32(*topPFlag),
				minP:           float32(*minPFlag),
				allowedBitmask: allowedWords,
			})
			if serr != nil {
				return serr
			}
		}
		if grammarMatcher != nil {
			if aerr := grammarMatcher.AcceptToken(grammarVocab, nextTokID); aerr != nil {
				return fmt.Errorf("grammar rejected sampled token %d at step %d: %w", nextTokID, step, aerr)
			}
		}

		generatedTokens = append(generatedTokens, nextTokID)
		if err := printToken(tokenizer, nextTokID); err != nil {
			return err
		}

		if tracer != nil {
			err := tracer.SpanSource(nil, fmt.Sprintf("Decode Token %d", step+1), "forward_pass", "examples/nvidia/llama-infer/model.go", 100, "", map[string]any{
				"phase": "cpu_decode",
				"step":  step + 1,
				"pos":   curPos,
			}, func() {
				lastLogits = Forward(weights, cfg, kvCache, nextTokID, curPos)
			})
			if err != nil {
				return fmt.Errorf("trace decode token %d: %w", step+1, err)
			}
		} else {
			lastLogits = Forward(weights, cfg, kvCache, nextTokID, curPos)
		}

		curPos++
	}

	genDuration := time.Since(tGen0)

	tokensPerSec := float64(len(generatedTokens)) / genDuration.Seconds()
	fmt.Printf("\n\n--- Performance Metrics ---\n")
	fmt.Printf("Prefill Time : %v (%d prompt tokens)\n", prefillDuration, len(promptTokens))
	fmt.Printf("Decode Time  : %v (%d generated tokens)\n", genDuration, len(generatedTokens))
	fmt.Printf("Throughput   : %.2f tokens/sec\n", tokensPerSec)

	return nil
}

// encodePrompt encodes prompt with tok. If tracer is not nil, it records
// the encoding as a span and returns an error if the span cannot be
// written; the encoding may not have run.
func encodePrompt(tracer *cudatrace.Tracer, tok Tokenizer, prompt string) ([]int32, error) {
	if tracer == nil {
		return tok.Encode(prompt)
	}
	var tokens []int32
	var err error
	if terr := tracer.Span(nil, "Tokenize Prompt", "tokenizer", func() {
		tokens, err = tok.Encode(prompt)
	}); terr != nil {
		return nil, fmt.Errorf("trace: %w", terr)
	}
	return tokens, err
}

// printToken decodes the token id with tok and prints its text to
// standard output.
func printToken(tok Tokenizer, id int32) error {
	s, err := tok.Decode([]int32{id})
	if err != nil {
		return fmt.Errorf("decode token %d: %w", id, err)
	}
	fmt.Print(s)
	return nil
}

// closeInto closes c and joins its error, if any, wrapped with what, into
// *errp.
func closeInto(errp *error, c io.Closer, what string) {
	if err := c.Close(); err != nil {
		*errp = errors.Join(*errp, fmt.Errorf("%s: %w", what, err))
	}
}

// archOr returns the compute capability cc reports. If cc fails, it
// returns major and minor with cc's error.
func archOr(cc func() (int, int, error), major, minor int) (int, int, error) {
	m, n, err := cc()
	if err != nil {
		return major, minor, err
	}
	return m, n, nil
}

func findForwardCUPath() string {
	candidates := []string{
		"forward.cu",
		"nvidia/llama-infer/forward.cu",
		"examples/nvidia/llama-infer/forward.cu",
	}
	for _, c := range candidates {
		if abs, err := filepath.Abs(c); err == nil && fileExists(abs) {
			return abs
		}
	}
	return "forward.cu"
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func sampleArgmax(logits []float32) int32 {
	if len(logits) == 0 {
		return 0
	}
	maxIdx := int32(0)
	maxVal := logits[0]
	for i, v := range logits {
		if v > maxVal {
			maxVal = v
			maxIdx = int32(i)
		}
	}
	return maxIdx
}

func sampleTemperature(logits []float32, temp float32) int32 {
	if len(logits) == 0 {
		return 0
	}
	if temp <= 0 {
		return sampleArgmax(logits)
	}

	maxVal := logits[0]
	for _, v := range logits {
		if v > maxVal {
			maxVal = v
		}
	}

	var sumExp float32
	probs := make([]float32, len(logits))
	for i, v := range logits {
		e := float32(math.Exp(float64((v - maxVal) / temp)))
		probs[i] = e
		sumExp += e
	}

	r := rand.Float32() * sumExp
	var acc float32
	for i, p := range probs {
		acc += p
		if r <= acc {
			return int32(i)
		}
	}
	return int32(len(logits) - 1)
}

func printKernelDisassembly(targetKernel, mode string) (rerr error) {
	archMajor, archMinor := 8, 0
	if dev, err := cuda.NewDevice(0); err == nil {
		defer closeInto(&rerr, dev, "close device")
		if archMajor, archMinor, err = archOr(dev.ComputeCapability, archMajor, archMinor); err != nil {
			fmt.Fprintf(os.Stderr, "compute capability: %v; compiling for sm_%d%d\n", err, archMajor, archMinor)
		}
	}

	opts := []nvrtcx.Option{
		nvrtcx.WithName("llama_forward.cu"),
		nvrtcx.WithArch(archMajor, archMinor),
	}
	mod, err := nvrtcx.Compile(ForwardKernelsCUDA, opts...)
	if err != nil {
		return fmt.Errorf("compile for disasm: %w", releaseErrorOwner(err))
	}

	ptx := mod.PTX()

	showPTX := mode == "ptx" || mode == "both"
	showSASS := mode == "sass" || mode == "both"

	if showPTX {
		fmt.Printf("\n=======================================================\n")
		fmt.Printf("=== 1. JIT-Compiled PTX Virtual Assembly (compute_%d%d) ===\n", archMajor, archMinor)
		fmt.Printf("=======================================================\n")
		if targetKernel == "all" {
			fmt.Println(ptx)
		} else {
			lines := strings.Split(ptx, "\n")
			inTarget := false
			printed := 0
			for _, l := range lines {
				if strings.Contains(l, ".entry "+targetKernel) {
					inTarget = true
				} else if inTarget && strings.HasPrefix(l, ".visible .entry ") {
					break
				}
				if inTarget {
					fmt.Println(l)
					printed++
				}
			}
			if printed == 0 {
				fmt.Printf("[Disasm Notice] Kernel %q not found in PTX.\nAvailable kernels: EmbeddingLookup, RMSNorm, LinearGEMV, ApplyRoPE, StoreKVCache, SwiGLU, AddResidual\n", targetKernel)
			}
		}
	}

	if showSASS {
		fmt.Printf("\n=======================================================\n")
		fmt.Printf("=== 2. Hardware SASS Instruction Microarchitecture  ===\n")
		fmt.Printf("=======================================================\n")
		return printSASSDisassembly(targetKernel)
	}

	return nil
}

//go:embed forward.sass
var forwardSASS string

func printSASSDisassembly(targetKernel string) error {
	d, err := cudasass.ParseString(forwardSASS)
	if err != nil {
		return fmt.Errorf("parse SASS disassembly: %w", err)
	}

	if targetKernel == "all" {
		for _, fn := range d.Functions {
			fmt.Printf("\n--- Kernel: %s ---\n", fn.Name)
			for _, inst := range fn.Instructions {
				fmt.Println(inst.Raw)
			}
		}
		return nil
	}

	if fn := d.Function(targetKernel); fn != nil {
		fmt.Printf("\n--- Kernel: %s ---\n", fn.Name)
		for _, inst := range fn.Instructions {
			fmt.Println(inst.Raw)
		}
	} else {
		fmt.Printf("[SASS Notice] %q is not among the kernels in forward.sass; run go run gen.go to regenerate it.\n", targetKernel)
	}
	return nil
}

// parseTokenIDs parses a comma-separated list of int32 token IDs and checks
// that each ID is within [0, vocabSize).
func parseTokenIDs(spec string, vocabSize int) ([]int32, error) {
	trimmed := strings.TrimSpace(spec)
	if trimmed == "" {
		return nil, errors.New("empty token ID list")
	}
	parts := strings.Split(trimmed, ",")
	tokens := make([]int32, 0, len(parts))
	for i, part := range parts {
		field := strings.TrimSpace(part)
		if field == "" {
			return nil, fmt.Errorf("token %d: empty token ID", i)
		}
		v, err := strconv.ParseInt(field, 10, 32)
		if err != nil {
			return nil, fmt.Errorf("token %d (%q): %w", i, field, err)
		}
		tok := int32(v)
		if err := checkToken(tok, vocabSize); err != nil {
			return nil, fmt.Errorf("token %d: %w", i, err)
		}
		tokens = append(tokens, tok)
	}
	return tokens, nil
}

// effectivePrompt returns the tokens to run: tokens, or token 1 when tokens
// is empty. Every token it returns, including that default, must be in a
// vocabulary of vocab tokens.
func effectivePrompt(tokens []int32, vocab int) ([]int32, error) {
	if len(tokens) == 0 {
		tokens = []int32{1}
	}
	for i, tok := range tokens {
		if err := checkToken(tok, vocab); err != nil {
			return nil, fmt.Errorf("prompt token %d: %w", i, err)
		}
	}
	return tokens, nil
}

// loadTokenizer loads the tokenizer in dir. With no dir it returns the
// byte-level fallback tokenizer. A tokenizer that fails to load is an error:
// it never falls back to the byte tokenizer.
func loadTokenizer(dir string) (Tokenizer, string, error) {
	if dir == "" {
		return NewByteTokenizer(), "Byte-Level BPE Fallback", nil
	}
	tokFile := filepath.Join(dir, "tokenizer.json")
	hfTok, err := LoadHFTokenizer(tokFile)
	if err != nil {
		return nil, "", fmt.Errorf("load tokenizer %s: %w", tokFile, err)
	}
	return hfTok, fmt.Sprintf("HuggingFace BPE (vocab: %d)", hfTok.VocabSize()), nil
}

// loadWeights loads the checkpoint in dir. With no dir it returns the
// deterministic testing model. A checkpoint that fails to load is an error:
// it never falls back to the testing model.
func loadWeights(dir string) (*ModelWeights, *ModelConfig, error) {
	if dir == "" {
		cfg := DefaultModelConfig()
		return GenerateDeterministicWeights(&cfg), &cfg, nil
	}
	w, cfg, err := LoadModel(dir)
	if err != nil {
		return nil, nil, fmt.Errorf("load model %s: %w", dir, err)
	}
	return w, cfg, nil
}

func findDefaultModelPath() string {
	if *modelPath != "" {
		return *modelPath
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	// 1. HuggingFace cache snapshots (TinyLlama-1.1B)
	pattern := filepath.Join(home, ".cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/*")
	matches, _ := filepath.Glob(pattern)
	for _, m := range matches {
		if _, err := os.Stat(filepath.Join(m, "model.safetensors")); err == nil {
			return m
		}
	}
	// 2. Common model directories
	candidates := []string{
		filepath.Join(home, "models/TinyLlama-1.1B-Chat-v1.0"),
		"/models/TinyLlama-1.1B-Chat-v1.0",
		"/tmp/TinyLlama-1.1B-Chat-v1.0",
	}
	for _, c := range candidates {
		if _, err := os.Stat(filepath.Join(c, "model.safetensors")); err == nil {
			return c
		}
	}
	return ""
}
