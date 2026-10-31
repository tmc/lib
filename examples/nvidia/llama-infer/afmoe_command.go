package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/cuda"
)

// afmoeRunOptions is separate from the dense decoder's flags. The AFMoE
// command requires LLAMA_AFMOE_NATIVE_ENABLE=1; default support stays closed.
type afmoeRunOptions struct {
	Model, TokenIDs string
	Steps, Context  int
	Scatter         string
	Warmup          bool
}

func runAfmoeMini(opt afmoeRunOptions) (rerr error) {
	if opt.Model == "" || opt.TokenIDs == "" || opt.Steps < 1 || opt.Context < 1 || opt.Context > 12032 {
		return fmt.Errorf("afmoe run: requires model, explicit token IDs, positive steps and context <= 12032")
	}
	if opt.Scatter != "bf16-expert-order" && opt.Scatter != "fp32-then-bf16" {
		return fmt.Errorf("afmoe run: explicitly select bf16-expert-order or fp32-then-bf16 scatter")
	}
	model, err := filepath.Abs(opt.Model)
	if err != nil {
		return err
	}
	ck, cfg, err := openAfmoeCheckpoint(model)
	if ck != nil {
		defer closeInto(&rerr, ck, "close afmoe checkpoint")
	}
	if err != nil {
		return err
	}
	prompt, err := parseTokenIDs(opt.TokenIDs, cfg.VocabSize)
	if err != nil {
		return fmt.Errorf("afmoe prompt: %w", err)
	}
	if len(prompt) == 0 || opt.Context > cfg.MaxSeqLen || len(prompt) > opt.Context || opt.Steps-1 > opt.Context-len(prompt) {
		return fmt.Errorf("afmoe run: prompt and output count exceed context")
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		return fmt.Errorf("afmoe requires CUDA: %w", err)
	}
	var weights *afmoeWeights
	var decoder *afmoeDecoder
	defer func() {
		// A failed drain or retained table/module owner must not be followed
		// by releasing its weights or context. Report the failure as HOLD.
		if decoder != nil {
			if err := decoder.Close(); err != nil {
				rerr = errors.Join(rerr, errReleasesHeld, fmt.Errorf("close afmoe decoder: %w", err))
				return
			}
		}
		if weights != nil {
			if err := weights.Free(); err != nil {
				rerr = errors.Join(rerr, errReleasesHeld, fmt.Errorf("free afmoe weights: %w", err))
				return
			}
		}
		rerr = errors.Join(rerr, dev.Close())
	}()
	if weights, err = streamAfmoeWeights(dev, ck, cfg); err != nil {
		return err
	}
	if decoder, err = newAfmoeDecoder(dev, cfg, weights, opt.Context, opt.Scatter == "fp32-then-bf16"); err != nil {
		return err
	}
	fmt.Printf("[Model Loader] Loaded real weights from %s (layers: %d, hidden: %d)\n", model, cfg.NumHiddenLayers, cfg.HiddenSize)
	fmt.Printf("Tokens    : %v\nGenerating: %d tokens (greedy, no EOS stop)\n", prompt, opt.Steps)
	fmt.Printf("AFMOE_MODE: tokenwise eager BF16 values; physical KV=f32; scatter=%s; context=%d\n", opt.Scatter, opt.Context)
	fmt.Printf("AFMOE_TIMING: CPU wall input staging through synchronization and scalar host ID; predrain excluded; no GPU event intervals\n")
	if opt.Warmup {
		if _, _, _, err := generateAfmoe(prompt, opt.Steps, opt.Context, decoder.drain, decoder.forward); err != nil {
			return fmt.Errorf("afmoe warmup: %w", err)
		}
		if err := decoder.reset(); err != nil {
			return fmt.Errorf("afmoe reset after warmup: %w", err)
		}
		fmt.Printf("AFMOE_WARMUP: complete prompt=%d outputs=%d KV-reset=physical-zero outside-timing\n", len(prompt), opt.Steps)
	} else {
		fmt.Println("AFMOE_WARMUP: disabled (cold first forward)")
	}
	fmt.Printf("Ready Time   : %v after startup\n", time.Since(processStart))
	ids, prefill, decode, gerr := generateAfmoe(prompt, opt.Steps, opt.Context, decoder.drain, decoder.forward)
	fmt.Printf("GEN_IDS: %v\n", ids)
	if gerr != nil {
		return gerr
	}
	fmt.Printf("Prefill Time : %v (%d prompt tokens, sequential)\n", prefill, len(prompt))
	fmt.Printf("STEP 0 wall_ns=%d\n", prefill.Nanoseconds())
	for i, d := range decode {
		fmt.Printf("STEP %d wall_ns=%d\n", i+1, d.Nanoseconds())
	}
	return nil
}

func routeAfmoeFromFlags() (bool, error) {
	enabled := os.Getenv("LLAMA_AFMOE_NATIVE_ENABLE")
	if enabled != "" && enabled != "0" && enabled != "1" {
		return true, fmt.Errorf("LLAMA_AFMOE_NATIVE_ENABLE must be 0 or 1")
	}
	if *modelPath == "" {
		if enabled == "1" || *afmoeScatterFlag != "" {
			return true, fmt.Errorf("afmoe requires explicit -model")
		}
		return false, nil
	}
	data, err := os.ReadFile(filepath.Join(*modelPath, "config.json"))
	if err != nil {
		if enabled == "1" {
			return true, fmt.Errorf("afmoe config: %w", err)
		}
		return false, nil
	}
	var config struct {
		ModelType string `json:"model_type"`
	}
	if err := json.Unmarshal(data, &config); err != nil {
		if enabled == "1" {
			return true, fmt.Errorf("afmoe config: %w", err)
		}
		return false, nil
	}
	if config.ModelType != "afmoe" {
		if enabled == "1" || *afmoeScatterFlag != "" {
			return true, fmt.Errorf("AFMoE mode requires model_type=afmoe")
		}
		return false, nil
	}
	if enabled != "1" {
		return true, fmt.Errorf("afmoe runtime is gated: set LLAMA_AFMOE_NATIVE_ENABLE=1 only after native conformance admission")
	}
	allowed := map[string]bool{"model": true, "token-ids": true, "steps": true, "max-context": true, "weight-dtype": true, "warmup": true, "afmoe-scatter": true, "temp": true}
	var unsupported []string
	flag.Visit(func(f *flag.Flag) {
		if !allowed[f.Name] {
			unsupported = append(unsupported, "-"+f.Name)
		}
	})
	if len(unsupported) > 0 {
		return true, fmt.Errorf("afmoe mode does not support %s", strings.Join(unsupported, ", "))
	}
	if *weightDTypeFlag != "bf16" || *tempFlag != 0 {
		return true, fmt.Errorf("afmoe mode requires -weight-dtype=bf16 and greedy temperature zero")
	}
	return true, runAfmoeMini(afmoeRunOptions{Model: *modelPath, TokenIDs: *tokenIDsFlag, Steps: *stepsFlag, Context: *maxContextFlag, Scatter: *afmoeScatterFlag, Warmup: *warmUpFlag})
}
