//go:build linux && tpu

package main

import (
	"context"
	"fmt"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/openxla"
)

// This exercises the production TPU ModelEngine with a CPU PJRT API. It
// qualifies the Go adapter methods, not libtpu or physical accelerator behavior.
func TestNativeTPUModelEngineWithCPU(t *testing.T) {
	path := os.Getenv("XLA_PJRT_CPU_PLUGIN")
	if path == "" {
		t.Skip("set XLA_PJRT_CPU_PLUGIN to a CPU PJRT library")
	}
	plugin, err := openxla.OpenPlugin(path)
	if err != nil {
		t.Fatal(err)
	}
	cfg := DefaultTinyLlamaConfig()
	cfg.Dim = 8
	cfg.HiddenDim = 11
	cfg.NumHeads = 2
	cfg.NumKVHeads = 1
	cfg.HeadDim = 4
	cfg.MaxSeqLen = 8
	p, tokenizer, err := preparePJRTModel(cfg)
	if err != nil {
		t.Fatal(err)
	}
	eng, err := newTPUModelEngine(cfg, p, tokenizer, plugin.Api())
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := eng.Free(); err != nil {
			t.Error(err)
		}
	}()
	kv := NewKVCacheState(p.config.NumHiddenLayers, p.config.MaxSeqLen, p.config.NumKeyValueHeads*p.config.HeadDim())
	var logits []float32
	for pos, id := range []int32{'h', 'i'} {
		logits = Forward(p.weights, &p.config, kv, id, pos)
	}
	var want []int32
	for i := 0; i < 3; i++ {
		id := sampleArgmax(logits)
		want = append(want, id)
		if i < 2 {
			logits = Forward(p.weights, &p.config, kv, id, 2+i)
		}
	}
	got, err := eng.GenerateTokenIDs(context.Background(), cfg, "hi", 3)
	if err != nil || !slices.Equal(got, want) {
		t.Fatalf("text request %v %v, want %v", got, err, want)
	}
	if _, err := eng.ReadLogits(); err != nil {
		t.Fatal(err)
	}
	if err := eng.ResetKV(); err != nil {
		t.Fatal(err)
	}
	if _, err := eng.ReadLogits(); err == nil {
		t.Fatal("stale logits after reset")
	}
	got = nil
	err = eng.GenerateFromTokenIDs(context.Background(), []int32{'h', 'i'}, 3, func(id int32) bool { got = append(got, id); return true })
	if err != nil || !slices.Equal(got, want) {
		t.Fatalf("raw request %v %v, want %v", got, err, want)
	}
	eng.tokenizer = nil
	if err := eng.Generate(context.Background(), cfg, "hi", 3, func(int32, string) bool { return true }); err == nil || !strings.Contains(err.Error(), "tokenization is unsupported") {
		t.Fatalf("checkpoint text fallback: %v", err)
	}
	if err := eng.Free(); err != nil {
		t.Fatal(err)
	}
	if _, err := eng.ReadLogits(); err == nil {
		t.Fatal("freed engine exposed logits")
	}
	if err := eng.GenerateFromTokenIDs(context.Background(), []int32{1}, 1, func(int32) bool { return true }); err == nil {
		t.Fatal("freed engine generated tokens")
	}
}

func ExampleModelEngine_GenerateFromTokenIDs() {
	var eng *ModelEngine
	err := eng.GenerateFromTokenIDs(context.Background(), []int32{1, 42}, 4, func(id int32) bool {
		fmt.Println(id)
		return true
	})
	fmt.Println(err)
	// Output: tpu: engine is closed
}

func TestTPUCheckpointTextRejectedBeforeLoading(t *testing.T) {
	err := runPlatform(Config{ModelPath: "nonexistent-checkpoint"})
	if err == nil || !strings.Contains(err.Error(), "tokenization is unsupported") {
		t.Fatalf("got %v", err)
	}
}
