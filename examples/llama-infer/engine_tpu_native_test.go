//go:build linux && tpu

package main

import (
	"context"
	"fmt"
	"github.com/tmc/lib/x/xla"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"unicode/utf8"

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
	if err == nil || !strings.Contains(err.Error(), "config.json") {
		t.Fatalf("got %v", err)
	}
}

func TestNativeTPUCheckpointTextWithCPU(t *testing.T) {
	path := os.Getenv("XLA_PJRT_CPU_PLUGIN")
	if path == "" {
		t.Skip("set XLA_PJRT_CPU_PLUGIN to a CPU PJRT library")
	}
	plugin, err := openxla.OpenPlugin(path)
	if err != nil {
		t.Fatal(err)
	}
	c, _ := pjrtFixture()
	c.VocabSize = 270
	w := GenerateDeterministicWeights(c)
	dir := t.TempDir()
	p := &pjrtCheckpoint{config: *c, weights: w, theta: float64(c.RopeTheta)}
	eng, err := newTPUModelEngine(Config{ModelPath: dir}, p, nil, plugin.Api())
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := eng.Free(); err != nil {
			t.Error(err)
		}
	}()
	// Raw IDs require no tokenizer file.
	if err := eng.GenerateFromTokenIDs(context.Background(), []int32{1}, 1, func(int32) bool { return true }); err != nil {
		t.Fatal(err)
	}
	if err := eng.Generate(context.Background(), Config{}, "hi", 1, func(int32, string) bool { return true }); err == nil {
		t.Fatal("missing tokenizer accepted")
	}
	data, err := os.ReadFile("../../x/hftokenizer/testdata/llama.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), data, 0600); err != nil {
		t.Fatal(err)
	}
	tok, err := loadPJRTTokenizer(dir, c.VocabSize)
	if err != nil {
		t.Fatal(err)
	}
	prompt, err := tok.Encode("hi")
	if err != nil {
		t.Fatal(err)
	}
	var want []int32
	if err := eng.GenerateFromTokenIDs(context.Background(), prompt, 3, func(id int32) bool { want = append(want, id); return true }); err != nil {
		t.Fatal(err)
	}
	var got []int32
	var text strings.Builder
	// Per-call paths and BOS/EOS defaults must not replace checkpoint metadata.
	cfg := Config{ModelPath: "wrong-directory", BOSTokenID: 999, EOSTokenID: 999}
	err = eng.Generate(context.Background(), cfg, "hi", 3, func(id int32, fragment string) bool {
		if !utf8.ValidString(fragment) {
			t.Errorf("invalid fragment %x", fragment)
		}
		got = append(got, id)
		text.WriteString(fragment)
		return true
	})
	if err != nil || !slices.Equal(got, want) {
		t.Fatalf("IDs %v %v; want %v", got, err, want)
	}
	whole, err := tok.Decode(got)
	if err != nil || text.String() != whole {
		t.Fatalf("text %q %v; want %q", text.String(), err, whole)
	}
	eng.checkpointEOS = []int32{want[0]}
	calls := 0
	if err := eng.Generate(context.Background(), cfg, "hi", 3, func(int32, string) bool { calls++; return true }); err != nil || calls != 1 {
		t.Fatalf("EOS calls %d: %v", calls, err)
	}
}

func TestNativeTPUWeightReport(t *testing.T) {
	path := os.Getenv("XLA_PJRT_CPU_PLUGIN")
	if path == "" {
		t.Skip("set XLA_PJRT_CPU_PLUGIN")
	}
	engine, err := xla.NewEngine(path)
	if err != nil {
		t.Fatal(err)
	}
	defer engine.Close()
	for _, tt := range []struct {
		precision string
		element   xla.ElementType
		storage   string
		perElem   int
	}{
		{"F32", xla.F32, "F32", 4},
		{"F16", xla.F16, "BF16", 2},
		{"BF16", xla.BF16, "BF16", 2},
	} {
		t.Run(tt.precision, func(t *testing.T) {
			dir, _ := writeRetainedTestCheckpoint(t, "BF16")
			p, err := loadPJRTCheckpoint(dir, 6, tt.element)
			if err != nil {
				t.Fatal(err)
			}
			model, err := p.newModel(context.Background(), engine)
			if err != nil {
				t.Fatal(err)
			}
			eng := &ModelEngine{decoder: model}
			report := eng.WeightReport()
			if report.DType != tt.precision || report.Storage != tt.storage || report.PerElem != tt.perElem || report.Bytes <= 0 {
				t.Fatalf("incorrect report: %v", report)
			}
			if eng.IsF16() != (tt.element == xla.F16) {
				t.Fatal("incorrect F16 flag")
			}
			if err := model.close(); err != nil {
				t.Fatal(err)
			}
			if eng.WeightReport().DType != "" || eng.IsF16() {
				t.Fatal("closed model reports active precision")
			}
		})
	}
	var eng *ModelEngine
	if eng.WeightReport().DType != "" || eng.IsF16() {
		t.Fatal("nil model reports active precision")
	}
}
