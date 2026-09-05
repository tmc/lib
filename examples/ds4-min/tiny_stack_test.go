package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/tmc/lib/gguf"
)

// TestTinyStackFixture replays the checked-in multi-layer fixture through the
// package implementation.
func TestTinyStackFixture(t *testing.T) {
	m := readFixture(t, "testdata/fixture.json")
	stack, err := newTinyStackFromSeed(m.Stack.Seed)
	if err != nil {
		t.Fatal(err)
	}
	replayStack(t, stack, m.Stack)
	checkStackCoverage(t, stack)
}

// TestTinyStackGGUF writes the tiny stack as a real GGUF file, reads it back
// through readGGUF, and replays the same fixture from the loaded weights. It
// also asserts that the pinned-identity gate still rejects the tiny file: the
// test path goes through the parser, never through validatePinnedProfile.
func TestTinyStackGGUF(t *testing.T) {
	m := readFixture(t, "testdata/fixture.json")
	path := filepath.Join(t.TempDir(), "tiny.gguf")
	if err := os.WriteFile(path, tinyStackGGUF(makeTinyStackWeights(m.Stack.Seed)), 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		t.Fatal(err)
	}
	if info.Size() > 1<<20 {
		t.Fatalf("tiny GGUF is %d bytes; it must stay small", info.Size())
	}
	g, err := gguf.Read(f, info.Size())
	if err != nil {
		t.Fatalf("read tiny GGUF: %v", err)
	}
	if err := validatePinnedProfile(g); err == nil {
		t.Fatal("pinned-identity gate accepted the synthetic tiny model")
	}
	stack, err := tinyStackFromGGUF(g, f)
	if err != nil {
		t.Fatalf("load tiny GGUF: %v", err)
	}
	replayStack(t, stack, m.Stack)
	checkStackCoverage(t, stack)
}

func TestTinyStackGGUFRejectsWrongShape(t *testing.T) {
	b := tinyStackGGUF(makeTinyStackWeights(0x445334))
	i := bytes.Index(b, []byte("deepseek4.block_count"))
	if i < 0 {
		t.Fatal("block count metadata missing")
	}
	put32(b[i+len("deepseek4.block_count")+4:], tinyStackLayers+1)
	g, err := gguf.Read(bytes.NewReader(b), int64(len(b)))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := tinyStackFromGGUF(g, bytes.NewReader(b)); err == nil {
		t.Fatal("loader accepted a mismatched block count")
	}
}

func replayStack(t *testing.T, stack *tinyStack, want stackPayload) {
	t.Helper()
	tokens := append(append([]int(nil), want.Prompt...), want.Decode...)
	if len(tokens) != len(want.Traces) {
		t.Fatalf("tokens = %d, traces = %d", len(tokens), len(want.Traces))
	}
	for i, token := range tokens {
		got, err := stack.Step(token)
		if err != nil {
			t.Fatalf("step %d: %v", i, err)
		}
		compareStackTrace(t, i, got, want.Traces[i])
	}
}

// checkStackCoverage fails unless the replayed sequence actually crossed every
// architectural path the stack exists to exercise.
func checkStackCoverage(t *testing.T, stack *tinyStack) {
	t.Helper()
	for i, cache := range stack.caches {
		kind, _ := tinyLayerShape(i)
		switch kind {
		case tinyRawLayer:
			if len(cache.compressed) != 0 {
				t.Fatalf("layer %d is raw but holds %d compressed rows", i, len(cache.compressed))
			}
		case tinyFineLayer:
			if len(cache.compressed) < 2 {
				t.Fatalf("layer %d emitted %d fine rows; the sequence must cross the ratio twice", i, len(cache.compressed))
			}
			if !cache.indexed || !cache.dual {
				t.Fatalf("layer %d is not the indexed dual-lane path", i)
			}
		case tinyCoarseLayer:
			if len(cache.compressed) < 2 {
				t.Fatalf("layer %d emitted %d coarse rows; the sequence must cross the ratio twice", i, len(cache.compressed))
			}
			if cache.indexed || cache.dual {
				t.Fatalf("layer %d is not the single-lane unindexed path", i)
			}
		}
	}
}

func compareStackTrace(t *testing.T, step int, got, want tinyStackTrace) {
	t.Helper()
	if got.Token != want.Token || got.Position != want.Position || len(got.Layers) != len(want.Layers) {
		t.Fatalf("step %d header differs: got %d/%d/%d want %d/%d/%d",
			step, got.Token, got.Position, len(got.Layers), want.Token, want.Position, len(want.Layers))
	}
	closeSlice(t, fmt.Sprintf("step %d logits", step), got.Logits, want.Logits)
	for i := range got.Layers {
		a, b := got.Layers[i], want.Layers[i]
		if a.Layer != b.Layer || a.Kind != b.Kind ||
			!reflect.DeepEqual(a.Compressed, b.Compressed) || !reflect.DeepEqual(a.Experts, b.Experts) {
			t.Fatalf("step %d layer %d discrete trace differs: got %+v want %+v", step, i, a, b)
		}
		for _, field := range []struct {
			name      string
			got, want []float32
		}{
			{"norm", a.Norm, b.Norm}, {"hc attention mix", a.HCAttnMix, b.HCAttnMix},
			{"hc FFN mix", a.HCFFNMix, b.HCFFNMix}, {"query", a.Query, b.Query},
			{"key", a.Key, b.Key}, {"attention", a.Attention, b.Attention},
			{"weights", a.Weights, b.Weights}, {"moe", a.MoE, b.MoE}, {"hc", a.HC, b.HC},
		} {
			closeSlice(t, fmt.Sprintf("step %d layer %d %s", step, i, field.name), field.got, field.want)
		}
	}
}

// tinyStackGGUF encodes the tiny stack as a version 3 GGUF file with float32
// tensors. It is test scaffolding: the command never writes GGUF.
func tinyStackGGUF(w tinyStackWeights) []byte {
	type tensor struct {
		name string
		dims []uint64
		data []float32
	}
	var tensors []tensor
	add := func(name string, data []float32, dims ...uint64) {
		tensors = append(tensors, tensor{name, dims, data})
	}
	add("token_embd.weight", w.embedding, tinyDim, tinyVocab)
	add("output_norm.weight", w.outputNorm, tinyDim)
	add("output.weight", w.outputHead, tinyDim, tinyVocab)
	for i, l := range w.layers {
		p := fmt.Sprintf("blk.%d.", i)
		add(p+"attn_norm.weight", l.norm, tinyDim)
		add(p+"attn_q_b.weight", l.q, tinyDim, tinyDim)
		add(p+"attn_kv.weight", l.kv, tinyDim, tinyDim)
		add(p+"attn_output_b.weight", l.o, tinyDim, tinyDim)
		add(p+"attn_sinks.weight", []float32{l.sink}, 1)
		add(p+"ffn_gate_inp.weight", l.router, tinyDim, tinyExperts)
		add(p+"exp_probs_b.bias", l.routerBias, tinyExperts)
		var gate, up, down []float32
		for _, e := range l.experts {
			gate = append(gate, e.gate...)
			up = append(up, e.up...)
			down = append(down, e.down...)
		}
		add(p+"ffn_gate_exps.weight", gate, tinyDim, tinyInter, tinyExperts)
		add(p+"ffn_up_exps.weight", up, tinyDim, tinyInter, tinyExperts)
		add(p+"ffn_down_exps.weight", down, tinyInter, tinyDim, tinyExperts)
		add(p+"hc_attn_fn.weight", l.hcAttnFn, tinyDim*tinyStreams, tinyStreams*(tinyStreams+2))
		add(p+"hc_attn_scale.weight", l.hcAttnScale, 3)
		add(p+"hc_attn_base.weight", l.hcAttnBase, tinyStreams*(tinyStreams+2))
		add(p+"hc_ffn_fn.weight", l.hcFFNFn, tinyDim*tinyStreams, tinyStreams*(tinyStreams+2))
		add(p+"hc_ffn_scale.weight", l.hcFFNScale, 3)
		add(p+"hc_ffn_base.weight", l.hcFFNBase, tinyStreams*(tinyStreams+2))
		if l.kind == tinyRawLayer {
			continue
		}
		width := uint64(tinyDim)
		if l.kind == tinyFineLayer {
			width *= 2
		}
		add(p+"attn_compressor_kv.weight", l.attention.kv, tinyDim, width)
		add(p+"attn_compressor_gate.weight", l.attention.gate, tinyDim, width)
		add(p+"attn_compressor_ape.weight", l.attention.ape, width, uint64(l.ratio))
		add(p+"attn_compressor_norm.weight", l.attention.norm, tinyDim)
		if l.kind != tinyFineLayer {
			continue
		}
		add(p+"indexer_compressor_kv.weight", l.indexer.kv, tinyDim, width)
		add(p+"indexer_compressor_gate.weight", l.indexer.gate, tinyDim, width)
		add(p+"indexer_compressor_ape.weight", l.indexer.ape, width, uint64(l.ratio))
		add(p+"indexer_compressor_norm.weight", l.indexer.norm, tinyDim)
		add(p+"indexer.attn_q_b.weight", l.indexQuery, tinyDim, tinyDim)
		add(p+"indexer.proj.weight", l.indexWeight, tinyDim)
	}

	type meta struct {
		key   string
		value uint32
	}
	numbers := []meta{
		{"deepseek4.block_count", uint32(len(w.layers))},
		{"deepseek4.embedding_length", tinyDim},
		{"deepseek4.vocab_size", tinyVocab},
		{"deepseek4.expert_count", tinyExperts},
		{"deepseek4.expert_used_count", tinyTopK},
		{"deepseek4.feed_forward_length", tinyInter},
		{"deepseek4.hyper_connection.stream_count", tinyStreams},
		{"deepseek4.hyper_connection.sinkhorn_iterations", tinySinkhornIterations},
		{"deepseek4.attention.raw_window", tinyStackRawWindow},
		{"deepseek4.attention.compression_ratio", tinyStackFineRatio},
		{"deepseek4.attention.coarse_compression_ratio", tinyStackCoarseRatio},
		{"deepseek4.attention.selected_row_cap", tinyStackRowCap},
		{"general.alignment", 32},
	}

	var b bytes.Buffer
	b.WriteString("GGUF")
	write32(&b, 3)
	write64(&b, uint64(len(tensors)))
	write64(&b, uint64(len(numbers)+1))
	writeString(&b, "general.architecture")
	write32(&b, 8)
	writeString(&b, "deepseek4")
	for _, m := range numbers {
		writeString(&b, m.key)
		write32(&b, 4)
		write32(&b, m.value)
	}
	offset := uint64(0)
	for _, t := range tensors {
		writeString(&b, t.name)
		write32(&b, uint32(len(t.dims)))
		for _, d := range t.dims {
			write64(&b, d)
		}
		write32(&b, 0) // float32
		write64(&b, offset)
		offset = (offset + uint64(len(t.data))*4 + 31) &^ 31
	}
	for b.Len()%32 != 0 {
		b.WriteByte(0)
	}
	base := b.Len()
	b.Write(make([]byte, offset))
	out := b.Bytes()
	offset = 0
	for _, t := range tensors {
		for i, v := range t.data {
			binary.LittleEndian.PutUint32(out[base+int(offset)+i*4:], math.Float32bits(v))
		}
		offset = (offset + uint64(len(t.data))*4 + 31) &^ 31
	}
	return out
}

// GGUF little-endian writers, used to build the tiny model file in memory.
func writeString(b *bytes.Buffer, s string) { write64(b, uint64(len(s))); b.WriteString(s) }
func write32(b *bytes.Buffer, n uint32)     { var p [4]byte; put32(p[:], n); b.Write(p[:]) }
func write64(b *bytes.Buffer, n uint64)     { var p [8]byte; put64(p[:], n); b.Write(p[:]) }
func put32(b []byte, n uint32)              { binary.LittleEndian.PutUint32(b, n) }
func put64(b []byte, n uint64)              { binary.LittleEndian.PutUint64(b, n) }
