//go:build darwin

package main

import (
	"math"
	"reflect"
	"sync"
	"testing"

	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

func TestBatchedModelTailMatchesCPUAndSequentialGPU(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	hidden := tailTestHidden(8, cfg.Dim)
	for _, b := range []int{1, 2, 4, 8} {
		tail, err := eng.NewBatchedModelTail(cfg, 8)
		if err != nil {
			t.Fatal(err)
		}
		got, err := tail.Run(eng, cfg, hidden[:b*cfg.Dim], make([]SamplerConfig, b))
		if err != nil {
			t.Fatal(err)
		}
		if got.Dispatches != 3 {
			t.Fatalf("B=%d dispatches=%d, want 3", b, got.Dispatches)
		}
		wantNorm, wantLogits, wantTokens := cpuModelTail(eng, cfg, hidden[:b*cfg.Dim])
		checkClose(t, "tail normalized", got.Normalized, wantNorm, 2e-4)
		checkClose(t, "tail logits", got.Logits, wantLogits, 5e-4)
		if !reflect.DeepEqual(got.Tokens, wantTokens) {
			t.Fatalf("B=%d tokens=%v, CPU=%v", b, got.Tokens, wantTokens)
		}
		for row := range b {
			logits, token := sequentialGPUTail(t, eng, cfg, hidden[row*cfg.Dim:(row+1)*cfg.Dim])
			checkClose(t, "sequential GPU logits", got.Logits[row*cfg.VocabSize:(row+1)*cfg.VocabSize], logits, 1e-6)
			if got.Tokens[row] != token {
				t.Fatalf("B=%d row=%d token=%d, sequential GPU=%d", b, row, got.Tokens[row], token)
			}
		}
		tail.Free()
		tail.Free()
		if _, err := tail.Run(eng, cfg, hidden[:cfg.Dim], nil); err == nil {
			t.Fatal("Run after Free succeeded")
		}
	}
}

func TestBatchedModelTailLargeVocabAndCompaction(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	cfg.Name = "TinyLlama-Large-Vocab-Tail-Test"
	cfg.VocabSize = 128256
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	tail, err := eng.NewBatchedModelTail(cfg, 8)
	if err != nil {
		t.Fatal(err)
	}
	defer tail.Free()
	hidden := tailTestHidden(8, cfg.Dim)
	all, err := tail.Run(eng, cfg, hidden, nil)
	if err != nil {
		t.Fatal(err)
	}
	activeRows := []int{7, 2, 5}
	compacted := make([]float32, 0, len(activeRows)*cfg.Dim)
	for _, row := range activeRows {
		compacted = append(compacted, hidden[row*cfg.Dim:(row+1)*cfg.Dim]...)
	}
	active, err := tail.Run(eng, cfg, compacted, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i, row := range activeRows {
		if active.Tokens[i] != all.Tokens[row] {
			t.Fatalf("compacted row %d token=%d, full=%d", row, active.Tokens[i], all.Tokens[row])
		}
		checkClose(t, "compacted logits", active.Logits[i*cfg.VocabSize:(i+1)*cfg.VocabSize], all.Logits[row*cfg.VocabSize:(row+1)*cfg.VocabSize], 1e-6)
	}
}

func TestBatchedModelTailRejectsUnsupportedSamplers(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	tail, err := eng.NewBatchedModelTail(cfg, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer tail.Free()
	hidden := tailTestHidden(2, cfg.Dim)
	tests := []struct {
		name     string
		samplers []SamplerConfig
	}{
		{name: "count", samplers: []SamplerConfig{{Method: "argmax"}}},
		{name: "temperature", samplers: []SamplerConfig{{Method: "argmax"}, {Method: "argmax", Temperature: 0.7}}},
		{name: "method", samplers: []SamplerConfig{{Method: "argmax"}, {Method: "top-p"}}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := tail.Run(eng, cfg, hidden, test.samplers); err == nil {
				t.Fatal("unsupported sampler succeeded")
			}
		})
	}
}

func TestBatchedModelTailConcurrentLifetime(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	tail, err := eng.NewBatchedModelTail(cfg, 8)
	if err != nil {
		t.Fatal(err)
	}
	hidden := tailTestHidden(8, cfg.Dim)
	want := cpuModelTailTokens(eng, cfg, hidden)
	var wg sync.WaitGroup
	for i := range 8 {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			got, err := tail.Run(eng, cfg, hidden, nil)
			if err != nil {
				t.Errorf("worker %d: %v", i, err)
				return
			}
			if !reflect.DeepEqual(got.Tokens, want) {
				t.Errorf("worker %d tokens=%v, want %v", i, got.Tokens, want)
			}
		}(i)
	}
	wg.Wait()
	tail.Free()
}

func cpuModelTailTokens(eng *ModelEngine, cfg Config, hidden []float32) []int32 {
	_, _, tokens := cpuModelTail(eng, cfg, hidden)
	return tokens
}

func tailTestHidden(rows, dim int) []float32 {
	x := make([]float32, rows*dim)
	for i := range x {
		x[i] = float32(math.Sin(float64(i+1)*0.173) + math.Cos(float64(i+7)*0.071))
	}
	return x
}

func cpuModelTail(eng *ModelEngine, cfg Config, hidden []float32) ([]float32, []float32, []int32) {
	b := len(hidden) / cfg.Dim
	norm := make([]float32, len(hidden))
	logits := make([]float32, b*cfg.VocabSize)
	tokens := make([]int32, b)
	for row := range b {
		x := hidden[row*cfg.Dim : (row+1)*cfg.Dim]
		var squares float64
		for _, value := range x {
			squares += float64(value * value)
		}
		scale := float32(1 / math.Sqrt(squares/float64(cfg.Dim)+float64(cfg.Eps)))
		for i, value := range x {
			norm[row*cfg.Dim+i] = value * scale * eng.finalNormW.HostSlice()[i]
		}
		projectCPU(norm[row*cfg.Dim:(row+1)*cfg.Dim], eng.lmHeadW.HostSlice(), logits[row*cfg.VocabSize:(row+1)*cfg.VocabSize], cfg.Dim)
		tokens[row] = sampleArgmax(logits[row*cfg.VocabSize : (row+1)*cfg.VocabSize])
	}
	return norm, logits, tokens
}

func sequentialGPUTail(t *testing.T, eng *ModelEngine, cfg Config, hidden []float32) ([]float32, int32) {
	t.Helper()
	x, err := metalmem.Alloc[float32](eng.dev, cfg.Dim)
	if err != nil {
		t.Fatal(err)
	}
	defer x.Free()
	norm, err := metalmem.Alloc[float32](eng.dev, cfg.Dim)
	if err != nil {
		t.Fatal(err)
	}
	defer norm.Free()
	logits, err := metalmem.Alloc[float32](eng.dev, cfg.VocabSize)
	if err != nil {
		t.Fatal(err)
	}
	defer logits.Free()
	token, err := metalmem.Alloc[int32](eng.dev, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer token.Free()
	copy(x.HostSlice(), hidden)
	cb := eng.dev.Queue().CommandBuffer()
	ce := cb.ComputeCommandEncoder()
	dim, vocab, eps := uint32(cfg.Dim), uint32(cfg.VocabSize), cfg.Eps
	ce.SetComputePipelineState(eng.pso.Norm)
	ce.SetBufferWithOffsetAtIndex(x.Buffer(), 0, 0)
	ce.SetBufferWithOffsetAtIndex(eng.finalNormW.Buffer(), 0, 1)
	ce.SetBufferWithOffsetAtIndex(norm.Buffer(), 0, 2)
	metal.SetScalar(ce, dim, 3)
	metal.SetScalar(ce, eps, 4)
	ce.DispatchThreadgroupsThreadsPerThreadgroup(metal.Threads(1), metal.Threads(128))
	ce.SetComputePipelineState(eng.pso.GEMV)
	ce.SetBufferWithOffsetAtIndex(norm.Buffer(), 0, 0)
	ce.SetBufferWithOffsetAtIndex(eng.lmHeadW.Buffer(), 0, 1)
	ce.SetBufferWithOffsetAtIndex(logits.Buffer(), 0, 2)
	metal.SetScalar(ce, dim, 3)
	metal.SetScalar(ce, vocab, 4)
	ce.DispatchThreadgroupsThreadsPerThreadgroup(metal.Threads(cfg.VocabSize), metal.Threads(128))
	ce.SetComputePipelineState(eng.pso.Sample)
	ce.SetBufferWithOffsetAtIndex(logits.Buffer(), 0, 0)
	ce.SetBufferWithOffsetAtIndex(token.Buffer(), 0, 1)
	metal.SetScalar(ce, vocab, 2)
	ce.DispatchThreadsThreadsPerThreadgroup(metal.Threads(1), metal.Threads(1))
	ce.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()
	return append([]float32(nil), logits.HostSlice()...), token.HostSlice()[0]
}
