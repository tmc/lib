//go:build linux

package main

import (
	"context"
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func TestCUDALMHeadMatrixCeilingProbe(t *testing.T) {
	path := os.Getenv("LLAMA_MODEL_PATH")
	if path == "" || os.Getenv("LLAMA_CUDA_MATRIX_PROBE") != "1" {
		t.Skip("CUDA matrix probe is disabled")
	}
	cfg := Llama3_2_1BConfig(path)
	cfg.MaxSeqLen = 8
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	sp, err := newGPUBatchScratchpad(eng.dev, &eng.mCfg, 8)
	if err != nil {
		t.Fatal(err)
	}
	defer sp.Free()
	start, err := eng.dev.NewEvent()
	if err != nil {
		t.Fatal(err)
	}
	defer start.Destroy()
	end, err := eng.dev.NewEvent()
	if err != nil {
		t.Fatal(err)
	}
	defer end.Destroy()
	launcher := cudalaunch.Kernel(eng.modules.BatchGEMV).On(eng.stream)
	var noBias cudamem.Slice[float32]
	hostInput := make([]float32, sp.NormX.Len())
	for i := range hostInput {
		hostInput[i] = float32(i%29-14) / 17
	}
	if err := sp.NormX.CopyFromHost(hostInput); err != nil {
		t.Fatal(err)
	}
	for _, batch := range []int{2, 4, 8} {
		if err := launcher.Grid(uint32(eng.mCfg.VocabSize), uint32(batch)).Block(128).Launch(sp.NormX, eng.weights.LMHead, noBias, sp.Logits, int32(eng.mCfg.HiddenSize), int32(eng.mCfg.VocabSize), int32(batch)); err != nil {
			t.Fatal(err)
		}
		if err := eng.stream.Synchronize(context.Background()); err != nil {
			t.Fatal(err)
		}
		scalarOutput := make([]float32, batch*eng.mCfg.VocabSize)
		if err := sp.Logits.Slice(0, len(scalarOutput)).CopyToHost(scalarOutput); err != nil {
			t.Fatal(err)
		}
		if err := sp.lmHead.run(eng.stream, sp.NormX, eng.weights.LMHead, sp.Logits, batch); err != nil {
			t.Fatal(err)
		}
		if err := eng.stream.Synchronize(context.Background()); err != nil {
			t.Fatal(err)
		}
		matrixOutput := make([]float32, len(scalarOutput))
		if err := sp.Logits.Slice(0, len(matrixOutput)).CopyToHost(matrixOutput); err != nil {
			t.Fatal(err)
		}
		checkCUDAFloatSlice(t, "matrix LM head", matrixOutput, scalarOutput, 2e-3)
		for range 5 {
			if err := launcher.Grid(uint32(eng.mCfg.VocabSize), uint32(batch)).Block(128).Launch(sp.NormX, eng.weights.LMHead, noBias, sp.Logits, int32(eng.mCfg.HiddenSize), int32(eng.mCfg.VocabSize), int32(batch)); err != nil {
				t.Fatal(err)
			}
		}
		if err := eng.stream.Synchronize(context.Background()); err != nil {
			t.Fatal(err)
		}
		if err := start.Record(eng.stream); err != nil {
			t.Fatal(err)
		}
		const iterations = 20
		for range iterations {
			if err := launcher.Grid(uint32(eng.mCfg.VocabSize), uint32(batch)).Block(128).Launch(sp.NormX, eng.weights.LMHead, noBias, sp.Logits, int32(eng.mCfg.HiddenSize), int32(eng.mCfg.VocabSize), int32(batch)); err != nil {
				t.Fatal(err)
			}
		}
		if err := end.Record(eng.stream); err != nil {
			t.Fatal(err)
		}
		if err := end.Synchronize(context.Background()); err != nil {
			t.Fatal(err)
		}
		d, err := end.ElapsedSince(start)
		if err != nil {
			t.Fatal(err)
		}
		weightBytes := int64(batch) * int64(eng.mCfg.HiddenSize) * int64(eng.mCfg.VocabSize) * 4
		t.Logf("scalar_lm_head batch=%d iterations=%d ns_per_op=%d weight_bytes_per_op=%d effective_weight_GBps=%.3f", batch, iterations, d.Nanoseconds()/iterations, weightBytes, float64(weightBytes*iterations)/d.Seconds()/1e9)
		for range 5 {
			if err := sp.lmHead.run(eng.stream, sp.NormX, eng.weights.LMHead, sp.Logits, batch); err != nil {
				t.Fatal(err)
			}
		}
		if err := eng.stream.Synchronize(context.Background()); err != nil {
			t.Fatal(err)
		}
		if err := start.Record(eng.stream); err != nil {
			t.Fatal(err)
		}
		for range iterations {
			if err := sp.lmHead.run(eng.stream, sp.NormX, eng.weights.LMHead, sp.Logits, batch); err != nil {
				t.Fatal(err)
			}
		}
		if err := end.Record(eng.stream); err != nil {
			t.Fatal(err)
		}
		if err := end.Synchronize(context.Background()); err != nil {
			t.Fatal(err)
		}
		matrixDuration, err := end.ElapsedSince(start)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("matrix_lm_head batch=%d iterations=%d ns_per_op=%d scalar_to_matrix=%.3fx", batch, iterations, matrixDuration.Nanoseconds()/iterations, float64(d)/float64(matrixDuration))
	}
}
