//go:build linux

package main

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func TestCUDAProjectionFamilyCeilingProbe(t *testing.T) {
	path := os.Getenv("LLAMA_MODEL_PATH")
	if path == "" || os.Getenv("LLAMA_CUDA_PROJECTION_PROBE") != "1" {
		t.Skip("CUDA projection probe is disabled")
	}
	cfg := Llama3_2_1BConfig(path)
	cfg.MaxSeqLen = 8
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	sp, err := newGPUBatchScratchpad(eng.dev, eng.memory, &eng.mCfg, 8)
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
	layer := &eng.weights.Layers[0]
	model := &eng.mCfg
	hostInput := make([]float32, sp.NormX.Len())
	for i := range hostInput {
		hostInput[i] = float32(i%29-14) / 17
	}
	if err := sp.NormX.CopyFromHost(hostInput); err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name          string
		input, weight cudamem.Slice[float32]
		output        cudamem.Slice[float32]
		inputDim      int
		outputDim     int
		matrix        bool
	}{
		{"q", sp.NormX, layer.WQ, sp.Q, model.HiddenSize, model.NumAttentionHeads * model.HeadDim(), false},
		{"k", sp.NormX, layer.WK, sp.K, model.HiddenSize, model.NumKeyValueHeads * model.HeadDim(), false},
		{"v", sp.NormX, layer.WV, sp.V, model.HiddenSize, model.NumKeyValueHeads * model.HeadDim(), false},
		{"output", sp.AttnOut, layer.WO, sp.FFNOut, model.HiddenSize, model.HiddenSize, false},
		{"gate", sp.NormX, layer.WGate, sp.Gate, model.HiddenSize, model.IntermediateSize, true},
		{"up", sp.NormX, layer.WUp, sp.Up, model.HiddenSize, model.IntermediateSize, true},
		{"down", sp.FFNAct, layer.WDown, sp.FFNOut, model.IntermediateSize, model.HiddenSize, false},
	}
	for _, batch := range []int{2, 4, 8} {
		for _, test := range tests {
			launch := func() error {
				return launcher.Grid(uint32(test.outputDim), uint32(batch)).Block(128).
					Launch(test.input, test.weight, noBias, test.output, int32(test.inputDim), int32(test.outputDim), int32(batch))
			}
			for range 5 {
				if err := launch(); err != nil {
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
				if err := launch(); err != nil {
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
			weightBytes := int64(batch) * int64(test.inputDim) * int64(test.outputDim) * 4
			t.Logf("scalar_projection family=%s batch=%d iterations=%d ns_per_op=%d weight_bytes_per_op=%d effective_weight_GBps=%.3f", test.name, batch, iterations, (d / time.Duration(iterations)).Nanoseconds(), weightBytes, float64(weightBytes*iterations)/d.Seconds()/1e9)
			if !test.matrix {
				continue
			}
			scalarOutput := make([]float32, batch*test.outputDim)
			if err := test.output.Slice(0, len(scalarOutput)).CopyToHost(scalarOutput); err != nil {
				t.Fatal(err)
			}
			if err := sp.lmHead.runPlanSet(sp.gateUp, eng.stream, test.input, test.weight, test.output, batch); err != nil {
				t.Fatal(err)
			}
			if err := eng.stream.Synchronize(context.Background()); err != nil {
				t.Fatal(err)
			}
			matrixOutput := make([]float32, len(scalarOutput))
			if err := test.output.Slice(0, len(matrixOutput)).CopyToHost(matrixOutput); err != nil {
				t.Fatal(err)
			}
			checkCUDAFloatSlice(t, "matrix "+test.name, matrixOutput, scalarOutput, 2e-3)
			for range 5 {
				if err := sp.lmHead.runPlanSet(sp.gateUp, eng.stream, test.input, test.weight, test.output, batch); err != nil {
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
				if err := sp.lmHead.runPlanSet(sp.gateUp, eng.stream, test.input, test.weight, test.output, batch); err != nil {
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
			t.Logf("matrix_projection family=%s batch=%d iterations=%d ns_per_op=%d scalar_to_matrix=%.3fx", test.name, batch, iterations, (matrixDuration / time.Duration(iterations)).Nanoseconds(), float64(d)/float64(matrixDuration))
		}
	}
}

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
	sp, err := newGPUBatchScratchpad(eng.dev, eng.memory, &eng.mCfg, 8)
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
