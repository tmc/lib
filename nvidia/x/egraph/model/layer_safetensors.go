package model

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// RunCompiledDenseLayer executes all steps of a compiled dense layer on the
// host CPU via [cudahostemu.Run] using weights and returns the final layer
// output [batchM * cfg.HiddenSize].
func RunCompiledDenseLayer(
	ctx context.Context,
	layer *CompiledLayer,
	weights *RepackedLayerWeights,
	hiddenIn []float32,
	kCache, vCache []float32,
	pageTable []int32,
	seqLens []int32,
) ([]float32, error) {
	if layer == nil || layer.Plan == nil {
		return nil, fmt.Errorf("run compiled dense layer: nil compiled layer or plan")
	}
	if weights == nil {
		return nil, fmt.Errorf("run compiled dense layer: nil weights")
	}
	p := layer.Plan
	repackedConsumer := make(map[string]bool)
	for _, k := range p.Kernels {
		if k.Tile != nil && k.Tile.Weight == hopper.WeightNVFP4 && k.Tile.WGMMA && k.Tile.Repacked {
			for _, a := range k.Args {
				if strings.HasSuffix(a, "_q") {
					repackedConsumer[strings.TrimSuffix(a, "_q")] = true
				}
			}
		}
	}

	toPlanSpec := func(t hopper.Tensor) PlanTensorSpec {
		return PlanTensorSpec{
			Name:  t.Name,
			DType: DType(t.DType),
			Shape: t.Shape,
		}
	}
	inputs := make([]PlanTensorSpec, len(p.Inputs))
	for i, in := range p.Inputs {
		inputs[i] = toPlanSpec(in)
	}
	tempsAndOut := make([]PlanTensorSpec, 0, len(p.Temps)+1)
	for _, tmp := range p.Temps {
		tempsAndOut = append(tempsAndOut, toPlanSpec(tmp))
	}
	tempsAndOut = append(tempsAndOut, toPlanSpec(p.Output))

	cbufs, err := BuildDenseLayerBuffers(
		layer.Config,
		layer.BatchM,
		weights,
		repackedConsumer,
		inputs,
		tempsAndOut,
		hiddenIn,
		kCache,
		vCache,
		pageTable,
		seqLens,
	)
	if err != nil {
		return nil, err
	}

	var src strings.Builder
	src.WriteString(hopper.Prelude)
	seenHelpers := make(map[string]bool)
	var launches []string
	for _, k := range p.Kernels {
		switch {
		case k.GEMM != nil:
			s, err := k.GEMM.CUDA()
			if err != nil {
				return nil, err
			}
			src.WriteString(hopper.DeduplicateCUDAHelpers(s, seenHelpers))
			grid, block, _ := k.GEMM.Grid()
			launches = append(launches, fmt.Sprintf("eg_emu_launch(dim3(%d, %d), dim3(%d), 0, %s<%d>, %s)",
				grid[0], grid[1], block[0], k.GEMM.Name, k.M, strings.Join(k.Args, ", ")))
		case k.Tile != nil:
			s, err := k.Tile.CUDA()
			if err != nil {
				return nil, err
			}
			src.WriteString(hopper.DeduplicateCUDAHelpers(s, seenHelpers))
			if k.Tile.Grouped {
				grid, block := k.Tile.Grid(k.MaxRows, k.Experts)
				launches = append(launches, fmt.Sprintf("eg_emu_launch(dim3(%d, %d, %d), dim3(%d), %d, %s, %d, %s)",
					grid[0], grid[1], grid[2], block[0], k.Tile.SharedBytes(), k.Tile.Name, k.M, strings.Join(k.Args, ", ")))
			} else {
				grid, block := k.Tile.Grid(k.M, 1)
				launches = append(launches, fmt.Sprintf("eg_emu_launch(dim3(%d, %d), dim3(%d), %d, %s, %d, %s)",
					grid[0], grid[1], block[0], k.Tile.SharedBytes(), k.Tile.Name, k.M, strings.Join(k.Args, ", ")))
			}
		case k.Attention != nil:
			s, err := k.Attention.CUDA()
			if err != nil {
				return nil, err
			}
			l, err := k.Attention.Launcher()
			if err != nil {
				return nil, err
			}
			src.WriteString(s + "\n" + l)
			args := append([]string(nil), k.Args[:4]...)
			args = append(args, fmt.Sprint(k.MaxPages))
			args = append(args, k.Args[4:]...)
			launches = append(launches, fmt.Sprintf("eg_emu_sms = 16; eg_emu_blocks_per_sm = 1; %s_launch(%d, %d, %s, nullptr)",
				k.Attention.Name, k.Batch, k.MaxKeys, strings.Join(args, ", ")))
		case k.Row != nil:
			s, err := k.Row.CUDA()
			if err != nil {
				return nil, err
			}
			src.WriteString(s)
			if k.Row.Cluster {
				l, err := k.Row.Launcher()
				if err != nil {
					return nil, err
				}
				src.WriteString("\n" + l)
				launches = append(launches, fmt.Sprintf("%s_launch(%d, %s, nullptr)",
					k.Row.Name, k.Rows, strings.Join(k.Args, ", ")))
			} else {
				grid, block := k.Row.Grid(k.Rows)
				launches = append(launches, fmt.Sprintf("eg_emu_launch(dim3(%d), dim3(%d), 0, %s, %d, %s)",
					grid[0], block[0], k.Row.Name, k.Rows, strings.Join(k.Args, ", ")))
			}
		default:
			return nil, fmt.Errorf("run compiled dense layer: unsupported kernel step in plan")
		}
	}

	tmpDir, err := os.MkdirTemp("", "model-safetensors-layer-*")
	if err != nil {
		return nil, fmt.Errorf("run compiled dense layer: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	res, err := cudahostemu.Run(ctx, tmpDir, cudahostemu.Program{
		Source:  src.String(),
		Launch:  strings.Join(launches, ";\n    "),
		Buffers: cbufs,
		Flags:   strings.Fields(os.Getenv("EG_EMU_FLAGS")),
	})
	if err != nil {
		return nil, fmt.Errorf("run compiled dense layer: %w", err)
	}

	return DecodeOutputF32(DType(p.Output.DType), res[p.Output.Name]), nil
}
