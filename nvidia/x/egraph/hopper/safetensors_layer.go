package hopper

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
	"github.com/tmc/lib/nvidia/x/egraph/model"
)

// LayerSafetensorsSpec aliases [model.LayerSafetensorsSpec].
type LayerSafetensorsSpec = model.LayerSafetensorsSpec

// RepackedLayerWeights holds extracted and hardware-repacked weight and scale
// buffers for a single transformer decoder layer loaded from a safetensors archive.
type RepackedLayerWeights struct {
	Prefix        string
	WeightDType   DType
	RepackedNVFP4 bool
	InputNorm     []uint16 // BF16 [HiddenSize]
	QKVWeight     []byte   // BF16, E4M3, or repacked NVFP4 [QKV_N, HiddenSize]
	QKVScale      []byte   // optional E4M3 block scales for NVFP4
	OWeight       []byte   // [HiddenSize, NumAttentionHeads * HeadDim]
	OScale        []byte
	PostAttnNorm  []uint16 // BF16 [HiddenSize]
	// Dense MLP weights:
	GateUpWeight []byte // [2 * IntermediateSize, HiddenSize]
	GateUpScale  []byte
	DownWeight   []byte // [HiddenSize, IntermediateSize]
	DownScale    []byte
}

func (w *RepackedLayerWeights) toModelWeights() *model.PackedLayerWeights {
	if w == nil {
		return nil
	}
	return &model.PackedLayerWeights{
		Prefix:        w.Prefix,
		WeightDType:   model.DType(w.WeightDType),
		RepackedNVFP4: w.RepackedNVFP4,
		InputNorm:     w.InputNorm,
		QKVWeight:     w.QKVWeight,
		QKVScale:      w.QKVScale,
		OWeight:       w.OWeight,
		OScale:        w.OScale,
		PostAttnNorm:  w.PostAttnNorm,
		GateUpWeight:  w.GateUpWeight,
		GateUpScale:   w.GateUpScale,
		DownWeight:    w.DownWeight,
		DownScale:     w.DownScale,
	}
}

func fromModelWeights(mw *model.PackedLayerWeights) *RepackedLayerWeights {
	if mw == nil {
		return nil
	}
	return &RepackedLayerWeights{
		Prefix:        mw.Prefix,
		WeightDType:   DType(mw.WeightDType),
		RepackedNVFP4: mw.RepackedNVFP4,
		InputNorm:     mw.InputNorm,
		QKVWeight:     mw.QKVWeight,
		QKVScale:      mw.QKVScale,
		OWeight:       mw.OWeight,
		OScale:        mw.OScale,
		PostAttnNorm:  mw.PostAttnNorm,
		GateUpWeight:  mw.GateUpWeight,
		GateUpScale:   mw.GateUpScale,
		DownWeight:    mw.DownWeight,
		DownScale:     mw.DownScale,
	}
}

// RepackNVFP4Weights losslessly repacks an [n, k] NVFP4 weight matrix into
// Hopper WGMMA 2-word bitopt dequant order.
func RepackNVFP4Weights(packed []byte, n, k int) ([]byte, error) {
	return model.RepackNVFP4Weights(packed, n, k)
}

// UnpackNVFP4Weights reverses [RepackNVFP4Weights], recovering the original
// [n, k/2] packed E2M1 bytes bit-for-bit.
func UnpackNVFP4Weights(repacked []byte, n, k int) ([]byte, error) {
	return model.UnpackNVFP4Weights(repacked, n, k)
}

// RepackNVFP4Scales losslessly repacks an [n, k/16] E4M3 block-scale matrix
// into Hopper WGMMA per-thread scale layout.
func RepackNVFP4Scales(scales []byte, n, k int) ([]byte, error) {
	return model.RepackNVFP4Scales(scales, n, k)
}

// UnpackNVFP4Scales reverses [RepackNVFP4Scales], recovering the original
// [n, k/16] E4M3 block-scale bytes bit-for-bit.
func UnpackNVFP4Scales(repacked []byte, n, k int) ([]byte, error) {
	return model.UnpackNVFP4Scales(repacked, n, k)
}

// LoadLayerSafetensors parses stData using nvidia/x/safetensors, extracts the
// weights and scales of layerPrefix, and optionally repacks NVFP4 weights and
// E4M3 block scales for Hopper WGMMA.
func LoadLayerSafetensors(stData []byte, cfg ArchConfig, layerPrefix string, repackForWGMMA bool) (*RepackedLayerWeights, error) {
	mw, err := model.LoadLayerSafetensors(stData, cfg.toModelArch(), layerPrefix, repackForWGMMA)
	if err != nil {
		return nil, err
	}
	return fromModelWeights(mw), nil
}

// EncodeLayerSafetensors serializes w into a valid .safetensors binary archive.
func EncodeLayerSafetensors(w *RepackedLayerWeights, cfg ArchConfig) ([]byte, error) {
	return model.EncodeLayerSafetensors(w.toModelWeights(), cfg.toModelArch())
}

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
		if k.Tile != nil && k.Tile.Weight == WeightNVFP4 && k.Tile.WGMMA && k.Tile.Repacked {
			for _, a := range k.Args {
				if strings.HasSuffix(a, "_q") {
					repackedConsumer[strings.TrimSuffix(a, "_q")] = true
				}
			}
		}
	}

	toPlanSpec := func(t Tensor) model.PlanTensorSpec {
		return model.PlanTensorSpec{
			Name:  t.Name,
			DType: model.DType(t.DType),
			Shape: t.Shape,
		}
	}
	inputs := make([]model.PlanTensorSpec, len(p.Inputs))
	for i, in := range p.Inputs {
		inputs[i] = toPlanSpec(in)
	}
	tempsAndOut := make([]model.PlanTensorSpec, 0, len(p.Temps)+1)
	for _, tmp := range p.Temps {
		tempsAndOut = append(tempsAndOut, toPlanSpec(tmp))
	}
	tempsAndOut = append(tempsAndOut, toPlanSpec(p.Output))

	cbufs, err := model.BuildDenseLayerBuffers(
		layer.Config.toModelArch(),
		layer.BatchM,
		weights.toModelWeights(),
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
	src.WriteString(Prelude)
	var (
		sawNVFP4Helpers   bool
		sawSwizzleHelpers bool
		sawBWFP4Helpers   bool
	)
	dedupHelpers := func(s string) string {
		if strings.Contains(s, nvfp4WGMMAHelpers) {
			if sawNVFP4Helpers {
				s = strings.Replace(s, nvfp4WGMMAHelpers, "", 1)
			}
			sawNVFP4Helpers = true
		}
		if strings.Contains(s, wgmmaSwizzleHelpers) {
			if sawSwizzleHelpers {
				s = strings.Replace(s, wgmmaSwizzleHelpers, "", 1)
			}
			sawSwizzleHelpers = true
		}
		if strings.Contains(s, blackwellFP4Helpers) {
			if sawBWFP4Helpers {
				s = strings.Replace(s, blackwellFP4Helpers, "", 1)
			}
			sawBWFP4Helpers = true
		}
		return s
	}
	var launches []string
	for _, k := range p.Kernels {
		switch {
		case k.GEMM != nil:
			s, err := k.GEMM.CUDA()
			if err != nil {
				return nil, err
			}
			src.WriteString(dedupHelpers(s))
			grid, block, _ := k.GEMM.Grid()
			launches = append(launches, fmt.Sprintf("eg_emu_launch(dim3(%d, %d), dim3(%d), 0, %s<%d>, %s)",
				grid[0], grid[1], block[0], k.GEMM.Name, k.M, strings.Join(k.Args, ", ")))
		case k.Tile != nil:
			s, err := k.Tile.CUDA()
			if err != nil {
				return nil, err
			}
			src.WriteString(dedupHelpers(s))
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

	tmpDir, err := os.MkdirTemp("", "hopper-safetensors-layer-*")
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

	return model.DecodeOutputF32(model.DType(p.Output.DType), res[p.Output.Name]), nil
}

func encodeU16LE(v []uint16) []byte {
	return model.EncodeU16LE(v)
}
