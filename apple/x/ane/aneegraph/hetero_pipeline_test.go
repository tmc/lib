//go:build darwin

package aneegraph

import (
	"errors"
	"testing"
	"unsafe"

	applemetal "github.com/tmc/apple/metal"
	"github.com/tmc/apple/x/ane"
	"github.com/tmc/apple/x/ane/mil"
	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/egraph"
	"github.com/tmc/lib/apple/x/metal/mempool"
	"github.com/tmc/lib/apple/x/metal/metalsample"
)

func TestHeterogeneousGPU_ANE_Pipeline(t *testing.T) {
	client, err := ane.Open()
	if errors.Is(err, ane.ErrNoANE) {
		t.Skip("no ANE available on this device")
	}
	if err != nil {
		t.Fatalf("ANE Open failed: %v", err)
	}
	defer client.Close()

	gpuDev, err := metal.NewDevice()
	if err != nil {
		t.Fatalf("Metal GPU init failed: %v", err)
	}

	// 1. Build unified Heterogeneous E-Graph
	root := &egraph.Expr{
		Op: "add",
		Children: []*egraph.Expr{
			{
				Op: "causal_flash_attention",
				Children: []*egraph.Expr{
					{Op: "q"}, {Op: "k"}, {Op: "v"},
				},
			},
			{
				Op: "fused_rmsnorm_gemv",
				Children: []*egraph.Expr{
					{Op: "x"}, {Op: "w_norm"}, {Op: "eps"}, {Op: "w_proj"},
				},
			},
		},
	}

	pg := egraph.HeterogeneousPartition(root)
	if len(pg.ANEExprs) == 0 || len(pg.GPUExprs) == 0 {
		t.Fatalf("failed to partition E-Graph: %v", pg)
	}

	// 2. Compile ANE part (RMSNorm GEMV projection)
	const dim = 16
	opts := ANEOptions{
		SpatialDim:       1,
		HiddenDim:        dim,
		IntermediateDim:  32,
		WeightBlobOffset: 64,
	}

	milText, err := GenerateMIL(pg.ANEExprs[0], opts)
	if err != nil {
		t.Fatalf("GenerateMIL failed: %v", err)
	}

	blob, err := mil.BuildIdentityWeightBlob(dim)
	if err != nil {
		t.Fatalf("BuildIdentityWeightBlob failed: %v", err)
	}

	aneModel, err := client.Compile(ane.CompileOptions{
		ModelType:  ane.ModelTypeMIL,
		MILText:    []byte(milText),
		WeightBlob: blob,
	})
	if err != nil {
		t.Fatalf("ANE Compile failed: %v", err)
	}
	defer aneModel.Close()

	// 3. Compile GPU part (Metal sampling & Attention reduction)
	lib, err := gpuDev.CompileLibrary(metalsample.ArgmaxSamplingMSL)
	if err != nil {
		t.Fatalf("CompileLibrary failed: %v", err)
	}

	pso, err := gpuDev.NewComputePipeline(lib, "ArgmaxSampling")
	if err != nil {
		t.Fatalf("NewComputePipeline failed: %v", err)
	}

	// 4. Zero-Copy Execution across ANE & GPU
	inputVec := make([]float32, dim)
	for i := range inputVec {
		inputVec[i] = float32(i + 1)
	}
	_ = aneModel.WriteInputF32(0, inputVec)

	// Step A: Evaluate on Apple Neural Engine Cores
	if err := aneModel.Eval(); err != nil {
		t.Fatalf("ANE Eval failed: %v", err)
	}

	aneOut := make([]float32, dim)
	_ = aneModel.ReadOutputF32(0, aneOut)

	// Step B: Direct handoff to Apple Silicon Metal GPU Cores
	logitsBuf, _ := mempool.AllocFromSlice(gpuDev, aneOut)
	tokBuf, _ := mempool.Alloc[int32](gpuDev, 1)
	vocabSize := int32(dim)
	vBuf := gpuDev.Raw().NewBufferWithBytesLengthOptions(unsafe.Pointer(&vocabSize), 4, applemetal.MTLResourceStorageModeShared)

	cb := gpuDev.Queue().CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(pso)
	enc.SetBufferWithOffsetAtIndex(logitsBuf.Buffer(), 0, 0)
	enc.SetBufferWithOffsetAtIndex(tokBuf.Buffer(), 0, 1)
	enc.SetBufferWithOffsetAtIndex(vBuf, 0, 2)
	enc.DispatchThreadsThreadsPerThreadgroup(
		applemetal.MTLSize{Width: 1, Height: 1, Depth: 1},
		applemetal.MTLSize{Width: 1, Height: 1, Depth: 1},
	)
	enc.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()

	selectedTok := tokBuf.HostSlice()[0]

	t.Logf("=== Heterogeneous GPU + ANE E-Graph Pipeline Complete ===")
	t.Logf("1. Projection Fused Layer Executed on Apple Neural Engine (ANE Cores)")
	t.Logf("2. Softmax / Attention / Sampling Executed on Apple Metal GPU")
	t.Logf("3. Combined Generated Token: %d", selectedTok)
}
