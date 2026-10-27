package nvl72

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
)

func TestRubinFabricAsyncTensorValidationAndPTX(t *testing.T) {
	if FabricSuccess != 0 {
		t.Fatalf("FabricSuccess = %d, want 0", FabricSuccess)
	}
	if CountedBytesAlignment != 256 {
		t.Fatalf("CountedBytesAlignment = %d, want 256", CountedBytesAlignment)
	}

	// 1. Verify PTXInstruction() for try_get, try_put (.counted::bytes), and try_red (.multimem...add.f32).
	getK := RubinFabricAsyncTensorKernel{
		Op:         FabricTensorTryGet,
		Dim:        2,
		Multimem:   false,
		Completion: FabricCompletionMbarrier,
		Target:     "sm_107f",
	}
	gotGetPTX, err := getK.PTXInstruction()
	if err != nil {
		t.Fatalf("getK.PTXInstruction: %v", err)
	}
	wantGetPTX := "fabric.try_get.async.tensor.2d.shared::cluster.fabric.mbarrier::complete_tx::16B.mbarrier::report::fabric.b128"
	if gotGetPTX != wantGetPTX {
		t.Fatalf("try_get PTX = %q, want %q", gotGetPTX, wantGetPTX)
	}

	putK := RubinFabricAsyncTensorKernel{
		Op:         FabricTensorTryPut,
		Dim:        2,
		Multimem:   true,
		Completion: FabricCompletionCountedBytes,
		Target:     "sm_107f",
		Endpoint:   FabricEndpointHandle{LeID: 7, Offset: 512},
	}
	gotPutPTX, err := putK.PTXInstruction()
	if err != nil {
		t.Fatalf("putK.PTXInstruction: %v", err)
	}
	wantPutPTX := "fabric.try_put.async.tensor.2d.multimem.fabric.shared::cta.counted::bytes.b128"
	if gotPutPTX != wantPutPTX {
		t.Fatalf("try_put PTX = %q, want %q", gotPutPTX, wantPutPTX)
	}

	redK := RubinFabricAsyncTensorKernel{
		Op:         FabricTensorTryRed,
		Dim:        2,
		Multimem:   true,
		Completion: FabricCompletionMbarrier,
		RedOp:      "add",
		Target:     "sm_107f",
	}
	gotRedPTX, err := redK.PTXInstruction()
	if err != nil {
		t.Fatalf("redK.PTXInstruction: %v", err)
	}
	wantRedPTX := "fabric.try_red.async.tensor.2d.multimem.fabric.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.add.f32"
	if gotRedPTX != wantRedPTX {
		t.Fatalf("try_red PTX = %q, want %q", gotRedPTX, wantRedPTX)
	}

	cudaSrc, err := redK.CUDA()
	if err != nil {
		t.Fatalf("redK.CUDA: %v", err)
	}
	for _, wantSub := range []string{
		"__CUDA_ARCH__ == 1070",
		wantRedPTX,
		FabricTryAtomAsyncIncU32PTX,
		"eg_rubin_fabric_async_tensor",
	} {
		if !strings.Contains(cudaSrc, wantSub) {
			t.Fatalf("redK.CUDA missing %q", wantSub)
		}
	}

	// 2. Verify validation errors: try_get + Multimem, try_get + counted_bytes, sm_100a rejection, unaligned counted_bytes offset.
	errCases := []struct {
		name string
		k    RubinFabricAsyncTensorKernel
	}{
		{
			name: "try_get_with_multimem",
			k: RubinFabricAsyncTensorKernel{
				Op:       FabricTensorTryGet,
				Dim:      2,
				Multimem: true,
				Target:   "sm_107f",
			},
		},
		{
			name: "try_get_with_counted_bytes",
			k: RubinFabricAsyncTensorKernel{
				Op:         FabricTensorTryGet,
				Dim:        2,
				Completion: FabricCompletionCountedBytes,
				Target:     "sm_107f",
			},
		},
		{
			name: "sm_100a_rejected",
			k: RubinFabricAsyncTensorKernel{
				Op:     FabricTensorTryRed,
				Dim:    2,
				Target: "sm_100a",
			},
		},
		{
			name: "invalid_dim_6",
			k: RubinFabricAsyncTensorKernel{
				Op:     FabricTensorTryRed,
				Dim:    6,
				Target: "sm_107f",
			},
		},
		{
			name: "unaligned_counted_bytes_offset",
			k: RubinFabricAsyncTensorKernel{
				Op:         FabricTensorTryPut,
				Dim:        2,
				Completion: FabricCompletionCountedBytes,
				Target:     "sm_107f",
				Endpoint:   FabricEndpointHandle{LeID: 1, Offset: 128},
			},
		},
	}
	for _, tc := range errCases {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.k.Validate(); err == nil {
				t.Fatalf("expected error for %s, got nil", tc.name)
			}
		})
	}
}

func TestRubinFabricAsyncTensorMultiRank(t *testing.T) {
	const (
		numRanks = 4
		tileRows = 8
		tileCols = 8
		n        = tileRows * tileCols
	)

	rankTiles := make([][]float32, numRanks)
	srcFlat := make([]float32, numRanks*n)
	for r := 0; r < numRanks; r++ {
		rankTiles[r] = make([]float32, n)
		for i := 0; i < n; i++ {
			v := float32(((r+1)*(i+3))%11-5) * 0.25
			rankTiles[r][i] = v
			srcFlat[r*n+i] = v
		}
	}

	for _, op := range []FabricTensorOp{FabricTensorTryRed, FabricTensorTryPut, FabricTensorTryGet} {
		t.Run(string(op), func(t *testing.T) {
			comp := FabricCompletionMbarrier
			multimem := op != FabricTensorTryGet
			if op == FabricTensorTryPut {
				comp = FabricCompletionCountedBytes
			}
			k := RubinFabricAsyncTensorKernel{
				Op:         op,
				Dim:        2,
				TileRows:   tileRows,
				TileCols:   tileCols,
				Multimem:   multimem,
				Completion: comp,
				RedOp:      "add",
				NumRanks:   numRanks,
				Target:     "sm_107f",
				Endpoint:   FabricEndpointHandle{LeID: 3, Offset: 512},
			}
			src, err := k.CUDAFile()
			if err != nil {
				t.Fatalf("CUDAFile(%s): %v", op, err)
			}
			want, err := k.Reference(rankTiles)
			if err != nil {
				t.Fatalf("Reference(%s): %v", op, err)
			}

			dstBytes := cudahostemu.Poison(numRanks * n * 4)
			srcBytes := f32Bytes(srcFlat)
			statusBytes := cudahostemu.Poison(numRanks * 4)

			grid, block := k.Grid()
			err = cudahostemu.RunMultiRank(context.Background(), src, numRanks, cudahostemu.Launch{
				Kernel: "eg_rubin_fabric_async_tensor",
				Grid:   cudahostemu.Dim3{X: grid[0], Y: grid[1], Z: grid[2]},
				Block:  cudahostemu.Dim3{X: block[0], Y: block[1], Z: block[2]},
			},
				cudahostemu.BufferArg("dst_tile", "float", dstBytes),
				cudahostemu.BufferArg("src_tile", "float", srcBytes),
				cudahostemu.BufferArg("out_status", "unsigned int", statusBytes),
				cudahostemu.IntArg(n),
			)
			if errors.Is(err, cudahostemu.ErrNoCompiler) {
				t.Skip(err)
			}
			if err != nil {
				t.Fatalf("RunMultiRank(%s): %v", op, err)
			}

			gotDst := bytesF32(dstBytes)
			gotStatus := bytesU32(statusBytes)

			for r := 0; r < numRanks; r++ {
				if gotStatus[r] != uint32(FabricSuccess) {
					t.Fatalf("rank %d out_status[0] = %d, want %d", r, gotStatus[r], FabricSuccess)
				}
			}

			checkRanks := numRanks
			if op == FabricTensorTryGet {
				checkRanks = 1 // Reference returns rank 0's pulled tile
			}
			for r := 0; r < checkRanks; r++ {
				for i := 0; i < n; i++ {
					got := gotDst[r*n+i]
					w := want[i]
					if math.Abs(float64(got-w)) > 1e-5 {
						t.Fatalf("%s rank %d dst_tile[%d] = %v, want %v", op, r, i, got, w)
					}
				}
			}
		})
	}
}

func ExampleRubinFabricAsyncTensorKernel() {
	k := RubinFabricAsyncTensorKernel{
		Op:         FabricTensorTryRed,
		Dim:        2,
		TileRows:   32,
		TileCols:   32,
		Multimem:   true,
		Completion: FabricCompletionMbarrier,
		RedOp:      "add",
		NumRanks:   4,
		Target:     "sm_107f",
	}
	inst, err := k.PTXInstruction()
	if err != nil {
		panic(err)
	}
	fmt.Println(inst)
	// Output:
	// fabric.try_red.async.tensor.2d.multimem.fabric.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.add.f32
}
