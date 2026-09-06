package nvrtcx_test

import (
	"crypto/sha256"
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func TestMixedTileCompilation(t *testing.T) {
	if os.Getenv("CUDA_TILE_TEST") != "1" {
		t.Skip("CUDA_TILE_TEST=1 requires NVRTC supporting mixed Tile/SIMT compilation")
	}
	cm, err := nvrtcx.Compile(`#include "cuda_tile.h"
extern "C" __tile_global__ void tile_fill(int* out) {
 auto i = cuda::tiles::bid().x;
 out[i] = i + 7;
}
extern "C" __global__ void simt_fill(int* out) {
 out[threadIdx.x] = threadIdx.x + 3;
}`, nvrtcx.WithArch(12, 1), nvrtcx.WithFlag("--std=c++20"), nvrtcx.WithOutput(nvrtcx.OutputPTXTileIR), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatal(err)
	}
	if cm.PTX() == "" || len(cm.TileIR()) == 0 {
		t.Fatal("missing mixed output")
	}
	t.Logf("PTX bytes=%d sha256=%x; TileIR bytes=%d sha256=%x", len(cm.PTX()), sha256.Sum256([]byte(cm.PTX())), len(cm.TileIR()), sha256.Sum256(cm.TileIR()))
	t.Log("compilation only; neither artifact was loaded or executed")
}

func TestMixedTileExecution(t *testing.T) {
	runTileMatrix(t, true)
}
