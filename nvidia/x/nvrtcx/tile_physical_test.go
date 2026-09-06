package nvrtcx_test

import (
	"crypto/sha256"
	"os"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func TestTileIRExecution(t *testing.T) {
	if os.Getenv("CUDA_TILE_TEST") != "1" {
		t.Skip("CUDA_TILE_TEST=1 requires NVRTC 13.3 and a TileIR-capable driver")
	}
	check := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
	}
	dev, err := cuda.NewDevice(0)
	check(err)
	defer func() { check(dev.Close()) }()
	major, minor, err := dev.ComputeCapability()
	check(err)
	cm, err := nvrtcx.Compile(`#include "cuda_tile.h"
extern "C" __tile_global__ void fill(int* out) {
 auto i = cuda::tiles::bid().x;
 out[i] = i + 7;
}`, nvrtcx.WithArch(int(major), int(minor)), nvrtcx.WithFlag("--std=c++20"), nvrtcx.WithOutput(nvrtcx.OutputTileIR), nvrtcx.WithNoCache())
	check(err)
	ir := cm.TileIR()
	if len(ir) == 0 {
		t.Fatal("empty TileIR")
	}
	t.Logf("TileIR bytes=%d sha256=%x", len(ir), sha256.Sum256(ir))
	if path := os.Getenv("CUDA_TILE_ARTIFACT"); path != "" {
		check(os.WriteFile(path, ir, 0600))
	}
	mod, log, err := cm.LoadOwnedOutput(dev, nvrtcx.OutputTileIR)
	t.Logf("TileIR driver info=%q error=%q load=%v", log.Info, log.Error, err)
	check(err)
	defer func() { check(mod.Close()) }()
	fn, err := mod.Function("fill")
	check(err)
	var p driver.DevicePtr
	check(dev.WithContext(func() error { var err error; p, err = driver.MemAlloc(64 * 4); return err }))
	defer func() { check(dev.WithContext(p.MemFree)) }()
	check(fn.Launch([3]uint32{64, 1, 1}, [3]uint32{1, 1, 1}, 0, 0, []unsafe.Pointer{unsafe.Pointer(&p)}))
	var got [64]int32
	check(dev.WithContext(func() error {
		if err := driver.CtxSynchronize(); err != nil {
			return err
		}
		return p.MemcpyDtoH(unsafe.Pointer(&got[0]), 256)
	}))
	for i, v := range got {
		if v != int32(i+7) {
			t.Fatalf("output[%d]=%d want %d", i, v, i+7)
		}
	}
	t.Log("verified 64 TileIR kernel outputs")
}

func TestTileIRMatrixExecution(t *testing.T) {
	runTileMatrix(t, false)
}
