package nvrtcx_test

import (
	"crypto/sha256"
	"os"
	"runtime"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
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
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	check(driver.Init(0))
	dev, err := driver.NewDeviceGet(0)
	check(err)
	major, err := dev.GetAttribute(driver.DevAttrComputeCapabilityMajor)
	check(err)
	minor, err := dev.GetAttribute(driver.DevAttrComputeCapabilityMinor)
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
	ctx, err := driver.NewContext(nil, 0, dev)
	check(err)
	defer func() { check(ctx.Destroy()) }()
	mod, err := driver.NewModuleLoadData(unsafe.Pointer(unsafe.SliceData(ir)))
	check(err)
	runtime.KeepAlive(ir)
	defer func() { check(mod.Unload()) }()
	fn, err := mod.GetFunction("fill")
	check(err)
	p, err := driver.MemAlloc(64 * 4)
	check(err)
	defer func() { check(p.MemFree()) }()
	check(fn.LaunchKernel(64, 1, 1, 1, 1, 1, 0, 0, []unsafe.Pointer{unsafe.Pointer(&p)}))
	check(driver.CtxSynchronize())
	var got [64]int32
	check(p.MemcpyDtoH(unsafe.Pointer(&got[0]), 256))
	for i, v := range got {
		if v != int32(i+7) {
			t.Fatalf("output[%d]=%d want %d", i, v, i+7)
		}
	}
	t.Log("verified 64 TileIR kernel outputs")
}
