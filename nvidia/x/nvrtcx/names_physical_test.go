package nvrtcx

import (
	"fmt"
	"github.com/tmc/lib/nvidia/driver"
	"os"
	"runtime"
	"testing"
	"unsafe"
)

func TestPhysicalLoweredName(t *testing.T) {
	if os.Getenv("NVRTC_NAMES_TEST") != "1" {
		t.Skip("NVRTC_NAMES_TEST=1 requires CUDA")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	check := func(e error) {
		t.Helper()
		if e != nil {
			t.Fatal(e)
		}
	}
	old := cacheDirOverride
	cacheDirOverride = t.TempDir()
	defer func() { cacheDirOverride = old }()
	check(driver.Init(0))
	dev, e := driver.DeviceGet(0)
	check(e)
	major, e := dev.Attribute(driver.DevAttrComputeCapabilityMajor)
	check(e)
	minor, e := dev.Attribute(driver.DevAttrComputeCapabilityMinor)
	check(e)
	_, badErr := Compile(`template<int N> __global__ void fill(int *p) { p[threadIdx.x] = N; }`, WithFlag(fmt.Sprintf("--gpu-architecture=sm_%d%d", major, minor)), WithOutput(OutputCUBIN), WithNameExpressions("missing_symbol"))
	if badErr == nil {
		t.Fatal("invalid expression compiled successfully")
	}
	entries, e := os.ReadDir(cacheDirOverride)
	check(e)
	if len(entries) != 0 {
		t.Fatalf("failed compile left %d cache entries", len(entries))
	}
	t.Logf("invalid expression rejected without cache publication: %v", badErr)
	cm, e := Compile(`template<int N> __global__ void fill(int *p) { p[threadIdx.x] = N + threadIdx.x; }`, WithFlag(fmt.Sprintf("--gpu-architecture=sm_%d%d", major, minor)), WithOutput(OutputCUBIN), WithNameExpressions("fill<7>"))
	check(e)
	other, e := Compile(`template<int N> __global__ void fill(int *p) { p[threadIdx.x] = N + threadIdx.x; }`, WithFlag(fmt.Sprintf("--gpu-architecture=sm_%d%d", major, minor)), WithOutput(OutputCUBIN), WithNameExpressions("fill<8>"))
	check(e)
	if other.key == cm.key {
		t.Fatal("name expressions did not change cache key")
	}
	if _, ok := other.LoweredName("fill<7>"); ok {
		t.Fatal("name leaked between cache entries")
	}
	name, ok := cm.LoweredName("fill<7>")
	if !ok || name == "" {
		t.Fatal("missing lowered name")
	}
	memCacheMu.Lock()
	delete(memCache, cm.key)
	memCacheMu.Unlock()
	cached, ok := lookupCache(cm.key)
	if !ok {
		t.Fatal("disk cache miss")
	}
	if got, ok := cached.LoweredName("fill<7>"); !ok || got != name {
		t.Fatalf("cached name %q differs from %q", got, name)
	}
	ctx, e := driver.NewContext(dev, 0)
	check(e)
	defer func() { check(ctx.Close()) }()
	cubin := cached.CUBIN()
	mod, e := driver.ModuleLoadData(cubin)
	check(e)
	runtime.KeepAlive(cubin)
	defer func() { check(mod.Close()) }()
	fn, e := mod.Function(name)
	check(e)
	p, e := driver.MemAllocBytes(64 * 4)
	check(e)
	defer func() { check(p.Free()) }()
	check(fn.LaunchKernel(driver.Dim3{X: 1, Y: 1, Z: 1}, driver.Dim3{X: 64, Y: 1, Z: 1}, 0, 0, []unsafe.Pointer{unsafe.Pointer(&p)}))
	check(driver.CtxSynchronize())
	var got [64]int32
	check(p.CopyToHostPtr(unsafe.Pointer(&got[0]), 256))
	for i, v := range got {
		if v != int32(i+7) {
			t.Fatalf("output %d=%d", i, v)
		}
	}
	t.Logf("lowered name %q survived compiler destruction and disk cache; 64 outputs matched", name)
}
