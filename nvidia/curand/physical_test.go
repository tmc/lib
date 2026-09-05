//go:build curand_physical

package curand_test

import (
	"math"
	"runtime"
	"slices"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/curand"
	"github.com/tmc/lib/nvidia/driver"
)

// TestDeviceUniform checks device generation, repeatability, and seed changes.
// It is not a statistical-quality test. Enabling the tag requires CUDA.
func TestDeviceUniform(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	check := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
	}
	check(driver.Init(0))
	dev, err := driver.NewDeviceGet(0)
	check(err)
	ctx, err := driver.NewContext(nil, 0, dev)
	check(err)
	defer func() { check(ctx.Destroy()) }()
	const count = 4096
	ptr, err := driver.MemAlloc(count * 4)
	check(err)
	defer func() { check(ptr.MemFree()) }()
	generate := func(seed uint64) []float32 {
		t.Helper()
		gen, err := curand.CreateGenerator(curand.CURAND_RNG_PseudoDefault)
		check(err)
		defer func() { check(gen.Destroy()) }()
		check(gen.SetPseudoRandomGeneratorSeed(seed))
		check(gen.GenerateUniform((*float32)(unsafe.Pointer(uintptr(ptr))), count))
		check(driver.CtxSynchronize())
		out := make([]float32, count)
		check(ptr.MemcpyDtoH(unsafe.Pointer(&out[0]), count*4))
		for i, v := range out {
			if math.IsNaN(float64(v)) || v <= 0 || v > 1 {
				t.Fatalf("sample %d = %v, want (0, 1]", i, v)
			}
		}
		return out
	}
	a, b, c := generate(42), generate(42), generate(43)
	if !slices.Equal(a, b) {
		t.Fatal("same seed produced different samples")
	}
	if slices.Equal(a, c) {
		t.Fatal("different seeds produced identical samples")
	}
	varied := false
	for _, v := range a[1:] {
		if v != a[0] {
			varied = true
			break
		}
	}
	if !varied {
		t.Fatal("all samples are identical")
	}
	t.Logf("%d device samples per run; same seed matched, different seed differed; first values %v", count, a[:4])
}
