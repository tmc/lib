//go:build curand_physical

package curand_test

import (
	"math"
	"os"
	"runtime"
	"slices"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/curand"
	"github.com/tmc/lib/nvidia/driver"
)

// Set CURAND_PHYSICAL_PERTURB to the name of an assertion to run the negative
// control: the expected value is corrupted, and a test that is not genuinely
// comparing content will not notice.
func perturb(name string) bool { return os.Getenv("CURAND_PHYSICAL_PERTURB") == name }

// wantSeed42 is the start of the XORWOW sequence CURAND_RNG_PSEUDO_DEFAULT
// produces for seed 42 with the default ordering, recorded from
// libcurand.so.10 (10.4.0.35) on a GB10 and confirmed identical from
// libcurand.so.10 (10.3.1.124) on the same host.
//
// These are the oracle. cuRAND does not promise the sequence across major
// versions, so a future library that changes it fails here rather than
// quietly returning different numbers, which is the point: the alternative
// is a test that accepts whatever it is given as long as it lands in (0, 1].
var wantSeed42 = []float32{
	0.070020944, 0.16014557, 0.42049608, 0.57637084,
	0.15770254, 0.40192622, 0.5076846, 0.108753294,
}

// TestDeviceUniformMatchesOracle checks the device generator against recorded
// values and against cuRAND's own host generator, which implements the same
// algorithm on the CPU and must agree bit for bit. Enabling the tag requires
// CUDA.
func TestDeviceUniformMatchesOracle(t *testing.T) {
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

	device := func(seed uint64) []float32 {
		t.Helper()
		gen, err := curand.CreateGenerator(curand.CURAND_RNG_PseudoDefault)
		check(err)
		defer func() { check(gen.Destroy()) }()
		check(gen.SetPseudoRandomGeneratorSeed(seed))
		check(gen.GenerateUniform((*float32)(unsafe.Pointer(uintptr(ptr))), count))
		check(driver.CtxSynchronize())
		out := make([]float32, count)
		check(ptr.MemcpyDtoH(unsafe.Pointer(&out[0]), count*4))
		return out
	}
	host := func(seed uint64) []float32 {
		t.Helper()
		gen, err := curand.CreateGeneratorHost(curand.CURAND_RNG_PseudoDefault)
		check(err)
		defer func() { check(gen.Destroy()) }()
		check(gen.SetPseudoRandomGeneratorSeed(seed))
		out := make([]float32, count)
		check(gen.GenerateUniform(&out[0], count))
		return out
	}

	a := device(42)
	want := slices.Clone(wantSeed42)
	if perturb("recorded") {
		want[0] = -want[0]
	}
	for i, w := range want {
		if a[i] != w {
			t.Errorf("sample %d = %v, want %v", i, a[i], w)
		}
	}

	h := host(42)
	if perturb("host") {
		h[0] = -h[0]
	}
	if !slices.Equal(a, h) {
		for i := range a {
			if a[i] != h[i] {
				t.Fatalf("device and host disagree at %d: %v != %v", i, a[i], h[i])
			}
		}
	}

	// The same seed must repeat and a different seed must not, or the
	// comparison above would pass on a generator that ignores the seed.
	if !slices.Equal(a, device(42)) {
		t.Error("same seed produced different samples")
	}
	if c := device(43); slices.Equal(a, c) {
		t.Error("different seeds produced identical samples")
	}
	for i, v := range a {
		if math.IsNaN(float64(v)) || v <= 0 || v > 1 {
			t.Fatalf("sample %d = %v, want (0, 1]", i, v)
		}
	}
	t.Logf("%d device samples matched %d recorded values and the host generator", count, len(wantSeed42))
}
