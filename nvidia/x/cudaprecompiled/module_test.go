package cudaprecompiled

import (
	"fmt"
	"os"
	"testing"
	"testing/fstest"
	"unsafe"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

const scaleCU = `#include "factor.cuh"
extern "C" __global__ void scale(float* x, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) x[i] *= FACTOR;
}
`

func files(factor string) fstest.MapFS {
	return fstest.MapFS{
		"scale.cu":   {Data: []byte(scaleCU)},
		"factor.cuh": {Data: []byte("#define FACTOR " + factor + "\n")},
	}
}

func module(fsys fstest.MapFS) *Module {
	return &Module{Files: fsys, Sources: []string{"scale.cu"}, Headers: []string{"factor.cuh"}}
}

// build returns m with an image built from built for the given targets.
func build(t *testing.T, m *Module, built fstest.MapFS, native []string, portable string) *Module {
	t.Helper()
	b := module(built).config()
	image, _, err := b.Image(native, portable)
	if err != nil {
		t.Fatal(err)
	}
	sum, err := b.Sum()
	if err != nil {
		t.Fatal(err)
	}
	m.Image, m.Sum = image, sum
	return m
}

func device(t *testing.T) *cuda.Device {
	t.Helper()
	if os.Getenv("CUDA_LINK_EXEC_TEST") != "1" {
		t.Skip("CUDA_LINK_EXEC_TEST=1 requires a GPU and NVRTC")
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { dev.Close() })
	return dev
}

// scale loads m, runs scale on 1, 2, ..., 256, and returns the result.
func scale(t *testing.T, dev *cuda.Device, m *Module) []float32 {
	t.Helper()
	mod, err := m.Load(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer mod.Close()
	k, err := mod.Function("scale")
	if err != nil {
		t.Fatal(err)
	}
	const n = 256
	host := make([]float32, n)
	for i := range host {
		host[i] = float32(i + 1)
	}
	buf, err := cudamem.Alloc[float32](dev, n)
	if err != nil {
		t.Fatal(err)
	}
	defer buf.Free()
	if err := buf.CopyFromHost(host); err != nil {
		t.Fatal(err)
	}
	ptr, count := buf.Ptr(), int32(n)
	if err := k.Launch([3]uint32{1, 1, 1}, [3]uint32{n, 1, 1}, 0, 0, []unsafe.Pointer{unsafe.Pointer(&ptr), unsafe.Pointer(&count)}); err != nil {
		t.Fatal(err)
	}
	if err := buf.CopyToHost(host); err != nil {
		t.Fatal(err)
	}
	return host
}

func TestLoad(t *testing.T) {
	dev := device(t)
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	native := fmt.Sprintf("sm_%d%d", major, minor)
	tests := []struct {
		name   string
		module func(*testing.T) *Module
		stale  bool
		want   float32
	}{
		{"no image", func(*testing.T) *Module { return module(files("2")) }, false, 2},
		{"native", func(t *testing.T) *Module {
			return build(t, module(files("2")), files("2"), []string{native}, "")
		}, false, 2},
		{"portable", func(t *testing.T) *Module {
			return build(t, module(files("2")), files("2"), nil, "compute_80")
		}, false, 2},
		{"stale", func(t *testing.T) *Module {
			return build(t, module(files("3")), files("2"), []string{native}, "")
		}, true, 3},
		{"no code for device", func(t *testing.T) *Module {
			other := "sm_80"
			if native == other {
				other = "sm_90"
			}
			return build(t, module(files("2")), files("2"), []string{other}, "")
		}, false, 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := tt.module(t)
			stale, err := m.Stale()
			if err != nil || stale != tt.stale {
				t.Fatalf("Stale() = %v, %v, want %v", stale, err, tt.stale)
			}
			got := scale(t, dev, m)
			for i, v := range got {
				if want := tt.want * float32(i+1); v != want {
					t.Fatalf("x[%d] = %v, want %v", i, v, want)
				}
			}
		})
	}
}

func TestLoweredName(t *testing.T) {
	dev := device(t)
	fsys := fstest.MapFS{"fill.cu": {Data: []byte("template <typename T> __global__ void fill(T* x, T v) { x[threadIdx.x] = v; }\n")}}
	m := &Module{Files: fsys, Sources: []string{"fill.cu"}, Names: map[string]string{"fill<float>": ""}}
	if name, _ := m.LoweredName("fill<float>"); name != "" {
		t.Fatalf("LoweredName before Load = %q", name)
	}
	mod, err := m.Load(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer mod.Close()
	name, ok := m.LoweredName("fill<float>")
	if !ok || name != "_Z4fillIfEvPT_S0_" {
		t.Fatalf("LoweredName = %q, %v", name, ok)
	}
	if _, err := mod.Function(name); err != nil {
		t.Fatal(err)
	}
}

func TestStaleSum(t *testing.T) {
	a, err := module(files("2")).config().Sum()
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range []*Module{
		module(files("3")),
		{Files: files("2"), Sources: []string{"scale.cu"}},
		{Files: files("2"), Sources: []string{"scale.cu"}, Headers: []string{"factor.cuh"}, Options: []string{"-DX"}},
		{Files: files("2"), Sources: []string{"scale.cu"}, Headers: []string{"factor.cuh"}, NoLTO: true},
	} {
		m.Image, m.Sum = []byte{1}, a
		if stale, err := m.Stale(); err != nil || !stale {
			t.Errorf("Stale(%+v) = %v, %v, want true", m, stale, err)
		}
	}
}
