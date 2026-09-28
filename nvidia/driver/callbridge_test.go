//go:build darwin || linux

package driver

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

// stubDriver stands in for libcuda. Each function returns what its
// arguments predict, so a wrapper that forwards the wrong value fails.
const stubDriver = `
#include <stddef.h>
#include <stdint.h>
int cuDriverGetVersion(int *v) { *v = 13010; return 0; }
int cuCtxSynchronize(void) { return 0; }
int cuDeviceGet(int *d, int ordinal) { *d = ordinal + 100; return 0; }
int cuDeviceGetAttribute(int *v, int attr, int dev) { *v = attr*1000 + dev; return 0; }
int cuMemAlloc_v2(uint64_t *p, size_t n) { *p = 0x7f0000000000ull + n; return 0; }
int cuMemsetD8_v2(uint64_t p, unsigned char c, size_t n) {
	return p == 0x7f0000001000ull && c == 0xa5 && n == 64 ? 0 : 1;
}
typedef struct { int id; char pad[4]; union { char pad[64]; int i; unsigned d[3]; } v; } attr;
typedef struct {
	unsigned grid[3], block[3], shmem;
	void *stream;
	attr *attrs;
	unsigned numAttrs;
} config;
int cuLaunchKernelEx(const config *c, void *f, void **params, void **extra) {
	return c->grid[0] == 8 && c->block[0] == 256 && c->shmem == 1024 &&
		f == (void *)0x1234 && params == NULL && extra == NULL &&
		c->numAttrs == 2 &&
		c->attrs[0].id == 2 && c->attrs[0].v.i == 1 &&
		c->attrs[1].id == 4 && c->attrs[1].v.d[0] == 2 && c->attrs[1].v.d[2] == 1 ? 0 : 1;
}
`

// TestCallBridge checks, without a GPU, that each common wrapper shape
// forwards its arguments and returns its outputs, and that none of them
// allocates.
func TestCallBridge(t *testing.T) {
	bindStub(t, stubDriver, map[string]*uintptr{
		"cuDriverGetVersion":   &symDriverGetVersion,
		"cuCtxSynchronize":     &symCtxSynchronize,
		"cuDeviceGet":          &symdeviceGet,
		"cuDeviceGetAttribute": &symdeviceGetAttribute,
		"cuMemAlloc_v2":        &symMemAlloc,
		"cuMemsetD8_v2":        &symmemsetD8,
		"cuLaunchKernelEx":     &symlaunchKernelEx,
	})
	cfg := LaunchConfig{
		GridDim:        Dim3{X: 8, Y: 1, Z: 1},
		BlockDim:       Dim3{X: 256, Y: 1, Z: 1},
		SharedMemBytes: 1024,
		Attrs:          make([]LaunchAttribute, 2),
	}
	cfg.Attrs[0].Id = LaunchAttributeCooperative
	*cfg.Attrs[0].Value.Cooperative() = 1
	cfg.Attrs[1].Id = LaunchAttributeClusterDimension
	*cfg.Attrs[1].Value.ClusterDim() = Dim3{X: 2, Y: 1, Z: 1}
	for _, tt := range []struct {
		name string
		call func() error
	}{
		{"none", func() error { return CtxSynchronize() }},
		{"out", func() error {
			v, err := DriverGetVersion()
			if err == nil && v != 13010 {
				t.Errorf("DriverGetVersion = %d, want 13010", v)
			}
			return err
		}},
		{"handle", func() error {
			d, err := DeviceGet(2)
			if err == nil && d != 102 {
				t.Errorf("DeviceGet(2) = %d, want 102", d)
			}
			return err
		}},
		{"method", func() error {
			v, err := Device(7).Attribute(DeviceAttribute(4))
			if err == nil && v != 4007 {
				t.Errorf("Attribute = %d, want 4007", v)
			}
			return err
		}},
		{"create", func() error {
			p, err := MemAllocBytes(0x1000)
			if err == nil && p != 0x7f0000001000 {
				t.Errorf("MemAllocBytes = %#x, want 0x7f0000001000", p)
			}
			return err
		}},
		{"devptr", func() error { return DevicePtr(0x7f0000001000).MemsetD8(0xa5, 64) }},
		{"launchex", func() error { return Function(0x1234).LaunchKernelEx(&cfg, nil) }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.call(); err != nil {
				t.Fatal(err)
			}
			if n := testing.AllocsPerRun(100, func() { tt.call() }); n != 0 {
				t.Errorf("%v allocs/op, want 0", n)
			}
		})
	}
}

// bindStub compiles csrc with the host C compiler and, until the test ends,
// binds each symbol in syms to it and marks it present. It skips the test
// when there is no compiler.
func bindStub(t *testing.T, csrc string, syms map[string]*uintptr) {
	cc := os.Getenv("CC")
	if cc == "" {
		cc = "cc"
	}
	if _, err := exec.LookPath(cc); err != nil {
		t.Skip("no C compiler")
	}
	dir := t.TempDir()
	src := filepath.Join(dir, "stub.c")
	if err := os.WriteFile(src, []byte(csrc), 0o644); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "libstub.so")
	args := []string{"-shared", "-fPIC", "-o", path, src}
	if runtime.GOOS == "darwin" {
		arch := runtime.GOARCH
		if arch == "amd64" {
			arch = "x86_64"
		}
		args = append([]string{"-arch", arch}, args...)
	}
	if out, err := exec.Command(cc, args...).CombinedOutput(); err != nil {
		t.Fatalf("cc: %v\n%s", err, out)
	}
	h, err := openDynamicLibrary(path)
	if err != nil {
		t.Fatal(err)
	}
	oldReady, oldPresent := ready.Load(), present
	old := make(map[string]uintptr)
	present = make(map[string]bool)
	for name, p := range syms {
		old[name] = *p
		if *p, err = lookupLibrarySymbol(h, name); err != nil {
			t.Fatal(err)
		}
		present[name] = true
	}
	ready.Store(true)
	t.Cleanup(func() {
		for name, p := range syms {
			*p = old[name]
		}
		present = oldPresent
		ready.Store(oldReady)
		closeLibrary(h)
	})
}
