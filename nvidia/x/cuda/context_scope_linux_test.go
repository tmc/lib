//go:build linux

package cuda_test

import (
	"errors"
	"runtime"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func TestContextScopeRestoration(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	dev := cudatest.RequireDevice(t)
	defer dev.Close()
	ctx, err := dev.PrimaryContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Release()
	previous, err := driver.CtxGetCurrent()
	if err != nil {
		t.Fatal(err)
	}
	defer previous.SetCurrent()
	foreign, err := driver.NewContext(dev.Raw(), 0)
	if err != nil {
		t.Fatal(err)
	}
	defer foreign.Close()
	check := func(want driver.Context) {
		t.Helper()
		got, err := driver.CtxGetCurrent()
		if err != nil || got != want {
			t.Fatalf("current=%v, %v; want %v", got, err, want)
		}
	}
	for _, before := range []driver.Context{0, ctx.Raw(), foreign} {
		if err := before.SetCurrent(); err != nil {
			t.Fatal(err)
		}
		if err := ctx.Run(func() error {
			check(ctx.Raw())
			if err := ctx.Run(func() error { check(ctx.Raw()); return nil }); err != nil {
				return err
			}
			check(ctx.Raw())
			return nil
		}); err != nil {
			t.Fatal(err)
		}
		check(before)
		sentinel := errors.New("callback failure")
		if err := ctx.Run(func() error {
			if err := driver.Context(0).SetCurrent(); err != nil {
				return err
			}
			return sentinel
		}); !errors.Is(err, sentinel) {
			t.Fatalf("Run error=%v", err)
		}
		check(before)
		func() {
			defer func() {
				if recover() != sentinel {
					t.Error("wrong panic")
				}
			}()
			_ = ctx.Run(func() error {
				if err := driver.Context(0).SetCurrent(); err != nil {
					t.Fatal(err)
				}
				panic(sentinel)
			})
		}()
		check(before)
	}
	if err := ctx.Release(); err != nil {
		t.Fatal(err)
	}
	called := false
	if err := ctx.Run(func() error { called = true; return nil }); err == nil || called {
		t.Fatalf("released Run=%v, called=%v", err, called)
	}
}

func TestKernelLaunchContextScope(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	dev := cudatest.RequireDevice(t)
	defer dev.Close()
	previous, err := driver.CtxGetCurrent()
	if err != nil {
		t.Fatal(err)
	}
	defer previous.SetCurrent()
	own, err := dev.PrimaryContext()
	if err != nil {
		t.Fatal(err)
	}
	defer own.Release()
	foreign, err := driver.NewContext(dev.Raw(), 0)
	if err != nil {
		t.Fatal(err)
	}
	defer foreign.Close()
	cm, err := nvrtcx.Compile(`extern "C" __global__ void empty(){}`, nvrtcx.WithDeviceArch(dev))
	if err != nil {
		t.Fatal(err)
	}
	var k struct {
		Empty *cuda.Kernel `cuda:"empty"`
	}
	mod, err := cm.Bind(dev, &k)
	if err != nil {
		t.Fatal(err)
	}
	defer mod.Close()
	for _, before := range []driver.Context{0, own.Raw(), foreign} {
		if err := before.SetCurrent(); err != nil {
			t.Fatal(err)
		}
		for _, launch := range []func() error{
			func() error { return k.Empty.Launch([3]uint32{1, 1, 1}, [3]uint32{1, 1, 1}, 0, 0, nil) },
			func() error {
				return k.Empty.LaunchEx(&driver.LaunchConfig{GridDim: driver.Dim3{X: 1, Y: 1, Z: 1}, BlockDim: driver.Dim3{X: 1, Y: 1, Z: 1}}, nil)
			},
		} {
			if err := launch(); err != nil {
				t.Fatal(err)
			}
			got, err := driver.CtxGetCurrent()
			if err != nil || got != before {
				t.Fatalf("after launch current=%v,%v; want %v", got, err, before)
			}
		}
	}
}
