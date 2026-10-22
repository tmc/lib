//go:build linux

package main

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/cublas"
	"github.com/tmc/lib/nvidia/driver"
)

func TestFlashVisionOwnerRelease(t *testing.T) {
	for _, nativeFailure := range []bool{false, true} {
		name := "restore failure"
		if nativeFailure {
			name = "native failure"
		}
		t.Run(name, func(t *testing.T) {
			e := &flashVisionEncoder{module: driver.Module(7), ready: true}
			e.graph.blas = cublas.Handle(9)
			failure := errors.New("release failure")
			blas, module := 0, 0
			ops := flashVisionCloseOperations{
				withContext: func(f func() error) error {
					if err := f(); err != nil {
						return err
					}
					return failure
				},
				destroyBLAS: func(cublas.Handle) error {
					blas++
					if nativeFailure {
						return failure
					}
					return nil
				},
				unloadModule: func(driver.Module) error {
					module++
					if nativeFailure {
						return failure
					}
					return nil
				},
			}
			if err := e.closeWithOperations(ops); !errors.Is(err, failure) {
				t.Fatalf("close: %v", err)
			}
			if e.ready {
				t.Fatal("failed close left encoder ready")
			}
			retained := e.graph.blas != 0 && e.module != 0
			ops.withContext = func(f func() error) error { return f() }
			ops.destroyBLAS = func(cublas.Handle) error { blas++; return nil }
			ops.unloadModule = func(driver.Module) error { module++; return nil }
			if err := e.closeWithOperations(ops); err != nil {
				t.Fatal(err)
			}
			want := 1
			if nativeFailure {
				want = 2
			}
			if blas != want || module != want || e.module != 0 || e.graph.blas != 0 {
				t.Fatalf("release counts %d/%d, want %d", blas, module, want)
			}
			if nativeFailure != retained {
				t.Fatal("release ownership differs from native outcome")
			}
		})
	}
}

func TestFlashVisionOwnerSynchronizeFailure(t *testing.T) {
	e := &flashVisionEncoder{module: driver.Module(7), ready: true}
	e.graph.blas = cublas.Handle(9)
	failure := errors.New("synchronize failure")
	releases := 0
	ops := flashVisionCloseOperations{
		synchronize: func() error { return failure },
		withContext: func(f func() error) error { releases++; return f() },
	}
	if err := e.closeWithOperations(ops); !errors.Is(err, failure) {
		t.Fatalf("close: %v", err)
	}
	if releases != 0 || e.module != 7 || e.graph.blas != 9 || e.ready {
		t.Fatal("synchronize failure lost ownership or released resources")
	}
}
