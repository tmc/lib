//go:build linux

package main

import (
	"errors"
	"io"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func TestCUDAProbeCleanup(t *testing.T) {
	failure := errors.New("cleanup fault")
	for _, fault := range []string{"none", "drain", "free", "unload", "restore"} {
		t.Run(fault, func(t *testing.T) {
			var calls []string
			attempt := 0
			owner := &cudaProbeResources{module: 1}
			owner.run = func(fn func() error) error {
				attempt++
				err := fn()
				if err == nil && fault == "restore" && attempt == 1 {
					return failure
				}
				return err
			}
			owner.synchronize = func() error {
				calls = append(calls, "drain")
				if fault == "drain" && attempt == 1 {
					return failure
				}
				return nil
			}
			owner.buffer = func() error {
				calls = append(calls, "free")
				if fault == "free" && attempt == 1 {
					return failure
				}
				return nil
			}
			owner.unload = func(driver.Module) error {
				calls = append(calls, "unload")
				if fault == "unload" && attempt == 1 {
					return failure
				}
				return nil
			}
			err := finishBackendCleanup(nil, owner)
			if fault == "none" {
				if err != nil {
					t.Fatal(err)
				}
			} else {
				if !errors.Is(err, failure) {
					t.Fatal("hidden cleanup failure", err)
				}
				var retained io.Closer
				if !errors.As(err, &retained) {
					t.Fatal("lost cleanup owner")
				}
				if err := retained.Close(); err != nil {
					t.Fatal(err)
				}
			}
			if owner.module != 0 || owner.buffer != nil {
				t.Fatal("owner not released")
			}
			if err := owner.Close(); err != nil {
				t.Fatal(err)
			}
			want := []string{"drain", "free", "unload"}
			switch fault {
			case "drain":
				want = []string{"drain", "drain", "free", "unload"}
			case "free":
				want = []string{"drain", "free", "free", "unload"}
			case "unload":
				want = []string{"drain", "free", "unload", "unload"}
			}
			if !reflect.DeepEqual(calls, want) {
				t.Fatalf("release order %v want %v", calls, want)
			}
		})
	}
}

func TestCUDAProbePartialCleanup(t *testing.T) {
	primary := errors.New("load or allocation failed")
	release := errors.New("release failed")
	for _, name := range []string{"load_zero", "load_partial", "allocation_zero", "allocation_partial"} {
		t.Run(name, func(t *testing.T) {
			drains, frees, unloads := 0, 0, 0
			owner := &cudaProbeResources{
				run:         func(fn func() error) error { return fn() },
				synchronize: func() error { drains++; return nil },
				unload: func(driver.Module) error {
					unloads++
					if name == "load_partial" && unloads == 1 {
						return release
					}
					return nil
				},
			}
			if name != "load_zero" {
				owner.module = 1
			}
			if name == "allocation_zero" {
				var buffer cudamem.Slice[float32]
				owner.buffer = buffer.Free
			}
			if name == "allocation_partial" {
				owner.buffer = func() error {
					frees++
					if frees == 1 {
						return release
					}
					return nil
				}
			}
			err := finishBackendCleanup(primary, owner)
			if !errors.Is(err, primary) {
				t.Fatal("lost primary error", err)
			}
			if name == "load_partial" || name == "allocation_partial" {
				if !errors.Is(err, release) || owner.module == 0 {
					t.Fatal("lost failed release owner", err)
				}
				var retained io.Closer
				if !errors.As(err, &retained) {
					t.Fatal("lost retryable cleanup")
				}
				if err := retained.Close(); err != nil {
					t.Fatal(err)
				}
			}
			if owner.module != 0 || owner.buffer != nil {
				t.Fatal("partial owner remains")
			}
			if name == "load_zero" && (drains != 0 || unloads != 0) {
				t.Fatal("zero module used")
			}
			if name == "allocation_partial" && (frees != 2 || unloads != 1) {
				t.Fatal("allocation cleanup order")
			}
		})
	}
}

func TestCUDAProbeZeroFreeMethods(t *testing.T) {
	var floats cudamem.Slice[float32]
	var integers cudamem.Slice[uint32]
	for _, free := range []func() error{floats.Free, integers.Free} {
		if err := free(); err != nil {
			t.Fatal(err)
		}
	}
}
