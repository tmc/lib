package cudafuse

import (
	"context"
	"errors"
	"fmt"
	"github.com/tmc/lib/nvidia/driver"
	"math"
	"reflect"
	"testing"
)

func TestMHCNonfiniteAdmission(t *testing.T) {
	for _, v := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		if mhcValidateValues([]float32{0, v, 1}) == nil {
			t.Fatal("nonfinite input accepted")
		}
	}
	if err := mhcValidateValues([]float32{-math.MaxFloat32, math.SmallestNonzeroFloat32, 0, math.MaxFloat32}); err != nil {
		t.Fatal(err)
	}
}

func TestMHCTwoStageCompletionOwnership(t *testing.T) {
	for _, mode := range []string{"healthy", "cancel-after-first", "second-launch-error", "failed-drain"} {
		t.Run(mode, func(t *testing.T) {
			var calls []string
			next := driver.DevicePtr(1000)
			allocs, frees, launches, syncs, released := 0, 0, 0, 0, false
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			fault := errors.New("device drain failed")
			ops := executionOps{
				alloc: func(n int) (*executionBuffer, error) {
					allocs++
					next += 1000
					return &executionBuffer{ptr: next, read: func(out []float32) error { calls = append(calls, fmt.Sprintf("read%d", n)); return nil }, free: func() error { frees++; calls = append(calls, "free"); return nil }}, nil
				},
				sync: func(ctx context.Context) error {
					syncs++
					calls = append(calls, "sync")
					if ctx.Err() != nil {
						return ctx.Err()
					}
					if mode == "failed-drain" && syncs <= 2 {
						return fault
					}
					return nil
				},
				compile: func(s FusedKernel) (*executionKernel, error) {
					calls = append(calls, "compile:"+s.Name)
					return &executionKernel{receipt: KernelReceipt{Name: s.Name}, close: func() error { calls = append(calls, "unload"); return nil }, launch: func(p []driver.DevicePtr, n int) error {
						launches++
						calls = append(calls, "launch:"+s.Name)
						if launches == 1 {
							if !reflect.DeepEqual(p, []driver.DevicePtr{10, 2000}) || n != 2 {
								t.Fatal("wrong normalization bindings", p, n)
							}
							if mode == "cancel-after-first" {
								cancel()
							}
						}
						if launches == 2 {
							if !reflect.DeepEqual(p, []driver.DevicePtr{20, 30, 40, 2000, 3000}) || n != 264 {
								t.Fatal("wrong post bindings", p, n)
							}
							if mode == "second-launch-error" {
								return fault
							}
						}
						return nil
					}}, nil
				},
			}
			e, err := mhcExecute(ctx, MHCConfig{2, 33, 10, 1e-6}, []driver.DevicePtr{10, 20, 30, 40}, ops, []func(){func() { released = true }})
			if mode == "healthy" {
				if err != nil {
					t.Fatal(err)
				}
				if _, err := e.Readback(ctx); err != nil {
					t.Fatal(err)
				}
				if launches != 2 || syncs != 1 {
					t.Fatal("wrong successful calls")
				}
			} else if err == nil {
				t.Fatal("fault or cancellation accepted")
			}
			if allocs != 2 || frees != 0 || released {
				t.Fatal("resources released before explicit close")
			}
			if mode == "failed-drain" {
				if err := e.Close(); !errors.Is(err, fault) {
					t.Fatal(err)
				}
				if frees != 0 || released {
					t.Fatal("failed drain freed live resources")
				}
			}
			if err := e.Close(); err != nil {
				t.Fatal(err)
			}
			if frees != 2 || !released {
				t.Fatal("ownership not discharged")
			}
			if err := e.Close(); err != nil || frees != 2 {
				t.Fatal("close not idempotent", err)
			}
			if e.Stats().Allocations != 2 || e.Stats().Frees != 2 || e.Stats().Launches != launches && mode != "second-launch-error" {
				t.Fatal(e.Stats())
			}
			t.Log(calls)
		})
	}
}
