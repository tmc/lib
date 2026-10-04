package cudafuse

import (
	"context"
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

func TestMHCFusedCompletionOwnership(t *testing.T) {
	for _, mode := range []string{"healthy", "launch-error", "failed-drain"} {
		t.Run(mode, func(t *testing.T) {
			frees, syncs, unloads, launches := 0, 0, 0, 0
			released := false
			fault := errors.New("fused drain failed")
			ops := executionOps{alloc: func(n int) (*executionBuffer, error) {
				if n != 264 {
					t.Fatal(n)
				}
				return &executionBuffer{ptr: 1000, read: func([]float32) error { return nil }, free: func() error { frees++; return nil }}, nil
			}, sync: func(context.Context) error {
				syncs++
				if mode == "failed-drain" && syncs <= 2 {
					return fault
				}
				return nil
			}, compile: func(s FusedKernel) (*executionKernel, error) {
				if s.Name != "mhc_fused_fp32" {
					t.Fatal(s.Name)
				}
				return &executionKernel{close: func() error { unloads++; return nil }, launch: func(ptrs []driver.DevicePtr, n int) error {
					launches++
					if len(ptrs) != 5 || ptrs[4] != 1000 {
						t.Fatal(ptrs)
					}
					if mode == "launch-error" {
						return fault
					}
					return nil
				}}, nil
			}}
			e, err := mhcFusedExecute(context.Background(), MHCConfig{2, 33, 10, 1e-6}, []driver.DevicePtr{10, 20, 30, 40}, ops, []func(){func() { released = true }})
			if mode == "healthy" && err != nil || mode != "healthy" && err == nil {
				t.Fatal(err)
			}
			if frees != 0 || unloads != 0 || released || launches != 1 {
				t.Fatal("premature release")
			}
			if mode == "failed-drain" {
				if err := e.Close(); !errors.Is(err, fault) {
					t.Fatal(err)
				}
				if frees != 0 || unloads != 0 || released {
					t.Fatal("failed drain released owners")
				}
			}
			if err := e.Close(); err != nil {
				t.Fatal(err)
			}
			if frees != 1 || unloads != 1 || !released {
				t.Fatal("fused ownership not discharged")
			}
			if err := e.Close(); err != nil || frees != 1 || unloads != 1 {
				t.Fatal("close not idempotent", err)
			}
		})
	}
}
