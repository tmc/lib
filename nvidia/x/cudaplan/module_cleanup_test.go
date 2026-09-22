package cudaplan

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

func TestModuleLoadOwnership(t *testing.T) {
	failure := errors.New("injected")
	for _, stage := range []string{"load", "own", "success"} {
		t.Run(stage, func(t *testing.T) {
			p := &ExecutablePlan{modules: make(map[*cudakernel.Artifact]*cuda.Module)}
			art := &cudakernel.Artifact{}
			owner := &cuda.Module{}
			err := p.loadModule(art, func() (driver.Module, error) {
				if stage == "load" {
					return 7, failure
				}
				return 7, nil
			}, func(raw driver.Module) (*cuda.Module, error) {
				if stage == "load" {
					t.Fatal("adopted failed load")
				}
				if raw != 7 {
					t.Fatalf("raw = %v", raw)
				}
				if stage == "own" {
					return nil, failure
				}
				return owner, nil
			})
			if stage == "success" {
				if err != nil || p.modules[art] != owner || len(p.pendingModules) != 0 {
					t.Fatal("ownership not transferred")
				}
				return
			}
			if !errors.Is(err, failure) || len(p.pendingModules) != 1 || p.pendingModules[0] != 7 {
				t.Fatal("lost failed-load module")
			}
		})
	}
}

func TestPendingModuleCleanup(t *testing.T) {
	failure := errors.New("injected")
	for _, stage := range []string{"entry", "unload", "restore"} {
		t.Run(stage, func(t *testing.T) {
			p := &ExecutablePlan{pendingModules: []driver.Module{7}}
			calls := 0
			ops := planCloseOps{
				run: func(fn func() error) error {
					if stage == "entry" {
						return failure
					}
					if err := fn(); err != nil {
						return err
					}
					return failure
				},
				unload: func(raw driver.Module) error {
					if raw != 7 {
						t.Fatalf("raw = %v", raw)
					}
					calls++
					if stage == "unload" {
						return failure
					}
					return nil
				},
			}
			if err := p.closeWith(ops); !errors.Is(err, failure) {
				t.Fatalf("error = %v", err)
			}
			if (len(p.pendingModules) != 0) != (stage != "restore") {
				t.Fatal("incorrect retained state")
			}
			ops.run = func(fn func() error) error { return fn() }
			ops.unload = func(driver.Module) error { calls++; return nil }
			for range 2 {
				if err := p.closeWith(ops); err != nil {
					t.Fatal(err)
				}
			}
			want := 1
			if stage == "unload" {
				want = 2
			}
			if calls != want || len(p.pendingModules) != 0 {
				t.Fatalf("calls = %d, want %d", calls, want)
			}
		})
	}
}
