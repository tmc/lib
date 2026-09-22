package cublasltx

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/cublaslt"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func TestEngineConstructionOwnership(t *testing.T) {
	failure := errors.New("injected construction failure")
	for _, stage := range []string{"entry", "create", "partial", "restore", "workspace", "success"} {
		t.Run(stage, func(t *testing.T) {
			size := 0
			if stage == "workspace" {
				size = 256
			}
			dev := &cuda.Device{}
			allocated := false
			e, err := newEngine(dev, engineConfig{workspaceBytes: &size}, func(fn func() error) error {
				if stage == "entry" {
					return failure
				}
				if err := fn(); err != nil {
					return err
				}
				if stage == "restore" {
					return failure
				}
				return nil
			}, func() (cublaslt.Handle, error) {
				if stage == "create" {
					return 0, failure
				}
				if stage == "partial" {
					return 7, failure
				}
				return 7, nil
			}, func(bytes int) (cudamem.Slice[byte], error) {
				allocated = true
				if stage != "workspace" || bytes != 256 {
					t.Fatal("unexpected allocation")
				}
				return cudamem.Slice[byte]{}, failure
			})
			if errors.Is(err, failure) != (stage != "success") {
				t.Fatalf("error=%v", err)
			}
			if allocated != (stage == "workspace") {
				t.Fatal("allocation after failed creation")
			}
			owned := stage != "entry" && stage != "create"
			if (e != nil) != owned {
				t.Fatal("lost partial engine owner")
			}
			if e == nil {
				return
			}
			if e.dev != dev || e.handle != 7 {
				t.Fatal("wrong engine identity")
			}
			destroys := 0
			for range 2 {
				if err := e.closeWith(func(fn func() error) error { return fn() }, func(h cublaslt.Handle) error {
					if h != 7 {
						t.Fatal("wrong destroyed handle")
					}
					destroys++
					return nil
				}, func() error { t.Fatal("free without allocation"); return nil }); err != nil {
					t.Fatal(err)
				}
			}
			if destroys != 1 {
				t.Fatalf("destroy count=%d", destroys)
			}
		})
	}
}
