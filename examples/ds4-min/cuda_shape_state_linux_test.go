//go:build linux

package main

import (
	"errors"
	"io"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func TestShapeStateCloseRetainsBorrowGraph(t *testing.T) {
	blocked := true
	cause := errors.New("consumer drain fault")
	s := &cudaShapeDecode{
		consumers: &cudaShapeConsumers{modules: map[string]driver.Module{"entry": 1}, entries: map[string]driver.Function{"entry": 1}, ops: shapeConsumerOperations{
			drain: func() error {
				if blocked {
					return cause
				}
				return nil
			},
			closeRun: func(f func() error) error { return f() },
			unload:   func(driver.Module) error { return nil },
		}},
		mix: new(cudaShapeHCMix), hc: new(cudaShapeHC), forward: new(cudaShapeForward),
	}
	if err := s.Close(); !errors.Is(err, cause) {
		t.Fatalf("close: %v", err)
	}
	if !s.closed || !s.consumers.closed || s.mix.closed || s.hc.closed || s.forward.closed || s.consumers.modules["entry"] != 1 {
		t.Fatal("child failure released a borrowed dependency")
	}
	blocked = false
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if !s.mix.closed || !s.hc.closed || !s.forward.closed || s.consumers.modules["entry"] != 0 {
		t.Fatal("retry did not complete ordered cleanup")
	}
}

func TestShapeStatePartialConstruction(t *testing.T) {
	for _, stage := range []string{"forward", "consumers", "hc", "mix", "attach"} {
		t.Run(stage, func(t *testing.T) {
			cause := errors.New("construction fault")
			blocked := true
			freed := 0
			forward := &cudaShapeForward{scratch: map[string]cudamem.Slice[float32]{"cur": cacheTestSlice(t, 7168*4, 1)}, ops: shapeForwardOperations{
				drain: func() error {
					if blocked {
						return cause
					}
					return nil
				},
				free: func(v *cudamem.Slice[float32]) error { freed++; *v = cudamem.Slice[float32]{}; return nil },
			}}
			ops := shapeStateOperations{
				forward: func() (*cudaShapeForward, error) {
					if stage == "forward" {
						return forward, cause
					}
					return forward, nil
				},
				consumers: func() (*cudaShapeConsumers, error) {
					if stage == "consumers" {
						return nil, cause
					}
					return new(cudaShapeConsumers), nil
				},
				hc: func() (*cudaShapeHC, error) {
					if stage == "hc" {
						return nil, cause
					}
					return new(cudaShapeHC), nil
				},
				mix: func() (*cudaShapeHCMix, error) {
					if stage == "mix" {
						return nil, cause
					}
					return new(cudaShapeHCMix), nil
				},
			}
			s, err := newCUDAShapeStateWithOperations(nil, shapeDecodeWeights{}, ops)
			var owner io.Closer
			if s != nil || err == nil || !errors.As(err, &owner) || freed != 0 {
				t.Fatalf("partial state lost: state=%v error=%v freed=%d", s, err, freed)
			}
			blocked = false
			if err := owner.Close(); err != nil {
				t.Fatal(err)
			}
			if err := owner.Close(); err != nil {
				t.Fatal(err)
			}
			if freed != 1 {
				t.Fatalf("free count=%d want=1", freed)
			}
		})
	}
}

func TestShapeStateMissingConstructor(t *testing.T) {
	if s, err := newCUDAShapeStateWithOperations(nil, shapeDecodeWeights{}, shapeStateOperations{}); s != nil || err == nil {
		t.Fatalf("state=%v error=%v", s, err)
	}
}
