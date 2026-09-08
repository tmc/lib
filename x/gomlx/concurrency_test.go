package gomlx

import (
	"fmt"
	"os"
	"sync"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
)

func TestNativeConcurrentRuntime(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Finalize()
	builder := b.Builder("concurrent")
	f := builder.Main()
	shape := shapes.Make(dtypes.Float32, 1)
	p, err := f.Parameter("x", shape, nil)
	if err != nil {
		t.Fatal(err)
	}
	sum, err := f.Add(p, p)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Return([]compute.Value{sum}, nil); err != nil {
		t.Fatal(err)
	}
	e, err := builder.Compile()
	if err != nil {
		t.Fatal(err)
	}
	input, err := b.BufferFromFlatData(0, []float32{3}, shape)
	if err != nil {
		t.Fatal(err)
	}
	const workers = 8
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for range workers {
		wg.Go(func() {
			for range 10 {
				out, err := e.Execute([]compute.Buffer{input}, nil, 0)
				if err != nil {
					errs <- err
					return
				}
				dst := make([]float32, 1)
				err = out[0].ToFlatData(dst)
				closeErr := out[0].Finalize()
				if err != nil {
					errs <- err
					return
				}
				if closeErr != nil {
					errs <- closeErr
					return
				}
				if dst[0] != 6 {
					errs <- fmt.Errorf("got %v, want 6", dst)
					return
				}
			}
		})
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
	if t.Failed() {
		return
	}
	t.Log("80 executions across 8 goroutines returned exact result 6")

	// Race runtime calls against backend shutdown. Each operation must either
	// complete or reject the closed resource; no caller owns native state alone.
	start := make(chan struct{})
	for range workers {
		wg.Go(func() {
			<-start
			for range 10 {
				out, err := e.Execute([]compute.Buffer{input}, nil, 0)
				if err != nil {
					continue
				}
				dst := make([]float32, 1)
				if err := out[0].ToFlatData(dst); err == nil && dst[0] != 6 {
					t.Errorf("concurrent close result %v", dst)
				}
				if err := out[0].Finalize(); err != nil {
					t.Error(err)
				}
				_, _ = input.Shape()
			}
		})
	}
	wg.Go(func() { <-start; b.Finalize() })
	close(start)
	wg.Wait()
	if b.closeErr != nil {
		t.Fatal(b.closeErr)
	}
	if !b.IsFinalized() {
		t.Fatal("backend remains open")
	}
	if len(b.buffers) != 0 || len(b.executables) != 0 {
		t.Fatal("native resources remain")
	}
	t.Log("concurrent shutdown completed with no tracked resources")
}
