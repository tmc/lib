package metalx_test

import (
	"errors"
	"sync"
	"testing"

	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalx"
)

const sampleKernel = `
#include <metal_stdlib>
using namespace metal;

kernel void VectorAdd(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c       [[buffer(2)]],
    uint id               [[thread_position_in_grid]]
) {
    c[id] = a[id] + b[id];
}
`

func TestCompile(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Skipf("Metal device not available: %v", err)
	}

	compiler, err := metalx.NewCompiler(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer compiler.Close()
	pso, err := compiler.Compile(sampleKernel, "VectorAdd")
	if err != nil {
		t.Fatalf("metalx.Compile failed: %v", err)
	}
	if pso.GetID() == 0 {
		t.Fatal("expected valid PSO handle")
	}

	// Test cache hit
	pso2, err := compiler.Compile(sampleKernel, "VectorAdd")
	if err != nil {
		t.Fatalf("cached compile failed: %v", err)
	}
	if pso2.GetID() != pso.GetID() {
		t.Errorf("expected cached PSO ID %v, got %v", pso.GetID(), pso2.GetID())
	}
}

func TestCompilerClose(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Skipf("Metal device not available: %v", err)
	}
	compiler, err := metalx.NewCompiler(dev)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := compiler.Compile(sampleKernel, "VectorAdd"); err != nil {
		t.Fatal(err)
	}
	if err := compiler.Close(); err != nil {
		t.Fatal(err)
	}
	if err := compiler.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := compiler.Compile(sampleKernel, "VectorAdd"); !errors.Is(err, metalx.ErrClosed) {
		t.Fatalf("Compile after Close = %v, want ErrClosed", err)
	}
}

func TestCompilerConcurrentCompileClose(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Skipf("Metal device not available: %v", err)
	}
	compiler, err := metalx.NewCompiler(dev)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	errs := make(chan error, 16)
	for range 16 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := compiler.Compile(sampleKernel, "VectorAdd")
			errs <- err
		}()
	}
	if err := compiler.Close(); err != nil {
		t.Fatal(err)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil && !errors.Is(err, metalx.ErrClosed) {
			t.Errorf("Compile = %v", err)
		}
	}
}

func TestCompilerRejectsNil(t *testing.T) {
	if _, err := metalx.NewCompiler(nil); err == nil {
		t.Fatal("NewCompiler(nil) succeeded")
	}
	var compiler *metalx.Compiler
	if _, err := compiler.Compile(sampleKernel, "VectorAdd"); err == nil {
		t.Fatal("nil Compiler.Compile succeeded")
	}
	if err := compiler.Close(); err != nil {
		t.Fatalf("nil Compiler.Close: %v", err)
	}
}

func TestCompilerOwnersAreIndependent(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Skipf("Metal device not available: %v", err)
	}
	a, err := metalx.NewCompiler(dev)
	if err != nil {
		t.Fatal(err)
	}
	b, err := metalx.NewCompiler(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	if _, err := a.Compile(sampleKernel, "VectorAdd"); err != nil {
		t.Fatal(err)
	}
	if _, err := b.Compile(sampleKernel, "VectorAdd"); err != nil {
		t.Fatal(err)
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := b.Compile(sampleKernel, "VectorAdd"); err != nil {
		t.Fatalf("second owner after first Close: %v", err)
	}
}
