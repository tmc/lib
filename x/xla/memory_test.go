package xla_test

import (
	"fmt"
	"os"
	"testing"

	"github.com/tmc/lib/x/xla"
)

func ExampleEngine_MemoryStats() {
	var engine xla.Engine
	_, err := engine.MemoryStats(nil)
	fmt.Println(err)
	// Output: xla: engine is closed
}

// TestNativeMemoryStats records plugin support without treating an unsupported
// native call as a successful allocator observation.
func TestNativeMemoryStats(t *testing.T) {
	path := os.Getenv("XLA_PJRT_MEMORY_STATS_PLUGIN")
	if path == "" {
		t.Skip("set XLA_PJRT_MEMORY_STATS_PLUGIN to a PJRT library with memory statistics")
	}
	engine, err := xla.NewEngine(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := engine.Close(); err != nil {
			t.Error(err)
		}
	}()
	devices, err := engine.Devices()
	if err != nil || len(devices) == 0 {
		t.Fatalf("devices=%v, error=%v", devices, err)
	}
	stats, err := engine.MemoryStats(devices[0])
	if err != nil {
		t.Fatalf("native allocator statistics unavailable: %v", err)
	}
	t.Logf("bytes in use: %d; optional counters: %+v", stats.BytesInUse, stats)
}

func ExampleExecutable_CompiledMemoryStats() {
	var executable xla.Executable
	_, err := executable.CompiledMemoryStats()
	fmt.Println(err)
	// Output: xla: executable is closed
}

func TestNativeCompiledMemoryStats(t *testing.T) {
	path := os.Getenv("XLA_PJRT_CPU_PLUGIN")
	if path == "" {
		t.Skip("set XLA_PJRT_CPU_PLUGIN to a CPU PJRT library")
	}
	engine, err := xla.NewEngine(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := engine.Close(); err != nil {
			t.Error(err)
		}
	}()
	program := []byte(`module {
  func.func @main(%a: tensor<2x4xf32>, %b: tensor<4x3xf32>) -> tensor<2x3xf32> {
    %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<2x4xf32>, tensor<4x3xf32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}`)
	executable, err := engine.Compile(program, "mlir", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := executable.Close(); err != nil {
			t.Error(err)
		}
	}()
	for i := 0; i < 2; i++ {
		stats, err := executable.CompiledMemoryStats()
		if err != nil {
			t.Fatal(err)
		}
		if stats.ArgumentBytes != 80 || stats.OutputBytes != 24 {
			t.Fatalf("got %+v, want 80 argument bytes and 24 output bytes", stats)
		}
		t.Logf("compiler estimates: %+v", stats)
	}
}
