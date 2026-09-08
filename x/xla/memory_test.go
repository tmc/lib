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
