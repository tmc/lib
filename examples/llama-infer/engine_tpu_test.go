//go:build tpu

package main

import (
	"context"
	"strings"
	"testing"
)


func skipIfNoTPU(t *testing.T, err error) {
	if err != nil {
		msg := err.Error()
		if strings.Contains(msg, "load libtpu") || strings.Contains(msg, "no addressable tpu devices") || strings.Contains(msg, "libtpu") {
			t.Skipf("Skipping TPU engine test due to missing hardware or libtpu: %v", err)
		}
		t.Fatalf("TPU engine initialization failed: %v", err)
	}
}

func TestTPUEngineCompilation(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	// Skip actual hardware initialization if libtpu.so is missing, but verify interface.
	eng, err := GetOrCompileTPUEngine(cfg)
	skipIfNoTPU(t, err)
	if eng == nil {
		t.Fatal("Expected non-nil engine")
	}

	// Verify StableHLO graph compilation and buffer execution
	if err := eng.ForwardPassMockTPU(); err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}
}

func TestTPUEngineGeneration(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	eng, err := GetOrCompileTPUEngine(cfg)
	skipIfNoTPU(t, err)

	err = eng.Generate(context.Background(), cfg, "test", 10, func(id int32, text string) bool {
		return true
	})
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
}
