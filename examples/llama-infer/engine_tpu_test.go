//go:build tpu

package main

import (
	"context"
	"testing"
)

func TestTPUEngineCompilation(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	// Skip actual hardware initialization if libtpu.so is missing, but verify interface.
	eng, err := GetOrCompileTPUEngine(cfg)
	if err != nil {
		t.Skipf("Skipping TPU engine test due to missing hardware or libtpu: %v", err)
	}
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
	if err != nil {
		t.Skipf("Skipping TPU engine test: %v", err)
	}

	err = eng.Generate(context.Background(), cfg, "test", 10, func(id int32, text string) bool {
		return true
	})
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
}
