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
		// Only skip if the dynamic library load failed because the file was not found or no hardware devices are present.
		// Any ABI mismatch, compilation error, or initialization bug must fail the test.
		if strings.Contains(msg, "load libtpu: tpu: failed to load libtpu.so") ||
			strings.Contains(msg, "no addressable tpu devices") {
			t.Skipf("Skipping TPU engine test due to missing hardware or libtpu library: %v", err)
		}
		t.Fatalf("TPU engine initialization failed unexpectedly: %v", err)
	}
}

func TestTPUEngineCompilation(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	eng, err := GetOrCompileTPUEngine(cfg)
	skipIfNoTPU(t, err)
	if eng == nil {
		t.Fatal("Expected non-nil engine")
	}
	defer eng.Free()

	// Verify StableHLO graph compilation and mock forward pass execution
	if err := eng.ForwardPassMockTPU(); err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}
}

func TestTPUEngineSmokeMatmul(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	err := RunTPUSmokeTest(cfg)
	skipIfNoTPU(t, err)
}

func TestTPUEngineGenerationUnsupported(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	eng, err := GetOrCompileTPUEngine(cfg)
	skipIfNoTPU(t, err)
	defer eng.Free()

	err = eng.Generate(context.Background(), cfg, "test", 10, func(id int32, text string) bool {
		return true
	})
	if err == nil {
		t.Fatal("Expected Generate to return unsupported error on TPU backend")
	}
	if !strings.Contains(err.Error(), "unsupported") {
		t.Fatalf("Expected unsupported error, got %v", err)
	}
}
