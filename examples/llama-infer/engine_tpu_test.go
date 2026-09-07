//go:build linux && tpu

package main

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"
)

func skipIfNoTPU(t *testing.T, err error) {
	if err == nil {
		return
	}
	// Hardware testing gate: when RUN_TPU_HARDWARE_TESTS=1 is set, construction failures MUST fail the test.
	if os.Getenv("RUN_TPU_HARDWARE_TESTS") != "" {
		t.Fatalf("TPU engine initialization failed on enabled hardware test: %v", err)
	}

	// Only skip when libtpu is genuinely not found or no addressable devices are present on host
	if errors.Is(err, ErrLibTPUNotFound) || errors.Is(err, ErrNoAddressableTPUDevices) {
		t.Skipf("Skipping TPU test due to absent hardware/library: %v", err)
	}
	// Any ABI mismatch, compilation failure, or unexpected initialization error MUST fail the test!
	t.Fatalf("TPU engine initialization failed unexpectedly: %v", err)
}

func TestTPUEngineAvailabilityClassifier(t *testing.T) {
	// Unit-test the availability classifier separately without hardware
	tests := []struct {
		name      string
		err       error
		wantSkip  bool
		wantFatal bool
	}{
		{
			name:     "libtpu_not_found",
			err:      ErrLibTPUNotFound,
			wantSkip: true,
		},
		{
			name:     "no_devices",
			err:      ErrNoAddressableTPUDevices,
			wantSkip: true,
		},
		{
			name:      "abi_or_compile_error",
			err:       errors.New("compile stablehlo: syntax error in MLIR"),
			wantFatal: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			isAvailable := errors.Is(tt.err, ErrLibTPUNotFound) || errors.Is(tt.err, ErrNoAddressableTPUDevices)
			if tt.wantSkip && !isAvailable {
				t.Errorf("expected error %v to be recognized as absent hardware/library", tt.err)
			}
			if tt.wantFatal && isAvailable {
				t.Errorf("expected error %v NOT to be classified as absent hardware", tt.err)
			}
		})
	}
}

func TestTPUEngineGenerationUnsupported(t *testing.T) {
	// Test directly on a zero ModelEngine without requiring hardware or library presence
	var eng ModelEngine
	cfg := DefaultTinyLlamaConfig()

	err := eng.Generate(context.Background(), cfg, "test", 10, func(id int32, text string) bool {
		return true
	})
	if err == nil {
		t.Fatal("Expected Generate to return unsupported error on TPU backend")
	}
	if !strings.Contains(err.Error(), "unsupported") {
		t.Fatalf("Expected unsupported error, got %v", err)
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

	if err := eng.ForwardPassMockTPU(); err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}
}

func TestTPUEngineSmokeMatmul(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	err := RunTPUSmokeTest(cfg)
	skipIfNoTPU(t, err)
}
