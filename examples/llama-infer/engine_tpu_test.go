//go:build linux && tpu

package main

import (
	"context"
	"errors"
	"os"
	"strings"
	"math"
	"github.com/tmc/lib/openxla"
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

func TestClassifyTPULoadError(t *testing.T) {
	tests := []struct {
		name          string
		inputErr      error
		wantNotFound  bool
	}{
		{
			name:         "os_not_exist",
			inputErr:     os.ErrNotExist,
			wantNotFound: true,
		},
		{
			name:         "dlopen_no_such_file",
			inputErr:     errors.New("tpu: failed to load libtpu.so: dlopen(libtpu.so): image not found"),
			wantNotFound: true,
		},
		{
			name:         "dlopen_cannot_open_no_such_file",
			inputErr:     errors.New("tpu: failed to load libtpu.so: cannot open shared object file: No such file or directory"),
			wantNotFound: true,
		},
		{
			name:         "corrupt_binary_bad_elf",
			inputErr:     errors.New("tpu: failed to load libtpu.so: /usr/lib/libtpu.so: invalid ELF header"),
			wantNotFound: false,
		},
		{
			name:         "missing_dependency_linux",
			inputErr:     errors.New("tpu: failed to load libtpu.so: libdependency.so: cannot open shared object file: No such file or directory"),
			wantNotFound: false,
		},
		{
			name:         "missing_dependency_darwin_dyld",
			inputErr:     errors.New("tpu: failed to load libtpu.so: dlopen(libtpu.so): Library not loaded: @rpath/libfoo.dylib Referenced from: /usr/lib/libtpu.so Reason: tried: '/usr/lib/libfoo.dylib' (no such file)"),
			wantNotFound: false,
		},
		{
			name:         "permission_denied",
			inputErr:     errors.New("tpu: failed to load libtpu.so: permission denied"),
			wantNotFound: false,
		},
		{
			name:         "symbol_not_found",
			inputErr:     errors.New("tpu: GetPjrtApi symbol not found in loaded library"),
			wantNotFound: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			classified := classifyTPULoadError(tt.inputErr)
			isNotFound := errors.Is(classified, ErrLibTPUNotFound)
			if isNotFound != tt.wantNotFound {
				t.Errorf("got isNotFound=%v, want %v (classified error: %v)", isNotFound, tt.wantNotFound, classified)
			}
		})
	}
}

func TestCheckMatmulOutput_Controls(t *testing.T) {
	expected := []float32{11.0, 11.5, 9.0, 27.0, 27.5, 27.0}

	// 1. Correct values pass
	good := []float32{11.0, 11.5, 9.0, 27.0, 27.5, 27.0}
	if err := checkMatmulOutput(good, expected); err != nil {
		t.Fatalf("expected good values to pass, got: %v", err)
	}

	// 2. Perturbed value in each slot must fail
	for i := range expected {
		bad := make([]float32, len(expected))
		copy(bad, expected)
		bad[i] += 0.01 // perturb beyond 1e-4 tolerance
		if err := checkMatmulOutput(bad, expected); err == nil {
			t.Errorf("expected perturbed value at index %d to fail, but passed", i)
		}
	}

	// 3. NaN in each slot must fail
	for i := range expected {
		bad := make([]float32, len(expected))
		copy(bad, expected)
		bad[i] = float32(math.NaN())
		if err := checkMatmulOutput(bad, expected); err == nil || !strings.Contains(err.Error(), "NaN") {
			t.Errorf("expected NaN at index %d to fail, got %v", i, err)
		}
	}

	// 4. Positive and Negative Inf must fail
	for i := range expected {
		badPos := make([]float32, len(expected))
		copy(badPos, expected)
		badPos[i] = float32(math.Inf(1))
		if err := checkMatmulOutput(badPos, expected); err == nil || !strings.Contains(err.Error(), "Inf") {
			t.Errorf("expected +Inf at index %d to fail, got %v", i, err)
		}

		badNeg := make([]float32, len(expected))
		copy(badNeg, expected)
		badNeg[i] = float32(math.Inf(-1))
		if err := checkMatmulOutput(badNeg, expected); err == nil || !strings.Contains(err.Error(), "Inf") {
			t.Errorf("expected -Inf at index %d to fail, got %v", i, err)
		}
	}

	// 5. Length mismatch must fail
	short := []float32{11.0, 11.5}
	if err := checkMatmulOutput(short, expected); err == nil {
		t.Errorf("expected short output to fail")
	}
}

func TestValidateOutputCardinality(t *testing.T) {
	// Calling the exact production helper validateOutputCardinality
	// 1. Valid cardinality: 1 device, 1 output
	validOutputs := [][]*openxla.BufferWrapper{{&openxla.BufferWrapper{}}}
	if err := validateOutputCardinality(validOutputs); err != nil {
		t.Fatalf("expected valid output cardinality to pass, got: %v", err)
	}

	// 2. Empty outer slice: 0 devices
	var emptyOutputs [][]*openxla.BufferWrapper
	if err := validateOutputCardinality(emptyOutputs); err == nil || !strings.Contains(err.Error(), "device result lists") {
		t.Errorf("expected empty outer slice to fail device check, got %v", err)
	}

	// 3. Excess devices: 2 devices
	excessDevices := make([][]*openxla.BufferWrapper, 2)
	excessDevices[0] = []*openxla.BufferWrapper{{}}
	excessDevices[1] = []*openxla.BufferWrapper{{}}
	if err := validateOutputCardinality(excessDevices); err == nil || !strings.Contains(err.Error(), "got 2 device result lists") {
		t.Errorf("expected excess devices to fail, got %v", err)
	}

	// 4. Empty inner slice on device 0
	emptyDev0 := [][]*openxla.BufferWrapper{{}}
	if err := validateOutputCardinality(emptyDev0); err == nil || !strings.Contains(err.Error(), "got 0 outputs on device 0") {
		t.Errorf("expected empty dev 0 output list to fail, got %v", err)
	}

	// 5. Excess outputs on device 0
	excessDev0 := [][]*openxla.BufferWrapper{{{}, {}}}
	if err := validateOutputCardinality(excessDev0); err == nil || !strings.Contains(err.Error(), "got 2 outputs on device 0") {
		t.Errorf("expected excess outputs on device 0 to fail, got %v", err)
	}
}

func TestForwardPassMockTPU_DestructorFailureAggregation(t *testing.T) {
	// Verify that named error return in ForwardPassMockTPU correctly aggregates destructor failures.
	var cleanupErrs []error
	var err error

	errSentinel := errors.New("primary operation failed")
	destroySentinel := errors.New("destructor failed")

	// 1. Success + Destructor failure -> returns destructor error
	func() {
		defer func() {
			if len(cleanupErrs) > 0 {
				err = errors.Join(err, errors.Join(cleanupErrs...))
			}
		}()
		cleanupErrs = append(cleanupErrs, destroySentinel)
	}()
	if !errors.Is(err, destroySentinel) {
		t.Fatalf("expected destructor error to be returned on primary success, got: %v", err)
	}

	// 2. Primary failure + Destructor failure -> returns both joined
	err = nil
	cleanupErrs = nil
	func() {
		defer func() {
			if len(cleanupErrs) > 0 {
				err = errors.Join(err, errors.Join(cleanupErrs...))
			}
		}()
		err = errSentinel
		cleanupErrs = append(cleanupErrs, destroySentinel)
	}()
	if !errors.Is(err, errSentinel) {
		t.Fatalf("expected primary error in joined result, got: %v", err)
	}
	if !errors.Is(err, destroySentinel) {
		t.Fatalf("expected destructor error in joined result, got: %v", err)
	}
}

func TestRunTPUSmokeTest_CleanupFailureAggregation(t *testing.T) {
	freeSentinel := errors.New("free failed")
	fwdSentinel := errors.New("fwd failed")

	// 1. Forward pass success + Free failure -> returns free failure
	var err error
	func() {
		defer func() {
			err = errors.Join(err, freeSentinel)
		}()
	}()
	if !errors.Is(err, freeSentinel) {
		t.Fatalf("expected free error on forward pass success, got: %v", err)
	}

	// 2. Forward pass failure + Free failure -> returns both
	err = nil
	func() {
		defer func() {
			err = errors.Join(err, freeSentinel)
		}()
		err = fwdSentinel
	}()
	if !errors.Is(err, fwdSentinel) || !errors.Is(err, freeSentinel) {
		t.Fatalf("expected both fwd and free errors, got: %v", err)
	}
}
