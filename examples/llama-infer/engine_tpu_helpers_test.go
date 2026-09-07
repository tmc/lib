package main

import (
	"errors"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/openxla"
)

func TestClassifyTPULoadError(t *testing.T) {
	tests := []struct {
		name         string
		inputErr     error
		wantNotFound bool
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
