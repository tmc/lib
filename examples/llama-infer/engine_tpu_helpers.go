package main

import (
	"errors"
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/tmc/lib/openxla"
)

var (
	ErrLibTPUNotFound          = errors.New("tpu: libtpu library not found")
	ErrNoAddressableTPUDevices = errors.New("openxla: no addressable tpu devices found")
)

// classifyTPULoadError distinguishes genuinely absent library files from corruption or dependency failures.
func classifyTPULoadError(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("%w: %v", ErrLibTPUNotFound, err)
	}

	// Examine underlying error stripped of our wrapper prefix
	msg := strings.ToLower(err.Error())
	const wrapperPrefix = "tpu: failed to load libtpu.so: "
	underlying := msg
	if idx := strings.Index(msg, wrapperPrefix); idx != -1 {
		underlying = msg[idx+len(wrapperPrefix):]
	}

	// If the underlying error mentions a missing secondary dependency (e.g. "libfoo.so: cannot open shared object file"
	// where the missing library is not libtpu.so) or mentions dynamic linker resolution failures ("library not loaded"),
	// this is a dependency resolution failure of an existing binary, NOT an absent libtpu plugin!
	if strings.Contains(underlying, "library not loaded") ||
		strings.Contains(underlying, "referenced from:") ||
		strings.Contains(underlying, "reason: tried:") {
		return err
	}

	if strings.Contains(underlying, "cannot open shared object file") {
		// Example: "libdependency.so: cannot open shared object file: no such file or directory"
		// Only consider it absent libtpu if the missing file is specifically libtpu.so
		// (e.g. "cannot open shared object file: no such file or directory" or "libtpu.so: cannot open shared object file")
		target := underlying
		if colon := strings.Index(underlying, ":"); colon != -1 {
			target = strings.TrimSpace(underlying[:colon])
		}
		if target != "cannot open shared object file" && !strings.Contains(target, "libtpu") {
			return err
		}
		return fmt.Errorf("%w: %v", ErrLibTPUNotFound, err)
	}

	// Check for dlopen image not found / no such file where the target is libtpu
	if strings.Contains(underlying, "image not found") && strings.Contains(underlying, "libtpu") {
		return fmt.Errorf("%w: %v", ErrLibTPUNotFound, err)
	}
	if strings.Contains(underlying, "no such file or directory") && (strings.Contains(underlying, "libtpu") || !strings.Contains(underlying, ".so")) {
		return fmt.Errorf("%w: %v", ErrLibTPUNotFound, err)
	}

	return err
}

// validateOutputCardinality validates that execution produced exactly 1 device output list with 1 buffer.
func validateOutputCardinality(outputs [][]*openxla.BufferWrapper) error {
	if len(outputs) != 1 {
		return fmt.Errorf("unexpected device output cardinality: got %d device result lists, want 1", len(outputs))
	}
	if len(outputs[0]) != 1 {
		return fmt.Errorf("unexpected per-device output cardinality: got %d outputs on device 0, want 1", len(outputs[0]))
	}
	return nil
}

// checkMatmulOutput validates the shape, finite values, and numerical accuracy against an independent oracle.
func checkMatmulOutput(got, expected []float32) error {
	if len(got) != len(expected) {
		return fmt.Errorf("output length mismatch: got %d, want %d", len(got), len(expected))
	}
	for i := range expected {
		val := got[i]
		if math.IsNaN(float64(val)) {
			return fmt.Errorf("matmul output contains NaN at index %d", i)
		}
		if math.IsInf(float64(val), 0) {
			return fmt.Errorf("matmul output contains Inf (%f) at index %d", val, i)
		}
		diff := val - expected[i]
		if diff < -1e-4 || diff > 1e-4 {
			return fmt.Errorf("matmul numerical mismatch at index %d: got %f, want %f", i, val, expected[i])
		}
	}
	return nil
}
