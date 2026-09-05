//go:build linux

package main

import (
	"context"
	"crypto/sha256"
	"fmt"
	"math"
	"os"
	"runtime"
	"testing"
)

// cudaQualificationMeasure brackets one actual launch with device events.
// These durations establish execution, not a performance comparison.
func cudaQualificationMeasure(t *testing.T, r *cudaFoundation, launch func() error) {
	t.Helper()
	if err := r.device.WithContext(r.loadMathKernels); err != nil {
		t.Fatal(err)
	}
	start, err := r.device.NewEvent()
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := start.Destroy(); err != nil {
			t.Error(err)
		}
	}()
	end, err := r.device.NewEvent()
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := end.Destroy(); err != nil {
			t.Error(err)
		}
	}()
	stream := r.device.DefaultStream()
	if err := start.Record(stream); err != nil {
		t.Fatal(err)
	}
	if err := launch(); err != nil {
		t.Fatal(err)
	}
	if err := end.Record(stream); err != nil {
		t.Fatal(err)
	}
	if err := end.Synchronize(context.Background()); err != nil {
		t.Fatal(err)
	}
	elapsed, err := end.ElapsedSince(start)
	if err != nil {
		t.Fatal(err)
	}
	if elapsed <= 0 {
		t.Fatalf("device event duration %s does not establish execution", elapsed)
	}
	t.Logf("CUDA device event duration: %s", elapsed)
}

func TestCUDAQualificationCompile(t *testing.T) {
	r := cudaSynthFoundation(t)
	host, err := os.Hostname()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("host=%s runtime=%s os=%s arch=%s CUDA source SHA256=%x", host, runtime.Version(), runtime.GOOS, runtime.GOARCH, sha256.Sum256([]byte(cudaMathSource)))
	if err := r.device.WithContext(r.loadMathKernels); err != nil {
		t.Fatal(err)
	}
	for _, kernel := range r.cudaKernels() {
		if *kernel.dst == 0 {
			t.Fatalf("kernel %s has no loaded function", kernel.name)
		}
		t.Logf("compiled and loaded: %s", kernel.name)
	}
}

// cudaQualificationClose rejects invalid samples before applying the tolerance
// registered by each physical differential. Empty comparisons prove nothing.
func cudaQualificationClose(got, want []float32, atol, rtol float64) error {
	if math.IsNaN(atol) || math.IsInf(atol, 0) || atol < 0 ||
		math.IsNaN(rtol) || math.IsInf(rtol, 0) || rtol < 0 {
		return fmt.Errorf("invalid tolerance: atol=%g rtol=%g", atol, rtol)
	}
	if len(got) == 0 || len(got) != len(want) {
		return fmt.Errorf("sample lengths: got %d, want %d nonempty values", len(got), len(want))
	}
	for i, value := range got {
		g, w := float64(value), float64(want[i])
		if math.IsNaN(g) || math.IsInf(g, 0) || math.IsNaN(w) || math.IsInf(w, 0) {
			return fmt.Errorf("sample %d: non-finite values %g, %g", i, g, w)
		}
		limit := atol + rtol*math.Abs(w)
		if math.IsInf(limit, 0) {
			return fmt.Errorf("sample %d: tolerance overflow", i)
		}
		if delta := math.Abs(g - w); delta > limit {
			return fmt.Errorf("sample %d: got %g, want %g (delta %g > %g)", i, g, w, delta, limit)
		}
	}
	return nil
}
