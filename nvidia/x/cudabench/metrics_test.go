package cudabench

import (
	"math"
	"testing"
	"time"
)

func TestComputeMetrics(t *testing.T) {
	tests := []struct {
		name       string
		elapsed    time.Duration
		n          int
		bytesPerOp int64
		flopsPerOp int64
		wantOK     bool
		want       metrics
	}{
		{
			// 1000 iterations occupying 1s of device time is 1ms per op.
			name:    "per_op_time",
			elapsed: time.Second,
			n:       1000,
			wantOK:  true,
			want:    metrics{gpuNsPerOp: 1e6},
		},
		{
			// 1e9 bytes moved in exactly 1s is 1.0 GB/s in decimal units.
			// Dividing by 1<<30 instead would report 0.9313.
			name:       "decimal_gb_per_sec",
			elapsed:    time.Second,
			n:          1,
			bytesPerOp: 1e9,
			wantOK:     true,
			want:       metrics{gpuNsPerOp: 1e9, gbPerSec: 1.0},
		},
		{
			name:       "tflops",
			elapsed:    2 * time.Second,
			n:          4,
			flopsPerOp: 5e11,
			wantOK:     true,
			want:       metrics{gpuNsPerOp: 5e8, tflops: 1.0},
		},
		{
			// Both rates from one run, with n > 1 so the iteration count is
			// load-bearing in each numerator.
			//
			//   bytes: 2.5e8 * 4 = 1e9 bytes / 1e9 = 1 GB, over 0.5s = 2 GB/s.
			//          A 1<<30 divisor would report 1.8626 GB/s.
			//   flops: 1.25e11 * 4 = 5e11 / 1e12 = 0.5 TFLOP, over 0.5s = 1 TFLOPS.
			//   time:  5e8 ns / 4 = 1.25e8 gpu-ns/op.
			name:       "bytes_and_flops_together",
			elapsed:    500 * time.Millisecond,
			n:          4,
			bytesPerOp: 2.5e8,
			flopsPerOp: 1.25e11,
			wantOK:     true,
			want:       metrics{gpuNsPerOp: 1.25e8, gbPerSec: 2.0, tflops: 1.0},
		},
		{
			// Zero bytes and zero flops leave their rates at zero rather than
			// publishing a column derived from an unset size.
			name:    "zero_bytes_and_flops_report_no_rates",
			elapsed: 2 * time.Second,
			n:       8,
			wantOK:  true,
			want:    metrics{gpuNsPerOp: 2.5e8},
		},
		{
			name:       "zero_elapsed_yields_no_rate",
			elapsed:    0,
			n:          10,
			bytesPerOp: 1024,
			wantOK:     false,
		},
		{
			name:       "negative_elapsed_yields_no_rate",
			elapsed:    -time.Millisecond,
			n:          10,
			bytesPerOp: 1024,
			wantOK:     false,
		},
		{
			name:    "zero_iterations_yields_no_rate",
			elapsed: time.Second,
			n:       0,
			wantOK:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := computeMetrics(tt.elapsed, tt.n, tt.bytesPerOp, tt.flopsPerOp)
			if ok != tt.wantOK {
				t.Fatalf("computeMetrics ok = %v, want %v", ok, tt.wantOK)
			}
			if !ok {
				return
			}
			close := func(a, b float64) bool { return math.Abs(a-b) <= 1e-9*math.Max(1, math.Abs(b)) }
			if !close(got.gpuNsPerOp, tt.want.gpuNsPerOp) {
				t.Errorf("gpuNsPerOp = %g, want %g", got.gpuNsPerOp, tt.want.gpuNsPerOp)
			}
			if !close(got.gbPerSec, tt.want.gbPerSec) {
				t.Errorf("gbPerSec = %g, want %g", got.gbPerSec, tt.want.gbPerSec)
			}
			if !close(got.tflops, tt.want.tflops) {
				t.Errorf("tflops = %g, want %g", got.tflops, tt.want.tflops)
			}
		})
	}
}

// TestComputeMetricsUnitScale pins the unit of the elapsed argument.
//
// cuda.Event.ElapsedSince returns a time.Duration (nanoseconds). An earlier
// version of Time treated that value as float64 milliseconds, which scaled
// gpu-ns/op up by 1e6 and both rates down by 1e6.
func TestComputeMetricsUnitScale(t *testing.T) {
	// One iteration taking exactly 1ms of device time.
	got, ok := computeMetrics(time.Millisecond, 1, 0, 0)
	if !ok {
		t.Fatal("computeMetrics reported no rate for a 1ms run")
	}
	if want := 1e6; got.gpuNsPerOp != want {
		t.Errorf("gpuNsPerOp = %g, want %g (1ms in nanoseconds)", got.gpuNsPerOp, want)
	}
}
