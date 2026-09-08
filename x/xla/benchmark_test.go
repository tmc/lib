package xla_test

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"runtime/pprof"
	"testing"

	"github.com/tmc/lib/x/xla"
)

// BenchmarkNativePJRT measures synchronous public API operations using a
// caller-supplied CPU plugin. Resource creation benchmarks include destruction;
// download reuses its host destination. Setup and validation are outside timing.
func BenchmarkNativePJRT(b *testing.B) {
	path := os.Getenv("XLA_PJRT_CPU_PLUGIN")
	if path == "" {
		b.Skip("set XLA_PJRT_CPU_PLUGIN to a CPU PJRT library")
	}
	for _, size := range []int{64, 512} {
		b.Run(fmt.Sprint(size), func(b *testing.B) {
			benchmarkPJRT(b, path, size)
		})
	}
}

func benchmarkPJRT(b *testing.B, path string, size int) {
	check := func(err error) {
		b.Helper()
		if err != nil {
			b.Fatal(err)
		}
	}
	engine, err := xla.NewEngine(path)
	check(err)
	defer func() { check(engine.Close()) }()
	devices, err := engine.Devices()
	check(err)
	if len(devices) == 0 {
		b.Fatal("plugin has no addressable devices")
	}
	program := []byte(fmt.Sprintf(`module {
  func.func @main(%%a: tensor<%[1]dx%[1]dxf32>, %%b: tensor<%[1]dx%[1]dxf32>) -> tensor<%[1]dx%[1]dxf32> {
    %%0 = stablehlo.dot_general %%a, %%b, contracting_dims = [1] x [0] : (tensor<%[1]dx%[1]dxf32>, tensor<%[1]dx%[1]dxf32>) -> tensor<%[1]dx%[1]dxf32>
    return %%0 : tensor<%[1]dx%[1]dxf32>
  }
}`, size))
	left := make([]byte, size*size*4)
	right := make([]byte, len(left))
	for i := 0; i < size*size; i++ {
		binary.LittleEndian.PutUint32(left[4*i:], math.Float32bits(float32(i%31-15)))
	}
	for i := 0; i < size; i++ {
		binary.LittleEndian.PutUint32(right[4*(i*size+i):], math.Float32bits(1))
	}
	dims := []int64{int64(size), int64(size)}
	x, err := engine.NewBufferFromHost(left, xla.F32, dims, devices[0])
	check(err)
	defer func() { check(x.Close()) }()
	y, err := engine.NewBufferFromHost(right, xla.F32, dims, devices[0])
	check(err)
	defer func() { check(y.Close()) }()
	executable, err := engine.Compile(program, "mlir", nil)
	check(err)
	defer func() { check(executable.Close()) }()
	args := []*xla.Buffer{x, y}
	outputs, err := executable.Execute(context.Background(), args)
	check(err)
	if len(outputs) != 1 {
		b.Fatalf("output count %d, want 1", len(outputs))
	}
	output := outputs[0]
	defer func() { check(output.Close()) }()
	dst := make([]byte, len(left))
	validate := func(b *testing.B) {
		b.Helper()
		for i := 0; i < len(left); i += 4 {
			got := math.Float32frombits(binary.LittleEndian.Uint32(dst[i:]))
			want := math.Float32frombits(binary.LittleEndian.Uint32(left[i:]))
			if got != want {
				b.Fatalf("element %d: got %g, want %g", i/4, got, want)
			}
		}
	}
	verify := func(b *testing.B, buffer *xla.Buffer) {
		b.Helper()
		if err := buffer.ToHost(dst); err != nil {
			b.Fatal(err)
		}
		validate(b)
	}
	verify(b, output)
	for _, operation := range []string{"compile_close", "upload_close", "download", "execute_close"} {
		b.Run(operation, func(b *testing.B) {
			check := func(err error) {
				b.Helper()
				if err != nil {
					b.Fatal(err)
				}
			}
			labels := pprof.Labels("pjrt.operation", operation, "pjrt.shape", fmt.Sprintf("%dx%d", size, size), "pjrt.dtype", "f32")
			if operation == "download" {
				for i := range dst {
					dst[i] = 0xff
				}
			}
			b.ReportAllocs()
			if operation == "download" || operation == "upload_close" {
				b.SetBytes(int64(len(left)))
			}
			pprof.Do(context.Background(), labels, func(ctx context.Context) {
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					switch operation {
					case "compile_close":
						v, err := engine.Compile(program, "mlir", nil)
						check(err)
						if i == b.N-1 {
							b.StopTimer()
							values, err := v.Execute(ctx, args)
							check(err)
							if len(values) != 1 {
								b.Fatal("unexpected output count")
							}
							verify(b, values[0])
							check(values[0].Close())
							b.StartTimer()
						}
						check(v.Close())
					case "upload_close":
						v, err := engine.NewBufferFromHost(left, xla.F32, dims, devices[0])
						check(err)
						if i == b.N-1 {
							b.StopTimer()
							verify(b, v)
							b.StartTimer()
						}
						check(v.Close())
					case "download":
						check(output.ToHost(dst))
					case "execute_close":
						values, err := executable.Execute(ctx, args)
						check(err)
						if len(values) != 1 {
							b.Fatalf("output count %d, want 1", len(values))
						}
						if i == b.N-1 {
							b.StopTimer()
							verify(b, values[0])
							b.StartTimer()
						}
						check(values[0].Close())
					}
				}
				b.StopTimer()
			})
			if operation == "download" {
				validate(b)
			} else {
				verify(b, output)
			}
		})
	}
}
