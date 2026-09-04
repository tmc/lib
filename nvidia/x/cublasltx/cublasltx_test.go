package cublasltx_test

import (
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/x/cublasltx"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

func TestMatMulValues(t *testing.T) {
	dev := cudatest.Require(t)
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	defer stream.Destroy()

	engine, err := cublasltx.NewEngine(dev)
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}
	defer engine.Close()

	const m, k, n = 2, 3, 2
	a, err := cudamem.Alloc[float32](dev, m*k)
	if err != nil {
		t.Fatalf("alloc A: %v", err)
	}
	defer a.Free()
	b, err := cudamem.Alloc[float32](dev, k*n)
	if err != nil {
		t.Fatalf("alloc B: %v", err)
	}
	defer b.Free()
	d, err := cudamem.Alloc[float32](dev, m*n)
	if err != nil {
		t.Fatalf("alloc D: %v", err)
	}
	defer d.Free()

	// cuBLASLt uses column-major layouts here. These encode
	// A=[[1,2,3],[4,5,6]] and B=[[7,8],[9,10],[11,12]].
	if err := a.CopyFromHost([]float32{1, 4, 2, 5, 3, 6}); err != nil {
		t.Fatalf("copy A: %v", err)
	}
	if err := b.CopyFromHost([]float32{7, 9, 11, 8, 10, 12}); err != nil {
		t.Fatalf("copy B: %v", err)
	}
	if err := engine.MatMul(stream, a, m, k, b, n, d); err != nil {
		t.Fatalf("MatMul: %v", err)
	}
	if err := stream.Synchronize(nil); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}
	got := make([]float32, m*n)
	if err := d.CopyToHost(got); err != nil {
		t.Fatalf("copy D: %v", err)
	}
	want := []float32{58, 139, 64, 154}
	if !slices.Equal(got, want) {
		t.Fatalf("D = %v, want %v", got, want)
	}
}

func TestMatMul(t *testing.T) {
	dev := cudatest.Require(t)
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatalf("NewStream failed: %v", err)
	}
	defer stream.Destroy()

	engine, err := cublasltx.NewEngine(dev)
	if err != nil {
		t.Fatalf("NewEngine failed: %v", err)
	}
	defer func() {
		if err := engine.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	}()

	m, k, n := 128, 64, 128
	aBuf, err := cudamem.Alloc[float32](dev, m*k)
	if err != nil {
		t.Fatalf("alloc a: %v", err)
	}
	defer aBuf.Free()

	bBuf, err := cudamem.Alloc[float32](dev, k*n)
	if err != nil {
		t.Fatalf("alloc b: %v", err)
	}
	defer bBuf.Free()

	dBuf, err := cudamem.Alloc[float32](dev, m*n)
	if err != nil {
		t.Fatalf("alloc d: %v", err)
	}
	defer dBuf.Free()

	if err := engine.MatMul(stream, aBuf, m, k, bBuf, n, dBuf); err != nil {
		t.Fatalf("MatMul failed: %v", err)
	}

	if err := stream.Synchronize(nil); err != nil {
		t.Fatalf("Synchronize failed: %v", err)
	}
	if err := engine.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := engine.MatMul(stream, aBuf, m, k, bBuf, n, dBuf); err == nil {
		t.Fatal("MatMul after Close succeeded")
	}
}

func BenchmarkCuBLASLtGEMM(b *testing.B) {
	dev := cudatest.Require(b)
	stream, err := dev.NewStream()
	if err != nil {
		b.Fatalf("NewStream failed: %v", err)
	}
	defer stream.Destroy()

	engine, err := cublasltx.NewEngine(dev)
	if err != nil {
		b.Fatalf("NewEngine failed: %v", err)
	}
	defer func() {
		if err := engine.Close(); err != nil {
			b.Errorf("Close: %v", err)
		}
	}()

	// Benchmark large batch prefill matrix sizes: M=64 (batch), K=2048, N=5632 (intermediate)
	m, k, n := 64, 2048, 5632
	aBuf, err := cudamem.Alloc[float32](dev, m*k)
	if err != nil {
		b.Fatalf("alloc A: %v", err)
	}
	defer aBuf.Free()
	bBuf, err := cudamem.Alloc[float32](dev, k*n)
	if err != nil {
		b.Fatalf("alloc B: %v", err)
	}
	defer bBuf.Free()
	dBuf, err := cudamem.Alloc[float32](dev, m*n)
	if err != nil {
		b.Fatalf("alloc D: %v", err)
	}
	defer dBuf.Free()

	// Warmup
	if err := engine.MatMul(stream, aBuf, m, k, bBuf, n, dBuf); err != nil {
		b.Fatalf("warmup MatMul: %v", err)
	}
	if err := stream.Synchronize(nil); err != nil {
		b.Fatalf("warmup synchronize: %v", err)
	}

	flops := float64(2 * m * k * n)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := engine.MatMul(stream, aBuf, m, k, bBuf, n, dBuf); err != nil {
			b.Fatalf("MatMul: %v", err)
		}
	}
	if err := stream.Synchronize(nil); err != nil {
		b.Fatalf("synchronize: %v", err)
	}
	elapsed := b.Elapsed().Seconds()

	tflops := (float64(b.N) * flops / 1e12) / elapsed
	b.ReportMetric(tflops, "TFLOPS")
}

func TestMatMulWithGELUBias(t *testing.T) {
	dev := cudatest.Require(t)
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatalf("NewStream failed: %v", err)
	}
	defer stream.Destroy()

	engine, err := cublasltx.NewEngine(dev)
	if err != nil {
		t.Fatalf("NewEngine failed: %v", err)
	}
	defer func() {
		if err := engine.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	}()

	m, k, n := 32, 64, 32
	aBuf, err := cudamem.Alloc[float32](dev, m*k)
	if err != nil {
		t.Fatalf("alloc A: %v", err)
	}
	defer aBuf.Free()
	bBuf, err := cudamem.Alloc[float32](dev, k*n)
	if err != nil {
		t.Fatalf("alloc B: %v", err)
	}
	defer bBuf.Free()
	biasBuf, err := cudamem.Alloc[float32](dev, n)
	if err != nil {
		t.Fatalf("alloc bias: %v", err)
	}
	defer biasBuf.Free()
	dBuf, err := cudamem.Alloc[float32](dev, m*n)
	if err != nil {
		t.Fatalf("alloc D: %v", err)
	}
	defer dBuf.Free()

	if err := engine.MatMulWithGELUBias(stream, aBuf, m, k, bBuf, n, biasBuf, dBuf); err != nil {
		t.Fatalf("MatMulWithGELUBias failed: %v", err)
	}

	if err := stream.Synchronize(nil); err != nil {
		t.Fatalf("Synchronize failed: %v", err)
	}
}
