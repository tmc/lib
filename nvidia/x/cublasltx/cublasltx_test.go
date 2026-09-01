package cublasltx_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cublasltx"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

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
	defer engine.Close()

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
	defer engine.Close()

	// Benchmark large batch prefill matrix sizes: M=64 (batch), K=2048, N=5632 (intermediate)
	m, k, n := 64, 2048, 5632
	aBuf, _ := cudamem.Alloc[float32](dev, m*k)
	defer aBuf.Free()
	bBuf, _ := cudamem.Alloc[float32](dev, k*n)
	defer bBuf.Free()
	dBuf, _ := cudamem.Alloc[float32](dev, m*n)
	defer dBuf.Free()

	// Warmup
	_ = engine.MatMul(stream, aBuf, m, k, bBuf, n, dBuf)
	_ = stream.Synchronize(nil)

	flops := float64(2 * m * k * n)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = engine.MatMul(stream, aBuf, m, k, bBuf, n, dBuf)
	}
	_ = stream.Synchronize(nil)
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
	defer engine.Close()

	m, k, n := 32, 64, 32
	aBuf, _ := cudamem.Alloc[float32](dev, m*k)
	defer aBuf.Free()
	bBuf, _ := cudamem.Alloc[float32](dev, k*n)
	defer bBuf.Free()
	biasBuf, _ := cudamem.Alloc[float32](dev, m)
	defer biasBuf.Free()
	dBuf, _ := cudamem.Alloc[float32](dev, m*n)
	defer dBuf.Free()

	if err := engine.MatMulWithGELUBias(stream, aBuf, m, k, bBuf, n, biasBuf, dBuf); err != nil {
		t.Fatalf("MatMulWithGELUBias failed: %v", err)
	}

	if err := stream.Synchronize(nil); err != nil {
		t.Fatalf("Synchronize failed: %v", err)
	}
}
