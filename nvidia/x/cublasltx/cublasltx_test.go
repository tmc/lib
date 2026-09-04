package cublasltx_test

import (
	"math"
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

func TestMatMulShapeDifferential(t *testing.T) {
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

	shapes := []struct {
		name    string
		m, k, n int
	}{
		{name: "vector", m: 1, k: 7, n: 5},
		{name: "rectangular", m: 3, k: 5, n: 2},
		{name: "tile boundary", m: 17, k: 9, n: 33},
	}
	for _, shape := range shapes {
		t.Run(shape.name, func(t *testing.T) {
			aHost := make([]float32, shape.m*shape.k)
			bHost := make([]float32, shape.k*shape.n)
			for col := range shape.k {
				for row := range shape.m {
					aHost[col*shape.m+row] = float32((row+2*col)%7-3) / 4
				}
			}
			for col := range shape.n {
				for row := range shape.k {
					bHost[col*shape.k+row] = float32((2*row+col)%9-4) / 5
				}
			}
			want := hostMatMul(aHost, bHost, shape.m, shape.k, shape.n)
			a, err := cudamem.Alloc[float32](dev, len(aHost))
			if err != nil {
				t.Fatal(err)
			}
			defer a.Free()
			b, err := cudamem.Alloc[float32](dev, len(bHost))
			if err != nil {
				t.Fatal(err)
			}
			defer b.Free()
			d, err := cudamem.Alloc[float32](dev, len(want))
			if err != nil {
				t.Fatal(err)
			}
			defer d.Free()
			if err := a.CopyFromHost(aHost); err != nil {
				t.Fatal(err)
			}
			if err := b.CopyFromHost(bHost); err != nil {
				t.Fatal(err)
			}
			if err := engine.MatMul(stream, a, shape.m, shape.k, b, shape.n, d); err != nil {
				t.Fatal(err)
			}
			if err := stream.Synchronize(nil); err != nil {
				t.Fatal(err)
			}
			got := make([]float32, len(want))
			if err := d.CopyToHost(got); err != nil {
				t.Fatal(err)
			}
			for i := range want {
				if diff := math.Abs(float64(got[i] - want[i])); diff > 1e-5 {
					t.Fatalf("D[%d] = %g, want %g (difference %g)", i, got[i], want[i], diff)
				}
			}
		})
	}
}

func hostMatMul(a, b []float32, m, k, n int) []float32 {
	d := make([]float32, m*n)
	for col := range n {
		for row := range m {
			for inner := range k {
				d[col*m+row] += a[inner*m+row] * b[col*k+inner]
			}
		}
	}
	return d
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

func TestMatMulSubmissionFailureDoesNotPoisonEngine(t *testing.T) {
	dev := cudatest.Require(t)
	engine, err := cublasltx.NewEngine(dev)
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}
	defer engine.Close()
	a, err := cudamem.Alloc[float32](dev, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Free()
	b, err := cudamem.Alloc[float32](dev, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Free()
	d, err := cudamem.Alloc[float32](dev, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Free()
	if err := a.CopyFromHost([]float32{3}); err != nil {
		t.Fatal(err)
	}
	if err := b.CopyFromHost([]float32{4}); err != nil {
		t.Fatal(err)
	}

	closed, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	if err := closed.Destroy(); err != nil {
		t.Fatal(err)
	}
	for range 32 {
		if err := engine.MatMul(closed, a, 1, 1, b, 1, d); err == nil {
			t.Fatal("MatMul on closed stream succeeded")
		}
	}
	live, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer live.Destroy()
	if err := engine.MatMul(live, a, 1, 1, b, 1, d); err != nil {
		t.Fatalf("MatMul after rejected submissions: %v", err)
	}
	if err := live.Synchronize(nil); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, 1)
	if err := d.CopyToHost(got); err != nil {
		t.Fatal(err)
	}
	if got[0] != 12 {
		t.Fatalf("D = %v, want [12]", got)
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

	m, k, n := 3, 5, 4
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
	biasBuf, err := cudamem.Alloc[float32](dev, m)
	if err != nil {
		t.Fatalf("alloc bias: %v", err)
	}
	defer biasBuf.Free()
	dBuf, err := cudamem.Alloc[float32](dev, m*n)
	if err != nil {
		t.Fatalf("alloc D: %v", err)
	}
	defer dBuf.Free()

	a := []float32{-.5, .25, .75, 1, -.25, .5, -.75, 1.25, .125, .375, -.625, .875, -.125, .625, -1}
	b := []float32{.5, -.5, .25, .75, -1, -.25, .75, .5, -1, .25, 1, .125, -.375, .625, -.875, .25, -.75, .375, .875, -.125}
	bias := []float32{-.5, .25, 1}
	if err := aBuf.CopyFromHost(a); err != nil {
		t.Fatal(err)
	}
	if err := bBuf.CopyFromHost(b); err != nil {
		t.Fatal(err)
	}
	if err := biasBuf.CopyFromHost(bias); err != nil {
		t.Fatal(err)
	}

	if err := engine.MatMulWithGELUBias(stream, aBuf, m, k, bBuf, n, biasBuf, dBuf); err != nil {
		t.Fatalf("MatMulWithGELUBias failed: %v", err)
	}

	if err := stream.Synchronize(nil); err != nil {
		t.Fatalf("Synchronize failed: %v", err)
	}
	got := make([]float32, m*n)
	if err := dBuf.CopyToHost(got); err != nil {
		t.Fatal(err)
	}
	want := hostMatMul(a, b, m, k, n)
	for col := range n {
		for row := range m {
			i := col*m + row
			want[i] = gelu(want[i] + bias[row])
			if diff := math.Abs(float64(got[i] - want[i])); diff > 2e-3 {
				t.Fatalf("D[%d] = %g, want %g (difference %g)", i, got[i], want[i], diff)
			}
		}
	}
}

func gelu(x float32) float32 {
	v := float64(x)
	return float32(.5 * v * (1 + math.Erf(v/math.Sqrt2)))
}
