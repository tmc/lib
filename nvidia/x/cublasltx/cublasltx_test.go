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

func TestMatMulRowMajor(t *testing.T) {
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
	aHost := []float32{1, 2, 3, 4, 5, 6}
	bHost := []float32{7, 8, 9, 10, 11, 12}
	want := hostMatMulRowMajor(aHost, bHost, m, k, n)

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

	if err := engine.MatMulRowMajor(stream, a, m, k, b, n, d); err != nil {
		t.Fatalf("MatMulRowMajor: %v", err)
	}
	if err := stream.Synchronize(nil); err != nil {
		t.Fatal(err)
	}

	got := make([]float32, len(want))
	if err := d.CopyToHost(got); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, want) {
		t.Fatalf("MatMulRowMajor = %v, want %v", got, want)
	}
}

func TestLinearRowMajor(t *testing.T) {
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

	const tokens, inFeatures, outFeatures = 2, 3, 2
	xHost := []float32{1, 2, 3, 4, 5, 6}
	wHost := []float32{7, 8, 9, 10, 11, 12}
	want := hostLinearRowMajor(xHost, wHost, tokens, inFeatures, outFeatures)

	x, err := cudamem.Alloc[float32](dev, len(xHost))
	if err != nil {
		t.Fatal(err)
	}
	defer x.Free()
	w, err := cudamem.Alloc[float32](dev, len(wHost))
	if err != nil {
		t.Fatal(err)
	}
	defer w.Free()
	y, err := cudamem.Alloc[float32](dev, len(want))
	if err != nil {
		t.Fatal(err)
	}
	defer y.Free()

	if err := x.CopyFromHost(xHost); err != nil {
		t.Fatal(err)
	}
	if err := w.CopyFromHost(wHost); err != nil {
		t.Fatal(err)
	}

	if err := engine.LinearRowMajor(stream, x, tokens, inFeatures, w, outFeatures, y); err != nil {
		t.Fatalf("LinearRowMajor: %v", err)
	}
	if err := stream.Synchronize(nil); err != nil {
		t.Fatal(err)
	}

	got := make([]float32, len(want))
	if err := y.CopyToHost(got); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, want) {
		t.Fatalf("LinearRowMajor = %v, want %v", got, want)
	}
}

func TestMatMulF16(t *testing.T) {
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
	aF32 := []float32{1, 4, 2, 5, 3, 6}
	bF32 := []float32{7, 9, 11, 8, 10, 12}
	aHost := make([]uint16, len(aF32))
	bHost := make([]uint16, len(bF32))
	for i, v := range aF32 {
		aHost[i] = f32ToF16Bits(v)
	}
	for i, v := range bF32 {
		bHost[i] = f32ToF16Bits(v)
	}

	a, err := cudamem.Alloc[uint16](dev, len(aHost))
	if err != nil {
		t.Fatal(err)
	}
	defer a.Free()
	b, err := cudamem.Alloc[uint16](dev, len(bHost))
	if err != nil {
		t.Fatal(err)
	}
	defer b.Free()
	d, err := cudamem.Alloc[uint16](dev, m*n)
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

	if err := engine.MatMulF16(stream, a, m, k, b, n, d); err != nil {
		t.Fatalf("MatMulF16: %v", err)
	}
	if err := stream.Synchronize(nil); err != nil {
		t.Fatal(err)
	}

	gotBits := make([]uint16, m*n)
	if err := d.CopyToHost(gotBits); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, m*n)
	for i, b := range gotBits {
		got[i] = f16ToF32Bits(b)
	}
	want := []float32{58, 139, 64, 154}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 0.5 {
			t.Fatalf("MatMulF16[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestLinearRowMajorF16(t *testing.T) {
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

	const tokens, inFeatures, outFeatures = 2, 3, 2
	xF32 := []float32{1, 2, 3, 4, 5, 6}
	wF32 := []float32{7, 8, 9, 10, 11, 12}
	xHost := make([]uint16, len(xF32))
	wHost := make([]uint16, len(wF32))
	for i, v := range xF32 {
		xHost[i] = f32ToF16Bits(v)
	}
	for i, v := range wF32 {
		wHost[i] = f32ToF16Bits(v)
	}

	x, err := cudamem.Alloc[uint16](dev, len(xHost))
	if err != nil {
		t.Fatal(err)
	}
	defer x.Free()
	w, err := cudamem.Alloc[uint16](dev, len(wHost))
	if err != nil {
		t.Fatal(err)
	}
	defer w.Free()
	y, err := cudamem.Alloc[uint16](dev, tokens*outFeatures)
	if err != nil {
		t.Fatal(err)
	}
	defer y.Free()

	if err := x.CopyFromHost(xHost); err != nil {
		t.Fatal(err)
	}
	if err := w.CopyFromHost(wHost); err != nil {
		t.Fatal(err)
	}

	if err := engine.LinearRowMajorF16(stream, x, tokens, inFeatures, w, outFeatures, y); err != nil {
		t.Fatalf("LinearRowMajorF16: %v", err)
	}
	if err := stream.Synchronize(nil); err != nil {
		t.Fatal(err)
	}

	gotBits := make([]uint16, tokens*outFeatures)
	if err := y.CopyToHost(gotBits); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, len(gotBits))
	for i, b := range gotBits {
		got[i] = f16ToF32Bits(b)
	}
	want := []float32{50, 68, 122, 167}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 0.5 {
			t.Fatalf("LinearRowMajorF16[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func hostMatMulRowMajor(a, b []float32, m, k, n int) []float32 {
	d := make([]float32, m*n)
	for r := 0; r < m; r++ {
		for c := 0; c < n; c++ {
			var sum float32
			for i := 0; i < k; i++ {
				sum += a[r*k+i] * b[i*n+c]
			}
			d[r*n+c] = sum
		}
	}
	return d
}

func hostLinearRowMajor(x, w []float32, tokens, inFeatures, outFeatures int) []float32 {
	y := make([]float32, tokens*outFeatures)
	for t := 0; t < tokens; t++ {
		for o := 0; o < outFeatures; o++ {
			var sum float32
			for i := 0; i < inFeatures; i++ {
				sum += x[t*inFeatures+i] * w[o*inFeatures+i]
			}
			y[t*outFeatures+o] = sum
		}
	}
	return y
}

func f32ToF16Bits(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits >> 23) & 0xff) - 127 + 15
	mant := bits & 0x007fffff
	if exp <= 0 {
		return sign
	} else if exp >= 31 {
		return sign | 0x7c00
	}
	return sign | uint16(exp<<10) | uint16(mant>>13)
}

func f16ToF32Bits(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := (h >> 10) & 0x1f
	mant := uint32(h & 0x3ff)
	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		for (mant & 0x400) == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x3ff
		exp32 := uint32(exp - 15 + 127)
		return math.Float32frombits(sign | (exp32 << 23) | (mant << 13))
	} else if exp == 31 {
		return math.Float32frombits(sign | 0x7f800000 | (mant << 13))
	}
	exp32 := uint32(exp - 15 + 127)
	return math.Float32frombits(sign | (exp32 << 23) | (mant << 13))
}

