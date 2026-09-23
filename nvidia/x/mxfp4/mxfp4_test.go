package mxfp4

import (
	"context"
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func TestQuantizeDequantizeRoundtrip(t *testing.T) {
	groupSize := 32
	numElements := 128
	src := make([]float32, numElements)
	for i := range src {
		src[i] = float32(math.Sin(float64(i+1)*0.1)) * 3.5
	}

	packed, scales, err := Quantize(src, groupSize)
	if err != nil {
		t.Fatalf("Quantize failed: %v", err)
	}

	if len(packed) != numElements/2 {
		t.Errorf("expected packed length %d, got %d", numElements/2, len(packed))
	}
	if len(scales) != numElements/groupSize {
		t.Errorf("expected %d scales, got %d", numElements/groupSize, len(scales))
	}

	dst := make([]float32, numElements)
	if err := Dequantize(packed, scales, groupSize, dst); err != nil {
		t.Fatalf("Dequantize failed: %v", err)
	}

	// Verify quantization preserves approximate sign and scale
	for i := 0; i < numElements; i++ {
		orig := src[i]
		rec := dst[i]
		diff := float32(math.Abs(float64(orig - rec)))
		// Quantization error for 4-bit should be reasonably bounded
		if diff > 1.5 {
			t.Errorf("element %d: large discrepancy orig=%f rec=%f (diff=%f)", i, orig, rec, diff)
		}
	}
}

func TestCUDAMXFP4LinearGEMV(t *testing.T) {
	if err := driver.Init(0); err != nil {
		t.Skipf("skipping: CUDA driver not available: %v", err)
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Skipf("skipping: CUDA device acquisition: %v", err)
	}
	defer dev.Close()

	engine, err := NewEngine(dev)
	if err != nil {
		t.Fatalf("NewEngine failed: %v", err)
	}
	defer engine.Close()

	inDim := 64
	outDim := 32
	groupSize := 32

	wHost := make([]float32, outDim*inDim)
	for i := range wHost {
		wHost[i] = float32(i%7-3) * 0.5
	}
	xHost := make([]float32, inDim)
	for i := range xHost {
		xHost[i] = 1.0
	}

	qm, err := QuantizeMatrix(outDim, inDim, wHost, groupSize)
	if err != nil {
		t.Fatalf("QuantizeMatrix failed: %v", err)
	}

	xDev, err := cudamem.Alloc[float32](dev, inDim)
	if err != nil {
		t.Fatal(err)
	}
	defer xDev.Free()
	if err := xDev.CopyFromHost(xHost); err != nil {
		t.Fatal(err)
	}

	packedDev, err := cudamem.Alloc[uint8](dev, len(qm.Packed))
	if err != nil {
		t.Fatal(err)
	}
	defer packedDev.Free()
	if err := packedDev.CopyFromHost(qm.Packed); err != nil {
		t.Fatal(err)
	}

	scalesDev, err := cudamem.Alloc[float32](dev, len(qm.Scales))
	if err != nil {
		t.Fatal(err)
	}
	defer scalesDev.Free()
	if err := scalesDev.CopyFromHost(qm.Scales); err != nil {
		t.Fatal(err)
	}

	outDev, err := cudamem.Alloc[float32](dev, outDim)
	if err != nil {
		t.Fatal(err)
	}
	defer outDev.Free()

	if err := engine.LinearGEMV(xDev, packedDev, scalesDev, outDev, inDim, outDim, groupSize); err != nil {
		t.Fatalf("LinearGEMV failed: %v", err)
	}
	if err := engine.LinearGEMV(xDev, packedDev.Slice(0, packedDev.Len()-1), scalesDev, outDev, inDim, outDim, groupSize); err == nil {
		t.Fatal("LinearGEMV accepted a short packed slice")
	}

	dequantDev, err := cudamem.Alloc[float32](dev, len(wHost))
	if err != nil {
		t.Fatal(err)
	}
	defer dequantDev.Free()
	if err := engine.Dequantize(packedDev, scalesDev, dequantDev, len(wHost), groupSize); err != nil {
		t.Fatalf("Dequantize failed: %v", err)
	}
	dequantHost := make([]float32, len(wHost))
	if err := dequantDev.CopyToHost(dequantHost); err != nil {
		t.Fatal(err)
	}
	wantDequant := make([]float32, len(wHost))
	if err := Dequantize(qm.Packed, qm.Scales, groupSize, wantDequant); err != nil {
		t.Fatal(err)
	}
	for i := range wantDequant {
		if dequantHost[i] != wantDequant[i] {
			t.Fatalf("dequantized[%d] = %v, want %v", i, dequantHost[i], wantDequant[i])
		}
	}

	outHost := make([]float32, outDim)
	if err := outDev.CopyToHost(outHost); err != nil {
		t.Fatalf("CopyToHost failed: %v", err)
	}

	for row, got := range outHost {
		var want float32
		for col, x := range xHost {
			want += wantDequant[row*inDim+col] * x
		}
		if math.IsNaN(float64(got)) || math.Abs(float64(got-want)) > 1e-5 {
			t.Fatalf("GEMV row %d = %v, want %v", row, got, want)
		}
	}
	if err := engine.Close(); err != nil {
		t.Fatal(err)
	}
	if err := engine.Close(); err != nil {
		t.Fatal(err)
	}
	if err := engine.LinearGEMV(xDev, packedDev, scalesDev, outDev, inDim, outDim, groupSize); err == nil {
		t.Fatal("closed engine accepted a launch")
	}
}

func BenchmarkMXFP4GEMV_Qwen38_Layer(b *testing.B) {
	if err := driver.Init(0); err != nil {
		b.Skipf("skipping: CUDA driver not available: %v", err)
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		b.Skipf("skipping: CUDA device: %v", err)
	}
	defer dev.Close()

	engine, err := NewEngine(dev)
	if err != nil {
		b.Fatalf("NewEngine: %v", err)
	}
	defer engine.Close()

	// Qwen3.8 FFN Intermediate Dimension
	inDim := 5120
	outDim := 17408
	groupSize := 32

	xDev, _ := cudamem.Alloc[float32](dev, inDim)
	defer xDev.Free()

	packedDev, _ := cudamem.Alloc[uint8](dev, outDim*(inDim/2))
	defer packedDev.Free()

	scalesDev, _ := cudamem.Alloc[float32](dev, outDim*(inDim/groupSize))
	defer scalesDev.Free()

	outDev, _ := cudamem.Alloc[float32](dev, outDim)
	defer outDev.Free()

	stream, _ := dev.NewStream()
	defer stream.Destroy()

	b.SetBytes(int64(outDim * (inDim / 2))) // 4-bit weight bytes transferred per step
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		if err := engine.LinearGEMV(xDev, packedDev, scalesDev, outDev, inDim, outDim, groupSize); err != nil {
			b.Fatalf("LinearGEMV: %v", err)
		}
	}
	_ = stream.Synchronize(context.Background())
}
