package mxfp4

import (
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

	xDev, _ := cudamem.Alloc[float32](dev, inDim)
	defer xDev.Free()
	_ = xDev.CopyFromHost(xHost)

	packedDev, _ := cudamem.Alloc[uint8](dev, len(qm.Packed))
	defer packedDev.Free()
	_ = packedDev.CopyFromHost(qm.Packed)

	scalesDev, _ := cudamem.Alloc[float32](dev, len(qm.Scales))
	defer scalesDev.Free()
	_ = scalesDev.CopyFromHost(qm.Scales)

	outDev, _ := cudamem.Alloc[float32](dev, outDim)
	defer outDev.Free()

	if err := engine.LinearGEMV(xDev, packedDev, scalesDev, outDev, inDim, outDim, groupSize); err != nil {
		t.Fatalf("LinearGEMV failed: %v", err)
	}

	outHost := make([]float32, outDim)
	if err := outDev.CopyToHost(outHost); err != nil {
		t.Fatalf("CopyToHost failed: %v", err)
	}

	t.Logf("Computed MXFP4 GEMV output sample: %v", outHost[:4])
}
