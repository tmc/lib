//go:build darwin && arm64

package main

import (
	"encoding/binary"
	"math"
	"testing"

	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

func TestMetalBatchRMSMatvecF16(t *testing.T) {
	dev, err := libmetal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	ops, err := newMetalPinnedOps(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer ops.Close()

	arena := metalmem.NewArena(dev)
	defer arena.Free()
	in := metalmem.UploadIn(arena, "input", []float32{1, -2, 3, -4, 2, 1, -1, -2})
	rmsOut := metalmem.AllocIn[float32](arena, "rms-output", in.Len())
	matvecOut := metalmem.AllocIn[float32](arena, "matvec-output", 3)
	weightData := metalmem.UploadIn(arena, "rms-weight", []byte{
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
	})
	for i, value := range []float32{1, 0.5, 2, 1.5} {
		binary.LittleEndian.PutUint32(weightData.HostSlice()[4*i:], math.Float32bits(value))
	}
	weight := boundMetalTensor{name: "norm", typeID: 0, dimensions: []uint64{4}, data: weightData}

	// Three rows of eight exactly representable F16 values.
	matrixValues := []uint16{
		0x3c00, 0, 0x3800, 0, 0xbc00, 0, 0x4000, 0,
		0, 0x3c00, 0, 0x3800, 0, 0xbc00, 0, 0x4000,
		0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00,
	}
	matrixData := metalmem.AllocIn[byte](arena, "matrix", 2*len(matrixValues))
	for i, value := range matrixValues {
		binary.LittleEndian.PutUint16(matrixData.HostSlice()[2*i:], value)
	}
	matrix := boundMetalTensor{name: "matrix", typeID: 1, dimensions: []uint64{8, 3}, data: matrixData}

	wantRMS, err := ops.RMS(append([]float32(nil), in.HostSlice()...), weight, 2, true)
	if err != nil {
		t.Fatal(err)
	}
	wantMatvec, err := ops.MatvecF16(matrix, wantRMS)
	if err != nil {
		t.Fatal(err)
	}

	batch, err := newMetalBatch(ops)
	if err != nil {
		t.Fatal(err)
	}
	if err := batch.EncodeRMS(in, weight, rmsOut, 2, true); err != nil {
		t.Fatal(err)
	}
	if err := batch.EncodeMatvecF16(matrix, rmsOut, matvecOut); err != nil {
		t.Fatal(err)
	}
	if got := batch.Stats(); got != (metalBatchStats{Encoded: 2}) {
		t.Fatalf("stats before wait = %+v, want {Encoded:2}", got)
	}
	if err := batch.Wait(t.Context()); err != nil {
		t.Fatal(err)
	}
	if got := batch.Stats(); got != (metalBatchStats{Encoded: 2, Commits: 1, Waits: 1}) {
		t.Fatalf("stats after wait = %+v, want {Encoded:2 Commits:1 Waits:1}", got)
	}
	closeSlice(t, "batched RMS", rmsOut.HostSlice(), wantRMS)
	closeSlice(t, "batched matvec", matvecOut.HostSlice(), wantMatvec)
}
