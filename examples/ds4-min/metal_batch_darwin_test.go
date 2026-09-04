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

func TestMetalBatchHCPreQKV(t *testing.T) {
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
	stateValues := make([]float32, 4*4096)
	for i := range stateValues {
		stateValues[i] = float32(i%29-14) / 16
	}
	state := metalmem.UploadIn(arena, "state", stateValues)
	hcOut := metalmem.AllocIn[float32](arena, "hc-output", 2*4096)
	attnNorm, err := hcOut.Slice(4096, 2*4096)
	if err != nil {
		t.Fatal(err)
	}
	qOut := metalmem.AllocIn[float32](arena, "query-output", 64)
	kvOut := metalmem.AllocIn[float32](arena, "kv-output", 32)

	hcFn := testBatchTensor(arena, "hc-fn", 1, []uint64{4 * 4096, 24}, 2*24*4*4096)
	hcScale := testBatchF32Tensor(arena, "hc-scale", []float32{1, 1, 1})
	hcBase := testBatchF32Tensor(arena, "hc-base", make([]float32, 24))
	normValues := make([]float32, 4096)
	for i := range normValues {
		normValues[i] = 0.75 + float32(i%7)/16
	}
	norm := testBatchF32Tensor(arena, "attention-norm", normValues)
	q := testBatchQ80Tensor(arena, "query", 4096, 64)
	kv := testBatchQ80Tensor(arena, "key-value", 4096, 32)

	_, wantNorm, err := ops.HCPre(stateValues, hcFn, hcScale, hcBase, norm)
	if err != nil {
		t.Fatal(err)
	}
	wantQ, err := ops.MatvecQ80(q, wantNorm)
	if err != nil {
		t.Fatal(err)
	}
	wantKV, err := ops.MatvecQ80(kv, wantNorm)
	if err != nil {
		t.Fatal(err)
	}

	batch, err := newMetalBatch(ops)
	if err != nil {
		t.Fatal(err)
	}
	if err := batch.EncodeHCPre(state, hcFn, hcScale, hcBase, norm, hcOut); err != nil {
		t.Fatal(err)
	}
	if err := batch.EncodeMatvecQ80(q, attnNorm, qOut); err != nil {
		t.Fatal(err)
	}
	if err := batch.EncodeMatvecQ80(kv, attnNorm, kvOut); err != nil {
		t.Fatal(err)
	}
	if got := batch.Stats(); got != (metalBatchStats{Encoded: 3}) {
		t.Fatalf("stats before wait = %+v, want {Encoded:3}", got)
	}
	if err := batch.Wait(t.Context()); err != nil {
		t.Fatal(err)
	}
	if got := batch.Stats(); got != (metalBatchStats{Encoded: 3, Commits: 1, Waits: 1}) {
		t.Fatalf("stats after wait = %+v, want {Encoded:3 Commits:1 Waits:1}", got)
	}
	closeSlice(t, "batched HC norm", attnNorm.HostSlice(), wantNorm)
	closeSlice(t, "batched query", qOut.HostSlice(), wantQ)
	closeSlice(t, "batched key-value", kvOut.HostSlice(), wantKV)
}

func testBatchTensor(arena *metalmem.Arena, name string, typeID uint32, dimensions []uint64, bytes int) boundMetalTensor {
	return boundMetalTensor{name: name, typeID: typeID, dimensions: dimensions, data: metalmem.AllocIn[byte](arena, name, bytes)}
}

func testBatchF32Tensor(arena *metalmem.Arena, name string, values []float32) boundMetalTensor {
	data := metalmem.AllocIn[byte](arena, name, 4*len(values))
	for i, value := range values {
		binary.LittleEndian.PutUint32(data.HostSlice()[4*i:], math.Float32bits(value))
	}
	return boundMetalTensor{name: name, typeID: 0, dimensions: []uint64{uint64(len(values))}, data: data}
}

func testBatchQ80Tensor(arena *metalmem.Arena, name string, cols, rows int) boundMetalTensor {
	blocks := cols / 32
	data := metalmem.AllocIn[byte](arena, name, rows*blocks*34)
	for row := range rows {
		for block := range blocks {
			p := data.HostSlice()[(row*blocks+block)*34:]
			binary.LittleEndian.PutUint16(p, 0x3000) // 0.125 in F16.
			for i := range 32 {
				p[2+i] = byte(int8((row+block+i)%9 - 4))
			}
		}
	}
	return boundMetalTensor{name: name, typeID: 8, dimensions: []uint64{uint64(cols), uint64(rows)}, data: data}
}
