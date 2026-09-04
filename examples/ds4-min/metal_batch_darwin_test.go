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
	hcProjected, err := hcOut.Slice(0, 4096)
	if err != nil {
		t.Fatal(err)
	}
	attnNorm, err := hcOut.Slice(4096, 2*4096)
	if err != nil {
		t.Fatal(err)
	}
	qLora := metalmem.AllocIn[float32](arena, "query-lora", 64)
	qHeads := metalmem.AllocIn[float32](arena, "query-heads", 128)
	kvRaw := metalmem.AllocIn[float32](arena, "key-value-raw", 32)
	kvNorm := metalmem.AllocIn[float32](arena, "key-value-norm", 32)

	hcFn := testBatchTensor(arena, "hc-fn", 1, []uint64{4 * 4096, 24}, 2*24*4*4096)
	hcScale := testBatchF32Tensor(arena, "hc-scale", []float32{1, 1, 1})
	hcBase := testBatchF32Tensor(arena, "hc-base", make([]float32, 24))
	normValues := make([]float32, 4096)
	for i := range normValues {
		normValues[i] = 0.75 + float32(i%7)/16
	}
	norm := testBatchF32Tensor(arena, "attention-norm", normValues)
	qa := testBatchQ80Tensor(arena, "query-a", 4096, 64)
	qaNorm := testBatchF32Tensor(arena, "query-a-norm", batchWeights(64))
	qb := testBatchQ80Tensor(arena, "query-b", 64, 128)
	kv := testBatchQ80Tensor(arena, "key-value", 4096, 32)
	kvWeight := testBatchF32Tensor(arena, "key-value-norm", batchWeights(32))

	wantHC, wantNorm, err := ops.HCPre(stateValues, hcFn, hcScale, hcBase, norm)
	if err != nil {
		t.Fatal(err)
	}
	wantQLora, err := ops.MatvecQ80(qa, wantNorm)
	if err != nil {
		t.Fatal(err)
	}
	wantQNorm, err := ops.RMS(wantQLora, qaNorm, 1, true)
	if err != nil {
		t.Fatal(err)
	}
	wantQHeadsRaw, err := ops.MatvecQ80(qb, wantQNorm)
	if err != nil {
		t.Fatal(err)
	}
	wantQHeads, err := ops.RMS(wantQHeadsRaw, qaNorm, 64, false)
	if err != nil {
		t.Fatal(err)
	}
	wantKVRaw, err := ops.MatvecQ80(kv, wantNorm)
	if err != nil {
		t.Fatal(err)
	}
	wantKVNorm, err := ops.RMS(wantKVRaw, kvWeight, 1, true)
	if err != nil {
		t.Fatal(err)
	}
	for name, values := range map[string][]float32{
		"HC projection":   wantHC,
		"attention norm":  wantNorm,
		"query LoRA":      wantQLora,
		"query norm":      wantQNorm,
		"query heads raw": wantQHeadsRaw,
		"query heads":     wantQHeads,
		"key-value raw":   wantKVRaw,
		"key-value norm":  wantKVNorm,
	} {
		assertNonzeroFinite(t, name, values)
	}

	batch, err := newMetalBatch(ops)
	if err != nil {
		t.Fatal(err)
	}
	if err := batch.EncodeHCPre(state, hcFn, hcScale, hcBase, norm, hcOut); err != nil {
		t.Fatal(err)
	}
	if err := batch.EncodeMatvecQ80(qa, attnNorm, qLora); err != nil {
		t.Fatal(err)
	}
	if err := batch.EncodeRMS(qLora, qaNorm, qLora, 1, true); err != nil {
		t.Fatal(err)
	}
	if err := batch.EncodeMatvecQ80(qb, qLora, qHeads); err != nil {
		t.Fatal(err)
	}
	if err := batch.EncodeRMS(qHeads, qaNorm, qHeads, 64, false); err != nil {
		t.Fatal(err)
	}
	if err := batch.EncodeMatvecQ80(kv, attnNorm, kvRaw); err != nil {
		t.Fatal(err)
	}
	if err := batch.EncodeRMS(kvRaw, kvWeight, kvNorm, 1, true); err != nil {
		t.Fatal(err)
	}
	if got := batch.Stats(); got != (metalBatchStats{Encoded: 7}) {
		t.Fatalf("stats before wait = %+v, want {Encoded:7}", got)
	}
	if err := batch.Wait(t.Context()); err != nil {
		t.Fatal(err)
	}
	if got := batch.Stats(); got != (metalBatchStats{Encoded: 7, Commits: 1, Waits: 1}) {
		t.Fatalf("stats after wait = %+v, want {Encoded:7 Commits:1 Waits:1}", got)
	}
	closeSlice(t, "batched HC projection", hcProjected.HostSlice(), wantHC)
	closeSlice(t, "batched HC norm", attnNorm.HostSlice(), wantNorm)
	closeSlice(t, "batched query norm", qLora.HostSlice(), wantQNorm)
	closeSlice(t, "batched query heads", qHeads.HostSlice(), wantQHeads)
	closeSlice(t, "batched raw key-value", kvRaw.HostSlice(), wantKVRaw)
	closeSlice(t, "batched key-value norm", kvNorm.HostSlice(), wantKVNorm)
}

func TestMetalBatchFFNElementwiseResidual(t *testing.T) {
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

	const width = 4096
	arena := metalmem.NewArena(dev)
	defer arena.Free()
	gateValues := make([]float32, width)
	upValues := make([]float32, width)
	stateValues := make([]float32, 4*width)
	for i := range width {
		gateValues[i] = float32(i%31-15) / 2
		upValues[i] = float32(i%23-11) / 2
	}
	for i := range stateValues {
		stateValues[i] = float32(i%17-8) / 16
	}
	gate := metalmem.UploadIn(arena, "gate", gateValues)
	up := metalmem.UploadIn(arena, "up", upValues)
	clampedGate := metalmem.AllocIn[float32](arena, "clamped-gate", width)
	clampedUp := metalmem.AllocIn[float32](arena, "clamped-up", width)
	mids := metalmem.AllocIn[float32](arena, "expert-mids", 2*width)
	weightedMid, err := mids.Slice(0, width)
	if err != nil {
		t.Fatal(err)
	}
	plainMid, err := mids.Slice(width, 2*width)
	if err != nil {
		t.Fatal(err)
	}
	block := metalmem.AllocIn[float32](arena, "block", width)
	state := metalmem.UploadIn(arena, "state", stateValues)
	out := metalmem.AllocIn[float32](arena, "residual", len(stateValues))
	hcFn := testBatchTensor(arena, "hc-fn", 1, []uint64{4 * width, 24}, 2*24*4*width)
	hcScale := testBatchF32Tensor(arena, "hc-scale", []float32{1, 1, 1})
	hcBase := testBatchF32Tensor(arena, "hc-base", make([]float32, 24))

	wantGate, err := ops.ClampGate10(gateValues)
	if err != nil {
		t.Fatal(err)
	}
	wantUp, err := ops.Clamp10(upValues)
	if err != nil {
		t.Fatal(err)
	}
	wantWeighted, err := ops.SwiGLUWeighted(wantGate, wantUp, 0.375)
	if err != nil {
		t.Fatal(err)
	}
	wantPlain, err := ops.SwiGLU(wantGate, wantUp)
	if err != nil {
		t.Fatal(err)
	}
	wantBlock, err := ops.SumVectors(append(append([]float32(nil), wantWeighted...), wantPlain...), 2)
	if err != nil {
		t.Fatal(err)
	}
	wantOut, err := ops.HCPost(stateValues, wantBlock, hcFn, hcScale, hcBase)
	if err != nil {
		t.Fatal(err)
	}
	for name, values := range map[string][]float32{
		"clamped gate": wantGate,
		"clamped up":   wantUp,
		"weighted mid": wantWeighted,
		"plain mid":    wantPlain,
		"block":        wantBlock,
		"residual":     wantOut,
	} {
		assertNonzeroFinite(t, name, values)
	}

	batch, err := newMetalBatch(ops)
	if err != nil {
		t.Fatal(err)
	}
	if err := batch.EncodeClampGate10(gate, clampedGate); err != nil {
		t.Fatal(err)
	}
	if err := batch.EncodeClamp10(up, clampedUp); err != nil {
		t.Fatal(err)
	}
	if err := batch.EncodeSwiGLUWeighted(clampedGate, clampedUp, weightedMid, 0.375); err != nil {
		t.Fatal(err)
	}
	if err := batch.EncodeSwiGLU(clampedGate, clampedUp, plainMid); err != nil {
		t.Fatal(err)
	}
	if err := batch.EncodeSumVectors(mids, block, 2); err != nil {
		t.Fatal(err)
	}
	if err := batch.EncodeHCPost(state, block, hcFn, hcScale, hcBase, out); err != nil {
		t.Fatal(err)
	}
	if got := batch.Stats(); got != (metalBatchStats{Encoded: 6}) {
		t.Fatalf("stats before wait = %+v, want {Encoded:6}", got)
	}
	if err := batch.Wait(t.Context()); err != nil {
		t.Fatal(err)
	}
	if got := batch.Stats(); got != (metalBatchStats{Encoded: 6, Commits: 1, Waits: 1}) {
		t.Fatalf("stats after wait = %+v, want {Encoded:6 Commits:1 Waits:1}", got)
	}
	closeSlice(t, "batched clamped gate", clampedGate.HostSlice(), wantGate)
	closeSlice(t, "batched clamped up", clampedUp.HostSlice(), wantUp)
	closeSlice(t, "batched weighted mid", weightedMid.HostSlice(), wantWeighted)
	closeSlice(t, "batched plain mid", plainMid.HostSlice(), wantPlain)
	closeSlice(t, "batched block", block.HostSlice(), wantBlock)
	closeSlice(t, "batched residual", out.HostSlice(), wantOut)
}

func batchWeights(n int) []float32 {
	weights := make([]float32, n)
	for i := range weights {
		weights[i] = 0.75 + float32(i%7)/16
	}
	return weights
}

func assertNonzeroFinite(t *testing.T, name string, values []float32) {
	t.Helper()
	nonzero := false
	for _, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("%s contains non-finite value %g", name, value)
		}
		nonzero = nonzero || value != 0
	}
	if !nonzero {
		t.Fatalf("%s is all zero", name)
	}
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
