//go:build linux

package main

// The math kernels read every weight through a device pointer and never touch
// the GGUF mapping, so they can be checked against the CPU references with
// synthetic tensors on any CUDA device. These tests do that: they require a
// device but not the pinned checkpoint, which keeps kernel coverage separate
// from checkpoint identity. Tests that must see real weights still go through
// newCUDAFoundation, which keeps validatePinnedProfile on every path that
// opens a model.

import (
	"context"
	"encoding/binary"
	"math"
	"math/rand"
	"slices"
	"strconv"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// cudaSynthFoundation binds CUDA device 0 with no model mapping. The result
// owns only the device, so Close releases the context and nothing else.
func cudaSynthFoundation(t *testing.T) *cudaFoundation {
	t.Helper()
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	name, err := dev.Name()
	if err != nil {
		dev.Close()
		t.Fatal(err)
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		dev.Close()
		t.Fatal(err)
	}
	r := &cudaFoundation{device: dev, name: name, major: major, minor: minor}
	t.Cleanup(func() {
		if err := r.Close(); err != nil {
			t.Error(err)
		}
	})
	t.Logf("synthetic foundation on %q sm_%d%d", name, major, minor)
	return r
}

// cudaSynthTensor uploads host bytes and describes them the way the GGUF
// binding does, so the encode methods cannot tell the difference.
func cudaSynthTensor(t *testing.T, r *cudaFoundation, name string, typeID uint32, dimensions []uint64, host []byte) cudaModelTensor {
	t.Helper()
	d, err := cudamem.Alloc[byte](r.device, len(host))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = d.Free() })
	if err := d.CopyFromHost(host); err != nil {
		t.Fatal(err)
	}
	return cudaModelTensor{
		name:       name,
		typeID:     typeID,
		dimensions: dimensions,
		device:     d.Ptr(),
		bytes:      uint64(len(host)),
	}
}

// cudaSynthHalf builds a binary16 value with the given unbiased exponent and a
// random mantissa. Generating the encoding directly keeps cudaTestFloat16, the
// existing widening reference, as the only conversion in the test.
func cudaSynthHalf(rnd *rand.Rand, exp int) uint16 {
	return uint16(rnd.Intn(2))<<15 | uint16(exp+15)&0x1f<<10 | uint16(rnd.Intn(1024))
}

// cudaSynthF16 returns a row-major F16 tensor and its widened values.
func cudaSynthF16(rnd *rand.Rand, rows, cols, exp int) ([]byte, []float32) {
	p := make([]byte, 2*rows*cols)
	v := make([]float32, rows*cols)
	for i := range v {
		h := cudaSynthHalf(rnd, exp)
		binary.LittleEndian.PutUint16(p[2*i:], h)
		v[i] = cudaTestFloat16(h)
	}
	return p, v
}

// cudaSynthQ80 returns a row-major Q8_0 tensor: one binary16 scale and 32
// signed values per block.
func cudaSynthQ80(rnd *rand.Rand, rows, cols, exp int) []byte {
	blocks := rows * cols / 32
	p := make([]byte, blocks*34)
	for block := range blocks {
		binary.LittleEndian.PutUint16(p[block*34:], cudaSynthHalf(rnd, exp))
		for i := range 32 {
			p[block*34+2+i] = byte(rnd.Intn(256))
		}
	}
	return p
}

func cudaSynthF32(rnd *rand.Rand, n int, scale float32) ([]byte, []float32) {
	p := make([]byte, 4*n)
	v := make([]float32, n)
	for i := range v {
		v[i] = (rnd.Float32()*2 - 1) * scale
		binary.LittleEndian.PutUint32(p[4*i:], math.Float32bits(v[i]))
	}
	return p, v
}

func cudaSynthI32(rnd *rand.Rand, n, limit int) ([]byte, []int32) {
	p := make([]byte, 4*n)
	v := make([]int32, n)
	for i := range v {
		v[i] = int32(rnd.Intn(limit))
		binary.LittleEndian.PutUint32(p[4*i:], uint32(v[i]))
	}
	return p, v
}

func cudaSynthVector(rnd *rand.Rand, n int, scale float32) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = (rnd.Float32()*2 - 1) * scale
	}
	return v
}

func cudaSynthUpload(t *testing.T, r *cudaFoundation, v []float32) cudamem.Slice[float32] {
	t.Helper()
	d := cudaTestAlloc(t, r, len(v))
	if err := d.CopyFromHost(v); err != nil {
		t.Fatal(err)
	}
	return d
}

func cudaSynthSync(t *testing.T, r *cudaFoundation) {
	t.Helper()
	if err := r.device.DefaultStream().Synchronize(context.Background()); err != nil {
		t.Fatal(err)
	}
}

// TestCUDASynthKernelsCompile loads every math kernel by name. It separates a
// compilation or symbol failure from a numerical one in the tests below.
func TestCUDASynthKernelsCompile(t *testing.T) {
	r := cudaSynthFoundation(t)
	err := r.device.WithContext(func() error { return r.loadMathKernels() })
	if err != nil {
		t.Fatal(err)
	}
	for _, kernel := range r.cudaKernels() {
		if *kernel.dst == 0 {
			t.Errorf("kernel %q did not load", kernel.name)
		}
	}
	t.Logf("loaded %d kernels", len(r.cudaKernels()))
}

func TestCUDASynthF16Matvec(t *testing.T) {
	r := cudaSynthFoundation(t)
	rnd := rand.New(rand.NewSource(1))
	const rows, cols = 16, 512
	p, _ := cudaSynthF16(rnd, rows, cols, -2)
	w := cudaSynthTensor(t, r, "synth.f16", 1, []uint64{cols, rows}, p)
	x := cudaSynthVector(rnd, cols, 1)
	dx := cudaSynthUpload(t, r, x)
	dout := cudaTestAlloc(t, r, rows)
	if err := r.encodeF16Matvec(w, dx, dout); err != nil {
		t.Fatal(err)
	}
	cudaSynthSync(t, r)
	cudaTestClose(t, "synthetic F16 matvec", cudaTestRead(t, dout), cudaTestF16MatvecBytes(p, x, rows))
}

func TestCUDASynthQ80Matvec(t *testing.T) {
	r := cudaSynthFoundation(t)
	rnd := rand.New(rand.NewSource(2))
	const rows, cols = 16, 512
	p := cudaSynthQ80(rnd, rows, cols, -6)
	w := cudaSynthTensor(t, r, "synth.q8_0", 8, []uint64{cols, rows}, p)
	x := cudaSynthVector(rnd, cols, 1)
	dx := cudaSynthUpload(t, r, x)
	dout := cudaTestAlloc(t, r, rows)
	if err := r.encodeQ80Matvec(w, dx, dout); err != nil {
		t.Fatal(err)
	}
	cudaSynthSync(t, r)
	want := make([]float32, rows)
	rowBytes := cols / 32 * 34
	for row := range want {
		want[row] = cudaTestQ80Row(p[row*rowBytes:], x)
	}
	cudaTestClose(t, "synthetic Q8_0 matvec", cudaTestRead(t, dout), want)
}

func TestCUDASynthRMS(t *testing.T) {
	r := cudaSynthFoundation(t)
	rnd := rand.New(rand.NewSource(3))
	const width = 512
	x := cudaSynthVector(rnd, 4*width, 1)
	dx := cudaSynthUpload(t, r, x)
	dout := cudaTestAlloc(t, r, len(x))

	// A null weight pointer is the unweighted path.
	if err := r.encodeRMS(dx, cudaModelTensor{}, dout, 4, false); err != nil {
		t.Fatal(err)
	}
	cudaSynthSync(t, r)
	cudaTestClose(t, "synthetic RMS, 4 groups, unweighted", cudaTestRead(t, dout), cudaTestRMS(x, nil, 4))

	p, weight := cudaSynthF32(rnd, len(x), 1)
	w := cudaSynthTensor(t, r, "synth.norm", 0, []uint64{uint64(len(x))}, p)
	if err := r.encodeRMS(dx, w, dout, 1, true); err != nil {
		t.Fatal(err)
	}
	cudaSynthSync(t, r)
	cudaTestClose(t, "synthetic RMS, weighted", cudaTestRead(t, dout), cudaTestRMS(x, weight, 1))
}

func TestCUDASynthRMSPair(t *testing.T) {
	r := cudaSynthFoundation(t)
	rnd := rand.New(rand.NewSource(5))
	const width0 = 1024
	const width1 = 512

	x0 := cudaSynthVector(rnd, width0, 1)
	p0, weight0 := cudaSynthF32(rnd, width0, 1)
	w0 := cudaSynthTensor(t, r, "synth.norm0", 0, []uint64{width0}, p0)
	dx0 := cudaSynthUpload(t, r, x0)
	dout0 := cudaTestAlloc(t, r, width0)

	x1 := cudaSynthVector(rnd, width1, 1)
	p1, weight1 := cudaSynthF32(rnd, width1, 1)
	w1 := cudaSynthTensor(t, r, "synth.norm1", 0, []uint64{width1}, p1)
	dx1 := cudaSynthUpload(t, r, x1)
	dout1 := cudaTestAlloc(t, r, width1)

	if err := r.encodeRMSPair(dx0, w0, dout0, dx1, w1, dout1); err != nil {
		t.Fatal(err)
	}
	cudaSynthSync(t, r)

	cudaTestClose(t, "synthetic RMS pair tensor 0 (1024)", cudaTestRead(t, dout0), cudaTestRMS(x0, weight0, 1))
	cudaTestClose(t, "synthetic RMS pair tensor 1 (512)", cudaTestRead(t, dout1), cudaTestRMS(x1, weight1, 1))
}


func TestCUDASynthEmbeddingRow(t *testing.T) {
	r := cudaSynthFoundation(t)
	rnd := rand.New(rand.NewSource(4))
	const rows, cols = 8, 4096
	p, v := cudaSynthF16(rnd, rows, cols, -2)
	e := cudaSynthTensor(t, r, "synth.embd", 1, []uint64{cols, rows}, p)
	dout := cudaTestAlloc(t, r, cols)
	const row = 5
	if err := r.encodeEmbeddingRow(e, row, dout); err != nil {
		t.Fatal(err)
	}
	cudaSynthSync(t, r)
	cudaTestClose(t, "synthetic embedding row", cudaTestRead(t, dout), v[row*cols:(row+1)*cols])
}

// cudaSynthHC builds the hyper-connection tensor set shared by the pre, post,
// and output-head kernels. The projection is deliberately small so the mixture
// gates stay away from saturation.
func cudaSynthHC(t *testing.T, r *cudaFoundation, rnd *rand.Rand, state []float32) (hcFn cudaModelTensor, fn []byte, scale, base cudaModelTensor, scaleV, baseV []float32) {
	t.Helper()
	fn, _ = cudaSynthF16(rnd, 24, 4*4096, -11)
	hcFn = cudaSynthTensor(t, r, "synth.hc_fn", 1, []uint64{4 * 4096, 24}, fn)
	scaleBytes, scaleV := cudaSynthF32(rnd, 3, 1)
	baseBytes, baseV := cudaSynthF32(rnd, 24, 1)
	scale = cudaSynthTensor(t, r, "synth.hc_scale", 0, []uint64{3}, scaleBytes)
	base = cudaSynthTensor(t, r, "synth.hc_base", 0, []uint64{24}, baseBytes)
	return hcFn, fn, scale, base, scaleV, baseV
}

func TestCUDASynthHCPre(t *testing.T) {
	r := cudaSynthFoundation(t)
	rnd := rand.New(rand.NewSource(5))
	state := cudaSynthVector(rnd, 4*4096, 1)
	hcFn, fn, scale, base, scaleV, baseV := cudaSynthHC(t, r, rnd, state)
	normBytes, normV := cudaSynthF32(rnd, 4096, 1)
	norm := cudaSynthTensor(t, r, "synth.attn_norm", 0, []uint64{4096}, normBytes)

	dState := cudaSynthUpload(t, r, state)
	dOut := cudaTestAlloc(t, r, 2*4096)
	if err := r.encodeHCPre(dState, hcFn, scale, base, norm, dOut); err != nil {
		t.Fatal(err)
	}
	cudaSynthSync(t, r)
	mix := cudaTestF16MatvecBytes(fn, cudaTestRMS(state, nil, 1), 24)
	wantCur, wantX := cudaTestHCPre(state, mix, scaleV, baseV, normV)
	cudaTestClose(t, "synthetic HC pre", cudaTestRead(t, dOut), append(wantCur, wantX...))
}

func TestCUDASynthHCPost(t *testing.T) {
	r := cudaSynthFoundation(t)
	rnd := rand.New(rand.NewSource(6))
	state := cudaSynthVector(rnd, 4*4096, 1)
	block := cudaSynthVector(rnd, 4096, 1)
	hcFn, fn, scale, base, scaleV, baseV := cudaSynthHC(t, r, rnd, state)

	dState := cudaSynthUpload(t, r, state)
	dBlock := cudaSynthUpload(t, r, block)
	dOut := cudaTestAlloc(t, r, len(state))
	if err := r.encodeHCPost(dState, dBlock, hcFn, scale, base, dOut); err != nil {
		t.Fatal(err)
	}
	cudaSynthSync(t, r)
	mix := cudaTestF16MatvecBytes(fn, cudaTestRMS(state, nil, 1), 24)
	want := cudaTestHCPost(state, block, mix, scaleV, baseV)
	cudaTestClose(t, "synthetic HC post", cudaTestRead(t, dOut), want)
}

func TestCUDASynthOutputHead(t *testing.T) {
	r := cudaSynthFoundation(t)
	rnd := rand.New(rand.NewSource(7))
	state := cudaSynthVector(rnd, 4*4096, 1)
	hcFn, fn, scale, base, scaleV, baseV := cudaSynthHC(t, r, rnd, state)
	normBytes, normV := cudaSynthF32(rnd, 4096, 1)
	const vocabulary = 128
	weightBytes := cudaSynthQ80(rnd, vocabulary, 4096, -6)
	out := cudaOutput{
		hcFn:    hcFn,
		hcScale: scale,
		hcBase:  base,
		norm:    cudaSynthTensor(t, r, "synth.output_norm", 0, []uint64{4096}, normBytes),
		weight:  cudaSynthTensor(t, r, "synth.output", 8, []uint64{4096, vocabulary}, weightBytes),
	}

	dState := cudaSynthUpload(t, r, state)
	dMixed := cudaTestAlloc(t, r, 2*4096)
	dLogits := cudaTestAlloc(t, r, vocabulary)
	if err := r.encodeOutputHead(dState, out, dMixed, dLogits); err != nil {
		t.Fatal(err)
	}
	cudaSynthSync(t, r)
	mix := cudaTestF16MatvecBytes(fn, cudaTestRMS(state, nil, 1), 24)
	flat, normalized := cudaTestHCMix(state, mix, scaleV, baseV, normV)
	cudaTestClose(t, "synthetic output transform", cudaTestRead(t, dMixed), append(flat, normalized...))
	want := make([]float32, vocabulary)
	rowBytes := 4096 / 32 * 34
	for row := range want {
		want[row] = cudaTestQ80Row(weightBytes[row*rowBytes:], normalized)
	}
	cudaTestClose(t, "synthetic logits", cudaTestRead(t, dLogits), want)
}

// TestCUDASynthRawAttention walks consecutive positions past the cache wrap.
// Position zero leaves the cache empty, so a wrong ring index would still
// produce the right answer there; the cases below run deep enough that prior
// keys must be read back, and the second uses the layer loop's real window so
// the wraparound is the one the model would hit.
func TestCUDASynthRawAttention(t *testing.T) {
	r := cudaSynthFoundation(t)
	for _, tt := range []struct {
		name     string
		capacity int
		last     int
	}{
		{"small window", 4, 6},
		{"model window", cudaRawCap, cudaRawCap + 3},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if testing.Short() && tt.capacity == cudaRawCap {
				t.Skip("the model window reference is slow")
			}
			cudaSynthRawAttention(t, r, tt.capacity, tt.last)
		})
	}
}

func cudaSynthRawAttention(t *testing.T, r *cudaFoundation, capacity, last int) {
	t.Helper()
	rnd := rand.New(rand.NewSource(8))
	sinksBytes, sinksV := cudaSynthF32(rnd, 64, 1)
	sinks := cudaSynthTensor(t, r, "synth.attn_sinks", 0, []uint64{64}, sinksBytes)

	dQuery := cudaTestAlloc(t, r, 64*512)
	dKV := cudaTestAlloc(t, r, 512)
	dCache := cudaSynthUpload(t, r, make([]float32, capacity*512))
	dScores := cudaTestAlloc(t, r, 64*capacity)
	dOut := cudaTestAlloc(t, r, 32768+512+32768)
	cache := make([]float32, capacity*512)

	for position := range last {
		query := cudaSynthVector(rnd, 64*512, 1)
		kv := cudaSynthVector(rnd, 512, 1)
		if err := dQuery.CopyFromHost(query); err != nil {
			t.Fatal(err)
		}
		if err := dKV.CopyFromHost(kv); err != nil {
			t.Fatal(err)
		}
		if err := r.encodeRawAttention(dQuery, dKV, sinks, dCache, dScores, dOut, position); err != nil {
			t.Fatal(err)
		}
		cudaSynthSync(t, r)
		want := cudaTestRawAttention(query, kv, sinksV, cache, position, capacity)
		// Every position advances both caches; the wrap boundary and the
		// ends get the full output comparison.
		if position < 2 || position+3 >= last || position%capacity < 2 {
			cudaTestClose(t, "synthetic raw attention at position "+strconv.Itoa(position), cudaTestRead(t, dOut), want)
		}
		cudaTestClose(t, "synthetic raw cache at position "+strconv.Itoa(position), cudaTestRead(t, dCache), cache)
	}
	t.Logf("raw attention: %d positions over a %d row cache", last, capacity)
}

func TestCUDASynthRouters(t *testing.T) {
	r := cudaSynthFoundation(t)
	rnd := rand.New(rand.NewSource(9))
	logits := cudaSynthVector(rnd, 256, 4)
	dLogits := cudaSynthUpload(t, r, logits)
	dScores := cudaTestAlloc(t, r, 256)
	dWeights := cudaTestAlloc(t, r, 6)
	dSelected, err := cudamem.Alloc[int32](r.device, 6)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = dSelected.Free() })
	gotSelected := make([]int32, 6)

	biasBytes, biasV := cudaSynthF32(rnd, 256, 1)
	bias := cudaSynthTensor(t, r, "synth.exp_probs_b", 0, []uint64{256}, biasBytes)
	if err := r.encodeRouterBiased(dLogits, bias, dScores, dSelected, dWeights); err != nil {
		t.Fatal(err)
	}
	cudaSynthSync(t, r)
	wantScores, wantSelected, wantWeights := cudaTestRouterBiased(logits, biasV)
	cudaTestClose(t, "synthetic biased router scores", cudaTestRead(t, dScores), wantScores)
	cudaTestClose(t, "synthetic biased router weights", cudaTestRead(t, dWeights), wantWeights)
	if err := dSelected.CopyToHost(gotSelected); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(gotSelected, wantSelected) {
		t.Fatalf("biased router experts = %v, want %v", gotSelected, wantSelected)
	}

	const token, tokens = 7, 64
	tableBytes, tableV := cudaSynthI32(rnd, 6*tokens, 256)
	table := cudaSynthTensor(t, r, "synth.tid2eid", 26, []uint64{6, tokens}, tableBytes)
	if err := r.encodeRouterHash(dLogits, table, dScores, dSelected, dWeights, token); err != nil {
		t.Fatal(err)
	}
	cudaSynthSync(t, r)
	wantScores, wantSelected, wantWeights = cudaTestRouterHash(logits, tableV, token)
	cudaTestClose(t, "synthetic hash router scores", cudaTestRead(t, dScores), wantScores)
	cudaTestClose(t, "synthetic hash router weights", cudaTestRead(t, dWeights), wantWeights)
	if err := dSelected.CopyToHost(gotSelected); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(gotSelected, wantSelected) {
		t.Fatalf("hash router experts = %v, want %v", gotSelected, wantSelected)
	}
}

func TestCUDASynthSwiGLUAndSum(t *testing.T) {
	r := cudaSynthFoundation(t)
	rnd := rand.New(rand.NewSource(10))
	const slots, width = 6, 2048
	gate := cudaSynthVector(rnd, slots*width, 12)
	up := cudaSynthVector(rnd, slots*width, 12)
	weights := cudaSynthVector(rnd, slots, 1)
	dGate := cudaSynthUpload(t, r, gate)
	dUp := cudaSynthUpload(t, r, up)
	dWeights := cudaSynthUpload(t, r, weights)
	dMid := cudaTestAlloc(t, r, len(gate))
	dSum := cudaTestAlloc(t, r, width)
	if err := r.encodeSwiGLUSlots(dGate, dUp, dWeights, dMid, slots, 0); err != nil {
		t.Fatal(err)
	}
	if err := r.encodeSumVectors(dMid, dSum, slots); err != nil {
		t.Fatal(err)
	}
	cudaSynthSync(t, r)
	wantMid := cudaTestSwiGLU(gate, up, weights, width)
	wantSum := make([]float32, width)
	for slot := range slots {
		for i := range width {
			wantSum[i] += wantMid[slot*width+i]
		}
	}
	cudaTestClose(t, "synthetic slotted SwiGLU", cudaTestRead(t, dMid), wantMid)
	cudaTestClose(t, "synthetic sum vectors", cudaTestRead(t, dSum), wantSum)

	// A zero weights slice is the scalar-weight path and passes a null pointer.
	dOne := cudaTestAlloc(t, r, len(gate))
	if err := r.encodeSwiGLU(dGate, dUp, dOne, 0.5); err != nil {
		t.Fatal(err)
	}
	cudaSynthSync(t, r)
	scalar := make([]float32, slots)
	for i := range scalar {
		scalar[i] = 0.5
	}
	cudaTestClose(t, "synthetic scalar SwiGLU", cudaTestRead(t, dOne), cudaTestSwiGLU(gate, up, scalar, width))
}
