//go:build linux

package main

import (
	"context"
	"encoding/binary"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"math"
	"os"
	"strconv"
	"testing"
)

func TestCUDAHCPost(t *testing.T) {
	path := os.Getenv("DS4_MODEL")
	if path == "" {
		t.Skip("DS4_MODEL is not set")
	}
	r, err := newCUDAFoundation(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := r.Close(); err != nil {
			t.Error(err)
		}
	})
	if !r.hmm {
		t.Skip("HC post requires HMM tensor views")
	}

	hcFn := cudaTestTensor(t, r, "blk.0.hc_attn_fn.weight")
	hcScale := cudaTestTensor(t, r, "blk.0.hc_attn_scale.weight")
	hcBase := cudaTestTensor(t, r, "blk.0.hc_attn_base.weight")
	state := make([]float32, 4*4096)
	for i := range state {
		state[i] = float32((i*29+7)%251-125) / 125
	}
	block := make([]float32, 4096)
	for i := range block {
		block[i] = float32((i*13+3)%199-99) / 199
	}

	dState := cudaTestAlloc(t, r, len(state))
	dBlock := cudaTestAlloc(t, r, len(block))
	dOut := cudaTestAlloc(t, r, len(state))
	if err := dState.CopyFromHost(state); err != nil {
		t.Fatal(err)
	}
	if err := dBlock.CopyFromHost(block); err != nil {
		t.Fatal(err)
	}
	if err := r.encodeHCPost(dState, dBlock, hcFn, hcScale, hcBase, dOut); err != nil {
		t.Fatal(err)
	}
	if err := r.device.DefaultStream().Synchronize(context.Background()); err != nil {
		t.Fatal(err)
	}

	mix := cudaTestF16Matvec(t, r, hcFn, cudaTestRMS(state, nil, 1))
	want := cudaTestHCPost(state, block, mix, cudaTestF32Tensor(t, r, hcScale), cudaTestF32Tensor(t, r, hcBase))
	cudaTestClose(t, "HC post", cudaTestRead(t, dOut), want)
}

func TestCUDARawAttention(t *testing.T) {
	path := os.Getenv("DS4_MODEL")
	if path == "" {
		t.Skip("DS4_MODEL is not set")
	}
	r, err := newCUDAFoundation(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := r.Close(); err != nil {
			t.Error(err)
		}
	})
	if !r.hmm {
		t.Skip("raw attention requires HMM tensor views")
	}

	const capacity = 4
	sinksTensor := cudaTestTensor(t, r, "blk.0.attn_sinks.weight")
	sinks := cudaTestF32Tensor(t, r, sinksTensor)
	dQuery := cudaTestAlloc(t, r, 64*512)
	dKV := cudaTestAlloc(t, r, 512)
	dCache := cudaTestAlloc(t, r, capacity*512)
	dScores := cudaTestAlloc(t, r, 64*capacity)
	dOut := cudaTestAlloc(t, r, 32768+512+32768)
	if err := dCache.CopyFromHost(make([]float32, capacity*512)); err != nil {
		t.Fatal(err)
	}
	cache := make([]float32, capacity*512)

	// Six positions walk the rolling cache past its wrap.
	for position := range 6 {
		query := make([]float32, 64*512)
		for i := range query {
			query[i] = float32((i*31+position*17)%211-105) / 210
		}
		kv := make([]float32, 512)
		for i := range kv {
			kv[i] = float32((i*23+position*5)%181-90) / 180
		}
		if err := dQuery.CopyFromHost(query); err != nil {
			t.Fatal(err)
		}
		if err := dKV.CopyFromHost(kv); err != nil {
			t.Fatal(err)
		}
		if err := r.encodeRawAttention(dQuery, dKV, sinksTensor, dCache, dScores, dOut, position); err != nil {
			t.Fatal(err)
		}
		if err := r.device.DefaultStream().Synchronize(context.Background()); err != nil {
			t.Fatal(err)
		}
		want := cudaTestRawAttention(query, kv, sinks, cache, position, capacity)
		cudaTestClose(t, "raw attention at position "+strconv.Itoa(position), cudaTestRead(t, dOut), want)
		cudaTestClose(t, "raw cache at position "+strconv.Itoa(position), cudaTestRead(t, dCache), cache)
	}
}

func TestCUDARouters(t *testing.T) {
	path := os.Getenv("DS4_MODEL")
	if path == "" {
		t.Skip("DS4_MODEL is not set")
	}
	r, err := newCUDAFoundation(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := r.Close(); err != nil {
			t.Error(err)
		}
	})
	if !r.hmm {
		t.Skip("routers require HMM tensor views")
	}

	logits := make([]float32, 256)
	for i := range logits {
		logits[i] = float32((i*67+13)%173-86) / 43
	}
	dLogits := cudaTestAlloc(t, r, 256)
	dScores := cudaTestAlloc(t, r, 256)
	dWeights := cudaTestAlloc(t, r, 6)
	dSelected, err := cudamem.Alloc[int32](r.device, 6)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = dSelected.Free() })
	if err := dLogits.CopyFromHost(logits); err != nil {
		t.Fatal(err)
	}
	gotSelected := make([]int32, 6)

	bias := cudaTestTensor(t, r, "blk.3.exp_probs_b.bias")
	if err := r.encodeRouterBiased(dLogits, bias, dScores, dSelected, dWeights); err != nil {
		t.Fatal(err)
	}
	if err := r.device.DefaultStream().Synchronize(context.Background()); err != nil {
		t.Fatal(err)
	}
	wantScores, wantSelected, wantWeights := cudaTestRouterBiased(logits, cudaTestF32Tensor(t, r, bias))
	cudaTestClose(t, "biased router scores", cudaTestRead(t, dScores), wantScores)
	cudaTestClose(t, "biased router weights", cudaTestRead(t, dWeights), wantWeights)
	if err := dSelected.CopyToHost(gotSelected); err != nil {
		t.Fatal(err)
	}
	for i := range gotSelected {
		if gotSelected[i] != wantSelected[i] {
			t.Fatalf("biased router expert %d = %d, want %d", i, gotSelected[i], wantSelected[i])
		}
	}

	const token = 7
	table := cudaTestTensor(t, r, "blk.0.ffn_gate_tid2eid.weight")
	if err := r.encodeRouterHash(dLogits, table, dScores, dSelected, dWeights, token); err != nil {
		t.Fatal(err)
	}
	if err := r.device.DefaultStream().Synchronize(context.Background()); err != nil {
		t.Fatal(err)
	}
	wantScores, wantSelected, wantWeights = cudaTestRouterHash(logits, cudaTestI32Tensor(t, r, table), token)
	cudaTestClose(t, "hash router scores", cudaTestRead(t, dScores), wantScores)
	cudaTestClose(t, "hash router weights", cudaTestRead(t, dWeights), wantWeights)
	if err := dSelected.CopyToHost(gotSelected); err != nil {
		t.Fatal(err)
	}
	for i := range gotSelected {
		if gotSelected[i] != wantSelected[i] {
			t.Fatalf("hash router expert %d = %d, want %d", i, gotSelected[i], wantSelected[i])
		}
	}
}

func TestCUDASwiGLUAndSum(t *testing.T) {
	path := os.Getenv("DS4_MODEL")
	if path == "" {
		t.Skip("DS4_MODEL is not set")
	}
	r, err := newCUDAFoundation(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := r.Close(); err != nil {
			t.Error(err)
		}
	})

	const slots, width = 6, 2048
	gate := make([]float32, slots*width)
	up := make([]float32, slots*width)
	weights := make([]float32, slots)
	for i := range gate {
		gate[i] = float32((i*37+5)%401-200) / 20
		up[i] = float32((i*53+11)%307-153) / 15
	}
	for i := range weights {
		weights[i] = float32(i+1) / 8
	}
	dGate := cudaTestAlloc(t, r, len(gate))
	dUp := cudaTestAlloc(t, r, len(up))
	dWeights := cudaTestAlloc(t, r, len(weights))
	dMid := cudaTestAlloc(t, r, len(gate))
	dSum := cudaTestAlloc(t, r, width)
	if err := dGate.CopyFromHost(gate); err != nil {
		t.Fatal(err)
	}
	if err := dUp.CopyFromHost(up); err != nil {
		t.Fatal(err)
	}
	if err := dWeights.CopyFromHost(weights); err != nil {
		t.Fatal(err)
	}
	if err := r.encodeSwiGLUSlots(dGate, dUp, dWeights, dMid, slots, 0); err != nil {
		t.Fatal(err)
	}
	if err := r.encodeSumVectors(dMid, dSum, slots); err != nil {
		t.Fatal(err)
	}
	if err := r.device.DefaultStream().Synchronize(context.Background()); err != nil {
		t.Fatal(err)
	}
	wantMid := cudaTestSwiGLU(gate, up, weights, width)
	wantSum := make([]float32, width)
	for slot := range slots {
		for i := range width {
			wantSum[i] += wantMid[slot*width+i]
		}
	}
	cudaTestClose(t, "slotted SwiGLU", cudaTestRead(t, dMid), wantMid)
	cudaTestClose(t, "sum vectors", cudaTestRead(t, dSum), wantSum)
}

func TestCUDAOutputHead(t *testing.T) {
	path := os.Getenv("DS4_MODEL")
	if path == "" {
		t.Skip("DS4_MODEL is not set")
	}
	r, err := newCUDAFoundation(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := r.Close(); err != nil {
			t.Error(err)
		}
	})
	if !r.hmm {
		t.Skip("the output head requires HMM tensor views")
	}
	m, err := r.bindModel()
	if err != nil {
		t.Fatal(err)
	}

	state := make([]float32, 4*4096)
	for i := range state {
		state[i] = float32((i*41+19)%263-131) / 131
	}
	dState := cudaTestAlloc(t, r, len(state))
	dMixed := cudaTestAlloc(t, r, 2*4096)
	dLogits := cudaTestAlloc(t, r, int(m.output.weight.dimensions[1]))
	if err := dState.CopyFromHost(state); err != nil {
		t.Fatal(err)
	}
	if err := r.encodeOutputHead(dState, m.output, dMixed, dLogits); err != nil {
		t.Fatal(err)
	}
	if err := r.device.DefaultStream().Synchronize(context.Background()); err != nil {
		t.Fatal(err)
	}

	mix := cudaTestF16Matvec(t, r, m.output.hcFn, cudaTestRMS(state, nil, 1))
	flat, normalized := cudaTestHCMix(state, mix,
		cudaTestF32Tensor(t, r, m.output.hcScale), cudaTestF32Tensor(t, r, m.output.hcBase),
		cudaTestF32Tensor(t, r, m.output.norm))
	cudaTestClose(t, "output transform", cudaTestRead(t, dMixed), append(flat, normalized...))

	// Checking every one of the 129280 rows on the CPU would dominate the
	// test, so compare a leading window and require the rest to be finite.
	const window = 64
	got := cudaTestRead(t, dLogits)
	weights := r.model[m.output.weight.fileOffset : m.output.weight.fileOffset+m.output.weight.bytes]
	rowBytes := len(normalized) / 32 * 34
	want := make([]float32, window)
	for row := range want {
		want[row] = cudaTestQ80Row(weights[row*rowBytes:], normalized)
	}
	cudaTestClose(t, "leading logits", got[:window], want)
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("logit %d = %g", i, v)
		}
	}
	t.Logf("output head: %d logits, %d compared against the CPU reference", len(got), window)
}

// TestCUDAFloat16RoundReference checks the half round-trip reference against
// the widening reference that the Q8_0 and F16 kernels already share. It needs
// no device.
func cudaTestI32Tensor(t *testing.T, r *cudaFoundation, tensor cudaModelTensor) []int32 {
	t.Helper()
	if tensor.typeID != 26 {
		t.Fatalf("invalid I32 test tensor %q", tensor.name)
	}
	p := r.model[tensor.fileOffset : tensor.fileOffset+tensor.bytes]
	out := make([]int32, tensor.bytes/4)
	for i := range out {
		out[i] = int32(binary.LittleEndian.Uint32(p[i*4:]))
	}
	return out
}

// cudaTestHCPost mirrors ds4_hc_post: it derives the output gate and the
// Sinkhorn normalized combination matrix, then folds the block output back into
// the four-stream state.
