//go:build linux

package main

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"slices"
	"strconv"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamem"
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

// TestCUDAFloat16RoundReference checks the half round-trip reference against
// the widening reference that the Q8_0 and F16 kernels already share. It needs
// no device.
func TestCUDAFloat16RoundReference(t *testing.T) {
	for _, tt := range []struct {
		name string
		in   float32
		want float32
	}{
		{"zero", 0, 0},
		{"negative zero", float32(math.Copysign(0, -1)), float32(math.Copysign(0, -1))},
		{"one", 1, 1},
		{"exact half", 0.5, 0.5},
		{"tie to even down", 1 + 1.0/2048, 1},
		{"tie to even up", 1 + 3.0/2048, 1 + 2.0/1024},
		{"smallest normal", 6.103515625e-05, 6.103515625e-05},
		{"smallest subnormal", 5.9604645e-08, 5.9604645e-08},
		{"below subnormal", 1e-08, 0},
		{"largest finite", 65504, 65504},
		{"overflow", 1e6, float32(math.Inf(1))},
		{"negative", -3.14159265, -3.140625},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got := cudaTestFloat16Round(tt.in)
			if got != tt.want || math.Signbit(float64(got)) != math.Signbit(float64(tt.want)) {
				t.Fatalf("cudaTestFloat16Round(%g) = %g, want %g", tt.in, got, tt.want)
			}
			if again := cudaTestFloat16Round(got); again != got {
				t.Fatalf("cudaTestFloat16Round is not idempotent: %g then %g", got, again)
			}
		})
	}
}

// TestCUDARouterBiasedReference pins the contract's routing rule: bias decides
// which experts run, and the unbiased scores decide their weights.
func TestCUDARouterBiasedReference(t *testing.T) {
	logits := make([]float32, 256)
	bias := make([]float32, 256)
	for i := range logits {
		logits[i] = float32(i) / 256
	}
	// Lift six otherwise low-scoring experts above every other candidate.
	promoted := []int{3, 17, 42, 99, 128, 200}
	for _, id := range promoted {
		bias[id] = 100
	}
	scores, selected, weights := cudaTestRouterBiased(logits, bias)
	got := slices.Clone(selected)
	slices.Sort(got)
	for i, id := range promoted {
		if int(got[i]) != id {
			t.Fatalf("selected experts = %v, want %v", got, promoted)
		}
	}
	var sum float32
	for _, id := range selected {
		sum += scores[id]
	}
	for slot, id := range selected {
		want := scores[id] / max(sum, 6.103515625e-5) * 1.5
		if math.Abs(float64(weights[slot]-want)) > 1e-6 {
			t.Fatalf("weight %d = %g, want %g; bias must not enter the mixture", slot, weights[slot], want)
		}
	}
	var total float32
	for _, w := range weights {
		total += w
	}
	if math.Abs(float64(total-1.5)) > 1e-5 {
		t.Fatalf("weights sum to %g, want 1.5", total)
	}
}

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
func cudaTestHCPost(state, block, mix, scale, base []float32) []float32 {
	post, comb := cudaTestHCSplit(mix, scale, base)
	out := make([]float32, len(state))
	for dst := range 4 {
		for d := range 4096 {
			v := block[d] * post[dst]
			for src := range 4 {
				v += comb[dst+src*4] * state[src*4096+d]
			}
			out[dst*4096+d] = v
		}
	}
	return out
}

func cudaTestHCSplit(mix, scale, base []float32) (post, comb []float32) {
	post = make([]float32, 4)
	comb = make([]float32, 16)
	for h := range 4 {
		post[h] = 2 / (1 + float32(math.Exp(-float64(mix[4+h]*scale[1]+base[4+h]))))
	}
	for dst := range 4 {
		mx := float32(-math.MaxFloat32)
		for src := range 4 {
			i := src + dst*4
			comb[i] = mix[8+i]*scale[2] + base[8+i]
			mx = max(mx, comb[i])
		}
		var sum float32
		for src := range 4 {
			i := src + dst*4
			comb[i] = float32(math.Exp(float64(comb[i] - mx)))
			sum += comb[i]
		}
		for src := range 4 {
			i := src + dst*4
			comb[i] = comb[i]/sum + 1e-6
		}
	}
	for src := range 4 {
		var sum float32
		for dst := range 4 {
			sum += comb[src+dst*4]
		}
		for dst := range 4 {
			comb[src+dst*4] /= sum + 1e-6
		}
	}
	for range 19 {
		for dst := range 4 {
			var sum float32
			for src := range 4 {
				sum += comb[src+dst*4]
			}
			for src := range 4 {
				comb[src+dst*4] /= sum + 1e-6
			}
		}
		for src := range 4 {
			var sum float32
			for dst := range 4 {
				sum += comb[src+dst*4]
			}
			for dst := range 4 {
				comb[src+dst*4] /= sum + 1e-6
			}
		}
	}
	return post, comb
}

// cudaTestRawAttention mirrors ds4_rope_attention_raw. It advances cache in
// place so a caller can walk consecutive positions, and returns the kernel's
// full output buffer.
func cudaTestRawAttention(q, kv, sinks, cache []float32, position, capacity int) []float32 {
	out := make([]float32, 32768+512+32768)
	copy(out, q)
	copy(out[32768:], kv)
	step := float32(math.Pow(10000, -2.0/64.0))
	rotate := func(base int, sign float32) {
		theta := float32(position)
		for i := 0; i < 64; i += 2 {
			c := float32(math.Cos(float64(theta)))
			s := sign * float32(math.Sin(float64(theta)))
			a, b := out[base+i], out[base+i+1]
			out[base+i] = a*c - b*s
			out[base+i+1] = a*s + b*c
			theta *= step
		}
	}
	for head := range 64 {
		rotate(head*512+448, 1)
	}
	rotate(32768+448, 1)
	for off := 0; off < 448; off += 64 {
		amax := float32(1e-4)
		for i := range 64 {
			amax = max(amax, float32(math.Abs(float64(out[32768+off+i]))))
		}
		scale := float32(math.Exp2(math.Ceil(math.Log2(float64(amax / 448)))))
		for i := range 64 {
			out[32768+off+i] = cudaTestE4M3Round(min(max(out[32768+off+i]/scale, -448), 448)) * scale
		}
	}
	row := position % capacity
	for i := range 512 {
		cache[row*512+i] = cudaTestFloat16Round(out[32768+i])
	}
	count := min(position+1, capacity)
	first := position + 1 - count
	for head := range 64 {
		qoff := head * 512
		scores := make([]float32, count)
		maximum := sinks[head]
		for t := range count {
			r := (first + t) % capacity
			var score float32
			for i := range 512 {
				score += out[qoff+i] * cache[r*512+i]
			}
			score /= float32(math.Sqrt(512))
			scores[t] = score
			maximum = max(maximum, score)
		}
		denom := float32(math.Exp(float64(sinks[head] - maximum)))
		for t := range count {
			denom += float32(math.Exp(float64(scores[t] - maximum)))
		}
		for i := range 512 {
			var value float32
			for t := range count {
				r := (first + t) % capacity
				value += float32(math.Exp(float64(scores[t]-maximum))) * cache[r*512+i]
			}
			out[33280+qoff+i] = value / denom
		}
		rotate(33280+qoff+448, -1)
	}
	return out
}

func cudaTestRouterScores(logits []float32) []float32 {
	out := make([]float32, len(logits))
	for i, x := range logits {
		softplus := x
		if x <= 20 {
			softplus = float32(math.Log(1 + math.Exp(float64(x))))
		}
		out[i] = float32(math.Sqrt(float64(softplus)))
	}
	return out
}

// cudaTestRouterBiased mirrors ds4_router_biased: bias selects, and the
// unbiased scores weight.
func cudaTestRouterBiased(logits, bias []float32) ([]float32, []int32, []float32) {
	scores := cudaTestRouterScores(logits)
	selected := make([]int32, 6)
	for k := range selected {
		best := -1
		for i := range scores {
			used := false
			for j := range k {
				used = used || selected[j] == int32(i)
			}
			if used {
				continue
			}
			if best < 0 || scores[i]+bias[i] > scores[best]+bias[best] {
				best = i
			}
		}
		selected[k] = int32(best)
	}
	return scores, selected, cudaTestRouterWeights(scores, selected)
}

// cudaTestRouterHash mirrors ds4_router_hash for the dense layers.
func cudaTestRouterHash(logits []float32, table []int32, token uint32) ([]float32, []int32, []float32) {
	scores := cudaTestRouterScores(logits)
	selected := slices.Clone(table[int(token)*6 : int(token)*6+6])
	return scores, selected, cudaTestRouterWeights(scores, selected)
}

func cudaTestRouterWeights(scores []float32, selected []int32) []float32 {
	var sum float32
	for _, id := range selected {
		sum += scores[id]
	}
	sum = max(sum, 6.103515625e-5)
	weights := make([]float32, len(selected))
	for i, id := range selected {
		weights[i] = scores[id] / sum * 1.5
	}
	return weights
}

func cudaTestSwiGLU(gate, up, weights []float32, width int) []float32 {
	out := make([]float32, len(gate))
	for i := range gate {
		x := min(gate[i], 10)
		y := min(max(up[i], -10), 10)
		out[i] = x / (1 + float32(math.Exp(float64(-x)))) * y * weights[i/width]
	}
	return out
}

func cudaTestE4M3Value(code int) float32 {
	e, m := (code>>3)&15, code&7
	if e == 0 {
		return float32(m) * 0.001953125
	}
	return (1 + float32(m)*0.125) * float32(math.Exp2(float64(e-7)))
}

// cudaTestE4M3Round mirrors ds4_e4m3fn_round, including its tie rule.
func cudaTestE4M3Round(value float32) float32 {
	sign := float32(1)
	if value < 0 {
		sign = -1
	}
	ax := min(float32(math.Abs(float64(value))), 448)
	lo, hi := 0, 126
	for lo < hi {
		mid := (lo + hi + 1) >> 1
		if cudaTestE4M3Value(mid) <= ax {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	best := lo
	if best < 126 {
		bd := math.Abs(float64(ax - cudaTestE4M3Value(best)))
		nd := math.Abs(float64(ax - cudaTestE4M3Value(best+1)))
		if nd < bd || (nd == bd && (best+1)&1 == 0 && best&1 != 0) {
			best++
		}
	}
	return sign * cudaTestE4M3Value(best)
}

// cudaTestFloat16Round mirrors ds4_half_round: one round trip through IEEE
// binary16 with round-half-to-even.
func cudaTestFloat16Round(value float32) float32 {
	x := math.Float32bits(value)
	sign := uint16(x>>16) & 0x8000
	mag := x & 0x7fffffff
	switch {
	case mag >= 0x7f800000:
		return value
	case mag >= 0x477ff000:
		return cudaTestFloat16(sign | 0x7c00)
	case mag >= 0x38800000:
		bits := mag + 0x00000fff + (mag>>13)&1
		return cudaTestFloat16(sign | uint16((bits-0x38000000)>>13))
	case mag >= 0x33000000:
		shift := 126 - int(mag>>23)
		m := mag&0x007fffff | 0x00800000
		h := m >> shift
		rem := m & (1<<shift - 1)
		mid := uint32(1) << (shift - 1)
		if rem > mid || (rem == mid && h&1 != 0) {
			h++
		}
		return cudaTestFloat16(sign | uint16(h))
	}
	return cudaTestFloat16(sign)
}
