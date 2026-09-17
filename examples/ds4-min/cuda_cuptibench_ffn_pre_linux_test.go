//go:build linux

package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamem"
)

// BenchmarkCUDAFFNPre measures the components of FFN preprocessing in isolation:
// Q8_1 input quantization (sequential vs batched), router GEMV, router top-6
// selection, and hidden connection post/pre transforms.
func BenchmarkCUDAFFNPre(b *testing.B) {
	if !*ds4CUPTIBench {
		b.Skip("enable with -ds4-cuptibench under gpu-lock")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		b.Fatal("gpu-lock is required")
	}
	for _, tokens := range cudaBenchSizes(b, "DS4_BENCH_TOKENS", []int{1, 16, 256}) {
		b.Run(fmt.Sprintf("tokens=%d", tokens), func(b *testing.B) {
			b.Run("quant_q81/sequential", func(b *testing.B) {
				cudaCUPTIFFNQuantQ81Sequential(b, tokens)
			})
			b.Run("quant_q81/batched", func(b *testing.B) {
				cudaCUPTIFFNQuantQ81Batched(b, tokens)
			})
			b.Run("router_gemv", func(b *testing.B) {
				cudaCUPTIFFNRouterGEMV(b, tokens)
			})
			b.Run("router_biased", func(b *testing.B) {
				cudaCUPTIFFNRouterBiased(b, tokens)
			})
			b.Run("hc/pre", func(b *testing.B) {
				cudaCUPTIFFNHCPre(b, tokens)
			})
			b.Run("hc/post_pre", func(b *testing.B) {
				cudaCUPTIFFNHCPostPre(b, tokens)
			})
		})
	}
}

func cudaCUPTIFFNQuantQ81Sequential(b *testing.B, tokens int) {
	r := cudaQ80BenchFoundation(b)
	const cols = cudaWidth
	rowBytes := cols / 32 * 36

	hostX := make([]float32, tokens*cols)
	for i := range hostX {
		hostX[i] = float32((i*37+i/17)%257-128) / 128
	}
	in := cudaCUPTIUpload(b, r, hostX)
	out := cudaCUPTIUpload(b, r, make([]byte, tokens*rowBytes))

	launch := func() error {
		for t := 0; t < tokens; t++ {
			if err := r.encodeQuantQ81(in.Slice(t*cols, (t+1)*cols), out.Slice(t*rowBytes, (t+1)*rowBytes), cols); err != nil {
				return err
			}
		}
		return nil
	}

	want := cudaQ81ActivationReference(hostX)
	cudaCUPTIRunBytes(b, r, tokens, launch, []cudamem.Slice[byte]{out}, [][]byte{want})
}

func cudaCUPTIFFNQuantQ81Batched(b *testing.B, tokens int) {
	r := cudaQ80BenchFoundation(b)
	const cols = cudaWidth
	rowBytes := cols / 32 * 36

	hostX := make([]float32, tokens*cols)
	for i := range hostX {
		hostX[i] = float32((i*37+i/17)%257-128) / 128
	}
	in := cudaCUPTIUpload(b, r, hostX)
	out := cudaCUPTIUpload(b, r, make([]byte, tokens*rowBytes))

	launch := func() error {
		return r.encodeQuantQ81Batch(in, out, cols, cols, tokens)
	}

	want := cudaQ81ActivationReference(hostX)
	cudaCUPTIRunBytes(b, r, tokens, launch, []cudamem.Slice[byte]{out}, [][]byte{want})
}

func cudaCUPTIFFNRouterGEMV(b *testing.B, tokens int) {
	r := cudaQ80BenchFoundation(b)
	const cols = cudaWidth
	const rows = 256

	hostW := make([]byte, cols*rows*2)
	for i := 0; i < cols*rows; i++ {
		w := float32((i*19+13)%127-63) / 2048
		bits := cudaQ81HalfReference(w)
		binary.LittleEndian.PutUint16(hostW[2*i:], bits)
	}
	weightTensor := cudaCUPTITensor(b, r, "router.weight", 1, cols, rows, hostW)

	hostX := make([]float32, tokens*cols)
	for i := range hostX {
		hostX[i] = float32((i*31+7)%251-125) / 128
	}
	in := cudaCUPTIUpload(b, r, hostX)
	out := cudaCUPTIUpload(b, r, makeNaNs(tokens*rows))

	launch := func() error {
		for t := 0; t < tokens; t++ {
			if err := r.encodeF16MatvecDefault(weightTensor, in.Slice(t*cols, (t+1)*cols), out.Slice(t*rows, (t+1)*rows)); err != nil {
				return err
			}
		}
		return nil
	}

	ref := cudaCUPTIUpload(b, r, makeNaNs(tokens*rows))
	for t := 0; t < tokens; t++ {
		if err := r.encodeF16MatvecDefault(weightTensor, in.Slice(t*cols, (t+1)*cols), ref.Slice(t*rows, (t+1)*rows)); err != nil {
			b.Fatal(err)
		}
	}
	if err := syncStream(r.stream()); err != nil {
		b.Fatal(err)
	}
	want := make([]float32, ref.Len())
	if err := ref.CopyToHost(want); err != nil {
		b.Fatal(err)
	}
	cudaCUPTIRun(b, r, tokens, launch, []cudamem.Slice[float32]{out}, [][]float32{want})
}

func cudaCUPTIFFNRouterBiased(b *testing.B, tokens int) {
	r := cudaQ80BenchFoundation(b)
	const experts = 256
	const topK = 6

	hostLogits := make([]float32, tokens*experts)
	for i := range hostLogits {
		hostLogits[i] = float32((i*41+17)%253-126) / 32
	}
	logits := cudaCUPTIUpload(b, r, hostLogits)

	hostBias := make([]float32, experts)
	for i := range hostBias {
		hostBias[i] = float32((i*13+5)%127-63) / 64
	}
	rawBias := make([]byte, experts*4)
	for i, v := range hostBias {
		binary.LittleEndian.PutUint32(rawBias[4*i:], math.Float32bits(v))
	}
	biasTensor := cudaCUPTIVector(b, r, "expert_bias", 0, experts, rawBias)

	scores := cudaCUPTIUpload(b, r, makeNaNs(tokens*experts))
	selected := cudaCUPTIUpload(b, r, make([]int32, tokens*topK))
	weights := cudaCUPTIUpload(b, r, makeNaNs(tokens*topK))

	launch := func() error {
		for t := 0; t < tokens; t++ {
			if err := r.encodeRouterBiased(
				logits.Slice(t*experts, (t+1)*experts),
				biasTensor,
				scores.Slice(t*experts, (t+1)*experts),
				selected.Slice(t*topK, (t+1)*topK),
				weights.Slice(t*topK, (t+1)*topK),
			); err != nil {
				return err
			}
		}
		return nil
	}

	refScores := cudaCUPTIUpload(b, r, makeNaNs(tokens*experts))
	refSelected := cudaCUPTIUpload(b, r, make([]int32, tokens*topK))
	refWeights := cudaCUPTIUpload(b, r, makeNaNs(tokens*topK))
	for t := 0; t < tokens; t++ {
		if err := r.encodeRouterBiased(
			logits.Slice(t*experts, (t+1)*experts),
			biasTensor,
			refScores.Slice(t*experts, (t+1)*experts),
			refSelected.Slice(t*topK, (t+1)*topK),
			refWeights.Slice(t*topK, (t+1)*topK),
		); err != nil {
			b.Fatal(err)
		}
	}
	if err := syncStream(r.stream()); err != nil {
		b.Fatal(err)
	}
	wantScores := make([]float32, refScores.Len())
	wantWeights := make([]float32, refWeights.Len())
	if err := refScores.CopyToHost(wantScores); err != nil {
		b.Fatal(err)
	}
	if err := refWeights.CopyToHost(wantWeights); err != nil {
		b.Fatal(err)
	}
	cudaCUPTIRun(b, r, tokens, launch, []cudamem.Slice[float32]{scores, weights}, [][]float32{wantScores, wantWeights})
}

func cudaCUPTIFFNHCPre(b *testing.B, tokens int) {
	r := cudaQ80BenchFoundation(b)
	hc := make([]*cudaHC, tokens)
	for t := range hc {
		h, err := newCUDAHC(r)
		if err != nil {
			b.Fatal(err)
		}
		hc[t] = h
	}
	b.Cleanup(func() {
		for _, h := range hc {
			_ = h.Close()
		}
	})

	const (
		hcState = 16384
		hcRows  = 24
		hcNorm  = 4096
	)
	state := cudaCUPTIUpload(b, r, make([]float32, tokens*hcState))
	hostState := make([]float32, tokens*hcState)
	for i := range hostState {
		hostState[i] = float32((i*23+7)%251-125) / 64
	}
	if err := state.CopyFromHost(hostState); err != nil {
		b.Fatal(err)
	}

	rawW := make([]byte, hcRows*hcState*2)
	for i := 0; i < hcRows*hcState; i++ {
		w := float32((i*17+3)%127-63) / 4096
		bits := cudaQ81HalfReference(w)
		binary.LittleEndian.PutUint16(rawW[2*i:], bits)
	}
	fn := cudaCUPTITensor(b, r, "hc-fn", 1, hcState, hcRows, rawW)

	rawScale := make([]byte, 3*4)
	for i := range 3 {
		binary.LittleEndian.PutUint32(rawScale[4*i:], math.Float32bits(1.0))
	}
	scale := cudaCUPTIVector(b, r, "hc-scale", 0, 3, rawScale)

	rawBase := make([]byte, hcRows*4)
	for i := range hcRows {
		binary.LittleEndian.PutUint32(rawBase[4*i:], math.Float32bits(0.0))
	}
	base := cudaCUPTIVector(b, r, "hc-base", 0, hcRows, rawBase)

	rawNorm := make([]byte, hcNorm*4)
	for i := range hcNorm {
		binary.LittleEndian.PutUint32(rawNorm[4*i:], math.Float32bits(1.0))
	}
	norm := cudaCUPTIVector(b, r, "hc-norm", 0, hcNorm, rawNorm)

	out := cudaCUPTIUpload(b, r, makeNaNs(tokens*8192))

	launch := func() error {
		for t := 0; t < tokens; t++ {
			if err := hc[t].pre(state.Slice(t*hcState, (t+1)*hcState), fn, scale, base, norm, out.Slice(t*8192, (t+1)*8192), false); err != nil {
				return err
			}
		}
		return nil
	}

	ref := cudaCUPTIUpload(b, r, makeNaNs(tokens*8192))
	for t := 0; t < tokens; t++ {
		if err := hc[t].pre(state.Slice(t*hcState, (t+1)*hcState), fn, scale, base, norm, ref.Slice(t*8192, (t+1)*8192), false); err != nil {
			b.Fatal(err)
		}
	}
	if err := syncStream(r.stream()); err != nil {
		b.Fatal(err)
	}
	want := make([]float32, ref.Len())
	if err := ref.CopyToHost(want); err != nil {
		b.Fatal(err)
	}
	cudaCUPTIRun(b, r, tokens, launch, []cudamem.Slice[float32]{out}, [][]float32{want})
}

func cudaCUPTIFFNHCPostPre(b *testing.B, tokens int) {
	r := cudaQ80BenchFoundation(b)
	hc := make([]*cudaHC, tokens)
	for t := range hc {
		h, err := newCUDAHC(r)
		if err != nil {
			b.Fatal(err)
		}
		hc[t] = h
	}
	b.Cleanup(func() {
		for _, h := range hc {
			_ = h.Close()
		}
	})

	const (
		hcState = 16384
		hcRows  = 24
		hcNorm  = 4096
	)
	state := cudaCUPTIUpload(b, r, make([]float32, tokens*hcState))
	hostState := make([]float32, tokens*hcState)
	for i := range hostState {
		hostState[i] = float32((i*23+7)%251-125) / 64
	}
	if err := state.CopyFromHost(hostState); err != nil {
		b.Fatal(err)
	}

	block := cudaCUPTIUpload(b, r, make([]float32, tokens*hcNorm))
	hostBlock := make([]float32, tokens*hcNorm)
	for i := range hostBlock {
		hostBlock[i] = float32((i*19+5)%127-63) / 32
	}
	if err := block.CopyFromHost(hostBlock); err != nil {
		b.Fatal(err)
	}

	rawW := make([]byte, hcRows*hcState*2)
	for i := 0; i < hcRows*hcState; i++ {
		w := float32((i*17+3)%127-63) / 4096
		bits := cudaQ81HalfReference(w)
		binary.LittleEndian.PutUint16(rawW[2*i:], bits)
	}
	fn := cudaCUPTITensor(b, r, "hc-fn", 1, hcState, hcRows, rawW)

	rawScale := make([]byte, 3*4)
	for i := range 3 {
		binary.LittleEndian.PutUint32(rawScale[4*i:], math.Float32bits(1.0))
	}
	scale := cudaCUPTIVector(b, r, "hc-scale", 0, 3, rawScale)

	rawBase := make([]byte, hcRows*4)
	for i := range hcRows {
		binary.LittleEndian.PutUint32(rawBase[4*i:], math.Float32bits(0.0))
	}
	base := cudaCUPTIVector(b, r, "hc-base", 0, hcRows, rawBase)

	rawNorm := make([]byte, hcNorm*4)
	for i := range hcNorm {
		binary.LittleEndian.PutUint32(rawNorm[4*i:], math.Float32bits(1.0))
	}
	norm := cudaCUPTIVector(b, r, "hc-norm", 0, hcNorm, rawNorm)

	postOut := cudaCUPTIUpload(b, r, makeNaNs(tokens*hcState))
	preOut := cudaCUPTIUpload(b, r, makeNaNs(tokens*8192))

	// Prime initial pre transform so ready flag is set for post.
	for t := 0; t < tokens; t++ {
		if err := hc[t].pre(state.Slice(t*hcState, (t+1)*hcState), fn, scale, base, norm, preOut.Slice(t*8192, (t+1)*8192), false); err != nil {
			b.Fatal(err)
		}
	}
	if err := syncStream(r.stream()); err != nil {
		b.Fatal(err)
	}

	launch := func() error {
		for t := 0; t < tokens; t++ {
			s := state.Slice(t*hcState, (t+1)*hcState)
			if err := hc[t].post(s, block.Slice(t*hcNorm, (t+1)*hcNorm), postOut.Slice(t*hcState, (t+1)*hcState)); err != nil {
				return err
			}
			if err := hc[t].pre(s, fn, scale, base, norm, preOut.Slice(t*8192, (t+1)*8192), false); err != nil {
				return err
			}
		}
		return nil
	}

	refPost := cudaCUPTIUpload(b, r, makeNaNs(tokens*hcState))
	refPre := cudaCUPTIUpload(b, r, makeNaNs(tokens*8192))
	for t := 0; t < tokens; t++ {
		s := state.Slice(t*hcState, (t+1)*hcState)
		if err := hc[t].post(s, block.Slice(t*hcNorm, (t+1)*hcNorm), refPost.Slice(t*hcState, (t+1)*hcState)); err != nil {
			b.Fatal(err)
		}
		if err := hc[t].pre(s, fn, scale, base, norm, refPre.Slice(t*8192, (t+1)*8192), false); err != nil {
			b.Fatal(err)
		}
	}
	if err := syncStream(r.stream()); err != nil {
		b.Fatal(err)
	}
	wantPost := make([]float32, refPost.Len())
	wantPre := make([]float32, refPre.Len())
	if err := refPost.CopyToHost(wantPost); err != nil {
		b.Fatal(err)
	}
	if err := refPre.CopyToHost(wantPre); err != nil {
		b.Fatal(err)
	}
	cudaCUPTIRun(b, r, tokens, launch, []cudamem.Slice[float32]{postOut, preOut}, [][]float32{wantPost, wantPre})
}
