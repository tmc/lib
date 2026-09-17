//go:build linux

package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cuptibench"
)

// BenchmarkCUDARMSNormQuant measures RMS normalization chained with Q8_1 and Q8_0
// quantization across sequence lengths (1, 16, 256).
func BenchmarkCUDARMSNormQuant(b *testing.B) {
	if !*ds4CUPTIBench {
		b.Skip("enable with -ds4-cuptibench under gpu-lock")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		b.Fatal("gpu-lock is required")
	}
	for _, tokens := range cudaBenchSizes(b, "DS4_BENCH_TOKENS", []int{1, 16, 256}) {
		b.Run(fmt.Sprintf("tokens=%d", tokens), func(b *testing.B) {
			b.Run("rms_quant_q81", func(b *testing.B) {
				cudaCUPTIRMSNormQuantQ81(b, tokens)
			})
			b.Run("rms_quant_q80", func(b *testing.B) {
				cudaCUPTIRMSNormQuantQ80(b, tokens)
			})
			b.Run("rms_only", func(b *testing.B) {
				cudaCUPTIRMSNormOnly(b, tokens)
			})
			b.Run("quant_q81_only", func(b *testing.B) {
				cudaCUPTIQuantQ81Only(b, tokens)
			})
			b.Run("quant_q80_only", func(b *testing.B) {
				cudaCUPTIQuantQ80Only(b, tokens)
			})
		})
	}
}

func cudaCUPTIRMSNormQuantQ81(b *testing.B, tokens int) {
	r := cudaQ80BenchFoundation(b)
	const cols = cudaWidth
	rowBytes := cols / 32 * 36

	hostX := make([]float32, tokens*cols)
	for i := range hostX {
		hostX[i] = float32((i*31+11)%251-125) / 64
	}
	x := cudaCUPTIUpload(b, r, hostX)

	hostWeight := make([]float32, cols)
	for i := range hostWeight {
		hostWeight[i] = float32((i*7+3)%127-63) / 128
	}
	rawWeight := make([]byte, cols*4)
	for i, v := range hostWeight {
		binary.LittleEndian.PutUint32(rawWeight[4*i:], math.Float32bits(v))
	}
	weightTensor := cudaCUPTIVector(b, r, "rms_weight", 0, cols, rawWeight)

	normOut := cudaCUPTIUpload(b, r, makeNaNs(tokens*cols))
	q81Out := cudaCUPTIUpload(b, r, make([]byte, tokens*rowBytes))

	launch := func() error {
		if err := r.encodeRMS(x, weightTensor, normOut, tokens, true); err != nil {
			return err
		}
		if tokens == 1 {
			return r.encodeQuantQ81(normOut, q81Out, cols)
		}
		return r.encodeQuantQ81Batch(normOut, q81Out, cols, cols, tokens)
	}

	// Warmup and verify valid output.
	if err := launch(); err != nil {
		b.Fatal(err)
	}
	if err := syncStream(r.stream()); err != nil {
		b.Fatal(err)
	}
	got := make([]byte, q81Out.Len())
	if err := q81Out.CopyToHost(got); err != nil {
		b.Fatal(err)
	}

	poisonNorm := makeNaNs(normOut.Len())
	poisonQ81 := make([]byte, q81Out.Len())
	for i := range poisonQ81 {
		poisonQ81[i] = 0xAA
	}
	prepare := func() error {
		if err := normOut.CopyFromHost(poisonNorm); err != nil {
			return err
		}
		return q81Out.CopyFromHost(poisonQ81)
	}

	cudaCUPTIRunWorkload(b, r, tokens, cuptibench.Workload{
		View:        "isolated",
		Streams:     []*cuda.Stream{r.stream()},
		Preparation: "synthetic device-resident immutable inputs; intermediates poisoned before each iteration",
		Prepare:     prepare,
		Synchronize: func() error { return syncStream(r.stream()) },
		Run:         func(*cuptibench.Scope) error { return launch() },
	})
}

func cudaCUPTIRMSNormQuantQ80(b *testing.B, tokens int) {
	r := cudaQ80BenchFoundation(b)
	const cols = cudaWidth
	rowBytes := cols / 32 * 36

	hostX := make([]float32, tokens*cols)
	for i := range hostX {
		hostX[i] = float32((i*31+11)%251-125) / 64
	}
	x := cudaCUPTIUpload(b, r, hostX)

	hostWeight := make([]float32, cols)
	for i := range hostWeight {
		hostWeight[i] = float32((i*7+3)%127-63) / 128
	}
	rawWeight := make([]byte, cols*4)
	for i, v := range hostWeight {
		binary.LittleEndian.PutUint32(rawWeight[4*i:], math.Float32bits(v))
	}
	weightTensor := cudaCUPTIVector(b, r, "rms_weight", 0, cols, rawWeight)

	normOut := cudaCUPTIUpload(b, r, makeNaNs(tokens*cols))
	q80Out := cudaCUPTIUpload(b, r, make([]byte, tokens*rowBytes))

	launch := func() error {
		if err := r.encodeRMS(x, weightTensor, normOut, tokens, true); err != nil {
			return err
		}
		return r.encodeQuantQ80(normOut, q80Out, cols)
	}

	// Warmup and verify valid output.
	if err := launch(); err != nil {
		b.Fatal(err)
	}
	if err := syncStream(r.stream()); err != nil {
		b.Fatal(err)
	}
	got := make([]byte, q80Out.Len())
	if err := q80Out.CopyToHost(got); err != nil {
		b.Fatal(err)
	}

	poisonNorm := makeNaNs(normOut.Len())
	poisonQ80 := make([]byte, q80Out.Len())
	for i := range poisonQ80 {
		poisonQ80[i] = 0xAA
	}
	prepare := func() error {
		if err := normOut.CopyFromHost(poisonNorm); err != nil {
			return err
		}
		return q80Out.CopyFromHost(poisonQ80)
	}

	cudaCUPTIRunWorkload(b, r, tokens, cuptibench.Workload{
		View:        "isolated",
		Streams:     []*cuda.Stream{r.stream()},
		Preparation: "synthetic device-resident immutable inputs; intermediates poisoned before each iteration",
		Prepare:     prepare,
		Synchronize: func() error { return syncStream(r.stream()) },
		Run:         func(*cuptibench.Scope) error { return launch() },
	})
}

func cudaCUPTIRMSNormOnly(b *testing.B, tokens int) {
	r := cudaQ80BenchFoundation(b)
	const cols = cudaWidth

	hostX := make([]float32, tokens*cols)
	for i := range hostX {
		hostX[i] = float32((i*31+11)%251-125) / 64
	}
	x := cudaCUPTIUpload(b, r, hostX)

	hostWeight := make([]float32, cols)
	for i := range hostWeight {
		hostWeight[i] = float32((i*7+3)%127-63) / 128
	}
	rawWeight := make([]byte, cols*4)
	for i, v := range hostWeight {
		binary.LittleEndian.PutUint32(rawWeight[4*i:], math.Float32bits(v))
	}
	weightTensor := cudaCUPTIVector(b, r, "rms_weight", 0, cols, rawWeight)

	out := cudaCUPTIUpload(b, r, makeNaNs(tokens*cols))

	launch := func() error {
		return r.encodeRMS(x, weightTensor, out, tokens, true)
	}

	// Warmup and check.
	if err := launch(); err != nil {
		b.Fatal(err)
	}
	if err := syncStream(r.stream()); err != nil {
		b.Fatal(err)
	}

	poison := makeNaNs(out.Len())
	prepare := func() error {
		return out.CopyFromHost(poison)
	}

	cudaCUPTIRunWorkload(b, r, tokens, cuptibench.Workload{
		View:        "isolated",
		Streams:     []*cuda.Stream{r.stream()},
		Preparation: "synthetic device-resident immutable inputs; output poisoned before each iteration",
		Prepare:     prepare,
		Synchronize: func() error { return syncStream(r.stream()) },
		Run:         func(*cuptibench.Scope) error { return launch() },
	})
}

func cudaCUPTIQuantQ81Only(b *testing.B, tokens int) {
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
		if tokens == 1 {
			return r.encodeQuantQ81(in, out, cols)
		}
		return r.encodeQuantQ81Batch(in, out, cols, cols, tokens)
	}

	want := cudaQ81ActivationReference(hostX)
	cudaCUPTIRunBytes(b, r, tokens, launch, []cudamem.Slice[byte]{out}, [][]byte{want})
}

func cudaCUPTIQuantQ80Only(b *testing.B, tokens int) {
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
		return r.encodeQuantQ80(in, out, cols)
	}

	want := cudaQ80ActivationReference(hostX)
	cudaCUPTIRunBytes(b, r, tokens, launch, []cudamem.Slice[byte]{out}, [][]byte{want})
}
