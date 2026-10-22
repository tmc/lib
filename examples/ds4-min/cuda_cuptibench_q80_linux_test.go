//go:build linux

package main

import (
	"fmt"
	"math/rand"
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamem"
)

// BenchmarkCUDAKernels measures isolated kernels, not a forward pass. Every
// leaf owns its fixture and plan, including when selected without its siblings.
func BenchmarkCUDAKernels(b *testing.B) {
	if !*ds4CUPTIBench {
		b.Skip("enable with -ds4-cuptibench under gpu-lock")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		b.Fatal("gpu-lock is required")
	}
	for _, phase := range []struct {
		name  string
		batch int
	}{
		{"decode", 1},
		{"prefill", 8},
	} {
		b.Run(phase.name, func(b *testing.B) {
			for _, shape := range cudaQ80BenchShapes {
				// Prefill emits the output head only for the terminal token.
				if phase.batch > 1 && shape.name == "output.weight" {
					continue
				}
				b.Run(fmt.Sprintf("q80/%s/%s/tokens=%d", shape.kernel, shape.name, phase.batch), func(b *testing.B) {
					cudaCUPTIQ80(b, shape, phase.batch)
				})
			}
			for _, rows := range []int{256, 512, 1024} {
				b.Run(fmt.Sprintf("f16_pair/rows=%d/tokens=%d", rows, phase.batch), func(b *testing.B) {
					cudaCUPTIF16(b, rows, phase.batch)
				})
			}
		})
	}
}

// BenchmarkCUDAQ80BatchWidth compares batch tiles on the attention projections.
// Each timed launch must match independent per-token launches bit for bit.

// BenchmarkCUDAF16PairBatchWidth compares inner tile batches across row dimensions and token counts.
func BenchmarkCUDAF16PairBatchWidth(b *testing.B) {
	if !*ds4CUPTIBench {
		b.Skip("enable with -ds4-cuptibench under gpu-lock")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		b.Fatal("gpu-lock is required")
	}
	for _, rows := range []int{256, 512, 1024} {
		for _, batch := range []int{8, 16, 32, 64, 256} {
			b.Run(fmt.Sprintf("rows=%d/tokens=%d", rows, batch), func(b *testing.B) {
				cudaCUPTIF16(b, rows, batch)
			})
		}
	}
}

func BenchmarkCUDAQ80BatchWidth(b *testing.B) {
	if !*ds4CUPTIBench {
		b.Skip("enable with -ds4-cuptibench under gpu-lock")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		b.Fatal("gpu-lock is required")
	}
	// The attention output has eight diagonal groups, each reading 4096
	// head values and writing 1024 low-rank values (evalBatchAttnOut).
	// Projection B maps the resulting 8192 values to cudaWidth (4096).
	for _, shape := range []cudaQ80BenchShape{
		{"attn_q_b", "q8_1", 1024, 32768, 0, 43},
		{"attn_output_b", "quantized", 8192, 4096, 0, 43},
		{"attn_output_a", "blockdiag", 4096, 8192, 8, 43},
	} {
		for _, batch := range []int{8, 16, 32, 256} {
			b.Run(fmt.Sprintf("%s/%s/tokens=%d", shape.kernel, shape.name, batch), func(b *testing.B) {
				cudaCUPTIQ80(b, shape, batch)
			})
		}
	}
}

func cudaCUPTIQ80(b *testing.B, s cudaQ80BenchShape, batch int) {
	r := cudaQ80BenchFoundation(b)
	weights := cudaSynthQ80(rand.New(rand.NewSource(1)), s.rows, s.cols, -8)
	w := cudaCUPTITensor(b, r, s.name, 8, s.cols, s.rows, weights)
	x := make([]float32, batch*s.activationBytes()/36*32)
	for i := range x {
		x[i] = float32((i*37+i/17)%257-128) / 128
	}
	var raw []byte
	if s.kernel == "q8_1" {
		raw = cudaQ81ActivationReference(x)
	} else {
		raw = cudaQ80ActivationReference(x)
	}
	in := cudaCUPTIUpload(b, r, raw)
	out := cudaCUPTIUpload(b, r, makeNaNs(batch*s.rows))
	launch := func() error {
		switch s.kernel {
		case "q8_1":
			return r.encodeQ80MatvecQ81Batch(w, in, out, batch)
		case "quantized":
			return r.encodeQ80MatvecQuantizedBatch(w, in, out, batch)
		case "blockdiag":
			return r.encodeQ80MatvecQuantizedBlockDiagBatch(w, in, out, s.diag, batch)
		}
		return fmt.Errorf("unknown Q8_0 kernel %q", s.kernel)
	}
	if batch == 1 {
		launch = cudaQ80BenchLaunch(b, r, s, w, in, out)
	}
	// Independent scalar launches establish the prefill result before timing.
	ref := cudaCUPTIUpload(b, r, makeNaNs(batch*s.rows))
	for token := range batch {
		call := cudaQ80BenchLaunch(b, r, s, w,
			in.Slice(token*s.activationBytes(), (token+1)*s.activationBytes()),
			ref.Slice(token*s.rows, (token+1)*s.rows))
		if err := call(); err != nil {
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
	cudaCUPTIRun(b, r, batch, launch, []cudamem.Slice[float32]{out}, [][]float32{want})
	// Report the bytes the launch must read so a shape can be divided into
	// gpu-ns/op to get its bandwidth. b.SetBytes is deliberately not used:
	// its MB/s column divides by wall-clock time, which carries host launch
	// and profiling overhead that the device span does not.
	b.ReportMetric(float64(s.weightBytes()+batch*s.activationBytes()), "bytes/op")
}

func cudaCUPTIF16(b *testing.B, rows, batch int) {
	r := cudaQ80BenchFoundation(b)
	const cols = 4096
	a, c, x := cudaF16PairBatchFixture(cols, rows, batch)
	wa := cudaCUPTITensor(b, r, "pair-a", 1, cols, rows, a)
	wb := cudaCUPTITensor(b, r, "pair-b", 1, cols, rows, c)
	in := cudaCUPTIUpload(b, r, x)
	out := []cudamem.Slice[float32]{cudaCUPTIUpload(b, r, makeNaNs(batch*rows)), cudaCUPTIUpload(b, r, makeNaNs(batch*rows))}
	ref := []cudamem.Slice[float32]{cudaCUPTIUpload(b, r, makeNaNs(batch*rows)), cudaCUPTIUpload(b, r, makeNaNs(batch*rows))}
	for token := range batch {
		if err := r.encodeF16PairOrdered(wa, wb, in.Slice(token*cols, (token+1)*cols),
			ref[0].Slice(token*rows, (token+1)*rows), ref[1].Slice(token*rows, (token+1)*rows)); err != nil {
			b.Fatal(err)
		}
	}
	if err := syncStream(r.stream()); err != nil {
		b.Fatal(err)
	}
	want := make([][]float32, 2)
	for i := range ref {
		want[i] = make([]float32, ref[i].Len())
		if err := ref[i].CopyToHost(want[i]); err != nil {
			b.Fatal(err)
		}
	}
	launch := func() error {
		if batch == 1 {
			return r.encodeF16PairOrdered(wa, wb, in, out[0], out[1])
		}
		return r.encodeF16PairOrderedBatch(wa, wb, in, out[0], out[1], batch)
	}
	cudaCUPTIRun(b, r, batch, launch, out, want)
}
