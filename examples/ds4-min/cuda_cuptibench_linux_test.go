//go:build linux

package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cuptibench"
	"github.com/tmc/lib/nvidia/x/cuptirange"
)

var ds4CUPTIBench = flag.Bool("ds4-cuptibench", false, "enable isolated synthetic DS4 CUDA kernel benchmarks (requires gpu-lock)")

func TestMain(m *testing.M) {
	cuptibench.Init()
	code := m.Run()
	// A failed output check must not publish an otherwise valid capture.
	if code == 0 {
		if err := cuptibench.Finish(); err != nil {
			fmt.Fprintln(os.Stderr, err)
			code = 1
		}
	}
	os.Exit(code)
}

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

func cudaCUPTIUpload[T any](b *testing.B, r *cudaFoundation, host []T) cudamem.Slice[T] {
	b.Helper()
	d, err := cudamem.Alloc[T](r.device, len(host))
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() {
		if err := d.Free(); err != nil {
			b.Error(err)
		}
	})
	if err := d.CopyFromHost(host); err != nil {
		b.Fatal(err)
	}
	return d
}

func cudaCUPTITensor(b *testing.B, r *cudaFoundation, name string, kind uint32, cols, rows int, host []byte) cudaModelTensor {
	b.Helper()
	d := cudaCUPTIUpload(b, r, host)
	return cudaModelTensor{name: name, typeID: kind, dimensions: []uint64{uint64(cols), uint64(rows)}, device: d.Ptr(), bytes: uint64(len(host))}
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

func cudaCUPTIRun(b *testing.B, r *cudaFoundation, tokens int, launch func() error, out []cudamem.Slice[float32], want [][]float32) {
	b.Helper()
	poison := make([][]float32, len(out))
	for i, d := range out {
		poison[i] = makeNaNs(d.Len())
	}
	prepare := func() error {
		for i, d := range out {
			if err := d.CopyFromHost(poison[i]); err != nil {
				return err
			}
		}
		return nil
	}
	check := func() {
		b.Helper()
		if err := syncStream(r.stream()); err != nil {
			b.Fatal(err)
		}
		for i, d := range out {
			got := make([]float32, d.Len())
			if err := d.CopyToHost(got); err != nil {
				b.Fatal(err)
			}
			for j, v := range got {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					b.Fatalf("output %d element %d is nonfinite", i, j)
				}
			}
			if err := exactFloat32Bits(got, want[i]); err != nil {
				b.Fatalf("output %d: %v", i, err)
			}
		}
	}
	// Load modules and fault in fixture pages before creating the profiling plan.
	if err := launch(); err != nil {
		b.Fatal(err)
	}
	check()
	metrics, err := cuptibench.Resolve("launches")
	if err != nil {
		b.Fatal(err)
	}
	plan, err := cuptibench.New(r.device, cuptibench.Config{
		Metrics: metrics, Mode: cuptibench.Kernels, Operations: 1,
		CachePolicy:    cuptirange.CachePreserve,
		Limits:         cuptirange.Limits{MaxRanges: 16, MaxRangeTreeNodes: 32, MaxCounterBytes: 16 << 20, MaxPasses: 32},
		MaxConfigBytes: 4 << 20,
	})
	if plan != nil {
		b.Cleanup(func() {
			if err := plan.Close(); err != nil {
				b.Error(err)
			}
		})
	}
	if err != nil {
		b.Fatal(err)
	}
	cuptibench.Benchmark(b, plan, cuptibench.Workload{
		View: "isolated", Streams: []*cuda.Stream{r.stream()},
		Preparation: "synthetic device-resident immutable weights and activations; poison output before each sequence; modules warmed; no cache flush; no model or KV state",
		Prepare:     prepare, Synchronize: func() error { return syncStream(r.stream()) },
		Run: func(*cuptibench.Scope) error { return launch() },
	})
	check()
	b.ReportMetric(float64(tokens), "tokens/op")
}
