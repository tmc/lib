//go:build linux

package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
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

const (
	defaultCUDAModelPath    = "/home/tmc/ds4-models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf"
	cudaPrefillBenchContext = 4096
)

var cudaPrefillBench struct {
	once    sync.Once
	runtime *cudaRuntime
	err     error
}

func cudaPrefillBenchRuntime(b *testing.B) *cudaRuntime {
	b.Helper()
	modelPath := os.Getenv("DS4_MODEL")
	if modelPath == "" {
		modelPath = defaultCUDAModelPath
	}
	if _, err := os.Stat(modelPath); err != nil {
		b.Skipf("model not accessible: %v", err)
	}
	cudaPrefillBench.once.Do(func() {
		context := cudaPrefillBenchContext
		if v := os.Getenv("DS4_BENCH_CONTEXT"); v != "" {
			n, err := strconv.Atoi(v)
			if err != nil {
				cudaPrefillBench.err = fmt.Errorf("DS4_BENCH_CONTEXT: %w", err)
				return
			}
			context = n
		}
		cudaPrefillBench.runtime, cudaPrefillBench.err = newCUDARuntime(modelPath, context)
	})
	if cudaPrefillBench.err != nil {
		b.Fatal(cudaPrefillBench.err)
	}
	return cudaPrefillBench.runtime
}

func cudaBenchTokens(n int) []int {
	tokens := make([]int, n)
	for i := range tokens {
		tokens[i] = (i*7919 + 13) % 129280
	}
	return tokens
}

func cudaBenchSizes(b *testing.B, key string, sweep []int) []int {
	b.Helper()
	v := os.Getenv(key)
	if v == "" {
		return sweep
	}
	var out []int
	for _, part := range strings.Split(v, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		n, err := strconv.Atoi(part)
		if err != nil {
			b.Fatalf("%s: invalid integer %q: %v", key, part, err)
		}
		out = append(out, n)
	}
	if len(out) == 0 {
		return sweep
	}
	return out
}

// BenchmarkCUDAPrefill evaluates prompt chunks from a fresh session using
// cuptibench to measure prefill launches, wall time, and GPU duration.
func BenchmarkCUDAPrefill(b *testing.B) {
	if !*ds4CUPTIBench {
		b.Skip("enable with -ds4-cuptibench under gpu-lock")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		b.Fatal("gpu-lock is required")
	}
	for _, chunk := range cudaBenchSizes(b, "DS4_BENCH_CHUNKS", []int{32, 64, 128, 256}) {
		b.Run(fmt.Sprintf("tokens=%d", chunk), func(b *testing.B) {
			r := cudaPrefillBenchRuntime(b)
			if chunk >= r.ctx {
				b.Skipf("chunk %d does not fit context %d", chunk, r.ctx)
			}
			tokens := cudaBenchTokens(chunk)
			if err := r.Reset(); err != nil {
				b.Fatal(err)
			}
			if err := r.EvalBatch(tokens, 0); err != nil {
				b.Fatal(err)
			}
			if err := r.Reset(); err != nil {
				b.Fatal(err)
			}
			metrics, err := cuptibench.Resolve("launches")
			if err != nil {
				b.Fatal(err)
			}
			plan, err := cuptibench.New(r.device, cuptibench.Config{
				Metrics:     metrics,
				Mode:        cuptibench.Whole,
				Operations:  1,
				CachePolicy: cuptirange.CachePreserve,
				Limits: cuptirange.Limits{
					MaxRanges:         64,
					MaxRangeTreeNodes: 128,
					MaxCounterBytes:   16 << 20,
					MaxPasses:         32,
				},
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
			var hostLaunches uint64
			work := cuptibench.Workload{
				View:        "full",
				Streams:     []*cuda.Stream{r.stream()},
				Preparation: "reset runtime state before prefill",
				Prepare:     func() error { return r.Reset() },
				Synchronize: func() error { return syncStream(r.stream()) },
				Run: func(s *cuptibench.Scope) error {
					before := cudalaunch.TotalLaunches()
					err := r.EvalBatch(tokens, 0)
					hostLaunches = cudalaunch.TotalLaunches() - before
					return err
				},
			}
			res := plan.Measure(b, work)
			b.ReportMetric(float64(chunk), "tokens/op")
			b.ReportMetric(float64(hostLaunches), "host-launches/op")
			if res != nil {
				counters := res
				if res.Aggregate != nil {
					counters = res.Aggregate
				}
				for _, rg := range counters.Ranges {
					for _, m := range rg.Metrics {
						if m.Name == "sm__ctas_launched.sum" {
							b.ReportMetric(m.Value, "launches/op")
						}
					}
				}
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
