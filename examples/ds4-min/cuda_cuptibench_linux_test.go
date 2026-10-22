//go:build linux

package main

import (
	"bytes"
	"flag"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"sync"
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
	// A Go test binary leaves through runtime.exit, which issues exit_group
	// directly and runs neither atexit handlers nor ELF destructors. The CUPTI
	// agent finalizes its journal from exactly those two paths and from a
	// SIGTERM terminator that the Go runtime preempts, so under cudaprof
	// capture every record still buffered in the CUPTI pool dies there and the
	// journal ends after its two init records -- a capture that collects
	// nothing and says so only as an unfinalized journal. Leaving through libc
	// exit runs the destructor and finalizes. Gated, because the detour skips
	// Go's own exit path and is wanted only under capture.
	if os.Getenv("DS4_CAPTURE_LIBC_EXIT") == "1" {
		exitProcess(code)
	}
	os.Exit(code)
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

func cudaCUPTIVector(b *testing.B, r *cudaFoundation, name string, kind uint32, length int, host []byte) cudaModelTensor {
	b.Helper()
	d := cudaCUPTIUpload(b, r, host)
	return cudaModelTensor{name: name, typeID: kind, dimensions: []uint64{uint64(length)}, device: d.Ptr(), bytes: uint64(len(host))}
}

func cudaCUPTIRunWorkload(b *testing.B, r *cudaFoundation, tokens int, work cuptibench.Workload) {
	b.Helper()
	metrics, err := cuptibench.Resolve("launches")
	if err != nil {
		b.Fatal(err)
	}
	plan, err := cuptibench.New(r.device, cuptibench.Config{
		Metrics:        metrics,
		Mode:           cuptibench.Kernels,
		Operations:     1,
		CachePolicy:    cuptirange.CachePreserve,
		Limits:         cuptirange.Limits{MaxRanges: 4096, MaxRangeTreeNodes: 8192, MaxCounterBytes: 64 << 20, MaxPasses: 32},
		MaxConfigBytes: 8 << 20,
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
	cuptibench.Benchmark(b, plan, work)
	b.ReportMetric(float64(tokens), "tokens/op")
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
			if len(want) > i && want[i] != nil {
				if err := exactFloat32Bits(got, want[i]); err != nil {
					b.Fatalf("output %d: %v", i, err)
				}
			}
		}
	}
	// Load modules and fault in fixture pages before creating the profiling plan.
	if err := launch(); err != nil {
		b.Fatal(err)
	}
	check()
	cudaCUPTIRunWorkload(b, r, tokens, cuptibench.Workload{
		View:        "isolated",
		Streams:     []*cuda.Stream{r.stream()},
		Preparation: "synthetic device-resident immutable weights and activations; poison output before each sequence; modules warmed; no cache flush; no model or KV state",
		Prepare:     prepare,
		Synchronize: func() error { return syncStream(r.stream()) },
		Run:         func(*cuptibench.Scope) error { return launch() },
	})
	check()
}

func cudaCUPTIRunBytes(b *testing.B, r *cudaFoundation, tokens int, launch func() error, out []cudamem.Slice[byte], want [][]byte) {
	b.Helper()
	poison := make([][]byte, len(out))
	for i, d := range out {
		poison[i] = make([]byte, d.Len())
		for j := range poison[i] {
			poison[i][j] = 0xAA
		}
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
			got := make([]byte, d.Len())
			if err := d.CopyToHost(got); err != nil {
				b.Fatal(err)
			}
			if len(want) > i && want[i] != nil {
				if !bytes.Equal(got, want[i]) {
					b.Fatalf("output %d: byte mismatch", i)
				}
			}
		}
	}
	if err := launch(); err != nil {
		b.Fatal(err)
	}
	check()
	cudaCUPTIRunWorkload(b, r, tokens, cuptibench.Workload{
		View:        "isolated",
		Streams:     []*cuda.Stream{r.stream()},
		Preparation: "synthetic device-resident immutable weights and activations; poison output before each sequence; modules warmed; no cache flush; no model or KV state",
		Prepare:     prepare,
		Synchronize: func() error { return syncStream(r.stream()) },
		Run:         func(*cuptibench.Scope) error { return launch() },
	})
	check()
}
