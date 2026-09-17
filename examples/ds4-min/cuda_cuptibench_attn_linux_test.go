//go:build linux

package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cuptibench"
)

// BenchmarkCUDAAttention measures isolated mixed attention score tiles and finalize
// across sequence lengths (16, 64, 256).
func BenchmarkCUDAAttention(b *testing.B) {
	if !*ds4CUPTIBench {
		b.Skip("enable with -ds4-cuptibench under gpu-lock")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		b.Fatal("gpu-lock is required")
	}
	for _, tokens := range cudaBenchSizes(b, "DS4_BENCH_TOKENS", []int{16, 64, 256}) {
		b.Run(fmt.Sprintf("tokens=%d", tokens), func(b *testing.B) {
			b.Run("scores_tile", func(b *testing.B) {
				cudaCUPTIAttentionScoresTile(b, tokens)
			})
			b.Run("finalize", func(b *testing.B) {
				cudaCUPTIAttentionFinalize(b, tokens)
			})
			b.Run("fused", func(b *testing.B) {
				cudaCUPTIAttentionFused(b, tokens)
			})
		})
	}
}

type cudaAttentionFixture struct {
	r            *cudaFoundation
	tokens       int
	position     int
	capacity     int
	nComp        int
	total        int
	tileGridX    uint32
	shmemFinal   uint32
	dr           cudamem.Slice[float32]
	dc           cudamem.Slice[float32]
	sink         cudaModelTensor
	allowed      cudamem.Slice[uint32]
	scores       cudamem.Slice[float32]
	denominators cudamem.Slice[float32]
	out          cudamem.Slice[float32]
}

func newCUDAAttentionFixture(b *testing.B, r *cudaFoundation, tokens int) *cudaAttentionFixture {
	b.Helper()
	if err := r.device.WithContext(r.loadMathKernels); err != nil {
		b.Fatal(err)
	}
	if r.mixedAttentionTileScoresFn == 0 || r.mixedAttentionFinalizeFn == 0 {
		b.Fatal("mixed attention tile or finalize function not loaded")
	}

	capacity := tokens
	nComp := 0
	total := capacity + nComp
	position := tokens - 1

	rawHost := make([]float32, capacity*512)
	for i := range rawHost {
		rawHost[i] = float32((i*13+i/512*7)%97-48) / 64
	}
	dr := cudaCUPTIUpload(b, r, rawHost)

	compHost := make([]float32, max(1, nComp)*512)
	dc := cudaCUPTIUpload(b, r, compHost)

	sinksHost := make([]float32, 64)
	for i := range sinksHost {
		sinksHost[i] = float32(i%7-3) / 4
	}
	rawSinks := make([]byte, 64*4)
	for i, v := range sinksHost {
		binary.LittleEndian.PutUint32(rawSinks[4*i:], math.Float32bits(v))
	}
	sink := cudaCUPTITensor(b, r, "sinks", 0, 64, 1, rawSinks)

	maskHost := make([]uint32, max(1, nComp))
	for i := range maskHost {
		maskHost[i] = 1
	}
	allowed := cudaCUPTIUpload(b, r, maskHost)

	scores := cudaCUPTIUpload(b, r, makeNaNs(64*total))
	denominators := cudaCUPTIUpload(b, r, makeNaNs(64))
	out := cudaCUPTIUpload(b, r, makeNaNs(32768+512+32768))

	tileGridX := uint32((total + 15) / 16)
	shmemFinal := uint32(total * 4)

	return &cudaAttentionFixture{
		r:            r,
		tokens:       tokens,
		position:     position,
		capacity:     capacity,
		nComp:        nComp,
		total:        total,
		tileGridX:    tileGridX,
		shmemFinal:   shmemFinal,
		dr:           dr,
		dc:           dc,
		sink:         sink,
		allowed:      allowed,
		scores:       scores,
		denominators: denominators,
		out:          out,
	}
}

func cudaCUPTIAttentionScoresTile(b *testing.B, tokens int) {
	r := cudaQ80BenchFoundation(b)
	f := newCUDAAttentionFixture(b, r, tokens)

	launch := func() error {
		return r.device.WithContext(func() error {
			return cudalaunch.Kernel(r.mixedAttentionTileScoresFn).Grid(f.tileGridX, 4).Block(256).SharedMem(66048).On(r.stream()).Launch(
				f.dr, f.dc, f.allowed, f.scores, f.out, uint32(f.position), uint32(f.capacity), uint32(f.nComp))
		})
	}

	// Warmup and check.
	if err := launch(); err != nil {
		b.Fatal(err)
	}
	if err := syncStream(r.stream()); err != nil {
		b.Fatal(err)
	}

	poison := makeNaNs(f.scores.Len())
	prepare := func() error {
		return f.scores.CopyFromHost(poison)
	}

	cudaCUPTIRunWorkload(b, r, tokens, cuptibench.Workload{
		View:        "isolated",
		Streams:     []*cuda.Stream{r.stream()},
		Preparation: "synthetic device-resident immutable KV activations; scores poisoned before each iteration",
		Prepare:     prepare,
		Synchronize: func() error { return syncStream(r.stream()) },
		Run:         func(*cuptibench.Scope) error { return launch() },
	})
}

func cudaCUPTIAttentionFinalize(b *testing.B, tokens int) {
	r := cudaQ80BenchFoundation(b)
	f := newCUDAAttentionFixture(b, r, tokens)

	// Populate scores before benchmarking finalize.
	if err := r.device.WithContext(func() error {
		return cudalaunch.Kernel(r.mixedAttentionTileScoresFn).Grid(f.tileGridX, 4).Block(256).SharedMem(66048).On(r.stream()).Launch(
			f.dr, f.dc, f.allowed, f.scores, f.out, uint32(f.position), uint32(f.capacity), uint32(f.nComp))
	}); err != nil {
		b.Fatal(err)
	}
	if err := syncStream(r.stream()); err != nil {
		b.Fatal(err)
	}

	launch := func() error {
		return r.device.WithContext(func() error {
			return cudalaunch.Kernel(r.mixedAttentionFinalizeFn).Grid(64).Block(512).SharedMem(f.shmemFinal).On(r.stream()).Launch(
				f.sink.device, f.dr, f.dc, f.allowed, f.scores, f.denominators, f.out, uint32(f.position), uint32(f.capacity), uint32(f.nComp))
		})
	}

	// Warmup and check.
	if err := launch(); err != nil {
		b.Fatal(err)
	}
	if err := syncStream(r.stream()); err != nil {
		b.Fatal(err)
	}

	poisonOut := makeNaNs(f.out.Len())
	poisonDenom := makeNaNs(f.denominators.Len())
	prepare := func() error {
		if err := f.out.CopyFromHost(poisonOut); err != nil {
			return err
		}
		return f.denominators.CopyFromHost(poisonDenom)
	}

	cudaCUPTIRunWorkload(b, r, tokens, cuptibench.Workload{
		View:        "isolated",
		Streams:     []*cuda.Stream{r.stream()},
		Preparation: "synthetic device-resident immutable activations; out and denominators poisoned before each iteration",
		Prepare:     prepare,
		Synchronize: func() error { return syncStream(r.stream()) },
		Run:         func(*cuptibench.Scope) error { return launch() },
	})
}

func cudaCUPTIAttentionFused(b *testing.B, tokens int) {
	r := cudaQ80BenchFoundation(b)
	f := newCUDAAttentionFixture(b, r, tokens)

	launch := func() error {
		return r.device.WithContext(func() error {
			if err := cudalaunch.Kernel(r.mixedAttentionTileScoresFn).Grid(f.tileGridX, 4).Block(256).SharedMem(66048).On(r.stream()).Launch(
				f.dr, f.dc, f.allowed, f.scores, f.out, uint32(f.position), uint32(f.capacity), uint32(f.nComp)); err != nil {
				return err
			}
			return cudalaunch.Kernel(r.mixedAttentionFinalizeFn).Grid(64).Block(512).SharedMem(f.shmemFinal).On(r.stream()).Launch(
				f.sink.device, f.dr, f.dc, f.allowed, f.scores, f.denominators, f.out, uint32(f.position), uint32(f.capacity), uint32(f.nComp))
		})
	}

	// Warmup and check.
	if err := launch(); err != nil {
		b.Fatal(err)
	}
	if err := syncStream(r.stream()); err != nil {
		b.Fatal(err)
	}

	poisonOut := makeNaNs(f.out.Len())
	poisonScores := makeNaNs(f.scores.Len())
	prepare := func() error {
		if err := f.out.CopyFromHost(poisonOut); err != nil {
			return err
		}
		return f.scores.CopyFromHost(poisonScores)
	}

	cudaCUPTIRunWorkload(b, r, tokens, cuptibench.Workload{
		View:        "isolated",
		Streams:     []*cuda.Stream{r.stream()},
		Preparation: "synthetic device-resident immutable activations; scores and out poisoned before each iteration",
		Prepare:     prepare,
		Synchronize: func() error { return syncStream(r.stream()) },
		Run:         func(*cuptibench.Scope) error { return launch() },
	})
}
