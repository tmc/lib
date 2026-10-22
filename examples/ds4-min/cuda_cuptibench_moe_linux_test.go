//go:build linux

package main

import (
	"fmt"
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cuptibench"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// BenchmarkCUDAMoEGather measures isolated MoE gather kernels across tile sizes
// (16, 32) and token counts (16, 64, 256) for Q2_K and IQ2_XXS.
func BenchmarkCUDAMoEGather(b *testing.B) {
	if !*ds4CUPTIBench {
		b.Skip("enable with -ds4-cuptibench under gpu-lock")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		b.Fatal("gpu-lock is required")
	}
	for _, tile := range cudaBenchSizes(b, "DS4_BENCH_MOE_TILES", []int{4, 8, 16, 32}) {
		for _, tokens := range cudaBenchSizes(b, "DS4_BENCH_TOKENS", []int{16, 64, 256}) {
			b.Run(fmt.Sprintf("q2_k/tile=%d/tokens=%d", tile, tokens), func(b *testing.B) {
				cudaCUPTIMoEGatherQ2K(b, tile, tokens)
			})
			b.Run(fmt.Sprintf("iq2_xxs/tile=%d/tokens=%d", tile, tokens), func(b *testing.B) {
				cudaCUPTIMoEGatherIQ2(b, tile, tokens)
			})
		}
	}
}

func cudaMoEGatherFunction(b *testing.B, r *cudaFoundation, tile int, symbol string) driver.Function {
	b.Helper()
	if symbol == "ds4_q2_k_selected_q8_1_gather" && tile == cudaMoEQ2GatherTile {
		if err := r.device.WithContext(r.loadMathKernels); err != nil {
			b.Fatal(err)
		}
		return r.q2KGatherQ81Fn
	}
	if symbol == "ds4_iq2_xxs_pair_selected_q8_1_gather" && tile == cudaMoEIQ2GatherTile {
		if err := r.device.WithContext(r.loadMathKernels); err != nil {
			b.Fatal(err)
		}
		return r.iq2GatherQ81Fn
	}
	program, err := nvrtcx.Compile(cudaMathSource,
		nvrtcx.WithName(fmt.Sprintf("ds4_gather_tile%d.cu", tile)),
		nvrtcx.WithFlag(fmt.Sprintf("--gpu-architecture=sm_%d%d", r.major, r.minor)),
		nvrtcx.WithFlag(fmt.Sprintf("-DDS4_MOE_IQ2_GATHER_TILE=%d", tile)),
		nvrtcx.WithFlag(fmt.Sprintf("-DDS4_MOE_Q2_GATHER_TILE=%d", tile)),
		nvrtcx.WithFlag(fmt.Sprintf("-DDS4_MOE_GATHER_TILE=%d", tile)),
		nvrtcx.WithFlag("--generate-line-info"),
		nvrtcx.WithFlag("--extra-device-vectorization"),
	)
	if err != nil {
		b.Fatal(err)
	}
	module, err := program.Load(r.device)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() {
		if err := module.Close(); err != nil {
			b.Error(err)
		}
	})
	fn, err := module.Function(symbol)
	if err != nil {
		b.Fatal(err)
	}
	return fn
}

func cudaCUPTIMoEGatherQ2K(b *testing.B, tile, tokens int) {
	r := cudaQ80BenchFoundation(b)
	const (
		cols    = 2048
		rows    = 256
		experts = 256
		slots   = 6
	)
	pairs := tokens * slots
	fn := cudaMoEGatherFunction(b, r, tile, "ds4_q2_k_selected_q8_1_gather")

	ids := make([]int32, pairs)
	for i := range ids {
		ids[i] = int32((i*37 + 13) % experts)
	}
	g, err := newCUDAMoEGrouping(r.device, experts, pairs)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = g.Free() })

	dIDs := cudaCUPTIUpload(b, r, ids)
	if err := r.encodeMoEGroupCSR(dIDs, g); err != nil {
		b.Fatal(err)
	}

	downBytes := cudaExpertFixture(10, cols, rows, experts, 43)
	downTensor := cudaModelTensor{
		name:       "expert_down",
		typeID:     10,
		dimensions: []uint64{cols, rows, experts},
		device:     cudaCUPTIUpload(b, r, downBytes).Ptr(),
		bytes:      uint64(len(downBytes)),
	}

	hostX := make([]float32, pairs*cols)
	for i := range hostX {
		hostX[i] = float32((i*17+i/cols*31+7)%251-125) / 128
	}
	q81X := cudaQ81ActivationReference(hostX)
	in := cudaCUPTIUpload(b, r, q81X)
	out := cudaCUPTIUpload(b, r, makeNaNs(pairs*rows))

	x, y := g.gatherGrid(uint64(rows))
	launch := func() error {
		return r.device.WithContext(func() error {
			return cudalaunch.Kernel(fn).Grid(x, y, 1).Block(32*cudaExpertGatherWarps, 1, 1).On(r.stream()).Launch(
				downTensor.device, g.offsets, g.pairs, g.active, g.count, in, out,
				uint32(cols), uint32(rows), uint32(experts))
		})
	}

	// Warmup and check that kernel produces finite numbers.
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
		Preparation: "synthetic device-resident immutable weights and activations; poison output before each sequence",
		Prepare:     prepare,
		Synchronize: func() error { return syncStream(r.stream()) },
		Run:         func(*cuptibench.Scope) error { return launch() },
	})
}

func cudaCUPTIMoEGatherIQ2(b *testing.B, tile, tokens int) {
	r := cudaQ80BenchFoundation(b)
	const (
		cols    = 2048
		rows    = 256
		experts = 256
		slots   = 6
	)
	pairs := tokens * slots
	fn := cudaMoEGatherFunction(b, r, tile, "ds4_iq2_xxs_pair_selected_q8_1_gather")

	ids := make([]int32, pairs)
	for i := range ids {
		ids[i] = int32((i*37 + 13) % experts)
	}
	g, err := newCUDAMoEGrouping(r.device, experts, pairs)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = g.Free() })

	dIDs := cudaCUPTIUpload(b, r, ids)
	if err := r.encodeMoEGroupCSR(dIDs, g); err != nil {
		b.Fatal(err)
	}

	gateBytes := cudaExpertFixture(16, cols, rows, experts, 17)
	upBytes := cudaExpertFixture(16, cols, rows, experts, 29)
	gateTensor := cudaModelTensor{
		name:       "expert_gate",
		typeID:     16,
		dimensions: []uint64{cols, rows, experts},
		device:     cudaCUPTIUpload(b, r, gateBytes).Ptr(),
		bytes:      uint64(len(gateBytes)),
	}
	upTensor := cudaModelTensor{
		name:       "expert_up",
		typeID:     16,
		dimensions: []uint64{cols, rows, experts},
		device:     cudaCUPTIUpload(b, r, upBytes).Ptr(),
		bytes:      uint64(len(upBytes)),
	}

	hostX := make([]float32, tokens*cols)
	for i := range hostX {
		hostX[i] = float32((i*37+i/17)%257-128) / 128
	}
	q81X := cudaQ81ActivationReference(hostX)
	in := cudaCUPTIUpload(b, r, q81X)
	gateOut := cudaCUPTIUpload(b, r, makeNaNs(pairs*rows))
	upOut := cudaCUPTIUpload(b, r, makeNaNs(pairs*rows))

	x, y := g.gatherGrid(uint64(rows))
	launch := func() error {
		return r.device.WithContext(func() error {
			return cudalaunch.Kernel(fn).Grid(x, y, 1).Block(32*cudaExpertGatherWarps, 1, 1).On(r.stream()).Launch(
				gateTensor.device, upTensor.device, g.offsets, g.pairs, g.active, g.count, in, gateOut, upOut,
				uint32(cols), uint32(rows), uint32(slots), uint32(experts))
		})
	}

	// Warmup and check.
	if err := launch(); err != nil {
		b.Fatal(err)
	}
	if err := syncStream(r.stream()); err != nil {
		b.Fatal(err)
	}

	poisonGate := makeNaNs(gateOut.Len())
	poisonUp := makeNaNs(upOut.Len())
	prepare := func() error {
		if err := gateOut.CopyFromHost(poisonGate); err != nil {
			return err
		}
		return upOut.CopyFromHost(poisonUp)
	}
	cudaCUPTIRunWorkload(b, r, tokens, cuptibench.Workload{
		View:        "isolated",
		Streams:     []*cuda.Stream{r.stream()},
		Preparation: "synthetic device-resident immutable weights and activations; poison output before each sequence",
		Prepare:     prepare,
		Synchronize: func() error { return syncStream(r.stream()) },
		Run:         func(*cuptibench.Scope) error { return launch() },
	})
}
