//go:build darwin && arm64

package main

import (
	"encoding/binary"
	"math"
	"testing"

	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// These benchmarks run the pinned Metal kernels on synthetic buffers at the DS4
// shapes, with no model file. That keeps them fast and, unlike a full run, keeps
// the captured buffers small enough for Apple's replay profiler: a capture of
// the real CLI maps the model as two multi-gigabyte buffers, and
// GTMTLReplayController crashes restoring them, so no per-shader timing is
// available from a full-model trace.
//
// Capture one with:
//
//	go test -c -o ds4-bench.test
//	gputrace capture -o bench.gputrace -- ./ds4-bench.test -test.bench=BenchmarkMetalRMS -test.benchtime=20x
//	gputrace profile-replay bench.gputrace

func benchOps(b *testing.B) (*libmetal.Device, *metalPinnedOps, *metalmem.Arena) {
	b.Helper()
	dev, err := libmetal.NewDevice()
	if err != nil {
		b.Skipf("Metal device: %v", err)
	}
	ops, err := newMetalPinnedOps(dev)
	if err != nil {
		dev.Close()
		b.Fatal(err)
	}
	arena := metalmem.NewArena(dev)
	b.Cleanup(func() {
		arena.Free()
		ops.Close()
		dev.Close()
	})
	return dev, ops, arena
}

func benchF32Tensor(b *testing.B, arena *metalmem.Arena, name string, n int) boundMetalTensor {
	b.Helper()
	data := metalmem.AllocIn[byte](arena, name, 4*n)
	for i := range n {
		binary.LittleEndian.PutUint32(data.HostSlice()[4*i:], math.Float32bits(1))
	}
	return boundMetalTensor{name: name, typeID: 0, dimensions: []uint64{uint64(n)}, data: data}
}

func benchF16Tensor(b *testing.B, arena *metalmem.Arena, name string, cols, rows int) boundMetalTensor {
	b.Helper()
	data := metalmem.AllocIn[byte](arena, name, 2*cols*rows)
	for i := range cols * rows {
		binary.LittleEndian.PutUint16(data.HostSlice()[2*i:], 0x3c00) // 1.0 in F16.
	}
	return boundMetalTensor{name: name, typeID: 1, dimensions: []uint64{uint64(cols), uint64(rows)}, data: data}
}

func benchRamp(n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(i%17)/8 - 1
	}
	return x
}

// BenchmarkMetalSubmitOverhead measures one whole ops round trip on the smallest
// possible payload: allocate, upload, submit a command buffer, block on it, copy
// back, free. The arithmetic is four elements, so what it reports is the fixed
// cost every ops call pays. The full CLI issues 3,638 command buffers where the
// C oracle issues 8, so this is the number that governs that gap.
func BenchmarkMetalSubmitOverhead(b *testing.B) {
	_, ops, _ := benchOps(b)
	x := []float32{1, -2, 3, -4}
	b.ResetTimer()
	for b.Loop() {
		if _, err := ops.Clamp10(x); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMetalRMS(b *testing.B) {
	for _, n := range []int{ds4Width, ds4HCWidth} {
		b.Run(map[int]string{ds4Width: "w4096", ds4HCWidth: "w16384"}[n], func(b *testing.B) {
			_, ops, arena := benchOps(b)
			weight := benchF32Tensor(b, arena, "norm", n)
			if err := arena.Err(); err != nil {
				b.Fatal(err)
			}
			x := benchRamp(n)
			b.ResetTimer()
			for b.Loop() {
				if _, err := ops.RMS(x, weight, 1, true); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkMetalRopeAttentionOne covers the kernel the f16 Q-rounding fix landed
// in. It is one thread per head over a scalar 512-element loop, against the
// oracle's simdgroup half4 with simd_sum.
func BenchmarkMetalRopeAttentionOne(b *testing.B) {
	_, ops, arena := benchOps(b)
	sinks := benchF32Tensor(b, arena, "sinks", 64)
	if err := arena.Err(); err != nil {
		b.Fatal(err)
	}
	q, kv := benchRamp(64*ds4KVWidth), benchRamp(ds4KVWidth)
	b.ResetTimer()
	for b.Loop() {
		if _, _, _, err := ops.RopeAttentionOne(q, kv, sinks, 0); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkMetalHCPre covers hc_parallel_mix, the fused 1024-virtual-thread RMS
// plus eight-simdgroup F16 projection ported from the oracle.
func BenchmarkMetalHCPre(b *testing.B) {
	_, ops, arena := benchOps(b)
	fn := benchF16Tensor(b, arena, "hc-fn", ds4HCWidth, 24)
	scale := benchF32Tensor(b, arena, "hc-scale", 3)
	base := benchF32Tensor(b, arena, "hc-base", 24)
	norm := benchF32Tensor(b, arena, "attn-norm", ds4Width)
	if err := arena.Err(); err != nil {
		b.Fatal(err)
	}
	state := benchRamp(ds4HCWidth)
	b.ResetTimer()
	for b.Loop() {
		if _, _, err := ops.HCPre(state, fn, scale, base, norm); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMetalMatvecF16(b *testing.B) {
	_, ops, arena := benchOps(b)
	w := benchF16Tensor(b, arena, "kv", ds4Width, ds4KVWidth)
	if err := arena.Err(); err != nil {
		b.Fatal(err)
	}
	x := benchRamp(ds4Width)
	b.ResetTimer()
	for b.Loop() {
		if _, err := ops.MatvecF16(w, x); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMetalSwiGLU(b *testing.B) {
	_, ops, _ := benchOps(b)
	gate, up := benchRamp(2048), benchRamp(2048)
	b.ResetTimer()
	for b.Loop() {
		if _, err := ops.SwiGLU(gate, up); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMetalSumVectors(b *testing.B) {
	_, ops, _ := benchOps(b)
	v := benchRamp(8 * ds4Width)
	b.ResetTimer()
	for b.Loop() {
		if _, err := ops.SumVectors(v, 8); err != nil {
			b.Fatal(err)
		}
	}
}
