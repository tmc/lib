//go:build linux

package main

import (
	"context"
	"fmt"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func TestCUDAAttentionFusedBits(t *testing.T) {
	r := cudaSynthFoundation(t)
	if err := r.device.WithContext(r.loadMathKernels); err != nil {
		t.Fatal(err)
	}
	if r.mixedAttentionTileScoresFn == 0 || r.mixedAttentionFinalizeFn == 0 {
		t.Fatal("mixedAttentionTileScoresFn or mixedAttentionFinalizeFn is nil")
	}

	start := strings.Index(cudaMathSource, `extern "C" __global__ void ds4_attention_mixed(`)
	if start < 0 {
		t.Fatal("reference kernel boundary missing")
	}
	end := strings.Index(cudaMathSource[start:], "\n}")
	if end < 0 {
		t.Fatal("reference kernel end missing")
	}
	referenceSource := cudaMathSource[:start] + cudaAttentionValueReference + cudaMathSource[start+end+2:]
	reference := cudaAttentionKernel(t, r, referenceSource, "attention_fused_reference")

	type attentionFns struct {
		tile  driver.Function
		final driver.Function
	}

	compile := func(source, name string) attentionFns {
		t.Helper()
		program, err := nvrtcx.Compile(source, nvrtcx.WithName(name+".cu"), nvrtcx.WithArch(r.major, r.minor))
		if err != nil {
			t.Fatal(err)
		}
		module, err := program.Load(r.device)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := module.Unload(); err != nil {
				t.Error(err)
			}
		})
		tile, err := module.GetFunction("ds4_attention_mixed_scores_tile")
		if err != nil {
			t.Fatal(err)
		}
		_ = tile.SetAttribute(driver.FuncAttributeMaxDynamicSharedSizeBytes, 66048)
		final, err := module.GetFunction("ds4_attention_mixed_finalize")
		if err != nil {
			t.Fatal(err)
		}
		_ = final.SetAttribute(driver.FuncAttributeMaxDynamicSharedSizeBytes, 66048)
		return attentionFns{tile: tile, final: final}
	}

	mutants := []struct {
		name, old, new string
		fns            attentionFns
	}{
		{
			name: "score-scale",
			old:  "s = dot * 0.04419417382415922f;",
			new:  "s = dot * 0.04419417382415922f * 1.00001f;",
		},
		{
			name: "denominator",
			old:  "float denom = expf(sinks[h] - mx);",
			new:  "float denom = expf(sinks[h] - mx) * 1.001f;",
		},
		{
			name: "rotation",
			old:  "float theta = ds4_compressed_theta(position, r), c = cosf(theta), s = -sinf(theta);",
			new:  "float theta = ds4_compressed_theta(position, r), c = cosf(theta), s = -sinf(theta) + 1e-6f;",
		},
	}
	fusedStart := strings.Index(cudaMathSource, `extern "C" __global__ void ds4_attention_mixed_scores_tile(`)
	if fusedStart < 0 {
		t.Fatal("tiled kernel boundary missing")
	}
	prefix := cudaMathSource[:fusedStart]
	fusedBody := cudaMathSource[fusedStart:]

	for i := range mutants {
		m := &mutants[i]
		if !strings.Contains(fusedBody, m.old) {
			t.Fatalf("mutant target %q missing from source", m.old)
		}
		m.fns = compile(prefix+strings.Replace(fusedBody, m.old, m.new, 1), "attention_mutant_"+m.name)
	}

	const capacity = 128
	const maxComp = 513
	raw := make([]float32, capacity*512)
	for i := range raw {
		raw[i] = float32((i*13+i/512*7)%97-48) / 64
	}
	comp := make([]float32, maxComp*512)
	for i := range comp {
		comp[i] = float32((i*7+i/512*11)%89-44) / 64
	}
	sinks := make([]float32, 64)
	for i := range sinks {
		sinks[i] = float32(i%7-3) / 4
	}
	sink := cudaCompressedTensor(t, r, "sinks", sinks)
	dq, dk := cudaTestAlloc(t, r, 64*512), cudaTestAlloc(t, r, 512)
	dr, dc := cudaTestAlloc(t, r, len(raw)), cudaTestAlloc(t, r, len(comp))
	scores := cudaTestAlloc(t, r, 64*(capacity+maxComp))
	denominators := cudaTestAlloc(t, r, 64)
	out := cudaTestAlloc(t, r, 32768+512+32768)
	allowed := cudaCompressedMask(t, r, maxComp)
	cudaCompressedUpload(t, dc, comp)

	finite := func(t *testing.T, name string, values []float32) {
		t.Helper()
		for i, v := range values {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("%s[%d] non-finite", name, i)
			}
		}
	}

	run := func(t *testing.T, fns attentionFns, position, n int, maskAll bool, old bool) ([]float32, []float32) {
		t.Helper()
		q, kv := cudaAttentionQualificationInputs(position)
		for i := range q {
			q[i] += float32((i*17+i/512*5)%31-15) / 128
		}
		cudaCompressedUpload(t, dq, q)
		cudaCompressedUpload(t, dk, kv)
		cudaCompressedUpload(t, dr, raw)
		mask := make([]uint32, maxComp)
		for i := 0; i < n; i++ {
			if !maskAll && i%2 == 0 {
				mask[i] = 1
			}
		}
		if err := allowed.CopyFromHost(mask); err != nil {
			t.Fatal(err)
		}
		poison := make([]float32, out.Len())
		for i := range poison {
			poison[i] = float32(math.NaN())
		}
		cudaCompressedUpload(t, out, poison)
		dp := make([]float32, 64)
		for i := range dp {
			dp[i] = float32(math.NaN())
		}
		cudaCompressedUpload(t, denominators, dp)

		savedOld, savedTile, savedFinal := r.mixedAttentionFn, r.mixedAttentionTileScoresFn, r.mixedAttentionFinalizeFn
		defer func() {
			r.mixedAttentionFn, r.mixedAttentionTileScoresFn, r.mixedAttentionFinalizeFn = savedOld, savedTile, savedFinal
		}()
		if old {
			r.mixedAttentionFn = reference
		} else {
			if fns.tile != 0 {
				r.mixedAttentionTileScoresFn = fns.tile
			}
			if fns.final != 0 {
				r.mixedAttentionFinalizeFn = fns.final
			}
		}

		cudaQualificationMeasure(t, r, func() error {
			if old {
				return r.encodeMixedAttention(dq, dk, sink, dr, dc, n, allowed, scores, out, position, 2)
			}
			return r.encodeMixedAttentionParallel(dq, dk, sink, dr, dc, n, allowed, scores, denominators, out, position, 2)
		})
		got, cache := cudaTestRead(t, out), cudaTestRead(t, dr)
		finite(t, "output", got)
		finite(t, "raw cache", cache)
		if !old {
			d := cudaTestRead(t, denominators)
			finite(t, "denominators", d)
			for i, v := range d {
				if v <= 0 {
					t.Fatalf("denominator[%d]=%g", i, v)
				}
			}
		}
		return got, cache
	}

	defaultFns := attentionFns{tile: r.mixedAttentionTileScoresFn, final: r.mixedAttentionFinalizeFn}
	for _, position := range []int{127, 128, 129} {
		for _, tc := range []struct {
			n      int
			masked bool
		}{{1, false}, {512, false}, {513, false}, {0, false}, {513, true}} {
			t.Run(fmt.Sprintf("position%d/comp%d/masked%t", position, tc.n, tc.masked), func(t *testing.T) {
				want, wantRaw := run(t, defaultFns, position, tc.n, tc.masked, true)
				got, gotRaw := run(t, defaultFns, position, tc.n, tc.masked, false)
				if err := exactFloat32Bits(got, want); err != nil {
					t.Fatalf("output bits: %v", err)
				}
				if err := exactFloat32Bits(gotRaw, wantRaw); err != nil {
					t.Fatalf("raw cache bits: %v", err)
				}
			})
		}
	}

	want, wantRaw := run(t, defaultFns, 129, 513, false, true)
	for _, m := range mutants {
		t.Run("mutation/"+m.name, func(t *testing.T) {
			got, gotRaw := run(t, m.fns, 129, 513, false, false)
			if err := exactFloat32Bits(gotRaw, wantRaw); err != nil {
				t.Fatalf("mutation changed preparation/cache: %v", err)
			}
			if exactFloat32Bits(got, want) == nil {
				t.Fatal("candidate-only mutation survived exact output comparison")
			}
		})
	}
}

func TestCUDAAttentionProfile(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	r := cudaSynthFoundation(t)
	if err := r.device.WithContext(r.loadMathKernels); err != nil {
		t.Fatal(err)
	}
	const capacity = 128
	const maxComp = 513
	raw := make([]float32, capacity*512)
	comp := make([]float32, maxComp*512)
	sinks := make([]float32, 64)
	sink := cudaCompressedTensor(t, r, "sinks", sinks)
	dq, dk := cudaTestAlloc(t, r, 64*512), cudaTestAlloc(t, r, 512)
	dr, dc := cudaTestAlloc(t, r, len(raw)), cudaTestAlloc(t, r, len(comp))
	scores := cudaTestAlloc(t, r, 64*(capacity+maxComp))
	denominators := cudaTestAlloc(t, r, 64)
	out := cudaTestAlloc(t, r, 32768+512+32768)
	allowed := cudaCompressedMask(t, r, maxComp)

	mask := make([]uint32, maxComp)
	for i := range mask {
		mask[i] = 1
	}
	if err := allowed.CopyFromHost(mask); err != nil {
		t.Fatal(err)
	}

	measure := func(name string, fn func() error) time.Duration {
		t.Helper()
		stream := r.device.DefaultStream()
		for range 3 {
			if err := fn(); err != nil {
				t.Fatal(err)
			}
		}
		if err := stream.Synchronize(context.Background()); err != nil {
			t.Fatal(err)
		}
		start, _ := r.device.NewEvent()
		end, _ := r.device.NewEvent()
		defer start.Destroy()
		defer end.Destroy()

		const iters = 20
		if err := start.Record(stream); err != nil {
			t.Fatal(err)
		}
		for range iters {
			if err := fn(); err != nil {
				t.Fatal(err)
			}
		}
		if err := end.Record(stream); err != nil {
			t.Fatal(err)
		}
		if err := end.Synchronize(context.Background()); err != nil {
			t.Fatal(err)
		}
		dur, err := end.ElapsedSince(start)
		if err != nil {
			t.Fatal(err)
		}
		avg := dur / iters
		t.Logf("%-32s: %v", name, avg)
		return avg
	}

	position := 2048
	for _, nComp := range []int{16, 512} {
		t.Logf("=== nComp = %d (total = %d) ===", nComp, capacity+nComp)
		measure(fmt.Sprintf("nComp=%d/mixed_parallel_full", nComp), func() error {
			return r.encodeMixedAttentionParallel(dq, dk, sink, dr, dc, nComp, allowed, scores, denominators, out, position, 2)
		})
		measure(fmt.Sprintf("nComp=%d/prepare", nComp), func() error {
			return r.device.WithContext(func() error {
				return cudalaunch.Kernel(r.mixedAttentionPrepareFn).Grid(1).Block(64).On(r.device.DefaultStream()).Launch(
					dq, dk, sink.device, dr, dc, allowed, scores, denominators, out, uint32(position), uint32(2), uint32(capacity), uint32(nComp))
			})
		})
		total := capacity + nComp
		tileGridX := uint32((total + 15) / 16)
		measure(fmt.Sprintf("nComp=%d/scores_tile", nComp), func() error {
			return r.device.WithContext(func() error {
				return cudalaunch.Kernel(r.mixedAttentionTileScoresFn).Grid(tileGridX, 4).Block(256).SharedMem(66048).On(r.device.DefaultStream()).Launch(
					dr, dc, allowed, scores, out, uint32(position), uint32(capacity), uint32(nComp))
			})
		})
		shmemFinal := uint32(total * 4)
		measure(fmt.Sprintf("nComp=%d/finalize", nComp), func() error {
			return r.device.WithContext(func() error {
				return cudalaunch.Kernel(r.mixedAttentionFinalizeFn).Grid(64).Block(512).SharedMem(shmemFinal).On(r.device.DefaultStream()).Launch(
					sink.device, dr, dc, allowed, scores, denominators, out, uint32(position), uint32(capacity), uint32(nComp))
			})
		})
	}
	measure("raw_attention (layers 0-1)", func() error {
		return r.encodeRawAttention(dq, dk, sink, dr, scores.Slice(0, 64*capacity), out, position)
	})
}

func TestCUDARawAttentionFastBits(t *testing.T) {
	r := cudaSynthFoundation(t)
	if err := r.device.WithContext(r.loadMathKernels); err != nil {
		t.Fatal(err)
	}
	if r.rawAttentionPrepareFn == 0 || r.rawAttentionEvalFn == 0 {
		t.Fatal("raw attention kernels not loaded")
	}

	const capacity = 128
	raw := make([]float32, capacity*512)
	for i := range raw {
		raw[i] = float32((i*13+i/512*7)%97-48) / 64
	}
	sinks := make([]float32, 64)
	for i := range sinks {
		sinks[i] = float32(i%7-3) / 4
	}
	sink := cudaCompressedTensor(t, r, "sinks", sinks)
	dq, dk := cudaTestAlloc(t, r, 64*512), cudaTestAlloc(t, r, 512)
	drRef, drFast := cudaTestAlloc(t, r, len(raw)), cudaTestAlloc(t, r, len(raw))
	scoresRef, scoresFast := cudaTestAlloc(t, r, 64*capacity), cudaTestAlloc(t, r, 64*capacity)
	outRef, outFast := cudaTestAlloc(t, r, 32768+512+32768), cudaTestAlloc(t, r, 32768+512+32768)

	for _, position := range []int{0, 1, 63, 127, 128, 129, 2048} {
		t.Run(fmt.Sprintf("position%d", position), func(t *testing.T) {
			q, kv := cudaAttentionQualificationInputs(position)
			cudaCompressedUpload(t, dq, q)
			cudaCompressedUpload(t, dk, kv)
			cudaCompressedUpload(t, drRef, raw)
			cudaCompressedUpload(t, drFast, raw)

			// Reference
			r.disableRawAttentionFused = true
			if err := r.encodeRawAttention(dq, dk, sink, drRef, scoresRef, outRef, position); err != nil {
				t.Fatal(err)
			}
			if err := r.device.DefaultStream().Synchronize(context.Background()); err != nil {
				t.Fatal(err)
			}

			// Fast
			r.disableRawAttentionFused = false
			if err := r.encodeRawAttention(dq, dk, sink, drFast, scoresFast, outFast, position); err != nil {
				t.Fatal(err)
			}
			if err := r.device.DefaultStream().Synchronize(context.Background()); err != nil {
				t.Fatal(err)
			}

			gotOut, wantOut := cudaTestRead(t, outFast), cudaTestRead(t, outRef)
			gotCache, wantCache := cudaTestRead(t, drFast), cudaTestRead(t, drRef)
			gotScores, wantScores := cudaTestRead(t, scoresFast), cudaTestRead(t, scoresRef)

			if err := exactFloat32Bits(gotCache, wantCache); err != nil {
				t.Fatalf("cache bits mismatch at pos %d: %v", position, err)
			}
			if err := exactFloat32Bits(gotScores, wantScores); err != nil {
				t.Fatalf("scores bits mismatch at pos %d: %v", position, err)
			}
			if err := exactFloat32Bits(gotOut, wantOut); err != nil {
				t.Fatalf("output bits mismatch at pos %d: %v", position, err)
			}
		})
	}
}
