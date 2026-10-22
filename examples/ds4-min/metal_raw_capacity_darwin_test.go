//go:build darwin && arm64

package main

import (
	"crypto/sha256"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/tmc/apple/objectivec"
	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// Registered before execution. A failed fixture must not widen this bound.
const metalRawCapacityATol = 2e-5
const metalRawCapacityRTol = 2e-4

func rawCapacityClose(got, want []float32) error {
	if len(got) != len(want) {
		return fmt.Errorf("length %d, want %d", len(got), len(want))
	}
	for i, x := range got {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return fmt.Errorf("nonfinite value at %d", i)
		}
		if math.Abs(float64(x-want[i])) > metalRawCapacityATol+metalRawCapacityRTol*math.Abs(float64(want[i])) {
			return fmt.Errorf("value %d=%g, want %g", i, x, want[i])
		}
	}
	return nil
}

func rawCapacityExact(got, want []float32) error {
	if len(got) != len(want) {
		return fmt.Errorf("length %d, want %d", len(got), len(want))
	}
	for i, x := range got {
		if math.Float32bits(x) != math.Float32bits(want[i]) {
			return fmt.Errorf("value %d bits %08x, want %08x", i, math.Float32bits(x), math.Float32bits(want[i]))
		}
	}
	return nil
}

func TestMetalRawAttentionCapacity(t *testing.T) {
	dev, err := libmetal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	ops, err := newMetalPinnedOps(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer ops.Close()
	start := strings.Index(metalPinnedMSL, "kernel void rope_attention_raw(")
	end := strings.Index(metalPinnedMSL, "kernel void rope_attention_one(")
	if start < 0 || end <= start {
		t.Fatal("raw attention mutation boundaries changed")
	}
	kernel := metalPinnedMSL[start:end]
	for _, variant := range []struct{ name, anchor, replacement string }{
		{name: "production"},
		{"omitted_history", "const uint count=min(position+1,cache_cap);", "const uint count=1;"},
		{"wrong_store_slot", "const uint row=position%cache_cap;", "const uint row=(position+1)%cache_cap;"},
	} {
		passed := t.Run(variant.name, func(t *testing.T) {
			original := ops.kernels.RopeAttentionRaw
			defer func() { ops.kernels.RopeAttentionRaw = original }()
			if variant.anchor != "" {
				if strings.Count(kernel, variant.anchor) != 1 {
					t.Fatal("raw attention mutation anchor changed")
				}
				source := metalPinnedMSL[:start] + strings.Replace(kernel, variant.anchor, variant.replacement, 1) + metalPinnedMSL[end:]
				library, err := dev.CompileLibrary(source)
				if err != nil {
					t.Fatal(err)
				}
				defer objectivec.ObjectFromID(library.GetID()).Release()
				pipeline, err := dev.NewComputePipeline(library, "rope_attention_raw")
				if err != nil {
					t.Fatal(err)
				}
				defer objectivec.ObjectFromID(pipeline.GetID()).Release()
				ops.kernels.RopeAttentionRaw = pipeline
				t.Logf("compiled actual raw kernel mutation source_sha256=%x", sha256.Sum256([]byte(source)))
			} else {
				t.Logf("production source_sha256=%x", sha256.Sum256([]byte(metalPinnedMSL)))
			}
			for _, capacity := range []int{4, 128} {
				positions := []int{0, 1, 3, 4, 5}
				if capacity == 128 {
					positions = []int{0, 1, 127, 128, 129, 2048}
				}
				if variant.anchor != "" {
					if capacity != 128 {
						continue
					}
					positions = []int{127, 128, 129, 2048}
				}
				for _, position := range positions {
					t.Run(fmt.Sprintf("capacity_%d/position_%d", capacity, position), func(t *testing.T) {
						rawCapacityDeviceCase(t, ops, capacity, position, variant.name)
					})
				}
			}
		})
		if !passed {
			return // Mutants are meaningful only after the production oracle passes.
		}
	}
}

func rawCapacityDeviceCase(t *testing.T, ops *metalPinnedOps, capacity, position int, variant string) {
	t.Helper()
	arena := metalmem.NewArena(ops.dev)
	defer arena.Free()
	query, current, sinks := rawCapacityInputs(position)
	rotated := append([]float32(nil), query...)
	for head := range 64 {
		rawCapacityRotate(rotated[head*512:(head+1)*512], position, false)
	}
	history := rawCapacityHistory(position, capacity)
	cache := make([]float32, capacity*512)
	for i := range cache {
		cache[i] = 13 // Finite poison keeps a wrong-slot mutant a live arithmetic control.
	}
	for index, row := range history {
		absolute := position - len(history) + index
		copy(cache[(absolute%capacity)*512:], row)
	}
	rotatedCurrent := append([]float32(nil), current...)
	rawCapacityRotate(rotatedCurrent, position, false)
	stored := rawCapacityStoredRow(rotatedCurrent)
	wantCache := append([]float32(nil), cache...)
	copy(wantCache[(position%capacity)*512:], stored)
	history = append(history, stored)
	wantHeads, wantScores := rawCapacityAttention(rotated, history, sinks, position)
	q := metalmem.UploadIn(arena, "query", query)
	kv := metalmem.UploadIn(arena, "current", current)
	raw := metalmem.UploadIn(arena, "raw", cache)
	scores := metalmem.AllocIn[float32](arena, "scores", 64*capacity)
	out := metalmem.AllocIn[float32](arena, "output", 32768+512+32768)
	for _, values := range [][]float32{scores.HostSlice(), out.HostSlice()} {
		for i := range values {
			values[i] = float32(math.NaN())
		}
	}
	sink := testBatchF32Tensor(arena, "sinks", sinks)
	batch, err := newMetalBatch(ops)
	if err != nil {
		t.Fatal(err)
	}
	if err := batch.EncodeRopeAttentionRaw(q, kv, sink, raw, scores, out, position); err != nil {
		t.Fatal(err)
	}
	if err := batch.Wait(t.Context()); err != nil {
		t.Fatal(err)
	}
	got := out.HostSlice()
	if err := rawCapacityClose(got[:32768], rotated); err != nil {
		t.Fatalf("analytic query: %v", err)
	}
	// Verify the consumed half operands exactly before consulting the independent
	// chronological reference; do not derive expected scores from unchecked bits.
	for i, x := range got[:32768] {
		if math.Float32bits(rawCapacityRound(x, 10, -14)) != math.Float32bits(rawCapacityRound(rotated[i], 10, -14)) {
			t.Fatalf("analytic/device half query differs at %d", i)
		}
	}
	for i, x := range got[32768:33280] {
		if math.Float32bits(rawCapacityRound(x, 10, -14)) != math.Float32bits(stored[i]) {
			t.Fatalf("stored current operand differs at %d", i)
		}
	}
	cacheErr := rawCapacityExact(raw.HostSlice(), wantCache)
	headErr := rawCapacityClose(got[33280:], wantHeads)
	if variant != "production" {
		for i, x := range got[33280:] {
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				t.Fatalf("mutant is invalid, not discriminating: head value %d", i)
			}
		}
		if headErr == nil {
			t.Fatal("compiled mutation survived independent attention oracle")
		}
		if variant == "wrong_store_slot" && cacheErr == nil {
			t.Fatal("store mutation survived full cache identity")
		}
		return
	}
	if cacheErr != nil || headErr != nil {
		t.Fatalf("production cache=%v heads=%v", cacheErr, headErr)
	}
	for head := range 64 {
		if err := rawCapacityClose(scores.HostSlice()[head*capacity:head*capacity+len(history)], wantScores[head*len(history):(head+1)*len(history)]); err != nil {
			t.Fatalf("chronological scores head %d: %v", head, err)
		}
	}
}
