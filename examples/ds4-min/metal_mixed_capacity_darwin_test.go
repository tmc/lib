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

// Registered before execution; the independent mixed oracle cannot widen it.
const metalMixedCapacityATol = 2e-5
const metalMixedCapacityRTol = 2e-4

func mixedCapacityClose(got, want []float32) error {
	if len(got) != len(want) {
		return fmt.Errorf("length %d, want %d", len(got), len(want))
	}
	for i, value := range got {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || math.Abs(float64(value-want[i])) > metalMixedCapacityATol+metalMixedCapacityRTol*math.Abs(float64(want[i])) {
			return fmt.Errorf("value %d=%g want %g", i, value, want[i])
		}
	}
	return nil
}

func TestMetalMixedAttentionCapacity(t *testing.T) {
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
	start := strings.Index(metalPinnedMSL, "kernel void rope_attention_mixed_wide(")
	end := strings.Index(metalPinnedMSL, "kernel void rope_attention_raw(")
	if start < 0 || end <= start {
		t.Fatal("wide mutation boundary changed")
	}
	wide := metalPinnedMSL[start:end]
	for _, variant := range []struct {
		name, anchor, replacement string
		occurrences               int
	}{
		{name: "production"},
		{"ignore_allowed", "if(allowed[cidx])", "if(true)", 3},
		{"collapse_comp_rows", "comp[cidx*512+i]", "comp[i]", 2},
	} {
		passed := t.Run(variant.name, func(t *testing.T) {
			original := ops.kernels.RopeAttentionWide
			defer func() { ops.kernels.RopeAttentionWide = original }()
			source := metalPinnedMSL
			if variant.anchor != "" {
				if strings.Count(wide, variant.anchor) != variant.occurrences {
					t.Fatal("wide mutation anchor changed")
				}
				source = source[:start] + strings.ReplaceAll(wide, variant.anchor, variant.replacement) + source[end:]
				library, err := dev.CompileLibrary(source)
				if err != nil {
					t.Fatal(err)
				}
				defer objectivec.ObjectFromID(library.GetID()).Release()
				pipeline, err := dev.NewComputePipeline(library, "rope_attention_mixed_wide")
				if err != nil {
					t.Fatal(err)
				}
				defer objectivec.ObjectFromID(pipeline.GetID()).Release()
				ops.kernels.RopeAttentionWide = pipeline
			}
			t.Logf("actual_wide_source_sha256=%x", sha256.Sum256([]byte(source)))
			for _, position := range []int{127, 128, 129, 2048} {
				counts := []int{0, 1, 512, 513}
				if variant.anchor != "" {
					counts = []int{512, 513}
				}
				for _, count := range counts {
					if !t.Run(fmt.Sprintf("position_%d/compressed_%d", position, count), func(t *testing.T) {
						mixedCapacityDeviceCase(t, ops, position, count, variant.name)
					}) {
						return
					}
				}
			}
		})
		if !passed {
			return
		}
	}
}

func mixedCapacityDeviceCase(t *testing.T, ops *metalPinnedOps, position, count int, variant string) {
	t.Helper()
	arena := metalmem.NewArena(ops.dev)
	defer arena.Free()
	query, current, sinks := mixedCapacityInputs(position)
	rotated := append([]float32(nil), query...)
	for head := range 64 {
		mixedCapacityRotate(rotated[head*512:(head+1)*512], position, false)
	}
	currentRotated := append([]float32(nil), current...)
	mixedCapacityRotate(currentRotated, position, false)
	stored := rawCapacityStoredRow(currentRotated)
	history := rawCapacityHistory(position, 128)
	cache := make([]float32, 128*512)
	for i := range cache {
		cache[i] = 13
	}
	for i, row := range history {
		copy(cache[((position-len(history)+i)%128)*512:], row)
	}
	wantCache := append([]float32(nil), cache...)
	copy(wantCache[(position%128)*512:], stored)
	history = append(history, stored)
	compressed, mask := mixedCapacityRows(count)
	flat := make([]float32, max(1, count)*512)
	for i, row := range compressed {
		copy(flat[i*512:], row)
	}
	if count == 0 {
		mask = []uint32{0}
	}
	wantHeads, wantScores := mixedCapacityAttention(rotated, history, compressed, mask, sinks, position)
	q := metalmem.UploadIn(arena, "query", query)
	kv := metalmem.UploadIn(arena, "current", current)
	raw := metalmem.UploadIn(arena, "raw", cache)
	comp := metalmem.UploadIn(arena, "compressed", flat)
	allowed := metalmem.UploadIn(arena, "allowed", mask)
	scores := metalmem.AllocIn[float32](arena, "scores", 64*(len(history)+count))
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
	if err := batch.EncodeRopeAttentionMixed(q, kv, sink, raw, comp, count, allowed, scores, out, position, 2); err != nil {
		t.Fatal(err)
	}
	if err := batch.Wait(t.Context()); err != nil {
		t.Fatal(err)
	}
	got := out.HostSlice()
	if err := mixedCapacityClose(got[:32768], rotated); err != nil {
		t.Fatalf("analytic query: %v", err)
	}
	for i, x := range got[:32768] {
		if math.Float32bits(rawCapacityRound(x, 10, -14)) != math.Float32bits(rawCapacityRound(rotated[i], 10, -14)) {
			t.Fatalf("consumed half query mismatch at %d", i)
		}
	}
	if err := rawCapacityExact(raw.HostSlice(), wantCache); err != nil {
		t.Fatalf("raw cache: %v", err)
	}
	if err := rawCapacityExact(comp.HostSlice(), flat); err != nil {
		t.Fatalf("compressed input mutated: %v", err)
	}
	headErr := mixedCapacityClose(got[33280:], wantHeads)
	if variant != "production" {
		for i, x := range got[33280:] {
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				t.Fatalf("invalid mutant output at %d", i)
			}
		}
		if headErr == nil {
			t.Fatal("actual wide mutant survived independent oracle")
		}
		return
	}
	if headErr != nil {
		t.Fatalf("heads: %v", headErr)
	}
	if err := mixedCapacityClose(scores.HostSlice(), wantScores); err != nil {
		t.Fatalf("chronological/masked scores: %v", err)
	}
}
