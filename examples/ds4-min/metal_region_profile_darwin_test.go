//go:build darwin && arm64

package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"
)

// metalRegionTokens is the batch this instrument profiles. It matches the
// CUDA phase profile's warmed batch so the two rankings cover the same
// positions; like that one, it says nothing about the positions a
// 2048-token run reaches, because attention cost grows with position.
const metalRegionTokens = 8

// expectedMetalRegions returns the spans a run of tokens positions must
// produce, in completion order.
//
// evalLayer builds exactly two command buffers per layer per position:
// the attention opening, and the attention closing sharing a buffer with
// the FFN. Nothing else in the layer path commits, so any other span is
// an unaccounted buffer rather than a region this profile can name.
func expectedMetalRegions(tokens int) []metalRegion {
	if tokens < 1 {
		return nil
	}
	regions := make([]metalRegion, 0, tokens*ds4Layers*2)
	for position := range tokens {
		for layer := range ds4Layers {
			regions = append(regions,
				metalRegion{Region: "opening", Layer: layer, Position: position},
				metalRegion{Region: "closing-ffn", Layer: layer, Position: position},
			)
		}
	}
	return regions
}

// validateMetalRegions refuses a log that does not match the expected
// inventory exactly, coordinate by coordinate and in order.
//
// The check is over coordinates, not durations: an instrument that
// stopped recording, recorded a buffer twice, or attributed a span to the
// wrong layer produces a wrong ranking that reads as a plausible one, and
// only the inventory catches that. Durations are checked only for the
// properties a GPU clock cannot violate.
func validateMetalRegions(tokens int, regions []metalRegion) error {
	want := expectedMetalRegions(tokens)
	if len(want) == 0 {
		return fmt.Errorf("metal region profile: token count %d is not positive", tokens)
	}
	if len(regions) != len(want) {
		return fmt.Errorf("metal region profile: got %d spans, want %d", len(regions), len(want))
	}
	for i, got := range regions {
		if got.Region != want[i].Region || got.Layer != want[i].Layer || got.Position != want[i].Position {
			return fmt.Errorf("metal region profile: span %d is (%q,%d,%d), want (%q,%d,%d)",
				i, got.Region, got.Layer, got.Position, want[i].Region, want[i].Layer, want[i].Position)
		}
		if math.IsNaN(got.Start) || math.IsNaN(got.End) {
			return fmt.Errorf("metal region profile: span %d has a NaN bound", i)
		}
		if got.Seconds() < 0 {
			return fmt.Errorf("metal region profile: span %d ends %g s before it starts", i, -got.Seconds())
		}
	}
	return nil
}

// metalRegionTotals sums the spans by region name.
func metalRegionTotals(regions []metalRegion) map[string]float64 {
	totals := make(map[string]float64)
	for _, r := range regions {
		totals[r.Region] += r.Seconds()
	}
	return totals
}

func TestMetalRegionInventory(t *testing.T) {
	want := expectedMetalRegions(metalRegionTokens)
	if n := len(want); n != metalRegionTokens*ds4Layers*2 {
		t.Fatalf("inventory has %d spans, want %d", n, metalRegionTokens*ds4Layers*2)
	}
	fill := func(regions []metalRegion) []metalRegion {
		out := append([]metalRegion(nil), regions...)
		for i := range out {
			out[i].Start, out[i].End = 0, 1
		}
		return out
	}
	if err := validateMetalRegions(metalRegionTokens, fill(want)); err != nil {
		t.Fatalf("exact inventory rejected: %v", err)
	}
	for _, tt := range []struct {
		name    string
		regions []metalRegion
	}{
		{"short", fill(want[:len(want)-1])},
		{"long", fill(append(append([]metalRegion(nil), want...), want[0]))},
		{"swapped order", func() []metalRegion {
			out := fill(want)
			out[0], out[1] = out[1], out[0]
			return out
		}()},
		{"duplicated span", func() []metalRegion {
			out := fill(want)
			out[3] = out[1]
			return out
		}()},
		{"wrong layer", func() []metalRegion {
			out := fill(want)
			out[2].Layer++
			return out
		}()},
		{"unnamed buffer", func() []metalRegion {
			out := fill(want)
			out[4].Region = ""
			return out
		}()},
		{"negative span", func() []metalRegion {
			out := fill(want)
			out[5].End = out[5].Start - 1
			return out
		}()},
		{"NaN bound", func() []metalRegion {
			out := fill(want)
			out[6].End = math.NaN()
			return out
		}()},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if err := validateMetalRegions(metalRegionTokens, tt.regions); err == nil {
				t.Fatal("accepted a log the inventory should refuse")
			}
		})
	}
	if err := validateMetalRegions(0, nil); err == nil {
		t.Fatal("accepted a zero token count")
	}
}

// TestMetalRegionProfile ranks the attention opening against the closing
// and FFN, and refuses to report the ranking unless the instrumented run
// reproduces the uninstrumented run's logits bit for bit.
//
// Identity is the gate because the instrument is meant to be inert: it
// reads two properties of a command buffer that has already completed. A
// bit that moves means it is not, and no ranking from it is worth having.
func TestMetalRegionProfile(t *testing.T) {
	model := os.Getenv("DS4_METAL_REGION_PROFILE_MODEL")
	if model == "" {
		t.Skip("DS4_METAL_REGION_PROFILE_MODEL not set")
	}
	r, err := newMetalRuntime(model, 4096)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	tokens := make([]int, metalRegionTokens)
	for i := range tokens {
		tokens[i] = i + 1
	}
	eval := func() []uint32 {
		t.Helper()
		if err := r.Reset(); err != nil {
			t.Fatal(err)
		}
		if err := r.EvalBatch(tokens, 0); err != nil {
			t.Fatal(err)
		}
		logits := r.Logits()
		bits := make([]uint32, len(logits))
		for i, v := range logits {
			bits[i] = math.Float32bits(v)
		}
		return bits
	}

	reference := eval()
	if len(reference) == 0 {
		t.Fatal("reference run produced no logits")
	}

	log := &metalRegionLog{}
	r.ops.regions = log
	instrumented := eval()
	r.ops.regions = nil

	if len(instrumented) != len(reference) {
		t.Fatalf("instrumented run produced %d logits, reference %d", len(instrumented), len(reference))
	}
	for i := range reference {
		if instrumented[i] != reference[i] {
			t.Fatalf("logit %d differs: instrumented %#08x, reference %#08x", i, instrumented[i], reference[i])
		}
	}

	regions := log.all()
	if err := validateMetalRegions(metalRegionTokens, regions); err != nil {
		t.Fatal(err)
	}
	totals := metalRegionTotals(regions)
	for _, name := range []string{"opening", "closing-ffn"} {
		t.Logf("%s: %d spans, %.6f s GPU time", name, len(regions)/2, totals[name])
	}

	if out := os.Getenv("DS4_METAL_REGION_PROFILE_OUT"); out != "" {
		data, err := json.MarshalIndent(regions, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(out, data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
}
