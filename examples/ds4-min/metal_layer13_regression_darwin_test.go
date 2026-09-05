//go:build darwin && arm64

package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// Artifact tensors live outside git. This digest pins their manifest including
// each file SHA256, source tensor offset, oracle revision and extraction limits.
const metalLayer13ManifestSHA256 = "2416a9c85b9236363e7747083cf66706b303cdd94e85cff67537ef07f89b00b9"
const metalLayer13ATol = 1e-4
const metalLayer13RTol = 1e-4

type metalLayer13Fixture struct{ files map[string][]byte }

func metalLayer13Load(t *testing.T) metalLayer13Fixture {
	t.Helper()
	dir := os.Getenv("DS4_METAL_LAYER13_FIXTURE_DIR")
	if dir == "" {
		t.Skip("DS4_METAL_LAYER13_FIXTURE_DIR is not set")
	}
	manifest, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(manifest)) != metalLayer13ManifestSHA256 {
		t.Fatal("layer13 fixture manifest digest mismatch")
	}
	var parsed struct {
		Files map[string]struct {
			Bytes int    `json:"bytes"`
			Hash  string `json:"sha256"`
		} `json:"files"`
	}
	if err := json.Unmarshal(manifest, &parsed); err != nil {
		t.Fatal(err)
	}
	fixture := metalLayer13Fixture{make(map[string][]byte)}
	for name, size := range map[string]int{"KVraw.bin": 2048, "KVnorm.bin": 2048, "KVcur.bin": 2048, "attn_norm.bin": 16384, "norm-weight.bin": 2048, "kv-weight.bin": 2228224} {
		metadata, ok := parsed.Files[name]
		if !ok || metadata.Bytes != size {
			t.Fatalf("invalid fixture descriptor %s", name)
		}
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		if len(data) != size || fmt.Sprintf("%x", sha256.Sum256(data)) != metadata.Hash {
			t.Fatalf("fixture bytes mismatch %s", name)
		}
		fixture.files[name] = data
	}
	return fixture
}
func (f metalLayer13Fixture) floats(t *testing.T, name string) []float32 {
	t.Helper()
	data := f.files[name]
	out := make([]float32, len(data)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:]))
		if math.IsNaN(float64(out[i])) || math.IsInf(float64(out[i]), 0) {
			t.Fatalf("nonfinite fixture %s[%d]", name, i)
		}
	}
	return out
}
func metalLayer13Close(got, want []float32) error {
	if len(got) != len(want) {
		return fmt.Errorf("length%d want%d", len(got), len(want))
	}
	for i, v := range got {
		x, y := float64(v), float64(want[i])
		if math.IsNaN(x) || math.IsInf(x, 0) || math.IsNaN(y) || math.IsInf(y, 0) {
			return fmt.Errorf("nonfinite element%d", i)
		}
		if math.Abs(x-y) > metalLayer13ATol+metalLayer13RTol*math.Abs(y) {
			return fmt.Errorf("element%d=%g want%g", i, x, y)
		}
	}
	return nil
}

// Each gate receives the original oracle input, not a previous candidate's
// output. Stop on the first failed gate to preserve causal isolation.
func TestMetalLayer13Regression(t *testing.T) {
	fixture := metalLayer13Load(t)
	dev, err := libmetal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
	}()
	ops, err := newMetalPinnedOps(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer ops.Close()
	arena := metalmem.NewArena(dev)
	defer arena.Free()
	norm := boundMetalTensor{name: "blk.13.attn_kv_a_norm.weight", typeID: 0, dimensions: []uint64{512}, data: metalmem.UploadIn(arena, "norm", fixture.files["norm-weight.bin"])}
	weight := boundMetalTensor{name: "blk.13.attn_kv.weight", typeID: 8, dimensions: []uint64{4096, 512}, data: metalmem.UploadIn(arena, "kv", fixture.files["kv-weight.bin"])}
	sinks := boundMetalTensor{name: "zero sinks", typeID: 0, dimensions: []uint64{64}, data: metalmem.UploadIn(arena, "sinks", make([]byte, 256))}
	if err := arena.Err(); err != nil {
		t.Fatal(err)
	}
	t.Logf("device=%s; bounded replay weight bytes=%d; full model not opened", dev.Name(), len(fixture.files["kv-weight.bin"])+len(fixture.files["norm-weight.bin"]))
	if !t.Run("same_input_fp8", func(t *testing.T) {
		input, want := fixture.floats(t, "KVnorm.bin"), fixture.floats(t, "KVcur.bin")
		_, got, _, err := ops.RopeAttentionOne(make([]float32, 64*512), input, sinks, 0)
		if err != nil {
			t.Fatal(err)
		}
		if err := metalLayer13Close(got, want); err != nil {
			t.Fatal(err)
		}
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("exact FP8 element%d=%g want%g", i, got[i], want[i])
			}
		}
		// Only this element moves across the cell midpoint; blockamax is unchanged.
		input[362] += 3e-6
		_, bad, _, err := ops.RopeAttentionOne(make([]float32, 64*512), input, sinks, 0)
		if err != nil {
			t.Fatal(err)
		}
		if bad[362] != -.03125 || want[362] != -.03515625 {
			t.Fatalf("midpoint mutant=%g oracle=%g", bad[362], want[362])
		}
		if metalLayer13Close(bad, want) == nil {
			t.Fatal("midpoint mutant passed strict gate")
		}
	}) {
		return
	}
	if !t.Run("same_input_rms512", func(t *testing.T) {
		got, err := ops.RMS(fixture.floats(t, "KVraw.bin"), norm, 1, true)
		if err != nil {
			t.Fatal(err)
		}
		want := fixture.floats(t, "KVnorm.bin")
		t.Logf("RMS[362]=%.10g oracle=%.10g signed_midpoint_distance=%.10g", got[362], want[362], float64(got[362])+.033203125)
		if err := metalLayer13Close(got, want); err != nil {
			t.Fatal(err)
		}
		// The diagnostic's purpose is preserving the observed quantization cell;
		// the ordinary float tolerance alone is insufficient at this midpoint.
		if metalLayer13FP8(got)[362] != fixture.floats(t, "KVcur.bin")[362] {
			t.Fatal("same-input RMS crosses the layer13 FP8 cell midpoint")
		}
	}) {
		return
	}
	t.Run("same_input_q80_kv", func(t *testing.T) {
		got, err := ops.MatvecQ80(weight, fixture.floats(t, "attn_norm.bin"))
		if err != nil {
			t.Fatal(err)
		}
		want := fixture.floats(t, "KVraw.bin")
		t.Logf("KVraw[362]=%.10g oracle=%.10g", got[362], want[362])
		if err := metalLayer13Close(got, want); err != nil {
			t.Fatal(err)
		}
	})
}

func metalLayer13FP8(input []float32) []float32 {
	out := append([]float32(nil), input...)
	for off := 0; off < 448; off += 64 {
		amax := 1e-4
		for _, v := range input[off : off+64] {
			amax = math.Max(amax, math.Abs(float64(v)))
		}
		scale := math.Exp2(math.Ceil(math.Log2(amax / 448)))
		for i := off; i < off+64; i++ {
			v := float64(input[i]) / scale
			_, e := math.Frexp(math.Abs(v))
			step := math.Ldexp(1, max(e-1, -6)-3)
			out[i] = float32(math.RoundToEven(v/step) * step * scale)
		}
	}
	return out
}

// Checks retained artifact identity and cell sensitivity without creating a
// Metal device. It does not substitute for TestMetalLayer13Regression.
func TestMetalLayer13Fixture(t *testing.T) {
	fixture := metalLayer13Load(t)
	input, want := fixture.floats(t, "KVnorm.bin"), fixture.floats(t, "KVcur.bin")
	got := metalLayer13FP8(input)
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("scalar FP8[%d]=%g want%g", i, got[i], want[i])
		}
	}
	input[362] += 3e-6
	bad := metalLayer13FP8(input)
	if bad[362] != -.03125 || metalLayer13Close(bad, want) == nil {
		t.Fatal("midpoint control not rejected")
	}
}
