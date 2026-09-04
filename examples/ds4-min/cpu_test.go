package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strconv"
	"testing"
)

const fixtureTolerance = 1e-4

type fixturePayload struct {
	Schema string      `json:"schema"`
	Seed   uint64      `json:"seed"`
	Prompt []int       `json:"prompt"`
	Decode []int       `json:"decode"`
	Traces []tinyTrace `json:"traces"`
}

type fixtureManifest struct {
	Payload fixturePayload `json:"payload"`
	SHA256  string         `json:"payload_sha256"`
}

func TestTinyCPUFixture(t *testing.T) {
	m := readFixture(t, "testdata/fixture.json")
	cpu := newTinyCPU(m.Payload.Seed)
	tokens := append(append([]int(nil), m.Payload.Prompt...), m.Payload.Decode...)
	if len(tokens) != len(m.Payload.Traces) {
		t.Fatalf("tokens = %d, traces = %d", len(tokens), len(m.Payload.Traces))
	}
	for i, token := range tokens {
		got, err := cpu.Step(token)
		if err != nil {
			t.Fatalf("step %d: %v", i, err)
		}
		compareTrace(t, i, got, m.Payload.Traces[i])
	}
	if len(cpu.cache.compressed) == 0 {
		t.Fatal("fixture did not cross the ratio-4 compression boundary")
	}
}

func TestFixtureRegeneration(t *testing.T) {
	dir := t.TempDir()
	paths := []string{filepath.Join(dir, "first.json"), filepath.Join(dir, "second.json")}
	for _, path := range paths {
		cmd := exec.Command("go", "run", "./testdata/generate.go", "-out", path)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("generate: %v\n%s", err, out)
		}
	}
	first, err := os.ReadFile(paths[0])
	if err != nil {
		t.Fatal(err)
	}
	second, err := os.ReadFile(paths[1])
	if err != nil {
		t.Fatal(err)
	}
	checked, err := os.ReadFile("testdata/fixture.json")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first, second) {
		t.Fatal("two fixture generations differ")
	}
	if !bytes.Equal(first, checked) {
		t.Fatal("checked fixture is stale; run go generate")
	}
	readFixture(t, paths[0])
}

func TestOperators(t *testing.T) {
	t.Run("rms norm", func(t *testing.T) {
		got, err := rmsNorm([]float32{3, 4}, []float32{1, 2}, 1e-5)
		if err != nil {
			t.Fatal(err)
		}
		wantScale := float32(1 / math.Sqrt(12.5+1e-5))
		closeSlice(t, "rms", got, []float32{3 * wantScale, 8 * wantScale})
	})
	t.Run("matvec", func(t *testing.T) {
		got, err := matvec([]float32{1, 2, 3, 4}, []float32{2, -1}, 2, 2)
		if err != nil {
			t.Fatal(err)
		}
		closeSlice(t, "matvec", got, []float32{0, 2})
	})
	t.Run("rotary", func(t *testing.T) {
		got, err := rotary([]float32{1, 0}, 1, 10000)
		if err != nil {
			t.Fatal(err)
		}
		closeSlice(t, "rotary", got, []float32{float32(math.Cos(1)), float32(math.Sin(1))})
	})
	t.Run("swiglu", func(t *testing.T) {
		got, err := swiglu([]float32{0, 1}, []float32{3, 2})
		if err != nil {
			t.Fatal(err)
		}
		closeSlice(t, "swiglu", got, []float32{0, 2 / (1 + float32(math.Exp(-1)))})
	})
	t.Run("expert ties", func(t *testing.T) {
		got, err := routeExperts([]float32{0, 0, 0, 0}, []float32{0, 0, 0, 0}, 2)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got.SelectedExperts, []int{0, 1}) {
			t.Fatalf("experts = %v", got.SelectedExperts)
		}
		closeSlice(t, "weights", got.Weights, []float32{.5, .5})
	})
	t.Run("biased selection unbiased weights", func(t *testing.T) {
		got, err := routeExperts([]float32{4, 3, 2}, []float32{-10, 0, 10}, 2)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got.SelectedExperts, []int{2, 1}) {
			t.Fatalf("experts = %v", got.SelectedExperts)
		}
		want0 := float32(1 / (1 + math.Exp(1)))
		closeSlice(t, "weights", got.Weights, []float32{want0, 1 - want0})
	})
}

func TestCompressedAttentionBoundaries(t *testing.T) {
	for _, ratio := range []int{4, 128} {
		t.Run(strconv.Itoa(ratio), func(t *testing.T) {
			cache := testCompressedAttention(t, ratio)
			if _, _, err := cache.Attend([]float32{1, 0}, []float32{1, 0}, []float32{1, 0}, 0, 1); err == nil {
				t.Fatal("Attend before Add succeeded")
			}
			for pos := 0; pos < ratio-1; pos++ {
				if err := cache.Add([]float32{1, 2}, []float32{1, 0}, pos); err != nil {
					t.Fatal(err)
				}
				if len(cache.compressed) != 0 {
					t.Fatalf("ratio %d emitted at position %d", ratio, pos)
				}
			}
			if err := cache.Add([]float32{2, 3}, []float32{0, 1}, ratio-1); err != nil {
				t.Fatal(err)
			}
			if len(cache.compressed) != 1 || !allFinite(cache.compressed[0].key, cache.compressed[0].index) {
				t.Fatalf("ratio %d compressed rows = %d", ratio, len(cache.compressed))
			}
		})
	}
}

func TestCompressedAttentionUsesLearnedWeights(t *testing.T) {
	a := testCompressedAttention(t, 4)
	b := testCompressedAttention(t, 4)
	b.attention.gate[2*2] = 3
	b.attention.ape[2] = 2
	for pos := 0; pos < 4; pos++ {
		x := []float32{float32(pos + 1), float32(pos + 2)}
		for _, cache := range []*compressedAttention{a, b} {
			if err := cache.Add(x, []float32{1, 0}, pos); err != nil {
				t.Fatal(err)
			}
		}
	}
	if reflect.DeepEqual(a.compressed[0].key, b.compressed[0].key) {
		t.Fatal("compressor output ignored learned gate and APE weights")
	}
}

func TestCompressedAttentionLargeTopK(t *testing.T) {
	cache := testCompressedAttention(t, 4)
	for pos := 0; pos < 6; pos++ {
		if err := cache.Add([]float32{float32(pos + 1), 1}, []float32{1, 0}, pos); err != nil {
			t.Fatal(err)
		}
	}
	got, selected, err := cache.Attend([]float32{1, 0}, []float32{1, 0}, []float32{1, 0}, 5, math.MaxInt)
	if err != nil {
		t.Fatal(err)
	}
	if len(selected) != 1 || len(got) != 2 || !allFinite(got) {
		t.Fatalf("selected = %v, output = %v", selected, got)
	}
}

func TestMalformedCPUInputs(t *testing.T) {
	tests := []struct {
		name string
		fn   func() error
	}{
		{"rms", func() error { _, err := rmsNorm(nil, nil, 1e-5); return err }},
		{"matvec", func() error { _, err := matvec([]float32{1}, []float32{1}, 2, 1); return err }},
		{"rotary", func() error { _, err := rotary([]float32{1}, 0, 10000); return err }},
		{"swiglu", func() error { _, err := swiglu([]float32{1}, nil); return err }},
		{"routing top-k", func() error { _, err := routeExperts([]float32{1}, []float32{0}, 2); return err }},
		{"hc split", func() error { _, _, _, err := hcSplit(nil, nil, nil, 2, 3, 1e-6); return err }},
		{"hc sum", func() error { _, err := hcWeightedSum([]float32{1}, []float32{1}, 2); return err }},
		{"hc post", func() error { _, err := hcPost(nil, nil, nil, nil); return err }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if err := test.fn(); err == nil {
				t.Fatal("malformed input succeeded")
			}
		})
	}
}

func FuzzCPUShapes(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4}, uint8(2), uint8(2))
	f.Add([]byte{}, uint8(0), uint8(0))
	f.Fuzz(func(t *testing.T, raw []byte, rowsByte, colsByte uint8) {
		if len(raw) > 256 {
			t.Skip()
		}
		values := make([]float32, len(raw))
		for i := range raw {
			values[i] = float32(int8(raw[i]))
		}
		rows, cols := int(rowsByte%17), int(colsByte%17)
		_, _ = matvec(values, values, rows, cols)
		_, _ = rmsNorm(values, values, 1e-5)
		_, _ = rotary(values, rows, 10000)
	})
}

func testCompressedAttention(t *testing.T, ratio int) *compressedAttention {
	t.Helper()
	dim, hidden := 2, 2
	width := dim
	if ratio == 4 {
		width *= 2
	}
	w := compressorWeights{
		kv: make([]float32, width*hidden), gate: make([]float32, width*hidden),
		ape: make([]float32, ratio*width), norm: []float32{1, 1},
	}
	for i := 0; i < width && i < hidden; i++ {
		w.kv[i*hidden+i] = 1
	}
	if ratio == 4 {
		w.kv[2*hidden] = 1
		w.kv[3*hidden+1] = 1
	}
	cache, err := newCompressedAttention(dim, hidden, 2, ratio, w, w, []float32{1, 0, 0, 1}, []float32{1, 1}, 0)
	if err != nil {
		t.Fatal(err)
	}
	return cache
}

func readFixture(t *testing.T, path string) fixtureManifest {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var m fixtureManifest
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	payload, err := json.Marshal(m.Payload)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(payload)
	if hex.EncodeToString(sum[:]) != m.SHA256 {
		t.Fatal("fixture payload SHA-256 mismatch")
	}
	return m
}

func compareTrace(t *testing.T, step int, got, want tinyTrace) {
	t.Helper()
	if got.Token != want.Token || got.Position != want.Position || !reflect.DeepEqual(got.Compressed, want.Compressed) || !reflect.DeepEqual(got.Experts, want.Experts) {
		t.Fatalf("step %d discrete trace differs: got %+v want %+v", step, got, want)
	}
	for _, field := range []struct {
		name      string
		got, want []float32
	}{
		{"norm", got.Norm, want.Norm}, {"hc attention mix", got.HCAttnMix, want.HCAttnMix},
		{"hc FFN mix", got.HCFFNMix, want.HCFFNMix},
		{"query", got.Query, want.Query}, {"key", got.Key, want.Key},
		{"attention", got.Attention, want.Attention}, {"weights", got.Weights, want.Weights},
		{"moe", got.MoE, want.MoE}, {"hc", got.HC, want.HC}, {"logits", got.Logits, want.Logits},
	} {
		closeSlice(t, field.name, field.got, field.want)
	}
}

func closeSlice(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length = %d, want %d", name, len(got), len(want))
	}
	for i := range got {
		delta := math.Abs(float64(got[i] - want[i]))
		limit := fixtureTolerance + fixtureTolerance*math.Abs(float64(want[i]))
		if delta > limit {
			t.Fatalf("%s[%d] = %g, want %g (delta %g > %g)", name, i, got[i], want[i], delta, limit)
		}
	}
}
