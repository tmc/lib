//go:build darwin && arm64

package main

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	applemetal "github.com/tmc/apple/metal"
	"github.com/tmc/apple/objectivec"
	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// Numerical packets freeze source and input/oracle bytes before native device
// creation. Freeze is a separate host test; physical gates only compare bytes.
func metalDSparkFrozenFixture(t *testing.T, name string, values any) {
	t.Helper()
	dir := os.Getenv("DS4_METAL_DSPARK_FIXTURE_DIR")
	if dir == "" {
		t.Fatal("sealed numerical fixture directory required")
	}
	want, err := json.Marshal(values)
	if err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(filepath.Join(dir, name+".json"))
	if err != nil || string(got) != string(want) {
		t.Fatalf("frozen fixture %s mismatch: %v", name, err)
	}
}

// Test-owned exact-source executor. Scratch/library/pipeline stay live until
// terminal completion; an unexpected nonterminal result retains every owner.
func metalDSparkOperatorArm(t *testing.T, source, entry string, encode func(*metalmem.Arena, applemetal.MTLComputeCommandEncoder), check func()) {
	t.Helper()
	dev, err := libmetal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	retained := false
	t.Cleanup(func() {
		if !retained {
			if err := dev.Close(); err != nil {
				t.Errorf("device close: %v", err)
			}
		}
	})
	ops, err := newMetalPinnedOps(dev)
	t.Cleanup(func() {
		if ops != nil && !retained {
			ops.Close()
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	arena := metalmem.NewArena(dev)
	t.Cleanup(func() {
		if !retained {
			arena.Free()
		}
	})
	lib, err := dev.CompileLibrary(source)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if !retained {
			objectivec.ObjectFromID(lib.GetID()).Release()
		}
	})
	pipeline, err := dev.NewComputePipeline(lib, entry)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if !retained {
			objectivec.ObjectFromID(pipeline.GetID()).Release()
		}
	})
	b, err := newMetalBatch(ops)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if !b.done {
			b.endEncoding()
			b.ops.hc.complete(b)
		} else if !metalGroupedTerminal(b) {
			retained = true
			t.Error("operator retains nonterminal owners")
		}
	})
	e := b.encoder()
	e.SetComputePipelineState(pipeline)
	encode(arena, e)
	b.encoded++
	if err := b.Wait(t.Context()); err != nil {
		t.Fatal(err)
	}
	check()
}

func TestMetalDSparkMarkovPrimitiveGate(t *testing.T) {
	if os.Getenv("DS4_METAL_DSPARK_OPERATOR_GATE") != "1" {
		t.Skip("opt-in model-free operator gate")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("gpu-lock required")
	}
	const site = "previous=best;"
	if strings.Count(metalDSparkAttentionMSL, site) != 1 {
		t.Fatal("nonunique Markov mutation")
	}
	h := make([]float64, 3*4096)
	h[0], h[4096] = 1, -1
	c := make([]float64, 4098)
	c[0], c[4096], c[4097] = .5, .25, -.25
	w1, w2 := []float64{1, 0, 0, 1, 1, 1}, []float64{0, 0, 2, 0, 0, 2}
	metalDSparkFrozenFixture(t, "markov", [][]float64{h, c, w1, w2})
	for _, threshold := range []float32{.2, .5} {
		for _, mutant := range []bool{false, true} {
			source := metalDSparkAttentionMSL
			if mutant {
				source = strings.Replace(source, site, "previous=args[3];", 1)
			}
			want, probabilities := metalDSparkHeadReference(make([]float64, 9), h, w1, w2, c, 3, 3, 2, 0, float64(threshold), false)
			var ids metalmem.Slice[int32]
			var p metalmem.Slice[float32]
			var confidenceLogits metalmem.Slice[float32]
			var result metalmem.Slice[uint32]
			metalDSparkOperatorArm(t, source, "dspark_proposal_head", func(a *metalmem.Arena, e applemetal.MTLComputeCommandEncoder) {
				for i, values := range [][]float64{make([]float64, 9), h, w1, w2, c} {
					f := make([]float32, len(values))
					for j, v := range values {
						f[j] = float32(v)
					}
					buf := metalmem.UploadIn(a, "head-input", f)
					e.SetBufferWithOffsetAtIndex(buf.Buffer(), buf.Offset(), uint(i))
				}
				var args [28]byte
				for i, v := range []uint32{3, 3, 2, 0, 0, 0, 0} {
					binary.LittleEndian.PutUint32(args[i*4:], v)
				}
				e.SetBytesLengthAtIndex(args[:], 5)
				libmetal.SetScalar(e, threshold, 6)
				ids = metalmem.UploadIn(a, "ids", []int32{-99, -99, -99})
				p = metalmem.UploadIn(a, "probabilities", []float32{-99, -99, -99})
				result = metalmem.AllocIn[uint32](a, "result", 2)
				e.SetBufferWithOffsetAtIndex(ids.Buffer(), ids.Offset(), 7)
				e.SetBufferWithOffsetAtIndex(p.Buffer(), p.Offset(), 8)
				e.SetBufferWithOffsetAtIndex(result.Buffer(), result.Offset(), 9)
				confidenceLogits = metalmem.UploadIn(a, "confidence-logits", []float32{-99, -99, -99})
				e.SetBufferWithOffsetAtIndex(confidenceLogits.Buffer(), confidenceLogits.Offset(), 10)
				e.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(1), libmetal.Threads(1))
			}, func() {
				if result.HostSlice()[1] != 0 || int(result.HostSlice()[0]) != len(want) {
					t.Fatalf("head terminal count/status %v", result.HostSlice())
				}
				if confidenceLogits.HostSlice()[0] != .75 {
					t.Fatalf("raw first confidence %g want .75", confidenceLogits.HostSlice()[0])
				}
				changed := 0
				for i, v := range want {
					if int(ids.HostSlice()[i]) != v {
						changed++
					}
				}
				if (!mutant && changed != 0) || (mutant && threshold == .2 && changed == 0) {
					t.Fatalf("Markov mutant=%v changed=%d/%d", mutant, changed, len(want))
				}
				for i, v := range probabilities {
					got := float64(p.HostSlice()[i])
					if math.IsNaN(got) || math.IsInf(got, 0) {
						t.Fatal("nonfinite confidence")
					}
					if !mutant && math.Abs(got-v) > 2e-6 {
						t.Fatalf("confidence row%d got%g want%g", i, got, v)
					}
				}
				for i := len(want); i < 3; i++ {
					if ids.HostSlice()[i] != -99 {
						t.Fatal("declined ID canary modified")
					}
				}
				t.Logf("OPERATOR Markov threshold=%g mutant=%v accepted=%d/3 confidence=%d/3 changed=%d", threshold, mutant, len(want), len(probabilities), changed)
			})
		}
	}
}

func TestMetalDSparkFeaturePrimitiveGate(t *testing.T) {
	if os.Getenv("DS4_METAL_DSPARK_OPERATOR_GATE") != "1" {
		t.Skip("opt-in model-free operator gate")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("gpu-lock required")
	}
	const site = "features[slot*4096+d]=mean;"
	if strings.Count(metalDSparkFeatureMSL, site) != 1 {
		t.Fatal("nonunique feature mutation")
	}
	input := make([]float32, 16384)
	want := make([]float64, 4096)
	for lane := 0; lane < 4; lane++ {
		for d := 0; d < 4096; d++ {
			v := float32((d%17)-8)*.125 + float32(lane)*.25
			input[lane*4096+d] = v
			want[d] += float64(v) * .25
		}
	}
	metalDSparkFrozenFixture(t, "features", struct {
		Input []float32
		Want  []float64
	}{input, want})
	for _, mutant := range []bool{false, true} {
		source := metalDSparkFeatureMSL
		if mutant {
			source = strings.Replace(source, site, "features[slot*4096+d]=mean+1.0f;", 1)
		}
		var output metalmem.Slice[float32]
		var status metalmem.Slice[uint32]
		metalDSparkOperatorArm(t, source, "dspark_capture_feature", func(a *metalmem.Arena, e applemetal.MTLComputeCommandEncoder) {
			in := metalmem.UploadIn(a, "four-lane-hc", input)
			canary := make([]float32, 3*4096)
			for i := range canary {
				canary[i] = -99
			}
			output = metalmem.UploadIn(a, "captured-slots", canary)
			status = metalmem.AllocIn[uint32](a, "finite-status", 1)
			e.SetBufferWithOffsetAtIndex(in.Buffer(), in.Offset(), 0)
			e.SetBufferWithOffsetAtIndex(output.Buffer(), output.Offset(), 1)
			e.SetBufferWithOffsetAtIndex(status.Buffer(), status.Offset(), 2)
			libmetal.SetScalar(e, uint32(1), 3)
			e.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(4096), libmetal.Threads(256))
		}, func() {
			if status.HostSlice()[0] != 0 {
				t.Fatal("finite feature status")
			}
			changed := 0
			for d, v := range want {
				got := float64(output.HostSlice()[4096+d])
				if math.IsNaN(got) || math.IsInf(got, 0) {
					t.Fatal("nonfinite feature")
				}
				if got != v {
					changed++
				}
			}
			if (!mutant && changed != 0) || (mutant && changed != 4096) {
				t.Fatalf("feature mutant=%v changed=%d/4096", mutant, changed)
			}
			for i, v := range output.HostSlice() {
				if (i < 4096 || i >= 8192) && v != -99 {
					t.Fatalf("feature adjacent slot canary %d", i)
				}
			}
			t.Logf("OPERATOR feature mutant=%v compared=4096 changed=%d canary=8192", mutant, changed)
		})
	}
}

func TestMetalDSparkAttentionPrimitiveGate(t *testing.T) {
	if os.Getenv("DS4_METAL_DSPARK_OPERATOR_GATE") != "1" {
		t.Skip("opt-in model-free operator gate")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("gpu-lock required")
	}
	const site = "output[hrow*512+d]=value/denominator;"
	if strings.Count(metalDSparkAttentionMSL, site) != 1 {
		t.Fatal("nonunique attention mutation")
	}
	query, raw, sinks, want := metalDSparkAttentionFixture()
	metalDSparkFrozenFixture(t, "attention", [][]float32{query, raw, sinks, want})
	for _, mutant := range []bool{false, true} {
		source := metalDSparkAttentionMSL
		if mutant {
			source = strings.Replace(source, site, "output[hrow*512+d]=value/denominator+1.0f;", 1)
		}
		var output metalmem.Slice[float32]
		metalDSparkOperatorArm(t, source, "dspark_noncausal", func(a *metalmem.Arena, e applemetal.MTLComputeCommandEncoder) {
			for i, values := range [][]float32{query, raw, sinks} {
				buf := metalmem.UploadIn(a, "attention-input", values)
				e.SetBufferWithOffsetAtIndex(buf.Buffer(), buf.Offset(), uint(i))
			}
			out := make([]float32, len(want)+32)
			for i := range out {
				out[i] = -99
			}
			output = metalmem.UploadIn(a, "attention-output-canary", out)
			e.SetBufferWithOffsetAtIndex(output.Buffer(), output.Offset(), 3)
			for i, v := range []uint32{3, 0, 3, 1} {
				libmetal.SetScalar(e, v, i+4)
			}
			e.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(64), libmetal.Threads(32))
		}, func() {
			changed := 0
			for i, v := range want {
				got := output.HostSlice()[i]
				if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) {
					t.Fatal("nonfinite attention")
				}
				if math.Abs(float64(got-v)) > 2e-5 {
					changed++
				}
			}
			if (!mutant && changed != 0) || (mutant && changed != len(want)) {
				t.Fatalf("attention mutant=%v changed=%d/%d", mutant, changed, len(want))
			}
			for _, v := range output.HostSlice()[len(want):] {
				if v != -99 {
					t.Fatal("attention output canary changed")
				}
			}
			t.Logf("OPERATOR attention mutant=%v compared=%d changed=%d canary=32", mutant, len(want), changed)
		})
	}
}

func TestMetalDSparkRoPEPrimitiveGate(t *testing.T) {
	if os.Getenv("DS4_METAL_DSPARK_OPERATOR_GATE") != "1" {
		t.Skip("opt-in model-free operator gate")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("gpu-lock required")
	}
	const site = "values[offset]=a*c-b*s;"
	if strings.Count(metalDSparkAttentionMSL, site) != 1 {
		t.Fatal("nonunique RoPE mutation")
	}
	for _, inverse := range []uint32{0, 1} {
		for _, mutant := range []bool{false, true} {
			source := metalDSparkAttentionMSL
			if mutant {
				source = strings.Replace(source, site, "values[offset]=a*c-b*s+1.0f;", 1)
			}
			input := make([]float32, 2*2*512+32)
			for i := range input {
				input[i] = float32(i%13-6) / 8
			}
			want := append([]float32(nil), input...)
			for row := 0; row < 2; row++ {
				for head := 0; head < 2; head++ {
					for pair := 0; pair < 32; pair++ {
						angle := float64(64+row) * math.Pow(10000, -float64(pair*2)/64)
						if inverse != 0 {
							angle = -angle
						}
						offset := (row*2+head)*512 + 448 + pair*2
						a, b := float64(input[offset]), float64(input[offset+1])
						want[offset] = float32(a*math.Cos(angle) - b*math.Sin(angle))
						want[offset+1] = float32(a*math.Sin(angle) + b*math.Cos(angle))
					}
				}
			}
			var values metalmem.Slice[float32]
			metalDSparkFrozenFixture(t, "rope", input)
			metalDSparkOperatorArm(t, source, "dspark_rope_tail", func(a *metalmem.Arena, e applemetal.MTLComputeCommandEncoder) {
				values = metalmem.UploadIn(a, "rope-values", input)
				e.SetBufferWithOffsetAtIndex(values.Buffer(), values.Offset(), 0)
				for i, v := range []uint32{2, 64, inverse, 2} {
					libmetal.SetScalar(e, v, i+1)
				}
				e.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(128), libmetal.Threads(32))
			}, func() {
				changed := 0
				for i, v := range want {
					got := values.HostSlice()[i]
					if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) {
						t.Fatal("nonfinite rope")
					}
					if math.Abs(float64(got-v)) > 1e-5 {
						changed++
					}
				}
				if (!mutant && changed != 0) || (mutant && changed != 128) {
					t.Fatalf("rope mutant=%v changed=%d/%d", mutant, changed, len(want))
				}
				t.Logf("OPERATOR rope inverse=%d mutant=%v compared=%d changed=%d", inverse, mutant, len(want), changed)
			})
		}
	}
}

// Exhaustive float64 nearest-value enumeration, independent of the shader's
// binary search. Even-code tie breaking is part of the storage contract.
func metalDSparkRawReference(input []float32) []float32 {
	out := append([]float32(nil), input...)
	value := func(code int) float64 {
		e, m := code>>3, code&7
		if e == 0 {
			return float64(m) / 512
		}
		return (1 + float64(m)/8) * math.Exp2(float64(e)-7)
	}
	for row := 0; row < len(out)/512; row++ {
		for block := 0; block < 7; block++ {
			start := row*512 + block*64
			maximum := 1e-4
			for _, x := range input[start : start+64] {
				maximum = math.Max(maximum, math.Abs(float64(x)))
			}
			scale := math.Exp2(math.Ceil(math.Log2(maximum / 448)))
			for i := 0; i < 64; i++ {
				x := float64(input[start+i]) / scale
				magnitude := math.Min(math.Abs(x), 448)
				best := 0
				distance := math.Inf(1)
				for code := 0; code <= 126; code++ {
					d := math.Abs(magnitude - value(code))
					if d < distance || (d == distance && code%2 == 0) {
						best, distance = code, d
					}
				}
				v := value(best) * scale
				if x < 0 {
					v = -v
				}
				out[start+i] = metalDSparkOracleHalf(v)
			}
		}
		for i := 448; i < 512; i++ {
			out[row*512+i] = metalDSparkOracleHalf(float64(input[row*512+i]))
		}
	}
	return out
}

func metalDSparkOracleHalf(v float64) float32 {
	if v == 0 {
		return float32(v)
	}
	x := math.Abs(v)
	step := math.Exp2(-24)
	if x >= math.Exp2(-14) {
		_, exponent := math.Frexp(x)
		step = math.Exp2(float64(exponent - 11))
	}
	return float32(math.Copysign(math.RoundToEven(x/step)*step, v))
}

func TestMetalDSparkOperatorFixtures(t *testing.T) {
	input := make([]float32, 1024)
	for i := range input {
		input[i] = float32(i%29-14) * .0313
	}
	want := metalDSparkRawReference(input)
	if len(want) != 1024 || !allFinite(want) {
		t.Fatal("invalid independent raw fixture")
	}
	changed := 0
	for i, v := range want {
		if v != input[i] {
			changed++
		}
	}
	if changed == 0 {
		t.Fatal("raw fixture never exercises quantization")
	}
	t.Logf("RAW_REFERENCE changed=%d/1024 quantized=896/1024 rotary_half=128/1024", changed)
	for _, tc := range []struct {
		input float64
		want  float32
	}{{1, 1}, {1 + 1.0/2048, 1}, {1 + 3.0/2048, 1 + 2.0/1024}, {-1.5, -1.5}} {
		if got := metalDSparkOracleHalf(tc.input); got != tc.want {
			t.Fatalf("half oracle %g got%g want%g", tc.input, got, tc.want)
		}
	}
}

func TestMetalDSparkRawStorePrimitiveGate(t *testing.T) {
	if os.Getenv("DS4_METAL_DSPARK_OPERATOR_GATE") != "1" {
		t.Skip("opt-in model-free operator gate")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("gpu-lock required")
	}
	const site = "float(half(x[i]));"
	if strings.Count(metalDSparkAttentionMSL, site) != 1 {
		t.Fatal("nonunique store mutation")
	}
	input := make([]float32, 2*512)
	for i := range input {
		input[i] = float32(i%29-14) * .0313
	}
	want := metalDSparkRawReference(input)
	metalDSparkFrozenFixture(t, "raw", [][]float32{input, want})
	for _, mutant := range []bool{false, true} {
		source := metalDSparkAttentionMSL
		if mutant {
			source = strings.Replace(source, site, "float(half(x[i]))+1.0f;", 1)
		}
		var raw metalmem.Slice[float32]
		metalDSparkOperatorArm(t, source, "dspark_store_raw", func(a *metalmem.Arena, e applemetal.MTLComputeCommandEncoder) {
			kv := metalmem.UploadIn(a, "unquantized-kv", input)
			out := make([]float32, 3*512+32)
			for i := range out {
				out[i] = -99
			}
			raw = metalmem.UploadIn(a, "raw-canary", out)
			e.SetBufferWithOffsetAtIndex(kv.Buffer(), kv.Offset(), 0)
			e.SetBufferWithOffsetAtIndex(raw.Buffer(), raw.Offset(), 1)
			for i, v := range []uint32{3, 2, 2} {
				libmetal.SetScalar(e, v, i+2)
			}
			e.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(2), libmetal.Threads(1))
		}, func() {
			changed := 0
			for i, v := range want {
				index := ((2+i/512)%3)*512 + i%512
				got := raw.HostSlice()[index]
				if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) {
					t.Fatal("nonfinite raw store")
				}
				if got != v {
					changed++
				}
			}
			if (!mutant && changed != 0) || (mutant && changed != 1024) {
				t.Fatalf("raw mutant=%v changed=%d/1024", mutant, changed)
			}
			for i, v := range raw.HostSlice() {
				if (i >= 512 && i < 1024 || i >= 1536) && v != -99 {
					t.Fatalf("raw adjacent canary %d", i)
				}
			}
			t.Logf("OPERATOR raw mutant=%v compared=1024 changed=%d canary=544", mutant, changed)
		})
	}
}

func TestMetalDSparkFreezeOperatorFixtures(t *testing.T) {
	dir := os.Getenv("DS4_METAL_DSPARK_FIXTURE_DIR")
	if dir == "" {
		t.Skip("opt-in host fixture freeze")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("gpu-lock required")
	}
	if err := os.Mkdir(dir, 0700); err != nil {
		t.Fatal(err)
	}
	write := func(name string, values any) {
		t.Helper()
		data, err := json.Marshal(values)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, name+".json"), data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	h := make([]float64, 3*4096)
	h[0], h[4096] = 1, -1
	c := make([]float64, 4098)
	c[0], c[4096], c[4097] = .5, .25, -.25
	write("markov", [][]float64{h, c, {1, 0, 0, 1, 1, 1}, {0, 0, 2, 0, 0, 2}})
	input := make([]float32, 16384)
	want := make([]float64, 4096)
	for lane := 0; lane < 4; lane++ {
		for d := 0; d < 4096; d++ {
			v := float32(d%17-8)*.125 + float32(lane)*.25
			input[lane*4096+d] = v
			want[d] += float64(v) * .25
		}
	}
	write("features", struct {
		Input []float32
		Want  []float64
	}{input, want})
	q, raw, sinks, out := metalDSparkAttentionFixture()
	write("attention", [][]float32{q, raw, sinks, out})
	rope := make([]float32, 2*2*512+32)
	for i := range rope {
		rope[i] = float32(i%13-6) / 8
	}
	write("rope", rope)
	kv := make([]float32, 1024)
	for i := range kv {
		kv[i] = float32(i%29-14) * .0313
	}
	write("raw", [][]float32{kv, metalDSparkRawReference(kv)})
	for _, kind := range []uint32{0, 1, 8} {
		data, in, want := metalDSparkDenseFixture(kind)
		write(string(rune('0'+kind))+"-dense", struct {
			Data  []byte
			Input []float32
			Want  []float64
		}{data, in, want})
	}
}
