//go:build darwin && arm64

package main

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/tmc/apple/objectivec"
	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// metalQ4Reference is independent of the kernel's lane/reduction layout.
// Unpack all eight scale/min pairs first, then accumulate in float64.
func metalQ4Reference(raw []byte, input []float32) float64 {
	var sum float64
	for block := 0; block < len(input)/256; block++ {
		p := raw[block*144:]
		d := float64(cudaTestFloat16(binary.LittleEndian.Uint16(p)))
		dm := float64(cudaTestFloat16(binary.LittleEndian.Uint16(p[2:])))
		var scales, mins [8]byte
		for g := 0; g < 4; g++ {
			scales[g], mins[g] = p[4+g]&63, p[8+g]&63
			scales[g+4] = p[12+g]&15 | p[4+g]>>6<<4
			mins[g+4] = p[12+g]>>4 | p[8+g]>>6<<4
		}
		for j := 0; j < 128; j++ {
			g, lane := j/32*2, j%32
			for k := 0; k < 2; k++ {
				q := p[16+j] >> uint(4*k) & 15
				w := d*float64(scales[g+k])*float64(q) - dm*float64(mins[g+k])
				sum += w * float64(input[block*256+(g+k)*32+lane])
			}
		}
	}
	return sum
}

func TestMetalQ4ReferencePackedScales(t *testing.T) {
	raw := make([]byte, 144)
	binary.LittleEndian.PutUint16(raw, 0x3c00)
	for i := 4; i < 16; i++ {
		raw[i] = 0xff
	}
	for i := 16; i < len(raw); i++ {
		raw[i] = 0x11
	}
	x := make([]float32, 256)
	for i := range x {
		x[i] = 1
	}
	if got := metalQ4Reference(raw, x); got != 63*256 {
		t.Fatalf("dot=%g", got)
	}
	raw[4] &^= 0xc0
	if got := metalQ4Reference(raw, x); got == 63*256 {
		t.Fatal("high-scale mutation invisible")
	}
}

func TestMetalExpertLayerAdmission(t *testing.T) {
	layers := make([]metalLayer, 43)
	for i := range layers {
		l := &layers[i]
		l.expertGate = boundMetalTensor{typeID: typeIQ2XXS, dimensions: []uint64{4096, 2048, 256}}
		l.expertUp = l.expertGate
		l.expertDown = boundMetalTensor{typeID: typeQ2K, dimensions: []uint64{2048, 4096, 256}}
		if i >= 37 {
			l.expertGate.typeID, l.expertUp.typeID, l.expertDown.typeID = typeQ4K, typeQ4K, typeQ4K
		}
	}
	if err := planMetalExpertLayers(layers); err != nil {
		t.Fatal(err)
	}
	layers[36].expertGate.typeID, layers[36].expertUp.typeID, layers[36].expertDown.typeID = typeQ4K, typeQ4K, typeQ4K
	if err := planMetalExpertLayers(layers); err == nil {
		t.Fatal("accepted wrong splice boundary")
	}
	layers[36].expertGate.typeID, layers[36].expertUp.typeID, layers[36].expertDown.typeID = typeIQ2XXS, typeIQ2XXS, typeQ2K
	layers[37].expertDown.dimensions = []uint64{4096, 2048, 256}
	if err := planMetalExpertLayers(layers); err == nil {
		t.Fatal("accepted reversed down axes")
	}
}

func TestMetalQ4SelectedDeviceGate(t *testing.T) {
	if os.Getenv("DS4_METAL_Q4_EXPERT_GATE") != "1" {
		t.Skip("opt-in locked Metal gate required")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("gpu-lock required")
	}
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
	arena := metalmem.NewArena(dev)
	defer arena.Free()
	const cols, rows, experts, slots = 512, 7, 8, 6
	weights := metalmem.AllocIn[byte](arena, "q4-fixture", experts*rows*(cols/256)*144)
	raw := weights.HostSlice()
	for b := 0; b < len(raw)/144; b++ {
		p := raw[b*144 : (b+1)*144]
		binary.LittleEndian.PutUint16(p, 0x2400)
		binary.LittleEndian.PutUint16(p[2:], 0x2000)
		for j := 4; j < len(p); j++ {
			p[j] = byte(b*13 + j*29 + 193)
		}
	}
	ids := metalmem.AllocIn[int32](arena, "q4-ids", slots)
	copy(ids.HostSlice(), []int32{7, 1, 5, 1, 0, 6})
	x := metalmem.AllocIn[float32](arena, "q4-input", cols*slots)
	for i := range x.HostSlice() {
		x.HostSlice()[i] = float32((i*17)%101-50) / 53
	}
	out := metalmem.AllocIn[float32](arena, "q4-output", rows*slots)
	tensor := boundMetalTensor{name: "q4-fixture", typeID: typeQ4K, dimensions: []uint64{cols, rows, experts}, data: weights}
	production, err := ops.ensureQ4Experts()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { ops.q4Experts = production }()
	t.Logf("extent cols=%d rows=%d experts=%d slots=%d baseline_routes=2 compiled_mutant_routes=2 invalid_id_cases=1", cols, rows, experts, slots)
	for mutant := 0; mutant < 2; mutant++ {
		source := metalQ4ExpertsMSL
		if mutant != 0 {
			q := new(metalQ4Experts)
			const site = "((s[g-4]>>6)<<4)"
			if strings.Count(metalQ4ExpertsMSL, site) != 1 {
				t.Fatal("mutant site is not unique")
			}
			source = strings.Replace(metalQ4ExpertsMSL, site, "0u", 1)
			q.library, err = dev.CompileLibrary(source)
			if err != nil {
				t.Fatal(err)
			}
			fn := q.library.NewFunctionWithName("q4_selected_f32")
			q.pipeline, err = dev.Raw().NewComputePipelineStateWithFunctionError(fn)
			objectivec.ObjectFromID(fn.GetID()).Release()
			if err != nil {
				q.Close()
				t.Fatal(err)
			}
			ops.q4Experts = q
			defer q.Close()
		}
		name := "production"
		if mutant != 0 {
			name = "high_scale_mutant"
		}
		t.Logf("variant=%s source_sha256=%x", name, sha256.Sum256([]byte(source)))
		for _, slotted := range []bool{false, true} {
			route := "common"
			if slotted {
				route = "slotted"
			}
			passed := t.Run(name+"/"+route, func(t *testing.T) {
				input := x
				if !slotted {
					input, err = x.Slice(0, cols)
					if err != nil {
						t.Fatal(err)
					}
				}
				batch, err := newMetalBatch(ops)
				if err != nil {
					t.Fatal(err)
				}
				short, err := out.Slice(0, out.Len()-1)
				if err != nil {
					t.Fatal(err)
				}
				if batch.EncodeMatvecQ4Selected(tensor, ids, input, short, slotted) == nil || batch.encoded != 0 {
					t.Fatal("shape mutation admitted or encoded")
				}
				for i := range out.HostSlice() {
					out.HostSlice()[i] = float32(math.NaN())
				}
				if err := batch.EncodeMatvecQ4Selected(tensor, ids, input, out, slotted); err != nil {
					t.Fatal(err)
				}
				if err := batch.Wait(context.Background()); err != nil {
					t.Fatal(err)
				}
				got := append([]float32(nil), out.HostSlice()...)
				for i, v := range got {
					if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
						t.Fatalf("nonfinite device result %d", i)
					}
				}
				violations := 0
				for slot, id := range ids.HostSlice() {
					for row := 0; row < rows; row++ {
						lo := (int(id)*rows + row) * (cols / 256) * 144
						in := x.HostSlice()[:cols]
						if slotted {
							in = x.HostSlice()[slot*cols : (slot+1)*cols]
						}
						want := metalQ4Reference(raw[lo:lo+(cols/256)*144], in)
						value := float64(got[slot*rows+row])
						if math.IsNaN(value) || math.Abs(value-want) > 1e-4+1e-4*math.Abs(want) {
							violations++
							if mutant == 0 {
								t.Fatalf("route=%s slot=%d row=%d got=%g want=%g", route, slot, row, value, want)
							}
						}
					}
				}
				if mutant != 0 && violations == 0 {
					t.Fatal("compiled high-scale mutant survived independent oracle")
				}
				t.Logf("independent_oracle_values=%d violations=%d", rows*slots, violations)
			})
			if !passed {
				return
			}
		}
	}
	ops.q4Experts = production
	t.Run("production/invalid_id", func(t *testing.T) {
		ids.HostSlice()[0] = -1
		batch, err := newMetalBatch(ops)
		if err != nil {
			t.Fatal(err)
		}
		for i := range out.HostSlice() {
			out.HostSlice()[i] = float32(math.NaN())
		}
		// The invalid slot must transition to NaN; an initial NaN would allow
		// a missing write to pass this negative control.
		for i := range rows {
			out.HostSlice()[i] = 13
		}
		if err := batch.EncodeMatvecQ4Selected(tensor, ids, x, out, true); err != nil {
			t.Fatal(err)
		}
		if err := batch.Wait(context.Background()); err != nil {
			t.Fatal(err)
		}
		for i, v := range out.HostSlice()[:rows] {
			if !math.IsNaN(float64(v)) {
				t.Fatalf("invalid selected ID left stale output %d=%g", i, v)
			}
		}
	})
}
