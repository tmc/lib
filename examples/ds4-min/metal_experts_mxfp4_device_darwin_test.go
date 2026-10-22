//go:build darwin && arm64

package main

import (
	"context"
	"crypto/sha256"
	"math"
	"os"
	"strings"
	"testing"

	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

func TestMetalMXFP4SelectedUpstreamParityGate(t *testing.T) {
	if os.Getenv("DS4_METAL_MXFP4_SELECTED_GATE") != "1" {
		t.Skip("opt-in locked Metal selected MXFP4 gate required")
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
	const k, m, e, slots = 64, 5, 256, 6
	w := metalmem.AllocIn[byte](arena, "mxfp4-fixture", e*m*k/32*17)
	raw := w.HostSlice()
	for b := 0; b < len(raw)/17; b++ {
		p := raw[b*17:]
		p[0] = byte(125 + b%5)
		if b%19 == 0 {
			p[0] = 0
		}
		for i := 1; i < 17; i++ {
			p[i] = byte(b*29 + i*37)
		}
	}
	ids := metalmem.AllocIn[int32](arena, "selected-ids", slots)
	copy(ids.HostSlice(), []int32{255, 2, 7, 0, 2, 17})
	x := metalmem.AllocIn[float32](arena, "selected-input", slots*k)
	for i := range x.HostSlice() {
		x.HostSlice()[i] = float32((i*17)%251-125) / 73
	}
	out := metalmem.AllocIn[float32](arena, "selected-output", slots*m)
	tensor := boundMetalTensor{name: "mxfp4-fixture", typeID: typeMXFP4, dimensions: []uint64{k, m, e}, data: w}
	variants := []struct{ name, old, new string }{
		{"baseline", "", ""},
		{"scale_mutant", "total+=(scale*v)", "total+=(2*scale*v)"},
		{"nibble_mutant", "((lane>>4)*4)", "(((lane>>4)^1)*4)"},
		{"zero_scale_mutant", "p[0]==0 ? 0x00400000u", "p[0]==0 ? 0x3f800000u"},
	}
	for _, variant := range variants {
		source := metalMXFP4ExpertsMSL
		if variant.old != "" {
			if strings.Count(source, variant.old) != 1 {
				t.Fatal("mutant site not unique")
			}
			source = strings.Replace(source, variant.old, variant.new, 1)
		}
		q, err := newMetalMXFP4Experts(dev, source)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("variant=%s source_sha256=%x", variant.name, sha256.Sum256([]byte(source)))
		for _, slotted := range []bool{false, true} {
			if variant.name == "zero_scale_mutant" {
				break
			}
			input := x
			if !slotted {
				input, err = x.Slice(0, k)
				if err != nil {
					t.Fatal(err)
				}
			}
			batch, err := newMetalBatch(ops)
			if err != nil {
				t.Fatal(err)
			}
			if err := batch.EncodeMatvecMXFP4Selected(q, tensor, ids, input, out, slotted); err != nil {
				t.Fatal(err)
			}
			if err := batch.Wait(context.Background()); err != nil {
				t.Fatal(err)
			}
			violations := 0
			for slot, id := range ids.HostSlice() {
				for row := 0; row < m; row++ {
					lo := (int(id)*m + row) * k / 32 * 17
					in := x.HostSlice()[:k]
					if slotted {
						in = x.HostSlice()[slot*k : (slot+1)*k]
					}
					want := metalMXFP4Reference(raw[lo:lo+k/32*17], in)
					v := float64(out.HostSlice()[slot*m+row])
					if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v-want) > 1e-4+1e-4*math.Abs(want) {
						violations++
					}
				}
			}
			t.Logf("variant=%s slotted=%t violations=%d/%d", variant.name, slotted, violations, len(out.HostSlice()))
			if variant.old == "" && violations != 0 {
				t.Fatalf("baseline violations=%d", violations)
			}
			if variant.old != "" && violations == 0 {
				t.Fatal("compiled same-site mutant survived oracle")
			}
		}
		if variant.old == "" || variant.name == "zero_scale_mutant" {
			zeroWeights := metalmem.AllocIn[byte](arena, "zero-scale", 256*17)
			for i := 1; i < 17; i++ {
				zeroWeights.HostSlice()[255*17+i] = 0x22
			}
			zeroIDs := metalmem.AllocIn[int32](arena, "zero-ids", 1)
			zeroIDs.HostSlice()[0] = 255
			zeroInput := metalmem.AllocIn[float32](arena, "zero-input", 32)
			for i := range zeroInput.HostSlice() {
				zeroInput.HostSlice()[i] = 65504
			}
			zeroOut := metalmem.AllocIn[float32](arena, "zero-output", 1)
			zeroTensor := boundMetalTensor{name: "mxfp4-zero", typeID: typeMXFP4, dimensions: []uint64{32, 1, 256}, data: zeroWeights}
			batch, err := newMetalBatch(ops)
			if err != nil {
				t.Fatal(err)
			}
			if err := batch.EncodeMatvecMXFP4Selected(q, zeroTensor, zeroIDs, zeroInput, zeroOut, false); err != nil {
				t.Fatal(err)
			}
			if err := batch.Wait(context.Background()); err != nil {
				t.Fatal(err)
			}
			// Pinned 11e8cad full selected F32 kernel returned zero for
			// this fixture under default/Safe and NSG4/8 (96/96 values).
			// This device parity oracle is separate from the unchanged
			// mathematical CPU decoder's nonzero result, 0x0a7fe000.
			got := math.Float32bits(zeroOut.HostSlice()[0])
			t.Logf("variant=%s upstream-zero values=1 got=%08x want=00000000 mathematical=0a7fe000", variant.name, got)
			if variant.old == "" && got != 0 {
				t.Fatalf("upstream E8M0 zero parity bits=%08x want=00000000", got)
			}
			if variant.name == "zero_scale_mutant" && (got == 0 || math.IsNaN(float64(zeroOut.HostSlice()[0])) || math.IsInf(float64(zeroOut.HostSlice()[0]), 0)) {
				t.Fatalf("zero-scale mutant did not produce finite nonzero value: %08x", got)
			}
		}
		q.Close()
	}
}
