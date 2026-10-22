//go:build linux

package main

import (
	"encoding/binary"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamem"
)

func TestCUDAMXFP4SelectedValueAndMutants(t *testing.T) {
	if os.Getenv("DS4_CUDA_MXFP4_SELECTED_GATE") != "1" {
		t.Skip("opt-in locked CUDA selected MXFP4 gate required")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("gpu-lock required")
	}
	const k, m, e = 64, 5, 256
	ids := []int32{255, 2, 7, 0, 2, 17}
	weights := make([]byte, e*m*k/32*17)
	for b := 0; b < len(weights)/17; b++ {
		p := weights[b*17:]
		p[0] = byte(125 + b%5)
		if b%19 == 0 {
			p[0] = 0
		}
		for i := 1; i < 17; i++ {
			p[i] = byte(b*29 + i*37)
		}
	}
	x := make([]byte, len(ids)*k/32*36)
	for b := 0; b < len(x)/36; b++ {
		binary.LittleEndian.PutUint16(x[b*36:], 0x3800)
		for i := 0; i < 32; i++ {
			x[b*36+4+i] = byte(int8((b*17+i*13)%251 - 125))
		}
	}
	variants := []struct{ name, old, new string }{
		{"baseline", "", ""},
		{"scale_mutant", "sum += scale*code[nibble]", "sum += 2*scale*code[nibble]"},
		{"nibble_mutant", "unsigned nibble=(p[1+(lane&15)] >> (lane>=16 ? 4 : 0)) & 15;", "unsigned nibble=(p[1+(lane&15)] >> (lane>=16 ? 0 : 4)) & 15;"},
	}
	for _, variant := range variants {
		if !t.Run(variant.name, func(t *testing.T) {
			r := cudaSynthFoundation(t)
			source := cudaMXFP4SelectedSource
			if variant.old != "" {
				if strings.Count(source, variant.old) != 1 {
					t.Fatal("mutant site not unique")
				}
				source = strings.Replace(source, variant.old, variant.new, 1)
			}
			consumer, err := compileCUDAMXFP4Selected(r, source)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := consumer.Close(); err != nil {
					t.Error(err)
				}
			})
			tensor := cudaSynthTensor(t, r, "mxfp4-fixture", typeMXFP4, []uint64{k, m, e}, weights)
			dIDs, err := cudamem.Alloc[int32](r.device, len(ids))
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = dIDs.Free() })
			if err := dIDs.CopyFromHost(ids); err != nil {
				t.Fatal(err)
			}
			for _, slotted := range []bool{false, true} {
				input := x
				if !slotted {
					input = x[:k/32*36]
				}
				dInput, err := cudamem.Alloc[byte](r.device, len(input))
				if err != nil {
					t.Fatal(err)
				}
				defer dInput.Free()
				if err := dInput.CopyFromHost(input); err != nil {
					t.Fatal(err)
				}
				output := cudaSynthUpload(t, r, make([]float32, m*len(ids)))
				if err := consumer.Encode(tensor, dIDs, dInput, output, slotted); err != nil {
					t.Fatal(err)
				}
				if err := r.device.WithContext(func() error { return syncStream(r.stream()) }); err != nil {
					t.Fatal(err)
				}
				got := make([]float32, output.Len())
				if err := output.CopyToHost(got); err != nil {
					t.Fatal(err)
				}
				violations := 0
				for slot, id := range ids {
					for row := 0; row < m; row++ {
						lo := (int(id)*m + row) * k / 32 * 17
						in := input[:k/32*36]
						if slotted {
							in = input[slot*k/32*36 : (slot+1)*k/32*36]
						}
						want := mxfp4Q81DotReference(weights[lo:lo+k/32*17], in)
						v := float64(got[slot*m+row])
						if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v-want) > 1e-4+1e-4*math.Abs(want) {
							violations++
						}
					}
				}
				t.Logf("slotted=%t violations=%d/%d", slotted, violations, len(got))
				if variant.old == "" && violations != 0 {
					t.Fatalf("baseline violations=%d", violations)
				}
				if variant.old != "" && violations == 0 {
					t.Fatal("same-site mutant survived oracle")
				}
			}
			if variant.old == "" {
				testCUDAMXFP4ZeroScale(t, r, consumer)
			}
		}) {
			return
		}
	}
}

func testCUDAMXFP4ZeroScale(t *testing.T, r *cudaFoundation, consumer *cudaMXFP4Selected) {
	// All products and the balanced 32-lane sum are exact dyadic float32
	// values. Bit equality is the preregistered golden rule, not abs1e-4.
	raw := make([]byte, 256*17)
	for i := 1; i < 17; i++ {
		raw[255*17+i] = 0x22
	}
	tensor := cudaSynthTensor(t, r, "mxfp4-zero", typeMXFP4, []uint64{32, 1, 256}, raw)
	ids, err := cudamem.Alloc[int32](r.device, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer ids.Free()
	if err := ids.CopyFromHost([]int32{255}); err != nil {
		t.Fatal(err)
	}
	input := make([]byte, 36)
	binary.LittleEndian.PutUint16(input, 0x7bff)
	for i := 4; i < 36; i++ {
		input[i] = 1
	}
	x, err := cudamem.Alloc[byte](r.device, 36)
	if err != nil {
		t.Fatal(err)
	}
	defer x.Free()
	if err := x.CopyFromHost(input); err != nil {
		t.Fatal(err)
	}
	out := cudaSynthUpload(t, r, []float32{float32(math.NaN())})
	if err := consumer.Encode(tensor, ids, x, out, false); err != nil {
		t.Fatal(err)
	}
	if err := r.device.WithContext(func() error { return syncStream(r.stream()) }); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, 1)
	if err := out.CopyToHost(got); err != nil {
		t.Fatal(err)
	}
	want := float32(math.Ldexp(65504*32, -127))
	t.Logf("E8M0 zero golden values=1 got=%08x want=%08x", math.Float32bits(got[0]), math.Float32bits(want))
	if math.Float32bits(got[0]) != math.Float32bits(want) {
		t.Fatalf("E8M0 zero golden bits=%08x want=%08x", math.Float32bits(got[0]), math.Float32bits(want))
	}
}
