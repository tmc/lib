//go:build linux

package main

import (
	"bytes"
	"encoding/binary"
	"math"
	"math/bits"
	"math/rand"
	"testing"

	"github.com/tmc/lib/lm/quant"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// Registered before device execution, matching the synthetic f32 contract in
// manual-gates.md. These tests do not claim default CUDA Q8_K-route parity.
const cudaExpertATol = 1e-4
const cudaExpertRTol = 1e-4

func TestCUDAQualificationExperts(t *testing.T) {
	for _, tc := range []struct {
		name       string
		cols, rows int
	}{
		{"one_block", 256, 5},
		{"multiple_blocks", 768, 9},
		{"production_input_width", 4096, 17},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := cudaSynthFoundation(t)
			const experts = 9
			ids := []int32{8, 2, 5, 0, 2, 7}
			dIDs, err := cudamem.Alloc[int32](r.device, len(ids))
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := dIDs.Free(); err != nil {
					t.Error(err)
				}
			})
			if err := dIDs.CopyFromHost(ids); err != nil {
				t.Fatal(err)
			}
			input := make([]float32, tc.cols)
			slotInput := make([]float32, tc.cols*len(ids))
			for i := range input {
				input[i] = float32((i*37+i/17)%257-128) / 128
			}
			for i := range slotInput {
				slotInput[i] = float32((i*13+i/tc.cols*31+i/19)%251-125) / 128
			}
			dInput := cudaSynthUpload(t, r, input)
			dSlotInput := cudaSynthUpload(t, r, slotInput)
			g := cudaExpertFixture(16, tc.cols, tc.rows, experts, 17)
			u := cudaExpertFixture(16, tc.cols, tc.rows, experts, 29)
			q := cudaExpertFixture(10, tc.cols, tc.rows, experts, 43)
			dims := []uint64{uint64(tc.cols), uint64(tc.rows), experts}
			gate := cudaSynthTensor(t, r, "IQ2 gate", 16, dims, g)
			up := cudaSynthTensor(t, r, "IQ2 up", 16, dims, u)
			down := cudaSynthTensor(t, r, "Q2 down", 10, dims, q)
			gateOut := cudaTestAlloc(t, r, tc.rows*len(ids))
			upOut := cudaTestAlloc(t, r, tc.rows*len(ids))
			downOut := cudaTestAlloc(t, r, tc.rows*len(ids))
			poison := make([]float32, tc.rows*len(ids))
			for i := range poison {
				poison[i] = float32(math.NaN())
			}
			for _, dst := range []cudamem.Slice[float32]{gateOut, upOut, downOut} {
				if err := dst.CopyFromHost(poison); err != nil {
					t.Fatal(err)
				}
			}
			cudaQualificationMeasure(t, r, func() error { return r.encodeMatvecIQ2Selected(gate, up, dIDs, dInput, gateOut, upOut) })
			cudaQualificationMeasure(t, r, func() error { return r.encodeMatvecQ2KSelected(down, dIDs, dSlotInput, downOut) })
			for _, output := range []struct {
				name    string
				raw     []byte
				typeID  uint32
				input   []float32
				slotted bool
				device  cudamem.Slice[float32]
			}{
				{"gate", g, 16, input, false, gateOut}, {"up", u, 16, input, false, upOut}, {"down", q, 10, slotInput, true, downOut},
			} {
				want := cudaExpertReference(t, output.raw, output.typeID, tc.cols, tc.rows, ids, output.input, output.slotted)
				got := cudaTestRead(t, output.device)
				if err := cudaQualificationClose(got, want, cudaExpertATol, cudaExpertRTol); err != nil {
					t.Fatalf("%s: %v", output.name, err)
				}
				var maxAbs float64
				for i := range got {
					maxAbs = max(maxAbs, math.Abs(float64(got[i])-float64(want[i])))
				}
				t.Logf("%s: %d physical outputs, max absolute error %g, atol=%g rtol=%g", output.name, len(got), maxAbs, cudaExpertATol, cudaExpertRTol)
				wrongIDs := append([]int32(nil), ids...)
				for i := range wrongIDs {
					wrongIDs[i] = int32(i)
				}
				wrong := cudaExpertReference(t, output.raw, output.typeID, tc.cols, tc.rows, wrongIDs, output.input, output.slotted)
				if err := cudaQualificationClose(wrong, want, cudaExpertATol, cudaExpertRTol); err == nil {
					t.Fatalf("%s fixture accepts wrong experts", output.name)
				}
				if err := cudaQualificationClose(make([]float32, len(want)), want, cudaExpertATol, cudaExpertRTol); err == nil {
					t.Fatalf("%s fixture accepts zero output", output.name)
				}
				if output.slotted {
					wrong := cudaExpertReference(t, output.raw, output.typeID, tc.cols, tc.rows, ids, output.input[:tc.cols], false)
					if err := cudaQualificationClose(wrong, want, cudaExpertATol, cudaExpertRTol); err == nil {
						t.Fatal("fixture accepts broadcast down input")
					}
				}
			}
			for _, bad := range []struct {
				name   string
				tensor cudaModelTensor
			}{
				{"type", func() cudaModelTensor { v := gate; v.typeID = 0; return v }()},
				{"rank", func() cudaModelTensor { v := gate; v.dimensions = []uint64{uint64(tc.cols), uint64(tc.rows)}; return v }()},
				{"partial block", func() cudaModelTensor {
					v := gate
					v.dimensions = []uint64{uint64(tc.cols - 1), uint64(tc.rows), experts}
					return v
				}()},
				{"zero experts", func() cudaModelTensor {
					v := gate
					v.dimensions = []uint64{uint64(tc.cols), uint64(tc.rows), 0}
					return v
				}()},
				{"extent", func() cudaModelTensor { v := gate; v.bytes--; return v }()},
			} {
				if err := r.encodeMatvecIQ2Selected(bad.tensor, up, dIDs, dInput, gateOut, upOut); err == nil {
					t.Fatalf("accepted bad IQ2 %s", bad.name)
				}
				v := bad.tensor
				if bad.name != "type" {
					v.typeID = 10
				}
				if bad.name != "extent" {
					v.bytes = down.bytes
				} else {
					v.bytes = down.bytes - 1
				}
				if err := r.encodeMatvecQ2KSelected(v, dIDs, dSlotInput, downOut); err == nil {
					t.Fatalf("accepted bad Q2 %s", bad.name)
				}
			}
			for label, err := range map[string]error{
				"IQ2 input":       r.encodeMatvecIQ2Selected(gate, up, dIDs, dInput.Slice(0, tc.cols-1), gateOut, upOut),
				"IQ2 gate output": r.encodeMatvecIQ2Selected(gate, up, dIDs, dInput, gateOut.Slice(0, gateOut.Len()-1), upOut),
				"IQ2 up output":   r.encodeMatvecIQ2Selected(gate, up, dIDs, dInput, gateOut, upOut.Slice(0, upOut.Len()-1)),
				"Q2 input":        r.encodeMatvecQ2KSelected(down, dIDs, dSlotInput.Slice(0, dSlotInput.Len()-1), downOut),
				"Q2 output":       r.encodeMatvecQ2KSelected(down, dIDs, dSlotInput, downOut.Slice(0, downOut.Len()-1)),
			} {
				if err == nil {
					t.Fatalf("accepted bad %s", label)
				}
			}
			cudaExpertTestQ8Route(t, r, tc.cols, tc.rows, ids, dIDs, input, slotInput, gate, up, down, g, u, q)
			// Invalid device IDs must fail closed, including when outputs already hold
			// a valid result from the preceding launch. They must not leave stale values.
			invalidIDs := append([]int32(nil), ids...)
			invalidIDs[0], invalidIDs[1] = -1, experts
			if err := dIDs.CopyFromHost(invalidIDs); err != nil {
				t.Fatal(err)
			}
			cudaQualificationMeasure(t, r, func() error { return r.encodeMatvecIQ2Selected(gate, up, dIDs, dInput, gateOut, upOut) })
			cudaQualificationMeasure(t, r, func() error { return r.encodeMatvecQ2KSelected(down, dIDs, dSlotInput, downOut) })
			for _, out := range []cudamem.Slice[float32]{gateOut, upOut, downOut} {
				got := cudaTestRead(t, out)
				for i := range 2 * tc.rows {
					if !math.IsNaN(float64(got[i])) {
						t.Fatalf("invalid ID output[%d]=%g, want NaN", i, got[i])
					}
				}
			}
		})
	}
}

// cudaExpertFixture constructs raw, finite blocks without using a quantizer.
// Every expert, row, and block has a distinct pattern; IQ2 varies codebook
// and sign indices, while Q2 varies scales, minima, and quant bit planes.
func cudaExpertFixture(typeID uint32, cols, rows, experts int, seed int64) []byte {
	blockBytes := 66
	if typeID == 10 {
		blockBytes = 84
	}
	data := make([]byte, experts*rows*(cols/256)*blockBytes)
	rnd := rand.New(rand.NewSource(seed))
	for offset := 0; offset < len(data); offset += blockBytes {
		p := data[offset : offset+blockBytes]
		for i := range p {
			p[i] = byte(rnd.Intn(256))
		}
		if typeID == 16 {
			binary.LittleEndian.PutUint16(p, 0x1c00+uint16(rnd.Intn(1024)))
		} else {
			binary.LittleEndian.PutUint16(p[80:], 0x1c00+uint16(rnd.Intn(1024)))
			binary.LittleEndian.PutUint16(p[82:], 0x1800+uint16(rnd.Intn(1024)))
		}
	}
	return data
}

// cudaExpertReference uses the independently golden-tested CPU dequantizers and
// float64 dot products, rather than the CUDA decoder or its reduction tree.
func cudaExpertReference(t *testing.T, raw []byte, typeID uint32, cols, rows int, ids []int32, input []float32, slotted bool) []float32 {
	t.Helper()
	blockBytes := 66
	dequant := quant.DequantizeIQ2_XXS
	if typeID == 10 {
		blockBytes = 84
		dequant = quant.DequantizeQ2_K
	}
	rowBytes := cols / 256 * blockBytes
	out := make([]float32, len(ids)*rows)
	for slot, id := range ids {
		x := input[:cols]
		if slotted {
			x = input[slot*cols : (slot+1)*cols]
		}
		for row := range rows {
			offset := (int(id)*rows + row) * rowBytes
			weights, err := dequant(raw[offset:offset+rowBytes], cols)
			if err != nil {
				t.Fatal(err)
			}
			var sum float64
			for col, w := range weights {
				sum += float64(w) * float64(x[col])
			}
			out[slot*rows+row] = float32(sum)
		}
	}
	return out
}

func cudaExpertAllocBytes(t *testing.T, r *cudaFoundation, n int) cudamem.Slice[byte] {
	t.Helper()
	out, err := cudamem.Alloc[byte](r.device, n)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := out.Free(); err != nil {
			t.Error(err)
		}
	})
	return out
}

func cudaExpertCheckBytes(t *testing.T, device cudamem.Slice[byte], want []byte) {
	t.Helper()
	got := make([]byte, device.Len())
	if err := device.CopyToHost(got); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		for i := range min(len(got), len(want)) {
			if got[i] != want[i] {
				t.Fatalf("Q8_K byte %d (block %d offset %d): got %#x want %#x", i, i/292, i%292, got[i], want[i])
			}
		}
		t.Fatalf("Q8_K length %d want %d", len(got), len(want))
	}
	t.Logf("Q8_K %d bytes match exactly (scale, quants, block sums)", len(got))
}

// cudaExpertQ8Reference implements the pinned CUDA quantizer contract from
// ds4_cuda.cu:19883. Enumerating bit-reversed indices selects its reduction-tree
// tie winner without reproducing the device tree. Arithmetic explicitly rounds
// to float32 before round-to-even; no production quantizer is called.
func cudaExpertQ8Reference(in []float32) []byte {
	out := make([]byte, len(in)/256*292)
	for block := range len(in) / 256 {
		x := in[block*256 : (block+1)*256]
		p := out[block*292 : (block+1)*292]
		var largest float32
		for rank := range 256 {
			v := x[bits.Reverse8(uint8(rank))]
			if math.Abs(float64(v)) > math.Abs(float64(largest)) {
				largest = v
			}
		}
		if largest == 0 {
			continue
		}
		inverse := float32(-127) / largest
		binary.LittleEndian.PutUint32(p, math.Float32bits(float32(1)/inverse))
		var sums [16]int16
		for i, v := range x {
			scaled := float32(inverse * v)
			q := int(math.RoundToEven(float64(scaled)))
			q = min(127, max(-128, q))
			p[4+i] = byte(int8(q))
			sums[i/16] += int16(q)
		}
		for i, sum := range sums {
			binary.LittleEndian.PutUint16(p[260+i*2:], uint16(sum))
		}
	}
	return out
}

func cudaExpertQ8Decode(raw []byte) []float32 {
	out := make([]float32, len(raw)/292*256)
	for b := range len(raw) / 292 {
		p := raw[b*292 : (b+1)*292]
		d := math.Float32frombits(binary.LittleEndian.Uint32(p))
		for i := range 256 {
			out[b*256+i] = d * float32(int8(p[4+i]))
		}
	}
	return out
}

func TestCUDAQualificationQ8K(t *testing.T) {
	r := cudaSynthFoundation(t)
	for _, name := range []string{"zero", "positive_max", "negative_max", "opposite_tie", "reversed_tie", "fractional_scale"} {
		t.Run(name, func(t *testing.T) {
			// Half-integer values with maximum magnitude 127 give exact +/-1
			// inverse scale, exposing nearest-even rounding without division noise.
			x := make([]float32, 512)
			if name != "zero" {
				for i := range x {
					x[i] = float32((i*17)%201-100) / 2
				}
				for b := range 2 {
					switch name {
					case "positive_max":
						x[b*256] = 127
					case "negative_max":
						x[b*256] = -127
					case "opposite_tie":
						x[b*256+1] = -127
						x[b*256+128] = 127
					case "reversed_tie":
						x[b*256+1] = 127
						x[b*256+128] = -127
					case "fractional_scale":
						x[b*256+37] = 131.375
					}
				}
			}
			want := cudaExpertQ8Reference(x)
			if name == "opposite_tie" && math.Float32frombits(binary.LittleEndian.Uint32(want)) != -1 {
				t.Fatal("reference lost bit-reversed positive tie winner")
			}
			if name == "reversed_tie" && math.Float32frombits(binary.LittleEndian.Uint32(want)) != 1 {
				t.Fatal("reference lost bit-reversed negative tie winner")
			}
			if name == "positive_max" && (int8(want[4+3]) != 24 || int8(want[4+5]) != 8) {
				t.Fatal("reference lost nearest-even half-integer rounding")
			}
			in := cudaSynthUpload(t, r, x)
			out := cudaExpertAllocBytes(t, r, len(want))
			if err := out.CopyFromHost(bytes.Repeat([]byte{0xa5}, len(want))); err != nil {
				t.Fatal(err)
			}
			cudaQualificationMeasure(t, r, func() error { return r.encodeQuantQ8K(in, out, 256) })
			cudaExpertCheckBytes(t, out, want)
		})
	}
}

func cudaExpertTestQ8Route(t *testing.T, r *cudaFoundation, cols, rows int, ids []int32, dIDs cudamem.Slice[int32], input, slotInput []float32, gate, up, down cudaModelTensor, g, u, q []byte) {
	t.Helper()
	dInput := cudaSynthUpload(t, r, input)
	dSlotInput := cudaSynthUpload(t, r, slotInput)
	qInput := cudaExpertAllocBytes(t, r, len(input)/256*292)
	qSlotInput := cudaExpertAllocBytes(t, r, len(slotInput)/256*292)
	cudaQualificationMeasure(t, r, func() error { return r.encodeQuantQ8K(dInput, qInput, cols) })
	cudaQualificationMeasure(t, r, func() error { return r.encodeQuantQ8K(dSlotInput, qSlotInput, cols) })
	wantQ := cudaExpertQ8Reference(input)
	wantSlotQ := cudaExpertQ8Reference(slotInput)
	cudaExpertCheckBytes(t, qInput, wantQ)
	cudaExpertCheckBytes(t, qSlotInput, wantSlotQ)
	gateOut := cudaTestAlloc(t, r, rows*len(ids))
	upOut := cudaTestAlloc(t, r, rows*len(ids))
	downOut := cudaTestAlloc(t, r, rows*len(ids))
	poison := make([]float32, rows*len(ids))
	for i := range poison {
		poison[i] = float32(math.NaN())
	}
	for _, dst := range []cudamem.Slice[float32]{gateOut, upOut, downOut} {
		if err := dst.CopyFromHost(poison); err != nil {
			t.Fatal(err)
		}
	}
	cudaQualificationMeasure(t, r, func() error { return r.encodeMatvecIQ2SelectedQ8K(gate, up, dIDs, qInput, gateOut, upOut) })
	cudaQualificationMeasure(t, r, func() error { return r.encodeMatvecQ2KSelectedQ8K(down, dIDs, qSlotInput, downOut) })
	for _, output := range []struct {
		name      string
		raw       []byte
		typeID    uint32
		quantized []byte
		slotted   bool
		device    cudamem.Slice[float32]
	}{
		{"Q8_K gate", g, 16, wantQ, false, gateOut}, {"Q8_K up", u, 16, wantQ, false, upOut}, {"Q8_K down", q, 10, wantSlotQ, true, downOut},
	} {
		decoded := cudaExpertQ8Decode(output.quantized)
		want := cudaExpertReference(t, output.raw, output.typeID, cols, rows, ids, decoded, output.slotted)
		got := cudaTestRead(t, output.device)
		if err := cudaQualificationClose(got, want, cudaExpertATol, cudaExpertRTol); err != nil {
			t.Fatalf("%s: %v", output.name, err)
		}
		var maxAbs float64
		for i := range got {
			maxAbs = max(maxAbs, math.Abs(float64(got[i])-float64(want[i])))
		}
		t.Logf("%s: %d physical outputs, max absolute error %g", output.name, len(got), maxAbs)
		original := input
		if output.slotted {
			original = slotInput
		}
		unquantized := cudaExpertReference(t, output.raw, output.typeID, cols, rows, ids, original, output.slotted)
		if err := cudaQualificationClose(unquantized, want, cudaExpertATol, cudaExpertRTol); err == nil {
			t.Fatalf("%s fixture cannot distinguish omitted Q8_K quantization", output.name)
		}
		wrongIDs := append([]int32(nil), ids...)
		for i := range wrongIDs {
			wrongIDs[i] = int32(i)
		}
		wrong := cudaExpertReference(t, output.raw, output.typeID, cols, rows, wrongIDs, decoded, output.slotted)
		if err := cudaQualificationClose(wrong, want, cudaExpertATol, cudaExpertRTol); err == nil {
			t.Fatalf("%s accepts wrong experts", output.name)
		}
		if output.slotted {
			wrong := cudaExpertReference(t, output.raw, output.typeID, cols, rows, ids, decoded[:cols], false)
			if err := cudaQualificationClose(wrong, want, cudaExpertATol, cudaExpertRTol); err == nil {
				t.Fatal("Q8_K down accepts broadcast input")
			}
		}
	}
	for label, err := range map[string]error{
		"quant partial block": r.encodeQuantQ8K(dInput, qInput, cols-1),
		"quant output":        r.encodeQuantQ8K(dInput, qInput.Slice(0, qInput.Len()-1), cols),
		"IQ2 Q8_K input":      r.encodeMatvecIQ2SelectedQ8K(gate, up, dIDs, qInput.Slice(0, qInput.Len()-1), gateOut, upOut),
		"Q2 Q8_K input":       r.encodeMatvecQ2KSelectedQ8K(down, dIDs, qSlotInput.Slice(0, qSlotInput.Len()-1), downOut),
	} {
		if err == nil {
			t.Fatalf("accepted bad %s", label)
		}
	}
}
