//go:build linux

package main

import (
	"encoding/binary"
	"math"
	"strconv"
	"testing"

	"github.com/tmc/lib/lm/quant"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// The aligned IQ2 source truncates each scaled integer subgroup dot before
// floating conversion. This is a different contract from mathematical weight
// dequantization. Source-contract comparisons retain the registered 1e-4 limits;
// the mathematical delta is reported separately, never hidden by a wider limit.
func TestCUDAQualificationExpertsQ81(t *testing.T) {
	for _, cols := range []int{256, 1024, 4096} {
		t.Run(strconv.Itoa(cols), func(t *testing.T) {
			r := cudaSynthFoundation(t)
			const rows, experts = 17, 9
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
			x, slotX := make([]float32, cols), make([]float32, cols*len(ids))
			for i := range x {
				x[i] = float32((i*37+i/17)%257-128) / 128
			}
			for i := range slotX {
				slotX[i] = float32((i*13+i/cols*31+i/19)%251-125) / 128
			}
			q, sq := cudaExpertQ81Upload(t, r, x, cols), cudaExpertQ81Upload(t, r, slotX, cols)
			rawQ, rawSQ := cudaQ81ActivationReference(x), cudaQ81ActivationReference(slotX)
			g, u, d := cudaExpertFixture(16, cols, rows, experts, 17), cudaExpertFixture(16, cols, rows, experts, 29), cudaExpertFixture(10, cols, rows, experts, 43)
			dims := []uint64{uint64(cols), rows, experts}
			gate := cudaSynthTensor(t, r, "Q81 IQ2 gate", 16, dims, g)
			up := cudaSynthTensor(t, r, "Q81 IQ2 up", 16, dims, u)
			down := cudaSynthTensor(t, r, "Q81 Q2 down", 10, dims, d)
			gout, uout, dout := cudaTestAlloc(t, r, rows*len(ids)), cudaTestAlloc(t, r, rows*len(ids)), cudaTestAlloc(t, r, rows*len(ids))
			poison := make([]float32, rows*len(ids))
			for i := range poison {
				poison[i] = float32(math.NaN())
			}
			for _, out := range []cudamem.Slice[float32]{gout, uout, dout} {
				if err := out.CopyFromHost(poison); err != nil {
					t.Fatal(err)
				}
			}
			cudaQualificationMeasure(t, r, func() error { return r.encodeMatvecIQ2SelectedQ81(gate, up, dIDs, q, gout, uout) })
			cudaQualificationMeasure(t, r, func() error { return r.encodeMatvecQ2KSelectedQ81(down, dIDs, sq, dout) })
			for _, o := range []struct {
				name                string
				weights, activation []byte
				typeID              uint32
				input               []float32
				slotted             bool
				device              cudamem.Slice[float32]
			}{
				{"gate", g, rawQ, 16, x, false, gout}, {"up", u, rawQ, 16, x, false, uout}, {"down", d, rawSQ, 10, slotX, true, dout},
			} {
				decoded := cudaExpertQ81Decode(o.activation)
				mathematical := cudaExpertReference(t, o.weights, o.typeID, cols, rows, ids, decoded, o.slotted)
				want := mathematical
				if o.typeID == 16 {
					want = cudaExpertIQ2Q81Reference(t, o.weights, o.activation, cols, rows, ids)
				}
				got := cudaTestRead(t, o.device)
				if err := cudaQualificationClose(got, want, cudaExpertATol, cudaExpertRTol); err != nil {
					t.Fatalf("Q81 %s: %v", o.name, err)
				}
				var maxAbs, mathDelta float64
				for i := range got {
					maxAbs = max(maxAbs, math.Abs(float64(got[i])-float64(want[i])))
					mathDelta = max(mathDelta, math.Abs(float64(want[i])-float64(mathematical[i])))
				}
				t.Logf("Q81 %s: max source-reference delta%g mathematical-contract delta%g; atol=%g rtol=%g", o.name, maxAbs, mathDelta, cudaExpertATol, cudaExpertRTol)
				wrongIDs := append([]int32(nil), ids...)
				for i := range wrongIDs {
					wrongIDs[i] = int32(i)
				}
				wrong := cudaExpertReference(t, o.weights, o.typeID, cols, rows, wrongIDs, decoded, o.slotted)
				if o.typeID == 16 {
					wrong = cudaExpertIQ2Q81Reference(t, o.weights, o.activation, cols, rows, wrongIDs)
				}
				if err := cudaQualificationClose(wrong, want, cudaExpertATol, cudaExpertRTol); err == nil {
					t.Fatalf("Q81 %s accepts wrong experts", o.name)
				}
				q8k := cudaExpertQ8Decode(cudaExpertQ8Reference(o.input))
				wrong = cudaExpertReference(t, o.weights, o.typeID, cols, rows, ids, q8k, o.slotted)
				if err := cudaQualificationClose(wrong, want, cudaExpertATol, cudaExpertRTol); err == nil {
					t.Fatalf("Q81 %s fixture cannot distinguish native Q8_K", o.name)
				}
			}
			before := cudaTestRead(t, dout)
			// The aligned Q2 minimum term uses sums of quantized codes, not the Q8_1
			// half(sum original input) field. Poison that field to detect accidental use.
			changed := append([]byte(nil), rawSQ...)
			for b := range len(changed) / 36 {
				binary.LittleEndian.PutUint16(changed[b*36+2:], 0x7e00)
			}
			if err := sq.CopyFromHost(changed); err != nil {
				t.Fatal(err)
			}
			cudaQualificationMeasure(t, r, func() error { return r.encodeMatvecQ2KSelectedQ81(down, dIDs, sq, dout) })
			if err := cudaQualificationClose(cudaTestRead(t, dout), before, 0, 0); err != nil {
				t.Fatalf("Q2 consumed unused half-sum field: %v", err)
			}
			for label, err := range map[string]error{
				"IQ2 input":  r.encodeMatvecIQ2SelectedQ81(gate, up, dIDs, q.Slice(0, q.Len()-1), gout, uout),
				"Q2 input":   r.encodeMatvecQ2KSelectedQ81(down, dIDs, sq.Slice(0, sq.Len()-1), dout),
				"IQ2 output": r.encodeMatvecIQ2SelectedQ81(gate, up, dIDs, q, gout.Slice(0, gout.Len()-1), uout),
				"Q2 output":  r.encodeMatvecQ2KSelectedQ81(down, dIDs, sq, dout.Slice(0, dout.Len()-1)),
			} {
				if err == nil {
					t.Fatalf("accepted short %s", label)
				}
			}
		})
	}
}

func cudaExpertQ81Upload(t *testing.T, r *cudaFoundation, x []float32, cols int) cudamem.Slice[byte] {
	t.Helper()
	in := cudaSynthUpload(t, r, x)
	out := cudaExpertAllocBytes(t, r, len(x)/32*36)
	cudaQualificationMeasure(t, r, func() error { return r.encodeQuantQ81(in, out, cols) })
	cudaQ81CheckBytes(t, out, cudaQ81ActivationReference(x))
	return out
}

func cudaExpertQ81Decode(raw []byte) []float32 {
	out := make([]float32, len(raw)/36*32)
	for b := range len(raw) / 36 {
		p := raw[b*36:]
		d := quant.Float16ToFloat32(binary.LittleEndian.Uint16(p))
		for i := range 32 {
			out[b*32+i] = d * float32(int8(p[4+i]))
		}
	}
	return out
}

// cudaExpertIQ2Q81Reference obtains signed codebook entries through the
// independently golden-tested dequantizer, then applies the pinned aligned
// integer division. It does not share the device's byte decoder or lane tree.
func cudaExpertIQ2Q81Reference(t *testing.T, raw, activation []byte, cols, rows int, ids []int32) []float32 {
	t.Helper()
	out := make([]float32, len(ids)*rows)
	rowBytes := cols / 256 * 66
	for slot, id := range ids {
		for row := range rows {
			p := raw[(int(id)*rows+row)*rowBytes:][:rowBytes]
			weights, err := quant.DequantizeIQ2_XXS(p, cols)
			if err != nil {
				t.Fatal(err)
			}
			var sum float64
			for part := range cols / 32 {
				block, sub := part/8, part%8
				d := quant.Float16ToFloat32(binary.LittleEndian.Uint16(p[block*66:]))
				aux := binary.LittleEndian.Uint32(p[block*66+2+sub*8+4:])
				ls := int64(2*(aux>>28) + 1)
				scale := float64(d) * float64(ls) / 8
				var dot int64
				if scale != 0 {
					for i := range 32 {
						grid := int64(math.Round(float64(weights[part*32+i]) / scale))
						dot += grid * int64(int8(activation[part*36+4+i]))
					}
				}
				d8 := quant.Float16ToFloat32(binary.LittleEndian.Uint16(activation[part*36:]))
				sum += float64(d) * float64(d8) * float64(dot*ls/8)
			}
			out[slot*rows+row] = float32(sum)
		}
	}
	return out
}

func TestCUDAQualificationIQ2Q81IntegerTruncation(t *testing.T) {
	r := cudaSynthFoundation(t)
	// Codebook index1 starts with43; all signs positive, scale nibble0.
	// A single quantized activation1 therefore contributes integer43/8=5,
	// whereas mathematical dequantization gives43/8=5.375.
	raw := make([]byte, 66)
	binary.LittleEndian.PutUint16(raw, 0x3c00)
	raw[2] = 1
	act := make([]byte, 8*36)
	binary.LittleEndian.PutUint16(act, 0x3c00)
	act[4] = 1
	tensor := cudaSynthTensor(t, r, "IQ2 truncation", 16, []uint64{256, 1, 1}, raw)
	input := cudaExpertAllocBytes(t, r, len(act))
	if err := input.CopyFromHost(act); err != nil {
		t.Fatal(err)
	}
	ids, err := cudamem.Alloc[int32](r.device, 1)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := ids.Free(); err != nil {
			t.Error(err)
		}
	})
	if err := ids.CopyFromHost([]int32{0}); err != nil {
		t.Fatal(err)
	}
	outA, outB := cudaTestAlloc(t, r, 1), cudaTestAlloc(t, r, 1)
	cudaQualificationMeasure(t, r, func() error { return r.encodeMatvecIQ2SelectedQ81(tensor, tensor, ids, input, outA, outB) })
	got := cudaTestRead(t, outA)
	if len(got) != 1 || got[0] != 5 {
		t.Fatalf("source integer43/8: got%v want[5]", got)
	}
	reference := cudaExpertIQ2Q81Reference(t, raw, act, 256, 1, []int32{0})
	if reference[0] != 5 {
		t.Fatalf("independent integer oracle got%g want5", reference[0])
	}
	mathematical := cudaExpertReference(t, raw, 16, 256, 1, []int32{0}, cudaExpertQ81Decode(act), false)
	if mathematical[0] != 5.375 {
		t.Fatalf("mathematical oracle got%g want5.375", mathematical[0])
	}
	act[4] = 255
	if err := input.CopyFromHost(act); err != nil {
		t.Fatal(err)
	}
	cudaQualificationMeasure(t, r, func() error { return r.encodeMatvecIQ2SelectedQ81(tensor, tensor, ids, input, outA, outB) })
	negative := cudaTestRead(t, outA)
	if negative[0] != -5 {
		t.Fatalf("source signed integer-43/8: got%g want-5", negative[0])
	}
	reference = cudaExpertIQ2Q81Reference(t, raw, act, 256, 1, []int32{0})
	if reference[0] != -5 {
		t.Fatalf("independent negative integer oracle got%g want-5", reference[0])
	}
	mathematical = cudaExpertReference(t, raw, 16, 256, 1, []int32{0}, cudaExpertQ81Decode(act), false)
	if mathematical[0] != -5.375 {
		t.Fatalf("negative mathematical oracle got%g want-5.375", mathematical[0])
	}
	t.Log("confirmed pinned aligned IQ2 truncation toward zero: device +/-5, mathematical +/-5.375")
}
