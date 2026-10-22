//go:build linux

package main

import (
	"context"
	"encoding/binary"
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamem"
)

// TestCUDAExpertSIMDMatchesScalar gates the Step 2 SIMD dequantization optimization.
//
// In ds4_iq2_part_q8_1, SIMD byte instructions (__vcmpne4, __vsub4) apply the 8-bit signs
// to the grid lookups, and __dp4a computes the 32-element integer dot product against the
// signed Q8_1 activation bytes. Because ds4_cuda_iq2xxs_grid entries are bounded by 43 and
// activations are int8, the entire inner sum is exact integer accumulation without overflow.
// Integer addition is associative, so the SIMD formulation must match the scalar loop
// exactly bit-for-bit (math.Float32bits).
//
// In ds4_q2_window_q8_1, the 2-bit slice extraction (v >> (2*i)) & 0x03030303 is computed
// via __dp4a against the packed 4-byte activations, and minimum is factored as
// (scale>>4) * __dp4a(0x01010101, q_packed, 0).
//
// Coverage includes:
//  1. Production shape: cols=2048, rows=2048, slots=6, experts=256 (cudaExpertWarps=2).
//  2. Unaligned tail shape: cols=2048, rows=2047, slots=5, experts=256, where rows*slots is odd
//     and not an exact multiple of cudaExpertWarps (2), exercising the tail return guard.
//  3. Invalid expert sentinels: negative IDs and IDs >= experts return 0x7fffffff bit pattern.
func TestCUDAExpertSIMDMatchesScalar(t *testing.T) {
	r := cudaSynthFoundation(t)

	testCases := []struct {
		name    string
		cols    int
		rows    int
		experts int
		ids     []int32
	}{
		{
			name:    "ProductionShape",
			cols:    2048,
			rows:    2048,
			experts: 256,
			ids:     []int32{0, -1, 15, 255, 256, 128}, // 6 slots, including negative (-1) and out-of-bounds (256) sentinels
		},
		{
			name:    "OddTailShape",
			cols:    2048,
			rows:    2047,
			experts: 256,
			ids:     []int32{0, -42, 15, 256, 128}, // 5 slots; rows*slots = 10235 (odd, exercises tail warp return)
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			cols, rows, experts, ids := tc.cols, tc.rows, tc.experts, tc.ids
			slots := len(ids)

			dIDs, err := cudamem.Alloc[int32](r.device, len(ids))
			if err != nil {
				t.Fatal(err)
			}
			defer dIDs.Free()
			if err := dIDs.CopyFromHost(ids); err != nil {
				t.Fatal(err)
			}

			// 1. Gate & Up: IQ2_XXS
			t.Run("IQ2_XXS_Pair", func(t *testing.T) {
				gData := cudaExpertFastFixture(16, cols, rows, experts, 101)
				uData := cudaExpertFastFixture(16, cols, rows, experts, 202)
				gTensor := cudaSynthTensor(t, r, "gate.weight", 16, []uint64{uint64(cols), uint64(rows), uint64(experts)}, gData)
				uTensor := cudaSynthTensor(t, r, "up.weight", 16, []uint64{uint64(cols), uint64(rows), uint64(experts)}, uData)

				inFloats := make([]float32, cols)
				for i := range inFloats {
					inFloats[i] = float32(math.Sin(float64(i)*0.41)) * 2.5
				}
				dInFloats := cudaSynthUpload(t, r, inFloats)
				defer dInFloats.Free()

				q81In, err := cudamem.Alloc[byte](r.device, cols/32*36)
				if err != nil {
					t.Fatal(err)
				}
				defer q81In.Free()
				if err := r.encodeQuantQ81(dInFloats, q81In, cols); err != nil {
					t.Fatal(err)
				}

				runPair := func(scalar bool) ([]float32, []float32) {
					gateOut, err := cudamem.Alloc[float32](r.device, slots*rows)
					if err != nil {
						t.Fatal(err)
					}
					defer gateOut.Free()
					upOut, err := cudamem.Alloc[float32](r.device, slots*rows)
					if err != nil {
						t.Fatal(err)
					}
					defer upOut.Free()

					poison := make([]float32, slots*rows)
					for i := range poison {
						poison[i] = math.Float32frombits(0x7fc00000) // NaN poison
					}
					_ = gateOut.CopyFromHost(poison)
					_ = upOut.CopyFromHost(poison)

					if scalar {
						if err := r.encodeMatvecIQ2SelectedQ81Scalar(gTensor, uTensor, dIDs, q81In, gateOut, upOut); err != nil {
							t.Fatal(err)
						}
					} else {
						if err := r.encodeMatvecIQ2SelectedQ81(gTensor, uTensor, dIDs, q81In, gateOut, upOut); err != nil {
							t.Fatal(err)
						}
					}
					if err := r.device.DefaultStream().Synchronize(context.Background()); err != nil {
						t.Fatal(err)
					}
					gHost := make([]float32, slots*rows)
					uHost := make([]float32, slots*rows)
					_ = gateOut.CopyToHost(gHost)
					_ = upOut.CopyToHost(uHost)
					return gHost, uHost
				}

				simdG, simdU := runPair(false)
				scalG, scalU := runPair(true)

				for s, id := range ids {
					isSentinel := id < 0 || int(id) >= experts
					for row := 0; row < rows; row++ {
						idx := s*rows + row
						if isSentinel {
							if math.Float32bits(simdG[idx]) != 0x7fffffff {
								t.Fatalf("slot %d (expert %d) row %d: expected sentinel 0x7fffffff, got SIMD gate %v (%#08x)",
									s, id, row, simdG[idx], math.Float32bits(simdG[idx]))
							}
							if math.Float32bits(simdU[idx]) != 0x7fffffff {
								t.Fatalf("slot %d (expert %d) row %d: expected sentinel 0x7fffffff, got SIMD up %v (%#08x)",
									s, id, row, simdU[idx], math.Float32bits(simdU[idx]))
							}
						} else {
							if math.IsNaN(float64(simdG[idx])) {
								t.Fatalf("slot %d row %d: NaN in SIMD gate output", s, row)
							}
						}
						if math.Float32bits(simdG[idx]) != math.Float32bits(scalG[idx]) {
							t.Fatalf("slot %d (expert %d) gate row %d: SIMD %v (%#08x) != Scalar %v (%#08x)",
								s, id, row, simdG[idx], math.Float32bits(simdG[idx]), scalG[idx], math.Float32bits(scalG[idx]))
						}
						if math.Float32bits(simdU[idx]) != math.Float32bits(scalU[idx]) {
							t.Fatalf("slot %d (expert %d) up row %d: SIMD %v (%#08x) != Scalar %v (%#08x)",
								s, id, row, simdU[idx], math.Float32bits(simdU[idx]), scalU[idx], math.Float32bits(scalU[idx]))
						}
					}
				}
			})

			// 2. Down: Q2_K
			t.Run("Q2_K_Selected", func(t *testing.T) {
				downData := cudaExpertFastFixture(10, cols, rows, experts, 303)
				downTensor := cudaSynthTensor(t, r, "down.weight", 10, []uint64{uint64(cols), uint64(rows), uint64(experts)}, downData)

				slotFloats := make([]float32, cols*slots)
				for i := range slotFloats {
					slotFloats[i] = float32(math.Cos(float64(i)*0.29)) * 1.8
				}
				dSlotFloats := cudaSynthUpload(t, r, slotFloats)
				defer dSlotFloats.Free()

				q81In, err := cudamem.Alloc[byte](r.device, cols/32*36*slots)
				if err != nil {
					t.Fatal(err)
				}
				defer q81In.Free()
				for s := 0; s < slots; s++ {
					inSlice := dSlotFloats.Slice(s*cols, (s+1)*cols)
					outSlice := q81In.Slice(s*(cols/32*36), (s+1)*(cols/32*36))
					if err := r.encodeQuantQ81(inSlice, outSlice, cols); err != nil {
						t.Fatal(err)
					}
				}

				runDown := func(scalar bool) []float32 {
					downOut, err := cudamem.Alloc[float32](r.device, slots*rows)
					if err != nil {
						t.Fatal(err)
					}
					defer downOut.Free()

					poison := make([]float32, slots*rows)
					for i := range poison {
						poison[i] = math.Float32frombits(0x7fc00000)
					}
					_ = downOut.CopyFromHost(poison)

					if scalar {
						if err := r.encodeMatvecQ2KSelectedQ81Scalar(downTensor, dIDs, q81In, downOut); err != nil {
							t.Fatal(err)
						}
					} else {
						if err := r.encodeMatvecQ2KSelectedQ81(downTensor, dIDs, q81In, downOut); err != nil {
							t.Fatal(err)
						}
					}
					if err := r.device.DefaultStream().Synchronize(context.Background()); err != nil {
						t.Fatal(err)
					}
					res := make([]float32, slots*rows)
					_ = downOut.CopyToHost(res)
					return res
				}

				simdDown := runDown(false)
				scalDown := runDown(true)

				for s, id := range ids {
					isSentinel := id < 0 || int(id) >= experts
					for row := 0; row < rows; row++ {
						idx := s*rows + row
						if isSentinel {
							if math.Float32bits(simdDown[idx]) != 0x7fffffff {
								t.Fatalf("slot %d (expert %d) row %d: expected sentinel 0x7fffffff, got SIMD down %v (%#08x)",
									s, id, row, simdDown[idx], math.Float32bits(simdDown[idx]))
							}
						} else {
							if math.IsNaN(float64(simdDown[idx])) {
								t.Fatalf("slot %d row %d: NaN in SIMD down output", s, row)
							}
						}
						if math.Float32bits(simdDown[idx]) != math.Float32bits(scalDown[idx]) {
							t.Fatalf("slot %d (expert %d) down row %d: SIMD %v (%#08x) != Scalar %v (%#08x)",
								s, id, row, simdDown[idx], math.Float32bits(simdDown[idx]), scalDown[idx], math.Float32bits(scalDown[idx]))
						}
					}
				}
			})
		})
	}
}

func cudaExpertFastFixture(typeID uint32, cols, rows, experts int, seed int64) []byte {
	blockBytes := 66
	if typeID == 10 {
		blockBytes = 84
	}
	const templateBlocks = 1024
	template := make([]byte, templateBlocks*blockBytes)
	for b := 0; b < templateBlocks; b++ {
		p := template[b*blockBytes : (b+1)*blockBytes]
		for i := range p {
			p[i] = byte((int(seed) + b*37 + i*13 + 7) & 0xff)
		}
		if typeID == 16 {
			binary.LittleEndian.PutUint16(p, 0x1c00+uint16((b*17)&0x3ff))
		} else {
			binary.LittleEndian.PutUint16(p[80:], 0x1c00+uint16((b*17)&0x3ff))
			binary.LittleEndian.PutUint16(p[82:], 0x1800+uint16((b*19)&0x3ff))
		}
	}
	total := experts * rows * (cols / 256) * blockBytes
	data := make([]byte, total)
	for i := 0; i < total; i += len(template) {
		chunk := len(template)
		if i+chunk > total {
			chunk = total - i
		}
		copy(data[i:i+chunk], template[:chunk])
	}
	return data
}
