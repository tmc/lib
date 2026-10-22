//go:build linux

package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"testing"

	"github.com/tmc/lib/lm/quant"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// This is the default CUDA oracle-equivalence contract, separately registered
// after attempts12/13. TestCUDAHCDenseProjection and its retained FAIL receipts
// remain mathematical diagnostics. In particular, cuBLAS's cancellation loss
// is part of the selected oracle route, not a request to replace it with FP64.
// The fixtures and HC tolerance remain unchanged: 2e-5 absolute and relative.
func TestCUDAHCOracleEquivalence(t *testing.T) {
	for _, name := range []string{"cancellation", "varied"} {
		t.Run(name, func(t *testing.T) {
			r := cudaSynthFoundation(t)
			state, block := make([]float32, 16384), make([]float32, 4096)
			for i := range state {
				state[i] = 1
				if name == "varied" {
					state[i] = float32((i*37)%251-125) / 64
				}
			}
			for i := range block {
				block[i] = float32(i%31-15) / 16
			}
			raw := make([]byte, 24*16384*2)
			weights := make([]float32, 24*16384)
			for row := range 24 {
				for col := range 16384 {
					var w float32
					if name == "cancellation" {
						// Every weight is nonzero and representable in binary16.
						// For half-rounded unit input, the exact row dot is +/-8;
						// the current 256-lane float tree loses the small terms.
						w = 1.0 / 1024
						if col%256 < 64 {
							w = 65504
						} else if col%256 < 128 {
							w = -65504
						}
						if row%2 != 0 {
							w = -w
						}
					} else {
						w = float32((col*19+row*43)%127-63) / 4096
						if w == 0 {
							w = 1.0 / 8192
						}
					}
					bits := cudaQ81HalfReference(w)
					weights[row*16384+col] = quant.Float16ToFloat32(bits)
					binary.LittleEndian.PutUint16(raw[2*(row*16384+col):], bits)
				}
			}

			for _, rows := range []int{24, 4} {
				t.Run(fmt.Sprintf("rows%d", rows), func(t *testing.T) {
					h, err := newCUDAHC(r)
					if err != nil {
						t.Fatal(err)
					}
					t.Cleanup(func() {
						if err := h.Close(); err != nil {
							t.Error(err)
						}
					})
					fn := cudaSynthTensor(t, r, "oracle HC fn", 1, []uint64{16384, uint64(rows)}, raw[:rows*16384*2])
					ds, db := cudaSynthUpload(t, r, state), cudaSynthUpload(t, r, block)
					scale := []float32{.125, .125, .125}
					base := make([]float32, 24)
					for i := range base {
						base[i] = float32(i%7-3) / 16
					}
					norm := make([]float32, 4096)
					for i := range norm {
						norm[i] = 1 + float32(i%5-2)/32
					}
					ns := 3
					if rows == 4 {
						ns = 1
					}
					st := cudaCompressedTensor(t, r, "scale", scale[:ns])
					bt := cudaCompressedTensor(t, r, "base", base[:rows])
					nt := cudaCompressedTensor(t, r, "norm", norm)
					out, post := cudaTestAlloc(t, r, 8192), cudaTestAlloc(t, r, 16384)
					for _, v := range []cudamem.Slice[float32]{h.normalized, h.mix, h.split, out, post} {
						cudaHCDensePoison(t, v)
					}
					halfPoison := make([]uint16, 16384)
					for i := range halfPoison {
						halfPoison[i] = 0x7e00
					}
					if err := h.half.CopyFromHost(halfPoison); err != nil {
						t.Fatal(err)
					}
					cudaQualificationMeasure(t, r, func() error { return h.pre(ds, fn, st, bt, nt, out, rows == 4) })
					normalized := cudaTestRead(t, h.normalized)
					cudaHCDenseCheck(t, "oracle route RMS", normalized, cudaHCDefaultNormalized(state))
					gotHalf := make([]uint16, 16384)
					if err := h.half.CopyToHost(gotHalf); err != nil {
						t.Fatal(err)
					}
					for i, v := range normalized {
						want := cudaQ81HalfReference(v)
						if gotHalf[i] != want {
							t.Fatalf("activation half %d: got %#x want %#x", i, gotHalf[i], want)
						}
						if name == "cancellation" && want != 0x3c00 {
							t.Fatalf("cancellation input %d is not half one", i)
						}
					}
					blas := cudaF16DefaultCUBLAS(t, r, fn, normalized)
					actualMix := cudaTestRead(t, h.mix.Slice(0, rows))
					cudaHCDenseCheck(t, "production cuBLAS projection vs independent call", actualMix, blas)
					var mathDelta float64
					for row := range rows {
						var sum float64
						for col, v := range normalized {
							sum += float64(weights[row*16384+col]) * float64(quant.Float16ToFloat32(cudaQ81HalfReference(v)))
						}
						if name == "cancellation" {
							want := float64(8)
							if row%2 != 0 {
								want = -8
							}
							if sum != want {
								t.Fatalf("cancellation mathematical row%d: got%g want%g", row, sum, want)
							}
						}
						mathDelta = max(mathDelta, math.Abs(sum-float64(blas[row])))
					}
					t.Logf("separate mathematical diagnostic: cuBLAS max float64 delta %g; historical mathematical FAIL is not promoted by this oracle-equivalence test", mathDelta)
					mix := make([]float32, 24)
					copy(mix, blas)
					wantPre, wantPost := cudaHCDenseFromMix(state, block, mix, scale, base, norm)
					cudaHCDenseCheck(t, "production collapse from oracle projection", cudaTestRead(t, out), wantPre)
					wrong := append([]float32(nil), blas...)
					wrong[0] += 1
					if cudaQualificationClose(wrong, blas, cudaHCDenseATol, cudaHCDenseRTol) == nil {
						t.Fatal("accepted corrupted oracle projection")
					}
					if rows == 24 {
						// Post must consume the cached split, not recompute or read the mix.
						cudaHCDensePoison(t, h.mix)
						cudaQualificationMeasure(t, r, func() error { return h.post(ds, db, post) })
						cudaHCDenseCheck(t, "production cached post from oracle projection", cudaTestRead(t, post), wantPost)
						if err := h.post(ds, db, post); err == nil {
							t.Fatal("accepted already-consumed cached split")
						}
						if name == "cancellation" {
							half := make([]float32, len(normalized))
							for i, v := range normalized {
								half[i] = quant.Float16ToFloat32(cudaQ81HalfReference(v))
							}
							old := cudaTestAlloc(t, r, 24)
							cudaHCDensePoison(t, old)
							input := cudaSynthUpload(t, r, half)
							cudaQualificationMeasure(t, r, func() error { return r.encodeF16Matvec(fn, input, old) })
							if cudaQualificationClose(cudaTestRead(t, old), blas, cudaHCDenseATol, cudaHCDenseRTol) == nil {
								t.Fatal("cancellation no longer rejects old float32 projection")
							}
						}
					} else if err := h.post(ds, db, post); err == nil {
						t.Fatal("accepted output-head split as layer split")
					}
				})
			}
		})
	}
}
