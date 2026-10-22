//go:build linux

package main

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func TestCUDAHCBatchExact(t *testing.T) {
	r := cudaSynthFoundation(t)
	state, block, _, scale, base, norm, weights := cudaHCDefaultFixture(t)
	// Dense, signed weights exercise cuBLAS reductions beyond the sparse
	// default-precision fixture. Keep scalar pre/post as the device reference.
	weights, _ = cudaSynthF16(rand.New(rand.NewSource(19)), 24, 16384, -7)
	fn := cudaSynthTensor(t, r, "hc batch fn", 1, []uint64{16384, 24}, weights)
	st := cudaCompressedTensor(t, r, "scale", scale)
	bt := cudaCompressedTensor(t, r, "base", base)
	nt := cudaCompressedTensor(t, r, "norm", norm)
	batch, err := newCUDAHCBatch(r, cudaPrefillGroupWidth)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := batch.Close(); err != nil {
			t.Error(err)
		}
	})
	scalar, err := newCUDAHC(r)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := scalar.Close(); err != nil {
			t.Error(err)
		}
	})
	// The batch path has two arms and they make different promises. The
	// per-token arm issues one GEMM per row and reproduces the scalar reference
	// bit for bit, which is what it exists for. The default arm projects the
	// whole group in one GEMM; cuBLAS picks its algorithm from the shape, so a
	// wide N does not sum a column the way a single-column call did, and the
	// promise there is a bound rather than equality. Both ship, so both are
	// checked -- and the bound is the interesting one, since it is the arm the
	// prefill actually runs.
	for _, arm := range []struct {
		name   string
		perTok bool
	}{{"batched", false}, {"per-token", true}} {
		r.disableLibraryGEMM = arm.perTok
		for _, n := range []int{1, 2, 7, 8, 9, 15, 16, 17, 255, 256} {
			t.Run(arm.name+"/"+fmt.Sprint(n), func(t *testing.T) {
				x := make([]float32, n*16384)
				blocks := make([]float32, n*4096)
				for row := range n {
					for i, v := range state {
						x[row*16384+i] = v*float32(row%7+1)/4 + float32(row%3-1)/16
					}
					for i, v := range block {
						blocks[row*4096+i] = v + float32(row%11-5)/32
					}
				}
				ds := cudaSynthUpload(t, r, x)
				db := cudaSynthUpload(t, r, blocks)
				pre := cudaTestAlloc(t, r, n*8192)
				post := cudaTestAlloc(t, r, n*16384)
				wantPre := cudaTestAlloc(t, r, n*8192)
				wantPost := cudaTestAlloc(t, r, n*16384)
				// Copy scalar scratch only in the test, after each projection.
				mixHost, splitHost := make([]float32, n*24), make([]float32, n*24)
				for row := range n {
					cudaQualificationMeasure(t, r, func() error {
						if err := scalar.pre(ds.Slice(row*16384, (row+1)*16384), fn, st, bt, nt, wantPre.Slice(row*8192, (row+1)*8192), false); err != nil {
							return err
						}
						return scalar.post(ds.Slice(row*16384, (row+1)*16384), db.Slice(row*4096, (row+1)*4096), wantPost.Slice(row*16384, (row+1)*16384))
					})
					copy(mixHost[row*24:], cudaTestRead(t, scalar.mix))
					copy(splitHost[row*24:], cudaTestRead(t, scalar.split))
				}
				cudaQualificationMeasure(t, r, func() error {
					if err := batch.pre(ds, fn, st, bt, nt, pre, n); err != nil {
						return err
					}
					return batch.post(ds, db, post, n)
				})
				// The batched compressor reads the normalized rows packed.
				want := cudaTestRead(t, wantPre)
				wantNorm := make([]float32, 0, n*4096)
				for row := range n {
					wantNorm = append(wantNorm, want[row*8192+4096:(row+1)*8192]...)
				}
				for _, check := range []struct {
					name      string
					got, want []float32
				}{
					{"mix", cudaTestRead(t, batch.mix.Slice(0, n*24)), mixHost},
					{"split", cudaTestRead(t, batch.split.Slice(0, n*24)), splitHost},
					{"pre", cudaTestRead(t, pre), want},
					{"norm", cudaTestRead(t, batch.norm.Slice(0, n*4096)), wantNorm},
					{"post", cudaTestRead(t, post), cudaTestRead(t, wantPost)},
				} {
					var worst float64
					var scale float64
					for _, v := range check.want {
						if a := math.Abs(float64(v)); a > scale {
							scale = a
						}
					}
					for i, v := range check.got {
						if arm.perTok {
							if math.Float32bits(v) != math.Float32bits(check.want[i]) {
								t.Fatalf("%s[%d]: got %08x want %08x", check.name, i, math.Float32bits(v), math.Float32bits(check.want[i]))
							}
							continue
						}
						if d := math.Abs(float64(v) - float64(check.want[i])); d > worst {
							worst = d
						}
					}
					if !arm.perTok && scale > 0 {
						if rel := worst / scale; rel > cudaHCBatchTolerance {
							t.Fatalf("%s: deviation from the scalar reference %.3g of full scale, want at most %g", check.name, rel, cudaHCBatchTolerance)
						} else {
							t.Logf("%s: deviation %.3g of full scale", check.name, rel)
						}
					}
				}
				if err := batch.post(ds, db, post, n); err == nil {
					t.Fatal("post reused a consumed projection")
				}
				if err := batch.pre(ds, fn, st, bt, nt, pre, n+1); err == nil {
					t.Fatal("accepted wrong batch extent")
				}
				cudaQualificationMeasure(t, r, func() error { return batch.pre(ds, fn, st, bt, nt, pre, n) })
				if err := batch.post(post, db, post, n); err == nil {
					t.Fatal("accepted a different residual")
				}
				if err := batch.post(ds, db, post, n); err == nil {
					t.Fatal("failed post retained a projection")
				}
			})
		}
	}
}

// cudaHCBatchTolerance bounds how far the whole-group GEMM may sit from the
// scalar reference, as a fraction of the largest reference value in the buffer.
// A dot product of 16384 half products reassociated by a different cuBLAS
// algorithm moves the low bits; anything computing a different quantity moves
// the result by order one, so this separates the two by many orders of
// magnitude. The observed value is logged on every subtest, so a change that
// erodes it is visible before it crosses.
const cudaHCBatchTolerance = 1e-5
