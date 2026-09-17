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
	for _, n := range []int{1, 2, 7, 8, 9, 15, 16, 17, 255, 256} {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
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
			for _, check := range []struct {
				name      string
				got, want []float32
			}{
				{"mix", cudaTestRead(t, batch.mix.Slice(0, n*24)), mixHost},
				{"split", cudaTestRead(t, batch.split.Slice(0, n*24)), splitHost},
				{"pre", cudaTestRead(t, pre), cudaTestRead(t, wantPre)},
				{"post", cudaTestRead(t, post), cudaTestRead(t, wantPost)},
			} {
				for i, v := range check.got {
					if math.Float32bits(v) != math.Float32bits(check.want[i]) {
						t.Fatalf("%s[%d]: got %08x want %08x", check.name, i, math.Float32bits(v), math.Float32bits(check.want[i]))
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
