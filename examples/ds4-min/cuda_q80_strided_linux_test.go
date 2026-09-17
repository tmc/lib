//go:build linux

package main

import (
	"bytes"
	"fmt"
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamem"
)

func TestCUDAQualificationQ80StridedQuantization(t *testing.T) {
	r := cudaSynthFoundation(t)
	for _, batch := range []int{1, 2, 7, 8, 9, 17, 255, 256} {
		for _, cols := range []int{32, 96, cudaAttentionHeads} {
			t.Run(fmt.Sprintf("%d/%d", batch, cols), func(t *testing.T) {
				stride := cols + cudaAttentionScratch
				x := make([]float32, (batch-1)*stride+cols)
				for i := range x {
					x[i] = float32(math.NaN()) // Reading padding must poison the result.
				}
				for row := range batch {
					for i := range cols {
						x[row*stride+i] = float32((row*71+i*37)%257-128) / 17
					}
				}
				in := cudaSynthUpload(t, r, x)
				qstride := cols / 32 * 36
				out := cudaExpertAllocBytes(t, r, batch*qstride)
				// Odd numbers of Q8_0 blocks occupy a multiple of four bytes,
				// but kernel pointer arguments require eight-byte alignment.
				// Allocate each scalar reference row independently.
				want := make([]cudamem.Slice[byte], batch)
				for row := range want {
					want[row] = cudaExpertAllocBytes(t, r, qstride)
				}
				if err := out.CopyFromHost(bytes.Repeat([]byte{0xa5}, out.Len())); err != nil {
					t.Fatal(err)
				}
				cudaQualificationMeasure(t, r, func() error {
					for row := range batch {
						if err := r.encodeQuantQ80(in.Slice(row*stride, row*stride+cols), want[row], cols); err != nil {
							return err
						}
					}
					return r.encodeQuantQ80Strided(in, out, cols, stride, batch)
				})
				b := make([]byte, batch*qstride)
				for row := range want {
					if err := want[row].CopyToHost(b[row*qstride : (row+1)*qstride]); err != nil {
						t.Fatal(err)
					}
				}
				cudaQ80CheckBytes(t, out, b)
				for name, err := range map[string]error{
					"short input":   r.encodeQuantQ80Strided(in.Slice(0, in.Len()-1), out, cols, stride, batch),
					"short output":  r.encodeQuantQ80Strided(in, out.Slice(0, out.Len()-1), cols, stride, batch),
					"overlap":       r.encodeQuantQ80Strided(in, out, cols, cols-1, batch),
					"partial block": r.encodeQuantQ80Strided(in, out, cols-1, stride, batch),
					"zero batch":    r.encodeQuantQ80Strided(in, out, cols, stride, 0),
				} {
					if err == nil {
						t.Fatalf("accepted %s", name)
					}
				}
			})
		}
	}
}
