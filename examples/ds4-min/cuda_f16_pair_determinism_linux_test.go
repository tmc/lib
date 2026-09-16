//go:build linux

package main

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamem"
)

// TestCUDADeterminismF16PairBatch checks production projection shapes against
// independent single-token allocations, with a common baseline across modes.
func TestCUDADeterminismF16PairBatch(t *testing.T) {
	const batch, cols = 8, 4096
	for _, rows := range []int{256, 512, 1024} {
		t.Run(fmt.Sprintf("rows-%d", rows), func(t *testing.T) {
			r := cudaSynthFoundation(t)
			a, b, x := cudaF16PairBatchFixture(cols, rows, batch)
			dims := []uint64{cols, uint64(rows)}
			ta := cudaSynthTensor(t, r, "repeat batch A", 1, dims, a)
			tb := cudaSynthTensor(t, r, "repeat batch B", 1, dims, b)
			in := cudaSynthUpload(t, r, x)
			out := [2]cudamem.Slice[float32]{cudaTestAlloc(t, r, batch*rows), cudaTestAlloc(t, r, batch*rows)}
			var tokenIn [batch]cudamem.Slice[float32]
			var ref [batch][2]cudamem.Slice[float32]
			for token := range batch {
				tokenIn[token] = cudaSynthUpload(t, r, x[token*cols:(token+1)*cols])
				ref[token] = [2]cudamem.Slice[float32]{cudaTestAlloc(t, r, rows), cudaTestAlloc(t, r, rows)}
			}
			var baseline [2][]byte
			for _, mode := range []string{"queued", "stage_sync"} {
				t.Run(mode, func(t *testing.T) {
					for iteration := range cudaDeterminismRepeats {
						for _, s := range out {
							cudaDeterminismPoison(t, cudaDeterminismFloats("batch", s), iteration)
						}
						for _, pair := range ref {
							for _, s := range pair {
								cudaDeterminismPoison(t, cudaDeterminismFloats("reference", s), iteration)
							}
						}
						if err := r.encodeF16PairOrderedBatch(ta, tb, in, out[0], out[1], batch); err != nil {
							t.Fatal(err)
						}
						if mode == "stage_sync" {
							if err := syncStream(r.stream()); err != nil {
								t.Fatal(err)
							}
						}
						for token := range batch {
							if err := r.encodeF16PairOrdered(ta, tb, tokenIn[token], ref[token][0], ref[token][1]); err != nil {
								t.Fatal(err)
							}
							if mode == "stage_sync" {
								if err := syncStream(r.stream()); err != nil {
									t.Fatal(err)
								}
							}
						}
						if err := syncStream(r.stream()); err != nil {
							t.Fatal(err)
						}
						for projection, s := range out {
							got := cudaDeterminismRead(t, cudaDeterminismFloats("batch", s))
							var want []byte
							for token := range batch {
								want = append(want, cudaDeterminismRead(t, cudaDeterminismFloats("reference", ref[token][projection]))...)
							}
							if err := cudaDeterminismDifference(got, want, true); err != nil {
								t.Fatalf("iteration=%d projection=%d reference: %v", iteration, projection, err)
							}
							if baseline[projection] == nil {
								baseline[projection] = append([]byte(nil), got...)
							}
							if err := cudaDeterminismDifference(got, baseline[projection], true); err != nil {
								t.Fatalf("iteration=%d projection=%d baseline: %v", iteration, projection, err)
							}
							// A token-zero alias must fail the same comparator.
							alias := bytes.Repeat(want[:rows*4], batch)
							if err := cudaDeterminismDifference(got, alias, true); err == nil {
								t.Fatalf("iteration=%d projection=%d token alias control matched", iteration, projection)
							}
						}
						if iteration == 0 {
							t.Logf("device execution reached: batch=%d cols=%d rows=%d", batch, cols, rows)
						}
					}
					t.Logf("completed: iterations=%d batch_launches=%d reference_launches=%d float_comparisons=%d alias_controls=%d", cudaDeterminismRepeats, cudaDeterminismRepeats, batch*cudaDeterminismRepeats, 2*batch*rows*cudaDeterminismRepeats, 2*cudaDeterminismRepeats)
				})
				if t.Failed() {
					return
				}
			}
		})
	}
}
