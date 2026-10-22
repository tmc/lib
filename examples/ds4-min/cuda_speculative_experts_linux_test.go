//go:build linux

package main

import (
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamem"
)

func TestCUDAQualificationSpeculativeBatchContract(t *testing.T) {
	for _, tc := range []struct {
		batch, slots int
		wantErr      bool
	}{
		{batch: 1, slots: 1},
		{batch: 8, slots: 6},
		{batch: 0, slots: 1, wantErr: true},
		{batch: 9, slots: 1, wantErr: true},
		{batch: 2, slots: 0, wantErr: true},
	} {
		t.Run(fmt.Sprintf("batch-%d-slots-%d", tc.batch, tc.slots), func(t *testing.T) {
			err := checkSpeculativeBatch(tc.batch, tc.slots)
			if (err != nil) != tc.wantErr {
				t.Fatalf("checkSpeculativeBatch(%d, %d) = %v, wantErr=%v", tc.batch, tc.slots, err, tc.wantErr)
			}
		})
	}
}

// TestCUDAQualificationSpeculativeSelectedSmallBatch is the differential gate
// for the bounded adapter. Each batch row is compared with the existing
// one-token selected kernels and with the independent float64 CPU reference.
// The adapter intentionally makes one existing device launch per row; this
// proves the token-major contract without asserting an unimplemented fused
// weight layout.
func TestCUDAQualificationSpeculativeSelectedSmallBatch(t *testing.T) {
	r := cudaSynthFoundation(t)
	const (
		cols    = 512
		rows    = 5
		experts = 7
		slots   = 3
	)
	gateRaw := cudaExpertFixture(16, cols, rows, experts, 611)
	upRaw := cudaExpertFixture(16, cols, rows, experts, 612)
	downRaw := cudaExpertFixture(10, cols, rows, experts, 613)
	dims := []uint64{cols, rows, experts}
	gate := cudaSynthTensor(t, r, "speculative IQ2 gate", 16, dims, gateRaw)
	up := cudaSynthTensor(t, r, "speculative IQ2 up", 16, dims, upRaw)
	down := cudaSynthTensor(t, r, "speculative Q2 down", 10, dims, downRaw)

	for _, batch := range []int{1, 2, 8} {
		t.Run(fmt.Sprintf("batch-%d", batch), func(t *testing.T) {
			ids := make([]int32, batch*slots)
			for token := 0; token < batch; token++ {
				for slot := 0; slot < slots; slot++ {
					ids[token*slots+slot] = int32((token*3 + slot*2 + 1) % experts)
				}
			}
			dIDs, err := cudamem.Alloc[int32](r.device, len(ids))
			if err != nil {
				t.Fatal(err)
			}
			defer dIDs.Free()
			if err := dIDs.CopyFromHost(ids); err != nil {
				t.Fatal(err)
			}
			in := make([]float32, batch*cols)
			slotIn := make([]float32, batch*slots*cols)
			for i := range in {
				in[i] = float32((i*17+i/13)%251-125) / 97
			}
			for i := range slotIn {
				slotIn[i] = float32((i*19+i/11)%257-128) / 103
			}
			dIn := cudaSynthUpload(t, r, in)
			defer dIn.Free()
			dSlotIn := cudaSynthUpload(t, r, slotIn)
			defer dSlotIn.Free()

			gotGate := cudaTestAlloc(t, r, batch*slots*rows)
			gotUp := cudaTestAlloc(t, r, batch*slots*rows)
			gotDown := cudaTestAlloc(t, r, batch*slots*rows)
			wantGate := cudaTestAlloc(t, r, batch*slots*rows)
			wantUp := cudaTestAlloc(t, r, batch*slots*rows)
			wantDown := cudaTestAlloc(t, r, batch*slots*rows)
			cudaQualificationMeasure(t, r, func() error {
				if err := r.encodeMatvecIQ2SelectedSmallBatch(gate, up, dIDs, dIn, gotGate, gotUp, slots, batch); err != nil {
					return err
				}
				return r.encodeMatvecQ2KSelectedSmallBatch(down, dIDs, dSlotIn, gotDown, slots, batch)
			})
			cudaQualificationMeasure(t, r, func() error {
				for token := 0; token < batch; token++ {
					idRow := dIDs.Slice(token*slots, (token+1)*slots)
					inRow := dIn.Slice(token*cols, (token+1)*cols)
					outLo := token * slots * rows
					if err := r.encodeMatvecIQ2Selected(gate, up, idRow, inRow,
						wantGate.Slice(outLo, outLo+slots*rows), wantUp.Slice(outLo, outLo+slots*rows)); err != nil {
						return err
					}
					inLo := token * slots * cols
					if err := r.encodeMatvecQ2KSelected(down, idRow,
						dSlotIn.Slice(inLo, inLo+slots*cols), wantDown.Slice(outLo, outLo+slots*rows)); err != nil {
						return err
					}
				}
				return nil
			})
			gotG, gotU, gotD := cudaTestRead(t, gotGate), cudaTestRead(t, gotUp), cudaTestRead(t, gotDown)
			wantG, wantU, wantD := cudaTestRead(t, wantGate), cudaTestRead(t, wantUp), cudaTestRead(t, wantDown)
			if err := exactFloat32Bits(gotG, wantG); err != nil {
				t.Fatalf("IQ2 gate differs from sequential selected kernels: %v", err)
			}
			if err := exactFloat32Bits(gotU, wantU); err != nil {
				t.Fatalf("IQ2 up differs from sequential selected kernels: %v", err)
			}
			if err := exactFloat32Bits(gotD, wantD); err != nil {
				t.Fatalf("Q2_K down differs from sequential selected kernels: %v", err)
			}
			for token := 0; token < batch; token++ {
				idRow := ids[token*slots : (token+1)*slots]
				inRow := in[token*cols : (token+1)*cols]
				want := cudaExpertReference(t, gateRaw, 16, cols, rows, idRow, inRow, false)
				if err := cudaQualificationClose(gotG[token*slots*rows:(token+1)*slots*rows], want, cudaExpertATol, cudaExpertRTol); err != nil {
					t.Fatalf("token %d IQ2 gate differs from host reference: %v", token, err)
				}
				want = cudaExpertReference(t, downRaw, 10, cols, rows, idRow, slotIn[token*slots*cols:(token+1)*slots*cols], true)
				if err := cudaQualificationClose(gotD[token*slots*rows:(token+1)*slots*rows], want, cudaExpertATol, cudaExpertRTol); err != nil {
					t.Fatalf("token %d Q2_K down differs from host reference: %v", token, err)
				}
			}
		})
	}
}
