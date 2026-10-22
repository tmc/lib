//go:build linux

package main

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudamem"
)

// cudaSpeculativeMaxBatch is the largest verifier batch for which the
// small-batch adapter is defined.  The adapter deliberately reuses the
// already-qualified one-token selected-expert kernels.  There is no new
// packed multi-token weight layout here, so a future fused kernel can replace
// this implementation without changing its caller contract.
const cudaSpeculativeMaxBatch = 8

func checkSpeculativeBatch(batch, slots int) error {
	if batch <= 0 || batch > cudaSpeculativeMaxBatch {
		return fmt.Errorf("cuda: speculative batch %d outside 1..%d", batch, cudaSpeculativeMaxBatch)
	}
	if slots <= 0 {
		return fmt.Errorf("cuda: speculative selected slots %d must be positive", slots)
	}
	return nil
}

// encodeMatvecIQ2SelectedSmallBatch evaluates batch token rows with the
// existing selected IQ2_XXS kernel. ids is token-major, with slots expert IDs
// per row; in is token-major F32 input; outputs are token-major and
// slot-major. The one-token kernel is invoked once per row. This is a safe
// dispatch contract for speculative verification, not a claim of a fused
// multi-token kernel.
func (r *cudaFoundation) encodeMatvecIQ2SelectedSmallBatch(
	gate, up cudaModelTensor, ids cudamem.Slice[int32], in, gateOut, upOut cudamem.Slice[float32],
	slots, batch int,
) error {
	if err := checkSpeculativeBatch(batch, slots); err != nil {
		return err
	}
	if err := cudaExpertProfile(gate, 16, 66); err != nil {
		return err
	}
	if err := cudaExpertProfile(up, 16, 66); err != nil {
		return err
	}
	if len(gate.dimensions) != 3 || len(up.dimensions) != 3 {
		return fmt.Errorf("cuda: speculative IQ2 selected batch: invalid tensor dimensions")
	}
	cols, rows := gate.dimensions[0], gate.dimensions[1]
	if uint64(ids.Len()) != uint64(batch)*uint64(slots) || uint64(in.Len()) != uint64(batch)*cols ||
		uint64(gateOut.Len()) != uint64(batch)*uint64(slots)*rows || upOut.Len() != gateOut.Len() {
		return fmt.Errorf("cuda: speculative IQ2 selected batch: invalid shape")
	}
	colsInt := int(cols)
	rowsInt := int(rows)
	for token := 0; token < batch; token++ {
		idRow := ids.Slice(token*slots, (token+1)*slots)
		inRow := in.Slice(token*colsInt, (token+1)*colsInt)
		outLo := token * slots * rowsInt
		if err := r.encodeMatvecIQ2Selected(gate, up, idRow, inRow,
			gateOut.Slice(outLo, outLo+slots*rowsInt), upOut.Slice(outLo, outLo+slots*rowsInt)); err != nil {
			return fmt.Errorf("cuda: speculative IQ2 row %d: %w", token, err)
		}
	}
	return nil
}

// encodeMatvecQ2KSelectedSmallBatch is the Q2_K counterpart of
// encodeMatvecIQ2SelectedSmallBatch. Each token owns slots input vectors and
// output rows, matching the existing selected Q2_K kernel's layout.
func (r *cudaFoundation) encodeMatvecQ2KSelectedSmallBatch(
	t cudaModelTensor, ids cudamem.Slice[int32], in, out cudamem.Slice[float32],
	slots, batch int,
) error {
	if err := checkSpeculativeBatch(batch, slots); err != nil {
		return err
	}
	if err := cudaExpertProfile(t, 10, 84); err != nil {
		return err
	}
	if len(t.dimensions) != 3 {
		return fmt.Errorf("cuda: speculative Q2_K selected batch: invalid tensor dimensions")
	}
	cols, rows := t.dimensions[0], t.dimensions[1]
	if uint64(ids.Len()) != uint64(batch)*uint64(slots) || uint64(in.Len()) != uint64(batch)*uint64(slots)*cols ||
		uint64(out.Len()) != uint64(batch)*uint64(slots)*rows {
		return fmt.Errorf("cuda: speculative Q2_K selected batch: invalid shape")
	}
	colsInt := int(cols)
	rowsInt := int(rows)
	for token := 0; token < batch; token++ {
		idRow := ids.Slice(token*slots, (token+1)*slots)
		inLo := token * slots * colsInt
		outLo := token * slots * rowsInt
		if err := r.encodeMatvecQ2KSelected(t, idRow,
			in.Slice(inLo, inLo+slots*colsInt), out.Slice(outLo, outLo+slots*rowsInt)); err != nil {
			return fmt.Errorf("cuda: speculative Q2_K row %d: %w", token, err)
		}
	}
	return nil
}

// encodeMatvecIQ2SelectedQ81SmallBatch and
// encodeMatvecQ2KSelectedQ81SmallBatch provide the same bounded adapter for
// the Q8_1 activation route used by the CUDA decode path. They intentionally
// preserve the existing per-token input and output layouts.
func (r *cudaFoundation) encodeMatvecIQ2SelectedQ81SmallBatch(
	gate, up cudaModelTensor, ids cudamem.Slice[int32], in cudamem.Slice[byte], gateOut, upOut cudamem.Slice[float32],
	slots, batch int,
) error {
	if err := checkSpeculativeBatch(batch, slots); err != nil {
		return err
	}
	if err := cudaExpertProfile(gate, 16, 66); err != nil {
		return err
	}
	if err := cudaExpertProfile(up, 16, 66); err != nil {
		return err
	}
	if len(gate.dimensions) < 2 {
		return fmt.Errorf("cuda: speculative IQ2 Q8_1 selected batch: invalid tensor dimensions")
	}
	cols, rows := gate.dimensions[0], gate.dimensions[1]
	rowBytes := cols / 32 * 36
	if uint64(ids.Len()) != uint64(batch*slots) || uint64(in.Len()) != uint64(batch)*rowBytes ||
		uint64(gateOut.Len()) != uint64(batch*slots)*rows || upOut.Len() != gateOut.Len() {
		return fmt.Errorf("cuda: speculative IQ2 Q8_1 selected batch: invalid shape")
	}
	for token := 0; token < batch; token++ {
		idRow := ids.Slice(token*slots, (token+1)*slots)
		inRow := in.Slice(token*int(rowBytes), (token+1)*int(rowBytes))
		outLo := token * slots * int(rows)
		if err := r.encodeMatvecIQ2SelectedQ81(gate, up, idRow, inRow,
			gateOut.Slice(outLo, outLo+slots*int(rows)), upOut.Slice(outLo, outLo+slots*int(rows))); err != nil {
			return fmt.Errorf("cuda: speculative IQ2 Q8_1 row %d: %w", token, err)
		}
	}
	return nil
}

func (r *cudaFoundation) encodeMatvecQ2KSelectedQ81SmallBatch(
	t cudaModelTensor, ids cudamem.Slice[int32], in cudamem.Slice[byte], out cudamem.Slice[float32],
	slots, batch int,
) error {
	if err := checkSpeculativeBatch(batch, slots); err != nil {
		return err
	}
	if err := cudaExpertProfile(t, 10, 84); err != nil {
		return err
	}
	if len(t.dimensions) < 2 {
		return fmt.Errorf("cuda: speculative Q2_K Q8_1 selected batch: invalid tensor dimensions")
	}
	cols, rows := t.dimensions[0], t.dimensions[1]
	rowBytes := cols / 32 * 36
	if uint64(ids.Len()) != uint64(batch*slots) || uint64(in.Len()) != uint64(batch*slots)*rowBytes ||
		uint64(out.Len()) != uint64(batch*slots)*rows {
		return fmt.Errorf("cuda: speculative Q2_K Q8_1 selected batch: invalid shape")
	}
	for token := 0; token < batch; token++ {
		idRow := ids.Slice(token*slots, (token+1)*slots)
		inLo := token * slots * int(rowBytes)
		outLo := token * slots * int(rows)
		if err := r.encodeMatvecQ2KSelectedQ81(t, idRow,
			in.Slice(inLo, inLo+slots*int(rowBytes)), out.Slice(outLo, outLo+slots*int(rows))); err != nil {
			return fmt.Errorf("cuda: speculative Q2_K Q8_1 row %d: %w", token, err)
		}
	}
	return nil
}
