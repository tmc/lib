//go:build linux

package main

import (
	_ "embed"
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

const cudaExpertWarps = 2

// cudaExpertQ81Source follows the default aligned single-token MoE arithmetic
// in pinned antirez/ds4 b0a147a, cuda/mmq/ds4_mmq.cu:2554 and :4307. Reading raw
// GGUF weights avoids repacking but preserves the aligned kernels' lane mapping.
// IQ2 truncates the scaled integer dot before floating conversion. Q2 uses
// quantized codes for its minimum correction, not the stored half block sum.
//
//go:embed cuda_experts_q81.cu
var cudaExpertQ81Source string

func (r *cudaFoundation) encodeMatvecIQ2SelectedQ81(gate, up cudaModelTensor, ids cudamem.Slice[int32], in cudamem.Slice[byte], gateOut, upOut cudamem.Slice[float32]) error {
	if err := cudaExpertProfile(gate, 16, 66); err != nil {
		return err
	}
	if err := cudaExpertProfile(up, 16, 66); err != nil {
		return err
	}
	for i := range gate.dimensions {
		if gate.dimensions[i] != up.dimensions[i] {
			return fmt.Errorf("cuda: IQ2 Q8_1 selected pair: mismatched dimensions")
		}
	}
	cols, rows, experts := gate.dimensions[0], gate.dimensions[1], gate.dimensions[2]
	slots := uint64(ids.Len())
	if slots == 0 || slots > math.MaxInt32/rows || uint64(in.Len()) != cols/32*36 || uint64(gateOut.Len()) != slots*rows || upOut.Len() != gateOut.Len() {
		return fmt.Errorf("cuda: IQ2 Q8_1 selected pair: invalid shape")
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		blocks := (rows*slots + cudaExpertWarps - 1) / cudaExpertWarps
		// slots of the experts planes are selected, and the byte extent per
		// expert comes from the tensor rather than the quantization layout.
		r.countWeightBytes("ds4_iq2_xxs_pair_selected_q8_1", (gate.bytes+up.bytes)/experts*slots)
		if err := cudalaunch.Kernel(r.iq2SelectedQ81Fn).Grid(uint32(blocks), 1, 1).Block(32*cudaExpertWarps, 1, 1).On(r.stream()).Launch(gate.device, up.device, ids, in, gateOut, upOut, uint32(cols), uint32(rows), uint32(slots), uint32(experts)); err != nil {
			return fmt.Errorf("cuda: launch IQ2 Q8_1 selected pair: %w", err)
		}
		return nil
	})
}

func (r *cudaFoundation) encodeMatvecQ2KSelectedQ81(t cudaModelTensor, ids cudamem.Slice[int32], in cudamem.Slice[byte], out cudamem.Slice[float32]) error {
	if err := cudaExpertProfile(t, 10, 84); err != nil {
		return err
	}
	cols, rows, experts := t.dimensions[0], t.dimensions[1], t.dimensions[2]
	slots := uint64(ids.Len())
	if slots == 0 || slots > math.MaxInt32/rows || uint64(in.Len()) != cols/32*36*slots || uint64(out.Len()) != rows*slots {
		return fmt.Errorf("cuda: Q2_K Q8_1 selected: invalid shape")
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		blocks := (rows*slots + cudaExpertWarps - 1) / cudaExpertWarps
		r.countWeightBytes("ds4_q2_k_selected_q8_1", t.bytes/experts*slots)
		if err := cudalaunch.Kernel(r.q2KSelectedQ81Fn).Grid(uint32(blocks), 1, 1).Block(32*cudaExpertWarps, 1, 1).On(r.stream()).Launch(t.device, ids, in, out, uint32(cols), uint32(rows), uint32(slots), uint32(experts)); err != nil {
			return fmt.Errorf("cuda: launch Q2_K Q8_1 selected: %w", err)
		}
		return nil
	})
}

func (r *cudaFoundation) encodeMatvecIQ2SelectedQ81Scalar(gate, up cudaModelTensor, ids cudamem.Slice[int32], in cudamem.Slice[byte], gateOut, upOut cudamem.Slice[float32]) error {
	if err := cudaExpertProfile(gate, 16, 66); err != nil {
		return err
	}
	if err := cudaExpertProfile(up, 16, 66); err != nil {
		return err
	}
	for i := range gate.dimensions {
		if gate.dimensions[i] != up.dimensions[i] {
			return fmt.Errorf("cuda: IQ2 Q8_1 selected pair scalar: mismatched dimensions")
		}
	}
	cols, rows, experts := gate.dimensions[0], gate.dimensions[1], gate.dimensions[2]
	slots := uint64(ids.Len())
	if slots == 0 || slots > math.MaxInt32/rows || uint64(in.Len()) != cols/32*36 || uint64(gateOut.Len()) != slots*rows || upOut.Len() != gateOut.Len() {
		return fmt.Errorf("cuda: IQ2 Q8_1 selected pair scalar: invalid shape")
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		blocks := (rows*slots + cudaExpertWarps - 1) / cudaExpertWarps
		if err := cudalaunch.Kernel(r.iq2SelectedQ81ScalarFn).Grid(uint32(blocks), 1, 1).Block(32*cudaExpertWarps, 1, 1).On(r.stream()).Launch(gate.device, up.device, ids, in, gateOut, upOut, uint32(cols), uint32(rows), uint32(slots), uint32(experts)); err != nil {
			return fmt.Errorf("cuda: launch IQ2 Q8_1 selected pair scalar: %w", err)
		}
		return nil
	})
}

func (r *cudaFoundation) encodeMatvecQ2KSelectedQ81Scalar(t cudaModelTensor, ids cudamem.Slice[int32], in cudamem.Slice[byte], out cudamem.Slice[float32]) error {
	if err := cudaExpertProfile(t, 10, 84); err != nil {
		return err
	}
	cols, rows, experts := t.dimensions[0], t.dimensions[1], t.dimensions[2]
	slots := uint64(ids.Len())
	if slots == 0 || slots > math.MaxInt32/rows || uint64(in.Len()) != cols/32*36*slots || uint64(out.Len()) != rows*slots {
		return fmt.Errorf("cuda: Q2_K Q8_1 selected scalar: invalid shape")
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		blocks := (rows*slots + cudaExpertWarps - 1) / cudaExpertWarps
		if err := cudalaunch.Kernel(r.q2KSelectedQ81ScalarFn).Grid(uint32(blocks), 1, 1).Block(32*cudaExpertWarps, 1, 1).On(r.stream()).Launch(t.device, ids, in, out, uint32(cols), uint32(rows), uint32(slots), uint32(experts)); err != nil {
			return fmt.Errorf("cuda: launch Q2_K Q8_1 selected scalar: %w", err)
		}
		return nil
	})
}

// encodeMatvecIQ2SelectedQ81Aligned is encodeMatvecIQ2SelectedQ81 reading the
// layer's aligned artifact instead of the raw tensors. gate and up supply only
// the shape; their weights are never read.
func (r *cudaFoundation) encodeMatvecIQ2SelectedQ81Aligned(gate, up cudaModelTensor, a *cudaMoEArtifact, ids cudamem.Slice[int32], in cudamem.Slice[byte], gateOut, upOut cudamem.Slice[float32]) error {
	if err := cudaExpertShape(gate, 16, 66); err != nil {
		return err
	}
	if err := cudaExpertShape(up, 16, 66); err != nil {
		return err
	}
	for i := range gate.dimensions {
		if gate.dimensions[i] != up.dimensions[i] {
			return fmt.Errorf("cuda: IQ2 Q8_1 selected aligned pair: mismatched dimensions")
		}
	}
	cols, rows, experts := gate.dimensions[0], gate.dimensions[1], gate.dimensions[2]
	nblk := experts * rows * (cols / 256)
	dqBytes := (nblk*2 + 63) &^ 63
	slots := uint64(ids.Len())
	if slots == 0 || slots > math.MaxInt32/rows || uint64(in.Len()) != cols/32*36 || uint64(gateOut.Len()) != slots*rows || upOut.Len() != gateOut.Len() ||
		uint64(a.gate.Len()) < dqBytes+nblk*64 || a.up.Len() != a.gate.Len() {
		return fmt.Errorf("cuda: IQ2 Q8_1 selected aligned pair: invalid shape")
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		blocks := (rows*slots + cudaExpertWarps - 1) / cudaExpertWarps
		r.countWeightBytes("ds4_iq2_xxs_pair_selected_q8_1_aligned", (gate.bytes+up.bytes)/experts*slots)
		if err := cudalaunch.Kernel(r.iq2SelectedQ81AlignedFn).Grid(uint32(blocks), 1, 1).Block(32*cudaExpertWarps, 1, 1).On(r.stream()).Launch(
			a.gate.Ptr(), a.gate.Ptr()+driver.DevicePtr(dqBytes), a.up.Ptr(), a.up.Ptr()+driver.DevicePtr(dqBytes),
			ids, in, gateOut, upOut, uint32(cols), uint32(rows), uint32(slots), uint32(experts)); err != nil {
			return fmt.Errorf("cuda: launch IQ2 Q8_1 selected aligned pair: %w", err)
		}
		return nil
	})
}

// encodeMatvecQ2KSelectedQ81Aligned is encodeMatvecQ2KSelectedQ81 reading the
// layer's aligned down artifact. t supplies only the shape.
func (r *cudaFoundation) encodeMatvecQ2KSelectedQ81Aligned(t cudaModelTensor, down cudamem.Slice[byte], ids cudamem.Slice[int32], in cudamem.Slice[byte], out cudamem.Slice[float32]) error {
	if err := cudaExpertShape(t, 10, 84); err != nil {
		return err
	}
	cols, rows, experts := t.dimensions[0], t.dimensions[1], t.dimensions[2]
	npair := experts * (rows / 2) * (cols / 256)
	dmBytes := (npair*8 + 63) &^ 63
	scBytes := (npair*32 + 63) &^ 63
	slots := uint64(ids.Len())
	if rows%2 != 0 || slots == 0 || slots > math.MaxInt32/rows || uint64(in.Len()) != cols/32*36*slots || uint64(out.Len()) != rows*slots ||
		uint64(down.Len()) < dmBytes+scBytes+npair*128 {
		return fmt.Errorf("cuda: Q2_K Q8_1 selected aligned: invalid shape")
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		blocks := (rows*slots + cudaExpertWarps - 1) / cudaExpertWarps
		r.countWeightBytes("ds4_q2_k_selected_q8_1_aligned", t.bytes/experts*slots)
		if err := cudalaunch.Kernel(r.q2KSelectedQ81AlignedFn).Grid(uint32(blocks), 1, 1).Block(32*cudaExpertWarps, 1, 1).On(r.stream()).Launch(
			down.Ptr(), down.Ptr()+driver.DevicePtr(dmBytes), down.Ptr()+driver.DevicePtr(dmBytes+scBytes),
			ids, in, out, uint32(cols), uint32(rows), uint32(slots), uint32(experts)); err != nil {
			return fmt.Errorf("cuda: launch Q2_K Q8_1 selected aligned: %w", err)
		}
		return nil
	})
}
