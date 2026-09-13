//go:build linux

package main

import (
	_ "embed"
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

const cudaExpertWarps = 2

// cudaExpertQ81Source follows the default aligned single-token MoE arithmetic
// in pinned antirez/ds4 b0a147a, cuda/mmq/ds4_mmq.cu:2554 and :4307. Reading raw
// GGUF weights avoids repacking but preserves the aligned kernels' lane mapping.
// IQ2 truncates the scaled integer dot before floating conversion. Q2 uses
// quantized codes for its minimum correction, not the stored half block sum.
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
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		blocks := (rows*slots + cudaExpertWarps - 1) / cudaExpertWarps
		if err := cudalaunch.Kernel(r.iq2SelectedQ81Fn).Grid(uint32(blocks)).Block(32*cudaExpertWarps).On(r.stream()).Launch(gate.device, up.device, ids, in, gateOut, upOut, uint32(cols), uint32(rows), uint32(slots), uint32(experts)); err != nil {
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
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		blocks := (rows*slots + cudaExpertWarps - 1) / cudaExpertWarps
		if err := cudalaunch.Kernel(r.q2KSelectedQ81Fn).Grid(uint32(blocks)).Block(32*cudaExpertWarps).On(r.stream()).Launch(t.device, ids, in, out, uint32(cols), uint32(rows), uint32(slots), uint32(experts)); err != nil {
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
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		blocks := (rows*slots + cudaExpertWarps - 1) / cudaExpertWarps
		if err := cudalaunch.Kernel(r.iq2SelectedQ81ScalarFn).Grid(uint32(blocks)).Block(32*cudaExpertWarps).On(r.stream()).Launch(gate.device, up.device, ids, in, gateOut, upOut, uint32(cols), uint32(rows), uint32(slots), uint32(experts)); err != nil {
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
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		blocks := (rows*slots + cudaExpertWarps - 1) / cudaExpertWarps
		if err := cudalaunch.Kernel(r.q2KSelectedQ81ScalarFn).Grid(uint32(blocks)).Block(32*cudaExpertWarps).On(r.stream()).Launch(t.device, ids, in, out, uint32(cols), uint32(rows), uint32(slots), uint32(experts)); err != nil {
			return fmt.Errorf("cuda: launch Q2_K Q8_1 selected scalar: %w", err)
		}
		return nil
	})
}
