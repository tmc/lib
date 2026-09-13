//go:build linux

package main

import (
	_ "embed"
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// cudaExpertSource implements F32 selected-expert projections. Its codebooks
// are from antirez/ds4 b0a147a7fba6d1a104d047d5a140e9bb4bfc13cd,
// metal/moe.metal; block layouts match lm/quant's pinned C oracle fixtures.
//go:embed cuda_experts.cu
var cudaExpertSource string

// cudaExpertProfile checks both dimensions and the available weight extent.
func cudaExpertProfile(t cudaModelTensor, typeID uint32, blockBytes uint64) error {
	if t.typeID != typeID || len(t.dimensions) != 3 || t.device == 0 {
		return fmt.Errorf("cuda: expert tensor %q: invalid type or rank", t.name)
	}
	cols, rows, experts := t.dimensions[0], t.dimensions[1], t.dimensions[2]
	if cols == 0 || cols%256 != 0 || cols > math.MaxUint32 || rows == 0 || rows > math.MaxUint32 || experts == 0 || experts > math.MaxInt32 {
		return fmt.Errorf("cuda: expert tensor %q: invalid dimensions", t.name)
	}
	rowBytes := cols / 256 * blockBytes
	if rows > t.bytes/rowBytes || experts > t.bytes/(rowBytes*rows) {
		return fmt.Errorf("cuda: expert tensor %q: truncated weights", t.name)
	}
	return nil
}

// encodeMatvecIQ2Selected appends paired F32-input IQ2_XXS projections.
// Outputs are slot-major. Invalid device IDs produce non-finite outputs.
func (r *cudaFoundation) encodeMatvecIQ2Selected(gate, up cudaModelTensor, ids cudamem.Slice[int32], in, gateOut, upOut cudamem.Slice[float32]) error {
	if err := cudaExpertProfile(gate, 16, 66); err != nil {
		return err
	}
	if err := cudaExpertProfile(up, 16, 66); err != nil {
		return err
	}
	for i := range gate.dimensions {
		if gate.dimensions[i] != up.dimensions[i] {
			return fmt.Errorf("cuda: IQ2 selected pair: mismatched dimensions")
		}
	}
	cols, rows, experts := gate.dimensions[0], gate.dimensions[1], gate.dimensions[2]
	slots := uint64(ids.Len())
	if slots == 0 || slots > math.MaxInt32/rows || uint64(in.Len()) != cols || uint64(gateOut.Len()) != slots*rows || upOut.Len() != gateOut.Len() {
		return fmt.Errorf("cuda: IQ2 selected pair: invalid input or output shape")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.iq2SelectedFn).Grid(uint32(rows*slots)).Block(256).On(r.device.DefaultStream()).Launch(gate.device, up.device, ids, in, gateOut, upOut, uint32(cols), uint32(rows), uint32(experts)); err != nil {
			return fmt.Errorf("cuda: launch IQ2 selected pair: %w", err)
		}
		return nil
	})
}

// encodeMatvecQ2KSelected appends Q2_K projections for GPU-selected experts.
// Inputs and outputs are slot-major. Invalid IDs produce non-finite outputs.
func (r *cudaFoundation) encodeMatvecQ2KSelected(t cudaModelTensor, ids cudamem.Slice[int32], in, out cudamem.Slice[float32]) error {
	if err := cudaExpertProfile(t, 10, 84); err != nil {
		return err
	}
	cols, rows, experts := t.dimensions[0], t.dimensions[1], t.dimensions[2]
	slots := uint64(ids.Len())
	if slots == 0 || slots > math.MaxInt32/rows || uint64(in.Len()) != cols*slots || uint64(out.Len()) != rows*slots {
		return fmt.Errorf("cuda: Q2_K selected: invalid input or output shape")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.q2KSelectedFn).Grid(uint32(rows*slots)).Block(256).On(r.device.DefaultStream()).Launch(t.device, ids, in, out, uint32(cols), uint32(rows), uint32(experts)); err != nil {
			return fmt.Errorf("cuda: launch Q2_K selected: %w", err)
		}
		return nil
	})
}

// encodeQuantQ8K appends CUDA-oracle Q8_K activation quantization. Each row
// contains cols elements and each 256-element block occupies 292 bytes.
func (r *cudaFoundation) encodeQuantQ8K(in cudamem.Slice[float32], out cudamem.Slice[byte], cols int) error {
	if cols <= 0 || cols%256 != 0 || in.Len() == 0 || in.Len()%cols != 0 || uint64(in.Len()/256) > math.MaxInt32 || uint64(out.Len()) != uint64(in.Len()/256)*292 {
		return fmt.Errorf("cuda: Q8_K quantization: invalid shape")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.quantQ8KFn).Grid(uint32(in.Len()/256)).Block(256).On(r.device.DefaultStream()).Launch(in, out); err != nil {
			return fmt.Errorf("cuda: launch Q8_K quantization: %w", err)
		}
		return nil
	})
}

// encodeMatvecIQ2SelectedQ8K appends paired projections using Q8_K activations.
func (r *cudaFoundation) encodeMatvecIQ2SelectedQ8K(gate, up cudaModelTensor, ids cudamem.Slice[int32], in cudamem.Slice[byte], gateOut, upOut cudamem.Slice[float32]) error {
	if err := cudaExpertProfile(gate, 16, 66); err != nil {
		return err
	}
	if err := cudaExpertProfile(up, 16, 66); err != nil {
		return err
	}
	for i := range gate.dimensions {
		if gate.dimensions[i] != up.dimensions[i] {
			return fmt.Errorf("cuda: IQ2 Q8_K selected pair: mismatched dimensions")
		}
	}
	cols, rows, experts := gate.dimensions[0], gate.dimensions[1], gate.dimensions[2]
	slots := uint64(ids.Len())
	if slots == 0 || slots > math.MaxInt32/rows || uint64(in.Len()) != cols/256*292 || uint64(gateOut.Len()) != slots*rows || upOut.Len() != gateOut.Len() {
		return fmt.Errorf("cuda: IQ2 Q8_K selected pair: invalid shape")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.iq2SelectedQ8KFn).Grid(uint32(rows*slots)).Block(256).On(r.device.DefaultStream()).Launch(gate.device, up.device, ids, in, gateOut, upOut, uint32(cols), uint32(rows), uint32(experts)); err != nil {
			return fmt.Errorf("cuda: launch IQ2 Q8_K selected pair: %w", err)
		}
		return nil
	})
}

// encodeMatvecQ2KSelectedQ8K appends down projections with slot-major Q8_K input.
func (r *cudaFoundation) encodeMatvecQ2KSelectedQ8K(t cudaModelTensor, ids cudamem.Slice[int32], in cudamem.Slice[byte], out cudamem.Slice[float32]) error {
	if err := cudaExpertProfile(t, 10, 84); err != nil {
		return err
	}
	cols, rows, experts := t.dimensions[0], t.dimensions[1], t.dimensions[2]
	slots := uint64(ids.Len())
	if slots == 0 || slots > math.MaxInt32/rows || uint64(in.Len()) != cols/256*292*slots || uint64(out.Len()) != rows*slots {
		return fmt.Errorf("cuda: Q2_K Q8_K selected: invalid shape")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.q2KSelectedQ8KFn).Grid(uint32(rows*slots)).Block(256).On(r.device.DefaultStream()).Launch(t.device, ids, in, out, uint32(cols), uint32(rows), uint32(experts)); err != nil {
			return fmt.Errorf("cuda: launch Q2_K Q8_K selected: %w", err)
		}
		return nil
	})
}
