//go:build linux

package main

import (
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func cudaModelNeedsQ4(m *cudaModel) bool {
	for i := range m.layers {
		if m.layers[i].expertGate.typeID == typeQ4K {
			return true
		}
	}
	return false
}

func (r *cudaRuntime) needsMoED2RScratch() bool {
	if !r.d2rEnabled() {
		return false
	}
	for i := range r.model.layers {
		if r.model.layers[i].expertGate.typeID == typeIQ2XXS && !r.moeArtifacts[i].built() {
			return true
		}
	}
	return false
}

func (r *cudaFoundation) encodeMatvecQ4KSelectedQ8K(t cudaModelTensor, ids cudamem.Slice[int32], in cudamem.Slice[byte], out cudamem.Slice[float32], slotted bool) error {
	if err := cudaExpertProfile(t, typeQ4K, 144); err != nil {
		return err
	}
	cols, rows, experts := t.dimensions[0], t.dimensions[1], t.dimensions[2]
	slots, inputs := uint64(ids.Len()), uint64(1)
	if slotted {
		inputs = slots
	}
	if slots == 0 || slots > math.MaxInt32/rows || uint64(in.Len()) != cols/256*292*inputs || uint64(out.Len()) != rows*slots {
		return fmt.Errorf("cuda: Q4_K selected: invalid shape")
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		var perSlot uint32
		if slotted {
			perSlot = 1
		}
		return cudalaunch.Kernel(r.q4KSelectedQ8KFn).Grid(uint32(rows*slots), 1, 1).Block(256, 1, 1).On(r.stream()).Launch(t.device, ids, in, out, uint32(cols), uint32(rows), uint32(experts), perSlot)
	})
}

func (r *cudaRuntime) evalQ4Experts(s *cudaTokenState, index int) error {
	l := &r.model.layers[index]
	if _, err := ExpertFormatForTypes(l.expertGate.typeID, l.expertUp.typeID, l.expertDown.typeID); err != nil {
		return err
	}
	if err := r.encodeQuantQ8K(s.ffnPre.Slice(cudaWidth, 2*cudaWidth), s.q8KInput, cudaWidth); err != nil {
		return err
	}
	if err := r.encodeMatvecQ4KSelectedQ8K(l.expertGate, s.expertIDs, s.q8KInput, s.expertGate, false); err != nil {
		return err
	}
	if err := r.encodeMatvecQ4KSelectedQ8K(l.expertUp, s.expertIDs, s.q8KInput, s.expertUp, false); err != nil {
		return err
	}
	if err := r.encodeSwiGLUSlots(s.expertGate, s.expertUp, s.expertW, s.expertMid, 6, 1); err != nil {
		return err
	}
	if err := r.encodeQuantQ8K(s.expertMid, s.q8KExperts, 2048); err != nil {
		return err
	}
	return r.encodeMatvecQ4KSelectedQ8K(l.expertDown, s.expertIDs, s.q8KExperts, s.expertDown, true)
}
