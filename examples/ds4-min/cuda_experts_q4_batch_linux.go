//go:build linux

package main

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func q4GatherShape(t cudaModelTensor, g *cudaMoEGrouping, slots, inputBytes, outputValues int) error {
	if g == nil {
		return fmt.Errorf("cuda: Q4_K gather: missing grouping")
	}
	if err := cudaExpertProfile(t, typeQ4K, 144); err != nil {
		return err
	}
	cols, rows, experts := t.dimensions[0], t.dimensions[1], t.dimensions[2]
	if err := g.profile(experts); err != nil {
		return err
	}
	if g.offsets.Ptr() == 0 || g.pairs.Ptr() == 0 {
		return fmt.Errorf("cuda: Q4_K gather: released grouping")
	}
	// Grid y includes the sentinel invalid-ID bin. SSD compact groupings
	// may exceed the 256 logical experts, but cannot exceed CUDA grid y.
	if experts > 65534 {
		return fmt.Errorf("cuda: Q4_K gather: expert count exceeds launch bound")
	}
	return q4BatchShape(cols, rows, g.npairs, slots, inputBytes, outputValues)
}

func (r *cudaFoundation) encodeQ4KPairGather(gate, up cudaModelTensor, g *cudaMoEGrouping, input cudamem.Slice[byte], gateOut, upOut cudamem.Slice[float32]) error {
	if g == nil {
		return fmt.Errorf("cuda: Q4_K gather: missing grouping")
	}
	if r == nil || r.device == nil || r.closing || input.Ptr() == 0 || gateOut.Ptr() == 0 || upOut.Ptr() == 0 {
		return fmt.Errorf("cuda: Q4_K gather: closed or missing operands")
	}
	if err := q4GatherShape(gate, g, 6, input.Len(), gateOut.Len()); err != nil {
		return err
	}
	if err := q4GatherShape(up, g, 6, input.Len(), upOut.Len()); err != nil {
		return err
	}
	if gate.dimensions[0] != up.dimensions[0] || gate.dimensions[1] != up.dimensions[1] {
		return fmt.Errorf("cuda: Q4_K gather: gate/up dimensions differ")
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		return cudalaunch.Kernel(r.q4KPairGatherFn).Grid(uint32(gate.dimensions[1]), uint32(g.experts+1), 1).Block(256, 1, 1).On(r.stream()).Launch(
			gate.device, up.device, g.offsets, g.pairs, input, gateOut, upOut,
			uint32(gate.dimensions[0]), uint32(gate.dimensions[1]), uint32(g.experts), uint32(6), uint32(g.npairs))
	})
}

func (r *cudaFoundation) encodeQ4KDownGather(t cudaModelTensor, g *cudaMoEGrouping, input cudamem.Slice[byte], out cudamem.Slice[float32]) error {
	if g == nil {
		return fmt.Errorf("cuda: Q4_K gather: missing grouping")
	}
	if r == nil || r.device == nil || r.closing || input.Ptr() == 0 || out.Ptr() == 0 {
		return fmt.Errorf("cuda: Q4_K gather: closed or missing operands")
	}
	if err := q4GatherShape(t, g, 1, input.Len(), out.Len()); err != nil {
		return err
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		return cudalaunch.Kernel(r.q4KDownGatherFn).Grid(uint32(t.dimensions[1]), uint32(g.experts+1), 1).Block(256, 1, 1).On(r.stream()).Launch(
			t.device, g.offsets, g.pairs, input, out, uint32(t.dimensions[0]), uint32(t.dimensions[1]), uint32(g.experts), uint32(g.npairs))
	})
}

func (r *cudaRuntime) evalBatchQ4Experts(rows []cudaTokenState, index int) error {
	if r == nil || r.cudaFoundation == nil || r.device == nil || r.closing || r.model == nil || r.batch == nil || r.batch.closing ||
		r.batch.grouping == nil || index < 0 || index >= len(r.model.layers) {
		return fmt.Errorf("cuda: Q4_K batch: closed or missing runtime")
	}
	l, d, n := &r.model.layers[index], &r.batch.dense, len(rows)
	format, err := ExpertFormatForTypes(l.expertGate.typeID, l.expertUp.typeID, l.expertDown.typeID)
	if err != nil {
		return err
	}
	if format != ExpertsQ4K {
		return fmt.Errorf("cuda: Q4_K batch: layer %d has format %s", index, format)
	}
	profile := r.collectors != nil && r.collectors.imatrix != nil
	c := q4BatchCapacity{norm: d.attnNorm.Len(), input: d.q4Input.Len(), gate: d.expertGate.Len(), up: d.expertUp.Len(),
		mid: d.q4Mid.Len(), quantMid: d.q4Experts.Len(), down: d.expertDown.Len(), ids: d.expertIDs.Len(), weights: d.expertW.Len(),
		pairs: r.batch.grouping.pairs.Len(), profile: d.profileMid.Len()}
	if err := checkQ4BatchCapacity(n, c, profile); err != nil {
		return err
	}
	for _, p := range []uint64{uint64(d.attnNorm.Ptr()), uint64(d.q4Input.Ptr()), uint64(d.expertGate.Ptr()), uint64(d.expertUp.Ptr()), uint64(d.q4Mid.Ptr()), uint64(d.q4Experts.Ptr()), uint64(d.expertDown.Ptr()), uint64(d.expertIDs.Ptr()), uint64(d.expertW.Ptr())} {
		if p == 0 {
			return fmt.Errorf("cuda: Q4_K batch: released scratch")
		}
	}
	if profile && d.profileMid.Ptr() == 0 {
		return fmt.Errorf("cuda: Q4_K batch: released profile scratch")
	}
	for i := range rows {
		if rows[i].ffnNorm.Len() != cudaWidth || rows[i].ffnNorm.Ptr() == 0 {
			return fmt.Errorf("cuda: Q4_K batch: invalid input row %d", i)
		}
	}
	for i, t := range []cudaModelTensor{l.expertGate, l.expertUp, l.expertDown} {
		if err := cudaExpertProfile(t, typeQ4K, 144); err != nil {
			return err
		}
		cols, outRows := uint64(cudaWidth), uint64(2048)
		if i == 2 {
			cols, outRows = 2048, cudaWidth
		}
		if t.dimensions[0] != cols || t.dimensions[1] != outRows {
			return fmt.Errorf("cuda: Q4_K batch: invalid expert dimensions")
		}
	}
	g := *r.batch.grouping
	g.npairs = n * 6
	g.pairs = g.pairs.Slice(0, g.npairs)
	if err := g.profile(l.expertGate.dimensions[2]); err != nil {
		return err
	}
	if g.active.Ptr() == 0 || g.count.Ptr() == 0 {
		return fmt.Errorf("cuda: Q4_K batch: released grouping")
	}
	if l.expertUp.dimensions[2] != l.expertGate.dimensions[2] || l.expertDown.dimensions[2] != l.expertGate.dimensions[2] {
		return fmt.Errorf("cuda: Q4_K batch: expert counts differ")
	}
	if err := q4GatherShape(l.expertGate, &g, 6, n*cudaWidth/256*292, n*6*2048); err != nil {
		return err
	}
	if err := q4GatherShape(l.expertDown, &g, 1, n*6*2048/256*292, n*6*cudaWidth); err != nil {
		return err
	}
	// Attention is complete. Its packed norm scratch is no longer live;
	// gather the strided FFN norms without changing the Q8_1 shared path.
	input := d.attnNorm.Slice(0, n*cudaWidth)
	for i := range rows {
		if err := input.Slice(i*cudaWidth, (i+1)*cudaWidth).CopyDeviceToDeviceAsync(rows[i].ffnNorm, r.stream()); err != nil {
			return fmt.Errorf("Q4_K batch input %d: %w", i, err)
		}
	}
	q := d.q4Input.Slice(0, n*cudaWidth/256*292)
	if err := r.encodeQuantQ8K(input, q, cudaWidth); err != nil {
		return err
	}
	if err := r.encodeMoEGroupCSR(d.expertIDs.Slice(0, g.npairs), &g); err != nil {
		return err
	}
	gate, up := d.expertGate.Slice(0, g.npairs*2048), d.expertUp.Slice(0, g.npairs*2048)
	if err := r.encodeQ4KPairGather(l.expertGate, l.expertUp, &g, q, gate, up); err != nil {
		return err
	}
	mid := d.q4Mid.Slice(0, g.npairs*2048)
	if err := r.encodeSwiGLUSlots(gate, up, d.expertW.Slice(0, g.npairs), mid, g.npairs, 1); err != nil {
		return err
	}
	if r.collectors != nil && r.collectors.imatrix != nil {
		if err := d.profileMid.Slice(0, mid.Len()).CopyDeviceToDeviceAsync(mid, r.stream()); err != nil {
			return fmt.Errorf("Q4_K batch profile mid: %w", err)
		}
	}
	qmid := d.q4Experts.Slice(0, g.npairs*2048/256*292)
	if err := r.encodeQuantQ8K(mid, qmid, 2048); err != nil {
		return err
	}
	return r.encodeQ4KDownGather(l.expertDown, &g, qmid, d.expertDown.Slice(0, g.npairs*cudaWidth))
}
