//go:build linux

package main

import (
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// denseQ8AlignedBytes returns the byte capacity needed to store an aligned
// repacked Q8_0 weight matrix of dimensions [M, K].
func denseQ8AlignedBytes(m, k int) int {
	if m <= 0 || k <= 0 || k%32 != 0 {
		return 0
	}
	nblk := uint64(m) * uint64(k/32)
	dqBytes := (nblk*2 + 63) & ^uint64(63)
	return int(dqBytes + nblk*32)
}

// repackQ80 repacks raw 34-byte block_q8_0 weights into separate aligned
// half scale (dq) and int8 quant (qs) planes.
func (r *cudaFoundation) repackQ80(dst cudamem.Slice[byte], src driver.DevicePtr, nblk uint64) error {
	dqBytes := (nblk*2 + 63) & ^uint64(63)
	qsBytes := nblk * 32
	if uint64(dst.Len()) < dqBytes+qsBytes {
		return fmt.Errorf("cuda: repack Q8_0: dst buffer too small (%d < %d)", dst.Len(), dqBytes+qsBytes)
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		dq := dst.Slice(0, int(dqBytes))
		qs := dst.Slice(int(dqBytes), int(dqBytes+qsBytes))
		grid := uint32((nblk*2 + 255) / 256)
		if err := cudalaunch.Kernel(r.repackQ80Fn).Grid(grid).Block(256).On(r.stream()).Launch(
			dq, qs, src, nblk); err != nil {
			return fmt.Errorf("cuda: launch repack Q8_0: %w", err)
		}
		return nil
	})
}

// encodeDenseQ8D2R executes a dense matrix-matrix multiplication on Turing/Ampere/Blackwell
// INT8 Tensor Cores using m16n8k32 MMA instructions directly against block_q8_1_mmq activations.
func (r *cudaFoundation) encodeDenseQ8D2R(
	wAligned cudamem.Slice[byte],
	q8 cudamem.Slice[byte],
	out cudamem.Slice[float32],
	m, n, k, outStride int,
) error {
	if m <= 0 || m%128 != 0 || n <= 0 || k <= 0 || k%1024 != 0 {
		return fmt.Errorf("cuda: dense Q8 D2R: invalid shape m=%d n=%d k=%d", m, n, k)
	}
	nblk := uint64(m) * uint64(k/32)
	dqBytes := (nblk*2 + 63) & ^uint64(63)
	qsBytes := nblk * 32
	if uint64(wAligned.Len()) < dqBytes+qsBytes {
		return fmt.Errorf("cuda: dense Q8 D2R: wAligned buffer too small (%d < %d)", wAligned.Len(), dqBytes+qsBytes)
	}
	if outStride <= 0 {
		outStride = m
	}
	reqQ8Blocks := (k/128)*n + 128
	if q8.Len() < reqQ8Blocks*144 {
		return fmt.Errorf("cuda: dense Q8 D2R: q8 buffer too small (%d < %d)", q8.Len(), reqQ8Blocks*144)
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		wdPlane := wAligned.Slice(0, int(dqBytes))
		wqPlane := wAligned.Slice(int(dqBytes), int(dqBytes+qsBytes))
		groupM := int32(4)
		if k <= 2048 {
			groupM = 1
		}
		numM := m / 128
		numN := (n + 127) / 128
		grid := uint32(numM * numN)
		block := [3]uint32{32, 16, 1}
		const smem = 69120
		if err := cudalaunch.Kernel(r.denseQ8D2RFn).Grid(grid).Block(block[0], block[1], block[2]).SharedMem(smem).On(r.stream()).Launch(
			wdPlane, wqPlane, q8, out, int32(m), int32(n), int32(k), groupM, int32(outStride)); err != nil {
			return fmt.Errorf("cuda: launch dense Q8 D2R: %w", err)
		}
		return nil
	})
}

// repackLayerDenseQ8 repacks all dense Q8_0 weights for one layer into scratch buffers.
func (r *cudaFoundation) repackLayerDenseQ8(l *cudaLayer, d *cudaBatchDense) error {
	// sharedGate: [2048, 4096]
	if err := r.repackQ80(d.denseGate, l.sharedGate.device, 2048*(4096/32)); err != nil {
		return fmt.Errorf("repack sharedGate: %w", err)
	}
	// sharedUp: [2048, 4096]
	if err := r.repackQ80(d.denseUp, l.sharedUp.device, 2048*(4096/32)); err != nil {
		return fmt.Errorf("repack sharedUp: %w", err)
	}
	// sharedDown: [4096, 2048]
	if err := r.repackQ80(d.denseDown, l.sharedDown.device, 4096*(2048/32)); err != nil {
		return fmt.Errorf("repack sharedDown: %w", err)
	}
	// attnOutB: [4096, 8192]
	if err := r.repackQ80(d.denseAttnOutB, l.attnOutB.device, 4096*(8192/32)); err != nil {
		return fmt.Errorf("repack attnOutB: %w", err)
	}
	// attnOutA: 8 diagonal blocks, each [1024, 4096]
	const blockAlignedBytes = 4456448 // denseQ8AlignedBytes(1024, 4096)
	const blockRawBytes = 1024 * (4096 / 32 * 34)
	for g := range 8 {
		rawG := l.attnOutA.device + driver.DevicePtr(uint64(g)*blockRawBytes)
		dstG := d.denseAttnOutA.Slice(g*blockAlignedBytes, (g+1)*blockAlignedBytes)
		if err := r.repackQ80(dstG, rawG, 1024*(4096/32)); err != nil {
			return fmt.Errorf("repack attnOutA block %d: %w", g, err)
		}
	}
	return nil
}
