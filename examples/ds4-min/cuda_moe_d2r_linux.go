//go:build linux

package main

import (
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

const (
	cudaIQ2XXSRepackedBytes = 553648128
	cudaQ2KRepackedBytes    = 704643072
)

// encodeQuantMMQ quantizes float activations into column-chunked block_q8_1_mmq.
func (r *cudaFoundation) encodeQuantMMQ(in cudamem.Slice[float32], out cudamem.Slice[byte], stride, cols, nTokens int) error {
	if cols <= 0 || cols%128 != 0 || stride < cols || nTokens <= 0 ||
		uint64(out.Len()) < uint64(cols/128)*uint64(nTokens)*144 {
		return fmt.Errorf("cuda: quantize MMQ Q8_1: invalid shape")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		grid := [3]uint32{uint32(nTokens), uint32(cols / 128), 1}
		block := [3]uint32{32, 1, 1}
		if err := cudalaunch.Kernel(r.quantMMQFn).Grid(grid[0], grid[1], grid[2]).Block(block[0], block[1], block[2]).On(r.stream()).Launch(
			in, out, int64(stride), int64(cols), int32(nTokens)); err != nil {
			return fmt.Errorf("cuda: launch quantize MMQ Q8_1: %w", err)
		}
		return nil
	})
}

// encodeD2RMakeIdsSrc converts assignment pair IDs to token source row indices.
func (r *cudaFoundation) encodeD2RMakeIdsSrc(pairs cudamem.Slice[int32], idsSrc cudamem.Slice[int32], nAssign int) error {
	if nAssign <= 0 || pairs.Len() < nAssign || idsSrc.Len() < nAssign {
		return fmt.Errorf("cuda: D2R make ids_src: invalid shape")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		grid := uint32((nAssign + 255) / 256)
		if err := cudalaunch.Kernel(r.d2rMakeIdsSrcFn).Grid(grid).Block(256).On(r.stream()).Launch(
			pairs, idsSrc, int32(nAssign)); err != nil {
			return fmt.Errorf("cuda: launch D2R make ids_src: %w", err)
		}
		return nil
	})
}

// encodeD2RBuildWorklist32 builds the 32-column tile worklist for Gate/Up GEMMs.
func (r *cudaFoundation) encodeD2RBuildWorklist32(expertBounds, work, nItems cudamem.Slice[int32], nExperts int) error {
	if nExperts <= 0 || expertBounds.Len() < nExperts+1 || work.Len() == 0 || nItems.Len() == 0 {
		return fmt.Errorf("cuda: D2R build worklist 32: invalid shape")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.d2rBuild32Fn).Grid(1).Block(256).On(r.stream()).Launch(
			expertBounds, work, nItems, int32(nExperts)); err != nil {
			return fmt.Errorf("cuda: launch D2R build worklist 32: %w", err)
		}
		return nil
	})
}

// encodeD2RBuildWorklist64 builds the 64-column tile worklist for Down GEMMs.
func (r *cudaFoundation) encodeD2RBuildWorklist64(expertBounds, work, nItems cudamem.Slice[int32], nExperts int) error {
	if nExperts <= 0 || expertBounds.Len() < nExperts+1 || work.Len() == 0 || nItems.Len() == 0 {
		return fmt.Errorf("cuda: D2R build worklist 64: invalid shape")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.d2rBuild64Fn).Grid(1).Block(256).On(r.stream()).Launch(
			expertBounds, work, nItems, int32(nExperts)); err != nil {
			return fmt.Errorf("cuda: launch D2R build worklist 64: %w", err)
		}
		return nil
	})
}

// encodeGateUpD2R executes fused Gate + Up GEMMs, SwiGLU, router weighting, and Q8_1 quantization.
func (r *cudaFoundation) encodeGateUpD2R(
	gateSoa, upSoa cudamem.Slice[byte],
	inputQ8 cudamem.Slice[byte],
	idsSrc, idsDst, expertBounds cudamem.Slice[int32],
	routerWeights cudamem.Slice[float32],
	work, nItems cudamem.Slice[int32],
	downQ8 cudamem.Slice[byte],
	m, k, nAssign, nTokens, nExperts int,
	clamp float32,
) error {
	capacity32 := (nAssign+31)/32 + nExperts
	if m <= 0 || m%128 != 0 || k <= 0 || k%256 != 0 || nAssign <= 0 || nTokens <= 0 || nExperts <= 0 ||
		gateSoa.Len() < cudaIQ2XXSRepackedBytes || upSoa.Len() < cudaIQ2XXSRepackedBytes ||
		idsSrc.Len() < nAssign || idsDst.Len() < nAssign || expertBounds.Len() < nExperts+1 ||
		routerWeights.Len() < nAssign || work.Len() < capacity32 || nItems.Len() < 1 ||
		downQ8.Len() < (m/128)*nAssign*144 {
		return fmt.Errorf("cuda: D2R gate/up: invalid shape")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		grid := [3]uint32{uint32(m / 128), uint32(capacity32), 1}
		block := [3]uint32{32, 8, 1}
		if err := cudalaunch.Kernel(r.gateupD2RFn).Grid(grid[0], grid[1], grid[2]).Block(block[0], block[1], block[2]).On(r.stream()).Launch(
			gateSoa, upSoa, inputQ8, idsSrc, idsDst, expertBounds, routerWeights,
			work, nItems, downQ8, int32(m), int32(k), int32(nAssign), int32(nTokens), int32(nExperts), clamp); err != nil {
			return fmt.Errorf("cuda: launch D2R gate/up: %w", err)
		}
		return nil
	})
}

// encodeDownD2R executes the fused Down GEMM reading downQ8 directly.
func (r *cudaFoundation) encodeDownD2R(
	wSoa cudamem.Slice[byte],
	q8 cudamem.Slice[byte],
	idsDst, expertBounds, work, nItems cudamem.Slice[int32],
	out cudamem.Slice[float32],
	m, k, nAssign, nExperts int,
) error {
	capacity64 := (nAssign+63)/64 + nExperts
	if m <= 0 || k <= 0 || k%256 != 0 || nAssign <= 0 || nExperts <= 0 ||
		wSoa.Len() < cudaQ2KRepackedBytes || idsDst.Len() < nAssign ||
		expertBounds.Len() < nExperts+1 || work.Len() < capacity64 || nItems.Len() < 1 ||
		out.Len() < nAssign*m {
		return fmt.Errorf("cuda: D2R down: invalid shape")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		grid := [3]uint32{uint32((m + 127) / 128), uint32(capacity64), 1}
		block := [3]uint32{32, 8, 1}
		if err := cudalaunch.Kernel(r.downD2RFn).Grid(grid[0], grid[1], grid[2]).Block(block[0], block[1], block[2]).On(r.stream()).Launch(
			wSoa, q8, idsDst, expertBounds, work, nItems, out,
			int32(m), int32(k), int32(nAssign), int32(nExperts)); err != nil {
			return fmt.Errorf("cuda: launch D2R down: %w", err)
		}
		return nil
	})
}

// repackIQ2 repacks raw IQ2_XXS expert weights into SOA format.
func (r *cudaFoundation) repackIQ2(dst cudamem.Slice[byte], src driver.DevicePtr, nblk uint64) error {
	dqBytes := (nblk*2 + 63) & ^uint64(63)
	qsBytes := nblk * 8 * 8
	if uint64(dst.Len()) < dqBytes+qsBytes {
		return fmt.Errorf("cuda: repack IQ2_XXS: dst buffer too small")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		dq := dst.Slice(0, int(dqBytes))
		qs := dst.Slice(int(dqBytes), int(dqBytes+qsBytes))
		grid := uint32((nblk*8 + 255) / 256)
		if err := cudalaunch.Kernel(r.repackIQ2Fn).Grid(grid).Block(256).On(r.stream()).Launch(
			dq, qs, src, nblk); err != nil {
			return fmt.Errorf("cuda: launch repack IQ2_XXS: %w", err)
		}
		return nil
	})
}

// repackQ2K repacks raw Q2_K expert weights into SOA format.
func (r *cudaFoundation) repackQ2K(dst cudamem.Slice[byte], src driver.DevicePtr, g0, cblk uint64, nbRow, nrows uint32) error {
	npair := uint64(256) * uint64(nrows/2) * uint64(nbRow)
	dmBytes := (npair*8 + 63) & ^uint64(63)
	scBytes := (npair*32 + 63) & ^uint64(63)
	qsBytes := npair * 32 * 4
	if uint64(dst.Len()) < dmBytes+scBytes+qsBytes {
		return fmt.Errorf("cuda: repack Q2_K: dst buffer too small")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		dm2 := dst.Slice(0, int(dmBytes))
		sc4 := dst.Slice(int(dmBytes), int(dmBytes+scBytes))
		qs2 := dst.Slice(int(dmBytes+scBytes), int(dmBytes+scBytes+qsBytes))
		grid := uint32((cblk*16 + 255) / 256)
		if err := cudalaunch.Kernel(r.repackQ2KFn).Grid(grid).Block(256).On(r.stream()).Launch(
			dm2, sc4, qs2, src, g0, cblk, nbRow, nrows); err != nil {
			return fmt.Errorf("cuda: launch repack Q2_K: %w", err)
		}
		return nil
	})
}

// repackLayerMoE repacks one layer's Gate, Up, and Down MoE weights into the provided SOA scratch buffers.
func (r *cudaFoundation) repackLayerMoE(l *cudaLayer, gateDst, upDst, downDst cudamem.Slice[byte]) error {
	const (
		iq2Nblk  = 256 * 2048 * 16
		q2kCblk  = 256 * 4096 * 8
		q2kNbRow = 8
		q2kNrows = 4096
	)
	if err := r.repackIQ2(gateDst, l.expertGate.device, iq2Nblk); err != nil {
		return fmt.Errorf("repack gate: %w", err)
	}
	if err := r.repackIQ2(upDst, l.expertUp.device, iq2Nblk); err != nil {
		return fmt.Errorf("repack up: %w", err)
	}
	if err := r.repackQ2K(downDst, l.expertDown.device, 0, q2kCblk, q2kNbRow, q2kNrows); err != nil {
		return fmt.Errorf("repack down: %w", err)
	}
	return nil
}
