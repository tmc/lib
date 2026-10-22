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
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		dq := dst.Slice(0, int(dqBytes))
		qs := dst.Slice(int(dqBytes), int(dqBytes+qsBytes))
		grid := uint32((nblk*2 + 255) / 256)
		if err := cudalaunch.Kernel(r.repackQ80Fn).Grid(grid, 1, 1).Block(256, 1, 1).On(r.stream()).Launch(
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
	return r.withContext(func() error {
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
		if err := cudalaunch.Kernel(r.denseQ8D2RFn).Grid(grid, 1, 1).Block(block[0], block[1], block[2]).SharedMem(smem).On(r.stream()).Launch(
			wdPlane, wqPlane, q8, out, int32(m), int32(n), int32(k), groupM, int32(outStride)); err != nil {
			return fmt.Errorf("cuda: launch dense Q8 D2R: %w", err)
		}
		r.denseQ8D2RLaunches++
		return nil
	})
}

// encodeDenseQ8D2RBlockDiag executes a block-diagonal dense matrix-matrix multiplication on
// Turing/Ampere/Blackwell INT8 Tensor Cores with numBlocks independent (m x k) @ (k x n) matrix products
// in a single kernel launch.
func (r *cudaFoundation) encodeDenseQ8D2RBlockDiag(
	wAligned cudamem.Slice[byte],
	q8 cudamem.Slice[byte],
	out cudamem.Slice[float32],
	m, n, k, outStride, numBlocks int,
) error {
	if m <= 0 || m%128 != 0 || n <= 0 || k <= 0 || k%1024 != 0 || numBlocks <= 0 {
		return fmt.Errorf("cuda: dense Q8 D2R blockdiag: invalid shape m=%d n=%d k=%d numBlocks=%d", m, n, k, numBlocks)
	}
	nblk := uint64(m) * uint64(k/32)
	dqBytes := (nblk*2 + 63) & ^uint64(63)
	qsBytes := nblk * 32
	blockWBytes := dqBytes + qsBytes
	totalWBytes := blockWBytes * uint64(numBlocks)
	if uint64(wAligned.Len()) < totalWBytes {
		return fmt.Errorf("cuda: dense Q8 D2R blockdiag: wAligned buffer too small (%d < %d)", wAligned.Len(), totalWBytes)
	}
	if outStride <= 0 {
		outStride = m * numBlocks
	}
	blockQBlocks := uint64(k/128) * uint64(n)
	reqQ8Blocks := blockQBlocks*uint64(numBlocks) + 128
	if uint64(q8.Len()) < reqQ8Blocks*144 {
		return fmt.Errorf("cuda: dense Q8 D2R blockdiag: q8 buffer too small (%d < %d)", q8.Len(), reqQ8Blocks*144)
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		groupM := int32(4)
		if k <= 2048 {
			groupM = 1
		}
		numM := m / 128
		numN := (n + 127) / 128
		ctasPerBlock := numM * numN
		grid := uint32(ctasPerBlock * numBlocks)
		block := [3]uint32{32, 16, 1}
		const smem = 69120
		if err := cudalaunch.Kernel(r.denseQ8D2RBlockDiagFn).Grid(grid, 1, 1).Block(block[0], block[1], block[2]).SharedMem(smem).On(r.stream()).Launch(
			wAligned, q8, out, int32(m), int32(n), int32(k), groupM, int32(outStride),
			int32(numBlocks), uint64(blockWBytes), uint64(blockQBlocks)); err != nil {
			return fmt.Errorf("cuda: launch dense Q8 D2R blockdiag: %w", err)
		}
		r.denseQ8D2RLaunches++
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

// repackLayerAttnQ8 repacks one layer's query and key/value projections into
// the aligned dq/qs planes the D2R GEMM reads. Those three matrices were all of
// ds4_q8_0_matvec_q8_1_batch's work: that kernel holds one accumulator per
// thread per token and caps the batch at 32, so a group of 2048 tokens streams
// each weight matrix 64 times. The GEMM reads it once.
func (r *cudaFoundation) repackLayerAttnQ8(l *cudaLayer, d *cudaBatchDense) error {
	for _, w := range []struct {
		name string
		t    cudaModelTensor
		dst  cudamem.Slice[byte]
		m, k int
	}{
		{"attnQA", l.attnQA, d.denseAttnQA, cudaQLoraWidth, cudaWidth},
		{"attnQB", l.attnQB, d.denseAttnQB, cudaAttentionHeads, cudaQLoraWidth},
		{"attnKV", l.attnKV, d.denseAttnKV, cudaKVWidth, cudaWidth},
	} {
		if w.t.typeID != 8 || len(w.t.dimensions) != 2 ||
			w.t.dimensions[0] != uint64(w.k) || w.t.dimensions[1] != uint64(w.m) {
			return fmt.Errorf("repack %s: want [%d, %d] Q8_0, have %v type %d", w.name, w.k, w.m, w.t.dimensions, w.t.typeID)
		}
		if err := r.repackQ80(w.dst, w.t.device, uint64(w.m)*uint64(w.k/32)); err != nil {
			return fmt.Errorf("repack %s: %w", w.name, err)
		}
	}
	return nil
}

// cudaDenseArtifact is one layer's dense Q8_0 weights in the aligned dq/qs
// layout the D2R GEMM reads, built once at load instead of once per prefill
// group. The eight matrices are the same for every token in the group and do
// not depend on the batch, so nothing but the repack cost made them scratch.
type cudaDenseArtifact struct {
	gate, up, down         cudamem.Slice[byte]
	attnOutB, attnOutA     cudamem.Slice[byte]
	attnQA, attnQB, attnKV cudamem.Slice[byte]
}

func (a *cudaDenseArtifact) built() bool { return a.gate.Len() > 0 }

// slices returns the eight planes in the order buildDenseArtifact fills them.
func (a *cudaDenseArtifact) slices() []*cudamem.Slice[byte] {
	return []*cudamem.Slice[byte]{&a.gate, &a.up, &a.down, &a.attnOutB, &a.attnOutA, &a.attnQA, &a.attnQB, &a.attnKV}
}

// digest folds a strided sample of every plane into one FNV-1a value. It is
// the same shape as cudaMoEArtifact.digest and exists for the same reason:
// taken once after the build and once after the prefill, a pair of equal
// digests says the artifact was intact the whole time, so a wrong logit came
// from somewhere else.
func (a *cudaDenseArtifact) digest(h uint64, buf []byte) (uint64, error) {
	for _, s := range a.slices() {
		const samples = 16
		step := s.Len() / samples
		if step < len(buf) {
			continue
		}
		for i := range samples {
			lo := i * step
			if err := s.Slice(lo, lo+len(buf)).CopyToHost(buf); err != nil {
				return 0, err
			}
			for _, b := range buf {
				h = (h ^ uint64(b)) * 1099511628211
			}
		}
	}
	return h, nil
}

func (a *cudaDenseArtifact) free() error {
	return a.freeWith(defaultCUDAChildCleanupOperations().bytes)
}

func (a *cudaDenseArtifact) Close() error { return a.free() }

func (a *cudaDenseArtifact) freeWith(release func(*cudamem.Slice[byte]) error) error {
	return freeCUDAByteSlices(a.slices(), release)
}

// cudaDenseArtifactBytes is what one layer's dense artifact occupies: two
// orders of magnitude less than the expert artifact, because these are the
// shared and attention weights rather than 256 experts.
const cudaDenseArtifactBytes = 3*denseQ8AlignedGate + denseQ8AlignedOutB + 8*denseQ8AlignedABlock +
	denseQ8AlignedQA + denseQ8AlignedQB + denseQ8AlignedKV

// The aligned sizes are fixed by the architecture, so they are constants here
// rather than calls; denseQ8AlignedBytes computes the same values.
const (
	denseQ8AlignedGate   = 2048 * (4096 / 32) * 34
	denseQ8AlignedOutB   = 4096 * (8192 / 32) * 34
	denseQ8AlignedABlock = 1024 * (4096 / 32) * 34
	denseQ8AlignedQA     = cudaQLoraWidth * (cudaWidth / 32) * 34
	denseQ8AlignedQB     = cudaAttentionHeads * (cudaQLoraWidth / 32) * 34
	denseQ8AlignedKV     = cudaKVWidth * (cudaWidth / 32) * 34
)

// buildDenseTensor repacks one dense Q8_0 tensor into freshly allocated device
// memory, deriving its shape from the tensor instead of from the architecture
// constants. buildDenseArtifact names its shapes because a wrong one there
// means a mis-bound layer weight; the output projection has no sibling it could
// be confused with, so the tensor itself is the authority.
//
// The mapping is not released here. The output projection is the one dense
// matvec that still has a fallback reading the model mapping, and dropping the
// pages would make that fallback fault them straight back in.
func (r *cudaFoundation) buildDenseTensor(dst *cudamem.Slice[byte], src *cudaModelTensor, s *cudaMoEStage) error {
	if src.typeID != 8 || len(src.dimensions) != 2 || src.dimensions[0]%32 != 0 {
		return fmt.Errorf("dense tensor %s: want 2-D Q8_0 with a multiple of 32 columns, have %v type %d",
			src.name, src.dimensions, src.typeID)
	}
	m, k := src.dimensions[1], src.dimensions[0]
	nblk := m * (k / 32)
	if src.bytes != nblk*34 {
		return fmt.Errorf("dense tensor %s: want %d bytes of Q8_0, have %d", src.name, nblk*34, src.bytes)
	}
	out, err := cudamem.Alloc[byte](r.device, denseQ8AlignedBytes(int(m), int(k)))
	*dst = out
	if err != nil {
		return fmt.Errorf("allocate %s artifact: %w", src.name, err)
	}
	if err := r.load(s, src); err != nil {
		return err
	}
	if err := r.repackQ80(out, s.dev.Ptr(), nblk); err != nil {
		return fmt.Errorf("repack %s: %w", src.name, err)
	}
	return syncStream(r.stream())
}

// buildDenseArtifact repacks one layer's dense Q8_0 weights into freshly
// allocated device memory, reading each source off disk through stage rather
// than through the model mapping.
func (r *cudaFoundation) buildDenseArtifact(l *cudaLayer, a *cudaDenseArtifact, s *cudaMoEStage) (err error) {
	defer func() {
		if err != nil {
			err = finishBackendCleanup(err, a)
		}
	}()
	// attnOutA is eight diagonal blocks in one tensor, repacked block by block.
	const aBlocks = 8
	for _, part := range []struct {
		dst    *cudamem.Slice[byte]
		src    *cudaModelTensor
		m, k   int
		blocks int
	}{
		{&a.gate, &l.sharedGate, 2048, 4096, 1},
		{&a.up, &l.sharedUp, 2048, 4096, 1},
		{&a.down, &l.sharedDown, 4096, 2048, 1},
		{&a.attnOutB, &l.attnOutB, 4096, 8192, 1},
		{&a.attnOutA, &l.attnOutA, 1024, 4096, aBlocks},
		{&a.attnQA, &l.attnQA, cudaQLoraWidth, cudaWidth, 1},
		{&a.attnQB, &l.attnQB, cudaAttentionHeads, cudaQLoraWidth, 1},
		{&a.attnKV, &l.attnKV, cudaKVWidth, cudaWidth, 1},
	} {
		nblk := uint64(part.m) * uint64(part.k/32)
		if part.src.typeID != 8 || part.src.bytes != uint64(part.blocks)*nblk*34 {
			return fmt.Errorf("dense artifact %s: want %d bytes of Q8_0, have %d bytes of type %d",
				part.src.name, uint64(part.blocks)*nblk*34, part.src.bytes, part.src.typeID)
		}
		aligned := denseQ8AlignedBytes(part.m, part.k)
		dst, err := cudamem.Alloc[byte](r.device, part.blocks*aligned)
		*part.dst = dst
		if err != nil {
			return fmt.Errorf("allocate %s artifact: %w", part.src.name, err)
		}
		if err := r.load(s, part.src); err != nil {
			return err
		}
		for g := range part.blocks {
			src := s.dev.Ptr() + driver.DevicePtr(uint64(g)*nblk*34)
			if err := r.repackQ80(dst.Slice(g*aligned, (g+1)*aligned), src, nblk); err != nil {
				return fmt.Errorf("repack %s block %d: %w", part.src.name, g, err)
			}
		}
		// The next load overwrites the stage, and the next allocation
		// reclaims; both have to wait for this repack to finish reading.
		if err := syncStream(r.stream()); err != nil {
			return err
		}
		// The artifact now holds these bytes, so the mapping's copy is
		// dead weight against the working set that binds the budget.
		if err := r.releaseMapped(part.src); err != nil {
			return err
		}
	}
	return nil
}
