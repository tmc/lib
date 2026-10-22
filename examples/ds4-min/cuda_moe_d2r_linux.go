//go:build linux

package main

import (
	"errors"
	"fmt"
	"os"
	"syscall"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
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
	return r.withContext(func() error {
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

// encodeQuantMMQBlocks quantizes nSlices independent activation slices in
// one launch. The slices are separated by sliceStride input elements and the
// output is laid out as one block-major quantized region per slice.
func (r *cudaFoundation) encodeQuantMMQBlocks(in cudamem.Slice[float32], out cudamem.Slice[byte], stride, cols, nTokens, sliceStride, nSlices int) error {
	if cols <= 0 || cols%128 != 0 || stride < cols || nTokens <= 0 || nSlices <= 0 || sliceStride < cols ||
		uint64(out.Len()) < uint64(cols/128)*uint64(nTokens)*uint64(nSlices)*144 {
		return fmt.Errorf("cuda: quantize MMQ Q8_1 blocks: invalid shape")
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		grid := [3]uint32{uint32(nTokens), uint32(cols / 128), 1}
		block := [3]uint32{32, 1, 1}
		if err := cudalaunch.Kernel(r.quantMMQBlocksFn).Grid(grid[0], grid[1], grid[2]).Block(block[0], block[1], block[2]).On(r.stream()).Launch(
			in, out, int64(stride), int64(cols), int32(nTokens), int64(sliceStride), int32(nSlices)); err != nil {
			return fmt.Errorf("cuda: launch quantize MMQ Q8_1 blocks: %w", err)
		}
		return nil
	})
}

// encodeD2RMakeIdsSrc converts assignment pair IDs to token source row indices.
func (r *cudaFoundation) encodeD2RMakeIdsSrc(pairs cudamem.Slice[int32], idsSrc cudamem.Slice[int32], nAssign int) error {
	if nAssign <= 0 || pairs.Len() < nAssign || idsSrc.Len() < nAssign {
		return fmt.Errorf("cuda: D2R make ids_src: invalid shape")
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		grid := uint32((nAssign + 255) / 256)
		if err := cudalaunch.Kernel(r.d2rMakeIdsSrcFn).Grid(grid, 1, 1).Block(256, 1, 1).On(r.stream()).Launch(
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
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.d2rBuild32Fn).Grid(1, 1, 1).Block(256, 1, 1).On(r.stream()).Launch(
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
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.d2rBuild64Fn).Grid(1, 1, 1).Block(256, 1, 1).On(r.stream()).Launch(
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
	return r.withContext(func() error {
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

// encodeGateUpD2RMid uses the profiling-only entry point. mid is token-major,
// selected-slot-major and contains the actual weighted values before Q8_1.
func (r *cudaFoundation) encodeGateUpD2RMid(
	gateSoa, upSoa, inputQ8 cudamem.Slice[byte],
	idsSrc, idsDst, expertBounds cudamem.Slice[int32],
	routerWeights cudamem.Slice[float32], work, nItems cudamem.Slice[int32],
	downQ8 cudamem.Slice[byte], m, k, nAssign, nTokens, nExperts int,
	clamp float32, mid cudamem.Slice[float32],
) error {
	capacity := (nAssign+31)/32 + nExperts
	if m <= 0 || m%128 != 0 || k <= 0 || k%256 != 0 || nAssign <= 0 || nTokens <= 0 || nExperts <= 0 ||
		uint64(m)*uint64(nAssign)*4 > 256<<20 || uint64(mid.Len()) < uint64(m)*uint64(nAssign) ||
		gateSoa.Len() < cudaIQ2XXSRepackedBytes || upSoa.Len() < cudaIQ2XXSRepackedBytes ||
		idsSrc.Len() < nAssign || idsDst.Len() < nAssign || expertBounds.Len() < nExperts+1 ||
		routerWeights.Len() < nAssign || work.Len() < capacity || nItems.Len() < 1 ||
		uint64(downQ8.Len()) < uint64(m/128)*uint64(nAssign)*144 {
		return fmt.Errorf("cuda: D2R gate/up mid: invalid shape")
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.gateupD2RMidFn).Grid(uint32(m/128), uint32(capacity), 1).Block(32, 8, 1).On(r.stream()).Launch(
			gateSoa, upSoa, inputQ8, idsSrc, idsDst, expertBounds, routerWeights,
			work, nItems, downQ8, int32(m), int32(k), int32(nAssign), int32(nTokens), int32(nExperts), clamp, mid); err != nil {
			return fmt.Errorf("cuda: launch D2R gate/up mid: %w", err)
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
	return r.withContext(func() error {
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
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		dq := dst.Slice(0, int(dqBytes))
		qs := dst.Slice(int(dqBytes), int(dqBytes+qsBytes))
		grid := uint32((nblk*8 + 255) / 256)
		if err := cudalaunch.Kernel(r.repackIQ2Fn).Grid(grid, 1, 1).Block(256, 1, 1).On(r.stream()).Launch(
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
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		dm2 := dst.Slice(0, int(dmBytes))
		sc4 := dst.Slice(int(dmBytes), int(dmBytes+scBytes))
		qs2 := dst.Slice(int(dmBytes+scBytes), int(dmBytes+scBytes+qsBytes))
		grid := uint32((cblk*16 + 255) / 256)
		if err := cudalaunch.Kernel(r.repackQ2KFn).Grid(grid, 1, 1).Block(256, 1, 1).On(r.stream()).Launch(
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
	if l.expertGate.device == 0 || l.expertUp.device == 0 || l.expertDown.device == 0 {
		return fmt.Errorf("repack: raw expert weights not addressable")
	}
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

// cudaMoEArtifact holds one layer's expert weights in the repacked layout the
// D2R GEMMs read.
//
// The repack is a pure function of the model bytes and the layout is
// byte-neutral, so a layer that holds one never reads its raw expert tensors
// again. What it buys is the repack itself, which the layer loop otherwise pays
// once per layer per prefill group.
type cudaMoEArtifact struct {
	gate, up, down cudamem.Slice[byte]
}

func (a *cudaMoEArtifact) built() bool { return a.gate.Len() > 0 }

// digest samples the artifact. The build is a pure function of the model bytes,
// so this value must not change between runs; that it does is the finding the
// sampling exists to make visible. Reading a stride rather than the whole 1.7
// GiB keeps it usable at every layer count.
func (a *cudaMoEArtifact) digest(buf []byte) (uint64, error) {
	h := uint64(14695981039346656037)
	for _, s := range []cudamem.Slice[byte]{a.gate, a.up, a.down} {
		const samples = 64
		step := s.Len() / samples
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

func (a *cudaMoEArtifact) free() error {
	return a.freeWith(defaultCUDAChildCleanupOperations().bytes)
}

func (a *cudaMoEArtifact) Close() error { return a.free() }

func (a *cudaMoEArtifact) freeWith(release func(*cudamem.Slice[byte]) error) error {
	return freeCUDAByteSlices([]*cudamem.Slice[byte]{&a.gate, &a.up, &a.down}, release)
}

// cudaMoEArtifactBytes is what one layer's artifact occupies.
const cudaMoEArtifactBytes = 2*cudaIQ2XXSRepackedBytes + cudaQ2KRepackedBytes

// cudaMoEStage is the scratch one tensor is repacked through. The source is
// read from the file with pread rather than addressed through the model
// mapping: a build that reads the mapping faults tens of gigabytes of the model
// into the page cache, where it competes with the artifacts being allocated on
// the same integrated memory, and the result is reclaim -- which costs
// throughput and, measurably, correctness.
type cudaMoEStage struct {
	dev     cudamem.Slice[byte]
	release func(*cudamem.Slice[byte]) error
	// host is aligned to cudaDirectAlign and longer than the largest tensor,
	// because an O_DIRECT read needs an aligned buffer and a range rounded
	// outward to alignment at both ends. See stageDirect.
	host []byte
}

// cudaDirectAlign is the largest block size direct staging will honour. The
// descriptor's real requirement comes from fstat, and openDirectModel declines
// the flag rather than serving a filesystem that wants more than this.
const cudaDirectAlign = 1 << 16

// cudaMoEStageBytes is the largest expert tensor. The layouts are byte-neutral,
// so this is both the raw and the repacked size.
const cudaMoEStageBytes = cudaQ2KRepackedBytes

func newCUDAMoEStage(dev *cuda.Device) (*cudaMoEStage, error) {
	return newCUDAMoEStageWithOperations(dev, cudamem.Alloc[byte], defaultCUDAChildCleanupOperations().bytes)
}

func newCUDAMoEStageWithOperations(dev *cuda.Device, allocate func(*cuda.Device, int) (cudamem.Slice[byte], error), release func(*cudamem.Slice[byte]) error) (*cudaMoEStage, error) {
	d, err := allocate(dev, cudaMoEStageBytes)
	s := &cudaMoEStage{dev: d, release: release}
	if err != nil {
		return nil, finishBackendCleanup(fmt.Errorf("allocate moe stage: %w", err), s)
	}
	// Over-allocate and slice forward to an aligned address. Go's heap does
	// not move allocations, so the alignment computed once here holds for the
	// life of the stage.
	raw := make([]byte, cudaMoEStageBytes+3*cudaDirectAlign)
	pad := int(-uintptr(unsafe.Pointer(&raw[0])) & (cudaDirectAlign - 1))
	s.host = raw[pad : pad+cudaMoEStageBytes+2*cudaDirectAlign]
	return s, nil
}

func (s *cudaMoEStage) Close() error {
	if s == nil || s.dev.Len() == 0 {
		return nil
	}
	release := s.release
	if release == nil {
		release = defaultCUDAChildCleanupOperations().bytes
	}
	if err := release(&s.dev); err != nil {
		return err
	}
	s.dev, s.host = cudamem.Slice[byte]{}, nil
	return nil
}

// load reads one tensor off disk and leaves it in device memory.
func (r *cudaFoundation) load(s *cudaMoEStage, t *cudaModelTensor) error {
	if err := r.checkOpen(); err != nil {
		return err
	}
	if t.bytes > cudaMoEStageBytes {
		return fmt.Errorf("tensor %s is %d bytes, stage holds %d", t.name, t.bytes, uint64(cudaMoEStageBytes))
	}
	buf, err := r.stage(s, t)
	if err != nil {
		return err
	}
	if err := s.dev.Slice(0, int(t.bytes)).CopyFromHost(buf); err != nil {
		return fmt.Errorf("stage %s: %w", t.name, err)
	}
	return nil
}

// stage fills the stage's host buffer with t's bytes and returns them. It
// reads through the O_DIRECT descriptor when one is open, and falls back to
// the buffered one for good on the errors that mean this filesystem will not
// serve direct reads. Both paths count what they read; see stagedBytes.
func (r *cudaFoundation) stage(s *cudaMoEStage, t *cudaModelTensor) ([]byte, error) {
	if err := r.checkOpen(); err != nil {
		return nil, err
	}
	if r.direct != nil {
		buf, err := r.stageDirect(s, t)
		switch {
		case err == nil:
			r.directBytes += t.bytes
			return buf, nil
		case errors.Is(err, syscall.EINVAL), errors.Is(err, syscall.EFAULT), errors.Is(err, syscall.ENOTSUP):
			fmt.Fprintf(os.Stderr, "ds4: direct staging disabled: %v\n", err)
			r.direct.Close()
			r.direct = nil
		default:
			return nil, err
		}
	}
	buf := s.host[:t.bytes]
	for off := uint64(0); off < t.bytes; {
		n, err := syscall.Pread(int(r.file.Fd()), buf[off:], int64(t.fileOffset+off))
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", t.name, err)
		}
		if n == 0 {
			return nil, fmt.Errorf("read %s: short read at %d", t.name, off)
		}
		off += uint64(n)
	}
	r.bufferedBytes += t.bytes
	return buf, nil
}

// stageDirect reads t through the O_DIRECT descriptor, which requires the
// buffer address, the file offset and the length all aligned. The range is
// rounded outward to the descriptor's alignment and the payload sliced back
// out of the span, so the caller never sees the padding.
func (r *cudaFoundation) stageDirect(s *cudaMoEStage, t *cudaModelTensor) ([]byte, error) {
	if err := r.checkOpen(); err != nil {
		return nil, err
	}
	a := uint64(r.directAlign)
	lo := t.fileOffset &^ (a - 1)
	hi := (t.fileOffset + t.bytes + a - 1) &^ (a - 1)
	span, end := s.host[:hi-lo], t.fileOffset+t.bytes
	for off := uint64(0); lo+off < end; {
		n, err := syscall.Pread(int(r.direct.Fd()), span[off:], int64(lo+off))
		if err != nil {
			return nil, fmt.Errorf("direct read %s: %w", t.name, err)
		}
		if n == 0 {
			return nil, fmt.Errorf("direct read %s: short read at %d", t.name, off)
		}
		off += uint64(n)
	}
	return span[t.fileOffset-lo:][:t.bytes], nil
}

// buildMoEArtifact repacks one layer's expert weights into freshly allocated
// device memory, reading the source through stage.
func (r *cudaFoundation) buildMoEArtifact(l *cudaLayer, a *cudaMoEArtifact, s *cudaMoEStage) (err error) {
	defer func() {
		if err != nil {
			err = finishBackendCleanup(err, a)
		}
	}()
	const (
		iq2Nblk  = 256 * 2048 * 16
		q2kCblk  = 256 * 4096 * 8
		q2kNbRow = 8
		q2kNrows = 4096
	)
	for _, part := range []struct {
		dst *cudamem.Slice[byte]
		src *cudaModelTensor
		n   int
	}{
		{&a.gate, &l.expertGate, cudaIQ2XXSRepackedBytes},
		{&a.up, &l.expertUp, cudaIQ2XXSRepackedBytes},
		{&a.down, &l.expertDown, cudaQ2KRepackedBytes},
	} {
		dst, err := cudamem.Alloc[byte](r.device, part.n)
		*part.dst = dst
		if err != nil {
			return fmt.Errorf("allocate %s artifact: %w", part.src.name, err)
		}
		if err := r.load(s, part.src); err != nil {
			return err
		}
		if part.src == &l.expertDown {
			err = r.repackQ2K(dst, s.dev.Ptr(), 0, q2kCblk, q2kNbRow, q2kNrows)
		} else {
			err = r.repackIQ2(dst, s.dev.Ptr(), iq2Nblk)
		}
		if err != nil {
			return fmt.Errorf("repack %s: %w", part.src.name, err)
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
