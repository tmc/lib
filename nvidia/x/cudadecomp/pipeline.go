package cudadecomp

import (
	"context"
	"fmt"
	"io"
	"sync"

	"github.com/tmc/lib/nvidia/driver"
)

// A Copy is a device-to-device copy of a stored chunk, which the hardware
// decompression engine does not handle.
type Copy struct {
	Src, Dst uintptr
	Size     int
}

// Algorithm returns the driver decompression algorithm for c.
func (c Codec) Algorithm() (driver.MemDecompressAlgorithm, error) {
	switch c {
	case CodecLZ4:
		return driver.MemDecompressAlgorithmLZ4, nil
	case CodecDeflate:
		return driver.MemDecompressAlgorithmDeflate, nil
	}
	return 0, fmt.Errorf("cudadecomp: codec %v has no hardware algorithm", c)
}

// CheckDevice reports whether a device whose decompression attributes are
// algorithms (DevAttrMemDecompressAlgorithmMask) and maxLength
// (DevAttrMemDecompressMaximumLength) can decompress every compressed
// chunk of t.
func (t Tensor) CheckDevice(algorithms driver.MemDecompressAlgorithm, maxLength int) error {
	for i, c := range t.Chunks {
		if c.Codec == CodecStored {
			continue
		}
		a, err := c.Codec.Algorithm()
		if err != nil {
			return err
		}
		if algorithms&a == 0 {
			return fmt.Errorf("cudadecomp: chunk %d: device lacks %v (mask %#x)", i, c.Codec, int32(algorithms))
		}
		if int(c.DstSize) > maxLength || int(c.SrcSize) > maxLength {
			return fmt.Errorf("cudadecomp: chunk %d: %d bytes exceed device maximum %d", i, max(c.DstSize, c.SrcSize), maxLength)
		}
	}
	return nil
}

// BatchParams returns the cuMemBatchDecompressAsync parameters for t's
// compressed chunks, with the payload at device address src, the
// uncompressed tensor at dst, and one 8-byte actual-size word per
// parameter starting at actual. Stored chunks are returned as copies. The
// addresses must come from allocations usable by the decompression engine
// (see [driver.Stream.MemBatchDecompressAsync]); BatchParams only computes
// them.
func (t Tensor) BatchParams(src, dst, actual uintptr) ([]driver.MemDecompressParams, []Copy, error) {
	var params []driver.MemDecompressParams
	var copies []Copy
	for i, c := range t.Chunks {
		if c.SrcOffset+uint64(c.SrcSize) > uint64(len(t.Payload)) || c.DstOffset+uint64(c.DstSize) > uint64(t.Size) {
			return nil, nil, fmt.Errorf("%w: chunk %d out of range", ErrChunkCorrupted, i)
		}
		if c.Codec == CodecStored {
			copies = append(copies, Copy{Src: src + uintptr(c.SrcOffset), Dst: dst + uintptr(c.DstOffset), Size: int(c.DstSize)})
			continue
		}
		a, err := c.Codec.Algorithm()
		if err != nil {
			return nil, nil, err
		}
		params = append(params, driver.MemDecompressParams{
			SrcNumBytes: uint(c.SrcSize),
			DstNumBytes: uint(c.DstSize),
			DstActBytes: actual + uintptr(8*len(params)),
			Src:         src + uintptr(c.SrcOffset),
			Dst:         dst + uintptr(c.DstOffset),
			Algo:        a,
		})
	}
	return params, copies, nil
}

// A ReadOp is one read of a chunk's compressed bytes from a file into a
// staging buffer, as a cuFile batch read would issue it.
type ReadOp struct {
	FileOffset int64
	Size       int
	BufOffset  int64 // offset in the staging buffer
}

// ReadAlign is the staging slot alignment [Tensor.ReadOps] uses, the
// page size GPUDirect Storage prefers for direct reads.
const ReadAlign = 4096

// ReadOps returns the reads that stage t's chunks from a file holding the
// payload at fileOffset into a buffer of slots slots, chunk i going to slot
// i%slots. It returns the slot size, the largest chunk rounded up to
// [ReadAlign].
func (t Tensor) ReadOps(fileOffset int64, slots int) ([]ReadOp, int64) {
	var size int64
	for _, c := range t.Chunks {
		size = max(size, int64(c.SrcSize))
	}
	size = (size + ReadAlign - 1) / ReadAlign * ReadAlign
	ops := make([]ReadOp, len(t.Chunks))
	for i, c := range t.Chunks {
		ops[i] = ReadOp{FileOffset: fileOffset + int64(c.SrcOffset), Size: int(c.SrcSize), BufOffset: int64(i%slots) * size}
	}
	return ops, size
}

// A Step is one step of the staging pipeline: the chunks read,
// decompressed and consumed during it, concurrently, or -1 for none.
type Step struct {
	Read, Decompress, Consume int
}

// Schedule returns the three-stage schedule for n chunks: during step s,
// chunk s is read, chunk s-1 decompressed and chunk s-2 consumed.
func Schedule(n int) []Step {
	if n <= 0 {
		return nil
	}
	steps := make([]Step, n+2)
	for s := range steps {
		steps[s] = Step{Read: inRange(s, n), Decompress: inRange(s-1, n), Consume: inRange(s-2, n)}
	}
	return steps
}

func inRange(i, n int) int {
	if i < 0 || i >= n {
		return -1
	}
	return i
}

// VerifySchedule checks that steps reads, decompresses and consumes each of
// n chunks exactly once, each stage in a later step than the one before, and
// that with chunk i in slot i%slots of both the compressed and the
// decompressed buffer, no chunk's slot is overwritten between the step that
// fills it and the step that drains it, inclusive.
func VerifySchedule(steps []Step, n, slots int) error {
	if slots <= 0 {
		return fmt.Errorf("cudadecomp: %d slots", slots)
	}
	read := make([]int, n)
	dec := make([]int, n)
	use := make([]int, n)
	for i := range n {
		read[i], dec[i], use[i] = -1, -1, -1
	}
	set := func(at []int, chunk, step int, stage string) error {
		if chunk < 0 {
			return nil
		}
		if chunk >= n {
			return fmt.Errorf("cudadecomp: step %d: %s chunk %d of %d", step, stage, chunk, n)
		}
		if at[chunk] >= 0 {
			return fmt.Errorf("cudadecomp: chunk %d: %s twice", chunk, stage)
		}
		at[chunk] = step
		return nil
	}
	for s, st := range steps {
		if err := set(read, st.Read, s, "read"); err != nil {
			return err
		}
		if err := set(dec, st.Decompress, s, "decompressed"); err != nil {
			return err
		}
		if err := set(use, st.Consume, s, "consumed"); err != nil {
			return err
		}
	}
	for i := range n {
		if read[i] < 0 || dec[i] <= read[i] || use[i] <= dec[i] {
			return fmt.Errorf("cudadecomp: chunk %d: read at %d, decompressed at %d, consumed at %d", i, read[i], dec[i], use[i])
		}
		for j := range n {
			if j == i || j%slots != i%slots {
				continue
			}
			if read[i] <= read[j] && read[j] <= dec[i] {
				return fmt.Errorf("cudadecomp: chunk %d overwrites compressed slot %d at step %d before chunk %d is decompressed", j, i%slots, read[j], i)
			}
			if dec[i] <= dec[j] && dec[j] <= use[i] {
				return fmt.Errorf("cudadecomp: chunk %d overwrites decompressed slot %d at step %d before chunk %d is consumed", j, i%slots, dec[j], i)
			}
		}
	}
	return nil
}

// Run stages t through [Schedule] on the host with two slots per buffer:
// it reads each chunk's compressed bytes from r at base plus the chunk's
// payload offset, decompresses it, and calls consume with the chunk and its
// uncompressed bytes, which are valid only during the call. The three
// stages of a step run concurrently; consume is called in chunk order. Run
// stops at the first error.
func Run(ctx context.Context, r io.ReaderAt, base int64, t Tensor, consume func(Chunk, []byte) error) error {
	const slots = 2
	var srcSize, dstSize int
	for _, c := range t.Chunks {
		srcSize = max(srcSize, int(c.SrcSize))
		dstSize = max(dstSize, int(c.DstSize))
	}
	var src, dst [slots][]byte
	for i := range slots {
		src[i] = make([]byte, srcSize)
		dst[i] = make([]byte, dstSize)
	}
	for _, st := range Schedule(len(t.Chunks)) {
		if err := ctx.Err(); err != nil {
			return err
		}
		var wg sync.WaitGroup
		var errs [3]error
		if i := st.Read; i >= 0 {
			wg.Go(func() {
				c := t.Chunks[i]
				if _, err := r.ReadAt(src[i%slots][:c.SrcSize], base+int64(c.SrcOffset)); err != nil {
					errs[0] = fmt.Errorf("cudadecomp: read chunk %d: %w", i, err)
				}
			})
		}
		if i := st.Decompress; i >= 0 {
			wg.Go(func() {
				c := t.Chunks[i]
				if err := decodeChunk(dst[i%slots][:c.DstSize], src[i%slots][:c.SrcSize], c); err != nil {
					errs[1] = fmt.Errorf("chunk %d: %w", i, err)
				}
			})
		}
		if i := st.Consume; i >= 0 {
			wg.Go(func() {
				c := t.Chunks[i]
				errs[2] = consume(c, dst[i%slots][:c.DstSize])
			})
		}
		wg.Wait()
		for _, err := range errs {
			if err != nil {
				return err
			}
		}
	}
	return nil
}
