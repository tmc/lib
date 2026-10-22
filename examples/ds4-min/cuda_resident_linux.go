//go:build linux

package main

import (
	"cmp"
	"fmt"
	"os"
	"slices"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// cudaResidentModel reports whether newCUDAFoundation holds the model's
// weights in device memory on a device that uses host page tables. It does
// unless DS4_CUDA_MAPPED_MODEL=1, which restores kernels reading the file
// mapping and exists only as the control arm for loadResidentSet.
func cudaResidentModel() bool {
	return os.Getenv("DS4_CUDA_MAPPED_MODEL") != "1"
}

// cudaResidentChunk is the pinned staging buffer loadResidentSet reads
// through.
const cudaResidentChunk = 64 << 20

// Span merging follows the C oracle's accelerator_prepare_model_tensor_spans
// (ds4.c:3002): tensors closer than cudaResidentGap merge into one span, and
// a span stops growing at cudaResidentMaxSpan, the default of
// DS4_CUDA_WEIGHT_PRELOAD_SPAN_MB (ds4.c:2970).
const (
	cudaResidentGap     = 64 << 10
	cudaResidentMaxSpan = 1 << 30
)

// cudaExpertReplaced reports whether an aligned MoE artifact replaces the
// named tensor, so that no kernel reads its raw bytes. The C oracle asks the
// same question of each tensor with ds4_gpu_model_range_replaced
// (ds4_cuda.cu:4555).
func cudaExpertReplaced(name string) bool {
	return strings.HasSuffix(name, "_exps.weight")
}

// cudaFileExtent is a byte range of the model file.
type cudaFileExtent struct{ off, n uint64 }

// cudaResidentSpan is a run of the model file held in device memory: file
// bytes [off, off+n) are at byte dev of the resident allocation.
type cudaResidentSpan struct{ off, n, dev uint64 }

// cudaResidentSet is where every model tensor a kernel may read lives.
//
// It is the C oracle's policy on an integrated device with aligned artifacts
// (ds4_cuda.cu:3758 and ds4.c:3002). The expert tensors are replaced by
// artifacts and get no device address at all. Every other tensor is copied
// once, at startup, into device memory, merged into spans the way C merges
// them, so the tensors a kernel reads are served from memory the host cannot
// reclaim. The file mapping stays open for the CPU, unregistered, as C leaves
// it, and nothing hands its addresses to a kernel.
type cudaResidentSet struct {
	spans    []cudaResidentSpan // sorted by off
	replaced []cudaFileExtent   // sorted by off
	bytes    uint64             // size of the device allocation
}

// planResidentSet lays out the device copy of every tensor that replaced
// does not claim, in a file of size bytes. Each span is widened to whole
// pages, short of the end of the file, so a tensor keeps the alignment its
// mapping address had; spans may then share a page.
func planResidentSet(tensors map[string]cudaFileExtent, replaced func(string) bool, page, size uint64) cudaResidentSet {
	var s cudaResidentSet
	var keep []cudaFileExtent
	for name, t := range tensors {
		if t.n == 0 {
			continue
		}
		if replaced(name) {
			s.replaced = append(s.replaced, t)
			continue
		}
		keep = append(keep, t)
	}
	byOffset := func(a, b cudaFileExtent) int {
		return cmp.Or(cmp.Compare(a.off, b.off), cmp.Compare(a.n, b.n))
	}
	slices.SortFunc(keep, byOffset)
	slices.SortFunc(s.replaced, byOffset)
	for i := 0; i < len(keep); {
		lo, hi := keep[i].off, keep[i].off+keep[i].n
		for i++; i < len(keep) && keep[i].off <= hi+cudaResidentGap && keep[i].off+keep[i].n-lo <= cudaResidentMaxSpan; i++ {
			hi = max(hi, keep[i].off+keep[i].n)
		}
		lo &^= page - 1
		hi = min((hi+page-1)&^(page-1), size)
		s.spans = append(s.spans, cudaResidentSpan{off: lo, n: hi - lo, dev: s.bytes})
		s.bytes += hi - lo
	}
	return s
}

// lookup returns the device offset of file bytes [off, off+n), or false if
// no span holds all of them.
func (s *cudaResidentSet) lookup(off, n uint64) (uint64, bool) {
	i := sort.Search(len(s.spans), func(i int) bool { return s.spans[i].off > off })
	if i == 0 {
		return 0, false
	}
	sp := s.spans[i-1]
	if off+n > sp.off+sp.n {
		return 0, false
	}
	return sp.dev + off - sp.off, true
}

// isReplaced reports whether [off, off+n) is exactly a replaced tensor.
func (s *cudaResidentSet) isReplaced(off, n uint64) bool {
	i := sort.Search(len(s.replaced), func(i int) bool { return s.replaced[i].off >= off })
	return i < len(s.replaced) && s.replaced[i] == cudaFileExtent{off, n}
}

// loadResidentSet plans the resident set and copies it into one device
// allocation. It reads with pread through a pinned stage and then drops the
// span's pages from the page cache, as cuda_model_range_ptr_from_fd does
// (ds4_cuda.cu:2354), so the load does not leave a second copy of the weights
// in the pool the device allocates from.
//
// On GB10 a kernel that reads a PROT_READ MAP_SHARED file mapping through the
// host page tables can see wrong bytes while the host reclaims page cache: a
// plain copy kernel over the whole model, run beside 40 GiB of committed
// device memory, returned one wrong page in 3 of 6 passes over 21171902
// pages, each wrong in whole 128-byte lines, and a re-read of the same page
// moments later returned the file's bytes. With the mapping, the 2048-token
// prefill gave a wrong digest or non-finite logits in 4 of 4 runs beside a
// 32 GiB ballast. Device memory is not reclaimable, so reads from it cannot
// race reclaim.
func (r *cudaFoundation) loadResidentSet() (err error) {
	if err := r.checkOpen(); err != nil {
		return err
	}
	defer beginOp("model.resident").end()
	start := time.Now()
	tensors := make(map[string]cudaFileExtent, len(r.gguf.Tensors))
	artifactTypes := make(map[string]bool, len(r.gguf.Tensors))
	for _, t := range r.gguf.Tensors {
		n, err := t.Bytes()
		if err != nil {
			return fmt.Errorf("cuda: size tensor %q: %w", t.Name, err)
		}
		tensors[t.Name] = cudaFileExtent{r.gguf.DataOffset + t.Offset, n}
		artifactTypes[t.Name] = t.TypeID == typeIQ2XXS || t.TypeID == typeQ2K
	}
	set := planResidentSet(tensors, func(name string) bool {
		return cudaExpertReplaced(name) && artifactTypes[name]
	}, uint64(syscall.Getpagesize()), uint64(len(r.model)))
	var replaced uint64
	for _, t := range set.replaced {
		replaced += t.n
	}
	ops := r.residentOperations()
	mem, err := ops.allocate(int(set.bytes))
	if mem != nil {
		r.resident, _ = mem.(cudamem.Slice[byte])
		r.residentClose = mem.Free
	}
	r.residentSet = set
	if err != nil {
		return fmt.Errorf("cuda: allocate resident weights (%d bytes): %w", set.bytes, err)
	}
	if mem == nil || mem.Len() != int(set.bytes) || mem.Len() <= 0 {
		return fmt.Errorf("cuda: invalid resident allocation")
	}
	err = ops.run(func() error {
		buf, free, err := ops.stage(cudaResidentChunk)
		r.residentStageClose = free
		if err != nil {
			return fmt.Errorf("allocate staging buffer: %w", err)
		}
		if len(buf) == 0 || free == nil {
			return fmt.Errorf("resident staging unavailable")
		}
		fd := int(r.file.Fd())
		for _, sp := range set.spans {
			for done := uint64(0); done < sp.n; {
				n := min(uint64(len(buf)), sp.n-done)
				if _, err := r.file.ReadAt(buf[:n], int64(sp.off+done)); err != nil {
					return fmt.Errorf("read model at %d: %w", sp.off+done, err)
				}
				dst := sp.dev + done
				if err := ops.copy(mem, int(dst), buf[:n]); err != nil {
					return fmt.Errorf("copy model at %d: %w", sp.off+done, err)
				}
				done += n
			}
			if err := ops.drop(fd, sp.off, sp.n); err != nil {
				return fmt.Errorf("drop model pages at %d: %w", sp.off, err)
			}
		}
		if err := ops.drain(); err != nil {
			return fmt.Errorf("drain resident copies: %w", err)
		}
		if err := r.residentStageClose(); err != nil {
			return fmt.Errorf("free resident staging: %w", err)
		}
		r.residentStageClose = nil
		return nil
	})
	if err != nil {
		return fmt.Errorf("cuda: load resident weights: %w", err)
	}
	fmt.Fprintf(os.Stderr, "ds4: model resident: spans=%d bytes=%.2fGiB replaced=%d/%.2fGiB secs=%.2f availGiB=%.1f\n",
		len(set.spans), float64(set.bytes)/(1<<30), len(set.replaced), float64(replaced)/(1<<30),
		time.Since(start).Seconds(), memAvailableGiB())
	return nil
}

// residentRange is modelRange for a foundation holding a resident set. It
// never returns an address in the file mapping: a range in a span gets its
// device copy, a replaced expert tensor gets no address, and anything else is
// an error, because serving it would put a kernel back on the mapping.
func (r *cudaFoundation) residentRange(fileOffset, length uint64) (driver.DevicePtr, error) {
	if off, ok := r.residentSet.lookup(fileOffset, length); ok {
		return r.resident.Ptr() + driver.DevicePtr(off), nil
	}
	if r.residentSet.isReplaced(fileOffset, length) {
		return 0, nil
	}
	return 0, fmt.Errorf("cuda: model range [%d,%d) is neither resident nor replaced by an artifact", fileOffset, fileOffset+length)
}
