package cudareduce

import (
	"errors"
	"fmt"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// MaxGroups is the largest group count [Reducer.SegmentColumnSum] accepts:
// each thread keeps one double per group in 48 KiB of shared memory.
const MaxGroups = 48 << 10 / (Threads * 8)

// targetBlocks is the number of stage 1 blocks Split aims for.
const targetBlocks = 512

// minRowsPerSplit keeps each stage 1 thread's serial loop from being too
// short to be worth a partial.
const minRowsPerSplit = 32

// Split returns how stage 1 of a reduction over rows rows and n columns
// divides the rows: into splits consecutive ranges of rowsPerSplit rows,
// the last possibly shorter. It depends only on the shape, never on the
// device, so a reduction sums in the same order everywhere.
//
// A stage 1 kernel launches on the grid (ceil(n/[Threads]), splits).
func Split(rows, n int) (splits, rowsPerSplit int) {
	if rows <= 0 {
		return 1, 1
	}
	colBlocks := max((n+Threads-1)/Threads, 1)
	splits = max(targetBlocks/colBlocks, 1)
	splits = min(splits, max(rows/minRowsPerSplit, 1))
	rowsPerSplit = (rows + splits - 1) / splits
	splits = (rows + rowsPerSplit - 1) / rowsPerSplit
	return splits, rowsPerSplit
}

// ScratchLen returns the number of float64 elements of scratch a reduction
// over rows rows and n columns into k sums per column needs.
func ScratchLen(rows, n, k int) int {
	splits, _ := Split(rows, n)
	return k * splits * n
}

// A Reducer launches the column reductions of [Header]. Its methods enqueue
// work on the given stream and do not synchronize. They allocate nothing:
// the caller provides the float64 scratch, [ScratchLen] elements, so a
// reduction can be captured in a CUDA graph. A scratch buffer must not be
// shared by reductions that can run concurrently.
//
// When accumulate is true, a method adds the sums to the existing output,
// as gradient accumulation across micro-batches needs; otherwise it
// overwrites it. Either way each output element is rounded to float32 once.
type Reducer struct {
	mod                               *cuda.Module
	colsum, segment, layernorm, final *cuda.Kernel
}

// New compiles the reduction kernels for dev.
func New(dev *cuda.Device) (_ *Reducer, err error) {
	cm, err := nvrtcx.Compile(kernelSource,
		nvrtcx.WithName("cudareduce.cu"),
		nvrtcx.WithDeviceArch(dev),
		nvrtcx.WithHeader("cudareduce.cuh", Header))
	if err != nil {
		return nil, fmt.Errorf("cudareduce: compile kernels: %w", err)
	}
	r := &Reducer{}
	if r.mod, err = cm.LoadModule(dev); err != nil {
		return nil, fmt.Errorf("cudareduce: load kernels: %w", err)
	}
	for _, k := range []struct {
		dst  **cuda.Kernel
		name string
	}{
		{&r.colsum, "cudareduce_colsum"},
		{&r.segment, "cudareduce_segment_colsum"},
		{&r.layernorm, "cudareduce_layernorm_partials"},
		{&r.final, "cudareduce_finish_kernel"},
	} {
		if *k.dst, err = r.mod.Function(k.name); err != nil {
			r.Close()
			return nil, fmt.Errorf("cudareduce: %w", err)
		}
	}
	return r, nil
}

// Close unloads the kernels. Work the Reducer enqueued must be complete.
func (r *Reducer) Close() error {
	if r.mod == nil {
		return nil
	}
	err := r.mod.Close()
	r.mod = nil
	return err
}

// ColumnSum sets out[c] to the sum over rows of x[r*n+c], for the row-major
// matrix x of rows rows and n columns: the bias gradient of a linear layer.
func (r *Reducer) ColumnSum(stream *cuda.Stream, out, x cudamem.Slice[float32], rows, n int, scratch cudamem.Slice[float64], accumulate bool) error {
	if err := check(rows, n, 1, scratch, need{"out", out.Len(), n}, need{"x", x.Len(), rows * n}); err != nil {
		return err
	}
	splits, rps := Split(rows, n)
	xp, sp := x.Ptr(), scratch.Ptr()
	rows64, n64, rps64 := int64(rows), int64(n), int64(rps)
	if err := r.launch(r.colsum, stream, n, splits, 0,
		unsafe.Pointer(&xp), unsafe.Pointer(&rows64), unsafe.Pointer(&n64), unsafe.Pointer(&rps64), unsafe.Pointer(&sp)); err != nil {
		return err
	}
	return r.finish(stream, out.Ptr(), sp, rows, n, 1, accumulate)
}

// SegmentColumnSum sets out[g*n+c] to the sum of x[r*n+c] over the rows r
// with seg[r] == g, for groups groups: the gradient of an embedding table
// indexed by seg. Rows whose seg is outside [0, groups), such as -1 for
// padding, contribute nothing. groups must be at most [MaxGroups].
func (r *Reducer) SegmentColumnSum(stream *cuda.Stream, out, x cudamem.Slice[float32], seg cudamem.Slice[int32], rows, n, groups int, scratch cudamem.Slice[float64], accumulate bool) error {
	if groups < 1 || groups > MaxGroups {
		return fmt.Errorf("cudareduce: %d groups, want 1 to %d", groups, MaxGroups)
	}
	if err := check(rows, n, groups, scratch, need{"out", out.Len(), groups * n}, need{"x", x.Len(), rows * n}, need{"seg", seg.Len(), rows}); err != nil {
		return err
	}
	splits, rps := Split(rows, n)
	xp, segp, sp := x.Ptr(), seg.Ptr(), scratch.Ptr()
	g32 := int32(groups)
	rows64, n64, rps64 := int64(rows), int64(n), int64(rps)
	if err := r.launch(r.segment, stream, n, splits, uint32(groups*Threads*8),
		unsafe.Pointer(&xp), unsafe.Pointer(&segp), unsafe.Pointer(&g32),
		unsafe.Pointer(&rows64), unsafe.Pointer(&n64), unsafe.Pointer(&rps64), unsafe.Pointer(&sp)); err != nil {
		return err
	}
	return r.finish(stream, out.Ptr(), sp, rows, n, groups, accumulate)
}

// LayerNormGrad computes the LayerNorm weight and bias gradients
// dw[c] = Σ_r dy[r*n+c]*xhat[r*n+c] and db[c] = Σ_r dy[r*n+c], where
// xhat = (x - mean[r]) * rstd[r]. If mean and rstd are both empty slices,
// x is taken to be xhat already.
func (r *Reducer) LayerNormGrad(stream *cuda.Stream, dw, db, dy, x, mean, rstd cudamem.Slice[float32], rows, n int, scratch cudamem.Slice[float64], accumulate bool) error {
	needs := []need{{"dw", dw.Len(), n}, {"db", db.Len(), n}, {"dy", dy.Len(), rows * n}, {"x", x.Len(), rows * n}}
	if mean.Len() != 0 || rstd.Len() != 0 {
		needs = append(needs, need{"mean", mean.Len(), rows}, need{"rstd", rstd.Len(), rows})
	}
	if err := check(rows, n, 2, scratch, needs...); err != nil {
		return err
	}
	splits, rps := Split(rows, n)
	dyp, xp, mp, rp, sp := dy.Ptr(), x.Ptr(), mean.Ptr(), rstd.Ptr(), scratch.Ptr()
	if mean.Len() == 0 {
		mp, rp = 0, 0
	}
	rows64, n64, rps64 := int64(rows), int64(n), int64(rps)
	if err := r.launch(r.layernorm, stream, n, splits, 0,
		unsafe.Pointer(&dyp), unsafe.Pointer(&xp), unsafe.Pointer(&mp), unsafe.Pointer(&rp),
		unsafe.Pointer(&rows64), unsafe.Pointer(&n64), unsafe.Pointer(&rps64), unsafe.Pointer(&sp)); err != nil {
		return err
	}
	if err := r.finish(stream, dw.Ptr(), sp, rows, n, 1, accumulate); err != nil {
		return err
	}
	return r.finish(stream, db.Ptr(), sp+driver.DevicePtr(splits*n*8), rows, n, 1, accumulate)
}

// Finish runs stage 2 for a stage 1 kernel of the caller's own that wrote
// k sums per column into scratch, as cudareduce_column_partials or
// cudareduce_segment_partials lay them out for [Split](rows, n). It writes
// sum j of column c to out[j*n+c].
func (r *Reducer) Finish(stream *cuda.Stream, out cudamem.Slice[float32], scratch cudamem.Slice[float64], rows, n, k int, accumulate bool) error {
	if k < 1 {
		return fmt.Errorf("cudareduce: %d sums per column", k)
	}
	if err := check(rows, n, k, scratch, need{"out", out.Len(), k * n}); err != nil {
		return err
	}
	return r.finish(stream, out.Ptr(), scratch.Ptr(), rows, n, k, accumulate)
}

func (r *Reducer) finish(stream *cuda.Stream, out, partials driver.DevicePtr, rows, n, k int, accumulate bool) error {
	splits, _ := Split(rows, n)
	s64, n64 := int64(splits), int64(n)
	var acc int32
	if accumulate {
		acc = 1
	}
	return r.launch(r.final, stream, n, k, 0,
		unsafe.Pointer(&partials), unsafe.Pointer(&s64), unsafe.Pointer(&n64), unsafe.Pointer(&out), unsafe.Pointer(&acc))
}

func (r *Reducer) launch(k *cuda.Kernel, stream *cuda.Stream, n, gridY int, shmem uint32, args ...unsafe.Pointer) error {
	if n == 0 {
		return nil
	}
	grid := [3]uint32{uint32((n + Threads - 1) / Threads), uint32(gridY), 1}
	err := stream.WithHandle(func(h driver.Stream) error {
		return k.Launch(grid, [3]uint32{Threads, 1, 1}, shmem, h, args)
	})
	if err != nil {
		return fmt.Errorf("cudareduce: %w", err)
	}
	return nil
}

// need is a slice that must hold at least want elements.
type need struct {
	name       string
	have, want int
}

var errShape = errors.New("cudareduce: rows and n must not be negative")

func check(rows, n, k int, scratch cudamem.Slice[float64], needs ...need) error {
	if rows < 0 || n < 0 {
		return errShape
	}
	needs = append(needs, need{"scratch", scratch.Len(), ScratchLen(rows, n, k)})
	for _, nd := range needs {
		if nd.have < nd.want {
			return fmt.Errorf("cudareduce: %s has %d elements, need %d", nd.name, nd.have, nd.want)
		}
	}
	return nil
}
