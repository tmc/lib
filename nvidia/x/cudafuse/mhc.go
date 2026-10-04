package cudafuse

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// MHCConfig specifies contiguous FP32 four-stream residual mixing. Each batch
// has a 4x4 logit matrix and four post weights; Width is the channel count.
// Repeat includes the initial column normalization and must be in [1,16].
// Epsilon must be finite and positive. Empty batches and widths are refused.
type MHCConfig struct {
	Batch, Width, Repeat int
	Epsilon              float32
}

func (c MHCConfig) validate() error {
	if c.Batch <= 0 || c.Width <= 0 || c.Repeat < 1 || c.Repeat > 16 || !mhcFinite(c.Epsilon) || c.Epsilon <= 0 {
		return errors.New("cudafuse: invalid mhc dimensions, repeat or epsilon")
	}
	// Count is a signed 32-bit kernel parameter. Reserve the two allocation
	// guards and check both the matrix and four-stream output before multiplying.
	if c.Batch > (math.MaxInt32-2)/16 || c.Width > (math.MaxInt32-2)/4/c.Batch {
		return errors.New("cudafuse: mhc extent overflow")
	}
	return nil
}

func mhcFinite(x float32) bool { return !math.IsNaN(float64(x)) && !math.IsInf(float64(x), 0) }

func mhcValidateValues(xs []float32) error {
	for _, v := range xs {
		if !mhcFinite(v) {
			return errors.New("cudafuse: nonfinite mhc input")
		}
	}
	return nil
}

// MHCInputs supplies finite FP32 values in contiguous batch-major storage:
// Logits [Batch,4,4], X [Batch,Width], Post [Batch,4], and Residual
// [Batch,4,Width]. The four views must have disjoint byte ranges on the stream's
// device. ExecuteMHC borrows them until completion; callers must not mutate
// them while borrowed. No in-place variant is supported.
type MHCInputs struct{ Logits, X, Post, Residual cudamem.Slice[float32] }

type mhcRange struct {
	ptr, bytes uint64
	length     int
}

func mhcValidateRanges(c MHCConfig, ranges []mhcRange) error {
	if err := c.validate(); err != nil {
		return err
	}
	lengths := []int{c.Batch * 16, c.Batch * c.Width, c.Batch * 4, c.Batch * 4 * c.Width}
	if len(ranges) != len(lengths) {
		return errors.New("cudafuse: invalid mhc binding count")
	}
	for i, r := range ranges {
		if r.length != lengths[i] || r.bytes != uint64(lengths[i])*4 || r.ptr == 0 || r.ptr%4 != 0 || r.bytes > math.MaxUint64-r.ptr {
			return fmt.Errorf("cudafuse: invalid mhc binding %d", i)
		}
		for _, s := range ranges[:i] {
			if r.ptr < s.ptr+s.bytes && s.ptr < r.ptr+r.bytes {
				return errors.New("cudafuse: aliased mhc bindings")
			}
		}
	}
	return nil
}

// ExecuteMHC launches normalization and post mixing on the same stream:
// out[b,n,c] = X[b,c]*Post[b,n] + sum_m mix[b,m,n]*Residual[b,m,c].
// Normalization is stable row softmax plus epsilon, column normalization,
// then Repeat-1 row/column rounds, each using sum plus epsilon. Finite-round
// results need not have exact unit row or column sums. Accumulation and output
// are FP32, with no fast math or fused multiply-add.
//
// The returned Execution owns guarded matrix scratch, guarded output and both
// loaded kernels; Close is required even on an error result. Its existing
// completion, failed-drain retry and cancellation rules apply. Inputs are
// checked for nonfinite values before compilation; finite arithmetic may still
// overflow in post mixing, which the caller's numerical comparison must reject.
func ExecuteMHC(ctx context.Context, stream *cuda.Stream, c MHCConfig, in MHCInputs) (*Execution, error) {
	if err := c.validate(); err != nil {
		return nil, err
	}
	if stream == nil || stream.Device() == nil {
		return nil, errors.New("cudafuse: nil mhc stream or device")
	}
	if ctx != nil && ctx.Err() != nil {
		return nil, ctx.Err()
	}
	dev := stream.Device()
	inputs := []cudamem.Slice[float32]{in.Logits, in.X, in.Post, in.Residual}
	var releases []func()
	rollback := func() {
		for _, release := range releases {
			release()
		}
	}
	var ranges []mhcRange
	var ptrs []driver.DevicePtr
	for i, b := range inputs {
		release, err := b.Borrow(dev)
		if err != nil {
			rollback()
			return nil, fmt.Errorf("cudafuse: borrow mhc input %d: %w", i, err)
		}
		releases = append(releases, release)
		ranges = append(ranges, mhcRange{uint64(b.Ptr()), uint64(b.SizeBytes()), b.Len()})
		ptrs = append(ptrs, b.Ptr())
	}
	if err := mhcValidateRanges(c, ranges); err != nil {
		rollback()
		return nil, err
	}
	for i, b := range inputs {
		host := make([]float32, b.Len())
		if err := b.CopyToHost(host); err != nil {
			rollback()
			return nil, fmt.Errorf("cudafuse: inspect mhc input %d: %w", i, err)
		}
		if err := mhcValidateValues(host); err != nil {
			rollback()
			return nil, fmt.Errorf("cudafuse: mhc input %d: %w", i, err)
		}
	}
	return mhcExecute(ctx, c, ptrs, nativeExecutionOps(dev, stream), releases)
}

func mhcExecute(ctx context.Context, c MHCConfig, ptrs []driver.DevicePtr, ops executionOps, releases []func()) (*Execution, error) {
	e := &Execution{ops: ops, releases: releases}
	fail := func(err error) (*Execution, error) {
		e.failure = err
		if !e.attempted {
			return e, errors.Join(err, e.Close())
		}
		return e, err
	}
	if err := c.validate(); err != nil {
		return fail(err)
	}
	e.n = c.Batch * 4 * c.Width
	if len(ptrs) != 4 {
		return fail(errors.New("cudafuse: invalid mhc pointer count"))
	}
	for _, n := range []int{c.Batch * 16, e.n} {
		if ctx != nil && ctx.Err() != nil {
			return fail(ctx.Err())
		}
		b, err := ops.alloc(n)
		if b != nil {
			e.resources = append(e.resources, b)
			e.stats.Allocations++
		}
		if err != nil {
			return fail(fmt.Errorf("cudafuse: allocate mhc: %w", err))
		}
	}
	e.output = e.resources[1]
	sources, _ := MHCSource(c)
	for i, s := range sources {
		if ctx != nil && ctx.Err() != nil {
			return fail(ctx.Err())
		}
		k, err := ops.compile(s)
		if k != nil {
			e.kernels = append(e.kernels, k)
		}
		if err != nil {
			var pending io.Closer
			if errors.As(err, &pending) {
				e.pending = append(e.pending, pending)
			}
			return fail(fmt.Errorf("cudafuse: compile mhc: %w", err))
		}
		e.receipts = append(e.receipts, k.receipt)
		if ctx != nil && ctx.Err() != nil {
			return fail(ctx.Err())
		}
		args := []driver.DevicePtr{ptrs[0], e.resources[0].ptr}
		n := c.Batch
		if i == 1 {
			args = []driver.DevicePtr{ptrs[1], ptrs[2], ptrs[3], e.resources[0].ptr, e.output.ptr}
			n = e.n
		}
		e.attempted = true
		if err := k.launch(args, n); err != nil {
			return fail(fmt.Errorf("cudafuse: launch mhc: %w", err))
		}
		e.stats.Launches++
	}
	if err := e.Wait(ctx); err != nil {
		return e, err
	}
	// Check scratch guards and full publication independently of output checks.
	if err := e.resources[0].read(make([]float32, c.Batch*16)); err != nil {
		return fail(err)
	}
	return e, nil
}
