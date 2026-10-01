package cudamem

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudalayout"
)

var (
	// ErrWriteAfterWrite reports differing source addresses for one destination.
	ErrWriteAfterWrite = errors.New("copy has a write-after-write alias")
	// ErrNullspaceInvariant reports a nullspace coordinate with nonzero destination offset.
	ErrNullspaceInvariant = errors.New("destination nullspace has a nonzero offset")
	// ErrCopySizeMismatch reports layouts or slices with incompatible sizes.
	ErrCopySizeMismatch = errors.New("copy size mismatch")
)

// CopyRun describes a contiguous transfer. Offsets and Length are in elements.
type CopyRun struct {
	SourceOffset      int
	DestinationOffset int
	Length            int
}

// CopyPlan is a checked layout-derived copy plan. Obtain one with [PlanCopy];
// its zero value is invalid. VectorWidth is the size of each vector transfer;
// residual transfers have size one. CollapseFactor is the greatest number of
// logical writes collapsed into one destination address.
type CopyPlan struct {
	CommonDomain        cudalayout.Layout
	SourceResidual      cudalayout.Layout
	DestinationResidual cudalayout.Layout
	Nullspace           cudalayout.Layout
	CollapseFactor      int
	VectorWidth         int
	VectorReason        string
	ElementCount        int
	runs                []CopyRun
}

type copyAlgebra struct {
	nullspace    func(cudalayout.Layout) (cudalayout.Layout, error)
	rightInverse func(cudalayout.Layout) (cudalayout.Layout, error)
}

// PlanCopy divides equal-sized layouts by their common domain, collapses the
// compatible destination's stride-zero nullspace, and orders its remaining
// coordinates using the destination's right inverse. The coalesced source
// over that inverse determines a unit-stride vector width. Only compatible
// coordinates are optimized; incompatible modes retain their iteration order.
// Vectorization failures yield scalar transfers and a diagnostic reason.
//
// Unlike PyCuTe's stride-zero nullspace check, an additional safety check
// covers overlapping nonzero strides and aliases between incompatible tiles.
// Equal source addresses collapse; differing addresses return ErrWriteAfterWrite.
// ErrNullspaceInvariant is reserved for malformed owner-algebra results.
// Negative, basis and F2 strides are unsupported by the required algebra.
// The contiguous-run rule does not choose a particular CUDA instruction.
func PlanCopy(src, dst cudalayout.Layout) (CopyPlan, error) {
	return planCopy(src, dst, copyAlgebra{cudalayout.Nullspace, cudalayout.RightInverse})
}

func planCopy(src, dst cudalayout.Layout, algebra copyAlgebra) (CopyPlan, error) {
	if src.Size() == 0 || dst.Size() == 0 {
		return CopyPlan{}, fmt.Errorf("copy layout: %w", cudalayout.ErrUninitialized)
	}
	if src.Size() != dst.Size() {
		return CopyPlan{}, fmt.Errorf("%w: source size %d, destination size %d", ErrCopySizeMismatch, src.Size(), dst.Size())
	}
	if _, err := src.Cosize(); err != nil {
		return CopyPlan{}, fmt.Errorf("copy source storage: %w", err)
	}
	if _, err := dst.Cosize(); err != nil {
		return CopyPlan{}, fmt.Errorf("copy destination storage: %w", err)
	}
	// Cosize also accepts F2 layouts. Ask the owner's integer-only algebra
	// to validate the source kind before any vectorization is attempted.
	if _, err := cudalayout.Nullspace(src); err != nil {
		return CopyPlan{}, fmt.Errorf("copy source algebra: %w", err)
	}
	common, err := cudalayout.GreatestCommonDomain(src, dst)
	if err != nil {
		return CopyPlan{}, fmt.Errorf("copy common domain: %w", err)
	}
	srcCompatible, srcResidual, err := divideCopyLayout(src, common)
	if err != nil {
		return CopyPlan{}, fmt.Errorf("copy source common domain: %w", err)
	}
	dstCompatible, dstResidual, err := divideCopyLayout(dst, common)
	if err != nil {
		return CopyPlan{}, fmt.Errorf("copy destination common domain: %w", err)
	}
	null, err := algebra.nullspace(dstCompatible)
	if err != nil {
		return CopyPlan{}, fmt.Errorf("copy destination nullspace: %w", err)
	}
	if null.Size() == 0 {
		return CopyPlan{}, ErrNullspaceInvariant
	}
	// Check the definition independently from WAW. A bad destination coordinate
	// must not be confused with a source varying over a valid nullspace.
	for i := 0; i < null.Size(); i++ {
		coordinate, err := null.Index(i)
		if err != nil {
			return CopyPlan{}, fmt.Errorf("%w: %v", ErrNullspaceInvariant, err)
		}
		offset, err := dstCompatible.Index(coordinate)
		if err != nil || offset != 0 {
			return CopyPlan{}, fmt.Errorf("%w: coordinate %d", ErrNullspaceInvariant, coordinate)
		}
	}
	variation := 0
	for i := 0; i < null.Size(); i++ {
		coordinate, _ := null.Index(i)
		offset, err := srcCompatible.Index(coordinate)
		if err != nil {
			return CopyPlan{}, fmt.Errorf("copy source nullspace: %w", err)
		}
		if offset != 0 {
			variation = offset
		}
	}
	if variation != 0 {
		return CopyPlan{}, fmt.Errorf("%w: source offset %d across destination nullspace", ErrWriteAfterWrite, variation)
	}
	_, srcNonNull, err := divideCopyLayout(srcCompatible, null)
	if err != nil {
		return CopyPlan{}, fmt.Errorf("copy source nullspace divide: %w", err)
	}
	_, dstNonNull, err := divideCopyLayout(dstCompatible, null)
	if err != nil {
		return CopyPlan{}, fmt.Errorf("copy destination nullspace divide: %w", err)
	}

	srcInner, srcRest, dstInner, dstRest, width, reason := vectorizeCopy(srcNonNull, dstNonNull, algebra.rightInverse)
	plan := CopyPlan{CommonDomain: common, SourceResidual: srcResidual, DestinationResidual: dstResidual,
		Nullspace: null, CollapseFactor: null.Size(), VectorWidth: width, VectorReason: reason, ElementCount: src.Size()}
	var vectors, scalars []CopyRun
	// This map checks aliases outside the owner's stride-zero nullspace. It
	// neither computes the nullspace nor determines destination memory order.
	seen := make(map[int]int)
	counts := make(map[int]int)
	maxAliases := 1
	for residual := 0; residual < srcResidual.Size(); residual++ {
		sr, _ := srcResidual.Index(residual)
		dr, _ := dstResidual.Index(residual)
		for rest := 0; rest < srcRest.Size(); rest++ {
			ss, _ := srcRest.Index(rest)
			ds, _ := dstRest.Index(rest)
			for inner := 0; inner < srcInner.Size(); {
				length := width
				if srcInner.Size()-inner < width {
					length = 1
				}
				so, _ := srcInner.Index(inner)
				do, _ := dstInner.Index(inner)
				run := CopyRun{sr + ss + so, dr + ds + do, length}
				var fresh []CopyRun
				for lane := 0; lane < length; lane++ {
					source, destination := run.SourceOffset+lane, run.DestinationOffset+lane
					if previous, ok := seen[destination]; ok {
						if previous != source {
							return CopyPlan{}, fmt.Errorf("%w: destination offset %d receives source offsets %d and %d", ErrWriteAfterWrite, destination, previous, source)
						}
					} else {
						seen[destination] = source
						fresh = append(fresh, CopyRun{source, destination, 1})
					}
					counts[destination]++
					if counts[destination] > maxAliases {
						maxAliases = counts[destination]
					}
				}
				if length == width && width > 1 && len(fresh) == length {
					vectors = append(vectors, run)
				} else {
					scalars = append(scalars, fresh...)
				}
				inner += length
			}
		}
	}
	plan.CollapseFactor *= maxAliases
	if width > 1 && len(vectors) == 0 {
		plan.VectorWidth = 1
		plan.VectorReason = "overlapping aliases leave only scalar transfers"
	}
	// Dispatch every vector before any residual, without sorting by address.
	plan.runs = append(vectors, scalars...)
	return plan, nil
}

func divideCopyLayout(layout, tiler cudalayout.Layout) (cudalayout.Layout, cudalayout.Layout, error) {
	divided, err := cudalayout.LogicalDivide(layout, tiler)
	if err != nil {
		return cudalayout.Layout{}, cudalayout.Layout{}, err
	}
	inner, err := divided.Mode(0)
	if err != nil {
		return cudalayout.Layout{}, cudalayout.Layout{}, err
	}
	outer, err := divided.Mode(1)
	return inner, outer, err
}

func vectorizeCopy(src, dst cudalayout.Layout, inverse func(cudalayout.Layout) (cudalayout.Layout, error)) (cudalayout.Layout, cudalayout.Layout, cudalayout.Layout, cudalayout.Layout, int, string) {
	trivial, _ := cudalayout.NewLayout(cudalayout.IntShape(1), cudalayout.IntStride(0))
	fallback := func(err error) (cudalayout.Layout, cudalayout.Layout, cudalayout.Layout, cudalayout.Layout, int, string) {
		return src, trivial, dst, trivial, 1, "vectorization failed: " + err.Error()
	}
	order, err := inverse(dst)
	if err != nil {
		return fallback(fmt.Errorf("destination inverse: %w", err))
	}
	si, sr, err := divideCopyLayout(src, order)
	if err != nil {
		return fallback(fmt.Errorf("source over inverse: %w", err))
	}
	di, dr, err := divideCopyLayout(dst, order)
	if err != nil {
		return fallback(fmt.Errorf("destination over inverse: %w", err))
	}
	coalesced := cudalayout.Coalesce(si)
	inner, err := coalesced.Mode(0)
	if err != nil {
		return fallback(fmt.Errorf("coalesced source mode: %w", err))
	}
	width := 1
	reason := "no unit-stride source run over destination inverse"
	if inner.Size() > 1 {
		stride, err := inner.Index(1)
		if err != nil {
			return fallback(fmt.Errorf("source stride: %w", err))
		}
		if stride == 1 {
			width, reason = inner.Size(), "unit-stride coalesced source over destination inverse"
		}
	}
	return si, sr, di, dr, width, reason
}

// Copy executes all vector transfers first, followed by scalar residuals.
// The complete plan and slice bounds are checked before the first transfer.
// Source and destination storage must not overlap.
func Copy[T any](plan CopyPlan, dst, src Slice[T]) error {
	return dispatchCopy(plan, dst.Len(), src.Len(), func(run CopyRun) error {
		return dst.Slice(run.DestinationOffset, run.DestinationOffset+run.Length).CopyDeviceToDevice(src.Slice(run.SourceOffset, run.SourceOffset+run.Length))
	})
}

func dispatchCopy(plan CopyPlan, destinationLength, sourceLength int, transfer func(CopyRun) error) error {
	if plan.VectorWidth < 1 || plan.ElementCount <= 0 || len(plan.runs) == 0 {
		return errors.New("cudamem: invalid copy plan")
	}
	for _, run := range plan.runs {
		if run.Length != 1 && run.Length != plan.VectorWidth {
			return fmt.Errorf("cudamem: invalid copy run length %d", run.Length)
		}
		if run.SourceOffset < 0 || run.DestinationOffset < 0 || run.Length > sourceLength || run.Length > destinationLength || run.SourceOffset > sourceLength-run.Length || run.DestinationOffset > destinationLength-run.Length {
			return fmt.Errorf("%w: run %+v exceeds source length %d or destination length %d", ErrCopySizeMismatch, run, sourceLength, destinationLength)
		}
	}
	for _, run := range plan.runs {
		if err := transfer(run); err != nil {
			return fmt.Errorf("cudamem: copy run: %w", err)
		}
	}
	return nil
}

// CopyWithLayouts plans and copies non-overlapping device slices. Layout and
// aliasing errors are returned before any device transfer.
func CopyWithLayouts[T any](dst, src Slice[T], sourceLayout, destinationLayout cudalayout.Layout) (CopyPlan, error) {
	return planCopyAndApply(sourceLayout, destinationLayout, func(plan CopyPlan) error { return Copy(plan, dst, src) })
}

func planCopyAndApply(src, dst cudalayout.Layout, apply func(CopyPlan) error) (CopyPlan, error) {
	plan, err := PlanCopy(src, dst)
	if err != nil {
		return CopyPlan{}, err
	}
	if err := apply(plan); err != nil {
		return plan, err
	}
	return plan, nil
}
