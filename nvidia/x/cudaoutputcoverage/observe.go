package cudaoutputcoverage

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
)

// Observe executes the two-fill arm. run must fill the declared buffer with the
// supplied FP32 bit pattern, launch the same admitted artifact/input/geometry,
// synchronize, and return the entire buffer readback. It must return only after
// work is complete, including on cancellation. A nil run reports the shared
// hardware-pending sentinel, never all-written. Observe validates admission
// before calling run and copies each readback before the next execution.
// It returns raw completed readbacks for provenance and subsequent reporting.
func Observe(ctx context.Context, spec TwoFill, run func(context.Context, uint32) ([]uint32, error)) (TwoFill, Report, error) {
	if err := validateTwoFill(spec); err != nil {
		return TwoFill{}, Report{}, err
	}
	if run == nil {
		return TwoFill{}, Report{}, cudaerrors.ErrHardwarePending
	}
	if ctx == nil {
		return TwoFill{}, Report{}, fmt.Errorf("two fill: nil context")
	}
	spec.Contract.Domain = append([]Range(nil), spec.Contract.Domain...)
	spec.Reference = append([]uint32(nil), spec.Reference...)
	read := func(fill uint32) ([]uint32, error) {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		b, err := run(ctx, fill)
		if err != nil {
			return nil, err
		}
		if len(b) != spec.Contract.Elements {
			return nil, fmt.Errorf("two fill: completed readback extent differs")
		}
		return append([]uint32(nil), b...), ctx.Err()
	}
	var err error
	spec.ReadbackA, err = read(spec.FillA)
	if err != nil {
		return spec, Report{}, fmt.Errorf("two fill A: %w", err)
	}
	spec.ReadbackB, err = read(spec.FillB)
	if err != nil {
		return spec, Report{}, fmt.Errorf("two fill B: %w", err)
	}
	out, err := ClassifyTwoFill(spec)
	return spec, out, err
}
