package cudaroofline

import "fmt"

// Precision names the numeric format a unit of work is carried out in, and the
// pipe that carries it. Both halves matter to a peak: an SM's tensor cores and
// its FMA pipes have different FLOP-per-cycle rates, so "FP16" without a pipe
// names two numbers that differ by more than an order of magnitude.
//
// The vocabulary follows the counters the hardware actually separates. DCGM
// reports HMMA cycles (half tensor) and FP16 cycles (half non-tensor) as
// distinct fields, and a figure derived from one is not a figure about the
// other.
type Precision string

const (
	// FP64, FP32 and FP16 are the non-tensor floating-point pipes.
	FP64 Precision = "fp64"
	FP32 Precision = "fp32"
	FP16 Precision = "fp16"

	// TF32Tensor, FP16Tensor, BF16Tensor, FP8Tensor and INT8Tensor are the
	// tensor-core formats. TF32, FP8 and INT8 have no non-tensor pipe, so
	// there is deliberately no unsuffixed constant for them.
	TF32Tensor Precision = "tf32_tensor"
	FP16Tensor Precision = "fp16_tensor"
	BF16Tensor Precision = "bf16_tensor"
	FP8Tensor  Precision = "fp8_tensor"
	INT8Tensor Precision = "int8_tensor"
)

// PeakTFLOPSFor returns the compute peak for one architecture and precision,
// with the provenance of the number returned.
//
// It answers only for precisions this repository holds a number for, which is
// FP32 and nothing else. Ceiling.PeakTFLOPS is the FP32 non-tensor peak -- that
// is what every figure in the table is -- and this function reads that one
// table rather than introducing a second. A tensor-format peak is absent, and
// absent is the answer: an FP16 tensor peak inferred from an FP32 pipe peak by
// multiplying by a ratio from a datasheet nobody here checked would be a
// fabrication dressed as a lookup, and every efficiency computed against it
// would inherit the fabrication silently.
//
// Absence is not an obstacle to the memory axis. Bandwidth does not depend on
// the format of the arithmetic, so Model.Judge still bounds a tensor-precision
// operation from below by its bytes; it just says that the bound rests on one
// axis. See FloorBasis.
func PeakTFLOPSFor(arch string, p Precision) (float64, Provenance, error) {
	if p != FP32 {
		if err := validPrecision(p); err != nil {
			return 0, "", err
		}
		return 0, "", fmt.Errorf("cudaroofline: no %s peak for %q: the table holds FP32 non-tensor peaks only", p, arch)
	}
	// The compute peak does not depend on where the bytes live, so any
	// residency the table knows for this architecture carries the same
	// PeakTFLOPS. Device residency exists for every entry.
	c, err := LookupCeiling(arch, DeviceResident)
	if err != nil {
		return 0, "", err
	}
	return c.PeakTFLOPS, c.ComputeProvenance, nil
}

func validPrecision(p Precision) error {
	switch p {
	case FP64, FP32, FP16, TF32Tensor, FP16Tensor, BF16Tensor, FP8Tensor, INT8Tensor:
		return nil
	}
	return fmt.Errorf("cudaroofline: unknown precision %q", p)
}
