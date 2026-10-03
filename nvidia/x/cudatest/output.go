package cudatest

import (
	"errors"
	"fmt"
	"math"
	"strings"
)

// ErrInvalidCUDAOutput distinguishes a malformed policy or readback from a
// numerical mismatch. An empty domain is valid only when Elements is zero
// and both buffers are empty.
var ErrInvalidCUDAOutput = errors.New("invalid CUDA output comparison")

// CUDAOutputPolicy identifies a bounded CUDA FP32 comparison. Its zero value
// is invalid. SourceDigest and InputDigest are lowercase SHA-256 receipts;
// Target and MathMode describe the actual compiled consumer, not a default.
// NaN is "reject" or "semantic" (both values NaN, without payload equality).
// Infinity is "reject" or "same-sign". Zero is "equal" or "bits".
// Finite values pass when abs(actual-expected) <= Absolute + Relative *
// max(abs(actual), abs(expected)). The implementation rearranges this formula
// to avoid overflowing a tolerance product or sum.
type CUDAOutputPolicy struct {
	Target, DType, MathMode   string
	SourceDigest, InputDigest string
	Elements                  int
	Absolute, Relative        float64
	NaN, Infinity, Zero       string
}

// CUDAOutputMismatch retains the first bad logical index, FP32 values and bits,
// and the policy identities. It reports output disagreement, not bad input.
type CUDAOutputMismatch struct {
	Index                    int
	Actual, Expected         float32
	ActualBits, ExpectedBits uint32
	Policy                   CUDAOutputPolicy
}

// Error describes the first disagreement including its reproducible identities.
func (m *CUDAOutputMismatch) Error() string {
	return fmt.Sprintf("CUDA output[%d]: actual=%g (0x%08x), expected=%g (0x%08x); target=%s dtype=%s math=%s source=%s input=%s policy={abs:%g rel:%g nan:%s infinity:%s zero:%s elements:%d}", m.Index, m.Actual, m.ActualBits, m.Expected, m.ExpectedBits, m.Policy.Target, m.Policy.DType, m.Policy.MathMode, m.Policy.SourceDigest, m.Policy.InputDigest, m.Policy.Absolute, m.Policy.Relative, m.Policy.NaN, m.Policy.Infinity, m.Policy.Zero, m.Policy.Elements)
}

// CompareCUDAOutputFP32 checks every declared logical element. It does not
// normalize buffers, establish that elements were written, or replace trace
// hashing. Invalid reference values under the chosen policy are input errors;
// invalid actual values are mismatches.
func CompareCUDAOutputFP32(actual, expected []float32, p CUDAOutputPolicy) error {
	bad := func(s string) error { return fmt.Errorf("%w: %s", ErrInvalidCUDAOutput, s) }
	if strings.TrimSpace(p.Target) == "" || strings.TrimSpace(p.MathMode) == "" || p.DType != "fp32" || !outputDigest(p.SourceDigest) || !outputDigest(p.InputDigest) {
		return bad("missing or unsupported CUDA policy identity")
	}
	if p.Elements < 0 || len(actual) != p.Elements || len(expected) != p.Elements {
		return bad("declared extent and buffer lengths differ")
	}
	if p.Absolute < 0 || p.Relative < 0 || math.IsNaN(p.Absolute) || math.IsNaN(p.Relative) || math.IsInf(p.Absolute, 0) || math.IsInf(p.Relative, 0) {
		return bad("tolerances must be finite and nonnegative")
	}
	if (p.NaN != "reject" && p.NaN != "semantic") || (p.Infinity != "reject" && p.Infinity != "same-sign") || (p.Zero != "equal" && p.Zero != "bits") {
		return bad("unknown special-value policy")
	}
	for _, v := range expected {
		if (math.IsNaN(float64(v)) && p.NaN == "reject") || (math.IsInf(float64(v), 0) && p.Infinity == "reject") {
			return bad("reference contains a refused nonfinite value")
		}
	}
	for i, got := range actual {
		want := expected[i]
		g, w := float64(got), float64(want)
		ok := false
		switch {
		case math.IsNaN(g) || math.IsNaN(w):
			ok = p.NaN == "semantic" && math.IsNaN(g) && math.IsNaN(w)
		case math.IsInf(g, 0) || math.IsInf(w, 0):
			ok = p.Infinity == "same-sign" && g == w
		case g == 0 && w == 0:
			ok = p.Zero == "equal" || math.Float32bits(got) == math.Float32bits(want)
		default:
			diff := math.Abs(g - w) // FP32 inputs cannot overflow FP64 subtraction.
			scale := math.Max(math.Abs(g), math.Abs(w))
			ok = diff <= p.Absolute || (scale > 0 && (diff-p.Absolute)/scale <= p.Relative)
		}
		if !ok {
			return &CUDAOutputMismatch{i, got, want, math.Float32bits(got), math.Float32bits(want), p}
		}
	}
	return nil
}

func outputDigest(s string) bool {
	if len(s) != 64 {
		return false
	}
	for _, c := range s {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
}
