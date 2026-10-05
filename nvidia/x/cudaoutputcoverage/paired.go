package cudaoutputcoverage

import (
	"errors"
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cudatest"
)

// PairedReport keeps written-domain and numerical findings separate. A nil
// NumericalMismatch means the admitted numerical comparison passed; it says
// nothing about whether the launch wrote the output.
type PairedReport struct {
	Written           Report
	NumericalMismatch *cudatest.CUDAOutputMismatch
}

// CheckOutput checks a recorded ordinary readback numerically using the
// integrated CUDA output policy and checks the two-fill readbacks for writes.
// Policy.Elements is the number of domain elements, excluding declared padding.
// All execution identities must match. NumericalMismatch.Index is the logical
// buffer index, not a flattened-domain index.
func CheckOutput(t TwoFill, ordinary []uint32, p cudatest.CUDAOutputPolicy) (PairedReport, error) {
	written, err := ClassifyTwoFill(t)
	if err != nil {
		return PairedReport{}, err
	}
	if len(ordinary) != t.Contract.Elements {
		return PairedReport{}, fmt.Errorf("paired output: ordinary readback extent differs")
	}
	if p.SourceDigest != t.SourceDigest || p.InputDigest != t.InputDigest || p.Target != t.Target || p.MathMode != t.MathMode {
		return PairedReport{}, fmt.Errorf("paired output: numerical and write identities differ")
	}
	n := 0
	for _, r := range t.Contract.Domain {
		n += r.End - r.Start
	}
	if p.Elements != n {
		return PairedReport{}, fmt.Errorf("paired output: numerical extent differs from domain")
	}
	actual := make([]float32, 0, n)
	reference := make([]float32, 0, n)
	indices := make([]int, 0, n)
	for _, r := range t.Contract.Domain {
		for i := r.Start; i < r.End; i++ {
			actual = append(actual, math.Float32frombits(ordinary[i]))
			reference = append(reference, math.Float32frombits(t.Reference[i]))
			indices = append(indices, i)
		}
	}
	out := PairedReport{Written: written}
	err = cudatest.CompareCUDAOutputFP32(actual, reference, p)
	if err == nil {
		return out, nil
	}
	var mismatch *cudatest.CUDAOutputMismatch
	if !errors.As(err, &mismatch) {
		return PairedReport{}, err
	}
	copy := *mismatch
	copy.Index = indices[mismatch.Index]
	out.NumericalMismatch = &copy
	return out, nil
}
