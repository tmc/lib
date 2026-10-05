package cudaoutputcoverage

import (
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatest"
)

func recorded(n int) TwoFill {
	r := TwoFill{Contract: Contract{Buffer: "out", Elements: n + 3, Domain: []Range{{0, n}}, Kind: AtLeastOnce, Deterministic: true}, FillA: math.Float32bits(-101), FillB: math.Float32bits(-202), Geometry: Geometry{Grid: [3]uint32{4, 1, 1}, Block: [3]uint32{256, 1, 1}}, SourceDigest: strings.Repeat("a", 64), InputDigest: strings.Repeat("b", 64), Target: "sm_121a", MathMode: "precise"}
	r.Reference = make([]uint32, n+3)
	r.ReadbackA = make([]uint32, n+3)
	r.ReadbackB = make([]uint32, n+3)
	for i := 0; i < n; i++ {
		r.Reference[i] = math.Float32bits(float32(i + 1))
		r.ReadbackA[i] = r.Reference[i]
		r.ReadbackB[i] = r.Reference[i]
	}
	// Padding differs between executions and is explicitly outside the domain.
	for i := n; i < n+3; i++ {
		r.ReadbackA[i] = r.FillA
		r.ReadbackB[i] = r.FillB
	}
	return r
}

func TestTwoFill_ClassifiesFromRecordedReadbacks(t *testing.T) {
	r := recorded(1003)
	r.Geometry.Grid[0] = 3
	for i := 768; i < 1003; i++ {
		r.ReadbackA[i] = r.FillA
		r.ReadbackB[i] = r.FillB
	}
	out, err := ClassifyTwoFill(r)
	if err != nil || len(out.Unwritten) != 1 || out.Unwritten[0] != (Range{768, 1003}) || len(out.Nondeterministic) != 0 {
		t.Fatal(out, err)
	}
	r.Contract.Domain[0].End = 2
	if out.Contract.Domain[0].End != 1003 {
		t.Fatal("report domain aliases admission")
	}
	healthy, err := ClassifyTwoFill(recorded(1003))
	if err != nil || len(healthy.Unwritten) != 0 || len(healthy.Nondeterministic) != 0 {
		t.Fatal(healthy, err)
	}
}

func TestContract_ExactlyOnceOnAtomicOutputIsContractError(t *testing.T) {
	r := recorded(3)
	r.Contract.Kind = ExactlyOnce
	r.Contract.Atomic = true
	if err := r.Contract.Validate(); err == nil {
		t.Fatal("atomic exactly-once accepted")
	}
	r.Contract.Kind = WriterCount
	r.Contract.ExpectedWriters = 2
	if err := r.Contract.Validate(); err != nil {
		t.Fatal(err)
	}
	if _, err := ClassifyTwoFill(r); err == nil {
		t.Fatal("two-fill counted writers")
	}
}

func TestTwoFillRefusesInvalidAdmission(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*TwoFill)
	}{
		{"equal fills", func(r *TwoFill) { r.FillB = r.FillA }},
		{"signed zero fills", func(r *TwoFill) { r.FillA = 0; r.FillB = 1 << 31 }},
		{"fill matches reference", func(r *TwoFill) { r.FillA = r.Reference[0] }},
		{"nonfinite fill", func(r *TwoFill) { r.FillA = math.Float32bits(float32(math.Inf(1))) }},
		{"nan reference", func(r *TwoFill) { r.Reference[0] = math.Float32bits(float32(math.NaN())) }},
		{"infinite reference", func(r *TwoFill) { r.Reference[0] = math.Float32bits(float32(math.Inf(-1))) }},
		{"overlapping domain", func(r *TwoFill) { r.Contract.Domain = []Range{{0, 2}, {1, 3}} }},
		{"short readback", func(r *TwoFill) { r.ReadbackA = r.ReadbackA[:1] }},
		{"no identity", func(r *TwoFill) { r.InputDigest = "" }},
		{"nondeterministic contract", func(r *TwoFill) { r.Contract.Deterministic = false }},
		{"exactly once", func(r *TwoFill) { r.Contract.Kind = ExactlyOnce }},
		{"zero grid", func(r *TwoFill) { r.Geometry.Grid[0] = 0 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := recorded(3)
			tc.change(&r)
			if _, err := ClassifyTwoFill(r); err == nil {
				t.Fatal("invalid admission accepted")
			}
		})
	}
}

func TestTwoFillNondeterminismDistinct(t *testing.T) {
	r := recorded(4)
	r.ReadbackB[1] = math.Float32bits(50)
	r.ReadbackA[2] = r.FillA
	out, err := ClassifyTwoFill(r)
	if err != nil || len(out.Unwritten) != 0 || len(out.Nondeterministic) != 1 || out.Nondeterministic[0] != (Range{1, 3}) {
		t.Fatal(out, err)
	}
}

func policy(r TwoFill) cudatest.CUDAOutputPolicy {
	n := 0
	for _, d := range r.Contract.Domain {
		n += d.End - d.Start
	}
	return cudatest.CUDAOutputPolicy{Target: r.Target, DType: "fp32", MathMode: r.MathMode, SourceDigest: r.SourceDigest, InputDigest: r.InputDigest, Elements: n, NaN: "reject", Infinity: "reject", Zero: "bits"}
}

func TestOutputContract_ResidueAndWrongValueAreDistinct(t *testing.T) {
	for _, name := range []string{"residue", "wrong value", "healthy"} {
		t.Run(name, func(t *testing.T) {
			r := recorded(1003)
			ordinary := append([]uint32(nil), r.Reference...)
			if name == "residue" {
				r.Geometry.Grid[0] = 3
				for i := 768; i < 1003; i++ {
					r.ReadbackA[i] = r.FillA
					r.ReadbackB[i] = r.FillB
				}
			}
			if name == "wrong value" {
				ordinary[29] = math.Float32bits(-9)
				r.ReadbackA[29] = ordinary[29]
				r.ReadbackB[29] = ordinary[29]
			}
			out, err := CheckOutput(r, ordinary, policy(r))
			if err != nil {
				t.Fatal(err)
			}
			if (len(out.Written.Unwritten) != 0) != (name == "residue") {
				t.Fatal(out)
			}
			if (out.NumericalMismatch != nil) != (name == "wrong value") {
				t.Fatal(out)
			}
			if name == "wrong value" && out.NumericalMismatch.Index != 29 {
				t.Fatal(out.NumericalMismatch)
			}
		})
	}
}
