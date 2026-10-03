package cudatest

import (
	"errors"
	"math"
	"strings"
	"testing"
)

func outputPolicy(n int) CUDAOutputPolicy {
	return CUDAOutputPolicy{Target: "sm_90", DType: "fp32", MathMode: "default-fp32", SourceDigest: strings.Repeat("1", 64), InputDigest: strings.Repeat("2", 64), Elements: n, NaN: "reject", Infinity: "reject", Zero: "equal"}
}

func TestCUDAOutput_RejectsNaNAgainstFiniteReference(t *testing.T) {
	for _, bits := range []uint32{0x7fc00000, 0xffc00123, 0x7f800001} {
		err := CompareCUDAOutputFP32([]float32{math.Float32frombits(bits)}, []float32{1}, outputPolicy(1))
		var m *CUDAOutputMismatch
		if !errors.As(err, &m) || m.ActualBits != bits {
			t.Fatalf("NaN accepted or lost: %v", err)
		}
	}
}

func TestCUDAOutput_InfinityAndSignedZeroPolicies(t *testing.T) {
	for _, sign := range []int{1, -1} {
		inf := float32(math.Inf(sign))
		p := outputPolicy(1)
		if !errors.Is(CompareCUDAOutputFP32([]float32{inf}, []float32{inf}, p), ErrInvalidCUDAOutput) {
			t.Fatal("refused reference admitted")
		}
		p.Infinity = "same-sign"
		if err := CompareCUDAOutputFP32([]float32{inf}, []float32{inf}, p); err != nil {
			t.Fatal(err)
		}
		if CompareCUDAOutputFP32([]float32{-inf}, []float32{inf}, p) == nil {
			t.Fatal("opposite infinity admitted")
		}
		if CompareCUDAOutputFP32([]float32{inf}, []float32{1}, p) == nil {
			t.Fatal("infinite output against finite reference admitted")
		}
	}
	minus := math.Float32frombits(0x80000000)
	p := outputPolicy(1)
	if err := CompareCUDAOutputFP32([]float32{minus}, []float32{0}, p); err != nil {
		t.Fatal(err)
	}
	p.Zero = "bits"
	if CompareCUDAOutputFP32([]float32{minus}, []float32{0}, p) == nil {
		t.Fatal("signed-zero policy ignored")
	}
	if err := CompareCUDAOutputFP32([]float32{minus}, []float32{minus}, p); err != nil {
		t.Fatal(err)
	}
	p = outputPolicy(1)
	p.NaN = "semantic"
	if err := CompareCUDAOutputFP32([]float32{math.Float32frombits(0x7fc00001)}, []float32{math.Float32frombits(0xffc00456)}, p); err != nil {
		t.Fatal(err)
	}
	if CompareCUDAOutputFP32([]float32{1}, []float32{float32(math.NaN())}, p) == nil {
		t.Fatal("finite equals semantic NaN")
	}
}

func TestCUDAOutput_InvalidToleranceAndLength(t *testing.T) {
	for _, v := range []float64{-1, math.NaN(), math.Inf(1), math.Inf(-1)} {
		for _, rel := range []bool{false, true} {
			p := outputPolicy(1)
			if rel {
				p.Relative = v
			} else {
				p.Absolute = v
			}
			if !errors.Is(CompareCUDAOutputFP32([]float32{1}, []float32{1}, p), ErrInvalidCUDAOutput) {
				t.Fatal("invalid tolerance admitted")
			}
		}
	}
	for _, pair := range [][2][]float32{{nil, {1}}, {{1}, nil}, {{1}, {1, 2}}, {{1}, {1}}} {
		if !errors.Is(CompareCUDAOutputFP32(pair[0], pair[1], outputPolicy(2)), ErrInvalidCUDAOutput) {
			t.Fatal("short readback/reference admitted")
		}
	}
	if err := CompareCUDAOutputFP32(nil, nil, outputPolicy(0)); err != nil {
		t.Fatal(err)
	}
	for _, alter := range []func(*CUDAOutputPolicy){func(p *CUDAOutputPolicy) { p.Elements = -1 }, func(p *CUDAOutputPolicy) { p.DType = "bf16" }, func(p *CUDAOutputPolicy) { p.NaN = "payload" }, func(p *CUDAOutputPolicy) { p.Zero = "unknown" }, func(p *CUDAOutputPolicy) { p.Infinity = "any" }, func(p *CUDAOutputPolicy) { p.SourceDigest = "" }, func(p *CUDAOutputPolicy) { p.InputDigest = "ABC" }, func(p *CUDAOutputPolicy) { p.Target = "" }, func(p *CUDAOutputPolicy) { p.MathMode = "" }} {
		p := outputPolicy(1)
		alter(&p)
		if !errors.Is(CompareCUDAOutputFP32([]float32{1}, []float32{1}, p), ErrInvalidCUDAOutput) {
			t.Fatal("invalid policy admitted")
		}
	}
}

func TestCUDAOutput_ToleranceBoundary(t *testing.T) {
	for _, rel := range []bool{false, true} {
		p := outputPolicy(1)
		if rel {
			p.Relative = 0.5
		} else {
			p.Absolute = 1
		}
		for _, tc := range []struct {
			value float32
			pass  bool
		}{{math.Nextafter32(2, 1), true}, {2, true}, {math.Nextafter32(2, 3), false}} {
			err := CompareCUDAOutputFP32([]float32{tc.value}, []float32{1}, p)
			if (err == nil) != tc.pass {
				t.Fatalf("relative=%t value=%g: %v", rel, tc.value, err)
			}
		}
	}
	p := outputPolicy(1)
	p.Relative = math.MaxFloat64
	if err := CompareCUDAOutputFP32([]float32{math.MaxFloat32}, []float32{-math.MaxFloat32}, p); err != nil {
		t.Fatal("overflow-safe large tolerance:", err)
	}
	p = outputPolicy(1)
	p.Absolute, p.Relative = 0.5, 0.25
	if err := CompareCUDAOutputFP32([]float32{2}, []float32{1}, p); err != nil {
		t.Fatal("combined tolerance boundary:", err)
	}
	if CompareCUDAOutputFP32([]float32{math.Nextafter32(2, 3)}, []float32{1}, p) == nil {
		t.Fatal("combined tolerance above boundary admitted")
	}
}

func TestCUDAOutput_FirstMismatchHasIndexAndBits(t *testing.T) {
	got := []float32{1, 2, 3, 9}
	want := []float32{1, 2, 4, 5}
	p := outputPolicy(4)
	err := CompareCUDAOutputFP32(got, want, p)
	var m *CUDAOutputMismatch
	if !errors.As(err, &m) || m.Index != 2 || m.ActualBits != math.Float32bits(3) || m.ExpectedBits != math.Float32bits(4) || m.Policy != p {
		t.Fatalf("mismatch receipt: %v", err)
	}
	for _, s := range []string{p.SourceDigest, p.InputDigest, p.Target, p.MathMode, "0x40400000"} {
		if !strings.Contains(err.Error(), s) {
			t.Fatalf("receipt omits %q", s)
		}
	}
	got = []float32{1, 2, 4, 9}
	if CompareCUDAOutputFP32(got, want, p) == nil {
		t.Fatal("last element ignored")
	}
}
