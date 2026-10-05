package cudahotswap

import (
	"encoding/json"
	"fmt"
	"math"
	"testing"
)

func TestMeasureReplay(t *testing.T) {
	negativeZero := math.Float32frombits(0x80000000)
	for _, tc := range []struct {
		name      string
		got, want []float32
		abs, rms  float64
		ulp       uint32
		exact     bool
	}{
		{"equal", []float32{1, -2}, []float32{1, -2}, 0, 0, 0, true},
		{"signed-zero", []float32{negativeZero}, []float32{0}, 0, 0, 0, false},
		{"next-positive", []float32{math.Float32frombits(0x3f800001)}, []float32{1}, math.Ldexp(1, -23), math.Ldexp(1, -23), 1, false},
		{"next-negative", []float32{math.Float32frombits(0xbf800001)}, []float32{-1}, math.Ldexp(1, -23), math.Ldexp(1, -23), 1, false},
		{"cross-zero", []float32{math.SmallestNonzeroFloat32}, []float32{-math.SmallestNonzeroFloat32}, 2 * math.SmallestNonzeroFloat32, 2 * math.SmallestNonzeroFloat32, 2, false},
		{"rms", []float32{3, 4}, []float32{0, 0}, 4, math.Sqrt(12.5), 0x40800000, false},
		{"finite-extremes", []float32{math.MaxFloat32}, []float32{-math.MaxFloat32}, 2 * float64(math.MaxFloat32), 2 * float64(math.MaxFloat32), 2 * 0x7f7fffff, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m, err := MeasureReplay(tc.got, tc.want)
			if err != nil {
				t.Fatal(err)
			}
			if m.Elements != len(tc.want) || m.NonFinite != 0 || m.Exact != tc.exact || m.MaxAbs == nil || m.RMS == nil || m.MaxULP == nil || *m.MaxAbs != tc.abs || *m.RMS != tc.rms || *m.MaxULP != tc.ulp {
				t.Fatalf("metrics=%+v want abs=%g rms=%g ulp=%d exact=%t", m, tc.abs, tc.rms, tc.ulp, tc.exact)
			}
		})
	}
}

func TestMeasureReplayRefusals(t *testing.T) {
	for _, want := range [][]float32{nil, {float32(math.NaN())}, {float32(math.Inf(1))}} {
		if _, err := MeasureReplay(want, want); err == nil {
			t.Fatalf("invalid reference accepted: %v", want)
		}
	}
	if _, err := MeasureReplay([]float32{1}, []float32{1, 2}); err == nil {
		t.Fatal("length mismatch accepted")
	}
	m, err := MeasureReplay([]float32{float32(math.NaN()), float32(math.Inf(-1)), 3}, []float32{1, 2, 3})
	if err != nil || m.NonFinite != 2 || m.Exact || m.MaxAbs != nil || m.RMS != nil || m.MaxULP != nil {
		t.Fatalf("nonfinite metrics=%+v err=%v", m, err)
	}
	if _, err := json.Marshal(m); err != nil {
		t.Fatal(err)
	}
}

func ExampleMeasureReplay() {
	m, err := MeasureReplay([]float32{1, 2}, []float32{1.25, 2.25})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("%d %.2f %.2f %t\n", m.Elements, *m.MaxAbs, *m.RMS, m.Exact)
	// Output: 2 0.25 0.25 false
}

func ExampleReplayMetrics() {
	m, err := MeasureReplay([]float32{1}, []float32{1})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(m.Elements, m.Exact)
	// Output: 1 true
}

func BenchmarkMeasureReplay(b *testing.B) {
	got, want := make([]float32, 4096), make([]float32, 4096)
	for i := range got {
		got[i], want[i] = float32(i)/4, float32(i)/4
	}
	b.ReportAllocs()
	b.SetBytes(int64(len(got) * 8))
	for b.Loop() {
		if _, err := MeasureReplay(got, want); err != nil {
			b.Fatal(err)
		}
	}
}

func FuzzMeasureReplay(f *testing.F) {
	for _, pair := range [][2]uint32{{0, 0x80000000}, {1, 0x80000001}, {0x3f800000, 0x3f800001}, {0x7f7fffff, 0xff7fffff}, {0x7fc00001, 0x3f800000}} {
		f.Add(pair[0], pair[1])
	}
	f.Fuzz(func(t *testing.T, a, b uint32) {
		got, want := math.Float32frombits(a), math.Float32frombits(b)
		m, err := MeasureReplay([]float32{got}, []float32{want})
		if math.IsNaN(float64(want)) || math.IsInf(float64(want), 0) {
			if err == nil {
				t.Fatal("nonfinite reference accepted")
			}
			return
		}
		if err != nil {
			t.Fatal(err)
		}
		if _, err := json.Marshal(m); err != nil {
			t.Fatal(err)
		}
		if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) {
			if m.NonFinite != 1 || m.Exact || m.MaxAbs != nil || m.RMS != nil || m.MaxULP != nil {
				t.Fatal("nonfinite candidate has finite metrics")
			}
			return
		}
		reverse, err := MeasureReplay([]float32{want}, []float32{got})
		if err != nil || m.Exact != (a == b) || *m.MaxAbs != math.Abs(float64(got)-float64(want)) || *m.RMS != *m.MaxAbs || *m.MaxULP != *reverse.MaxULP {
			t.Fatalf("asymmetric or incorrect metrics: %+v %+v err=%v", m, reverse, err)
		}
	})
}
