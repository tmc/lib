package cudagraph_test

import (
	"math"
	"testing"
)

func TestCheckPDLNumerical(t *testing.T) {
	for _, tt := range []struct {
		name string
		want []float32
		got  []float32
		ok   bool
	}{
		{name: "equal", want: []float32{1, 2}, got: []float32{1, 2.00005}, ok: true},
		{name: "different", want: []float32{1}, got: []float32{1.001}, ok: false},
		{name: "nan", want: []float32{1}, got: []float32{float32(math.NaN())}, ok: false},
		{name: "length", want: []float32{1}, got: nil, ok: false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if err := checkPDLNumerical(tt.want, tt.got); (err == nil) != tt.ok {
				t.Fatalf("checkPDLNumerical() error = %v, want success %t", err, tt.ok)
			}
		})
	}
}
