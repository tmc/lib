package cudamoe_test

import (
	"math"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamoe"
)

func TestGatherExtentAdmission(t *testing.T) {
	for _, checked := range []bool{false, true} {
		name := "Gather16"
		if checked {
			name = "Gather16Checked"
		}
		t.Run(name, func(t *testing.T) {
			gather := func(experts [][]float32, indices []int, width int) ([]float32, error) {
				if checked {
					return cudamoe.Gather16Checked(experts, indices, width, len(experts))
				}
				return cudamoe.Gather16(experts, indices, width)
			}
			for _, tc := range []struct {
				name    string
				experts [][]float32
				indices []int
				width   int
			}{
				{"overflow", [][]float32{{}}, []int{0, 0}, math.MaxInt - 15},
				{"short row before allocation", [][]float32{{}}, []int{0}, math.MaxInt - 15},
				{"bad index before allocation", [][]float32{{}}, []int{1}, math.MaxInt - 15},
			} {
				t.Run(tc.name, func(t *testing.T) {
					got, err := gather(tc.experts, tc.indices, tc.width)
					if err == nil || got != nil {
						t.Fatalf("gather = %v, %v; want nil output and error", got, err)
					}
				})
			}
			a, b := make([]float32, 16), make([]float32, 16)
			for i := range a {
				a[i], b[i] = float32(i), float32(100+i)
			}
			want := append(append(append([]float32{}, b...), a...), b...)
			got, err := gather([][]float32{a, b}, []int{1, 0, 1}, 16)
			if err != nil || !reflect.DeepEqual(got, want) {
				t.Fatalf("gather = %v, %v; want %v", got, err, want)
			}
			got[0] = -1
			if b[0] != 100 {
				t.Fatal("output aliases expert row")
			}
			got, err = gather(nil, nil, 16)
			if err != nil || got == nil || len(got) != 0 {
				t.Fatalf("empty gather = %v, %v; want nonnil empty output", got, err)
			}
		})
	}
}
