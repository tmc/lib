package builder

import (
	"fmt"
	"math"
)

func checkType(t Type) error {
	if t.ElementType == "" {
		return fmt.Errorf("missing element type")
	}
	for _, d := range t.Shape {
		if d < -1 {
			return fmt.Errorf("invalid dimension %d", d)
		}
	}
	return nil
}

func sameElements(types ...Type) error {
	for _, t := range types {
		if err := checkType(t); err != nil {
			return err
		}
		if t.ElementType != types[0].ElementType {
			return fmt.Errorf("element types differ: %s and %s", types[0].ElementType, t.ElementType)
		}
	}
	return nil
}

func checkShape(got Type, want []int64) error {
	if len(got.Shape) != len(want) {
		return fmt.Errorf("result rank %d, want %d", len(got.Shape), len(want))
	}
	for i, d := range want {
		if d >= 0 && got.Shape[i] >= 0 && got.Shape[i] != d {
			return fmt.Errorf("result dimension %d is %d, want %d", i, got.Shape[i], d)
		}
	}
	return nil
}

func axes(rank int, groups ...[]int64) ([]bool, error) {
	used := make([]bool, rank)
	for _, group := range groups {
		for _, d := range group {
			if d < 0 || d >= int64(rank) {
				return nil, fmt.Errorf("axis %d outside rank %d", d, rank)
			}
			if used[d] {
				return nil, fmt.Errorf("duplicate axis %d", d)
			}
			used[d] = true
		}
	}
	return used, nil
}

func compatibleDim(a, b int64) bool { return a < 0 || b < 0 || a == b }

func dotShape(lhs, rhs Type, lhsBatch, rhsBatch, lhsContract, rhsContract []int64) ([]int64, error) {
	if len(lhsBatch) != len(rhsBatch) || len(lhsContract) != len(rhsContract) {
		return nil, fmt.Errorf("dimension lists have different lengths")
	}
	lused, err := axes(len(lhs.Shape), lhsBatch, lhsContract)
	if err != nil {
		return nil, err
	}
	rused, err := axes(len(rhs.Shape), rhsBatch, rhsContract)
	if err != nil {
		return nil, err
	}
	for _, pair := range [][2][]int64{{lhsBatch, rhsBatch}, {lhsContract, rhsContract}} {
		for i, l := range pair[0] {
			r := pair[1][i]
			if !compatibleDim(lhs.Shape[l], rhs.Shape[r]) {
				return nil, fmt.Errorf("paired axes %d and %d have different sizes", l, r)
			}
		}
	}
	var shape []int64
	for i, d := range lhsBatch {
		n := lhs.Shape[d]
		if n < 0 {
			n = rhs.Shape[rhsBatch[i]]
		}
		shape = append(shape, n)
	}
	for i, n := range lhs.Shape {
		if !lused[i] {
			shape = append(shape, n)
		}
	}
	for i, n := range rhs.Shape {
		if !rused[i] {
			shape = append(shape, n)
		}
	}
	return shape, nil
}

func elements(shape []int64) (n int64, known bool, err error) {
	for _, d := range shape {
		if d == 0 {
			return 0, true, nil
		}
	}
	for _, d := range shape {
		if d < 0 {
			return 0, false, nil
		}
	}
	n = 1
	for _, d := range shape {
		if d > math.MaxInt64/n {
			return 0, false, fmt.Errorf("element count overflows int64")
		}
		n *= d
	}
	return n, true, nil
}

func floatType(typ string) bool {
	switch typ {
	case "f16", "bf16", "f32", "f64":
		return true
	}
	return false
}
