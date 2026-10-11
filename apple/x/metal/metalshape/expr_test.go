package metalshape

import (
	"math"
	"testing"
)

func TestDimensionBoundsContainValues(t *testing.T) {
	a, _ := DimensionSymbol("a")
	b, _ := DimensionSymbol("b")
	ceil, _ := DimensionCeilDiv(a, b)
	floor, _ := DimensionFloorDiv(a, b)
	mod, _ := DimensionMod(a, b)
	tests := []struct {
		name  string
		expr  Expr
		value func(int64, int64) int64
	}{
		{"add", DimensionAdd(a, b), func(a, b int64) int64 { return a + b }},
		{"mul", DimensionMul(a, b), func(a, b int64) int64 { return a * b }},
		{"ceil", ceil, func(a, b int64) int64 { return (a + b - 1) / b }},
		{"floor", floor, func(a, b int64) int64 { return a / b }},
		{"mod", mod, func(a, b int64) int64 { return a % b }},
		{"min", DimensionMin(a, b), func(a, b int64) int64 {
			if a < b {
				return a
			}
			return b
		}},
		{"max", DimensionMax(a, b), func(a, b int64) int64 {
			if a > b {
				return a
			}
			return b
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for loA := int64(0); loA <= 4; loA++ {
				for hiA := loA; hiA <= 7; hiA++ {
					for loB := int64(1); loB <= 4; loB++ {
						for hiB := loB; hiB <= 7; hiB++ {
							intervals := map[string]Interval{"a": {loA, hiA}, "b": {loB, hiB}}
							bounds, err := tt.expr.Bounds(intervals)
							if err != nil {
								t.Fatal(err)
							}
							simplified, err := tt.expr.SimplifyWithIntervals(intervals)
							if err != nil {
								t.Fatal(err)
							}
							for av := loA; av <= hiA; av++ {
								for bv := loB; bv <= hiB; bv++ {
									want := tt.value(av, bv)
									values := map[string]int64{"a": av, "b": bv}
									got, err := tt.expr.Eval(values)
									if err != nil || got != want || want < bounds.Min || want > bounds.Max {
										t.Fatalf("a=%d b=%d got=%d want=%d bounds=%v err=%v", av, bv, got, want, bounds, err)
									}
									got, err = simplified.Eval(values)
									if err != nil || got != want {
										t.Fatalf("simplification changed value a=%d b=%d got=%d want=%d err=%v", av, bv, got, want, err)
									}
								}
							}
						}
					}
				}
			}
		})
	}
}

func TestDimensionRefusesUnknownZeroDivisorAndOverflow(t *testing.T) {
	a, _ := DimensionSymbol("a")
	b, _ := DimensionSymbol("b")
	div, _ := DimensionCeilDiv(a, b)
	for _, values := range []map[string]int64{{"a": 1}, {"a": 1, "b": 0}, {"a": -1, "b": 1}} {
		if _, err := div.Eval(values); err == nil {
			t.Fatalf("accepted %v", values)
		}
	}
	if _, err := div.Bounds(map[string]Interval{"a": {0, 3}, "b": {0, 1}}); err == nil {
		t.Fatal("accepted possible zero divisor")
	}
	max, _ := DimensionConstant(math.MaxInt64)
	one, _ := DimensionConstant(1)
	sum := DimensionAdd(max, one)
	if _, err := sum.Eval(nil); err == nil {
		t.Fatal("accepted overflow")
	}
	if _, err := sum.Bounds(nil); err == nil {
		t.Fatal("bounds accepted overflow")
	}
	var zero Expr
	if got, err := zero.Eval(nil); err != nil || got != 0 {
		t.Fatalf("zero=%d err=%v", got, err)
	}
}
