package cudalayout

import (
	"errors"
	"math"
	"strconv"
	"testing"
)

func TestXORSwizzleConstructorBounds(t *testing.T) {
	width := min(62, strconv.IntSize-1)
	tests := []struct {
		name              string
		bits, base, shift int
		want              error
	}{
		{"negative bits", -1, 0, 1, ErrUnsupported},
		{"negative base", 1, -1, 1, ErrUnsupported},
		{"positive overlap", 2, 0, 1, ErrUnsupported},
		{"negative overlap", 2, 0, -1, ErrUnsupported},
		{"max base addition", 1, math.MaxInt, 3, ErrOverflow},
		{"max shift addition", 1, 0, math.MaxInt, ErrOverflow},
		{"min shift negation", 1, 0, math.MinInt, ErrOverflow},
		{"min shift identity", 0, 0, math.MinInt, ErrOverflow},
		{"max bits", math.MaxInt, 0, math.MaxInt, ErrOverflow},
		{"min base", 1, math.MinInt, 1, ErrUnsupported},
		{"min bits", math.MinInt, 0, 1, ErrUnsupported},
		{"word width positive", 1, 0, strconv.IntSize, ErrOverflow},
		{"word width negative", 1, 0, -strconv.IntSize, ErrOverflow},
		{"last positive range", 1, width - 2, 1, nil},
		{"last negative range", 1, width - 2, -1, nil},
		{"positive range past end", 1, width - 1, 1, ErrOverflow},
		{"negative range past end", 1, width - 1, -1, ErrOverflow},
		{"identity base boundary", 0, width, 0, nil},
		{"identity shift boundary", 0, 0, width, nil},
		{"identity negative shift boundary", 0, 0, -width, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewXORSwizzle(tt.bits, tt.base, tt.shift)
			if !errors.Is(err, tt.want) {
				t.Fatalf("NewXORSwizzle(%d, %d, %d) = %v, want %v", tt.bits, tt.base, tt.shift, err, tt.want)
			}
		})
	}
}

// Evaluate by toggling individual destination bits, independently of Apply's mask.
func scalarXORSwizzle(offset, bits, base, shift int) int {
	out := offset
	for k := range bits {
		src, dst := base+shift+k, base+k
		if shift < 0 {
			src, dst = base+k, base-shift+k
		}
		if offset>>src&1 != 0 {
			out ^= 1 << dst
		}
	}
	return out
}

func TestXORSwizzleSignedBoundaryAlgebra(t *testing.T) {
	width := min(62, strconv.IntSize-1)
	for _, p := range [][3]int{{0, 0, 0}, {1, 0, 1}, {1, 0, -1}, {1, 5, -3}, {1, width - 2, 1}, {1, width - 2, -1}, {1, 0, width - 1}, {1, 0, 1 - width}} {
		s, err := NewXORSwizzle(p[0], p[1], p[2])
		if err != nil {
			t.Fatal(err)
		}
		values := []int{0, 1, 2, 3, 7, 31, 127, 1 << (width - 1), math.MaxInt}
		for _, x := range values {
			if got, want := s.Apply(x), scalarXORSwizzle(x, p[0], p[1], p[2]); got != want {
				t.Fatalf("%v.Apply(%d) = %d, want %d", s, x, got, want)
			}
			if s.Apply(s.Apply(x)) != x {
				t.Fatalf("%v inverse failed at %d", s, x)
			}
			for _, y := range values {
				if s.Apply(x^y) != s.Apply(x)^s.Apply(y) {
					t.Fatalf("%v linearity failed at %d, %d", s, x, y)
				}
			}
		}
		base, err := NewLayout(IntShape(8), IntStride(1))
		if err != nil {
			t.Fatal(err)
		}
		composed, err := s.Compose(base)
		if err != nil {
			t.Fatal(err)
		}
		for i := range base.Size() {
			got, err := composed.Index(i)
			if want := scalarXORSwizzle(i, p[0], p[1], p[2]); err != nil || got != want {
				t.Fatalf("%v composition at %d = %d, %v; want %d", s, i, got, err, want)
			}
		}
		for _, i := range []int{-1, composed.Size()} {
			if _, err := composed.Index(i); !errors.Is(err, ErrOutOfBounds) {
				t.Fatalf("%v Index(%d) = %v, want ErrOutOfBounds", s, i, err)
			}
		}
	}
}
