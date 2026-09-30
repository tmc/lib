package cudalayout

import (
	"cmp"
	"fmt"
	"slices"
)

// RightInverse returns the largest contiguous right inverse r, with
// l(r(i)) == i for every i in [0, r.Size()). A layout without a unit-stride
// start returns the trivial layout 1:0, as PyCuTe does.
//
// This operation supports nonnegative integer strides. Unlike PyCuTe, it
// returns an error for uninitialized, negative, basis and F2 layouts.
func RightInverse(l Layout) (Layout, error) {
	if l.size == 0 {
		return Layout{}, ErrUninitialized
	}
	if err := checkIntLayout(l, "right inverse operand"); err != nil {
		return Layout{}, err
	}
	type mode struct {
		leaf
		indexStride int
	}
	var modes []mode
	prefix := 1
	for _, f := range coalesce(l.leaves) {
		modes = append(modes, mode{f, prefix})
		prefix *= f.n // Bounded by l.Size().
	}
	slices.SortStableFunc(modes, func(a, b mode) int { return cmp.Compare(a.d, b.d) })
	var out []leaf
	covered := 1
	for _, m := range modes {
		if m.d == 0 || m.n == 1 || m.d != covered {
			continue
		}
		out = append(out, leaf{n: m.n, d: m.indexStride})
		covered *= m.n // Each selected mode occurs once in l's domain.
	}
	return NewLayout(leafTree(coalesce(out)))
}

// Nullspace returns the logical coordinates of l's stride-zero modes,
// flattened in colexicographic order with their original domain strides.
// It preserves the shape of those modes, including size-one modes, as
// PyCuTe does. With no stride-zero modes it returns the trivial layout 1:0.
//
// This is the full zero-offset set for nonnegative integer strides.
// Uninitialized, negative, basis and F2 layouts return an error; PyCuTe
// supports additional codomains that this operation does not implement.
func Nullspace(l Layout) (Layout, error) {
	if l.size == 0 {
		return Layout{}, ErrUninitialized
	}
	if err := checkIntLayout(l, "nullspace operand"); err != nil {
		return Layout{}, err
	}
	var out []leaf
	prefix := 1
	for _, f := range l.leaves {
		if f.d == 0 {
			out = append(out, leaf{n: f.n, d: prefix})
		}
		prefix *= f.n // Bounded by l.Size().
	}
	if len(out) == 0 {
		out = append(out, leaf{n: 1})
	}
	return NewLayout(leafTree(out))
}

// GreatestCommonDomain returns the largest ordered factorization shared by
// a.Shape() and b.Shape(), with each factor's aligned logical stride. The
// result depends only on shapes, not on the operands' strides. It is a tiler
// admissible to both layouts, rather than an inverse of their memory offsets.
//
// As in PyCuTe, the result is a flat tuple, even for a single factor. Shapes
// with no aligned common factor return (1):(0). Unlike PyCuTe's unbounded
// integers, an intermediate alignment that exceeds int returns ErrOverflow.
// Uninitialized layouts return ErrUninitialized.
func GreatestCommonDomain(a, b Layout) (Layout, error) {
	if a.size == 0 || b.size == 0 {
		return Layout{}, ErrUninitialized
	}
	as, bs := a.shape.flatten(nil), b.shape.flatten(nil)
	var shapes []Shape
	var strides []Stride
	pa, pb := 1, 1
	for i, j := 0, 0; i < len(as) && j < len(bs); {
		if as[i] == 1 {
			i++
			continue
		}
		if bs[j] == 1 {
			j++
			continue
		}
		aligned, err := mul(pa/gcd(pa, pb), pb)
		if err != nil {
			return Layout{}, fmt.Errorf("common domain alignment: %w", err)
		}
		ra, rb := aligned/pa, aligned/pb
		if as[i]%ra == 0 && bs[j]%rb == 0 {
			factor := gcd(as[i]/ra, bs[j]/rb)
			if factor != 1 {
				shapes = append(shapes, IntShape(factor))
				strides = append(strides, IntStride(aligned))
				as[i] /= ra * factor
				bs[j] /= rb * factor
				pa, err = mul(aligned, factor)
				if err != nil {
					return Layout{}, fmt.Errorf("common domain extent: %w", err)
				}
				pb = pa
				continue
			}
		}
		// Advance whichever leaf ends first in the divisibility order. When
		// neither end divides the other, advance both leaves.
		ea, eb := pa*as[i], pb*bs[j] // Bounded by the input sizes.
		qa, qb := eb%ea == 0, ea%eb == 0
		if qa || !qb {
			pa = ea
			i++
		}
		if qb || !qa {
			pb = eb
			j++
		}
	}
	if len(shapes) == 0 {
		shapes = append(shapes, IntShape(1))
		strides = append(strides, IntStride(0))
	}
	return NewLayout(TupleShape(shapes...), TupleStride(strides...))
}

func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}
