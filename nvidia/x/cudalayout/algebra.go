package cudalayout

import (
	"fmt"
	"math"
	"math/bits"
	"slices"
)

// A Layout is a function from coordinates to offsets, given by a hierarchical
// [Shape] and a congruent [Stride], in the manner of CuTe.
//
// The coordinate space is colexicographic: in an integral index, the first
// mode varies fastest. An offset is in whatever unit the strides are in; a
// Layout does not know about elements or bytes. [TMASwizzleLayout] documents
// that its offsets are bytes.
//
// The zero Layout is invalid. Construct one with [NewLayout], [ParseLayout]
// or [CompactLayout], or obtain one from the algebra ([Coalesce],
// [Composition], [Complement], [LogicalDivide] and the rest).
type Layout struct {
	shape  Shape
	stride Stride
	leaves []leaf
	modes  []int // top-level mode i owns leaves[modes[i]:modes[i+1]]
	size   int
	kind   strideKind
}

type leaf struct {
	n    int // extent
	kind strideKind
	d    int
	axis int
	bits []int
}

// NewLayout returns the layout shape:stride.
//
// The stride must be congruent with the shape: the same tree, leaf for leaf.
// Extents must be positive and a tuple must be non-empty. Every leaf with a
// nonzero contribution must be of one kind; an integer stride of 0 may appear
// anywhere. An F2 stride's mode must have extent 1<<len(bits) and its bits
// must be non-negative. The layout's size and every offset it can produce must
// fit in an int.
func NewLayout(shape Shape, stride Stride) (Layout, error) {
	l := Layout{shape: shape, stride: stride, kind: strideInt}
	if err := l.build(); err != nil {
		return Layout{}, fmt.Errorf("layout %v:%v: %w", shape, stride, err)
	}
	return l, nil
}

func (l *Layout) build() error {
	if err := congruent(l.shape, l.stride); err != nil {
		return err
	}
	l.leaves = collect(l.shape, l.stride, nil)
	if l.shape.tuple {
		l.modes = make([]int, 0, len(l.shape.modes)+1)
		n := 0
		for _, m := range l.shape.modes {
			l.modes = append(l.modes, n)
			n += len(m.flatten(nil))
		}
		l.modes = append(l.modes, n)
	} else {
		l.modes = []int{0, 1}
	}
	size, span := 1, 0
	kindSet := false
	for _, f := range l.leaves {
		if f.n <= 0 {
			return fmt.Errorf("extent %d is not positive: %w", f.n, ErrUnsupported)
		}
		var err error
		if size, err = mul(size, f.n); err != nil {
			return err
		}
		if f.kind == strideInt && f.d == 0 {
			continue
		}
		if kindSet && f.kind != l.kind {
			return fmt.Errorf("stride mixes integer, basis and F2 leaves: %w", ErrUnsupported)
		}
		l.kind, kindSet = f.kind, true
		switch f.kind {
		case strideF2:
			if f.n != 1<<len(f.bits) {
				return fmt.Errorf("F2 stride with %d bits on extent %d, want extent %d: %w", len(f.bits), f.n, 1<<len(f.bits), ErrUnsupported)
			}
			for _, b := range f.bits {
				if b < 0 {
					return fmt.Errorf("F2 stride bit %d is negative: %w", b, ErrUnsupported)
				}
			}
		case strideBasis:
			if f.axis < 0 {
				return fmt.Errorf("basis axis %d is negative: %w", f.axis, ErrUnsupported)
			}
			fallthrough
		default:
			if f.d == math.MinInt {
				return ErrOverflow
			}
			reach, err := mul(abs(f.d), f.n-1)
			if err != nil {
				return err
			}
			if span, err = add(span, reach); err != nil {
				return err
			}
		}
	}
	l.size = size
	return nil
}

// congruent reports whether stride has the same tree as shape.
func congruent(shape Shape, stride Stride) error {
	if shape.tuple != stride.tuple {
		return fmt.Errorf("shape %v and stride %v are not congruent: %w", shape, stride, ErrUnsupported)
	}
	if !shape.tuple {
		return nil
	}
	if len(shape.modes) == 0 {
		return fmt.Errorf("empty tuple: %w", ErrUnsupported)
	}
	if len(shape.modes) != len(stride.modes) {
		return fmt.Errorf("shape %v has %d modes, stride %v has %d: %w", shape, len(shape.modes), stride, len(stride.modes), ErrUnsupported)
	}
	for i := range shape.modes {
		if err := congruent(shape.modes[i], stride.modes[i]); err != nil {
			return err
		}
	}
	return nil
}

func collect(shape Shape, stride Stride, dst []leaf) []leaf {
	if !shape.tuple {
		return append(dst, leaf{n: shape.n, kind: stride.kind, d: stride.d, axis: stride.axis, bits: stride.bits})
	}
	for i := range shape.modes {
		dst = collect(shape.modes[i], stride.modes[i], dst)
	}
	return dst
}

// CompactLayout returns the colexicographic compact layout of shape: the
// first mode has stride 1 and each later mode's stride is the product of the
// extents before it.
func CompactLayout(shape Shape) (Layout, error) {
	next := 1
	var overflow bool
	var build func(Shape) Stride
	build = func(s Shape) Stride {
		if !s.tuple {
			d := next
			if n, err := mul(next, max(s.n, 1)); err != nil {
				overflow = true
			} else {
				next = n
			}
			return IntStride(d)
		}
		st := Stride{tuple: true}
		for _, m := range s.modes {
			st.modes = append(st.modes, build(m))
		}
		return st
	}
	stride := build(shape)
	if overflow {
		return Layout{}, fmt.Errorf("compact layout of %v: %w", shape, ErrOverflow)
	}
	return NewLayout(shape, stride)
}

// Shape returns the layout's shape.
func (l Layout) Shape() Shape { return l.shape }

// Stride returns the layout's stride.
func (l Layout) Stride() Stride { return l.stride }

// Rank returns the number of top-level modes.
func (l Layout) Rank() int { return len(l.modes) - 1 }

// Size returns the number of coordinates in the layout's domain.
func (l Layout) Size() int { return l.size }

// String formats the layout as "shape:stride".
func (l Layout) String() string { return l.shape.String() + ":" + l.stride.String() }

// Mode returns top-level mode i as a layout of its own.
func (l Layout) Mode(i int) (Layout, error) {
	if i < 0 || i >= l.Rank() {
		return Layout{}, fmt.Errorf("mode %d of rank-%d layout %v: %w", i, l.Rank(), l, ErrOutOfBounds)
	}
	if !l.shape.tuple {
		return l, nil
	}
	return NewLayout(l.shape.modes[i], l.stride.modes[i])
}

// Cosize returns one more than the largest offset the layout produces, the
// extent of storage it addresses. It is defined for integer layouts with
// non-negative strides and for F2 layouts.
func (l Layout) Cosize() (int, error) {
	if l.size == 0 {
		return 0, ErrUninitialized
	}
	switch l.kind {
	case strideF2:
		top := f2Max(l.leaves)
		if top == math.MaxInt {
			return 0, ErrOverflow
		}
		return top + 1, nil
	case strideInt:
		top := 0
		for _, f := range l.leaves {
			if f.d < 0 {
				return 0, fmt.Errorf("cosize of %v: negative stride: %w", l, ErrUnsupported)
			}
			top += f.d * (f.n - 1) // bounded by NewLayout
		}
		if top == math.MaxInt {
			return 0, ErrOverflow
		}
		return top + 1, nil
	default:
		return 0, fmt.Errorf("cosize of %v: basis strides have no integer codomain: %w", l, ErrUnsupported)
	}
}

// Index returns the offset of integral coordinate i, 0 <= i < Size().
//
// It does not allocate. It is defined for integer and F2 layouts; a layout
// with basis strides has a coordinate-valued codomain and uses
// [Layout.BasisOffset].
func (l Layout) Index(i int) (int, error) {
	if l.size == 0 {
		return 0, ErrUninitialized
	}
	if i < 0 || i >= l.size {
		return 0, ErrOutOfBounds
	}
	if l.kind == strideBasis {
		return 0, errBasisCodomain
	}
	return evalIndex(l.leaves, i), nil
}

var errBasisCodomain = fmt.Errorf("basis strides have a coordinate-valued codomain: %w", ErrUnsupported)

// evalIndex evaluates an in-range integral coordinate.
func evalIndex(leaves []leaf, i int) int {
	off := 0
	for k := range leaves {
		f := &leaves[k]
		c := i % f.n
		i /= f.n
		off = contribute(off, f, c)
	}
	return off
}

func contribute(off int, f *leaf, c int) int {
	if f.kind == strideF2 {
		return off ^ f2Eval(f.bits, c)
	}
	return off + c*f.d
}

// f2Eval returns the XOR of bits[k] over the set bits k of c: the product of
// the F2 matrix whose columns are bits with the bit vector c.
func f2Eval(bits []int, c int) int {
	x := 0
	for k, b := range bits {
		if c>>k&1 != 0 {
			x ^= b
		}
	}
	return x
}

// Offset returns the offset of a coordinate given in one of three forms:
//
//   - one integer, an integral coordinate as for [Layout.Index];
//   - Rank() integers, one integral coordinate per top-level mode;
//   - one integer per leaf, the flattened natural coordinate.
//
// When two forms have the same length they denote the same coordinate. It
// does not allocate.
func (l Layout) Offset(crd ...int) (int, error) {
	if l.size == 0 {
		return 0, ErrUninitialized
	}
	if l.kind == strideBasis {
		return 0, errBasisCodomain
	}
	switch len(crd) {
	case 1:
		return l.Index(crd[0])
	case len(l.leaves):
		off := 0
		for k, c := range crd {
			f := &l.leaves[k]
			if c < 0 || c >= f.n {
				return 0, ErrOutOfBounds
			}
			off = contribute(off, f, c)
		}
		return off, nil
	case l.Rank():
		off := 0
		for m, c := range crd {
			mode := l.leaves[l.modes[m]:l.modes[m+1]]
			n := 1
			for k := range mode {
				n *= mode[k].n
			}
			if c < 0 || c >= n {
				return 0, ErrOutOfBounds
			}
			off = combine(l.kind, off, evalIndex(mode, c))
		}
		return off, nil
	}
	return 0, fmt.Errorf("coordinate of length %d for layout %v of rank %d with %d leaves: %w", len(crd), l, l.Rank(), len(l.leaves), ErrOutOfBounds)
}

func combine(kind strideKind, a, b int) int {
	if kind == strideF2 {
		return a ^ b
	}
	return a + b
}

// Coordinate appends to dst the flattened natural coordinate of integral
// coordinate i: the colexicographic decomposition of i over the leaf extents,
// first leaf fastest. It is the inverse of [Layout.Offset] on the domain.
func (l Layout) Coordinate(i int, dst []int) ([]int, error) {
	if l.size == 0 {
		return dst, ErrUninitialized
	}
	if i < 0 || i >= l.size {
		return dst, ErrOutOfBounds
	}
	for _, f := range l.leaves {
		dst = append(dst, i%f.n)
		i /= f.n
	}
	return dst, nil
}

// BasisOffset returns the coordinate-valued offset of integral coordinate i
// for a layout with basis strides, in dst[:n] where n is one more than the
// largest axis. dst is reused if it is long enough.
func (l Layout) BasisOffset(i int, dst []int) ([]int, error) {
	if l.size == 0 {
		return dst, ErrUninitialized
	}
	if i < 0 || i >= l.size {
		return dst, ErrOutOfBounds
	}
	n := 1
	for _, f := range l.leaves {
		if f.kind == strideBasis {
			n = max(n, f.axis+1)
		}
	}
	dst = slices.Grow(dst[:0], n)[:n]
	clear(dst)
	for _, f := range l.leaves {
		c := i % f.n
		i /= f.n
		switch f.kind {
		case strideBasis:
			dst[f.axis] += c * f.d
		case strideInt:
			dst[0] += c * f.d
		default:
			return dst, fmt.Errorf("basis offset of F2 layout %v: %w", l, ErrUnsupported)
		}
	}
	return dst, nil
}

// Equal reports whether two layouts have the same shape and stride trees.
// Two layouts can be unequal and still compute the same function.
func (l Layout) Equal(m Layout) bool { return l.String() == m.String() }

// Coalesce returns the shortest flat layout computing the same function over
// integral coordinates as l.
//
// Extent-1 modes are dropped, and adjacent modes merge when the second
// continues the first: s0:d0 followed by s1:(s0*d0) becomes (s0*s1):d0. Basis
// modes merge only along the same axis, and adjacent F2 modes always merge.
// The result is a single mode, or a flat tuple of modes.
func Coalesce(l Layout) Layout {
	return fromLeaves(coalesce(l.leaves))
}

// CoalesceByMode coalesces each top-level mode of l separately. The result has
// the same rank, so a coordinate given per top-level mode addresses the same
// offset in both.
func CoalesceByMode(l Layout) Layout {
	if l.Rank() == 1 {
		return Coalesce(l)
	}
	var shape []Shape
	var stride []Stride
	for m := 0; m < l.Rank(); m++ {
		s, d := leafTree(coalesce(l.leaves[l.modes[m]:l.modes[m+1]]))
		shape, stride = append(shape, s), append(stride, d)
	}
	return mustLayout(TupleShape(shape...), TupleStride(stride...))
}

func coalesce(in []leaf) []leaf {
	var out []leaf
	for _, f := range in {
		if f.n == 1 {
			continue
		}
		if len(out) > 0 {
			last := &out[len(out)-1]
			switch {
			case last.kind == strideF2 && f.kind == strideF2:
				last.n *= f.n
				last.bits = append(slices.Clip(last.bits), f.bits...)
				continue
			case last.kind == f.kind && f.kind != strideF2 && last.axis == f.axis && last.n*last.d == f.d:
				last.n *= f.n
				continue
			}
		}
		out = append(out, f)
	}
	if len(out) == 0 {
		out = append(out, leaf{n: 1})
	}
	return out
}

// fromLeaves returns the flat layout of leaves: a single mode, or a tuple.
func fromLeaves(leaves []leaf) Layout {
	s, d := leafTree(leaves)
	return mustLayout(s, d)
}

func leafTree(leaves []leaf) (Shape, Stride) {
	if len(leaves) == 1 {
		return leafShape(leaves[0]), leafStride(leaves[0])
	}
	shape, stride := Shape{tuple: true}, Stride{tuple: true}
	for _, f := range leaves {
		shape.modes = append(shape.modes, leafShape(f))
		stride.modes = append(stride.modes, leafStride(f))
	}
	return shape, stride
}

func leafShape(f leaf) Shape { return IntShape(f.n) }

func leafStride(f leaf) Stride {
	return Stride{kind: f.kind, d: f.d, axis: f.axis, bits: f.bits}
}

// mustLayout builds a layout the algebra derived from a valid one. The
// algebra only shrinks or permutes validated extents and offsets, so an error
// here is a bug in this package.
func mustLayout(shape Shape, stride Stride) Layout {
	l, err := NewLayout(shape, stride)
	if err != nil {
		panic("cudalayout: internal error: " + err.Error())
	}
	return l
}

// Composition returns the layout a∘b, whose offset at coordinate i is
// a's offset at b's offset at i. The result has b's shape, refined where a's
// modes split it.
//
// b must be an integer layout with non-negative strides. b's offsets are
// integral coordinates of a; offsets past a.Size() continue a's last mode,
// as in CuTe. For an integer or basis a the
// composition exists only under CuTe's divisibility conditions: walking a's
// coalesced modes, each of b's strides must divide or be divided by the
// extent it meets (the stride divisibility condition), and the extent split
// off must divide what remains of b's mode (the shape divisibility
// condition). b's modes must also not carry into one another when their
// offsets add in a's coordinates (the additivity condition), since the
// result evaluates them separately. A violated condition is an error, not an
// approximation.
//
// For an F2 a, b must have power-of-two extents and offsets whose bit images
// are disjoint, so that b is itself F2-linear, and b's offsets must stay
// inside a's domain, and a's extents must be powers of two, so that a is
// F2-linear in its integral coordinate. The result is then an F2 layout.
func Composition(a, b Layout) (Layout, error) {
	if a.size == 0 || b.size == 0 {
		return Layout{}, ErrUninitialized
	}
	if err := checkIntLayout(b, "right operand of composition"); err != nil {
		return Layout{}, err
	}
	var s Shape
	var d Stride
	var err error
	if a.kind == strideF2 {
		for _, f := range a.leaves {
			if f.n&(f.n-1) != 0 {
				return Layout{}, fmt.Errorf("compose %v with %v: F2 layout has extent %d, not a power of two: %w", a, b, f.n, ErrUnsupported)
			}
		}
		if err = checkF2Exact(b.leaves); err != nil {
			return Layout{}, fmt.Errorf("compose %v with %v: %w", a, b, err)
		}
		s, d, err = composeF2Tree(a, b.shape, b.stride)
	} else {
		ca := coalesce(a.leaves)
		s, d, err = composeTree(ca, b.shape, b.stride, make([]int, len(ca)))
	}
	if err != nil {
		return Layout{}, fmt.Errorf("compose %v with %v: %w", a, b, err)
	}
	return NewLayout(s, d)
}

func checkIntLayout(l Layout, what string) error {
	for _, f := range l.leaves {
		if f.kind != strideInt {
			return fmt.Errorf("%s %v must have integer strides: %w", what, l, ErrUnsupported)
		}
		if f.d < 0 {
			return fmt.Errorf("%s %v has a negative stride: %w", what, l, ErrUnsupported)
		}
	}
	return nil
}

// composeTree composes the coalesced leaves a with each leaf of b's tree,
// keeping b's tree. used[j] accumulates the largest digit b's leaves place in
// a's mode j; see compose1.
func composeTree(a []leaf, bs Shape, bd Stride, used []int) (Shape, Stride, error) {
	if bs.tuple {
		s, d := Shape{tuple: true}, Stride{tuple: true}
		for i := range bs.modes {
			ms, md, err := composeTree(a, bs.modes[i], bd.modes[i], used)
			if err != nil {
				return Shape{}, Stride{}, err
			}
			s.modes, d.modes = append(s.modes, ms), append(d.modes, md)
		}
		return s, d, nil
	}
	out, err := compose1(a, bs.n, bd.d, used)
	if err != nil {
		return Shape{}, Stride{}, err
	}
	s, d := leafTree(out)
	return s, d, nil
}

// compose1 composes coalesced leaves a with the single mode shape:stride.
// It follows CuTe's composition_impl (include/cute/layout.hpp) and pycute's
// composition (python/pycute/layout.py), and checks both conditions at run
// time: pycute's stride condition and CuTe's shape condition. Each alone
// admits a wrong answer the other rejects.
//
// It checks a third condition neither checks. The result evaluates each of
// b's leaves separately and adds, which equals a at the sum only when
// adding the leaves' offsets carries no digit of a's mixed radix. compose1
// adds the largest digit this leaf places in each mode j of a to used[j], and
// rejects a total that reaches a's extent there. The last mode of a is
// unbounded and never carries.
func compose1(a []leaf, shape, stride int, used []int) ([]leaf, error) {
	if stride == 0 {
		return []leaf{{n: shape}}, nil
	}
	var out []leaf
	restShape, restStride := shape, stride
	last := a[len(a)-1]
	for j, cur := range a[:len(a)-1] {
		if cur.n%restStride != 0 && restStride%cur.n != 0 {
			return nil, fmt.Errorf("stride divisibility: stride %d against extent %d: %w", restStride, cur.n, ErrUnsupported)
		}
		nextShape := ceilDiv(cur.n, restStride)
		nextStride := ceilDiv(restStride, cur.n)
		if nextShape == 1 || restShape == 1 {
			restStride = nextStride
			continue
		}
		newShape := min(nextShape, restShape)
		if restShape%newShape != 0 {
			return nil, fmt.Errorf("shape divisibility: extent %d does not divide %d: %w", newShape, restShape, ErrUnsupported)
		}
		digit := (newShape - 1) * restStride
		if digit >= cur.n-used[j] {
			return nil, fmt.Errorf("additivity: modes of the right operand together reach digit %d of extent %d, so their offsets carry: %w", used[j]+digit, cur.n, ErrUnsupported)
		}
		used[j] += digit
		d, err := scale(cur, restStride)
		if err != nil {
			return nil, err
		}
		out = append(out, d.with(newShape))
		restShape /= newShape
		restStride = nextStride
	}
	if restShape != 1 || len(out) == 0 {
		d, err := scale(last, restStride)
		if err != nil {
			return nil, err
		}
		out = append(out, d.with(restShape))
	}
	return out, nil
}

func (f leaf) with(n int) leaf { f.n = n; return f }

// scale returns f with its stride multiplied by k.
func scale(f leaf, k int) (leaf, error) {
	d, err := mul(f.d, k)
	if err != nil {
		return leaf{}, err
	}
	f.d = d
	return f, nil
}

// composeF2Tree composes an F2 layout a with each leaf of b's tree. b is
// known to be F2-exact, so a(b(c)) is the XOR of a over b's bit images.
func composeF2Tree(a Layout, bs Shape, bd Stride) (Shape, Stride, error) {
	if bs.tuple {
		s, d := Shape{tuple: true}, Stride{tuple: true}
		for i := range bs.modes {
			ms, md, err := composeF2Tree(a, bs.modes[i], bd.modes[i])
			if err != nil {
				return Shape{}, Stride{}, err
			}
			s.modes, d.modes = append(s.modes, ms), append(d.modes, md)
		}
		return s, d, nil
	}
	if bd.d == 0 {
		return bs, IntStride(0), nil
	}
	var img []int
	for k := 0; 1<<k < bs.n; k++ {
		j := bd.d << k
		if j >= a.size {
			return Shape{}, Stride{}, fmt.Errorf("offset %d is outside the F2 domain of size %d: %w", j, a.size, ErrUnsupported)
		}
		img = append(img, evalIndex(a.leaves, j))
	}
	return bs, F2Stride(img...), nil
}

// checkF2Exact reports whether integer leaves compute an F2-linear function:
// every mode that contributes has a power-of-two extent, and the offsets of
// all single coordinate bits are pairwise disjoint, so no sum carries.
func checkF2Exact(leaves []leaf) error {
	seen := 0
	for _, f := range leaves {
		if f.d == 0 || f.n == 1 {
			continue
		}
		if f.kind != strideInt || f.d < 0 {
			return fmt.Errorf("stride %d is not F2-linear: %w", f.d, ErrUnsupported)
		}
		if f.n&(f.n-1) != 0 {
			return fmt.Errorf("extent %d is not a power of two: %w", f.n, ErrUnsupported)
		}
		for k := 0; 1<<k < f.n; k++ {
			img := f.d << k
			if img>>k != f.d {
				return ErrOverflow
			}
			if seen&img != 0 {
				return fmt.Errorf("bit image %#x overlaps %#x, so offsets carry: %w", img, seen&img, ErrUnsupported)
			}
			seen |= img
		}
	}
	return nil
}

// ComposeByMode composes each of a's first len(tiler) top-level modes with
// the corresponding tiler layout, and keeps a's remaining modes. It is
// CuTe's composition with a tuple of layouts as the tiler.
func ComposeByMode(a Layout, tiler []Layout) (Layout, error) {
	return byMode(a, tiler, Composition)
}

// ShapeTiler returns the by-mode tiler for a shape of integer extents: the
// layout n:1 for each extent n. This is how CuTe promotes a shape used as a
// tiler.
func ShapeTiler(extents ...int) ([]Layout, error) {
	var t []Layout
	for _, n := range extents {
		l, err := NewLayout(IntShape(n), IntStride(1))
		if err != nil {
			return nil, fmt.Errorf("shape tiler: %w", err)
		}
		t = append(t, l)
	}
	return t, nil
}

func byMode(a Layout, tiler []Layout, op func(a, b Layout) (Layout, error)) (Layout, error) {
	if a.size == 0 {
		return Layout{}, ErrUninitialized
	}
	if len(tiler) == 0 || len(tiler) > a.Rank() {
		return Layout{}, fmt.Errorf("tiler of %d modes for rank-%d layout %v: %w", len(tiler), a.Rank(), a, ErrUnsupported)
	}
	var shape []Shape
	var stride []Stride
	for m := 0; m < a.Rank(); m++ {
		mode, err := a.Mode(m)
		if err != nil {
			return Layout{}, err
		}
		if m < len(tiler) {
			if mode, err = op(mode, tiler[m]); err != nil {
				return Layout{}, fmt.Errorf("mode %d: %w", m, err)
			}
		}
		shape, stride = append(shape, mode.shape), append(stride, mode.stride)
	}
	return NewLayout(TupleShape(shape...), TupleStride(stride...))
}

// Complement returns the minimal layout r, ordered by increasing offset,
// such that the concatenation (l', r) is injective and its size covers
// cotarget, where l' is l without its stride-0 modes. r addresses exactly
// the offsets l skips, then repeats l's footprint until the product of the
// two sizes is at least cotarget. As in CuTe, a stride-0 mode of l
// contributes nothing to its image and nothing to r.
//
// l must be an integer layout with non-negative strides. Sorted by stride,
// each mode's stride must be a multiple of the footprint of the modes below
// it; otherwise no such r exists (l overlaps itself, or leaves holes that no
// ordered layout can fill), and Complement returns an error.
func Complement(l Layout, cotarget int) (Layout, error) {
	if l.size == 0 {
		return Layout{}, ErrUninitialized
	}
	if cotarget < 1 {
		return Layout{}, fmt.Errorf("complement of %v: cotarget %d is not positive: %w", l, cotarget, ErrUnsupported)
	}
	if err := checkIntLayout(l, "complement operand"); err != nil {
		return Layout{}, err
	}
	var modes []leaf
	for _, f := range l.leaves {
		if f.n != 1 && f.d != 0 {
			modes = append(modes, f)
		}
	}
	slices.SortStableFunc(modes, func(x, y leaf) int {
		if x.d != y.d {
			return x.d - y.d
		}
		return x.n - y.n
	})
	var out []leaf
	cur := 1
	for _, f := range modes {
		if f.d%cur != 0 {
			return Layout{}, fmt.Errorf("complement of %v: stride %d is not a multiple of the covered extent %d: %w", l, f.d, cur, ErrUnsupported)
		}
		out = append(out, leaf{n: f.d / cur, d: cur})
		var err error
		if cur, err = mul(f.d, f.n); err != nil {
			return Layout{}, fmt.Errorf("complement of %v: %w", l, err)
		}
	}
	out = append(out, leaf{n: ceilDiv(cotarget, cur), d: cur})
	c, err := NewLayout(leafTree(coalesce(out)))
	if err != nil {
		return Layout{}, fmt.Errorf("complement of %v with cotarget %d: %w", l, cotarget, err)
	}
	return c, nil
}

// LogicalDivide returns a∘(b, Complement(b, a.Size())): a rank-2 layout
// whose first mode is the tile b selects from a and whose second mode
// iterates over the tiles. b is subject to the conditions of [Complement]
// and [Composition].
func LogicalDivide(a, b Layout) (Layout, error) {
	if a.size == 0 || b.size == 0 {
		return Layout{}, ErrUninitialized
	}
	c, err := Complement(b, a.size)
	if err != nil {
		return Layout{}, fmt.Errorf("logical divide %v by %v: %w", a, b, err)
	}
	t, err := NewLayout(TupleShape(b.shape, c.shape), TupleStride(b.stride, c.stride))
	if err != nil {
		return Layout{}, fmt.Errorf("logical divide %v by %v: %w", a, b, err)
	}
	r, err := Composition(a, t)
	if err != nil {
		return Layout{}, fmt.Errorf("logical divide %v by %v: %w", a, b, err)
	}
	return r, nil
}

// LogicalDivideByMode divides each of a's first len(tiler) top-level modes by
// the corresponding tiler layout, and keeps a's remaining modes.
func LogicalDivideByMode(a Layout, tiler []Layout) (Layout, error) {
	return byMode(a, tiler, LogicalDivide)
}

// ZippedDivide divides a by mode, as [LogicalDivideByMode] does, and gathers
// the result into two modes: the tiles, (tile0, tile1, ...), and the rest,
// (rest0, rest1, ..., the undivided modes of a). Indexing the first mode
// walks within one tile; indexing the second picks the tile, as a CTA or warp
// index would.
func ZippedDivide(a Layout, tiler []Layout) (Layout, error) {
	d, err := LogicalDivideByMode(a, tiler)
	if err != nil {
		return Layout{}, err
	}
	tile, rest := Stride{tuple: true}, Stride{tuple: true}
	tileS, restS := Shape{tuple: true}, Shape{tuple: true}
	for m := 0; m < a.Rank(); m++ {
		s, st := d.shape.modes[m], d.stride.modes[m]
		if m >= len(tiler) {
			restS.modes, rest.modes = append(restS.modes, s), append(rest.modes, st)
			continue
		}
		tileS.modes, tile.modes = append(tileS.modes, s.modes[0]), append(tile.modes, st.modes[0])
		restS.modes, rest.modes = append(restS.modes, s.modes[1]), append(rest.modes, st.modes[1])
	}
	return NewLayout(TupleShape(tileS, restS), TupleStride(tile, rest))
}

// ceilDiv returns a/b rounded up, for a >= 0 and b > 0, without overflow.
func ceilDiv(a, b int) int {
	q := a / b
	if a%b != 0 {
		q++
	}
	return q
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func mul(a, b int) (int, error) {
	hi, lo := bits.Mul64(uint64(abs(a)), uint64(abs(b)))
	if hi != 0 || lo > math.MaxInt {
		return 0, ErrOverflow
	}
	return a * b, nil
}

func add(a, b int) (int, error) {
	s := a + b
	if (s > a) != (b > 0) {
		return 0, ErrOverflow
	}
	return s, nil
}
