package cudalayout

import (
	"fmt"
	"strconv"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
)

// An XORSwizzle is CuTe's Swizzle<B,M,S>: the F2-linear map that XORs the B
// bits of an offset starting at bit M+S into the B bits starting at bit M.
//
//	offset ^ ((offset >> S) & ((1<<B - 1) << M))
//
// A negative S shifts left instead, moving low bits into high ones. Because
// the map is linear over F2, composing it with an F2 layout, or with an
// integer layout whose offsets never carry, gives another F2 layout; see
// [XORSwizzle.Compose].
//
// It is not [Swizzle], this package's generic row-column permutation, and
// it is not [TMASwizzle], a tensor-map mode; [TMASwizzleXOR] gives the
// XORSwizzle a tensor-map mode applies to byte offsets.
//
// The zero XORSwizzle is Swizzle<0,0,0>, the identity.
type XORSwizzle struct {
	bits, base, shift int
}

// NewXORSwizzle returns Swizzle<bits,base,shift>. bits and base must be
// non-negative, and |shift| must be at least bits so that the source and
// destination bit ranges do not overlap, which keeps the map invertible.
// The two bit ranges must fit below bit 62 and the host int sign bit.
func NewXORSwizzle(bits, base, shift int) (XORSwizzle, error) {
	if bits < 0 || base < 0 {
		return XORSwizzle{}, fmt.Errorf("swizzle<%d,%d,%d>: negative bit count or base: %w", bits, base, shift, ErrUnsupported)
	}
	if -bits < shift && shift < bits {
		return XORSwizzle{}, fmt.Errorf("swizzle<%d,%d,%d>: |shift| < bits overlaps source and destination: %w", bits, base, shift, ErrUnsupported)
	}
	// Bound each term before negation or addition, then subtract from the
	// available width so malformed inputs cannot wrap the range check.
	width := min(62, strconv.IntSize-1)
	if bits > width || base > width || shift < -width || shift > width {
		return XORSwizzle{}, fmt.Errorf("swizzle<%d,%d,%d>: %w", bits, base, shift, ErrOverflow)
	}
	if base > width-bits || abs(shift) > width-bits-base {
		return XORSwizzle{}, fmt.Errorf("swizzle<%d,%d,%d>: %w", bits, base, shift, ErrOverflow)
	}
	return XORSwizzle{bits: bits, base: base, shift: shift}, nil
}

// TMASwizzleXOR returns the XORSwizzle a tensor-map mode applies to byte
// offsets within a tile: Swizzle<1,4,3>, <2,4,3> and <3,4,3> for 32, 64 and
// 128 bytes, and the identity for [TMASwizzleNone] or an unknown mode.
func TMASwizzleXOR(mode TMASwizzle) XORSwizzle {
	switch mode {
	case TMASwizzle32B:
		return XORSwizzle{bits: 1, base: 4, shift: 3}
	case TMASwizzle64B:
		return XORSwizzle{bits: 2, base: 4, shift: 3}
	case TMASwizzle128B:
		return XORSwizzle{bits: 3, base: 4, shift: 3}
	}
	return XORSwizzle{}
}

// String formats s as "Swizzle<B,M,S>".
func (s XORSwizzle) String() string {
	return fmt.Sprintf("Swizzle<%d,%d,%d>", s.bits, s.base, s.shift)
}

// Apply returns the swizzled offset. It is an involution: Apply(Apply(x)) == x.
func (s XORSwizzle) Apply(offset int) int {
	mask := (1<<s.bits - 1) << (s.base + max(s.shift, 0))
	if s.shift >= 0 {
		return offset ^ (offset&mask)>>s.shift
	}
	return offset ^ (offset&mask)<<-s.shift
}

// Compose returns s∘l, the layout whose offset at each coordinate is
// s.Apply of l's offset, as an F2 layout of l's shape.
//
// l must be an F2 layout, or an integer layout whose offsets never carry:
// every mode that contributes has a power-of-two extent, its stride is
// non-negative, and the offsets of distinct coordinate bits share no set
// bit. Only then is l's offset the XOR of its bit images, which s maps one by
// one.
func (s XORSwizzle) Compose(l Layout) (Layout, error) {
	if l.size == 0 {
		return Layout{}, ErrUninitialized
	}
	switch l.kind {
	case strideF2:
	case strideInt:
		if err := checkF2Exact(l.leaves); err != nil {
			return Layout{}, fmt.Errorf("compose %v with %v: %w", s, l, err)
		}
	default:
		return Layout{}, fmt.Errorf("compose %v with %v: basis strides: %w", s, l, ErrUnsupported)
	}
	st, err := s.stride(l.shape, l.stride)
	if err != nil {
		return Layout{}, fmt.Errorf("compose %v with %v: %w", s, l, err)
	}
	return NewLayout(l.shape, st)
}

func (s XORSwizzle) stride(sh Shape, st Stride) (Stride, error) {
	if st.tuple {
		out := Stride{tuple: true}
		for i := range st.modes {
			m, err := s.stride(sh.modes[i], st.modes[i])
			if err != nil {
				return Stride{}, err
			}
			out.modes = append(out.modes, m)
		}
		return out, nil
	}
	if st.kind == strideF2 {
		img := make([]int, len(st.bits))
		for k, b := range st.bits {
			img[k] = s.Apply(b)
		}
		return F2Stride(img...), nil
	}
	if st.d == 0 {
		return st, nil
	}
	var img []int
	for k := 0; 1<<k < sh.n; k++ {
		img = append(img, s.Apply(st.d<<k))
	}
	return F2Stride(img...), nil
}

// TMASwizzleLayout returns the byte-offset layout of a rows x cols tile of
// elementBytes-byte elements under a tensor-map swizzle mode, as an F2
// layout. Its offsets are bytes, and Offset(row, col) equals the
// [SwizzledTile.ByteOffset] of the same tile.
//
// Under a swizzle the layout is (rows,(spanCols,groups)):(span,(e,rows*span))
// before the XOR, where span is the mode's byte span, e is elementBytes,
// spanCols = span/e and groups = cols/spanCols: columns are split into spans
// and each span of every row is stored before the next span begins. Without a
// swizzle it is the row-major (rows,cols):(cols*e,e).
//
// The tile must be one [NewSwizzledTile] accepts, and rows, spanCols and
// groups (cols without a swizzle) must be powers of two, since only then is
// the layout F2-linear.
func TMASwizzleLayout(rows, cols, elementBytes int, mode TMASwizzle) (Layout, error) {
	if _, err := NewSwizzledTile(rows, cols, elementBytes, mode); err != nil {
		return Layout{}, err
	}
	var base Layout
	var err error
	if span := mode.Bytes(); span == 0 {
		base, err = NewLayout(
			TupleShape(IntShape(rows), IntShape(cols)),
			TupleStride(IntStride(cols*elementBytes), IntStride(elementBytes)))
	} else {
		spanCols := span / elementBytes
		base, err = NewLayout(
			TupleShape(IntShape(rows), TupleShape(IntShape(spanCols), IntShape(cols/spanCols))),
			TupleStride(IntStride(span), TupleStride(IntStride(elementBytes), IntStride(rows*span))))
	}
	if err != nil {
		return Layout{}, fmt.Errorf("tma swizzle layout: %w", err)
	}
	l, err := TMASwizzleXOR(mode).Compose(base)
	if err != nil {
		return Layout{}, fmt.Errorf("tma swizzle layout %dx%d of %d-byte elements under %s: %w", rows, cols, elementBytes, mode, err)
	}
	return l, nil
}

// CheckDevice reports whether s has been measured on target: whether a
// device addresses shared memory as s says, and whether the bank-conflict
// counts this package computes from s's offsets are the ones the hardware
// shows.
//
// No instrument in this package has made that measurement, so for the
// targets this package knows ([TargetSM100a], [TargetSM121a]) the answer is
// [cudaerrors.ErrHardwarePending]: unknown, not refused. An unknown target
// is a caller error, [ErrUnsupported].
func (s XORSwizzle) CheckDevice(target string) error {
	switch target {
	case TargetSM100a, TargetSM121a:
		return fmt.Errorf("cudalayout: %v on %s: no instrument has measured shared-memory addressing or bank conflicts: %w", s, target, cudaerrors.ErrHardwarePending)
	}
	return fmt.Errorf("cudalayout: unknown target %q: %w", target, ErrUnsupported)
}

// f2Max returns the largest offset an F2 layout reaches: the XOR span of its
// images. Over F2 the reachable offsets form a subspace, and the largest
// element of a subspace is found by reducing a basis to echelon form and
// XORing in each pivot that raises the running value.
func f2Max(leaves []leaf) int {
	var basis []int
	for _, f := range leaves {
		if f.kind != strideF2 {
			continue
		}
		for _, v := range f.bits {
			for _, b := range basis {
				v = min(v, v^b)
			}
			if v != 0 {
				basis = append(basis, v)
				// Keep the basis sorted high to low so reduction is greedy.
				for i := len(basis) - 1; i > 0 && basis[i] > basis[i-1]; i-- {
					basis[i], basis[i-1] = basis[i-1], basis[i]
				}
			}
		}
	}
	top := 0
	for _, b := range basis {
		top = max(top, top^b)
	}
	return top
}
