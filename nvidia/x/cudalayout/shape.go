package cudalayout

import (
	"fmt"
	"strconv"
	"strings"
)

// A Shape is a hierarchical extent: an integer, or a tuple of Shapes.
//
// A Shape is only a description. Its extents are checked when it is paired
// with a [Stride] by [NewLayout], so a Shape holding a non-positive extent can
// be built and printed but not used. The zero Shape is the integer 0.
type Shape struct {
	n     int
	modes []Shape
	tuple bool
}

// IntShape returns the integer shape n.
func IntShape(n int) Shape { return Shape{n: n} }

// TupleShape returns the tuple of the given modes. The modes are copied.
func TupleShape(modes ...Shape) Shape {
	return Shape{modes: append([]Shape(nil), modes...), tuple: true}
}

// Rank returns the number of top-level modes: 1 for an integer.
func (s Shape) Rank() int {
	if !s.tuple {
		return 1
	}
	return len(s.modes)
}

// String formats the shape in CuTe notation, such as "((4,2),8)".
func (s Shape) String() string {
	var b strings.Builder
	s.format(&b)
	return b.String()
}

func (s Shape) format(b *strings.Builder) {
	if !s.tuple {
		b.WriteString(strconv.Itoa(s.n))
		return
	}
	b.WriteByte('(')
	for i, m := range s.modes {
		if i > 0 {
			b.WriteByte(',')
		}
		m.format(b)
	}
	b.WriteByte(')')
}

// flatten appends the integer leaves of s in colexicographic order.
func (s Shape) flatten(dst []int) []int {
	if !s.tuple {
		return append(dst, s.n)
	}
	for _, m := range s.modes {
		dst = m.flatten(dst)
	}
	return dst
}

type strideKind uint8

const (
	strideInt   strideKind = iota // offset += c*d
	strideBasis                   // offset[axis] += c*d
	strideF2                      // offset ^= bits[k] for each set bit k of c
)

// A Stride is a hierarchical stride: a leaf, or a tuple of Strides.
//
// A leaf is one of three kinds, and the kind decides how a coordinate in that
// mode contributes to an offset:
//
//   - [IntStride] d adds c*d, as in an ordinary strided layout.
//   - [BasisStride] d@axis adds c*d to component axis of a coordinate-valued
//     offset, as CuTe's scaled basis strides E<axis> do.
//   - [F2Stride] b0, b1, ... XORs b_k into the offset for each set bit k of c.
//     This is carry-less arithmetic over F2, the form XOR swizzles take.
//
// The zero Stride is the integer stride 0.
type Stride struct {
	kind  strideKind
	d     int
	axis  int
	bits  []int
	modes []Stride
	tuple bool
}

// IntStride returns the integer stride d.
func IntStride(d int) Stride { return Stride{d: d} }

// BasisStride returns the stride d along coordinate axis, written d@axis.
func BasisStride(axis, d int) Stride { return Stride{kind: strideBasis, d: d, axis: axis} }

// F2Stride returns an F2-linear stride for a mode of extent 1<<len(bits).
// Coordinate bit k contributes bits[k], and contributions combine by XOR.
// The bits are copied.
func F2Stride(bits ...int) Stride {
	return Stride{kind: strideF2, bits: append([]int{}, bits...)}
}

// TupleStride returns the tuple of the given modes. The modes are copied.
func TupleStride(modes ...Stride) Stride {
	return Stride{modes: append([]Stride(nil), modes...), tuple: true}
}

// Rank returns the number of top-level modes: 1 for a leaf.
func (d Stride) Rank() int {
	if !d.tuple {
		return 1
	}
	return len(d.modes)
}

// String formats the stride in CuTe notation. A basis stride prints as
// "d@axis" and an F2 stride as "xor(b0,b1,...)".
func (d Stride) String() string {
	var b strings.Builder
	d.format(&b)
	return b.String()
}

func (d Stride) format(b *strings.Builder) {
	if d.tuple {
		b.WriteByte('(')
		for i, m := range d.modes {
			if i > 0 {
				b.WriteByte(',')
			}
			m.format(b)
		}
		b.WriteByte(')')
		return
	}
	switch d.kind {
	case strideBasis:
		fmt.Fprintf(b, "%d@%d", d.d, d.axis)
	case strideF2:
		b.WriteString("xor(")
		for i, v := range d.bits {
			if i > 0 {
				b.WriteByte(',')
			}
			b.WriteString(strconv.Itoa(v))
		}
		b.WriteByte(')')
	default:
		b.WriteString(strconv.Itoa(d.d))
	}
}

// ParseShape parses a shape in CuTe notation, such as "((4,2),8)".
func ParseShape(s string) (Shape, error) {
	p := parser{s: s}
	sh, err := p.shape()
	if err == nil {
		err = p.end()
	}
	if err != nil {
		return Shape{}, fmt.Errorf("parse shape %q: %w", s, err)
	}
	return sh, nil
}

// ParseStride parses a stride in CuTe notation, such as "((2,16),1)". A leaf
// written "d@axis" is a [BasisStride]. F2 strides have no written form.
func ParseStride(s string) (Stride, error) {
	p := parser{s: s}
	st, err := p.stride()
	if err == nil {
		err = p.end()
	}
	if err != nil {
		return Stride{}, fmt.Errorf("parse stride %q: %w", s, err)
	}
	return st, nil
}

// ParseLayout parses "shape:stride" in CuTe notation, such as
// "((4,2),8):((2,16),1)", and checks it with [NewLayout].
func ParseLayout(s string) (Layout, error) {
	shape, stride, ok := strings.Cut(s, ":")
	if !ok {
		return Layout{}, fmt.Errorf("parse layout %q: missing ':': %w", s, ErrUnsupported)
	}
	sh, err := ParseShape(shape)
	if err != nil {
		return Layout{}, err
	}
	st, err := ParseStride(stride)
	if err != nil {
		return Layout{}, err
	}
	return NewLayout(sh, st)
}

type parser struct {
	s   string
	pos int
}

func (p *parser) skip() {
	for p.pos < len(p.s) && (p.s[p.pos] == ' ' || p.s[p.pos] == '\t') {
		p.pos++
	}
}

func (p *parser) peek() byte {
	p.skip()
	if p.pos < len(p.s) {
		return p.s[p.pos]
	}
	return 0
}

func (p *parser) end() error {
	if p.peek() != 0 {
		return fmt.Errorf("unexpected %q at offset %d: %w", p.s[p.pos:], p.pos, ErrUnsupported)
	}
	return nil
}

func (p *parser) int() (int, error) {
	p.skip()
	start := p.pos
	if p.pos < len(p.s) && (p.s[p.pos] == '-' || p.s[p.pos] == '+') {
		p.pos++
	}
	for p.pos < len(p.s) && '0' <= p.s[p.pos] && p.s[p.pos] <= '9' {
		p.pos++
	}
	n, err := strconv.Atoi(p.s[start:p.pos])
	if err != nil {
		return 0, fmt.Errorf("integer at offset %d: %w", start, ErrUnsupported)
	}
	return n, nil
}

// list parses "(" elem ("," elem)* ")", calling elem for each element.
func (p *parser) list(elem func() error) error {
	p.pos++ // '('
	for {
		if err := elem(); err != nil {
			return err
		}
		switch p.peek() {
		case ',':
			p.pos++
		case ')':
			p.pos++
			return nil
		default:
			return fmt.Errorf("want ',' or ')' at offset %d: %w", p.pos, ErrUnsupported)
		}
	}
}

func (p *parser) shape() (Shape, error) {
	if p.peek() != '(' {
		n, err := p.int()
		return IntShape(n), err
	}
	s := Shape{tuple: true}
	err := p.list(func() error {
		m, err := p.shape()
		s.modes = append(s.modes, m)
		return err
	})
	return s, err
}

func (p *parser) stride() (Stride, error) {
	if p.peek() != '(' {
		d, err := p.int()
		if err != nil {
			return Stride{}, err
		}
		if p.peek() != '@' {
			return IntStride(d), nil
		}
		p.pos++
		axis, err := p.int()
		return BasisStride(axis, d), err
	}
	s := Stride{tuple: true}
	err := p.list(func() error {
		m, err := p.stride()
		s.modes = append(s.modes, m)
		return err
	})
	return s, err
}
