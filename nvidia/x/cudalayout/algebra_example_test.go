package cudalayout_test

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudalayout"
)

func ExampleParseLayout() {
	l, err := cudalayout.ParseLayout("((2,2),2):((4,1),2)")
	if err != nil {
		panic(err)
	}
	for i := range l.Size() {
		o, _ := l.Index(i)
		fmt.Print(o, " ")
	}
	fmt.Println()
	// Output: 0 4 1 5 2 6 3 7
}

func ExampleNewLayout() {
	l, err := cudalayout.NewLayout(
		cudalayout.TupleShape(cudalayout.IntShape(4), cudalayout.IntShape(8)),
		cudalayout.TupleStride(cudalayout.IntStride(8), cudalayout.IntStride(1)))
	fmt.Println(l, l.Size(), err)

	_, err = cudalayout.NewLayout(
		cudalayout.TupleShape(cudalayout.IntShape(4), cudalayout.IntShape(8)),
		cudalayout.IntStride(1))
	fmt.Println(errors.Is(err, cudalayout.ErrUnsupported))
	// Output:
	// (4,8):(8,1) 32 <nil>
	// true
}

func ExampleCompactLayout() {
	shape, _ := cudalayout.ParseShape("(4,(2,3))")
	l, _ := cudalayout.CompactLayout(shape)
	fmt.Println(l)
	// Output: (4,(2,3)):(1,(4,8))
}

// Offset takes an integral coordinate, one integer per top-level mode, or
// one per leaf. All three name the same element here.
func ExampleLayout_Offset() {
	l, _ := cudalayout.ParseLayout("((2,2),2):((4,1),2)")
	a, _ := l.Offset(7)
	b, _ := l.Offset(3, 1)
	c, _ := l.Offset(1, 1, 1)
	fmt.Println(a, b, c)
	// Output: 7 7 7
}

func ExampleLayout_Coordinate() {
	l, _ := cudalayout.ParseLayout("((2,2),2):((4,1),2)")
	crd, _ := l.Coordinate(5, nil)
	fmt.Println(crd)
	// Output: [1 0 1]
}

func ExampleLayout_Cosize() {
	l, _ := cudalayout.ParseLayout("(4,2):(1,8)")
	n, _ := l.Cosize()
	fmt.Println(l.Size(), n)
	// Output: 8 12
}

// A layout with basis strides maps into coordinates: this one is the
// identity over a 4x6 index space, tiled as (4,(2,3)).
func ExampleLayout_BasisOffset() {
	l, _ := cudalayout.ParseLayout("(4,(2,3)):(1@0,(1@1,2@1))")
	var buf []int
	for _, i := range []int{0, 5, 23} {
		buf, _ = l.BasisOffset(i, buf)
		fmt.Println(buf)
	}
	// Output:
	// [0 0]
	// [1 1]
	// [3 5]
}

func ExampleLayout_Mode() {
	l, _ := cudalayout.ParseLayout("(4,(2,3)):(1,(4,8))")
	m, _ := l.Mode(1)
	fmt.Println(l.Rank(), m)
	// Output: 2 (2,3):(4,8)
}

func ExampleCoalesce() {
	l, _ := cudalayout.ParseLayout("((2,2),(2,3)):((1,2),(8,16))")
	fmt.Println(cudalayout.Coalesce(l))
	fmt.Println(cudalayout.CoalesceByMode(l))
	// Output:
	// (4,6):(1,8)
	// (4,6):(1,8)
}

func ExampleCoalesceByMode() {
	l, _ := cudalayout.ParseLayout("(2,4):(1,2)")
	fmt.Println(cudalayout.Coalesce(l), cudalayout.CoalesceByMode(l))
	// Output: 8:1 (2,4):(1,2)
}

func ExampleComposition() {
	a, _ := cudalayout.ParseLayout("(6,2):(8,2)")
	b, _ := cudalayout.ParseLayout("(4,3):(3,1)")
	r, err := cudalayout.Composition(a, b)
	fmt.Println(r, err)

	// Stride 4 meets extent 6, which neither divides: rejected rather than
	// approximated.
	a, _ = cudalayout.ParseLayout("(6,2):(1,7)")
	b, _ = cudalayout.ParseLayout("4:4")
	_, err = cudalayout.Composition(a, b)
	fmt.Println(errors.Is(err, cudalayout.ErrUnsupported))
	// Output:
	// ((2,2),3):((24,2),8) <nil>
	// true
}

func ExampleComposeByMode() {
	a, _ := cudalayout.ParseLayout("(12,(4,8)):(59,(13,1))")
	t0, _ := cudalayout.ParseLayout("3:4")
	t1, _ := cudalayout.ParseLayout("8:2")
	r, _ := cudalayout.ComposeByMode(a, []cudalayout.Layout{t0, t1})
	fmt.Println(r)
	// Output: (3,(2,4)):(236,(26,1))
}

func ExampleShapeTiler() {
	a, _ := cudalayout.ParseLayout("(6,8):(1,6)")
	tiler, _ := cudalayout.ShapeTiler(3, 4)
	r, _ := cudalayout.ComposeByMode(a, tiler)
	fmt.Println(r)
	// Output: (3,4):(1,6)
}

func ExampleComplement() {
	l, _ := cudalayout.ParseLayout("(2,2):(1,6)")
	r, _ := cudalayout.Complement(l, 24)
	fmt.Println(r)
	// Output: (3,2):(2,12)
}

func ExampleLogicalDivide() {
	a, _ := cudalayout.ParseLayout("(4,2,3):(2,1,8)")
	b, _ := cudalayout.ParseLayout("4:2")
	r, _ := cudalayout.LogicalDivide(a, b)
	fmt.Println(r)
	// Output: ((2,2),(2,3)):((4,1),(2,8))
}

func ExampleLogicalDivideByMode() {
	a, _ := cudalayout.ParseLayout("(8,8):(1,8)")
	tiler, _ := cudalayout.ShapeTiler(4, 2)
	r, _ := cudalayout.LogicalDivideByMode(a, tiler)
	fmt.Println(r)
	// Output: ((4,2),(2,4)):((1,4),(8,16))
}

// ZippedDivide gathers the tiles into mode 0 and the tile index into mode 1:
// Offset(i, k) is element i of tile k.
func ExampleZippedDivide() {
	a, _ := cudalayout.ParseLayout("(8,8):(1,8)")
	tiler, _ := cudalayout.ShapeTiler(4, 2)
	z, _ := cudalayout.ZippedDivide(a, tiler)
	o, _ := z.Offset(0, 3)
	fmt.Println(z, o)
	// Output: ((4,2),(2,4)):((1,8),(4,16)) 20
}

func ExampleF2Stride() {
	l, _ := cudalayout.NewLayout(cudalayout.IntShape(4), cudalayout.F2Stride(3, 1))
	for i := range l.Size() {
		o, _ := l.Index(i)
		fmt.Print(o, " ")
	}
	fmt.Println()
	// Output: 0 3 1 2
}

func ExampleXORSwizzle_Apply() {
	s := cudalayout.TMASwizzleXOR(cudalayout.TMASwizzle128B)
	fmt.Println(s, s.Apply(0x80), s.Apply(0x390))
	// Output: Swizzle<3,4,3> 144 992
}

func ExampleXORSwizzle_Compose() {
	s, _ := cudalayout.NewXORSwizzle(1, 0, 2)
	l, _ := cudalayout.ParseLayout("8:1")
	r, _ := s.Compose(l)
	fmt.Println(r)
	// Output: 8:xor(1,2,5)
}

func ExampleNewXORSwizzle() {
	_, err := cudalayout.NewXORSwizzle(3, 4, 2)
	fmt.Println(errors.Is(err, cudalayout.ErrUnsupported))
	// Output: true
}

// The F2 layout of a tensor-map tile addresses the same bytes as
// SwizzledTile.
func ExampleTMASwizzleLayout() {
	l, _ := cudalayout.TMASwizzleLayout(8, 64, 2, cudalayout.TMASwizzle128B)
	tile, _ := cudalayout.NewSwizzledTile(8, 64, 2, cudalayout.TMASwizzle128B)
	a, _ := l.Offset(3, 9)
	b, _ := tile.ByteOffset(3, 9)
	fmt.Println(a, b)
	// Output: 418 418
}

// No instrument has measured a swizzle on a device, so the answer is
// pending, not refused.
func ExampleXORSwizzle_CheckDevice() {
	err := cudalayout.TMASwizzleXOR(cudalayout.TMASwizzle128B).CheckDevice(cudalayout.TargetSM100a)
	fmt.Println(errors.Is(err, cudaerrors.ErrHardwarePending), errors.Is(err, cudaerrors.ErrUnsupported))
	// Output: true false
}

func ExampleShape_String() {
	s := cudalayout.TupleShape(cudalayout.IntShape(4), cudalayout.TupleShape(cudalayout.IntShape(2), cudalayout.IntShape(3)))
	d := cudalayout.TupleStride(cudalayout.BasisStride(0, 1), cudalayout.TupleStride(cudalayout.IntStride(0), cudalayout.F2Stride(8, 16)))
	fmt.Println(s, s.Rank(), d, d.Rank())
	// Output: (4,(2,3)) 2 (1@0,(0,xor(8,16))) 2
}

func ExampleParseStride() {
	d, _ := cudalayout.ParseStride("(1@0,(4@0,1@1))")
	fmt.Println(d)
	// Output: (1@0,(4@0,1@1))
}

func ExampleLayout_Equal() {
	a, _ := cudalayout.ParseLayout("(2,4):(1,2)")
	b, _ := cudalayout.ParseLayout("8:1")
	fmt.Println(a.Equal(b), cudalayout.Coalesce(a).Equal(b))
	// Output: false true
}
