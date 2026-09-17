package cudalayout

import "fmt"

func Example() {
	tile, _ := NewTile(2, 3, 4, WithStride(4))
	p, _ := tile.Index(1, 2)
	fmt.Println(p.ByteOffset)
	// Output: 24
}

func ExampleNewTile() {
	tile, _ := NewTile(2, 4, 2, WithStride(16), WithSwizzle(Swizzle32B))
	p, _ := tile.Index(1, 2)
	fmt.Println(p.ElementOffset, p.ByteOffset, tile.StrideElements())
	// Output: 18 38 16
}

func ExampleTile_accessors() {
	t, _ := NewTile(2, 3, 2, WithStride(4), WithAlignment(16), WithSwizzle(SwizzleNone))
	f, _ := t.FootprintBytes()
	fmt.Println(t.Rows(), t.Cols(), t.ElementBytes(), t.StrideElements(), t.AlignmentBytes(), t.Swizzle(), f)
	// Output: 2 3 2 4 16 none 16
}

func ExampleTile_Index() {
	t, _ := NewTile(2, 3, 2)
	p, _ := t.Index(1, 2)
	fmt.Println(p.Row, p.Col, p.ElementOffset, p.ByteOffset)
	// Output: 1 2 5 10
}

func ExampleTile_Compatible() {
	a, _ := NewTile(2, 2, 4)
	b, _ := NewTile(2, 2, 4)
	c, _ := NewTile(2, 2, 4, WithStride(4))
	fmt.Println(a.Compatible(b), a.Compatible(c), len(a.Identity()))
	// Output: true false 64
}

func ExampleTile_Owner() {
	t, _ := NewTile(2, 2, 4)
	o, _ := t.Owner(1, 0)
	r, c, _ := t.OwnerFor(o.Lane, o.Register)
	fmt.Println(o.Lane, o.Register, r, c)
	// Output: 2 0 1 0
}

func ExampleTile_TransposeOwners() {
	t, _ := NewTile(2, 3, 1)
	o, _ := t.TransposeOwners(1, 2)
	fmt.Println(o.Lane, o.Register)
	// Output: 5 0
}

func ExampleRedistributionPermutation_Map() {
	p, _ := TransposePermutation(2, 3)
	r, c, _ := p.Map(1, 2)
	fmt.Println(r, c)
	// Output: 2 1
}

func ExampleFixtureSourceForTarget() {
	t, _ := NewTile(2, 2, 2)
	source, err := FixtureSourceForTarget(t, TargetSM100a)
	fmt.Println(err == nil, len(source) > 0)
	// Output: true true
}
