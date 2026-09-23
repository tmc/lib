// Package cudalayout describes checked physical tile layouts.
//
// It maps logical elements to byte offsets and register ownership without
// allocating storage or moving data. Strides are measured in elements and
// alignment in bytes. The package supports positive row-major tiles with
// optional 32-, 64-, and 128-byte generic software XOR row-column
// permutations. These permutations are not vendor TMA descriptor mappings
// and make no bank-conflict or performance claim.
//
// A zero Tile is invalid. There are no aliasing or broadcast layouts. Element
// slots are 1, 2, or 4 bytes; they describe logical slots and do not allocate
// GPRs. Padding can leave holes in the physical and register tile.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
//
// # Layout algebra
//
// A [Layout] is CuTe's shape:stride function from coordinates to offsets,
// written in CuTe notation ("((2,2),2):((4,1),2)") and evaluated
// colexicographically. [Coalesce], [Composition], [Complement],
// [LogicalDivide] and [ZippedDivide] implement CuTe's algebra over it, and
// reject rather than approximate an operation whose divisibility conditions
// fail. Offsets are in the unit of the strides; a Layout does not know
// elements from bytes.
//
// A [Stride] leaf is an integer, a basis stride that maps into one axis of a
// coordinate-valued offset, or an F2 stride whose contributions combine by
// XOR. An [XORSwizzle] composed with a layout gives an F2 layout, and
// [TMASwizzleLayout] gives the F2 layout, in bytes, of a tile under a
// tensor-map swizzle mode. Its offsets equal [SwizzledTile.ByteOffset]; the
// bank-conflict counts the tests compute from them are host arithmetic, and
// [XORSwizzle.CheckDevice] reports that no device has confirmed them.
package cudalayout
