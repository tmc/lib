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
package cudalayout
