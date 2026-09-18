package cudatma

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudalayout"
)

// AgreesWithTile reports whether a descriptor and a shared tile address the
// same bytes.
//
// A TMA store writes shared memory in the layout the descriptor names, and
// register loads read it in the layout the tile names. When the two disagree
// the result is wrong data, not an error: the hardware has no way to report
// it and neither side can detect it alone. This is the check that says so.
//
// The comparison lives here rather than on the tile because the box extent is
// half of it, and a tile knows nothing about a descriptor. [cudalayout] is
// closed-form arithmetic with no device dependency; keeping the direction of
// the dependency downward is what lets it stay that way.
//
// Agreement is element width, swizzle mode, and the box extent against the
// tile's own shape. The descriptor is validated first, because comparing
// against an unvalidated one compares against nothing.
func (d Descriptor) AgreesWithTile(t cudalayout.SwizzledTile) error {
	if err := d.Validate(); err != nil {
		return fmt.Errorf("descriptor does not describe a tensor map: %w", err)
	}
	if uint32(t.ElementBytes()) != d.ElementBytes {
		return fmt.Errorf("tile element size %d does not match descriptor element size %d",
			t.ElementBytes(), d.ElementBytes)
	}
	sw, err := cudalayout.TMASwizzleFromBytes(int(d.Swizzle))
	if err != nil {
		return fmt.Errorf("descriptor swizzle: %w", err)
	}
	if err := t.AgreesWith(sw); err != nil {
		return err
	}
	// Box[0] is the contiguous extent and the tile's columns; Box[1] is the
	// next dimension and the tile's rows. A descriptor of rank above 2 tiles
	// a larger tensor, but the two innermost extents are still the shared
	// tile a transfer lands in.
	if uint32(t.Cols()) != d.Box[0] {
		return fmt.Errorf("tile has %d columns, descriptor box extent 0 is %d", t.Cols(), d.Box[0])
	}
	if uint32(t.Rows()) != d.Box[1] {
		return fmt.Errorf("tile has %d rows, descriptor box extent 1 is %d", t.Rows(), d.Box[1])
	}
	return nil
}
