package cudalayout

import "fmt"

// FragmentLayout names the register layout of an MMA base tile. The two
// layouts are transposes of one another, and they differ in the order their
// packed slots visit the tile, not only in the element each lane starts from.
type FragmentLayout uint8

const (
	// FragmentRow is the mma.sync row layout: lane l begins at
	// row l/4, column 2*(l%4).
	FragmentRow FragmentLayout = iota
	// FragmentCol is the column layout: lane l begins at
	// row 2*(l%4), column l/4.
	FragmentCol
)

func (l FragmentLayout) String() string {
	switch l {
	case FragmentRow:
		return "row"
	case FragmentCol:
		return "col"
	default:
		return fmt.Sprintf("fragmentlayout(%d)", uint8(l))
	}
}

// A Fragment is the warp-level register layout of one MMA base tile: how a
// 16x16 tile is distributed across the 32 lanes of a warp by mma.sync.
//
// A Fragment is keyed by element width and layout because both change the
// mapping. It is distinct from [Tile.Owner], which orders elements linearly
// and knows neither. Owner remains for callers that want a flat ownership
// model; a register tile that must agree with an MMA fragment uses a
// Fragment.
//
// A Fragment is the logical mapping, which is what a load emitter must end
// up with, not the sequence it takes to get there. ThunderKittens reaches it
// by permuting the shared addresses of half the lanes for bank reasons and
// swapping the register halves back afterwards
// (include/ops/group/memory/tile/shared_to_register.cuh:79 and 100-104).
// An emitter must reproduce both steps or neither: doing the address
// permutation without the compensating swap transposes the pairs on half
// the lanes, and no comparison against this model can detect that, because
// the model is right.
//
// The zero Fragment is invalid. Construct one with [NewFragment].
type Fragment struct {
	elemBytes int
	layout    FragmentLayout
}

// NewFragment returns the fragment layout for an element width and layout.
//
// Element widths of 2 and 4 bytes are modeled. One-byte elements are
// rejected: their base tile is 16x32 rather than 16x16 and packs four
// elements per slot, and this package has no verified mapping for that
// distribution. [ErrUnsupported] there means unmodeled, not absent from the
// hardware.
func NewFragment(elementBytes int, layout FragmentLayout) (Fragment, error) {
	switch elementBytes {
	case 2, 4:
	case 1:
		return Fragment{}, fmt.Errorf("one-byte fragment distribution is not modeled: %w", ErrUnsupported)
	default:
		return Fragment{}, fmt.Errorf("element size %d is not 2 or 4 bytes: %w", elementBytes, ErrUnsupported)
	}
	if layout != FragmentRow && layout != FragmentCol {
		return Fragment{}, fmt.Errorf("unknown fragment layout %d: %w", uint8(layout), ErrUnsupported)
	}
	return Fragment{elemBytes: elementBytes, layout: layout}, nil
}

// Rows returns the base tile row count.
func (f Fragment) Rows() int { return 16 }

// Cols returns the base tile column count.
func (f Fragment) Cols() int { return 16 }

// Layout returns the fragment's layout.
func (f Fragment) Layout() FragmentLayout { return f.layout }

// ElementBytes returns the element width the fragment was built for.
func (f Fragment) ElementBytes() int { return f.elemBytes }

// ElementsPerThread returns the elements one lane holds: the tile's element
// count divided over the warp's 32 lanes.
func (f Fragment) ElementsPerThread() int { return f.Rows() * f.Cols() / 32 }

// ElementsPerSlot returns the elements in one packed slot: a pair along the
// layout's packed axis for every width [NewFragment] admits.
//
// It is 2 rather than a function of the element width only because one-byte
// elements are rejected. Their packing is four elements per slot while the
// slot count stays 4, so a one-byte Fragment would keep
// [Fragment.PackedPerThread] at 4 and silently break this.
func (f Fragment) ElementsPerSlot() int { return 2 }

// PackedPerThread returns the packed slots one lane holds.
func (f Fragment) PackedPerThread() int { return f.ElementsPerThread() / f.ElementsPerSlot() }

// RegistersPerThread returns the 32-bit registers one lane holds. A slot of
// two 16-bit elements occupies one register; a slot of two 32-bit elements
// occupies two.
func (f Fragment) RegistersPerThread() int { return f.ElementsPerThread() * f.elemBytes / 4 }

// A FragmentOwner identifies the lane and register position holding one
// element of a fragment.
type FragmentOwner struct {
	// Lane is the owning lane, in [0,32).
	Lane int
	// Slot is the packed register slot, in [0,PackedPerThread).
	Slot int
	// Index is the element's position within the packed slot, in
	// [0,ElementsPerSlot). A slot is a 2-vector of the packed type --
	// ThunderKittens stores dtype data[packed_per_thread] where dtype is
	// bf16_2, half_2 or float2 (rt_base.cuh:64-66) -- so Index 0 is the .x
	// component and 1 is .y, at both element widths. Whether a slot occupies
	// one 32-bit register or two does not appear here; see
	// [Fragment.RegistersPerThread], which answers that for occupancy and for
	// declaring .reg .b32, not for addressing.
	Index int
}

// lanePosition returns the tile position lane holds first.
func (f Fragment) lanePosition(lane int) (row, col int) {
	if f.layout == FragmentRow {
		return lane / 4, 2 * (lane % 4)
	}
	return 2 * (lane % 4), lane / 4
}

// Owner returns the lane and register position holding the element at row,
// col.
func (f Fragment) Owner(row, col int) (FragmentOwner, error) {
	if f.elemBytes == 0 {
		return FragmentOwner{}, ErrUnsupported
	}
	if row < 0 || row >= f.Rows() || col < 0 || col >= f.Cols() {
		return FragmentOwner{}, ErrOutOfBounds
	}
	// Each lane holds a 2x2 arrangement of pairs: one pair per quadrant of
	// the tile, the pair running along the layout's packed axis.
	var lane, slot, index int
	if f.layout == FragmentRow {
		// Row layout packs along columns: a lane owns columns c0, c0+1.
		lane = (row%8)*4 + (col%8)/2
		slot = (row / 8) + 2*(col/8)
		index = col % 2
	} else {
		// Column layout packs along rows: a lane owns rows r0, r0+1.
		lane = (col%8)*4 + (row%8)/2
		slot = 2*(row/8) + (col / 8)
		index = row % 2
	}
	return FragmentOwner{Lane: lane, Slot: slot, Index: index}, nil
}

// Element returns the tile position an owner slot holds. It is the inverse of
// [Fragment.Owner].
func (f Fragment) Element(o FragmentOwner) (row, col int, err error) {
	if f.elemBytes == 0 {
		return 0, 0, ErrUnsupported
	}
	if o.Lane < 0 || o.Lane >= 32 ||
		o.Slot < 0 || o.Slot >= f.PackedPerThread() ||
		o.Index < 0 || o.Index >= f.ElementsPerSlot() {
		return 0, 0, ErrOutOfBounds
	}
	r0, c0 := f.lanePosition(o.Lane)
	if f.layout == FragmentRow {
		return r0 + 8*(o.Slot%2), c0 + 8*(o.Slot/2) + o.Index, nil
	}
	return r0 + 8*(o.Slot/2) + o.Index, c0 + 8*(o.Slot%2), nil
}
