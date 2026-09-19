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

// elementsPerSlot is the elements one packed slot holds, by element width.
//
// It is a table and not a function of the width because it is a property of
// ThunderKittens' packed type rather than of arithmetic: bf16_2, half_2 and
// float2 are pairs at two and four bytes, while one-byte elements pack as a
// four-vector (fp8e4m3_4). Measured, see testdata/tklayout.json.
var elementsPerSlot = map[int]int{1: 4, 2: 2, 4: 2}

// NewFragment returns the fragment layout for an element width and layout.
//
// Element widths of 1, 2 and 4 bytes are modeled in the row layout. A
// one-byte column fragment returns [ErrUnmodeled]: its base tile is 16x32,
// so the column layout is not the transpose of a square tile, and no
// measurement of it exists. That is a gap in the evidence, not an
// unsupported combination, which is why it is not plain [ErrUnsupported].
//
// One byte means one byte of storage. A type whose unit is a pair of 4-bit
// values also has an element width of 1 and would be admitted here, and this
// package cannot tell it apart, because a Fragment is keyed on width alone.
// No such type was measured, and ThunderKittens' own register arithmetic for
// fp4 under-allocates by 2x, so do not read an fp4 answer out of this.
func NewFragment(elementBytes int, layout FragmentLayout) (Fragment, error) {
	if _, ok := elementsPerSlot[elementBytes]; !ok {
		return Fragment{}, fmt.Errorf("element size %d is not 1, 2 or 4 bytes: %w", elementBytes, ErrUnsupported)
	}
	if layout != FragmentRow && layout != FragmentCol {
		return Fragment{}, fmt.Errorf("unknown fragment layout %d: %w", uint8(layout), ErrUnsupported)
	}
	if elementBytes == 1 && layout == FragmentCol {
		return Fragment{}, fmt.Errorf("one-byte column fragment: %w", ErrUnmodeled)
	}
	return Fragment{elemBytes: elementBytes, layout: layout}, nil
}

// Rows returns the base tile row count.
func (f Fragment) Rows() int { return 16 }

// Cols returns the base tile column count. It is 16 at two and four bytes
// and 32 at one, where ThunderKittens doubles the base tile's column extent
// (util.cuh: TILE_COL_DIM is BASE_TILE_DIM*2 for a one-byte type).
func (f Fragment) Cols() int { return 8 * f.ElementsPerSlot() }

// Layout returns the fragment's layout.
func (f Fragment) Layout() FragmentLayout { return f.layout }

// ElementBytes returns the element width the fragment was built for.
func (f Fragment) ElementBytes() int { return f.elemBytes }

// ElementsPerThread returns the elements one lane holds: the tile's element
// count divided over the warp's 32 lanes.
func (f Fragment) ElementsPerThread() int { return f.Rows() * f.Cols() / 32 }

// ElementsPerSlot returns the elements in one packed slot, along the
// layout's packed axis: a pair at two and four bytes, a four-vector at one.
//
// The slot count stays 4 across all three widths, so this is the only place
// the one-byte packing appears. A Fragment that returned 2 here at one byte
// would report the right [Fragment.PackedPerThread] and address half the
// tile.
func (f Fragment) ElementsPerSlot() int { return elementsPerSlot[f.elemBytes] }

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
	// [0,ElementsPerSlot). A slot is a vector of the packed type --
	// ThunderKittens stores dtype data[packed_per_thread] where dtype is
	// bf16_2, half_2 or float2 (rt_base.cuh:64-66), or fp8e4m3_4 at one
	// byte -- so Index 0 is the .x component and 1 the .y. Whether a slot
	// occupies one 32-bit register or two does not appear here; see
	// [Fragment.RegistersPerThread], which answers that for occupancy and for
	// declaring .reg .b32, not for addressing.
	Index int
}

// lanePosition returns the tile position lane holds first. A lane's run
// along the packed axis is ElementsPerSlot long, so that is the stride
// between adjacent lanes on that axis.
func (f Fragment) lanePosition(lane int) (row, col int) {
	e := f.ElementsPerSlot()
	if f.layout == FragmentRow {
		return lane / 4, e * (lane % 4)
	}
	return e * (lane % 4), lane / 4
}

// Owner returns the lane and register position holding the element at row,
// col.
func (f Fragment) Owner(row, col int) (FragmentOwner, error) {
	if f.elemBytes == 0 {
		return FragmentOwner{}, ErrUninitialized
	}
	if row < 0 || row >= f.Rows() || col < 0 || col >= f.Cols() {
		return FragmentOwner{}, ErrOutOfBounds
	}
	// Each lane holds a 2x2 arrangement of runs: one run per quadrant of the
	// tile, the run going along the layout's packed axis and e elements
	// long. half is the extent of one quadrant along that axis -- 8 columns
	// of a 16-wide tile, 16 of a 32-wide one.
	e := f.ElementsPerSlot()
	var lane, slot, index int
	if f.layout == FragmentRow {
		// Row layout packs along columns: a lane owns a run of e columns.
		half := f.Cols() / 2
		lane = (row%8)*4 + (col%half)/e
		slot = (row / 8) + 2*(col/half)
		index = col % e
	} else {
		// Column layout packs along rows: a lane owns a run of e rows.
		half := f.Rows() / 2
		lane = (col%8)*4 + (row%half)/e
		slot = 2*(row/half) + (col / 8)
		index = row % e
	}
	return FragmentOwner{Lane: lane, Slot: slot, Index: index}, nil
}

// Element returns the tile position an owner slot holds. It is the inverse of
// [Fragment.Owner].
func (f Fragment) Element(o FragmentOwner) (row, col int, err error) {
	if f.elemBytes == 0 {
		return 0, 0, ErrUninitialized
	}
	if o.Lane < 0 || o.Lane >= 32 ||
		o.Slot < 0 || o.Slot >= f.PackedPerThread() ||
		o.Index < 0 || o.Index >= f.ElementsPerSlot() {
		return 0, 0, ErrOutOfBounds
	}
	r0, c0 := f.lanePosition(o.Lane)
	if f.layout == FragmentRow {
		return r0 + 8*(o.Slot%2), c0 + (f.Cols()/2)*(o.Slot/2) + o.Index, nil
	}
	return r0 + (f.Rows()/2)*(o.Slot/2) + o.Index, c0 + 8*(o.Slot%2), nil
}
