package cudalayout

import (
	"errors"
	"testing"
)

func TestNewFragmentRejects(t *testing.T) {
	tests := []struct {
		name        string
		elementByte int
		layout      FragmentLayout
	}{
		{"one byte is unmodeled", 1, FragmentRow},
		{"eight byte has no fragment", 8, FragmentRow},
		{"zero width", 0, FragmentRow},
		{"unknown layout", 4, FragmentLayout(7)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := NewFragment(tt.elementByte, tt.layout); !errors.Is(err, ErrUnsupported) {
				t.Fatalf("NewFragment(%d, %v) error = %v, want ErrUnsupported", tt.elementByte, tt.layout, err)
			}
		})
	}
}

func TestFragmentCounts(t *testing.T) {
	tests := []struct {
		name                                  string
		elementBytes                          int
		perThread, perSlot, packed, registers int
	}{
		// ThunderKittens rt_base.cuh:76-79: elements_per_thread is 8,
		// packed_per_thread 4, and registers_per_thread 4 for a 4-byte
		// packed type (bf16_2) or 8 for an 8-byte one (float2).
		{"16-bit", 2, 8, 2, 4, 4},
		{"32-bit", 4, 8, 2, 4, 8},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f, err := NewFragment(tt.elementBytes, FragmentRow)
			if err != nil {
				t.Fatal(err)
			}
			if got := f.ElementsPerThread(); got != tt.perThread {
				t.Errorf("ElementsPerThread() = %d, want %d", got, tt.perThread)
			}
			if got := f.ElementsPerSlot(); got != tt.perSlot {
				t.Errorf("ElementsPerSlot() = %d, want %d", got, tt.perSlot)
			}
			if got := f.PackedPerThread(); got != tt.packed {
				t.Errorf("PackedPerThread() = %d, want %d", got, tt.packed)
			}
			if got := f.RegistersPerThread(); got != tt.registers {
				t.Errorf("RegistersPerThread() = %d, want %d", got, tt.registers)
			}
		})
	}
}

// tkElement returns the tile position ThunderKittens gives an owner slot,
// transcribed from the loads in
// include/ops/group/memory/tile/shared_to_register.cuh. The row layout is
// lines 73-95 and the column layout lines 109-119; each reads four pairs per
// lane from a base position that the layout fixes.
func tkElement(layout FragmentLayout, lane, slot, index int) (row, col int) {
	if layout == FragmentRow {
		r0, c0 := lane/4, 2*(lane%4)
		switch slot {
		case 0:
			return r0, c0 + index
		case 1:
			return r0 + 8, c0 + index
		case 2:
			return r0, c0 + 8 + index
		default:
			return r0 + 8, c0 + 8 + index
		}
	}
	r0, c0 := 2*(lane%4), lane/4
	switch slot {
	case 0:
		return r0 + index, c0
	case 1:
		return r0 + index, c0 + 8
	case 2:
		return r0 + 8 + index, c0
	default:
		return r0 + 8 + index, c0 + 8
	}
}

func TestFragmentMatchesThunderKittens(t *testing.T) {
	for _, layout := range []FragmentLayout{FragmentRow, FragmentCol} {
		t.Run(layout.String(), func(t *testing.T) {
			f, err := NewFragment(4, layout)
			if err != nil {
				t.Fatal(err)
			}
			for lane := 0; lane < 32; lane++ {
				for slot := 0; slot < f.PackedPerThread(); slot++ {
					for index := 0; index < f.ElementsPerSlot(); index++ {
						o := FragmentOwner{Lane: lane, Slot: slot, Index: index}
						row, col, err := f.Element(o)
						if err != nil {
							t.Fatalf("Element(%+v): %v", o, err)
						}
						wantRow, wantCol := tkElement(layout, lane, slot, index)
						if row != wantRow || col != wantCol {
							t.Errorf("Element(%+v) = (%d,%d), ThunderKittens has (%d,%d)", o, row, col, wantRow, wantCol)
						}
					}
				}
			}
		})
	}
}

// TestFragmentIsABijection is the gate the mapping has to pass: every element
// of the base tile is held exactly once, and Owner inverts Element. A mapping
// that collided, skipped an element, or disagreed with its own inverse would
// fail here.
func TestFragmentIsABijection(t *testing.T) {
	for _, layout := range []FragmentLayout{FragmentRow, FragmentCol} {
		for _, elementBytes := range []int{2, 4} {
			f, err := NewFragment(elementBytes, layout)
			if err != nil {
				t.Fatal(err)
			}
			t.Run(layout.String()+"/"+string(rune('0'+elementBytes)), func(t *testing.T) {
				seen := make(map[FragmentOwner][2]int, f.Rows()*f.Cols())
				for row := 0; row < f.Rows(); row++ {
					for col := 0; col < f.Cols(); col++ {
						o, err := f.Owner(row, col)
						if err != nil {
							t.Fatalf("Owner(%d,%d): %v", row, col, err)
						}
						if o.Lane < 0 || o.Lane >= 32 ||
							o.Slot < 0 || o.Slot >= f.PackedPerThread() ||
							o.Index < 0 || o.Index >= f.ElementsPerSlot() {
							t.Fatalf("Owner(%d,%d) = %+v, out of range", row, col, o)
						}
						if prev, ok := seen[o]; ok {
							t.Fatalf("Owner(%d,%d) = %+v, already held (%d,%d)", row, col, o, prev[0], prev[1])
						}
						seen[o] = [2]int{row, col}
						gotRow, gotCol, err := f.Element(o)
						if err != nil {
							t.Fatalf("Element(%+v): %v", o, err)
						}
						if gotRow != row || gotCol != col {
							t.Errorf("Element(Owner(%d,%d)) = (%d,%d)", row, col, gotRow, gotCol)
						}
					}
				}
				if want := f.Rows() * f.Cols(); len(seen) != want {
					t.Errorf("distinct owners = %d, want %d", len(seen), want)
				}
			})
		}
	}
}

func TestFragmentOutOfBounds(t *testing.T) {
	f, err := NewFragment(2, FragmentRow)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Owner(16, 0); !errors.Is(err, ErrOutOfBounds) {
		t.Errorf("Owner(16,0) error = %v, want ErrOutOfBounds", err)
	}
	if _, err := f.Owner(0, -1); !errors.Is(err, ErrOutOfBounds) {
		t.Errorf("Owner(0,-1) error = %v, want ErrOutOfBounds", err)
	}
	if _, _, err := f.Element(FragmentOwner{Lane: 32}); !errors.Is(err, ErrOutOfBounds) {
		t.Errorf("Element(lane 32) error = %v, want ErrOutOfBounds", err)
	}
	if _, _, err := f.Element(FragmentOwner{Slot: 4}); !errors.Is(err, ErrOutOfBounds) {
		t.Errorf("Element(slot 4) error = %v, want ErrOutOfBounds", err)
	}
	var zero Fragment
	if _, err := zero.Owner(0, 0); !errors.Is(err, ErrUnsupported) {
		t.Errorf("zero Fragment Owner error = %v, want ErrUnsupported", err)
	}
}

// TestFragmentSlotQuadrants pins each slot to the quadrant of the base tile
// it reads, which neither the bijection test nor a single transcription of
// the source can establish on its own.
//
// TestFragmentIsABijection passes under all 24 orderings of the four slots,
// 23 of which are wrong, so it cannot witness the slot order at all.
// TestFragmentMatchesThunderKittens does pin the order, but tkElement and
// the implementation were transcribed from the same lines by the same
// author, so it is one witness, not two.
//
// This table is the second witness: it comes from the 32-bit path at
// ThunderKittens include/ops/group/memory/tile/shared_to_register.cuh:84-99,
// which spells addresses out rather than relying on ldmatrix's implicit
// register semantics. Lines 84-85 send addr_1 to row+0 and addr_2 to row+8;
// lines 96-99 assign tmp[0] and tmp[2] from addr_1 at byte offsets 0 and 32,
// tmp[1] and tmp[3] from addr_2 at the same offsets. That is slot 1 in the
// lower row half and slot 2 in the right column half, and it is derived
// independently of the mapping under test. Swapping two rows of this table
// must fail.
func TestFragmentSlotQuadrants(t *testing.T) {
	tests := []struct {
		layout                 FragmentLayout
		slot, rowHalf, colHalf int
	}{
		{FragmentRow, 0, 0, 0},
		{FragmentRow, 1, 1, 0},
		{FragmentRow, 2, 0, 1},
		{FragmentRow, 3, 1, 1},
		// The column layout is the transpose, reached through the argument
		// swap in the ldsm4t call at line 64: tmp[0], tmp[2], tmp[1], tmp[3].
		// It rests on that one line; there is no explicit-address 32-bit
		// column path to cross-check it against.
		{FragmentCol, 0, 0, 0},
		{FragmentCol, 1, 0, 1},
		{FragmentCol, 2, 1, 0},
		{FragmentCol, 3, 1, 1},
	}
	for _, tt := range tests {
		f, err := NewFragment(4, tt.layout)
		if err != nil {
			t.Fatal(err)
		}
		for lane := 0; lane < 32; lane++ {
			for index := 0; index < f.ElementsPerSlot(); index++ {
				o := FragmentOwner{Lane: lane, Slot: tt.slot, Index: index}
				row, col, err := f.Element(o)
				if err != nil {
					t.Fatalf("Element(%+v): %v", o, err)
				}
				if row/8 != tt.rowHalf || col/8 != tt.colHalf {
					t.Fatalf("%v Element(%+v) = (%d,%d), in quadrant (%d,%d), want (%d,%d)",
						tt.layout, o, row, col, row/8, col/8, tt.rowHalf, tt.colHalf)
				}
			}
		}
	}
}
