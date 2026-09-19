package cudalayout

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"
)

// tkLayout is testdata/tklayout.json: ThunderKittens' own layouts, measured
// on a GPU. See testdata/tklayout.md for provenance and for what each map
// means.
type tkLayout struct {
	TKCommit string    `json:"tkCommit"`
	Arch     string    `json:"arch"`
	Shapes   []tkShape `json:"shapes"`
}

type tkShape struct {
	Elem         string `json:"elem"`
	ElemBytes    int    `json:"elemBytes"`
	Rows         int    `json:"rows"`
	Cols         int    `json:"cols"`
	SwizzleBytes int    `json:"swizzleBytes"`

	BaseRows int `json:"baseRows"`
	BaseCols int `json:"baseCols"`
	Height   int `json:"height"`
	Width    int `json:"width"`

	BaseElementsPerThread  int `json:"baseElementsPerThread"`
	BasePackedPerThread    int `json:"basePackedPerThread"`
	BaseRegistersPerThread int `json:"baseRegistersPerThread"`
	ElementsPerSlot        int `json:"elementsPerSlot"`

	// Storage[i] is the logical (row, col) at raw storage element i.
	Storage [][2]int `json:"storage"`
	// Regs[lane][k] is the logical (row, col) lane holds in slot k.
	Regs [][][2]int `json:"regs"`
}

func (s tkShape) String() string {
	return fmt.Sprintf("%s_%dx%d", s.Elem, s.Rows, s.Cols)
}

func loadTKLayout(t *testing.T) tkLayout {
	t.Helper()
	b, err := os.ReadFile("testdata/tklayout.json")
	if err != nil {
		t.Fatal(err)
	}
	var l tkLayout
	if err := json.Unmarshal(b, &l); err != nil {
		t.Fatal(err)
	}
	if len(l.Shapes) == 0 {
		t.Fatal("fixture records no shapes: every comparison below would pass over nothing")
	}
	return l
}

// TestTKFixtureCoversTheWidths states the denominator the two differentials
// below run against, and fails if the fixture stops covering a width this
// package models. A fixture that quietly lost its one-byte shapes would
// leave every one-byte claim untested while both differentials stayed green.
func TestTKFixtureCoversTheWidths(t *testing.T) {
	l := loadTKLayout(t)
	widths := make(map[int]int)
	for _, s := range l.Shapes {
		widths[s.ElemBytes]++
	}
	for _, w := range []int{1, 2, 4} {
		if widths[w] == 0 {
			t.Errorf("no shape of %d-byte elements; %d-byte claims are untested", w, w)
		}
	}
	t.Logf("TK %s on %s: %d shapes, %d one-byte, %d two-byte, %d four-byte",
		l.TKCommit, l.Arch, len(l.Shapes), widths[1], widths[2], widths[4])
}

// TestFragmentMatchesMeasuredRegisters compares Fragment against what a GPU
// did, rather than against a reading of the headers.
//
// It does not replace TestFragmentMatchesThunderKittens, which stays: that
// one compares against tkElement, a transcription of shared_to_register.cuh
// by whoever wrote Fragment, so it is one reading wearing two hats -- but it
// is the only witness for FragmentCol, which warp::load does not produce and
// this fixture therefore does not record. Where the two overlap, the
// measurement is the stronger one and this is the test to believe.
//
// It walks the whole recorded tile, not one base tile, because a shape wider
// or taller than the base tiles it with rt_base[height][width] and the slot
// index runs across those tiles. A base-tile-only comparison would leave the
// tiling untested.
func TestFragmentMatchesMeasuredRegisters(t *testing.T) {
	l := loadTKLayout(t)
	for _, s := range l.Shapes {
		t.Run(s.String(), func(t *testing.T) {
			f, err := NewFragment(s.ElemBytes, FragmentRow)
			if err != nil {
				t.Fatalf("NewFragment(%d, row) = %v", s.ElemBytes, err)
			}
			if got := f.Rows(); got != s.BaseRows {
				t.Errorf("Rows() = %d, TK base tile has %d", got, s.BaseRows)
			}
			if got := f.Cols(); got != s.BaseCols {
				t.Errorf("Cols() = %d, TK base tile has %d", got, s.BaseCols)
			}
			if got := f.ElementsPerSlot(); got != s.ElementsPerSlot {
				t.Errorf("ElementsPerSlot() = %d, TK packs %d", got, s.ElementsPerSlot)
			}
			if got := f.ElementsPerThread(); got != s.BaseElementsPerThread {
				t.Errorf("ElementsPerThread() = %d, TK has %d", got, s.BaseElementsPerThread)
			}
			if got := f.PackedPerThread(); got != s.BasePackedPerThread {
				t.Errorf("PackedPerThread() = %d, TK has %d", got, s.BasePackedPerThread)
			}
			if got := f.RegistersPerThread(); got != s.BaseRegistersPerThread {
				t.Errorf("RegistersPerThread() = %d, TK has %d", got, s.BaseRegistersPerThread)
			}
			if t.Failed() {
				return
			}

			perTile := f.PackedPerThread() * f.ElementsPerSlot()
			checked := 0
			for lane := range s.Regs {
				for k, want := range s.Regs[lane] {
					// A lane's k-th element: which base tile, then which
					// slot and position inside it.
					tile, within := k/perTile, k%perTile
					o := FragmentOwner{
						Lane:  lane,
						Slot:  within / f.ElementsPerSlot(),
						Index: within % f.ElementsPerSlot(),
					}
					row, col, err := f.Element(o)
					if err != nil {
						t.Fatalf("lane %d k %d: Element(%+v) = %v", lane, k, o, err)
					}
					row += f.Rows() * (tile / s.Width)
					col += f.Cols() * (tile % s.Width)
					if row != want[0] || col != want[1] {
						t.Fatalf("lane %d k %d: Element = (%d,%d), TK measured (%d,%d)", lane, k, row, col, want[0], want[1])
					}
					// Owner must invert Element within the base tile.
					back, err := f.Owner(row-f.Rows()*(tile/s.Width), col-f.Cols()*(tile%s.Width))
					if err != nil {
						t.Fatalf("lane %d k %d: Owner = %v", lane, k, err)
					}
					if back != o {
						t.Fatalf("lane %d k %d: Owner(Element(%+v)) = %+v", lane, k, o, back)
					}
					checked++
				}
			}
			if want := s.Rows * s.Cols; checked != want {
				t.Errorf("compared %d assignments, tile holds %d", checked, want)
			}
		})
	}
}

// TestSwizzledTileMatchesMeasuredStorage gives SwizzledTile.ByteOffset a
// witness that is not a transcription. TestSwizzledTileMatchesThunderKittens
// compares it against tkByteOffset, a Go copy of st::idx written by the same
// author, which would reproduce a misreading faithfully.
//
// Every recorded shape witnesses the swizzle, including the four whose row
// byte width equals their span. It is tempting to think those cannot: their
// group index is 0 for every column, so the swizzle permutes nothing across
// columns. But swizzleXOR keys off the whole linear offset, and the row term
// is in it, so the permutation runs across rows instead. Measured: making
// swizzleXOR return 0 fails all ten shapes, and no shape here has a swizzled
// offset equal to its row-major one at any element.
func TestSwizzledTileMatchesMeasuredStorage(t *testing.T) {
	l := loadTKLayout(t)
	for _, s := range l.Shapes {
		t.Run(s.String(), func(t *testing.T) {
			sw, err := TMASwizzleFromBytes(s.SwizzleBytes)
			if err != nil {
				t.Fatalf("TMASwizzleFromBytes(%d) = %v", s.SwizzleBytes, err)
			}
			tile, err := NewSwizzledTile(s.Rows, s.Cols, s.ElemBytes, sw)
			if err != nil {
				t.Fatalf("NewSwizzledTile(%d, %d, %d, %s) = %v", s.Rows, s.Cols, s.ElemBytes, sw, err)
			}
			if len(s.Storage) != s.Rows*s.Cols {
				t.Fatalf("storage map holds %d entries, tile holds %d", len(s.Storage), s.Rows*s.Cols)
			}
			for i, rc := range s.Storage {
				got, err := tile.ByteOffset(rc[0], rc[1])
				if err != nil {
					t.Fatalf("ByteOffset(%d, %d) = %v", rc[0], rc[1], err)
				}
				if want := i * s.ElemBytes; got != want {
					t.Fatalf("ByteOffset(%d, %d) = %d, TK stored it at %d", rc[0], rc[1], got, want)
				}
			}
		})
	}
}
