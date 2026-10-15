package cudalayout

import (
	"errors"
	"fmt"
	"math"
	"math/big"
	"testing"
)

func TestSwizzledTileExtentOverflow(t *testing.T) {
	for _, mode := range []TMASwizzle{TMASwizzleNone, TMASwizzle32B, TMASwizzle64B, TMASwizzle128B} {
		for _, elem := range []int{1, 2, 4} {
			t.Run(fmt.Sprintf("%s/%d", mode, elem), func(t *testing.T) {
				rowBytes := mode.Bytes()
				if rowBytes == 0 {
					rowBytes = 16
				}
				rows := math.MaxInt/rowBytes + 1
				if tile, err := NewSwizzledTile(rows, rowBytes/elem, elem, mode); !errors.Is(err, ErrOverflow) {
					t.Fatalf("overflowing footprint: tile=%+v err=%v", tile, err)
				}
			})
		}
	}
	for _, mode := range []TMASwizzle{TMASwizzleNone, TMASwizzle32B, TMASwizzle64B, TMASwizzle128B} {
		for _, elem := range []int{2, 4} {
			// The product is a power of two above MaxInt and a multiple of
			// every supported span, so modulo alignment cannot reject it.
			if tile, err := NewSwizzledTile(1, math.MaxInt/elem+1, elem, mode); !errors.Is(err, ErrOverflow) {
				t.Fatalf("%s/%d overflowing row: tile=%+v err=%v", mode, elem, tile, err)
			}
		}
	}
}

func TestSwizzledTileExtentBoundary(t *testing.T) {
	for _, mode := range []TMASwizzle{TMASwizzleNone, TMASwizzle32B, TMASwizzle64B, TMASwizzle128B} {
		for _, elem := range []int{1, 2, 4} {
			t.Run(fmt.Sprintf("%s/%d", mode, elem), func(t *testing.T) {
				cols := 17
				if mode != TMASwizzleNone {
					cols = 2 * mode.Bytes() / elem
				}
				rowBytes := cols * elem
				// These descriptors allocate no tensor storage. Check both a
				// small odd row count and the largest representable row count.
				for _, rows := range []int{3, math.MaxInt / rowBytes} {
					tile, err := NewSwizzledTile(rows, cols, elem, mode)
					if err != nil {
						t.Fatal(err)
					}
					if tile.Bytes() != rows*rowBytes {
						t.Fatalf("size=%d want %d", tile.Bytes(), rows*rowBytes)
					}
					for _, p := range [][2]int{{0, 0}, {1, 1}, {rows - 1, cols - 1}, {rows - 1, cols / 2}} {
						got, err := tile.ByteOffset(p[0], p[1])
						if err != nil {
							t.Fatal(err)
						}
						// Evaluate the byte-address formula with arbitrary-precision
						// integers so a shared int overflow cannot bless the result.
						n := func(x int) *big.Int { return big.NewInt(int64(x)) }
						want := new(big.Int)
						if span := mode.Bytes(); span != 0 {
							spanCols := span / elem
							want.Mul(n(p[1]/spanCols), n(rows))
							want.Add(want, n(p[0]))
							want.Mul(want, n(span))
							want.Add(want, n((p[1]%spanCols)*elem))
							mask := new(big.Int).Mod(new(big.Int).Set(want), n(span*8))
							mask.Div(mask, n(128)).Mul(mask, n(16))
							want.Xor(want, mask)
						} else {
							want.Mul(n(p[0]), n(cols)).Add(want, n(p[1])).Mul(want, n(elem))
						}
						if got < 0 || got > tile.Bytes()-elem || want.Cmp(n(got)) != 0 {
							t.Fatalf("%dx%d element %v: offset=%d want=%s size=%d", rows, cols, p, got, want, tile.Bytes())
						}
					}
					if _, err := tile.ByteOffset(rows, 0); !errors.Is(err, ErrOutOfBounds) {
						t.Fatalf("row tail: %v", err)
					}
					if _, err := tile.ByteOffset(0, cols); !errors.Is(err, ErrOutOfBounds) {
						t.Fatalf("column tail: %v", err)
					}
				}
			})
		}
	}
}

func TestSwizzledTileElementIntervals(t *testing.T) {
	for _, mode := range []TMASwizzle{TMASwizzleNone, TMASwizzle32B, TMASwizzle64B, TMASwizzle128B} {
		t.Run(mode.String(), func(t *testing.T) {
			for rows := 1; rows <= 17; rows++ {
				for cols := 1; cols <= 129; cols++ {
					for elem := 1; elem <= 256; elem *= 2 {
						tile, err := NewSwizzledTile(rows, cols, elem, mode)
						if mode != TMASwizzleNone && (elem > 16 || cols*elem%mode.Bytes() != 0) {
							if !errors.Is(err, ErrUnsupported) {
								t.Fatalf("%dx%dx%d: unsupported mapping err=%v", rows, cols, elem, err)
							}
							continue
						}
						if err != nil {
							t.Fatalf("%dx%dx%d: %v", rows, cols, elem, err)
						}
						seen := make([]bool, rows*cols)
						for row := 0; row < rows; row++ {
							for col := 0; col < cols; col++ {
								off, err := tile.ByteOffset(row, col)
								if err != nil || off < 0 || off > tile.Bytes()-elem || off%elem != 0 {
									t.Fatalf("%dx%dx%d (%d,%d): offset=%d size=%d err=%v", rows, cols, elem, row, col, off, tile.Bytes(), err)
								}
								// Equal-width aligned intervals overlap exactly when they
								// occupy the same slot. No large tensor is allocated.
								if seen[off/elem] {
									t.Fatalf("%dx%dx%d (%d,%d): overlapping interval at %d", rows, cols, elem, row, col, off)
								}
								seen[off/elem] = true
							}
						}
						for slot, present := range seen {
							if !present {
								t.Fatalf("%dx%dx%d: missing interval %d", rows, cols, elem, slot)
							}
						}
					}
				}
			}
		})
	}
}
