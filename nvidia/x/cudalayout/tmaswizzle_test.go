package cudalayout

import (
	"errors"
	"testing"
)

func TestTMASwizzleFromBytes(t *testing.T) {
	tests := []struct {
		bytes int
		want  TMASwizzle
		ok    bool
	}{
		{0, TMASwizzleNone, true},
		{32, TMASwizzle32B, true},
		{64, TMASwizzle64B, true},
		{128, TMASwizzle128B, true},
		{16, 0, false},
		{256, 0, false},
	}
	for _, tt := range tests {
		got, err := TMASwizzleFromBytes(tt.bytes)
		if tt.ok {
			if err != nil || got != tt.want {
				t.Errorf("TMASwizzleFromBytes(%d) = %v, %v; want %v, nil", tt.bytes, got, err, tt.want)
			}
			if got.Bytes() != tt.bytes {
				t.Errorf("%v.Bytes() = %d, want %d", got, got.Bytes(), tt.bytes)
			}
			continue
		}
		if !errors.Is(err, ErrUnsupported) {
			t.Errorf("TMASwizzleFromBytes(%d) error = %v, want ErrUnsupported", tt.bytes, err)
		}
	}
}

func TestNewSwizzledTileRejects(t *testing.T) {
	tests := []struct {
		name                     string
		rows, cols, elementBytes int
		swizzle                  TMASwizzle
	}{
		{"empty tile", 0, 16, 2, TMASwizzle128B},
		{"element not a power of two", 16, 16, 3, TMASwizzle128B},
		// 16 columns of 2 bytes is 32 bytes, not a multiple of 128.
		{"row narrower than the span", 16, 16, 2, TMASwizzle128B},
		// 64 columns of 2 bytes is 128 bytes, not a multiple of... it is,
		// so use a width that straddles: 48 columns is 96 bytes.
		{"row width straddles the span", 16, 48, 2, TMASwizzle64B},
		{"element wider than the span", 16, 32, 64, TMASwizzle32B},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := NewSwizzledTile(tt.rows, tt.cols, tt.elementBytes, tt.swizzle); !errors.Is(err, ErrUnsupported) {
				t.Fatalf("NewSwizzledTile(%d,%d,%d,%v) error = %v, want ErrUnsupported",
					tt.rows, tt.cols, tt.elementBytes, tt.swizzle, err)
			}
		})
	}
}

// tkByteOffset transcribes ThunderKittens' st::idx
// (include/types/shared/st.cuh:108-120) against a tile base of zero, which is
// what [SwizzledTile.BaseAlignment] requires a real base to be congruent to.
//
// Being a transcription, it can only reproduce the reading that produced it.
// TestSwizzledTileMatchesMeasuredStorage is the witness that is not, and it
// agrees with this one on all ten recorded shapes.
func tkByteOffset(rows, elemBytes, span, row, col int) int {
	subtileCols := span / elemBytes
	outer := col / subtileCols
	addr := elemBytes * (outer*rows*subtileCols + row*subtileCols + col%subtileCols)
	swizzle := ((addr % (span * 8)) >> 7) << 4
	return addr ^ swizzle
}

func TestSwizzledTileMatchesThunderKittens(t *testing.T) {
	tests := []struct {
		name                     string
		rows, cols, elementBytes int
		swizzle                  TMASwizzle
	}{
		{"128B bf16 64 wide", 16, 64, 2, TMASwizzle128B},
		{"128B fp32 32 wide", 16, 32, 4, TMASwizzle128B},
		{"64B bf16 32 wide", 32, 32, 2, TMASwizzle64B},
		{"32B bf16 16 wide", 16, 16, 2, TMASwizzle32B},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tile, err := NewSwizzledTile(tt.rows, tt.cols, tt.elementBytes, tt.swizzle)
			if err != nil {
				t.Fatal(err)
			}
			for row := 0; row < tt.rows; row++ {
				for col := 0; col < tt.cols; col++ {
					got, err := tile.ByteOffset(row, col)
					if err != nil {
						t.Fatalf("ByteOffset(%d,%d): %v", row, col, err)
					}
					want := tkByteOffset(tt.rows, tt.elementBytes, tt.swizzle.Bytes(), row, col)
					if got != want {
						t.Fatalf("ByteOffset(%d,%d) = %d, ThunderKittens has %d", row, col, got, want)
					}
				}
			}
		})
	}
}

// TestSwizzledTileIsAPermutation is the gate: a swizzle rearranges a tile's
// bytes and must not lose, duplicate, or leave one. A formula that XORed an
// offset out of range, or collided two elements, would fail here.
func TestSwizzledTileIsAPermutation(t *testing.T) {
	tests := []struct {
		name                     string
		rows, cols, elementBytes int
		swizzle                  TMASwizzle
	}{
		{"none", 16, 16, 2, TMASwizzleNone},
		{"32B", 16, 16, 2, TMASwizzle32B},
		{"64B", 16, 32, 2, TMASwizzle64B},
		{"128B bf16", 16, 64, 2, TMASwizzle128B},
		{"128B fp32", 16, 32, 4, TMASwizzle128B},
		{"128B tall", 64, 64, 2, TMASwizzle128B},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tile, err := NewSwizzledTile(tt.rows, tt.cols, tt.elementBytes, tt.swizzle)
			if err != nil {
				t.Fatal(err)
			}
			seen := make(map[int][2]int, tt.rows*tt.cols)
			for row := 0; row < tt.rows; row++ {
				for col := 0; col < tt.cols; col++ {
					off, err := tile.ByteOffset(row, col)
					if err != nil {
						t.Fatalf("ByteOffset(%d,%d): %v", row, col, err)
					}
					if off < 0 || off+tt.elementBytes > tile.Bytes() {
						t.Fatalf("ByteOffset(%d,%d) = %d, outside a %d-byte tile", row, col, off, tile.Bytes())
					}
					if off%tt.elementBytes != 0 {
						t.Fatalf("ByteOffset(%d,%d) = %d, not element aligned", row, col, off)
					}
					if prev, ok := seen[off]; ok {
						t.Fatalf("ByteOffset(%d,%d) = %d, already held by (%d,%d)", row, col, off, prev[0], prev[1])
					}
					seen[off] = [2]int{row, col}
				}
			}
			if want := tt.rows * tt.cols; len(seen) != want {
				t.Errorf("distinct offsets = %d, want %d", len(seen), want)
			}
		})
	}
}

func TestSwizzledTileBaseAlignment(t *testing.T) {
	tests := []struct {
		swizzle TMASwizzle
		want    int
	}{
		{TMASwizzle32B, 256},
		{TMASwizzle64B, 512},
		{TMASwizzle128B, 1024},
	}
	for _, tt := range tests {
		tile, err := NewSwizzledTile(16, tt.swizzle.Bytes()/2*2, 2, tt.swizzle)
		if err != nil {
			t.Fatal(err)
		}
		if got := tile.BaseAlignment(); got != tt.want {
			t.Errorf("%v BaseAlignment() = %d, want %d", tt.swizzle, got, tt.want)
		}
	}
	plain, err := NewSwizzledTile(16, 16, 4, TMASwizzleNone)
	if err != nil {
		t.Fatal(err)
	}
	if got := plain.BaseAlignment(); got != 4 {
		t.Errorf("unswizzled BaseAlignment() = %d, want 4", got)
	}
}

func TestSwizzledTileAgreesWith(t *testing.T) {
	tile, err := NewSwizzledTile(16, 64, 2, TMASwizzle128B)
	if err != nil {
		t.Fatal(err)
	}
	if err := tile.AgreesWith(TMASwizzle128B); err != nil {
		t.Errorf("AgreesWith(128B) = %v, want nil", err)
	}
	if err := tile.AgreesWith(TMASwizzle64B); !errors.Is(err, ErrUnsupported) {
		t.Errorf("AgreesWith(64B) error = %v, want ErrUnsupported", err)
	}
	if err := tile.AgreesWith(TMASwizzleNone); !errors.Is(err, ErrUnsupported) {
		t.Errorf("AgreesWith(none) error = %v, want ErrUnsupported", err)
	}
}

func TestSwizzledTileOutOfBounds(t *testing.T) {
	tile, err := NewSwizzledTile(16, 64, 2, TMASwizzle128B)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := tile.ByteOffset(16, 0); !errors.Is(err, ErrOutOfBounds) {
		t.Errorf("ByteOffset(16,0) error = %v, want ErrOutOfBounds", err)
	}
	var zero SwizzledTile
	if _, err := zero.ByteOffset(0, 0); !errors.Is(err, ErrUnsupported) {
		t.Errorf("zero tile ByteOffset error = %v, want ErrUnsupported", err)
	}
}

// cuteSwizzleApply transcribes CUTLASS CuTe's Swizzle<BBits,MBase,SShift>
// from include/cute/swizzle.hpp:
//
//	bit_msk = (1 << BBits) - 1
//	yyy_msk = bit_msk << (MBase + max(0, SShift))
//	apply(offset) = offset ^ shiftr(offset & yyy_msk, SShift)
//
// CuTe's get_tma_swizzle_bits (include/cute/atom/copy_traits_sm90_tma_swizzle.hpp)
// pins the triple for each tensor-map mode: MBase is 4 and SShift is 3, and
// BBits is 3 for 128B, 2 for 64B, 1 for 32B and 0 for none.
func cuteSwizzleApply(offset, bbits int) int {
	const mbase, sshift = 4, 3
	bitMsk := (1 << bbits) - 1
	yyyMsk := bitMsk << (mbase + sshift)
	return offset ^ ((offset & yyyMsk) >> sshift)
}

// TestSwizzleXORMatchesCuTe is the second witness for the swizzle step.
//
// TestSwizzledTileMatchesThunderKittens compares against tkByteOffset, which
// this package's author transcribed from the same source the implementation
// was written from. That is one reading spelled twice, not two witnesses, and
// it is the weakness [TestFragmentSlotQuadrants] closes for the fragment
// mapping. CuTe is the independent route: a different project, a different
// author, and a different parameterization -- bit masks and a shift rather
// than modulo and divide arithmetic -- reaching the same permutation.
//
// It witnesses the XOR step only. CuTe expresses the column grouping as a
// separate Layout composed with the Swizzle, so the group ordering in
// [SwizzledTile.ByteOffset] still rests on the ThunderKittens reading alone.
func TestSwizzleXORMatchesCuTe(t *testing.T) {
	tests := []struct {
		swizzle TMASwizzle
		bbits   int
	}{
		{TMASwizzleNone, 0},
		{TMASwizzle32B, 1},
		{TMASwizzle64B, 2},
		{TMASwizzle128B, 3},
	}
	for _, tt := range tests {
		t.Run(tt.swizzle.String(), func(t *testing.T) {
			span := tt.swizzle.Bytes()
			// Sweep well past the eight-span repeat so a pattern that
			// restarts at the wrong period is caught.
			for offset := 0; offset < 4096; offset += swizzleChunk {
				var got int
				if span == 0 {
					got = offset
				} else {
					got = offset ^ swizzleXOR(offset, span)
				}
				if want := cuteSwizzleApply(offset, tt.bbits); got != want {
					t.Fatalf("offset %d: this package gives %d, CuTe gives %d", offset, got, want)
				}
			}
		})
	}
}
