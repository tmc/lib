package cudalayout

import (
	"math"
	"strconv"
	"strings"
	"testing"
)

func TestLayoutStoragePermutation(t *testing.T) {
	for _, elem := range []int{1, 2, 4} {
		for _, span := range []Swizzle{SwizzleNone, Swizzle32B, Swizzle64B, Swizzle128B} {
			stride := 256 / elem
			for _, cols := range []int{1, stride - 3, stride} {
				tile, err := NewTile(137, cols, elem, WithStride(stride), WithSwizzle(span))
				if err != nil {
					t.Fatal(err)
				}
				footprint, err := tile.FootprintBytes()
				if err != nil {
					t.Fatal(err)
				}
				seen := make(map[int]bool)
				for row := 0; row < tile.Rows(); row++ {
					for col := 0; col < cols; col++ {
						p, err := tile.Index(row, col)
						if err != nil {
							t.Fatal(err)
						}
						if p.ByteOffset < 0 || p.ByteOffset+elem > footprint || p.ByteOffset%elem != 0 || seen[p.ByteOffset] {
							t.Fatalf("bad physical index %+v", p)
						}
						seen[p.ByteOffset] = true
						// Invert in element units, independently of the byte-index formula.
						r := p.ByteOffset / (stride * elem)
						c := (p.ByteOffset / elem) % stride
						if span != SwizzleNone {
							c ^= r % (int(span) / elem)
						}
						if r != row || c != col {
							t.Fatalf("inverse (%d,%d), want (%d,%d)", r, c, row, col)
						}
					}
				}
			}
		}
	}
}

func TestLayoutInvalidContracts(t *testing.T) {
	var zero Tile
	if _, err := zero.FootprintBytes(); err == nil {
		t.Fatal("zero footprint accepted")
	}
	if _, _, err := zero.OwnerFor(0, 0); err == nil {
		t.Fatal("zero owner accepted")
	}
	if zero.Compatible(zero) {
		t.Fatal("invalid layouts are compatible")
	}
	for _, tc := range []struct {
		rows, cols, elem, stride, align int
		sw                              Swizzle
	}{
		{2, 1, 4, math.MaxInt, 16, 0}, {math.MaxInt, 1, 1, 2, 16, 0},
		{2, 33, 4, 33, 16, Swizzle128B}, {2, 2, 4, 2, 1, 0}, {2, 2, 4, 2, 16, Swizzle(3)},
	} {
		if _, err := NewTile(tc.rows, tc.cols, tc.elem, WithStride(tc.stride), WithAlignment(tc.align), WithSwizzle(tc.sw)); err == nil {
			t.Fatalf("accepted %+v", tc)
		}
	}
	tile, _ := NewTile(3, 5, 2, WithStride(8))
	if _, _, err := tile.OwnerFor(31, math.MaxInt); err == nil {
		t.Fatal("overflowing register accepted")
	}
	if _, _, err := tile.OwnerFor(5, 0); err == nil {
		t.Fatal("padding register accepted")
	}
	// Treating an element stride as bytes produces a different address.
	p, _ := tile.Index(2, 3)
	if p.ByteOffset == 2*tile.StrideElements()+3*tile.ElementBytes() {
		t.Fatal("byte/element mutant escaped")
	}
}

func TestTransposeRegisterPermutation(t *testing.T) {
	for _, dims := range [][2]int{{3, 5}, {4, 40}, {33, 17}} {
		rows, cols := dims[0], dims[1]
		tile, err := NewTile(rows, cols, 2, WithStride(cols+3))
		if err != nil {
			t.Fatal(err)
		}
		values := make(map[RegisterOwner]int)
		for row := 0; row < rows; row++ {
			for col := 0; col < cols; col++ {
				src, _ := tile.Owner(row, col)
				values[src] = row*cols + col + 1
			}
		}
		out := make([]int, rows*cols)
		for row := 0; row < rows; row++ {
			for col := 0; col < cols; col++ {
				src, _ := tile.Owner(row, col)
				dst, err := tile.TransposeOwners(row, col)
				if err != nil {
					t.Fatal(err)
				}
				index := dst.Register*32 + dst.Lane
				if index >= len(out) || out[index] != 0 {
					t.Fatal("invalid redistribution")
				}
				out[index] = values[src]
			}
		}
		for i, v := range out {
			want := (i%rows)*cols + i/rows + 1
			if v != want {
				t.Fatalf("transpose[%d]=%d want %d", i, v, want)
			}
		}
		out[0], out[1] = out[1], out[0]
		if out[0] == 1 {
			t.Fatal("wrong-lane mutant escaped")
		}
	}
}

func TestGeneratedWideIndex(t *testing.T) {
	if strconv.IntSize < 64 {
		t.Skip("large shape requires 64-bit host")
	}
	tile, err := NewTile(1<<20, 1<<12, 4)
	if err != nil {
		t.Fatal(err)
	}
	source, err := FixtureSource(tile)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(source, "4294967296ULL") || !strings.Contains(source, "(unsigned long long)blockIdx.x") {
		t.Fatal("generated arithmetic narrowed to 32 bits")
	}
}

func FuzzLayoutPermutation(f *testing.F) {
	f.Add(uint8(3), uint8(17), uint8(2), uint8(3))
	f.Fuzz(func(t *testing.T, r, c, e, s uint8) {
		elem := 1 << uint(e%3)
		span := []Swizzle{0, 32, 64, 128}[s%4]
		stride := 256 / elem
		cols := int(c)%stride + 1
		rows := int(r) + 1
		tile, err := NewTile(rows, cols, elem, WithStride(stride), WithSwizzle(span))
		if err != nil {
			t.Fatal(err)
		}
		row, col := rows-1, cols-1
		p, err := tile.Index(row, col)
		if err != nil {
			t.Fatal(err)
		}
		got := (p.ByteOffset / elem) % stride
		if span != 0 {
			got ^= row % (int(span) / elem)
		}
		if got != col || p.ByteOffset/(stride*elem) != row {
			t.Fatal("mapping failed inverse")
		}
	})
}
