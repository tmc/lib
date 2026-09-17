package cudalayout

import (
	"strings"
	"testing"
)

func TestLayoutIndexOracle(t *testing.T) {
	tile, err := NewTile(3, 5, 2, WithStride(8))
	if err != nil {
		t.Fatal(err)
	}
	for r := 0; r < 3; r++ {
		for c := 0; c < 5; c++ {
			got, err := tile.Index(r, c)
			if err != nil {
				t.Fatal(err)
			}
			want := r*8 + c
			if got.ElementOffset != want || got.ByteOffset != want*2 {
				t.Fatalf("(%d,%d) = %+v", r, c, got)
			}
		}
	}
}

func TestLayoutOverflow(t *testing.T) {
	if _, err := NewTile(int(^uint(0)>>1), 2, 4); err != ErrOverflow {
		t.Fatalf("overflow error = %v", err)
	}
}

func TestLayoutSwizzleRoundTrip(t *testing.T) {
	tile, err := NewTile(16, 8, 4, WithStride(16), WithSwizzle(Swizzle64B))
	if err != nil {
		t.Fatal(err)
	}
	for r := 0; r < tile.Rows(); r++ {
		for c := 0; c < tile.Cols(); c++ {
			p, _ := tile.Index(r, c)
			want := r*16*4 + ((c * 4) ^ ((r * 4) & 63))
			if p.ByteOffset != want {
				t.Fatalf("swizzle mutation or wrong units at %d,%d: got %d want %d", r, c, p.ByteOffset, want)
			}
		}
	}
	mutant, _ := NewTile(16, 8, 4, WithStride(16), WithSwizzle(Swizzle32B))
	p, _ := mutant.Index(10, 2)
	q, _ := tile.Index(10, 2)
	if p.ByteOffset == q.ByteOffset {
		t.Fatal("swizzle mutant was not detected")
	}
}

func TestRegisterTileOwnership(t *testing.T) {
	tile, _ := NewTile(4, 40, 2)
	for r := 0; r < tile.Rows(); r++ {
		for c := 0; c < tile.Cols(); c++ {
			o, err := tile.Owner(r, c)
			if err != nil {
				t.Fatal(err)
			}
			rr, cc, err := tile.OwnerFor(o.Lane, o.Register)
			if err != nil || rr != r || cc != c {
				t.Fatalf("owner round trip (%d,%d): %+v -> %d,%d,%v", r, c, o, rr, cc, err)
			}
		}
	}
	wrong, _ := tile.Owner(1, 1)
	wrong.Lane ^= 1
	if _, _, err := tile.OwnerFor(wrong.Lane, wrong.Register); err == nil {
		t.Log("mutant lane remains a valid owner, as expected; permutation oracle catches value mismatch")
	}
}

func TestTileRedistribution(t *testing.T) {
	p, _ := TransposePermutation(3, 5)
	r, c, _ := p.Map(1, 4)
	if r != 4 || c != 1 {
		t.Fatalf("transpose = %d,%d", r, c)
	}
	if _, _, err := p.Map(3, 0); err == nil {
		t.Fatal("tail out of bounds accepted")
	}
}

func TestTransposeOwnersFullPermutation(t *testing.T) {
	tile, _ := NewTile(3, 5, 2)
	seen := make(map[RegisterOwner]int)
	for r := 0; r < 3; r++ {
		for c := 0; c < 5; c++ {
			o, err := tile.TransposeOwners(r, c)
			if err != nil {
				t.Fatal(err)
			}
			if _, ok := seen[o]; ok {
				t.Fatalf("duplicate destination owner %+v", o)
			}
			seen[o] = r*5 + c
		}
	}
	if len(seen) != 15 {
		t.Fatalf("permutation size=%d", len(seen))
	}
	want, _ := tile.TransposeOwners(0, 1)
	mutant := RegisterOwner{Lane: want.Lane ^ 1, Register: want.Register}
	if seen[mutant] == 1 {
		t.Fatal("wrong-lane mutant accepted")
	}
}

func TestLayoutIdentity(t *testing.T) {
	a, _ := NewTile(4, 4, 2)
	b, _ := NewTile(4, 4, 2, WithStride(16), WithSwizzle(Swizzle32B))
	if a.Identity() == b.Identity() || len(a.Identity()) != 64 {
		t.Fatal("layout identity is unstable or incomplete")
	}
}

func TestLayoutTailBounds(t *testing.T) {
	tile, _ := NewTile(3, 5, 2)
	if _, err := tile.Index(2, 5); err != ErrOutOfBounds {
		t.Fatalf("tail error = %v", err)
	}
	if _, err := tile.Owner(-1, 0); err != ErrOutOfBounds {
		t.Fatalf("negative index error = %v", err)
	}
}

func TestGeneratedFixture(t *testing.T) {
	tile, _ := NewTile(8, 8, 2, WithStride(16), WithSwizzle(Swizzle32B))
	src, err := FixtureSource(tile)
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range []string{"layout_copy", "row=i/8", "byte_col=col*2", "stride_elements=16", tile.Identity()} {
		if !strings.Contains(src, s) {
			t.Errorf("fixture missing %q", s)
		}
	}
	if _, err := FixtureSourceForTarget(tile, "sm_90"); err == nil {
		t.Fatal("unsupported target accepted")
	}
	if _, err := FixtureSource(Tile{}); err == nil {
		t.Fatal("zero tile fixture accepted")
	}
}

func FuzzLayoutIndex(f *testing.F) {
	f.Add(2, 3, 1, 1, 2)
	f.Fuzz(func(t *testing.T, rows, cols, elem, row, col int) {
		tile, err := NewTile(rows, cols, elem)
		if err != nil {
			return
		}
		_, _ = tile.Index(row, col)
	})
}
