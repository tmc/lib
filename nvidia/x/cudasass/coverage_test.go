package cudasass_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

// TestSASS_EdgeCoverage_HashCollisionAndUniqueness checks EdgeIndex against
// values computed by hand from cuFuzz's inject_funcs.cu:58-59,
//
//	new_bb_id = (MAP_SIZE/2) + ((prev_cov_bb[tid] ^ bb_id) % (MAP_SIZE/2));
//	prev_cov_bb[tid] = bb_id >> 1;
//
// with MAP_SIZE 65536.
func TestSASS_EdgeCoverage_HashCollisionAndUniqueness(t *testing.T) {
	const a, b = 5, 9
	// A then B: 32768+(0^5) = 32773, prev 5>>1 = 2; 32768+(2^9) = 32779.
	// B then A: 32768+(0^9) = 32777, prev 9>>1 = 4; 32768+(4^5) = 32769.
	// Without the shift both orders end at 32768+(5^9) = 32780.
	ab1, next := cudasass.EdgeIndex(0, a)
	ab2, _ := cudasass.EdgeIndex(next, b)
	ba1, next := cudasass.EdgeIndex(0, b)
	ba2, _ := cudasass.EdgeIndex(next, a)
	for _, c := range []struct {
		name      string
		got, want int
	}{
		{"entry A", ab1, 32773},
		{"A→B", ab2, 32779},
		{"entry B", ba1, 32777},
		{"B→A", ba2, 32769},
	} {
		if c.got != c.want {
			t.Errorf("%s: index %d, want %d", c.name, c.got, c.want)
		}
	}
	if ab2 == ba2 {
		t.Errorf("A→B and B→A share index %d; the >>1 shift is what separates them", ab2)
	}

	// A negative ID sign-extends, as the C int does.
	if i, next := cudasass.EdgeIndex(0, -1); i != 65535 || next != ^uint64(0) {
		t.Errorf("EdgeIndex(0, -1) = %d, %#x; want 65535, 0xffffffffffffffff", i, next)
	}

	// The map is half the ID space: IDs 32768 apart collide from the same
	// location. The hash does not promise uniqueness, and a test that
	// assumed it would be testing something cuFuzz does not do.
	i1, _ := cudasass.EdgeIndex(0, 5)
	i2, _ := cudasass.EdgeIndex(0, 5+32768)
	if i1 != i2 {
		t.Errorf("blocks 5 and 32773 from location 0: %d and %d, want a collision", i1, i2)
	}
	for id := int32(-40000); id < 40000; id += 7 {
		for _, prev := range []uint64{0, 1, 0x7fff, ^uint64(0)} {
			if i, _ := cudasass.EdgeIndex(prev, id); i < 32768 || i >= cudasass.CoverageMapSize {
				t.Fatalf("EdgeIndex(%#x, %d) = %d, outside the upper half", prev, id, i)
			}
		}
	}

	// Per thread, the same path counts the same edges however threads
	// interleave.
	var c cudasass.EdgeCoverage
	c.Record(1, a)
	c.Record(2, b)
	c.Record(1, b)
	c.Record(2, a)
	for _, w := range []struct{ index, count int }{{32773, 1}, {32779, 1}, {32777, 1}, {32769, 1}, {32780, 0}} {
		if got := c.Count(w.index); int(got) != w.count {
			t.Errorf("Count(%d) = %d, want %d", w.index, got, w.count)
		}
	}

	// A warp counts its first lane's edge once, and every lane moves on.
	var warp cudasass.EdgeCoverage
	if i := warp.RecordWarp([]uint64{10, 11, 12}, a); i != 32773 || warp.Count(32773) != 1 {
		t.Errorf("RecordWarp entry: index %d count %d, want 32773 once", i, warp.Count(32773))
	}
	if i := warp.Record(12, b); i != 32779 {
		t.Errorf("lane 12 after the warp entered A: index %d, want 32779 (its prev advanced)", i)
	}
	if i := warp.RecordWarp(nil, a); i != -1 {
		t.Errorf("RecordWarp(nil) = %d, want -1", i)
	}
}

func TestEdgeCoverageBitmap(t *testing.T) {
	tests := []struct {
		count int
		want  byte
	}{
		{0, 0}, {1, 1}, {2, 2}, {3, 3}, {511, 3}, {512, 4}, {4095, 4},
		{4096, 5}, {16383, 5}, {16384, 6}, {65535, 6}, {65536, 7}, {70000, 7},
	}
	for _, tt := range tests {
		var c cudasass.EdgeCoverage
		var index int
		for range tt.count {
			index = c.Record(0, 0)
			c.Record(0, 1) // back to location 0: 1>>1 is 0
		}
		if tt.count == 0 {
			index, _ = cudasass.EdgeIndex(0, 0)
		}
		b := c.Bitmap()
		if len(b) != cudasass.CoverageMapSize {
			t.Fatalf("Bitmap has %d bytes", len(b))
		}
		if b[index] != tt.want {
			t.Errorf("count %d: bucket %d, want %d", tt.count, b[index], tt.want)
		}
	}
}
