package hopper

import "testing"

// TestPrefillV27PackLayout compares each input bit with the two-word packing
// table in the retained Moe8 prefill v27 source. It checks the word layout,
// not the full weight, scale, routing, or kernel contract.
func TestPrefillV27PackLayout(t *testing.T) {
	bits := [8][8]uint{
		{4, 5, 6, 13, 20, 21, 22, 29},
		{0, 1, 2, 9, 16, 17, 18, 25},
		{26, 27, 28, 3, 10, 11, 12, 19},
		{34, 35, 36, 43, 50, 51, 52, 59},
		{60, 61, 62, 37, 44, 45, 46, 53},
		{56, 57, 58, 33, 40, 41, 42, 49},
		{30, 31, 48, 55, 14, 15, 32, 39},
		{38, 7, 8, 47, 54, 23, 24, 63},
	}
	for code := range 16 {
		for bit := range 4 {
			var in [2]uint32
			in[code/8] = 1 << uint(4*(code%8)+bit)
			got := PackDequant2Words(in[0], in[1])
			pos := bits[code/2][(code%2)*4+bit]
			var want [2]uint32
			want[pos/32] = 1 << (pos % 32)
			if got != want {
				t.Fatalf("code %d bit %d: got %08x, want %08x", code, bit, got, want)
			}
		}
	}
}
