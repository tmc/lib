package egraph

import (
	"math/bits"
	"regexp"
	"strconv"
	"testing"
)

// TestIQ2XXSGridIsDequantCodebook checks the structural invariants that
// distinguish ggml's dequantization codebook from llama.cpp's quantizer-side
// search grid. The two tables have the same name-shape and the same length, so
// a value oracle that decodes with the table under test cannot tell them apart;
// these properties can.
//
// The dequant codebook packs eight bytes drawn from {8, 25, 43, 60} and its
// entries are strictly ascending. The quantizer grid packs eight 2-bit
// selectors, so its entries contain bytes outside that set.
func TestIQ2XXSGridIsDequantCodebook(t *testing.T) {
	allowed := map[byte]bool{8: true, 25: true, 43: true, 60: true}
	for i, entry := range iq2xxsGrid {
		for j := 0; j < 8; j++ {
			b := byte(entry >> (8 * j))
			if !allowed[b] {
				t.Errorf("iq2xxsGrid[%d] byte %d = %d, want one of {8, 25, 43, 60}", i, j, b)
			}
		}
		if i > 0 && entry <= iq2xxsGrid[i-1] {
			t.Errorf("iq2xxsGrid[%d] = %#016x, not greater than [%d] = %#016x", i, entry, i-1, iq2xxsGrid[i-1])
		}
	}
}

// TestIQ2XXSSignTableParity checks ksignsIQ2XS against its definition: the low
// seven bits are the index itself and bit 7 is the parity that makes the
// popcount of each entry even.
func TestIQ2XXSSignTableParity(t *testing.T) {
	for i, s := range ksignsIQ2XS {
		if int(s&0x7f) != i {
			t.Errorf("ksignsIQ2XS[%d] low bits = %d, want %d", i, s&0x7f, i)
		}
		if bits.OnesCount8(s)%2 != 0 {
			t.Errorf("ksignsIQ2XS[%d] = %d has odd popcount, want even parity", i, s)
		}
	}
}

// TestIQ2XXSTablesCUDARoundTrip parses the emitted device tables back and
// compares them to the Go tables they were generated from, so a formatting
// defect in the emitter cannot silently ship a different codebook to the GPU.
func TestIQ2XXSTablesCUDARoundTrip(t *testing.T) {
	src := iq2xxsTablesCUDA()

	grid := regexp.MustCompile(`0x([0-9a-f]{16})ULL`).FindAllStringSubmatch(src, -1)
	if len(grid) != len(iq2xxsGrid) {
		t.Fatalf("emitted %d grid entries, want %d", len(grid), len(iq2xxsGrid))
	}
	for i, m := range grid {
		v, err := strconv.ParseUint(m[1], 16, 64)
		if err != nil {
			t.Fatalf("grid entry %d: %v", i, err)
		}
		if v != iq2xxsGrid[i] {
			t.Errorf("emitted grid[%d] = %#016x, want %#016x", i, v, iq2xxsGrid[i])
		}
	}

	signsBlock := regexp.MustCompile(`(?s)ksigns_iq2xs\[128\] = \{(.*?)\}`).FindStringSubmatch(src)
	if signsBlock == nil {
		t.Fatal("emitted source has no ksigns_iq2xs table")
	}
	signs := regexp.MustCompile(`\d+`).FindAllString(signsBlock[1], -1)
	if len(signs) != len(ksignsIQ2XS) {
		t.Fatalf("emitted %d sign entries, want %d", len(signs), len(ksignsIQ2XS))
	}
	for i, s := range signs {
		v, err := strconv.ParseUint(s, 10, 8)
		if err != nil {
			t.Fatalf("sign entry %d: %v", i, err)
		}
		if uint8(v) != ksignsIQ2XS[i] {
			t.Errorf("emitted ksigns[%d] = %d, want %d", i, v, ksignsIQ2XS[i])
		}
	}
}
