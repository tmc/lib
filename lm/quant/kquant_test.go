// SPDX-License-Identifier: Apache-2.0

package quant_test

import (
	"bufio"
	"encoding/hex"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/lm/quant"
)

// golden holds the parsed contents of testdata/ds4_golden.txt: the two raw
// super-blocks and the element values the upstream C oracle produces for them.
type golden struct {
	q2kRaw, iq2Raw []byte
	q2k, iq2       []float32
}

func readGolden(t *testing.T) golden {
	t.Helper()
	f, err := os.Open("testdata/ds4_golden.txt")
	if err != nil {
		t.Fatalf("open golden: %v", err)
	}
	defer f.Close()

	g := golden{q2k: make([]float32, 256), iq2: make([]float32, 256)}
	scan := bufio.NewScanner(f)
	for scan.Scan() {
		line := scan.Text()
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Fields(line)
		switch fields[0] {
		case "q2k_raw", "iq2_raw":
			raw, err := hex.DecodeString(fields[1])
			if err != nil {
				t.Fatalf("decode %s: %v", fields[0], err)
			}
			if fields[0] == "q2k_raw" {
				g.q2kRaw = raw
			} else {
				g.iq2Raw = raw
			}
		case "q2k", "iq2":
			i, err := strconv.Atoi(fields[1])
			if err != nil {
				t.Fatalf("parse index %q: %v", fields[1], err)
			}
			v, err := strconv.ParseFloat(fields[2], 32)
			if err != nil {
				t.Fatalf("parse value %q: %v", fields[2], err)
			}
			if fields[0] == "q2k" {
				g.q2k[i] = float32(v)
			} else {
				g.iq2[i] = float32(v)
			}
		default:
			t.Fatalf("unexpected golden record %q", fields[0])
		}
	}
	if err := scan.Err(); err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if len(g.q2kRaw) != 84 || len(g.iq2Raw) != 66 {
		t.Fatalf("golden block sizes = %d, %d; want 84, 66", len(g.q2kRaw), len(g.iq2Raw))
	}
	return g
}

// TestDequantizeQ2_KOracle checks the CPU dequantizer against values produced
// by q2_k_value_f32 (ds4.c:3608) in the pinned ds4 oracle.
func TestDequantizeQ2_KOracle(t *testing.T) {
	g := readGolden(t)
	got, err := quant.DequantizeQ2_K(g.q2kRaw, quant.SuperBlockSize)
	if err != nil {
		t.Fatalf("DequantizeQ2_K: %v", err)
	}
	for i, want := range g.q2k {
		if got[i] != want {
			t.Errorf("element %d = %v, want %v", i, got[i], want)
		}
	}
}

// TestDequantizeIQ2_XXSOracle checks the CPU dequantizer against values
// recovered from ds4_vec_dot_iq2_xxs_f32 (ds4.c:3950) in the pinned ds4 oracle.
func TestDequantizeIQ2_XXSOracle(t *testing.T) {
	g := readGolden(t)
	got, err := quant.DequantizeIQ2_XXS(g.iq2Raw, quant.SuperBlockSize)
	if err != nil {
		t.Fatalf("DequantizeIQ2_XXS: %v", err)
	}
	for i, want := range g.iq2 {
		// The oracle accumulates through a dot product, so allow a
		// single-ulp difference rather than requiring bit equality.
		if diff := math.Abs(float64(got[i] - want)); diff > 1e-6*(1+math.Abs(float64(want))) {
			t.Errorf("element %d = %v, want %v", i, got[i], want)
		}
	}
}

// TestDequantizeQ2_KLayout exercises the block layout with hand-built blocks
// whose expected values follow directly from q2_k_value_f32 (ds4.c:3608).
func TestDequantizeQ2_KLayout(t *testing.T) {
	// d = 1.0 (0x3C00), dmin = 1.0, every group scale byte = 0x21:
	// quant scale 1, minimum 2.
	block := make([]byte, 84)
	for i := range 16 {
		block[i] = 0x21
	}
	block[80], block[81] = 0x00, 0x3C
	block[82], block[83] = 0x00, 0x3C
	// qs byte 0 holds elements 0, 32, 64 and 96 in its four 2-bit fields:
	// groups 0, 2, 4 and 6 all read q_base 0, l 0 at shifts 0, 2, 4 and 6.
	block[16] = 0b11_10_01_00

	tests := []struct {
		name  string
		index int
		want  float32
	}{
		{"group0 quant 0", 0, 1*0 - 1*2},
		{"group2 quant 1", 32, 1*1 - 1*2},
		{"group4 quant 2", 64, 1*2 - 1*2},
		{"group6 quant 3", 96, 1*3 - 1*2},
		{"unset element", 1, 1*0 - 1*2},
	}

	got, err := quant.DequantizeQ2_K(block, quant.SuperBlockSize)
	if err != nil {
		t.Fatalf("DequantizeQ2_K: %v", err)
	}
	for _, tt := range tests {
		if got[tt.index] != tt.want {
			t.Errorf("%s: element %d = %v, want %v", tt.name, tt.index, got[tt.index], tt.want)
		}
	}
}

// TestDequantizeIQ2_XXSLayout builds a block that selects codebook entry 0 of
// iq2xxs_grid (ds4.c:973), which is 0x0808080808080808, with sign index 0 of
// ksigns_iq2xs (ds4.c:962), which is 0: all eight bytes positive.
func TestDequantizeIQ2_XXSLayout(t *testing.T) {
	block := make([]byte, 66)
	block[0], block[1] = 0x00, 0x3C // d = 1.0
	// Group 0: indices word = 0 (grid entry 0 four times), aux word top
	// nibble = 0, so scale = 0.125*1.0*(2*0+1) = 0.125.
	// Every other group is zero too, which is the same codebook entry.
	plain, err := quant.DequantizeIQ2_XXS(block, quant.SuperBlockSize)
	if err != nil {
		t.Fatalf("DequantizeIQ2_XXS: %v", err)
	}
	for i, v := range plain {
		if want := float32(0.125 * 8); v != want {
			t.Fatalf("element %d = %v, want %v", i, v, want)
		}
	}

	// Set group 0's scale nibble to 15: scale = 0.125*(2*15+1) = 3.875.
	block[2+7] = 0xF0
	got, err := quant.DequantizeIQ2_XXS(block, quant.SuperBlockSize)
	if err != nil {
		t.Fatalf("DequantizeIQ2_XXS: %v", err)
	}
	if want := float32(0.125 * 31 * 8); got[0] != want {
		t.Errorf("scaled element 0 = %v, want %v", got[0], want)
	}
	if want := float32(0.125 * 8); got[32] != want {
		t.Errorf("next group element 32 = %v, want %v", got[32], want)
	}
}

func TestDequantizeKQuantErrors(t *testing.T) {
	tests := []struct {
		name        string
		fn          func([]byte, int) ([]float32, error)
		data        []byte
		numElements int
	}{
		{"q2_k partial block", quant.DequantizeQ2_K, make([]byte, 84), 128},
		{"q2_k negative", quant.DequantizeQ2_K, make([]byte, 84), -256},
		{"q2_k short buffer", quant.DequantizeQ2_K, make([]byte, 83), 256},
		{"iq2_xxs partial block", quant.DequantizeIQ2_XXS, make([]byte, 66), 32},
		{"iq2_xxs negative", quant.DequantizeIQ2_XXS, make([]byte, 66), -256},
		{"iq2_xxs short buffer", quant.DequantizeIQ2_XXS, make([]byte, 65), 256},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := tt.fn(tt.data, tt.numElements); err == nil {
				t.Fatal("expected an error")
			}
		})
	}
}
