//go:build darwin && arm64

package main

import (
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestMetalMXFP4SourceCompile(t *testing.T) {
	tool, err := exec.Command("xcrun", "--find", "metal").Output()
	if err != nil {
		t.Skip("Metal source compiler unavailable")
	}
	dir := t.TempDir()
	source := filepath.Join(dir, "selected.metal")
	if err := os.WriteFile(source, []byte(metalMXFP4ExpertsMSL), 0600); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(strings.TrimSpace(string(tool)), "-c", source, "-o", filepath.Join(dir, "selected.air"))
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("compile: %v\n%s", err, output)
	}
}

// Decode by coordinate rather than the kernel's lane traversal, accumulating
// in float64. E8M0 zero encodes 2^-127, not zero.
func metalMXFP4Reference(weights []byte, x []float32) float64 {
	magnitude := [...]float64{0, .5, 1, 1.5, 2, 3, 4, 6}
	var sum float64
	for i, v := range x {
		block, coordinate := i/32, i%32
		p := weights[block*17:]
		exponent := int(p[0]) - 127
		scale := math.Ldexp(1, exponent)
		if p[0] == 255 {
			scale = math.Inf(1)
		}
		code := p[1+coordinate%16] >> uint((coordinate/16)*4) & 15
		value := magnitude[code&7]
		if code&8 != 0 {
			value = -value
		}
		sum += scale * value * float64(v)
	}
	return sum
}

func TestMetalMXFP4Reference(t *testing.T) {
	raw := make([]byte, 34)
	raw[0], raw[17] = 127, 0
	for i := 1; i < 17; i++ {
		raw[i] = 0xf1
		raw[17+i] = 0x2a
	}
	x := make([]float32, 64)
	for i := range x {
		x[i] = 1
	}
	want := -88.0
	if got := metalMXFP4Reference(raw, x); got != want {
		t.Fatalf("dot=%g, want %g", got, want)
	}
	small := make([]float32, 32)
	for i := range small {
		small[i] = 1
	}
	// The signed pair -1,+1 cancels. Change just one nibble to expose zero scale.
	raw[18] = 0x3a
	if got := metalMXFP4Reference(raw[17:], small); got != math.Ldexp(.5, -127) {
		t.Fatalf("zero scale=%g", got)
	}
	raw[0] = 128
	if metalMXFP4Reference(raw, x) == want {
		t.Fatal("scale mutant invisible")
	}
	raw[1] ^= 0x80
	if metalMXFP4Reference(raw[:17], small) == -176 {
		t.Fatal("nibble sign mutant invisible")
	}
}

func TestMetalMXFP4SelectedSource(t *testing.T) {
	for _, site := range []string{"ids[slot]", "uint64_t(expert)*rows+row", "uint64_t(slot)*cols", "0x00400000u", "p[1+(lane&15)]", "((lane>>4)*4)", "if(code&8)v=-v"} {
		if strings.Count(metalMXFP4ExpertsMSL, site) != 1 {
			t.Fatalf("missing unique source site %q", site)
		}
	}
	q := new(metalMXFP4Experts)
	q.Close()
	q.Close()
	if q.library != nil || q.pipeline != nil {
		t.Fatal("closed owner retained resources")
	}
	for _, dims := range [][]uint64{{32, 1, 256}, {64, 3, 256}} {
		// No weight allocation: validation must reject before encoding.
		tensor := boundMetalTensor{typeID: typeMXFP4, dimensions: dims}
		if checkMetalMXFP4Selected(tensor, 6, int(dims[0])*6, int(dims[1])*6, true) == nil {
			t.Fatal("missing weights admitted")
		}
	}
}

func TestMetalMXFP4InvalidScaleAndOwner(t *testing.T) {
	tooManyExperts := boundMetalTensor{typeID: typeMXFP4, dimensions: []uint64{32, 1, uint64(math.MaxInt32) + 1}}
	if err := checkMetalMXFP4Selected(tooManyExperts, 1, 32, 1, false); err == nil || !strings.Contains(err.Error(), "geometry") {
		t.Fatalf("unreachable signed expert ID extent admitted: %v", err)
	}
	raw := make([]byte, 17)
	raw[0] = 255
	x := make([]float32, 32)
	// Pinned C Metal arithmetic maps exponent 255 to +Inf; multiplication
	// by zero makes this fixture NaN. The finite-state guard must see it.
	if got := metalMXFP4Reference(raw, x); !math.IsNaN(got) {
		t.Fatalf("invalid E8M0 scale produced %g", got)
	}
	if strings.Count(metalMXFP4ExpertsMSL, "uint(p[0])<<23") != 1 {
		t.Fatal("pinned Metal exponent conversion absent")
	}
	if q, err := newMetalMXFP4Experts(nil, metalMXFP4ExpertsMSL); err == nil || q != nil {
		t.Fatal("nil device admitted")
	}
	var q *metalMXFP4Experts
	q.Close()
}

func TestMetalMXFP4SelectedIndexReference(t *testing.T) {
	const cols, rows, experts = 64, 3, 256
	raw := make([]byte, experts*rows*34)
	for expert := 0; expert < experts; expert++ {
		for row := 0; row < rows; row++ {
			lo := (expert*rows + row) * 34
			for block := 0; block < 2; block++ {
				raw[lo+block*17] = 127
				for j := 1; j < 17; j++ {
					raw[lo+block*17+j] = byte((expert+row+j)%8) | byte(8+(row+j)%8)<<4
				}
			}
		}
	}
	ids := []int{255, 1, 255, 0, 128, 1}
	x := make([]float32, cols*len(ids))
	for i := range x {
		x[i] = float32(i%17-8) / 8
	}
	for slot, id := range ids {
		for row := 0; row < rows; row++ {
			lo := (id*rows + row) * 34
			got := metalMXFP4Reference(raw[lo:lo+34], x[slot*cols:(slot+1)*cols])
			// Separate scalar traversal with a switch decoder, not table lookup.
			var want float64
			for i := 0; i < cols; i++ {
				b := raw[lo+(i/32)*17+1+i%16]
				code := b >> uint((i%32/16)*4) & 15
				var v float64
				switch code & 7 {
				case 1:
					v = .5
				case 2:
					v = 1
				case 3:
					v = 1.5
				case 4:
					v = 2
				case 5:
					v = 3
				case 6:
					v = 4
				case 7:
					v = 6
				}
				if code&8 != 0 {
					v = -v
				}
				want += v * float64(x[slot*cols+i])
			}
			if got != want {
				t.Fatalf("slot %d expert %d row %d: %g != %g", slot, id, row, got, want)
			}
		}
	}
}
