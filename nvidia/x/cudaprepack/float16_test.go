package cudaprepack

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestFloat16Rounding(t *testing.T) {
	for _, tc := range []struct {
		name string
		bits uint32
		want uint16
	}{
		{"zero", 0, 0}, {"negative zero", 0x80000000, 0x8000},
		{"even tie", 0x3f801000, 0x3c00}, {"odd tie", 0x3f803000, 0x3c02},
		{"exponent carry", 0x3ffff000, 0x4000}, {"negative carry", 0xbffff000, 0xc000},
		{"smallest half", 0x33800000, 1}, {"underflow tie", 0x33000000, 0},
		{"above underflow tie", 0x33000001, 1}, {"subnormal tie", 0x33c00000, 2},
		{"normal transition", 0x387fe000, 0x0400},
		{"largest finite", 0x477fe000, 0x7bff}, {"overflow tie", 0x477ff000, 0x7c00},
		{"infinity", 0x7f800000, 0x7c00}, {"negative infinity", 0xff800000, 0xfc00},
		{"quiet nan", 0x7fc00000, 0x7e00}, {"signaling nan", 0x7f800001, 0x7e00},
		{"negative payload", 0xffcaa000, 0xfe55},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := Float32ToFloat16(math.Float32frombits(tc.bits)); got != tc.want {
				t.Fatalf("%08x -> %04x want %04x", tc.bits, got, tc.want)
			}
		})
	}
}

// The host compiler supplies an independent IEEE binary16 rounding oracle.
// This is CPU conversion evidence, not CUDA/native-kernel qualification.
func TestFloat16CompilerOracle(t *testing.T) {
	compiler, err := exec.LookPath("clang++")
	if err != nil {
		t.Skip("requires host clang++ with _Float16")
	}
	dir := t.TempDir()
	source := `#include <stdint.h>
#include <stdio.h>
#include <string.h>
int main() {
  unsigned char in[4];
  while (fread(in,1,4,stdin)==4) {
    uint32_t bits=(uint32_t)in[0]|(uint32_t)in[1]<<8|(uint32_t)in[2]<<16|(uint32_t)in[3]<<24;
    float value; memcpy(&value,&bits,4);
    _Float16 half=(_Float16)value;
    uint16_t result; memcpy(&result,&half,2);
    unsigned char out[2]={(unsigned char)result,(unsigned char)(result>>8)};
    if(fwrite(out,1,2,stdout)!=2)return 1;
  }
  return ferror(stdin)?2:0;
}
`
	path := filepath.Join(dir, "half.cc")
	if err := os.WriteFile(path, []byte(source), 0600); err != nil {
		t.Fatal(err)
	}
	binaryPath := filepath.Join(dir, "half")
	if out, err := exec.Command(compiler, "-std=c++17", "-O1", path, "-o", binaryPath).CombinedOutput(); err != nil {
		t.Fatalf("compile: %v\n%s", err, out)
	}
	var words []uint32
	seed := uint32(0x6a09e667)
	for len(words) < 65536 {
		seed ^= seed << 13
		seed ^= seed >> 17
		seed ^= seed << 5
		if seed&0x7f800000 != 0x7f800000 {
			words = append(words, seed)
		}
	}
	input := make([]byte, len(words)*4)
	for i, bits := range words {
		binary.LittleEndian.PutUint32(input[i*4:], bits)
	}
	command := exec.Command(binaryPath)
	command.Stdin = bytes.NewReader(input)
	out, err := command.Output()
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != len(words)*2 {
		t.Fatalf("oracle returned %d bytes", len(out))
	}
	for i, bits := range words {
		want := binary.LittleEndian.Uint16(out[i*2:])
		got := Float32ToFloat16(math.Float32frombits(bits))
		if got != want {
			t.Fatalf("%08x -> %04x, compiler wants %04x", bits, got, want)
		}
	}
}

func TestFloat16RoundTripAndMidpoints(t *testing.T) {
	// Every finite binary16 value is exact in binary32. Adjacent binary16
	// midpoints are also exact, so parity gives a rounding oracle directly.
	for h := uint32(0); h < 0x7c00; h++ {
		for _, sign := range []uint16{0, 0x8000} {
			bits := uint16(h) | sign
			if got := Float32ToFloat16(Float16ToFloat32(bits)); got != bits {
				t.Fatalf("round trip %04x -> %04x", bits, got)
			}
		}
		if h == 0x7bff {
			continue
		}
		a, b := Float16ToFloat32(uint16(h)), Float16ToFloat32(uint16(h+1))
		mid := (a + b) / 2
		want := uint16(h)
		if h&1 != 0 {
			want++
		}
		for _, negative := range []bool{false, true} {
			value, expected := mid, want
			if negative {
				value = -value
				expected |= 0x8000
			}
			if got := Float32ToFloat16(value); got != expected {
				t.Fatalf("midpoint %04x/%04x negative=%v -> %04x want %04x", h, h+1, negative, got, expected)
			}
		}
		if got := Float32ToFloat16(math.Nextafter32(mid, a)); got != uint16(h) {
			t.Fatalf("below midpoint %04x -> %04x", h, got)
		}
		if got := Float32ToFloat16(math.Nextafter32(mid, b)); got != uint16(h+1) {
			t.Fatalf("above midpoint %04x -> %04x", h, got)
		}
	}
}
