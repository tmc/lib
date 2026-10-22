package dsv4cpu

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestPinnedDenseQ80(t *testing.T) {
	s := string(pinnedCPUOracle(t))
	extract := func(name string) string {
		i := strings.Index(s, name+"(")
		if i < 0 {
			t.Fatal(name)
		}
		start := strings.LastIndex(s[:i], "\n") + 1
		brace := strings.Index(s[i:], "{") + i
		depth := 0
		for end := brace; end < len(s); end++ {
			switch s[end] {
			case '{':
				depth++
			case '}':
				depth--
				if depth == 0 {
					return s[start : end+1]
				}
			}
		}
		t.Fatal("unterminated function")
		return ""
	}
	c := "#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n#include <math.h>\n#undef __ARM_NEON\nstatic float f16_to_f32(uint16_t h){_Float16 v;memcpy(&v,&h,2);return (float)v;}\n"
	for _, name := range []string{"dot_i8_32", "dot_q8_0_row", "quantize_q8_0_activation"} {
		c += extract(name) + "\n"
	}
	c += `int main(void){float x[4096],scale[128];int8_t q[4096];uint8_t row[128*34];for(int i=0;i<4096;i++)x[i]=(float)(i%97-48)/17;quantize_q8_0_activation(x,q,scale,4096);fwrite(q,1,4096,stdout);fwrite(scale,4,128,stdout);for(int r=0;r<8;r++){for(int b=0;b<128;b++){uint16_t d=0x3800;memcpy(row+b*34,&d,2);for(int j=0;j<32;j++)row[b*34+2+j]=(uint8_t)((r*11+b*32+j)%251-125);}float dot=dot_q8_0_row(row,q,scale,4096,128);fwrite(&dot,4,1,stdout);}return 0;}`
	dir := t.TempDir()
	file, bin := filepath.Join(dir, "dense.c"), filepath.Join(dir, "dense")
	if err := os.WriteFile(file, []byte(c), 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("cc", "-O0", "-ffp-contract=off", file, "-lm", "-o", bin).CombinedOutput(); err != nil {
		t.Fatalf("compile %v %s", err, out)
	}
	raw, err := exec.Command(bin).Output()
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) != 4096+128*4+8*4 {
		t.Fatal("short dense oracle")
	}
	x := make([]float32, 4096)
	for i := range x {
		x[i] = float32(i%97-48) / 17
	}
	a := make([]Q80Activation, 128)
	if err := QuantizeQ80(a, x); err != nil {
		t.Fatal(err)
	}
	for b, v := range a {
		for j, q := range v.Values {
			if q != int8(raw[b*32+j]) {
				t.Fatalf("quant block%d value%d", b, j)
			}
		}
		if math.Float32bits(v.Scale) != binary.LittleEndian.Uint32(raw[4096+b*4:]) {
			t.Fatalf("quant scale%d", b)
		}
	}
	packed := make([]byte, 8*128*34)
	for r := 0; r < 8; r++ {
		for b := 0; b < 128; b++ {
			p := packed[(r*128+b)*34:]
			binary.LittleEndian.PutUint16(p, 0x3800)
			for j := 0; j < 32; j++ {
				p[2+j] = byte(int8((r*11+b*32+j)%251 - 125))
			}
		}
	}
	m, err := NewMatrix(bytes.NewReader(packed), 0, int64(len(packed)), 8, 4096, Q80)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, 8)
	if err := m.MulQ80(out, a); err != nil {
		t.Fatal(err)
	}
	for r, v := range out {
		want := math.Float32frombits(binary.LittleEndian.Uint32(raw[4096+128*4+r*4:]))
		if math.Float32bits(v) != math.Float32bits(want) {
			t.Fatalf("dense row%d got%g want%g", r, v, want)
		}
	}
}

func TestDenseDomain(t *testing.T) {
	x := make([]float32, 32)
	a := make([]Q80Activation, 1)
	for _, v := range []float32{math.SmallestNonzeroFloat32, float32(math.Inf(1)), float32(math.NaN())} {
		x[0] = v
		if QuantizeQ80(a, x) == nil {
			t.Fatalf("accepted %g", v)
		}
	}
	x[0] = 0
	if err := QuantizeQ80(a, x); err != nil || a[0] != (Q80Activation{}) {
		t.Fatalf("zero block: %v %v", a, err)
	}
	if err := QuantizeQ80(a, x[:31]); err == nil {
		t.Fatal("short block accepted")
	}
	var m *Matrix
	if m.MulQ80(nil, a) == nil {
		t.Fatal("nil matrix accepted")
	}
}

func ExampleQuantizeQ80() {
	x := make([]float32, 32)
	x[0] = 1
	a := make([]Q80Activation, 1)
	_ = QuantizeQ80(a, x)
	fmt.Println(a[0].Values[0])
	// Output: 127
}

func ExampleQ80Activation() {
	a := Q80Activation{Scale: 0.5}
	a.Values[0] = 2
	fmt.Println(a.Scale * float32(a.Values[0]))
	// Output: 1
}

func ExampleMatrix_MulQ80() {
	row := make([]byte, 34)
	binary.LittleEndian.PutUint16(row, 0x3c00)
	row[2] = 3
	m, _ := NewMatrix(bytes.NewReader(row), 0, 34, 1, 32, Q80)
	a := []Q80Activation{{Scale: 0.5}}
	a[0].Values[0] = 2
	out := make([]float32, 1)
	_ = m.MulQ80(out, a)
	fmt.Println(out)
	// Output: [3]
}
