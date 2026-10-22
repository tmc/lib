package dsv4cpu

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func ExampleQuantizeQ8K() {
	x := make([]float32, 256)
	x[0] = 1
	out := make([]Q8K, 1)
	_ = QuantizeQ8K(out, x)
	fmt.Println(out[0].Values[0], out[0].Sums[0])
	// Output: -127 -127
}

func ExampleSwiGLU() {
	out := make([]float32, 1)
	_ = SwiGLU(out, []float32{0}, []float32{2}, 10)
	fmt.Println(out)
	// Output: [0]
}

func TestQ8KEdges(t *testing.T) {
	x := make([]float32, 256)
	out := make([]Q8K, 1)
	if err := QuantizeQ8K(out, x); err != nil || out[0].Scale != 0 {
		t.Fatalf("zero: %v", err)
	}
	x[0] = -127
	x[1] = 127
	x[2] = .5
	x[3] = 1.5
	x[4] = 2.5
	if err := QuantizeQ8K(out, x); err != nil {
		t.Fatal(err)
	}
	if out[0].Scale != 1 || out[0].Values[0] != -127 || out[0].Values[1] != 127 || out[0].Values[2] != 0 || out[0].Values[3] != 2 || out[0].Values[4] != 2 {
		t.Fatal("signed amax/ties")
	}
	clear(x)
	x[0] = math.SmallestNonzeroFloat32
	if err := QuantizeQ8K(out, x); err == nil {
		t.Fatal("non-representable scale accepted")
	}
}

func TestPinnedActivations(t *testing.T) {
	s := string(pinnedCPUOracle(t))
	extract := func(name string) string {
		i := strings.Index(s, name+"(")
		if i < 0 {
			t.Fatal(name)
		}
		// silu has a forward declaration: start at its definition.
		if name == "silu" {
			i = strings.Index(s, "static float silu(float x) {") + len("static float ")
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
	c := "#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#include <math.h>\n#define QK_K 256\ntypedef struct {float d;int8_t qs[256];int16_t bsums[16];} block_q8_K;\nstatic void ds4_die(const char *s){abort();}\n"
	for _, name := range []string{"ds4_quantize_row_q8_K", "sigmoid_stable", "silu", "swiglu"} {
		c += extract(name) + "\n"
	}
	c += `int main(void){float x[4096],gate[2048],up[2048],out[2048];block_q8_K q[16];for(int i=0;i<4096;i++)x[i]=(float)(i%257-128)/11;for(int i=0;i<2048;i++){gate[i]=(float)(i%67-33);up[i]=(float)(i%59-29);}ds4_quantize_row_q8_K(x,q,4096);for(int b=0;b<16;b++){fwrite(&q[b].d,4,1,stdout);fwrite(q[b].qs,1,256,stdout);fwrite(q[b].bsums,2,16,stdout);}swiglu(out,gate,up,2048,10);fwrite(out,4,2048,stdout);return 0;}`
	dir := t.TempDir()
	file, bin := filepath.Join(dir, "activation.c"), filepath.Join(dir, "activation")
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
	if len(raw) != 16*292+2048*4 {
		t.Fatal("short activation oracle")
	}
	x := make([]float32, 4096)
	for i := range x {
		x[i] = float32(i%257-128) / 11
	}
	q := make([]Q8K, 16)
	if err := QuantizeQ8K(q, x); err != nil {
		t.Fatal(err)
	}
	for b, v := range q {
		r := raw[b*292:]
		if math.Float32bits(v.Scale) != binary.LittleEndian.Uint32(r) {
			t.Fatalf("Q8 scale block%d", b)
		}
		for j, got := range v.Values {
			if got != int8(r[4+j]) {
				t.Fatalf("Q8 value block%d i%d", b, j)
			}
		}
		for j, got := range v.Sums {
			if got != int16(binary.LittleEndian.Uint16(r[260+j*2:])) {
				t.Fatalf("Q8 sums block%d i%d", b, j)
			}
		}
	}
	gate, up, out := make([]float32, 2048), make([]float32, 2048), make([]float32, 2048)
	for i := range gate {
		gate[i] = float32(i%67 - 33)
		up[i] = float32(i%59 - 29)
	}
	if err := SwiGLU(out, gate, up, 10); err != nil {
		t.Fatal(err)
	}
	for i, v := range out {
		want := math.Float32frombits(binary.LittleEndian.Uint32(raw[16*292+i*4:]))
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.Abs(float64(v-want)) > 1e-6+1e-6*math.Abs(float64(want)) {
			t.Fatalf("SwiGLU i%d got%g want%g", i, v, want)
		}
	}
}
