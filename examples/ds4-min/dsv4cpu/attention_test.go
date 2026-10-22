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

func ExampleAttend() {
	out := make([]float32, 2)
	_ = Attend(out, []float32{0, 0}, []float32{2, 4}, []float32{0}, 2)
	fmt.Println(out)
	// Output: [1 2]
}

func TestPinnedAttentionOracle(t *testing.T) {
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
	c := `#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#undef __ARM_NEON
#define DS4_N_HEAD 64
#define DS4_N_HEAD_DIM 512
typedef struct {int unused;} ds4_model;
typedef struct {float *attn_sinks;} ds4_layer_weights;
static float *tensor_data(const ds4_model *m,float *x){return x;}
static void *xmalloc(size_t n){return malloc(n);}
`
	for _, n := range []string{"dot_f32", "axpy_f32", "scale_f32", "layer_attention_rows_one"} {
		c += extract(n) + "\n"
	}
	c += `int main(void){float q[32768],rows[65536],sink[64],out[32768];for(int i=0;i<32768;i++)q[i]=(float)(i%23-11)/17;for(int i=0;i<65536;i++)rows[i]=(float)(i%41-20)/29;for(int i=0;i<64;i++)sink[i]=(float)(i%13-6)/7;ds4_model m={0};ds4_layer_weights l={sink};layer_attention_rows_one(out,&m,&l,q,rows,128);fwrite(out,4,32768,stdout);return 0;}`
	dir := t.TempDir()
	file, bin := filepath.Join(dir, "attn.c"), filepath.Join(dir, "attn")
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
	if len(raw) != 32768*4 {
		t.Fatal("short C attention output")
	}
	q, rows, sinks, out := make([]float32, 32768), make([]float32, 65536), make([]float32, 64), make([]float32, 32768)
	for i := range q {
		q[i] = float32(i%23-11) / 17
	}
	for i := range rows {
		rows[i] = float32(i%41-20) / 29
	}
	for i := range sinks {
		sinks[i] = float32(i%13-6) / 7
	}
	if err := Attend(out, q, rows, sinks, 512); err != nil {
		t.Fatal(err)
	}
	var max float64
	for i, v := range out {
		want := math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.IsNaN(float64(want)) || math.IsInf(float64(want), 0) {
			t.Fatal("nonfinite attention")
		}
		delta := math.Abs(float64(v - want))
		if delta > max {
			max = delta
		}
		if delta > 1e-6+1e-6*math.Abs(float64(want)) {
			t.Fatalf("attention i=%d got=%g want=%g", i, v, want)
		}
	}
	t.Logf("attention values=32768 heads=64 head_dim=512 rows=128 max_abs=%g", max)
}
