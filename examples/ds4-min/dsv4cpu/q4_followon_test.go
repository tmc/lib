package dsv4cpu

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestPinnedQ4KFollowOn(t *testing.T) {
	s := string(pinnedCPUOracle(t))
	c := `#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#undef __ARM_NEON
#define QK_K 256
typedef struct{uint16_t d,dmin;uint8_t scales[12],qs[128];}block_q4_K;
typedef struct{float d;int8_t qs[256];int16_t bsums[16];}block_q8_K;
static void ds4_die(const char *s){abort();}
static float f16_to_f32(uint16_t h){_Float16 v;memcpy(&v,&h,2);return (float)v;}
`
	for _, name := range []string{"q4_k_get_scale_min", "ds4_quantize_row_q8_K", "ds4_vec_dot_q4_K_q8_K", "ds4_vec_dot_q4_K_f32"} {
		c += oracleFunction(t, s, name) + "\n"
	}
	c += `int main(void){block_q4_K blocks[16];float x[4096];block_q8_K q[16];for(int b=0;b<16;b++){blocks[b].d=0x3000+b;blocks[b].dmin=0x2800+b;for(int j=0;j<12;j++)blocks[b].scales[j]=(uint8_t)(b*13+j*17);for(int j=0;j<128;j++)blocks[b].qs[j]=(uint8_t)(b*19+j*7);}float basis[256]={0};for(int i=0;i<256;i++){basis[i]=1;float v=ds4_vec_dot_q4_K_f32(256,blocks,basis);fwrite(&v,4,1,stdout);basis[i]=0;}for(int arm=0;arm<3;arm++){for(int i=0;i<4096;i++)x[i]=(float)((i*7+arm*11)%97-48)/17;ds4_quantize_row_q8_K(x,q,4096);float dot;ds4_vec_dot_q4_K_q8_K(4096,&dot,blocks,q);fwrite(&dot,4,1,stdout);}return 0;}`
	dir := t.TempDir()
	file, bin := filepath.Join(dir, "q4.c"), filepath.Join(dir, "q4")
	if err := os.WriteFile(file, []byte(c), 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("cc", "-O0", "-ffp-contract=off", file, "-lm", "-o", bin).CombinedOutput(); err != nil {
		t.Fatalf("compile: %v %s", err, out)
	}
	raw, err := exec.Command(bin).Output()
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) != (256+3)*4 {
		t.Fatal("short Q4 oracle")
	}
	blocks := make([]byte, 16*144)
	for b := 0; b < 16; b++ {
		q := blocks[b*144:]
		binary.LittleEndian.PutUint16(q, uint16(0x3000+b))
		binary.LittleEndian.PutUint16(q[2:], uint16(0x2800+b))
		for j := 0; j < 12; j++ {
			q[4+j] = byte(b*13 + j*17)
		}
		for j := 0; j < 128; j++ {
			q[16+j] = byte(b*19 + j*7)
		}
	}
	for i := 0; i < 256; i++ {
		got := q4KValue(blocks, i)
		want := math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) || math.IsNaN(float64(want)) || math.IsInf(float64(want), 0) || math.Float32bits(got) != math.Float32bits(want) {
			t.Fatalf("Q4 value %d got %g want %g", i, got, want)
		}
	}
	matrix, err := NewMatrix(bytes.NewReader(blocks), 0, int64(len(blocks)), 1, 4096, Q4K)
	if err != nil {
		t.Fatal(err)
	}
	decoded := make([]float32, 4096)
	if err := matrix.Row(decoded, 0); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 256; i++ {
		if math.Float32bits(decoded[i]) != binary.LittleEndian.Uint32(raw[i*4:]) {
			t.Fatal("Q4 matrix row differs", i)
		}
	}
	for arm := 0; arm < 3; arm++ {
		x := make([]float32, 4096)
		for i := range x {
			x[i] = float32((i*7+arm*11)%97-48) / 17
		}
		q := make([]Q8K, 16)
		if err := QuantizeQ8K(q, x); err != nil {
			t.Fatal(err)
		}
		var got float32
		for b, a := range q {
			got += q4KDotBlock(blocks[b*144:], a)
		}
		want := math.Float32frombits(binary.LittleEndian.Uint32(raw[(256+arm)*4:]))
		if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) || math.IsNaN(float64(want)) || math.IsInf(float64(want), 0) || math.Float32bits(got) != math.Float32bits(want) {
			t.Fatalf("Q4 dot arm %d got %g want %g", arm, got, want)
		}
		out := make([]float32, 1)
		if err := matrix.MulQ8K(out, q); err != nil {
			t.Fatal(err)
		}
		if math.Float32bits(out[0]) != math.Float32bits(want) {
			t.Fatal("Q4 matrix integer dot differs", arm, out[0], want)
		}
	}
}
