package dsv4cpu

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// The scaffold supplies allocation and serial dispatch only. Quantization,
// dot products, activation and the shared-expert orchestration are literal C.
func TestPinnedFlashSharedFFN(t *testing.T) {
	s := string(pinnedCPUOracle(t))
	c := `#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#undef __ARM_NEON
#define DS4_N_FF_EXP 2048
#define DS4_SWIGLU_CLAMP_EXP 10.0f
typedef struct{int unused;}ds4_model;
typedef struct{int type;uint64_t dim[2];uint8_t *data;}ds4_tensor;
typedef struct{ds4_tensor *ffn_gate_shexp,*ffn_up_shexp,*ffn_down_shexp;}ds4_layer_weights;
static void *xmalloc(size_t n){void *p=malloc(n);if(!p)exit(2);return p;}
static void ds4_die(const char *s){fprintf(stderr,"%s",s);exit(2);}
static float f16_to_f32(uint16_t h){_Float16 v;memcpy(&v,&h,2);return (float)v;}
`
	for _, name := range []string{"dot_i8_32", "dot_q8_0_row", "quantize_q8_0_activation", "sigmoid_stable", "silu", "swiglu"} {
		ss := s
		if name == "silu" {
			i := strings.Index(s, "static float silu(float x) {")
			if i < 0 {
				t.Fatal(name)
			}
			ss = "\n" + s[i:]
		}
		c += oracleFunction(t, ss, name) + "\n"
	}
	c += `static void matvec_q8_0_pair_prequant(float *a,float *b,const ds4_model *m,const ds4_tensor *wa,const ds4_tensor *wb,const int8_t *q,const float *scale){uint64_t blocks=wa->dim[0]/32,stride=blocks*34;for(uint64_t r=0;r<wa->dim[1];r++){a[r]=dot_q8_0_row(wa->data+r*stride,q,scale,wa->dim[0],blocks);b[r]=dot_q8_0_row(wb->data+r*stride,q,scale,wb->dim[0],blocks);}}
static void matvec_q8_0(float *out,const ds4_model *m,const ds4_tensor *w,const float *x){uint64_t blocks=w->dim[0]/32,stride=blocks*34;int8_t *q=xmalloc(w->dim[0]);float *scale=xmalloc(blocks*4);quantize_q8_0_activation(x,q,scale,w->dim[0]);for(uint64_t r=0;r<w->dim[1];r++)out[r]=dot_q8_0_row(w->data+r*stride,q,scale,w->dim[0],blocks);free(q);free(scale);}
`
	c += oracleFunction(t, s, "layer_shared_ffn_one") + "\n"
	c += `static uint8_t *weights(int rows,int cols,int seed){int n=rows*(cols/32)*34;uint8_t *p=xmalloc(n);for(int b=0;b<n/34;b++){uint16_t d=0x1800;memcpy(p+b*34,&d,2);for(int j=0;j<32;j++)p[b*34+2+j]=(uint8_t)((b*7+j*11+seed)%31-15);}return p;}
int main(void){ds4_model m={0};ds4_tensor gate={8,{4096,2048},weights(2048,4096,1)},up={8,{4096,2048},weights(2048,4096,3)},down={8,{2048,4096},weights(4096,2048,5)};ds4_layer_weights l={&gate,&up,&down};float x[4096],out[4096];for(int i=0;i<4096;i++)x[i]=(float)(i%97-48)/17;layer_shared_ffn_one(out,&m,&l,x);fwrite(out,4,4096,stdout);return 0;}`
	dir := t.TempDir()
	file, bin := filepath.Join(dir, "shared.c"), filepath.Join(dir, "shared")
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
	if len(raw) != 4096*4 {
		t.Fatal("short shared oracle")
	}
	matrix := func(rows, cols, seed int) *Matrix {
		p := make([]byte, rows*(cols/32)*34)
		for b := 0; b < len(p)/34; b++ {
			binary.LittleEndian.PutUint16(p[b*34:], 0x1800)
			for j := 0; j < 32; j++ {
				p[b*34+2+j] = byte(int8((b*7+j*11+seed)%31 - 15))
			}
		}
		m, err := NewMatrix(bytes.NewReader(p), 0, int64(len(p)), rows, cols, Q80)
		if err != nil {
			t.Fatal(err)
		}
		return m
	}
	l := flashLayer{sharedGate: matrix(2048, 4096, 1), sharedUp: matrix(2048, 4096, 3), sharedDown: matrix(4096, 2048, 5)}
	x, out := make([]float32, 4096), make([]float32, 4096)
	for i := range x {
		x[i] = float32(i%97-48) / 17
	}
	if err := sharedFFN(out, x, &l); err != nil {
		t.Fatal(err)
	}
	for i, v := range out {
		want := math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.IsNaN(float64(want)) || math.IsInf(float64(want), 0) || math.Abs(float64(v-want)) > 1e-6+1e-6*math.Abs(float64(want)) {
			t.Fatalf("shared index %d got %g want %g", i, v, want)
		}
	}
}
