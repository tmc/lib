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

func TestPinnedFlashGroupedOutput(t *testing.T) {
	s := string(pinnedCPUOracle(t))
	c := `#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#undef __ARM_NEON
#define DS4_N_HEAD 64
#define DS4_N_HEAD_DIM 512
typedef struct{int unused;}ds4_model;
typedef struct{int type,ndim;uint64_t dim[2];uint8_t *data;}ds4_tensor;
typedef struct{ds4_tensor *attn_output_a,*attn_output_b;}ds4_layer_weights;
typedef struct{float *out;const uint8_t *data;const int8_t *xq;const float *xscale;uint64_t in_dim,blocks,rank;}matvec_q8_0_grouped_ctx;
static void *xmalloc(size_t n){void *p=malloc(n);if(!p)exit(2);return p;}
static void *xcalloc(size_t n,size_t size){void *p=calloc(n,size);if(!p)exit(2);return p;}
static void ds4_die(const char *s){fprintf(stderr,"%s",s);exit(2);}
static const void *tensor_data(const ds4_model *m,const ds4_tensor *w){return w->data;}
static float f16_to_f32(uint16_t h){_Float16 v;memcpy(&v,&h,2);return (float)v;}
static void ds4_parallel_for(uint64_t n,void (*worker)(void *,uint64_t,uint64_t),void *ctx){worker(ctx,0,n);}
`
	for _, name := range []string{"dot_i8_32", "dot_q8_0_row", "quantize_q8_0_activation", "matvec_q8_0_grouped_worker", "matvec_q8_0_grouped_rows"} {
		c += oracleFunction(t, s, name) + "\n"
	}
	c += `static void matvec_q8_0(float *out,const ds4_model *m,const ds4_tensor *w,const float *x){uint64_t blocks=w->dim[0]/32,stride=blocks*34;int8_t *q=xmalloc(w->dim[0]);float *scale=xmalloc(blocks*4);quantize_q8_0_activation(x,q,scale,w->dim[0]);for(uint64_t r=0;r<w->dim[1];r++)out[r]=dot_q8_0_row(w->data+r*stride,q,scale,w->dim[0],blocks);free(q);free(scale);}
`
	c += oracleFunction(t, s, "layer_grouped_out_one") + "\n"
	c += `static uint8_t *weights(int rows,int cols,int seed){int n=rows*(cols/32)*34;uint8_t *p=xmalloc(n);for(int b=0;b<n/34;b++){uint16_t d=0x1800;memcpy(p+b*34,&d,2);for(int j=0;j<32;j++)p[b*34+2+j]=(uint8_t)((b*7+j*11+seed)%31-15);}return p;}
int main(void){ds4_model m={0};ds4_tensor a={8,2,{4096,8192},weights(8192,4096,1)},b={8,2,{8192,4096},weights(4096,8192,3)};ds4_layer_weights l={&a,&b};float heads[32768],out[4096];for(int i=0;i<32768;i++)heads[i]=(float)(i%97-48)/17;layer_grouped_out_one(out,&m,&l,heads);fwrite(out,4,4096,stdout);return 0;}`
	dir := t.TempDir()
	file, bin := filepath.Join(dir, "grouped.c"), filepath.Join(dir, "grouped")
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
		t.Fatal("short grouped oracle")
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
	l := flashLayer{outA: matrix(8192, 4096, 1), outB: matrix(4096, 8192, 3)}
	heads, out := make([]float32, 32768), make([]float32, 4096)
	for i := range heads {
		heads[i] = float32(i%97-48) / 17
	}
	if err := groupedOutput(out, heads, &l); err != nil {
		t.Fatal(err)
	}
	for i, v := range out {
		want := math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.IsNaN(float64(want)) || math.IsInf(float64(want), 0) || math.Abs(float64(v-want)) > 1e-6+1e-6*math.Abs(float64(want)) {
			t.Fatalf("grouped index %d got %g want %g", i, v, want)
		}
	}
}
