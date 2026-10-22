package dsv4cpu

import (
	"encoding/binary"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// This control runs the actual streaming state transition and RMS code. The
// projection callbacks return already projected inputs; RoPE/quantization are
// identity seams here and are independently checked by their pinned tests.
func TestPinnedCompressorState(t *testing.T) {
	s := string(pinnedCPUOracle(t))
	c := `#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define DS4_NEG_INF (-1.0e30f)
#define DS4_RMS_EPS 1.0e-6f
#define DS4_N_ROT 64
#define DS4_N_HEAD_DIM 512
#define DS4_N_INDEXER_HEAD_DIM 128
typedef struct {int unused;} ds4_model;
typedef struct {int type,ndim;uint64_t dim[3];float *data;} ds4_tensor;
static void *xmalloc(size_t n){void *p=malloc(n);if(!p)exit(2);return p;}
static void matvec_any(float *out,const ds4_model *m,const ds4_tensor *w,const float *x){memcpy(out,w->data,w->dim[1]*4);}
static void quantize_q8_0_activation(const float *x,int8_t *q,float *s,uint64_t n){exit(3);}
static void matvec_q8_0_pair_prequant(float *a,float *b,const ds4_model *m,const ds4_tensor *wa,const ds4_tensor *wb,const int8_t *q,const float *s){exit(4);}
static float tensor_2d_value(const ds4_model *m,const ds4_tensor *w,uint32_t j,uint32_t p){return 0;}
static float tensor_1d_value(const ds4_model *m,const ds4_tensor *w,uint32_t i){return 1;}
static void rope_tail_layer_inplace(float *x,uint32_t heads,uint32_t dim,uint32_t rot,uint32_t pos,uint32_t il,bool inverse){}
static void dsv4_fp8_kv_quantize_row_inplace_cpu(float *x,uint32_t dim,uint32_t rot){}
static void dsv4_indexer_qat_row_inplace_cpu(float *x,uint32_t dim){}
`
	for _, name := range []string{"compressor_pool_decode_state", "compressor_decode_one"} {
		c += oracleFunction(t, s, name) + "\n"
	}
	c += `int main(void){ds4_model m={0};ds4_tensor dummy={0};for(int dim=128;dim<=512;dim*=4)for(int ratio=4;ratio<=128;ratio*=32){int lanes=ratio==4?2:1,width=lanes*dim,n=lanes*ratio*width;float *kv=calloc(n,4),*score=malloc(n*4),*k=malloc(width*4),*sc=malloc(width*4),*out=malloc(dim*4);for(int i=0;i<n;i++)score[i]=DS4_NEG_INF;ds4_tensor wk={0,2,{4096,width,0},k},ws={0,2,{4096,width,0},sc};for(int pos=0;pos<ratio*3;pos++){for(int i=0;i<width;i++){k[i]=(float)((pos*7+i)%97-48)/17;sc[i]=(float)((pos*3+i)%29-14)/7;}unsigned char emitted=compressor_decode_one(out,&m,&wk,&ws,&dummy,&dummy,NULL,kv,score,dim,ratio,2,pos);fwrite(&emitted,1,1,stdout);if(emitted)fwrite(out,4,dim,stdout);}fwrite(kv,4,n,stdout);fwrite(score,4,n,stdout);free(out);free(sc);free(k);free(score);free(kv);}return 0;}`
	dir := t.TempDir()
	file, bin := filepath.Join(dir, "state.c"), filepath.Join(dir, "state")
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
	off := 0
	readFloat := func() float32 {
		t.Helper()
		if off+4 > len(raw) {
			t.Fatal("short state oracle")
		}
		v := math.Float32frombits(binary.LittleEndian.Uint32(raw[off:]))
		off += 4
		return v
	}
	for _, dim := range []int{128, 512} {
		for _, ratio := range []int{4, 128} {
			c, err := NewCompressor(dim, ratio)
			if err != nil {
				t.Fatal(err)
			}
			k, score, out, norm, ones := make([]float32, c.projectionWidth), make([]float32, c.projectionWidth), make([]float32, dim), make([]float32, dim), make([]float32, dim)
			for i := range ones {
				ones[i] = 1
			}
			for pos := 0; pos < ratio*3; pos++ {
				for i := range k {
					k[i] = float32((pos*7+i)%97-48) / 17
					score[i] = float32((pos*3+i)%29-14) / 7
				}
				emitted, err := c.Push(out, k, score, pos)
				if err != nil {
					t.Fatal(err)
				}
				if off >= len(raw) {
					t.Fatal("short emit oracle")
				}
				wantEmit := raw[off] != 0
				off++
				if emitted != wantEmit {
					t.Fatalf("dim %d ratio %d pos %d emit", dim, ratio, pos)
				}
				if emitted {
					if err := RMSNorm(norm, out, ones, 1e-6); err != nil {
						t.Fatal(err)
					}
					for i, v := range norm {
						want := readFloat()
						delta := math.Abs(float64(v - want))
						if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || delta > 1e-6+1e-6*math.Abs(float64(want)) {
							t.Fatalf("dim %d ratio %d pos %d output %d got %g want %g", dim, ratio, pos, i, v, want)
						}
					}
				}
			}
			for kind, values := range [][]float32{c.kv, c.score} {
				for i, v := range values {
					want := readFloat()
					if math.Float32bits(v) != math.Float32bits(want) {
						t.Fatalf("dim %d ratio %d state %d index %d got %g want %g", dim, ratio, kind, i, v, want)
					}
				}
			}
		}
	}
	if off != len(raw) {
		t.Fatal("unconsumed state oracle bytes")
	}
	t.Log("actual compressor_decode_one: four real width/ratio schedules, 792 token updates, 12 emits, final rolling KV/score state bit-exact")
}
