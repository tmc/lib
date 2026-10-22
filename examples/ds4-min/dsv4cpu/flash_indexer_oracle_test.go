package dsv4cpu

import (
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// This qualifies selection after the projection/RoPE/QAT seams. The source
// executes its actual scoring and top-k loops, including its <=512 bypass.
func TestPinnedFlashIndexerSelection(t *testing.T) {
	s := string(pinnedCPUOracle(t))
	c := `#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#undef __ARM_NEON
#define DS4_N_INDEXER_TOP_K 512
#define DS4_N_INDEXER_HEAD_DIM 128
#define DS4_N_INDEXER_HEAD 64
#define DS4_N_ROT 64
#define DS4_NEG_INF (-1.0e30f)
typedef struct{int unused;}ds4_model;
typedef struct{float *data;int n;}ds4_tensor;
typedef struct{ds4_tensor *indexer_attn_q_b,*indexer_proj;}ds4_layer_weights;
static void *xmalloc(size_t n){void *p=malloc(n);if(!p)exit(2);return p;}
static void *xcalloc(size_t n,size_t size){void *p=calloc(n,size);if(!p)exit(2);return p;}
static void matvec_any(float *out,const ds4_model *m,const ds4_tensor *w,const float *x){memcpy(out,w->data,w->n*4);}
static void rope_tail_layer_inplace(float *x,uint32_t heads,uint32_t dim,uint32_t rot,uint32_t pos,uint32_t il,bool inverse){}
static void dsv4_indexer_qat_rows_inplace_cpu(float *x,uint32_t rows,uint32_t dim){}
`
	for _, name := range []string{"dot_f32", "indexer_allowed_decode_one"} {
		c += oracleFunction(t, s, name) + "\n"
	}
	c += `int main(void){ds4_model m={0};float q[8192],weight[64],rows[1024*128];ds4_tensor qw={q,8192},ww={weight,64};ds4_layer_weights l={&qw,&ww};int sizes[]={512,513,1024};for(int mode=0;mode<2;mode++)for(int arm=0;arm<3;arm++){for(int i=0;i<8192;i++)q[i]=mode==0?(float)(i%97-48)/17:0;for(int i=0;i<64;i++)weight[i]=(float)(i%11-5)/13;for(int i=0;i<1024*128;i++)rows[i]=(float)(i%53-26)/7;bool *allowed=indexer_allowed_decode_one(&m,&l,NULL,NULL,rows,sizes[arm],2,2049);fwrite(allowed,1,sizes[arm],stdout);free(allowed);}return 0;}`
	dir := t.TempDir()
	file, bin := filepath.Join(dir, "index.c"), filepath.Join(dir, "index")
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
	if len(raw) != 2*(512+513+1024) {
		t.Fatal("short selection oracle")
	}
	offset := 0
	for mode := 0; mode < 2; mode++ {
		for _, n := range []int{512, 513, 1024} {
			q, w, rows := make([]float32, 8192), make([]float32, 64), make([]float32, n*128)
			for i := range q {
				if mode == 0 {
					q[i] = float32(i%97-48) / 17
				}
			}
			scale := float32(1) / float32(math.Sqrt(8192))
			for i := range w {
				w[i] = float32(i%11-5) / 13
				w[i] *= scale
			}
			for i := range rows {
				rows[i] = float32(i%53-26) / 7
			}
			allowed, err := selectCompressedRows(q, w, rows, 512)
			if err != nil {
				t.Fatal(err)
			}
			count := 0
			for i, v := range allowed {
				if v != (raw[offset+i] != 0) {
					t.Fatalf("mode %d rows %d selected row %d differs", mode, n, i)
				}
				if v {
					count++
				}
			}
			if count != 512 {
				t.Fatal("wrong selected count", count)
			}
			offset += n
		}
	}
	t.Log("six actual 64x128 selection cases at512/513/1024 rows, including stable equal-score ties; all4098allowed bits match")
}
