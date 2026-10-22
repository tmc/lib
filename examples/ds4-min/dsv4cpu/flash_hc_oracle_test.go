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

func TestPinnedFlashHCPre(t *testing.T) {
	s := string(pinnedCPUOracle(t))
	c := `#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#undef __ARM_NEON
#define DS4_N_HC 4
#define DS4_N_EMBD 4096
#define DS4_RMS_EPS 1.0e-6f
#define DS4_NEG_INF (-1.0e30f)
#define DS4_N_HC_SINKHORN_ITER 20
typedef struct{int unused;}ds4_model;
typedef struct{const void *data;}ds4_tensor;
static const void *tensor_data(const ds4_model *m,const ds4_tensor *t){return t->data;}
static float f16_to_f32(uint16_t h){_Float16 v;memcpy(&v,&h,2);return (float)v;}
`
	for _, name := range []string{"dot_f16_row", "rms_norm_no_weight", "hc_split_sinkhorn_one", "hc_weighted_sum_one"} {
		c += oracleFunction(t, s, name) + "\n"
	}
	c += `static void matvec_f16(float *out,const ds4_model *m,const ds4_tensor *w,const float *x){const uint16_t *data=w->data;for(int r=0;r<24;r++)out[r]=dot_f16_row(data+r*16384,x,16384);}
static void matvec_f16_serial(float *out,const ds4_model *m,const ds4_tensor *w,const float *x){matvec_f16(out,m,w,x);}
`
	c += oracleFunction(t, s, "hc_pre_from_state_one_scratch") + "\n"
	c += `int main(void){uint16_t *fn=malloc(24*16384*2);float state[16384],flat[16384],out[4096],post[4],comb[16],scale[3]={0.5f,0.75f,0.25f},base[24];for(int i=0;i<24*16384;i++)fn[i]=0x1800+(i%257);for(int i=0;i<16384;i++)state[i]=(float)(i%97-48)/17;for(int i=0;i<24;i++)base[i]=(float)(i%7-3)/8;ds4_model m={0};ds4_tensor w={fn},sc={scale},b={base};hc_pre_from_state_one_scratch(&m,&w,&sc,&b,state,out,post,comb,flat,true);fwrite(out,4,4096,stdout);fwrite(post,4,4,stdout);fwrite(comb,4,16,stdout);fwrite(flat,4,16384,stdout);free(fn);return 0;}`
	dir := t.TempDir()
	file, bin := filepath.Join(dir, "hcpre.c"), filepath.Join(dir, "hcpre")
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
	if len(raw) != (4096+4+16+16384)*4 {
		t.Fatal("short HC pre oracle")
	}
	packed := make([]byte, 24*16384*2)
	for i := 0; i < 24*16384; i++ {
		binary.LittleEndian.PutUint16(packed[i*2:], uint16(0x1800+i%257))
	}
	projection, err := NewMatrix(bytes.NewReader(packed), 0, int64(len(packed)), 24, 16384, F16)
	if err != nil {
		t.Fatal(err)
	}
	h := flashHC{projection: projection, scale: []float32{0.5, 0.75, 0.25}, base: make([]float32, 24)}
	for i := range h.base {
		h.base[i] = float32(i%7-3) / 8
	}
	state, plain, post, comb, flat := make([]float32, 16384), make([]float32, 4096), make([]float32, 4), make([]float32, 16), make([]float32, 16384)
	for i := range state {
		state[i] = float32(i%97-48) / 17
	}
	if err := h.pre(plain, post, comb, state); err != nil {
		t.Fatal(err)
	}
	if err := RMSNorm(flat, state, nil, 1e-6); err != nil {
		t.Fatal(err)
	}
	offset := 0
	var maxAbs float64
	for kind, values := range [][]float32{plain, post, comb, flat} {
		for i, v := range values {
			want := math.Float32frombits(binary.LittleEndian.Uint32(raw[(offset+i)*4:]))
			delta := math.Abs(float64(v - want))
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.IsNaN(float64(want)) || math.IsInf(float64(want), 0) || delta > 1e-6+1e-6*math.Abs(float64(want)) {
				t.Fatalf("HC pre kind %d index %d got %g want %g", kind, i, v, want)
			}
			if delta > maxAbs {
				maxAbs = delta
			}
		}
		offset += len(values)
	}
	t.Logf("actual composed HC pre: 16384->24 F16 projection + RMS/Sinkhorn/collapse, 20500values maxabs %g", maxAbs)
}
