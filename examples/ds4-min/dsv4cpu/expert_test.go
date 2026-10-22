package dsv4cpu

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

func ExampleMatrix_MulQ8K() {
	weights := make([]byte, 66)
	binary.LittleEndian.PutUint16(weights, 0x3800)
	m, _ := NewMatrix(bytes.NewReader(weights), 0, 66, 1, 256, IQ2XXS)
	x := make([]float32, 256)
	x[0] = 1
	a := make([]Q8K, 1)
	_ = QuantizeQ8K(a, x)
	out := make([]float32, 1)
	_ = m.MulQ8K(out, a)
	fmt.Printf("%.3f\n", out[0])
	// Output: 0.500
}

func TestPinnedExpertQ8K(t *testing.T) {
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
#include <pthread.h>
#include <math.h>
#undef __ARM_NEON
#define DS4_MAYBE_UNUSED
#define QK_K 256
typedef struct {uint8_t scales[16],qs[64];uint16_t d,dmin;} block_q2_K;
typedef struct {uint16_t d,qs[32];} block_iq2_xxs;
typedef struct {float d;int8_t qs[256];int16_t bsums[16];} block_q8_K;
static float f16_to_f32(uint16_t h){_Float16 v;memcpy(&v,&h,2);return (float)v;}
static void ds4_die(const char *s){abort();}
`
	for _, name := range []string{"kmask_iq2xs", "ksigns_iq2xs", "iq2xxs_grid"} {
		r := regexp.MustCompile(`static const [^\n]+ ` + name + `\[[^]]+\] = \{[^}]+\};`)
		table := r.FindString(s)
		if table == "" {
			t.Fatal(name)
		}
		c += table + "\n"
	}
	c += "static int8_t iq2xxs_signed_grid[256][128][8];\nstatic int8_t iq2xxs_signs[128][8];\n"
	for _, name := range []string{"iq2xxs_signed_grid_init", "dot_iq2_pair_16", "dot_q2_16", "ds4_quantize_row_q8_K", "ds4_vec_dot_q2_K_q8_K", "ds4_vec_dot_iq2_xxs_q8_K"} {
		c += extract(name) + "\n"
	}
	c += `int main(void){enum{N=4096,NB=16};float x[N];block_q8_K a[NB];block_q2_K q[NB];block_iq2_xxs iq[NB];for(int i=0;i<N;i++)x[i]=(float)(i%97-48)/17;ds4_quantize_row_q8_K(x,a,N);iq2xxs_signed_grid_init();for(int r=0;r<7;r++){for(int b=0;b<NB;b++){q[b].d=0x3800;q[b].dmin=0x3400;for(int j=0;j<16;j++)q[b].scales[j]=(uint8_t)(r*7+b*31+j*13);for(int j=0;j<64;j++)q[b].qs[j]=(uint8_t)(r*11+b*17+j*23);iq[b].d=0x3800;uint8_t *p=(uint8_t*)iq[b].qs;for(int j=0;j<64;j++)p[j]=(uint8_t)(r*13+b*37+j*19);}float qdot,idot;ds4_vec_dot_q2_K_q8_K(N,&qdot,q,a);ds4_vec_dot_iq2_xxs_q8_K(N,&idot,iq,a);fwrite(&qdot,4,1,stdout);fwrite(&idot,4,1,stdout);}return 0;}`
	dir := t.TempDir()
	file, bin := filepath.Join(dir, "expert.c"), filepath.Join(dir, "expert")
	if err := os.WriteFile(file, []byte(c), 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("cc", "-O0", "-ffp-contract=off", file, "-lm", "-pthread", "-o", bin).CombinedOutput(); err != nil {
		t.Fatalf("compile %v %s", err, out)
	}
	raw, err := exec.Command(bin).Output()
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) != 14*4 {
		t.Fatal("short expert oracle")
	}
	x := make([]float32, 4096)
	for i := range x {
		x[i] = float32(i%97-48) / 17
	}
	a := make([]Q8K, 16)
	if err := QuantizeQ8K(a, x); err != nil {
		t.Fatal(err)
	}
	for typIndex, typ := range []Encoding{Q2K, IQ2XXS} {
		stride := 84
		if typ == IQ2XXS {
			stride = 66
		}
		packed := make([]byte, 7*16*stride)
		for r := 0; r < 7; r++ {
			for b := 0; b < 16; b++ {
				p := packed[(r*16+b)*stride:]
				if typ == Q2K {
					for j := 0; j < 16; j++ {
						p[j] = byte(r*7 + b*31 + j*13)
					}
					for j := 0; j < 64; j++ {
						p[16+j] = byte(r*11 + b*17 + j*23)
					}
					binary.LittleEndian.PutUint16(p[80:], 0x3800)
					binary.LittleEndian.PutUint16(p[82:], 0x3400)
				} else {
					binary.LittleEndian.PutUint16(p, 0x3800)
					for j := 0; j < 64; j++ {
						p[2+j] = byte(r*13 + b*37 + j*19)
					}
				}
			}
		}
		m, err := NewMatrix(bytes.NewReader(packed), 0, int64(len(packed)), 7, 4096, typ)
		if err != nil {
			t.Fatal(err)
		}
		out := make([]float32, 7)
		if err := m.MulQ8K(out, a); err != nil {
			t.Fatal(err)
		}
		for r, v := range out {
			want := math.Float32frombits(binary.LittleEndian.Uint32(raw[(r*2+typIndex)*4:]))
			if math.Float32bits(v) != math.Float32bits(want) {
				t.Fatalf("expert type%d row%d got%g want%g", typ, r, v, want)
			}
		}
	}
}
