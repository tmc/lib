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

func TestPinnedQ2Oracle(t *testing.T) {
	path := os.Getenv("DS4_CPU_ORACLE_SOURCE")
	if path == "" {
		t.Skip("DS4_CPU_ORACLE_SOURCE not set")
	}
	b := pinnedCPUOracle(t)
	s := string(b)
	start := strings.Index(s, "static inline float q2_k_value_f32(")
	if start < 0 {
		t.Fatal("missing Q2 helper")
	}
	end := strings.Index(s[start:], "\n}") + start + 2
	if start < 0 || end < start {
		t.Fatal("missing Q2 helper")
	}
	c := "#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#define QK_K 256\ntypedef struct {uint8_t scales[16],qs[64];uint16_t d,dmin;} block_q2_K;\nstatic float f16_to_f32(uint16_t h){_Float16 v;memcpy(&v,&h,2);return (float)v;}\n" + s[start:end] + "\n"
	start = strings.Index(s, "static float ds4_vec_dot_q2_K_f32(")
	if start < 0 {
		t.Fatal("missing Q2 dot")
	}
	end = strings.Index(s[start:], "\n}") + start + 2
	c += s[start:end] + "\n"
	c += `int main(int argc,char **argv){int n=atoi(argv[1]);block_q2_K *b=malloc(n/256*sizeof(*b));for(int i=0;i<n/256;i++){for(int j=0;j<16;j++)b[i].scales[j]=(uint8_t)(i*31+j*13);for(int j=0;j<64;j++)b[i].qs[j]=(uint8_t)(i*17+j*23);b[i].d=0x3800;b[i].dmin=0x3400;}for(int k=0;k<n;k++){float v=q2_k_value_f32(b,k);fwrite(&v,4,1,stdout);}float *x=malloc(n*4);for(int i=0;i<n;i++)x[i]=(float)(i%97-48)/17;float dot=ds4_vec_dot_q2_K_f32(n,b,x);fwrite(&dot,4,1,stdout);return 0;}`
	dir := t.TempDir()
	file, bin := filepath.Join(dir, "q2.c"), filepath.Join(dir, "q2")
	if err := os.WriteFile(file, []byte(c), 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("cc", "-O0", "-ffp-contract=off", file, "-o", bin).CombinedOutput(); err != nil {
		t.Fatalf("compile: %v %s", err, out)
	}
	for _, n := range []int{2048, 4096} {
		raw, err := exec.Command(bin, fmt.Sprint(n)).Output()
		if err != nil {
			t.Fatal(err)
		}
		if len(raw) != (n+1)*4 {
			t.Fatal("short oracle output")
		}
		packed := make([]byte, n/256*84)
		for i := 0; i < n/256; i++ {
			q := packed[i*84:]
			for j := 0; j < 16; j++ {
				q[j] = byte(i*31 + j*13)
			}
			for j := 0; j < 64; j++ {
				q[16+j] = byte(i*17 + j*23)
			}
			binary.LittleEndian.PutUint16(q[80:], 0x3800)
			binary.LittleEndian.PutUint16(q[82:], 0x3400)
		}
		m, err := NewMatrix(bytes.NewReader(packed), 0, int64(len(packed)), 1, n, Q2K)
		if err != nil {
			t.Fatal(err)
		}
		got := make([]float32, n)
		if err := m.Row(got, 0); err != nil {
			t.Fatal(err)
		}
		for i, v := range got {
			if math.Float32bits(v) != binary.LittleEndian.Uint32(raw[i*4:]) {
				t.Fatalf("Q2 n=%d i=%d got=%g want=%g", n, i, v, math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:])))
			}
		}
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(i%97-48) / 17
		}
		dot := make([]float32, 1)
		if err := m.Mul(dot, x); err != nil {
			t.Fatal(err)
		}
		want := math.Float32frombits(binary.LittleEndian.Uint32(raw[n*4:]))
		if dot[0] != want {
			t.Fatalf("Q2 nonbasis dot n=%d got=%g want=%g", n, dot[0], want)
		}
	}
}
