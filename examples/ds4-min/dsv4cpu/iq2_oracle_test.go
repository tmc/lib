package dsv4cpu

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

func TestPinnedIQ2Oracle(t *testing.T) {
	path := os.Getenv("DS4_CPU_ORACLE_SOURCE")
	if path == "" {
		t.Skip("DS4_CPU_ORACLE_SOURCE not set")
	}
	b := pinnedCPUOracle(t)
	s := string(b)
	c := "#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#include <pthread.h>\n#define QK_K 256\ntypedef struct {uint16_t d;uint16_t qs[32];} block_iq2_xxs;\nstatic float f16_to_f32(uint16_t h){_Float16 v;memcpy(&v,&h,2);return (float)v;}\n"
	for _, name := range []string{"kmask_iq2xs", "ksigns_iq2xs", "iq2xxs_grid"} {
		r := regexp.MustCompile(`static const [^\n]+ ` + name + `\[[^]]+\] = \{[^}]+\};`)
		table := r.FindString(s)
		if table == "" {
			t.Fatal(name)
		}
		c += table + "\n"
	}
	start := strings.Index(s, "static int8_t iq2xxs_signed_grid[")
	if start < 0 {
		t.Fatal("missing signed-grid initializer")
	}
	end := strings.Index(s[start:], "static inline DS4_MAYBE_UNUSED int32_t dot_iq2_pair_16") + start
	if start < 0 || end < start {
		t.Fatal("missing signed-grid initializer")
	}
	c += s[start:end] + "\n"
	start = strings.Index(s, "static float ds4_vec_dot_iq2_xxs_f32(")
	if start < 0 {
		t.Fatal("missing IQ2 dot")
	}
	end = strings.Index(s[start:], "\n}") + start + 2
	c += s[start:end] + "\n"
	c += `int main(void){enum{N=4096};block_iq2_xxs b[N/256];float y[N]={0};for(int i=0;i<N/256;i++){b[i].d=0x3800;uint8_t *q=(uint8_t*)b[i].qs;for(int j=0;j<64;j++)q[j]=(uint8_t)(i*37+j*19);}for(int i=0;i<N;i++){y[i]=1;float v=ds4_vec_dot_iq2_xxs_f32(N,b,y);fwrite(&v,4,1,stdout);y[i]=0;}for(int i=0;i<N;i++)y[i]=(float)(i%97-48)/17;float dot=ds4_vec_dot_iq2_xxs_f32(N,b,y);fwrite(&dot,4,1,stdout);return 0;}`
	dir := t.TempDir()
	file, bin := filepath.Join(dir, "iq2.c"), filepath.Join(dir, "iq2")
	if err := os.WriteFile(file, []byte(c), 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("cc", "-O0", "-ffp-contract=off", file, "-pthread", "-o", bin).CombinedOutput(); err != nil {
		t.Fatalf("compile %v %s", err, out)
	}
	raw, err := exec.Command(bin).Output()
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) != 4097*4 {
		t.Fatal("short oracle output")
	}
	packed := make([]byte, 4096/256*66)
	for i := 0; i < 4096/256; i++ {
		q := packed[i*66:]
		binary.LittleEndian.PutUint16(q, 0x3800)
		for j := 0; j < 64; j++ {
			q[2+j] = byte(i*37 + j*19)
		}
	}
	m, err := NewMatrix(bytes.NewReader(packed), 0, int64(len(packed)), 1, 4096, IQ2XXS)
	if err != nil {
		t.Fatal(err)
	}
	got := make([]float32, 4096)
	if err := m.Row(got, 0); err != nil {
		t.Fatal(err)
	}
	for i, v := range got {
		want := math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		if v != want {
			t.Fatalf("IQ2 i=%d got=%g want=%g", i, v, want)
		}
	}
	x := make([]float32, 4096)
	for i := range x {
		x[i] = float32(i%97-48) / 17
	}
	dot := make([]float32, 1)
	if err := m.Mul(dot, x); err != nil {
		t.Fatal(err)
	}
	want := math.Float32frombits(binary.LittleEndian.Uint32(raw[4096*4:]))
	if dot[0] != want {
		t.Fatalf("IQ2 nonbasis dot got=%g want=%g", dot[0], want)
	}
}
