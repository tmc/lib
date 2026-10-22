package dsv4cpu

import (
	"encoding/binary"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestPinnedHCOracle(t *testing.T) {
	path := os.Getenv("DS4_CPU_ORACLE_SOURCE")
	if path == "" {
		t.Skip("DS4_CPU_ORACLE_SOURCE not set")
	}
	b := pinnedCPUOracle(t)
	source := string(b)
	extract := func(name string) string {
		i := strings.Index(source, name+"(")
		if i < 0 {
			t.Fatal(name)
		}
		start := strings.LastIndex(source[:i], "\n") + 1
		brace := strings.Index(source[i:], "{") + i
		depth := 0
		for end := brace; end < len(source); end++ {
			switch source[end] {
			case '{':
				depth++
			case '}':
				depth--
				if depth == 0 {
					return source[start : end+1]
				}
			}
		}
		t.Fatal("unterminated function")
		return ""
	}
	c := "#include <stdio.h>\n#include <stdint.h>\n#include <math.h>\n#define DS4_NEG_INF (-INFINITY)\n"
	for _, name := range []string{"hc_split_sinkhorn_one", "hc_weighted_sum_one", "hc_post_one"} {
		c += extract(name) + "\n"
	}
	c += `int main(void){float mix[24],base[24],scale[3]={.3f,.7f,.2f},out[24],state[16384],plain[4096],next[16384];for(int i=0;i<24;i++){mix[i]=(float)(i%7-3)/5;base[i]=(float)(i%11-5)/9;}for(int i=0;i<16384;i++)state[i]=(float)(i%113-56)/31;hc_split_sinkhorn_one(out,mix,scale,base,4,20,1e-6f);hc_weighted_sum_one(plain,state,out,4096,4);hc_post_one(next,plain,state,out+4,out+8,4096,4);fwrite(out,4,24,stdout);fwrite(plain,4,4096,stdout);fwrite(next,4,16384,stdout);return 0;}`
	dir := t.TempDir()
	file := filepath.Join(dir, "hc.c")
	bin := filepath.Join(dir, "hc")
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
	if len(raw) != (24+4096+16384)*4 {
		t.Fatal("short oracle output")
	}
	mix, base := make([]float32, 24), make([]float32, 24)
	for i := range mix {
		mix[i] = float32(i%7-3) / 5
		base[i] = float32(i%11-5) / 9
	}
	out, state, plain, next := make([]float32, 24), make([]float32, 16384), make([]float32, 4096), make([]float32, 16384)
	for i := range state {
		state[i] = float32(i%113-56) / 31
	}
	if err := Split(out, mix, []float32{.3, .7, .2}, base, 4, 20, 1e-6); err != nil {
		t.Fatal(err)
	}
	if err := Collapse(plain, state, out[:4]); err != nil {
		t.Fatal(err)
	}
	if err := Combine(next, plain, state, out[4:8], out[8:]); err != nil {
		t.Fatal(err)
	}
	all := append(append(append([]float32{}, out...), plain...), next...)
	var max float64
	for i, v := range all {
		want := math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.IsNaN(float64(want)) || math.IsInf(float64(want), 0) {
			t.Fatalf("nonfinite HC value i=%d", i)
		}
		delta := math.Abs(float64(v - want))
		if delta > max {
			max = delta
		}
		if delta > 1e-6+1e-6*math.Abs(float64(want)) {
			t.Fatalf("HC i=%d got=%g want=%g", i, v, want)
		}
	}
	t.Logf("HC values=%d max_abs=%g scalar expf tolerance=abs1e-6+rel1e-6", len(all), max)
}
