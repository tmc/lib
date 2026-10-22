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

// This gate compiles literal functions from the pinned upstream checkout.
// It is opt-in because ordinary package tests need no C compiler or checkout.
func TestPinnedScalarOracle(t *testing.T) {
	path := os.Getenv("DS4_CPU_ORACLE_SOURCE")
	if path == "" {
		t.Skip("DS4_CPU_ORACLE_SOURCE not set")
	}
	b := pinnedCPUOracle(t)
	source := string(b)
	extract := func(name string) string {
		i := strings.Index(source, name+"(")
		if i < 0 {
			t.Fatalf("missing function %s", name)
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
	c := "#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#include <math.h>\nstatic float f16_to_f32(uint16_t h){_Float16 v;memcpy(&v,&h,2);return (float)v;}\n"
	c += extract("rms_norm_weight") + "\n" + extract("dot_q8_0_row_f32_ref") + "\n"
	c += `int main(int argc,char **argv){int n=atoi(argv[1]);float *x=malloc(n*4),*w=malloc(n*4),*out=malloc(n*4);uint8_t *q=malloc(n/32*34);for(int i=0;i<n;i++){x[i]=(float)(i%97-48)/17;w[i]=(float)(i%13+1)/9;}for(int i=0;i<n/32;i++){uint16_t h=0x3800;memcpy(q+i*34,&h,2);for(int j=0;j<32;j++)q[i*34+2+j]=(uint8_t)((i*32+j)%251-125);}rms_norm_weight(out,x,w,n,1e-6f);fwrite(out,4,n,stdout);float dot=dot_q8_0_row_f32_ref(q,x,n,n/32);fwrite(&dot,4,1,stdout);return 0;}`
	dir := t.TempDir()
	file := filepath.Join(dir, "oracle.c")
	binaryPath := filepath.Join(dir, "oracle")
	if err := os.WriteFile(file, []byte(c), 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("cc", "-std=c11", "-O0", "-ffp-contract=off", file, "-lm", "-o", binaryPath).CombinedOutput(); err != nil {
		t.Fatalf("compile: %v %s", err, out)
	}
	for _, n := range []int{512, 1024, 2048, 4096, 16384} {
		raw, err := exec.Command(binaryPath, fmt.Sprint(n)).Output()
		if err != nil {
			t.Fatal(err)
		}
		if len(raw) != (n+1)*4 {
			t.Fatalf("oracle bytes %d", len(raw))
		}
		x, w, got := make([]float32, n), make([]float32, n), make([]float32, n)
		q := make([]byte, n/32*34)
		for i := range x {
			x[i] = float32(i%97-48) / 17
			w[i] = float32(i%13+1) / 9
			binary.LittleEndian.PutUint16(q[i/32*34:], 0x3800)
			q[i/32*34+2+i%32] = byte(int8(i%251 - 125))
		}
		if err := RMSNorm(got, x, w, 1e-6); err != nil {
			t.Fatal(err)
		}
		for i, v := range got {
			if math.Float32bits(v) != binary.LittleEndian.Uint32(raw[i*4:]) {
				t.Fatalf("RMS n=%d i=%d got=%g", n, i, v)
			}
		}
		m, err := NewMatrix(bytes.NewReader(q), 0, int64(len(q)), 1, n, Q80)
		if err != nil {
			t.Fatal(err)
		}
		dot := make([]float32, 1)
		if err := m.Mul(dot, x); err != nil {
			t.Fatal(err)
		}
		if math.Float32bits(dot[0]) != binary.LittleEndian.Uint32(raw[n*4:]) {
			t.Fatalf("Q8 dot n=%d got=%g want=%g", n, dot[0], math.Float32frombits(binary.LittleEndian.Uint32(raw[n*4:])))
		}
	}
}
