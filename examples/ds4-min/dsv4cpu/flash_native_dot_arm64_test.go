package dsv4cpu

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestPinnedNativeF16Projection(t *testing.T) {
	s := string(pinnedCPUOracle(t))
	c := `#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static float f16_to_f32(uint16_t h){_Float16 v;memcpy(&v,&h,2);return(float)v;}
` + oracleFunction(t, s, "dot_f16_row") + `
int main(void){uint32_t rows,cols;if(fread(&rows,4,1,stdin)!=1||fread(&cols,4,1,stdin)!=1)return 2;
uint16_t *w=malloc((size_t)rows*cols*2);float *x=malloc((size_t)cols*4);
if(!w||!x||fread(w,2,(size_t)rows*cols,stdin)!=(size_t)rows*cols||fread(x,4,cols,stdin)!=cols)return 2;
for(uint32_t r=0;r<rows;r++){float y=dot_f16_row(w+(size_t)r*cols,x,cols);if(fwrite(&y,4,1,stdout)!=1)return 2;}
free(w);free(x);return 0;}`
	dir := t.TempDir()
	file, bin := filepath.Join(dir, "native.c"), filepath.Join(dir, "native")
	if err := os.WriteFile(file, []byte(c), 0600); err != nil {
		t.Fatal(err)
	}
	arch := "-march=native"
	if runtime.GOOS == "darwin" {
		arch = "-mcpu=native"
	}
	flags := []string{"-O3", "-ffast-math", "-fno-finite-math-only", arch, file, "-lm", "-o", bin}
	if out, err := exec.Command("cc", flags...).CombinedOutput(); err != nil {
		t.Fatalf("compile optimized pinned C: %v %s", err, out)
	}
	assembly := filepath.Join(dir, "native.s")
	asmFlags := []string{"-O3", "-ffast-math", "-fno-finite-math-only", arch, "-S", file, "-o", assembly}
	if out, err := exec.Command("cc", asmFlags...).CombinedOutput(); err != nil {
		t.Fatalf("inspect optimized pinned C: %v %s", err, out)
	}
	asm, err := os.ReadFile(assembly)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(asm), "fmla") || !strings.Contains(string(asm), "faddp") {
		t.Fatal("optimized oracle did not retain NEON FMA and pairwise reduction")
	}
	t.Logf("optimized C source sha256=%x assembly sha256=%x flags=%q", sha256.Sum256([]byte(c)), sha256.Sum256(asm), flags[:4])
	for _, cols := range []int{16384, 4096} {
		const rows = 24
		weights := make([]byte, rows*cols*2)
		x := make([]float32, cols)
		seed := uint32(17)
		next := func() uint32 { seed = seed*1664525 + 1013904223; return seed }
		for i := 0; i < rows*cols; i++ {
			v := next()
			binary.LittleEndian.PutUint16(weights[i*2:], uint16((v>>16)&0x83ff)|0x2800)
		}
		for i := range x {
			x[i] = float32(int32(next()>>8)-8388608) / 1048576
		}
		var in bytes.Buffer
		binary.Write(&in, binary.LittleEndian, uint32(rows))
		binary.Write(&in, binary.LittleEndian, uint32(cols))
		in.Write(weights)
		binary.Write(&in, binary.LittleEndian, x)
		cmd := exec.Command(bin)
		cmd.Stdin = &in
		raw, err := cmd.Output()
		if err != nil || len(raw) != rows*4 {
			t.Fatalf("oracle execution: %v bytes=%d", err, len(raw))
		}
		m, err := NewMatrix(bytes.NewReader(weights), 0, int64(len(weights)), rows, cols, F16)
		if err != nil {
			t.Fatal(err)
		}
		got := make([]float32, rows)
		model := make([]float32, rows)
		if err := nativeF16Mul(got, x, m); err != nil {
			t.Fatal(err)
		}
		if err := cpuDense(model, x, m); err != nil {
			t.Fatal(err)
		}
		for i, v := range got {
			want := math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.IsNaN(float64(want)) || math.IsInf(float64(want), 0) || math.Float32bits(v) != math.Float32bits(want) || math.Float32bits(model[i]) != math.Float32bits(want) {
				t.Fatalf("native F16 projection cols=%d row=%d got=%08x model=%08x want=%08x", cols, i, math.Float32bits(v), math.Float32bits(model[i]), math.Float32bits(want))
			}
		}
	}
}

func TestFlashNativeHCAdmission(t *testing.T) {
	tensors, size := virtualFlashDescriptors()
	m, err := newFlashModel(new(virtualFlashSource), size, tensors)
	if err != nil {
		t.Fatal(err)
	}
	for i, l := range m.layers {
		if !l.attentionHC.nativeF16 || !l.ffnHC.nativeF16 {
			t.Fatalf("native F16 HC model route missing at layer %d", i)
		}
	}
}
