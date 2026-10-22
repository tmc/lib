package dsv4cpu

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestPinnedNativeQ80Projection(t *testing.T) {
	s := string(pinnedCPUOracle(t))
	c := `#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define DS4_MAYBE_UNUSED
`
	for _, name := range []string{"f16_to_f32", "dot_i8_32", "dot_q8_0_row", "dot_q8_0_row_pair"} {
		c += oracleFunction(t, s, name) + "\n"
	}
	c += `int main(void){uint32_t rows,cols;if(fread(&rows,4,1,stdin)!=1||fread(&cols,4,1,stdin)!=1||cols%32)return 2;uint32_t blocks=cols/32;
uint8_t *w=malloc((size_t)rows*blocks*34);int8_t *xq=malloc(cols);float *scale=malloc((size_t)blocks*4);
if(!w||!xq||!scale||fread(w,34,(size_t)rows*blocks,stdin)!=(size_t)rows*blocks||fread(xq,1,cols,stdin)!=cols||fread(scale,4,blocks,stdin)!=blocks)return 2;
for(uint32_t r=0;r<rows;r+=2){float y[2];dot_q8_0_row_pair(w+(size_t)r*blocks*34,w+(size_t)(r+1)*blocks*34,xq,scale,cols,blocks,&y[0],&y[1]);for(int j=0;j<2;j++){float single=dot_q8_0_row(w+(size_t)(r+j)*blocks*34,xq,scale,cols,blocks);if(memcmp(&single,&y[j],4))return 3;}if(fwrite(y,4,2,stdout)!=2)return 2;}
free(w);free(xq);free(scale);return 0;}`
	dir := t.TempDir()
	file, bin, assembly := filepath.Join(dir, "native-q80.c"), filepath.Join(dir, "native-q80"), filepath.Join(dir, "native-q80.s")
	if err := os.WriteFile(file, []byte(c), 0600); err != nil {
		t.Fatal(err)
	}
	arch := "-march=native"
	if runtime.GOOS == "darwin" {
		arch = "-mcpu=native"
	}
	flags := []string{"-O3", "-ffast-math", "-fno-finite-math-only", arch}
	if out, err := exec.Command("cc", append(append([]string(nil), flags...), file, "-lm", "-o", bin)...).CombinedOutput(); err != nil {
		t.Fatalf("compile optimized Q8_0 oracle: %v %s", err, out)
	}
	if out, err := exec.Command("cc", append(append([]string(nil), flags...), "-S", file, "-o", assembly)...).CombinedOutput(); err != nil {
		t.Fatalf("inspect optimized Q8_0 oracle: %v %s", err, out)
	}
	asm, err := os.ReadFile(assembly)
	if err != nil {
		t.Fatal(err)
	}
	// These schedules are pinned to macOS clang and DGX GCC 13.3, respectively.
	// A compiler that selects a different schedule requires renewed qualification.
	ops := []string{"sdot", "fmla", "faddp"}
	if runtime.GOOS == "linux" {
		ops = []string{"fmadd", "addv"}
		if strings.Contains(string(asm), "sdot") {
			t.Fatal("owning GCC oracle unexpectedly selected DOTPROD")
		}
	}
	for _, op := range ops {
		if !strings.Contains(string(asm), op) {
			t.Fatalf("owning oracle did not exercise %s", op)
		}
	}
	t.Logf("optimized Q8_0 source sha256=%x assembly sha256=%x flags=%q", sha256.Sum256([]byte(c)), sha256.Sum256(asm), flags)
	widths := []int{96, 1024, 4096, 8192}
	if os.Getenv("DS4_CPU_Q80_ROWS") != "" {
		widths = []int{4096}
	}
	for _, cols := range widths {
		const rows = 24
		blocks := cols / 32
		weights := make([]byte, rows*blocks*34)
		x := make([]float32, cols)
		seed := uint32(31)
		next := func() uint32 { seed = seed*1664525 + 1013904223; return seed }
		for block := 0; block < rows*blocks; block++ {
			binary.LittleEndian.PutUint16(weights[block*34:], uint16(next()&0x83ff)|0x2400)
			for j := 0; j < 32; j++ {
				weights[block*34+2+j] = byte(next() >> 24)
			}
		}
		if path := os.Getenv("DS4_CPU_Q80_ROWS"); path != "" && cols == 4096 {
			actual, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if len(actual) != len(weights) {
				t.Fatalf("actual QA row fixture bytes=%d want=%d", len(actual), len(weights))
			}
			if fmt.Sprintf("%x", sha256.Sum256(actual)) != "f921dd72e0bb7abe2beb43a7c866d5779d182e13b1dc0634b22306e15f6d562a" {
				t.Fatal("actual QA row fixture identity differs")
			}
			weights = actual
			t.Logf("actual L0 QA row fixture sha256=%x rows=%d cols=%d", sha256.Sum256(actual), rows, cols)
		}
		for i := range x {
			x[i] = float32(int32(next()>>8)-8388608) / 1048576
		}
		activation := make([]Q80Activation, blocks)
		if err := QuantizeQ80(activation, x); err != nil {
			t.Fatal(err)
		}
		var in bytes.Buffer
		for _, v := range []uint32{rows, uint32(cols)} {
			if err := binary.Write(&in, binary.LittleEndian, v); err != nil {
				t.Fatal(err)
			}
		}
		in.Write(weights)
		for _, a := range activation {
			if err := binary.Write(&in, binary.LittleEndian, a.Values); err != nil {
				t.Fatal(err)
			}
		}
		for _, a := range activation {
			if err := binary.Write(&in, binary.LittleEndian, a.Scale); err != nil {
				t.Fatal(err)
			}
		}
		cmd := exec.Command(bin)
		cmd.Stdin = &in
		raw, err := cmd.Output()
		if err != nil || len(raw) != rows*4 {
			t.Fatalf("owning Q8_0 oracle: %v bytes=%d", err, len(raw))
		}
		m, err := NewMatrix(bytes.NewReader(weights), 0, int64(len(weights)), rows, cols, Q80)
		if err != nil {
			t.Fatal(err)
		}
		got, model := make([]float32, rows), make([]float32, rows)
		if err := nativeQ80Mul(got, activation, m); err != nil {
			t.Fatal(err)
		}
		if err := cpuDense(model, x, m); err != nil {
			t.Fatal(err)
		}
		shared := make([]float32, rows)
		if err := cpuDensePrequant(shared, activation, m); err != nil {
			t.Fatal(err)
		}
		for i, v := range got {
			want := binary.LittleEndian.Uint32(raw[i*4:])
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.Float32bits(v) != want || math.Float32bits(model[i]) != want || math.Float32bits(shared[i]) != want {
				t.Fatalf("native Q8_0 projection cols=%d row=%d got=%08x model=%08x want=%08x", cols, i, math.Float32bits(v), math.Float32bits(model[i]), want)
			}
		}
	}
}
