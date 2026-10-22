package dsv41

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestPrecisionPinnedC compiles the literal CUDA kernel body as a scalar host
// fixture. Warp peak and launch indices are emulated; CUDA execution is not
// qualified by this test. The same-site peak mutant must disagree.
func TestPrecisionPinnedC(t *testing.T) {
	repo := os.Getenv("DS41_ORACLE")
	if repo == "" {
		t.Skip("set DS41_ORACLE for literal source control")
	}
	const revision = "11e8cad4473716a41baf4d62e92ed9764a00fce6"
	sources := make(map[string]string)
	for name, want := range map[string]string{"ds4_deepseek41_cuda.cuh": "0dc287ea0e7db01eb1590ed73dfce0489595e6b6bc2fc3512d5a10e090f3041e", "ds4_cuda.cu": "c420f92fecf07af2113bad5b9d74801c5e1d10020f1094120d9f31ba430fc0b5"} {
		b, err := exec.Command("git", "-C", repo, "show", revision+":"+name).Output()
		if err != nil {
			t.Fatal(err)
		}
		if want != "" && fmt.Sprintf("%x", sha256.Sum256(b)) != want {
			t.Fatal("source hash mismatch", name)
		}
		sources[name] = string(b)
	}
	var functions string
	for _, name := range []string{"dsv4_e4m3fn_value_dev", "dsv4_e4m3fn_dequant_dev", "dsv4_e2m1fn_value_dev", "dsv4_e2m1fn_dequant_dev"} {
		functions += literalFunction(t, sources["ds4_cuda.cu"], name)
	}
	for _, name := range []string{"dsv41_bf16", "dsv41_pow2_ceil", "dsv41_quantize_kernel"} {
		functions += literalFunction(t, sources["ds4_deepseek41_cuda.cuh"], name)
	}
	const prefix = `#include <stdint.h>
#include <stdio.h>
#include <math.h>
#define __device__
#define __global__
#define DS4_V41_FP8_E8M0 1
#define DS4_V41_FP4_E8M0 2
#define DS4_V41_FP4_E4M3 3
struct index { unsigned x; } blockIdx, threadIdx;
static float peak;
static float warp_max_f32(float ignored) { return peak; }
#define __shfl_sync(mask,x,lane) (x)
static uint32_t __float_as_uint(float x) { union { float f; uint32_t u; } b={.f=x};return b.u; }
static float __uint_as_float(uint32_t x) { union { uint32_t u; float f; } b={.u=x};return b.f; }
static float __fmul_rn(float x,float y) { volatile float z=x*y;return z; }
static float __fdiv_rn(float x,float y) { volatile float z=x/y;return z; }
`
	const main = `int main(void) {
 uint32_t mode; float x[192];
 if(fread(&mode,4,1,stdin)!=1 || fread(x,4,192,stdin)!=192)return 2;
 unsigned block=mode==3?16:32;
 for(unsigned group=0;group<192/block;group++) {
  peak=0;for(unsigned lane=0;lane<block;lane++)peak=fmaxf(peak,fabsf(dsv41_bf16(x[group*block+lane])));
  blockIdx.x=group/8;
  for(unsigned lane=0;lane<block;lane++){threadIdx.x=(group%8)*32+lane;dsv41_quantize_kernel(x,64,3,mode);}
 }
 return fwrite(x,4,192,stdout)==192?0:3;
}
`
	x := make([]float32, 192)
	for i := range x {
		x[i] = float32(math.Sin(float64(i)*.73) * math.Exp(float64(i%17)/3))
	}
	for _, mutant := range []bool{false, true} {
		f := functions
		if mutant {
			f = strings.Replace(f, "__shfl_sync(0xffffffffu, warp_max_f32(fabsf(value)), 0)", "fabsf(value)", 1)
			if f == functions {
				t.Fatal("mutant site missing")
			}
		}
		dir := t.TempDir()
		path := filepath.Join(dir, "oracle.c")
		if err := os.WriteFile(path, []byte(prefix+f+main), 0600); err != nil {
			t.Fatal(err)
		}
		bin := filepath.Join(dir, "oracle")
		if out, err := exec.Command("cc", "-O0", "-ffp-contract=off", path, "-o", bin, "-lm").CombinedOutput(); err != nil {
			t.Fatalf("compile: %v %s", err, out)
		}
		changed := false
		for mode := FP8E8M0; mode <= FP4E4M3; mode++ {
			var input bytes.Buffer
			binary.Write(&input, binary.LittleEndian, uint32(mode))
			binary.Write(&input, binary.LittleEndian, x)
			cmd := exec.Command(bin)
			cmd.Stdin = &input
			out, err := cmd.Output()
			if err != nil {
				t.Fatal(err)
			}
			want := append([]float32(nil), x...)
			if err := Quantize(want, 64, mode); err != nil {
				t.Fatal(err)
			}
			if len(out) != len(x)*4 {
				t.Fatal("oracle output size")
			}
			for i, v := range want {
				bits := binary.LittleEndian.Uint32(out[4*i:])
				if bits != math.Float32bits(v) {
					changed = true
					if !mutant {
						t.Fatalf("mode %d index %d: C=%08x Go=%08x", mode, i, bits, math.Float32bits(v))
					}
				}
			}
		}
		if mutant && !changed {
			t.Fatal("peak mutant not detected")
		}
	}
}

func literalFunction(t *testing.T, source, name string) string {
	t.Helper()
	i := strings.Index(source, name+"(")
	if i < 0 {
		t.Fatal("missing function", name)
	}
	start := strings.LastIndex(source[:i], "\n") + 1
	open := strings.Index(source[i:], "{") + i
	depth := 0
	for j := open; j < len(source); j++ {
		switch source[j] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return source[start:j+1] + "\n"
			}
		}
	}
	t.Fatal("unterminated function", name)
	return ""
}
