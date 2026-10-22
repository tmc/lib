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

	"github.com/tmc/lib/examples/ds4-min/dsv4cpu"
)

func pinned41(t *testing.T) string {
	t.Helper()
	repo := os.Getenv("DS41_ORACLE")
	if repo == "" {
		t.Skip("set DS41_ORACLE for literal source control")
	}
	b, err := exec.Command("git", "-C", repo, "show", "11e8cad4473716a41baf4d62e92ed9764a00fce6:ds4_deepseek41_cuda.cuh").Output()
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(b)) != "0dc287ea0e7db01eb1590ed73dfce0489595e6b6bc2fc3512d5a10e090f3041e" {
		t.Fatal("source hash mismatch")
	}
	return string(b)
}

const operationPrefix = `#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#define __device__
#define __global__
struct index { unsigned x; } blockIdx,threadIdx,blockDim;
static uint32_t __float_as_uint(float x){union {float f;uint32_t u;}b={.f=x};return b.u;}
static float __uint_as_float(uint32_t x){union {uint32_t u;float f;}b={.u=x};return b.f;}
static float __fmul_rn(float x,float y){volatile float z=x*y;return z;}
static float __fadd_rn(float x,float y){volatile float z=x+y;return z;}
static float __fdiv_rn(float x,float y){volatile float z=x/y;return z;}
static float rsqrtf(float x){return 1.0f/sqrtf(x);}
static float sums[3];static unsigned calls;
static float warp_sum_f32(float x){return sums[calls++%3];}
#define __shfl_sync(mask,x,lane) (x)
`

func runLiteral(t *testing.T, source string, input []float32) []float32 {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "oracle.c")
	if err := os.WriteFile(path, []byte(source), 0600); err != nil {
		t.Fatal(err)
	}
	bin := filepath.Join(dir, "oracle")
	if out, err := exec.Command("c++", "-x", "c++", "-std=c++20", "-U__ARM_NEON", "-O0", "-ffp-contract=off", path, "-o", bin, "-lm").CombinedOutput(); err != nil {
		t.Fatalf("compile: %v %s", err, out)
	}
	var b bytes.Buffer
	if err := binary.Write(&b, binary.LittleEndian, input); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(bin)
	cmd.Stdin = &b
	out, err := cmd.Output()
	if err != nil {
		t.Fatal(err)
	}
	if len(out)%4 != 0 {
		t.Fatal("invalid oracle extent")
	}
	values := make([]float32, len(out)/4)
	for i := range values {
		values[i] = math.Float32frombits(binary.LittleEndian.Uint32(out[4*i:]))
	}
	return values
}

// TestEngramProjectionPinnedC covers the F16 projection and its output BF16
// boundary against the pinned scalar C matvec, not the GPU GEMM reduction.
func TestEngramProjectionPinnedC(t *testing.T) {
	repo := os.Getenv("DS41_ORACLE")
	if repo == "" {
		t.Skip("set DS41_ORACLE for literal source control")
	}
	b, err := exec.Command("git", "-C", repo, "show", "11e8cad4473716a41baf4d62e92ed9764a00fce6:ds4.c").Output()
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(b)) != "62e458f2c416148c0882f7247c7b16f9820054b0c2ae41374a9209541fe7a4c7" {
		t.Fatal("projection source hash mismatch")
	}
	f := literalFunction(t, string(b), "f16_to_f32") + literalFunction(t, string(b), "dot_f16_row") + literalFunction(t, pinned41(t), "dsv41_bf16")
	const main = `int main(void){uint16_t weights[5*6144];float x[6144],out[5];if(fread(weights,2,5*6144,stdin)!=5*6144||fread(x,4,6144,stdin)!=6144)return 2;for(unsigned r=0;r<5;r++)out[r]=dsv41_bf16(dot_f16_row(weights+r*6144,x,6144));return fwrite(out,4,5,stdout)==5?0:3;}`
	w := make([]byte, 5*6144*2)
	x := make([]float32, 6144)
	for i := range x {
		x[i] = BF16(float32(math.Sin(float64(i)*.137)) * .8)
	}
	for r := 0; r < 5; r++ {
		for i := 0; i < 6144; i++ {
			binary.LittleEndian.PutUint16(w[(r*6144+i)*2:], []uint16{0x3000, 0xb400, 0x3800, 0x2800}[((i+r)%4)])
		}
	}
	m, err := dsv4cpu.NewMatrix(bytes.NewReader(w), 0, int64(len(w)), 5, 6144, dsv4cpu.F16)
	if err != nil {
		t.Fatal(err)
	}
	want := make([]float32, 5)
	if err := m.Mul(want, x); err != nil {
		t.Fatal(err)
	}
	for i := range want {
		want[i] = BF16(want[i])
	}
	input := make([]float32, len(w)/4)
	for i := range input {
		input[i] = math.Float32frombits(binary.LittleEndian.Uint32(w[i*4:]))
	}
	input = append(input, x...)
	if got := runLiteral(t, operationPrefix+f+main, input); !equalBits(got, want) {
		t.Fatal("projection mismatch", got, want)
	}
	mutant := strings.ReplaceAll(f, "f16_to_f32(row[i]) * x[i]", "f16_to_f32(row[0]) * x[i]")
	if mutant == f {
		t.Fatal("projection operand mutant missing")
	}
	if equalBits(runLiteral(t, operationPrefix+mutant+main, input), want) {
		t.Fatal("projection operand mutant not detected")
	}
}

func equalBits(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return false
		}
	}
	return true
}

// TestPoolPinnedC runs the unchanged pool kernel on three chunk boundaries.
// The host launcher implements the wrapper's last-even carry copy. It does not
// qualify GPU memory copies, streams, or device transcendental instructions.
func TestPoolPinnedC(t *testing.T) {
	source := pinned41(t)
	f := literalFunction(t, source, "dsv41_bf16") + literalFunction(t, source, "dsv41_pool_kernel")
	const main = `int main(void){float kv[20],score[20],prev[4]={0},ps[4]={0},out[8];if(fread(kv,4,20,stdin)!=20||fread(score,4,20,stdin)!=20)return 2;
 unsigned chunks[3]={1,3,1},start=0,output=0;blockDim.x=256;blockIdx.x=0;
 for(unsigned c=0;c<3;c++){unsigned rows=chunks[c],pairs=(rows+(start&1))/2;
  for(unsigned i=0;i<4*pairs;i++){threadIdx.x=i;dsv41_pool_kernel(out+output,kv+start*4,score+start*4,prev,ps,4,pairs,start&1);}
  unsigned last_even=(start+rows-1)&~1u;if(last_even>=start)for(unsigned j=0;j<4;j++){prev[j]=kv[last_even*4+j];ps[j]=score[last_even*4+j];}
  output+=pairs*4;start+=rows;
 }return fwrite(out,4,8,stdout)==8?0:3;}
`
	kv, score := make([]float32, 20), make([]float32, 20)
	for i := range kv {
		kv[i] = float32(i)*.37 - 2
		score[i] = float32(i%7) * .19
	}
	input := append(append([]float32(nil), kv...), score...)
	want := make([]float32, 8)
	var p Pool
	if err := p.Append(nil, kv[:4], score[:4], 4, 0); err != nil {
		t.Fatal(err)
	}
	if err := p.Append(want, kv[4:16], score[4:16], 4, 1); err != nil {
		t.Fatal(err)
	}
	if err := p.Append(nil, kv[16:], score[16:], 4, 4); err != nil {
		t.Fatal(err)
	}
	got := runLiteral(t, operationPrefix+f+main, input)
	if !equalBits(got, want) {
		t.Fatal("pool literal mismatch", got, want)
	}
	mutant := strings.Replace(f, "prev_kv[col]", "0.0f", 1)
	if mutant == f {
		t.Fatal("carry mutant missing")
	}
	if equalBits(runLiteral(t, operationPrefix+mutant+main, input), want) {
		t.Fatal("carry mutant not detected")
	}
}

// TestEngramPinnedC runs the unchanged Engram kernel, emulating warp reductions
// from the same lane partials. Host rsqrt/exp are not device qualification.
func TestEngramPinnedC(t *testing.T) {
	source := pinned41(t)
	f := literalFunction(t, source, "dsv41_bf16") + literalFunction(t, source, "dsv41_engram_kernel")
	const main = `int main(void){float r[256],kv[320],q[256],k[256];if(fread(r,4,256,stdin)!=256||fread(kv,4,320,stdin)!=320||fread(q,4,256,stdin)!=256||fread(k,4,256,stdin)!=256)return 2;blockIdx.x=0;
 for(unsigned head=0;head<4;head++){float h2[32]={0},k2[32]={0},dot[32]={0};
  for(unsigned lane=0;lane<32;lane++)for(unsigned i=lane;i<64;i+=32){unsigned j=head*64+i;float h=r[j],v=dsv41_bf16(kv[j]);h2[lane]+=h*h;k2[lane]+=v*v;dot[lane]+=h*(q[j]*k[j])*v;}
  for(unsigned d=16;d;d/=2)for(unsigned i=0;i<d;i++){h2[i]+=h2[i+d];k2[i]+=k2[i+d];dot[i]+=dot[i+d];}
  sums[0]=h2[0];sums[1]=k2[0];sums[2]=dot[0];
  for(unsigned lane=0;lane<32;lane++){threadIdx.x=head*32+lane;calls=0;dsv41_engram_kernel(r,kv,q,k,0,64,1e-20f);}
 }return fwrite(r,4,256,stdout)==256?0:3;}
`
	r, kv, q, k := make([]float32, 256), make([]float32, 320), make([]float32, 256), make([]float32, 256)
	for i := range r {
		r[i] = float32(math.Sin(float64(i)*.3)) + .2
		q[i] = float32(i%7)*.03 + .1
		k[i] = float32(i%5)*.09 - .2
	}
	for i := range kv {
		kv[i] = float32(math.Cos(float64(i)*.17)) * 1.3
	}
	input := append(append(append(append([]float32(nil), r...), kv...), q...), k...)
	want := append([]float32(nil), r...)
	if err := AddEngram(want, kv, q, k, 64, 1e-20); err != nil {
		t.Fatal(err)
	}
	got := runLiteral(t, operationPrefix+f+main, input)
	if !equalBits(got, want) {
		t.Fatal("engram literal mismatch")
	}
	mutant := strings.Replace(f, "row * 5u + 4u", "row * 5u + 3u", 1)
	if mutant == f {
		t.Fatal("value operand mutant missing")
	}
	if equalBits(runLiteral(t, operationPrefix+mutant+main, input), want) {
		t.Fatal("value operand mutant not detected")
	}
}
