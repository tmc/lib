package cudamoe

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestPrefillScalarEmittedNumerics(t *testing.T) {
	compiler, err := exec.LookPath("clang++")
	if err != nil {
		t.Skip("requires host clang++")
	}
	dir := t.TempDir()
	header := `#include <stddef.h>
#define __global__
using half = _Float16;
inline float __half2float(half h){return (float)h;}
struct dim {unsigned x=1,y=1,z=1;};
extern dim blockIdx,blockDim,threadIdx;
`
	main := `
#include <algorithm>
dim blockIdx,blockDim,threadIdx;
int main(){
  half a[256*48],b[48*48]; float output[256*48+2];
  for(unsigned k: {16u,32u,48u})for(unsigned n: {16u,32u,48u}){
    for(unsigned i=0;i<256*k;i++)a[i]=(int)(i%7)-3;
    for(unsigned i=0;i<k*n;i++)b[i]=(int)((i*3)%11)-5;
    std::fill(output,output+256*48+2,1234567.0f);
    blockDim={128,1,1};
    for(unsigned i=0;i<256*n+17;i++){
      blockIdx={i/128,0,0};threadIdx={i%128,0,0};
      moe_prefill_scalar(a,b,output+1,k,n);
    }
    for(unsigned r=0;r<256;r++)for(unsigned c=0;c<n;c++){
      double expected=0;
      for(unsigned p=0;p<k;p++)expected+=(double)a[r*k+p]*(double)b[p*n+c];
      if(output[1+r*n+c]!=expected)return 1;
    }
    if(output[0]!=1234567.0f)return 2;
    for(unsigned i=1+256*n;i<256*48+2;i++)if(output[i]!=1234567.0f)return 3;
  }
}
`
	if err := os.WriteFile(filepath.Join(dir, "cuda_fp16.h"), []byte(header), 0600); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "scalar.cc")
	if err := os.WriteFile(path, []byte(prefillW256ScalarSource+main), 0600); err != nil {
		t.Fatal(err)
	}
	binary := filepath.Join(dir, "scalar")
	if out, err := exec.Command(compiler, "-std=c++17", "-O1", "-I", dir, path, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile: %v\n%s", err, out)
	}
	if out, err := exec.Command(binary).CombinedOutput(); err != nil {
		t.Fatalf("run: %v\n%s", err, out)
	}
}
