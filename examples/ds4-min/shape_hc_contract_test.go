package main

import (
	"crypto/sha256"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestShapeHCSource(t *testing.T) {
	pinned, err := os.ReadFile("testdata/shape-hc-upstream.cu")
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(pinned)) != "3f3d0b79260f204fc5b7fa0ca68063b95d8fc7236254b4d84aa6aa57c070cd03" {
		t.Fatal("pinned HC fixture changed")
	}
	src, err := os.ReadFile("cuda_shape_hc.cu")
	if err != nil {
		t.Fatal(err)
	}
	want := strings.ReplaceAll(string(pinned), "__global__ static void hc_weighted_sum_kernel", "extern \"C\" __global__ void ds4_shape_hc_weighted_sum")
	want = strings.ReplaceAll(want, "__global__ static void hc_expand_kernel", "extern \"C\" __global__ void ds4_shape_hc_expand")
	if !strings.HasSuffix(string(src), want) {
		t.Fatal("HC CUDA bodies differ from pinned source")
	}
}

func TestShapeHCValues(t *testing.T) {
	sourcePath := "cuda_shape_hc.cu"
	if p := os.Getenv("DS4_SHAPE_HC_TEST_SOURCE"); p != "" {
		sourcePath = p
	}
	src, err := os.ReadFile(sourcePath)
	if err != nil {
		t.Fatal(err)
	}
	cc, err := exec.LookPath("c++")
	if err != nil {
		t.Fatal(err)
	}
	prefix := `#include <cmath>
#include <cstdio>
#include <vector>
#define __global__
struct Dim {unsigned x;}; static Dim blockIdx,blockDim,threadIdx;
`
	suffix := `
int main(){
 for(unsigned width: {4096u,7168u}) {
  const unsigned hc=4; std::vector<float>x(width*hc),out(width),next(width*hc),block(width);
  float pre[4]={0.13f,0.27f,-0.18f,0.79f},post[4]={0.8f,-0.2f,0.3f,0.65f},comb[16];
  for(unsigned i=0;i<x.size();i++)x[i]=float(int(i%73)-31)/37.0f;
  for(unsigned i=0;i<width;i++)block[i]=float(int(i%41)-13)/23.0f;
  for(unsigned i=0;i<16;i++)comb[i]=float(int(i)-5)/19.0f;
  blockDim.x=256;
  for(unsigned i=0;i<width;i++){blockIdx.x=i/256;threadIdx.x=i%256;ds4_shape_hc_weighted_sum(out.data(),x.data(),pre,width,hc,1,4);}
  for(unsigned d=0;d<width;d++){double want=0;for(unsigned h=0;h<hc;h++)want+=double(x[h*width+d])*pre[h];if(std::fabs(out[d]-want)>2e-6){std::printf("HC_COLLAPSE_MISMATCH width=%u d=%u\n",width,d);return 1;}}
  for(unsigned i=0;i<next.size();i++){blockIdx.x=i/256;threadIdx.x=i%256;ds4_shape_hc_expand(next.data(),block.data(),nullptr,nullptr,x.data(),post,comb,width,hc,1,4,16,0,0);}
  for(unsigned h=0;h<hc;h++)for(unsigned d=0;d<width;d++){double want=double(block[d])*post[h];for(unsigned s=0;s<hc;s++)want+=double(comb[h+s*hc])*x[s*width+d];if(std::fabs(next[h*width+d]-want)>2e-6){std::printf("HC_EXPAND_MISMATCH width=%u h=%u d=%u\n",width,h,d);return 1;}}
  std::printf("HC_VALUES_PASS width=%u collapse=%u expand=%u\n",width,width,width*hc);
 }
}
`
	d := t.TempDir()
	input, out := filepath.Join(d, "hc.cpp"), filepath.Join(d, "hc")
	if err := os.WriteFile(input, []byte(prefix+string(src)+suffix), 0600); err != nil {
		t.Fatal(err)
	}
	if b, err := exec.Command(cc, "-std=c++11", input, "-o", out).CombinedOutput(); err != nil {
		t.Fatalf("compile: %v %s", err, b)
	}
	if b, err := exec.Command(out).CombinedOutput(); err != nil {
		t.Fatalf("HC value oracle: %v %s", err, b)
	} else {
		t.Log(string(b))
	}
}
