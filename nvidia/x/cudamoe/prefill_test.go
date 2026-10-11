package cudamoe

import (
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudakernel"
)

func TestPrefillW256Layouts(t *testing.T) {
	for _, dims := range [][2]int{{0, 16}, {16, 0}, {17, 16}, {16, 17}, {-16, 16}, {math.MaxInt - 15, math.MaxInt - 15}} {
		if _, err := PrefillW256Layouts(dims[0], dims[1]); err == nil {
			t.Fatalf("accepted dimensions %v", dims)
		}
	}
	tiles, err := PrefillW256Layouts(32, 48)
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []int{256 * 32 * 2, 32 * 48 * 2, 256 * 48 * 4} {
		got, err := tiles[i].FootprintBytes()
		if err != nil || got != want || tiles[i].AlignmentBytes() != 32 {
			t.Fatalf("matrix %d: bytes=%d err=%v", i, got, err)
		}
	}
	schema, err := prefillW256Schema()
	if err != nil || schema.TotalParamBytes != 32 || len(schema.Params) != 5 {
		t.Fatalf("schema=%+v err=%v", schema, err)
	}
}

func TestPrefillArtifactsRefuseTargets(t *testing.T) {
	for _, tc := range []struct {
		name    string
		compile func(string) (*cudakernel.Artifact, error)
	}{
		{"scalar", PrefillW256ScalarArtifact},
		{"wmma", PrefillW256Artifact},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, target := range []string{"", "sm_90", "sm_90a", "sm_100", "sm_121", "compute_100a"} {
				if _, err := tc.compile(target); err == nil || !strings.Contains(err.Error(), "unsupported prefill target") {
					t.Fatalf("compile %q = %v, want unsupported prefill target", target, err)
				}
			}
		})
	}
}

func ExamplePrefillW256Layouts() {
	tiles, err := PrefillW256Layouts(32, 48)
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, tile := range tiles {
		size, _ := tile.FootprintBytes()
		fmt.Println(size)
	}
	// Output:
	// 16384
	// 3072
	// 49152
}

// The host WMMA shim computes whole fragments once per simulated warp. It
// checks emitted matrix indexing and ownership, not physical fragment mapping,
// floating-point Tensor Core behavior, warp convergence or CUDA compilation.
func TestPrefillW256EmittedIndexing(t *testing.T) {
	compiler, err := exec.LookPath("clang++")
	if err != nil {
		t.Skip("requires host clang++")
	}
	for _, tc := range []struct {
		name, old, replacement string
		fail                   bool
	}{
		{name: "source"},
		{name: "overlapping_warps", old: "(threadIdx.x / 32) * 16", replacement: "0", fail: true},
		{name: "skipped_reduction_tile", old: "p += 16", replacement: "p += 32", fail: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			src, _ := PrefillW256Source()
			if tc.old != "" {
				if !strings.Contains(src, tc.old) {
					t.Fatal("mutation target missing")
				}
				src = strings.ReplaceAll(src, tc.old, tc.replacement)
			}
			for name, contents := range map[string]string{
				"cuda_fp16.h": "#pragma once\nusing half = _Float16;\nstatic_assert(sizeof(half)==2,\"half size\");\n",
				"mma.h":       prefillHostWMMA,
				"prefill.cc":  src + prefillHostMain,
			} {
				if err := os.WriteFile(filepath.Join(dir, name), []byte(contents), 0600); err != nil {
					t.Fatal(err)
				}
			}
			binary := filepath.Join(dir, "prefill")
			if out, err := exec.Command(compiler, "-std=c++17", "-O1", "-I", dir, filepath.Join(dir, "prefill.cc"), "-o", binary).CombinedOutput(); err != nil {
				t.Fatalf("compile: %v\n%s", err, out)
			}
			out, err := exec.Command(binary).CombinedOutput()
			if (err != nil) != tc.fail {
				t.Fatalf("run: %v want failure=%v\n%s", err, tc.fail, out)
			}
		})
	}
}

func TestPrefillW256Compile(t *testing.T) {
	if os.Getenv("CUDAMOE_COMPILE") != "1" {
		t.Skip("set CUDAMOE_COMPILE=1 and CUDAMOE_TARGET for CUDA compilation")
	}
	for _, tc := range []struct {
		name    string
		compile func(string) (*cudakernel.Artifact, error)
	}{
		{"scalar", PrefillW256ScalarArtifact},
		{"wmma", PrefillW256Artifact},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a, err := tc.compile(os.Getenv("CUDAMOE_TARGET"))
			if err != nil {
				t.Fatal(err)
			}
			info, err := cudakernel.ParseCUBIN(a.Raw)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Contains(info.Symbols, a.EntryPoint) {
				t.Fatal("missing compiled entry")
			}
			t.Logf("target=%s entry=%s digest=%s resources=%+v options=%v", a.Target, a.EntryPoint, a.Digest, a.Resources, a.CompilerOpts)
		})
	}
}

const prefillHostWMMA = `
#include <cstddef>
#define __global__
struct dim { unsigned x=1,y=1,z=1; };
extern dim blockDim,blockIdx,threadIdx;
namespace nvcuda { namespace wmma {
struct matrix_a {}; struct matrix_b {}; struct accumulator {}; struct row_major {};
constexpr int mem_row_major=0;
template<class Role,int M,int N,int K,class T,class Layout=void>
struct fragment { T data[256]; };
template<class F> void fill_fragment(F& f,float v) {for(int i=0;i<256;i++)f.data[i]=v;}
template<class F,class T> void load_matrix_sync(F& f,const T* p,unsigned ld) {
  for(int r=0;r<16;r++)for(int c=0;c<16;c++)f.data[r*16+c]=p[(size_t)r*ld+c];
}
template<class D,class A,class B,class C> void mma_sync(D& d,const A& a,const B& b,const C& c) {
  float values[256];
  for(int r=0;r<16;r++)for(int col=0;col<16;col++) {
    float sum=c.data[r*16+col];
    for(int p=0;p<16;p++)sum+=(float)a.data[r*16+p]*(float)b.data[p*16+col];
    values[r*16+col]=sum;
  }
  for(int i=0;i<256;i++)d.data[i]=values[i];
}
template<class F> void store_matrix_sync(float* p,const F& f,unsigned ld,int) {
  for(int r=0;r<16;r++)for(int c=0;c<16;c++)p[(size_t)r*ld+c]=f.data[r*16+c];
}
}}
`

const prefillHostMain = `
#include <algorithm>
#include <cstdio>
dim blockDim,blockIdx,threadIdx;
int main() {
  alignas(32) half a[256*48],b[48*48];
  alignas(32) float c[256*48];
  for(unsigned k: {16u,32u,48u})for(unsigned n: {16u,32u,48u}) {
    for(unsigned i=0;i<256*k;i++)a[i]=(int)(i%7)-3;
    for(unsigned i=0;i<k*n;i++)b[i]=(int)((i*3)%11)-5;
    unsigned writes[256*48]={};
    blockDim={128,1,1}; threadIdx={0,0,0};
    for(unsigned y=0;y<4;y++)for(unsigned x=0;x<n/16;x++)for(unsigned warp=0;warp<4;warp++) {
      std::fill(c,c+256*48,1234567.0f);
      blockIdx={x,y,0};threadIdx.x=warp*32;
      moe_prefill_w256(a,b,c,k,n);
      for(unsigned i=0;i<256*48;i++)if(c[i]!=1234567.0f) {
        if(i>=256*n)return 2;
        double expected=0;
        for(unsigned p=0;p<k;p++)expected+=(double)a[(i/n)*k+p]*(double)b[p*n+i%n];
        if(c[i]!=expected){std::fprintf(stderr,"wrong output k=%u n=%u index=%u\n",k,n,i);return 3;}
        writes[i]++;
      }
    }
    for(unsigned i=0;i<256*n;i++)if(writes[i]!=1){std::fprintf(stderr,"writers=%u index=%u\n",writes[i],i);return 4;}
  }
}
`
