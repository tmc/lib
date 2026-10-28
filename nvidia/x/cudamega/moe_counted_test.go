package cudamega

import (
	"crypto/sha256"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudamoe"
)

func TestMoECountedArtifact(t *testing.T) {
	config := MoEWeightConfig{Layers: 3, Experts: 2, Intermediate: 16, Hidden: 17}
	source, err := MoECountedWeightSource(config, "sm_121")
	if err != nil {
		t.Fatal(err)
	}
	makeArtifact := func() *cudakernel.Artifact {
		// Synthetic non-executable ELF metadata, never passed to a driver.
		spec := moeHostArtifact(t).Spec()
		spec.EntryPoint = "moe_weight_counted_worker"
		spec.Dependencies = []cudakernel.Dependency{{Path: "cudamega_moe_counted.cu", Tracked: true, Digest: fmt.Sprintf("%x", sha256.Sum256([]byte(source)))}}
		a, err := cudakernel.NewArtifactFromSpec(spec)
		if err != nil {
			t.Fatal(err)
		}
		return a
	}
	a := makeArtifact()
	original := *a
	owned, err := checkedMoECountedArtifact(config, a)
	if err != nil {
		t.Fatal(err)
	}
	spec := a.Spec()
	spec.Dependencies[0].Digest = "changed"
	other, err := cudakernel.NewArtifactFromSpec(spec)
	if err != nil {
		t.Fatal(err)
	}
	*a = *other
	if *owned != original || owned.Dependencies()[0].Digest == "changed" {
		t.Fatal("source identity follows caller replacement")
	}
	for _, mutate := range []func(*cudakernel.ArtifactSpec){
		func(a *cudakernel.ArtifactSpec) { a.EntryPoint = "moe_weight_worker" },
		func(a *cudakernel.ArtifactSpec) { a.Target = "sm_121a" },
		func(a *cudakernel.ArtifactSpec) { a.Dependencies[0].Digest = "stale" },
		func(a *cudakernel.ArtifactSpec) { a.Dependencies = nil },
		func(a *cudakernel.ArtifactSpec) { a.Dependencies = append(a.Dependencies, a.Dependencies[0]) },
		func(a *cudakernel.ArtifactSpec) { a.CompilerOpts[2] = "--fmad=true" },
	} {
		spec := makeArtifact().Spec()
		mutate(&spec)
		a, err := cudakernel.NewArtifactFromSpec(spec)
		if err != nil {
			t.Logf("constructor refused descriptor: %v", err)
			continue
		}
		if _, err := checkedMoECountedArtifact(config, a); err == nil {
			t.Fatal("wrong counted artifact admitted")
		}
	}
	if _, err := checkedMoECountedArtifact(config, nil); err == nil {
		t.Fatal("nil counted artifact admitted")
	}
}

func TestMoECountedGraph(t *testing.T) {
	config := MoEWeightConfig{Layers: 3, Experts: 2, Intermediate: 8, Hidden: 17}
	p, err := moeCountedProgram(config, "sm_121")
	if err != nil {
		t.Fatal(err)
	}
	if p.grid.Policy != CountedPolicy || p.grid.CTAs != 2 || len(p.steps) != 2 || p.steps[0].Task.Rank != p.steps[1].Task.Rank || p.steps[0].Owner.Count() != 1 || p.steps[1].Owner.Count() != 1 || p.steps[0].Owner.Lo == p.steps[1].Owner.Lo {
		t.Fatal("real expert tasks do not have disjoint counted owners")
	}
	for _, target := range []Target{SM80, SM90, "sm_121a"} {
		if _, err := MoECountedWeightSource(config, target); err == nil {
			t.Fatalf("unqualified target %s accepted", target)
		}
	}
	for _, tc := range []struct {
		blocks           int32
		sms, cooperative int
		valid            bool
	}{{1, 1, 1, false}, {1, 2, 1, true}, {2, 1, 1, true}, {0, 2, 1, false}, {1, 0, 1, false}, {1, 2, 0, false}} {
		if err := checkMoECountedResidency(tc.blocks, tc.sms, tc.cooperative); (err == nil) != tc.valid {
			t.Fatalf("residency %+v: %v", tc, err)
		}
	}
}

func countedBorrowFixture(dev *cuda.Device, expert int) cudamoe.MoEWeightBorrowed {
	return cudamoe.MoEWeightBorrowed{
		Descriptor:  cudamoe.MoEWeightDescriptor{Device: dev, WeightBase: driver.DevicePtr(0x1000), ScaleBase: driver.DevicePtr(0x10000), WeightOffsetBytes: uint64((2*2 + expert) * 16 * 17), ScaleOffsetBytes: uint64((2*2 + expert) * 16 * 4), Layer: 2, Expert: expert, Intermediate: 8, Hidden: 17, Generation: uint64(expert + 1), Format: "e4m3fn-gate-up-interleave8-v1"},
		Shape:       cudamoe.MoEWeightShape{Layers: 3, Experts: 2, Intermediate: 8, Hidden: 17, ActiveLayers: []int{0, 2}},
		WeightBytes: 3 * 2 * 16 * 17, ScaleBytes: 3 * 2 * 16 * 4,
	}
}

func TestMoECountedOwnersAndPublication(t *testing.T) {
	dev := &cuda.Device{} // Metadata only; no constructor or native call.
	c := MoEWeightConfig{Layers: 3, Experts: 2, Intermediate: 8, Hidden: 17}
	owned := []cudamoe.MoEWeightBorrowed{countedBorrowFixture(dev, 0), countedBorrowFixture(dev, 1)}
	if err := checkMoECountedBorrow(c, dev, owned); err != nil {
		t.Fatal(err)
	}
	if err := checkMoECountedBorrow(c, dev, owned[:1]); err == nil {
		t.Fatal("one expert's pin admitted two expert reads")
	}
	for _, mutate := range []func(*cudamoe.MoEWeightBorrowed){
		func(b *cudamoe.MoEWeightBorrowed) { b.Descriptor.Expert = 0 },
		func(b *cudamoe.MoEWeightBorrowed) { b.Descriptor.Layer = 0 },
		func(b *cudamoe.MoEWeightBorrowed) { b.Descriptor.Generation = 0 },
		func(b *cudamoe.MoEWeightBorrowed) { b.Descriptor.WeightBase++ },
		func(b *cudamoe.MoEWeightBorrowed) { b.Descriptor.ScaleBase++ },
		func(b *cudamoe.MoEWeightBorrowed) { b.Descriptor.Device = &cuda.Device{} },
	} {
		bad := append([]cudamoe.MoEWeightBorrowed(nil), owned...)
		mutate(&bad[1])
		if err := checkMoECountedBorrow(c, dev, bad); err == nil {
			t.Fatal("mismatched second owner admitted")
		}
	}
	for _, order := range [][]uint64{{2, 1, 0, 0, 2}, {2, 2, 0, 0, 1}} {
		order = append(order, 2, 1)
		if err := checkMoEPublicationWords([]uint32{1, 1}, []uint32{0}, order); err != nil {
			t.Fatal(err)
		}
	}
	for _, order := range [][]uint64{{2, 1, 0, 0, 1}, {2, 1, 1, 0, 2}, {2, 1, 0, 0, 0}, {1, 1, 0, 0, 2}, {2, 1, 0, 0, 3}} {
		order = append(order, 2, 1)
		if err := checkMoEPublicationWords([]uint32{1, 1}, []uint32{0}, order); err == nil {
			t.Fatal("wrong task/owner/ticket admission")
		}
	}
}

func TestMoECountedEmittedPayload(t *testing.T) {
	source, err := MoECountedWeightSource(MoEWeightConfig{Layers: 3, Experts: 2, Intermediate: 8, Hidden: 17}, "sm_121")
	if err != nil {
		t.Fatal(err)
	}
	ordinary, _, _ := cudamoe.MoEWeightSource(cudamoe.MoEWeightSlabConsumer)
	if !strings.HasPrefix(source, ordinary) {
		t.Fatal("ordinary two-expert source was replaced")
	}
	if strings.Contains(source, "op_0(") || !strings.Contains(source, "if (owns_1 && !done_1)") || !strings.Contains(source, "output[1u*intermediate+neuron]") {
		t.Fatal("real payload or second counted owner missing")
	}
	compiler, err := exec.LookPath("clang++")
	if err != nil {
		t.Fatal("host emitted-payload oracle requires clang++")
	}
	// Serially interpret each lane's emitted payload as host C++. This checks
	// arithmetic, expert addressing and owner guards only: fences/barriers are
	// explicitly erased and therefore have no GPU scheduling proof here.
	// The all-CTA entry protocol cannot run in a serial lane interpreter.
	// Its actual CPU implementation is checked by cudamegacpu entry controls.
	start, end := strings.Index(source, "// entry protocol begin"), strings.Index(source, "// entry protocol end")
	if start < 0 || end < start {
		t.Fatal("entry protocol missing")
	}
	source = source[:start] + source[end+len("// entry protocol end"):]
	source = strings.ReplaceAll(source, `asm volatile("fence.sc.gpu;" ::: "memory");`, "(void)0;")
	stub := `#include <cmath>
#include <cstring>
#define __device__
#define __global__
#define __shared__ static
#define __syncthreads() ((void)0)
struct Dim { unsigned x,y,z; };
Dim gridDim{2,1,1}, blockDim{32,1,1}, blockIdx{0,0,0}, threadIdx{0,0,0};
float __fadd_rn(float a,float b) { volatile float x=a+b; return x; }
float __fmul_rn(float a,float b) { volatile float x=a*b; return x; }
float __int_as_float(unsigned x) { float y; std::memcpy(&y,&x,4); return y; }
unsigned __float_as_int(float x) { unsigned y; std::memcpy(&y,&x,4); return y; }
unsigned atomicCAS(unsigned *p,unsigned old,unsigned v) { unsigned was=*p; if(was==old)*p=v; return was; }
template<class T> T atomicAdd(T *p,T v) { T was=*p; *p+=v; return was; }
`
	main := `
int main() {
  constexpr int I=8,H=17,L=2;
  unsigned char weights[3*2*2*I*H]={}; float scales[3*2*2*I]={},input[H];
  for(int h=0;h<H;++h) input[h]=(h%2 ? -0.03f:0.07f);
  for(int e=0;e<2;++e) for(int n=0;n<I;++n) {
    int row=(L*2+e)*2*I+(n/8)*16+n%8;
    scales[row]=1.0f+float(n)/16; scales[row+8]=0.5f+float(n)/32;
    for(int h=0;h<H;++h) { weights[row*H+h]=e?0x30:0x38; weights[(row+8)*H+h]=e?0x40:0x38; }
  }
  float want[2*I],ordinary[2*I]={},actual[2*I];
  for(int e=0;e<2;++e) for(int n=0;n<I;++n) {
    float gate=0,up=0;
    for(int h=0;h<H;++h) { gate=__fadd_rn(gate,__fmul_rn(input[h],e?0.5f:1.0f)); up=__fadd_rn(up,__fmul_rn(input[h],e?2.0f:1.0f)); }
    gate=__fmul_rn(gate,1.0f+float(n)/16); up=__fmul_rn(up,0.5f+float(n)/32);
    want[e*I+n]=__fmul_rn(gate/__fadd_rn(1.0f,expf(-gate)),up);
  }
  gridDim.x=1;
  for(int e=0;e<2;++e) for(threadIdx.x=0;threadIdx.x<32;++threadIdx.x)
    moe_weight_slab(weights,scales,input,ordinary+e*I,L,e,2,I,H);
  gridDim.x=2; unsigned completed[2]={},abort=0; unsigned long long order[7]={};
  for(float &x:actual) x=-99;
  for(blockIdx.x=0;blockIdx.x<2;++blockIdx.x) for(threadIdx.x=0;threadIdx.x<32;++threadIdx.x)
    moe_weight_counted_worker(weights,scales,input,actual,completed,order,&abort,L,0,I,1);
  if(abort || completed[0]!=1 || completed[1]!=1 || order[0]!=2 || order[1]!=1 || order[2] || order[3] || order[4]!=2) return 1;
  for(int n=0;n<2*I;++n) if(actual[n]!=want[n] || ordinary[n]!=want[n]) return 2;
  // Entry synchronization is erased in this arithmetic-only interpreter.
  threadIdx.x=0; blockIdx.x=0;
  abort=0; completed[0]=0;
  moe_weight_counted_worker(weights,scales,input,actual,completed,order,&abort,L,0,I+1,2);
  if(abort!=3) return 4;
}
`
	dir := t.TempDir()
	path, binary := filepath.Join(dir, "counted.cc"), filepath.Join(dir, "counted")
	if err := os.WriteFile(path, []byte(stub+source+main), 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command(compiler, "-std=c++11", "-ffp-contract=off", path, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("host payload compile: %v\n%s", err, out)
	}
	if out, err := exec.Command(binary).CombinedOutput(); err != nil {
		t.Fatalf("host payload arithmetic/owner guards: %v\n%s", err, out)
	}
}
