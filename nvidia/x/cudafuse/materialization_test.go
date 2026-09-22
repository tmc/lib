package cudafuse_test

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudafuse"
	"github.com/tmc/lib/nvidia/x/cudair"
)

func TestSecondaryConsumerMaterialization(t *testing.T) {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{5}, cudair.Float32)
	y := g.AddInput("y", []int{5}, cudair.Float32)
	sum := g.AddBinary("add", x, y)
	product := g.AddBinary("mul", sum, y)
	unused := g.AddBinary("sub", x, y)
	if err := g.SetOutputs(sum, x, product); err != nil {
		t.Fatal(err)
	}
	fk, err := cudafuse.FuseWithSourceMap(g, "materialize")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(fk.Outputs, []string{sum.Name, x.Name, product.Name}) {
		t.Fatalf("outputs=%v", fk.Outputs)
	}
	if strings.Contains(fk.Source, "out_"+unused.Name) {
		t.Fatal("materialized an unrequested terminal")
	}
	compiler, err := exec.LookPath("clang++")
	if err != nil {
		t.Skip("host emitted-source check requires clang++")
	}
	for _, mutant := range []string{"", "missing store", "index overflow"} {
		src := fk.Source
		if mutant == "missing store" {
			store := "out_" + sum.Name + "[idx] = v_" + sum.Name + ";"
			if !strings.Contains(src, store) {
				t.Fatal("missing mutation target")
			}
			src = strings.Replace(src, store, "/* omitted secondary-consumer value */", 1)
		}
		if mutant == "index overflow" {
			index := "(unsigned long long)blockDim.x * blockIdx.x"
			if !strings.Contains(src, index) {
				t.Fatal("missing index mutation target")
			}
			src = strings.Replace(src, index, "blockDim.x * blockIdx.x", 1)
		}
		dir := t.TempDir()
		header := `#define __global__
struct dim {unsigned int x=1,y=1,z=1;};
extern dim blockDim,blockIdx,threadIdx,gridDim;
`
		main := `
dim blockDim,blockIdx,threadIdx,gridDim;
int main(){
  float x[5]={-2,-1,0,1,2},y[5]={2,2,2,2,2};
  float sum[7],copy[7],product[7];
  for(int i=0;i<7;i++)sum[i]=copy[i]=product[i]=-99;
  blockDim.x=3;gridDim.x=2;
  for(blockIdx.x=0;blockIdx.x<2;blockIdx.x++)
    for(threadIdx.x=0;threadIdx.x<3;threadIdx.x++)
      materialize(x,y,sum+1,copy+1,product+1,5);
  for(int i=0;i<5;i++)if(sum[i+1]!=i || copy[i+1]!=i-2 || product[i+1]!=2*i)return 1;
  for(int i: {0,6})if(sum[i]!=-99 || copy[i]!=-99 || product[i]!=-99)return 2;
  // An initial index above 32 bits must not wrap into the output.
  blockDim.x=8;blockIdx.x=1u<<30;threadIdx.x=0;gridDim.x=(1u<<30)+1;
  sum[1]=-88;
  materialize(x,y,sum+1,copy+1,product+1,5);
  if(sum[1]!=-88)return 3;
  // The stride itself may exceed 32 bits; exactly one element is visited.
  blockDim.x=65536;gridDim.x=65536;blockIdx.x=0;
  materialize(x,y,sum+1,copy+1,product+1,5);
  if(sum[1]!=0)return 4;
  // Negative lengths and non-1D launches must not dereference pointers.
  materialize(nullptr,nullptr,nullptr,nullptr,nullptr,-1);
  gridDim.y=2;
  materialize(nullptr,nullptr,nullptr,nullptr,nullptr,5);
}
`
		// initializer_list supplies the two guard indices without CUDA headers.
		src = "#include <initializer_list>\n" + src + main
		if err := os.WriteFile(filepath.Join(dir, "cuda_runtime.h"), []byte(header), 0600); err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(dir, "materialize.cc")
		if err := os.WriteFile(path, []byte(src), 0600); err != nil {
			t.Fatal(err)
		}
		binary := filepath.Join(dir, "materialize")
		if out, err := exec.Command(compiler, "-std=c++17", "-O1", "-I", dir, path, "-o", binary).CombinedOutput(); err != nil {
			t.Fatalf("compile: %v\n%s", err, out)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		out, err := exec.CommandContext(ctx, binary).CombinedOutput()
		cancel()
		if (err != nil) != (mutant != "") {
			t.Fatalf("mutant=%q error=%v output=%s", mutant, err, out)
		}
	}
}

func TestFuseOutputAdmission(t *testing.T) {
	if _, err := cudafuse.Fuse(nil, "test"); err == nil {
		t.Fatal("accepted nil graph")
	}
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{1}, cudair.Float32)
	if _, err := cudafuse.Fuse(g, "test"); err == nil {
		t.Fatal("accepted graph with no outputs")
	}
	if err := g.SetOutputs(x, x); err != nil {
		t.Fatal(err)
	}
	if _, err := cudafuse.Fuse(g, "test"); err == nil {
		t.Fatal("accepted duplicate outputs")
	}
}
