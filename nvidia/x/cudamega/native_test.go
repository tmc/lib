package cudamega_test

import (
	"context"
	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

func programFixture(t *testing.T) (cudataskgraph.Graph, map[string]cudamega.Operation) {
	t.Helper()
	g, err := cudataskgraph.New([]cudataskgraph.Task{{ID: "add3", Rank: 0, Scope: cudataskgraph.ScopeCTA}, {ID: "multiply2", Deps: []string{"add3"}, Rank: 1, Scope: cudataskgraph.ScopeCTA}})
	if err != nil {
		t.Fatal(err)
	}
	return g, map[string]cudamega.Operation{"add3": {Kind: cudamega.Add, Scalar: 3}, "multiply2": {Kind: cudamega.Multiply, Scalar: 2}}
}

func TestNativePersistentReference(t *testing.T) {
	if os.Getenv("CUDA_MEGA_NATIVE") != "1" {
		t.Skip("native qualification opt-in: CUDA_MEGA_NATIVE=1")
	}
	if os.Getenv("GPU_LOCK_HELD") != "1" {
		t.Fatal("CUDA_MEGA_NATIVE=1 requires GPU_LOCK_HELD=1")
	}
	if os.Getenv("CUDA_MEGA_NATIVE_ARCH") != "sm_90" {
		t.Fatal("CUDA_MEGA_NATIVE=1 requires CUDA_MEGA_NATIVE_ARCH=sm_90")
	}
	nvcc, err := exec.LookPath("nvcc")
	if err != nil {
		t.Fatal(err)
	}
	g, ops := programFixture(t)
	src, err := cudamega.ProgramSource(g, ops)
	if err != nil {
		t.Fatal(err)
	}
	src += `
int main() { const unsigned n=32; unsigned h[n], a[n], b[n]; for(unsigned i=0;i<n;i++) h[i]=i;
unsigned *in=0,*scratch=0,*completed=0,*abort=0; if(cudaMalloc(&in,sizeof h)||cudaMalloc(&scratch,2*n*sizeof(unsigned))||cudaMalloc(&completed,2*sizeof(unsigned))||cudaMalloc(&abort,sizeof(unsigned))) return 2;
cudaMemset(completed,0,2*sizeof(unsigned)); cudaMemset(abort,0,sizeof(unsigned)); cudaMemcpy(in,h,sizeof h,cudaMemcpyHostToDevice);
int blocks=0; if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,dag_worker,128,0)!=cudaSuccess||blocks<1)return 3;
dag_worker<<<1,128>>>(in,scratch,completed,abort,n,1); if(cudaDeviceSynchronize())return 4; cudaMemcpy(a,scratch+n,sizeof a,cudaMemcpyDeviceToHost);
cudaMemset(scratch,0,2*n*sizeof(unsigned)); ordinary_task<<<1,128>>>(in,scratch,n,0); ordinary_task<<<1,128>>>(in,scratch,n,1); if(cudaDeviceSynchronize())return 5; cudaMemcpy(b,scratch+n,sizeof b,cudaMemcpyDeviceToHost);
for(unsigned i=0;i<n;i++) { unsigned want=(h[i]+3)*2; if(a[i]!=want||b[i]!=want||a[i]!=b[i])return 6; }
cudaMemset(scratch,0xa5,2*n*sizeof(unsigned)); cudaMemset(abort,1,sizeof(unsigned));
dag_worker<<<1,128>>>(in,scratch,completed,abort,n,2); if(cudaDeviceSynchronize())return 7;
cudaMemcpy(a,scratch+n,sizeof a,cudaMemcpyDeviceToHost); for(unsigned i=0;i<n;i++) if(a[i]!=0xa5a5a5a5u)return 8;
cudaFree(in);cudaFree(scratch);cudaFree(completed);cudaFree(abort);return 0; }
`
	home, _ := os.UserHomeDir()
	dir, err := os.MkdirTemp(filepath.Join(home, "tmp"), "cudamega-native-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })
	cu, exe := filepath.Join(dir, "fixture.cu"), filepath.Join(dir, "fixture")
	if err := os.WriteFile(cu, []byte(src), 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command(nvcc, "-arch=sm_90", cu, "-o", exe).CombinedOutput(); err != nil {
		t.Fatalf("compile: %v\n%s", err, out)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, exe).CombinedOutput(); err != nil {
		t.Fatalf("native fixture: %v\n%s", err, out)
	}
}

func TestCompileProgramSource(t *testing.T) {
	if os.Getenv("CUDA_MEGA_COMPILE") != "1" {
		t.Skip("compile qualification opt-in: CUDA_MEGA_COMPILE=1")
	}
	nvcc, err := exec.LookPath("nvcc")
	if err != nil {
		t.Fatal(err)
	}
	g, ops := programFixture(t)
	src, err := cudamega.ProgramSource(g, ops)
	if err != nil {
		t.Fatal(err)
	}
	home, _ := os.UserHomeDir()
	dir, err := os.MkdirTemp(filepath.Join(home, "tmp"), "cudamega-compile-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })
	cu := filepath.Join(dir, "fixture.cu")
	if err := os.WriteFile(cu, []byte(src), 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command(nvcc, "-arch=sm_90", "-c", cu, "-o", filepath.Join(dir, "fixture.o")).CombinedOutput(); err != nil {
		t.Fatalf("compile: %v\n%s", err, out)
	}
}
