package cudapipeline_test

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudapipeline"
)

func TestScheduleProtocolRefusals(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*cudapipeline.PipelineSchedule)
	}{
		{"negative stages", func(s *cudapipeline.PipelineSchedule) { s.Stages = -1 }},
		{"huge stages", func(s *cudapipeline.PipelineSchedule) { s.Stages = 1 << 30 }},
		{"unknown kind", func(s *cudapipeline.PipelineSchedule) { s.Steps[2].Kind = "hidden_control_flow" }},
		{"wrong role", func(s *cudapipeline.PipelineSchedule) { s.Steps[2].Role = cudapipeline.RoleConsumer }},
		{"negative bytes", func(s *cudapipeline.PipelineSchedule) {
			for i := range s.Steps {
				if s.Steps[i].Kind == cudapipeline.StepIssueTMA {
					s.Steps[i].Bytes = -1
					return
				}
			}
		}},
		{"foreign epoch", func(s *cudapipeline.PipelineSchedule) {
			for i := range s.Steps {
				if s.Steps[i].Kind == cudapipeline.StepConsumeStage {
					s.Steps[i].Epoch += 2
					return
				}
			}
		}},
		{"undrained", func(s *cudapipeline.PipelineSchedule) { s.Steps = s.Steps[:len(s.Steps)-1] }},
		{"missing consumer", func(s *cudapipeline.PipelineSchedule) { s.Roles = s.Roles[:1] }},
		{"wrong expectation", func(s *cudapipeline.PipelineSchedule) { s.Expectations[0].TransactionBytes++ }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s, err := cudapipeline.BuildWarpSpecializedSchedule(2, 128, 5)
			if err != nil {
				t.Fatal(err)
			}
			tc.mutate(s)
			if len(s.Validate()) == 0 {
				t.Fatal("unsafe schedule accepted")
			}
			if _, _, err := s.CopySources("copy"); err == nil {
				t.Fatal("unsafe schedule emitted")
			}
		})
	}
	for _, args := range [][3]int{{65, 128, 2}, {2, 128, 4097}, {2, 1 << 21, 1}} {
		if _, err := cudapipeline.BuildWarpSpecializedSchedule(args[0], int64(args[1]), args[2]); err == nil {
			t.Fatal("unbounded schedule generated")
		}
	}
}

func TestCopySourceHostOracle(t *testing.T) {
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("host C++ compiler unavailable")
	}
	for _, stages := range []int{2, 3} {
		t.Run(fmt.Sprint(stages), func(t *testing.T) {
			s, err := cudapipeline.BuildWarpSpecializedSchedule(stages, 128, 9)
			if err != nil {
				t.Fatal(err)
			}
			syncSource, asyncSource, err := s.CopySources("copy_fixture")
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(asyncSource, "cp.async.ca.shared.global") || !strings.Contains(asyncSource, "cp.async.wait_group 0;") || strings.Contains(syncSource, "cp.async") {
				t.Fatal("copy mode erased")
			}
			dir := t.TempDir()
			header := `#include <barrier>
#define __global__
#define __shared__ static
#define __align__(n) __attribute__((aligned(n)))
struct dimensions {unsigned int x,y,z;};
static thread_local dimensions threadIdx;
static dimensions blockDim{64,1,1},gridDim{1,1,1};
static std::barrier rendezvous(64);
static void __syncthreads(){rendezvous.arrive_and_wait();}
`
			if err := os.WriteFile(filepath.Join(dir, "cuda_runtime.h"), []byte(header), 0600); err != nil {
				t.Fatal(err)
			}
			main := `#include <thread>
#include <vector>
int main(){
alignas(16) unsigned char input[1152],output[1184];
for(int i=0;i<1152;i++)input[i]=(unsigned char)((i*17+31)%251);
for(int i=0;i<1184;i++)output[i]=219;
std::vector<std::thread> threads;
for(unsigned int i=0;i<64;i++)threads.emplace_back([&,i]{threadIdx={i,0,0};copy_fixture(input,output);});
for(auto& t:threads)t.join();
for(int i=0;i<1152;i++)if(input[i]!=output[i])return 1;
for(int i=1152;i<1184;i++)if(output[i]!=219)return 2;
return 0;}
`
			input := filepath.Join(dir, "source.cc")
			if err := os.WriteFile(input, []byte(syncSource+main), 0600); err != nil {
				t.Fatal(err)
			}
			bin := filepath.Join(dir, "oracle")
			ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
			defer cancel()
			if out, err := exec.CommandContext(ctx, compiler, "-std=c++20", "-pthread", "-I", dir, input, "-o", bin).CombinedOutput(); err != nil {
				t.Fatalf("host source compile: %v %s", err, out)
			}

			if out, err := exec.CommandContext(ctx, bin).CombinedOutput(); err != nil {
				t.Fatalf("independent byte/guard oracle: %v %s", err, out)
			}
		})
	}
}

func ExamplePipelineSchedule_CopySources() {
	s, _ := cudapipeline.BuildWarpSpecializedSchedule(2, 128, 4)
	syncSource, asyncSource, err := s.CopySources("copy")
	fmt.Println(strings.Contains(syncSource, "__syncthreads"), strings.Contains(asyncSource, "cp.async"), err)
	// Output: true true <nil>
}

func TestCopySourcesNativeReference(t *testing.T) {
	if os.Getenv("CUDA_PIPELINE_COPY_NATIVE") != "1" {
		t.Skip("native copy fixture disabled")
	}
	admission, err := pipelineNativeAdmission(os.Getenv)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("inherited_host_lease=%srequested_uuid=%s", admission.Owner.String(), admission.UUID)
	compiler := os.Getenv("NVCC")
	if compiler == "" {
		compiler = "nvcc"
	}
	compiler, err = exec.LookPath(compiler)
	if err != nil {
		t.Fatal(err)
	}
	s, _ := cudapipeline.BuildWarpSpecializedSchedule(3, 128, 9)
	syncSource, asyncSource, err := s.CopySources("copy_async")
	if err != nil {
		t.Fatal(err)
	}
	syncSource = strings.Replace(syncSource, "void copy_async(", "void copy_sync(", 1)
	main := pipelineDeviceAdmissionSource + `
#include <cstdio>
#define CHECK(call) do{cudaError_t e=(call);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));return 10;}}while(0)
int main(int argc, char **argv){
int ordinal = -1; if(argc != 2 || admittedDevice(argv[1], &ordinal) != 0) return 40;
CHECK(cudaSetDevice(ordinal));
unsigned char input[1152],output[1184];for(int i=0;i<1152;i++)input[i]=(i*17+31)%251;
unsigned char *dinput,*doutput;CHECK(cudaMalloc(&dinput,1152));CHECK(cudaMalloc(&doutput,1184));CHECK(cudaMemcpy(dinput,input,1152,cudaMemcpyHostToDevice));
for(int mode=0;mode<2;mode++){
CHECK(cudaMemset(doutput,219,1184));if(mode==0)copy_sync<<<1,64>>>(dinput,doutput);else copy_async<<<1,64>>>(dinput,doutput);
CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(output,doutput,1184,cudaMemcpyDeviceToHost));
for(int i=0;i<1152;i++)if(output[i]!=input[i])return 20+mode;
for(int i=1152;i<1184;i++)if(output[i]!=219)return 30+mode;
}CHECK(cudaFree(doutput));CHECK(cudaFree(dinput));return 0;}
`
	dir := t.TempDir()
	path := filepath.Join(dir, "source.cu")
	if err := os.WriteFile(path, []byte(syncSource+asyncSource+main), 0600); err != nil {
		t.Fatal(err)
	}
	bin := filepath.Join(dir, "copy-fixture")
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++20", "-arch=sm_80", path, "-o", bin, "-lcuda").CombinedOutput(); err != nil {
		t.Fatalf("CUDA compile: %v %s", err, out)
	}
	if current, err := pipelineNativeAdmission(os.Getenv); err != nil || current != admission {
		t.Fatalf("native admission changed before launch: %v", err)
	}
	if out, err := exec.CommandContext(ctx, bin, admission.UUID).CombinedOutput(); err != nil {
		t.Fatalf("native byte/guard oracle: %v %s", err, out)
	}
}
