package nvjitlinkx_test

import (
	"fmt"
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/nvjitlink"
	"github.com/tmc/lib/nvidia/x/nvjitlinkx"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func ExampleLink() {
	_, err := nvjitlinkx.Link(nil, "-arch=sm_80")
	fmt.Println(err)
	// Output: nvjitlinkx: no inputs
}

// TestLTOPipeline requires a compatible NVRTC/nvJitLink installation, but does
// not load a CUDA module or launch GPU work. It proves linking, not execution.
func TestLTOPipeline(t *testing.T) {
	if os.Getenv("CUDA_COMPILER_TEST") != "1" {
		t.Skip("CUDA_COMPILER_TEST=1 required")
	}
	sources := []string{
		`extern "C" __device__ float twice(float x) { return 2*x; }`,
		`extern "C" __device__ float twice(float);
extern "C" __global__ void apply(float* out) { out[0] = twice(3); }`,
	}
	var inputs []nvjitlinkx.Input
	for i, source := range sources {
		cm, err := nvrtcx.Compile(source, nvrtcx.WithName(fmt.Sprintf("unit%d.cu", i)), nvrtcx.WithArch(8, 0), nvrtcx.WithOutput(nvrtcx.OutputLTOIR), nvrtcx.WithNoCache())
		if err != nil {
			t.Fatal(err)
		}
		if len(cm.LTOIR()) == 0 || cm.PTX() != "" || len(cm.CUBIN()) != 0 {
			t.Fatal("wrong compiler artifact")
		}
		inputs = append(inputs, nvjitlinkx.Input{Name: fmt.Sprintf("unit%d", i), Kind: nvjitlink.InputLtoir, Data: cm.LTOIR()})
	}
	result, err := nvjitlinkx.Link(inputs, "-arch=sm_80", "-lto")
	if result != nil {
		defer func() {
			if err := result.Close(); err != nil {
				t.Error(err)
			}
		}()
	}
	if err != nil {
		t.Fatal(err)
	}
	if len(result.CUBIN) < 4 || string(result.CUBIN[:4]) != "\x7fELF" {
		t.Fatal("no ELF CUBIN")
	}
	bad, err := nvjitlinkx.Link(inputs[1:], "-arch=sm_80", "-lto")
	if bad != nil {
		defer func() {
			if err := bad.Close(); err != nil {
				t.Error(err)
			}
		}()
	}
	if err == nil {
		t.Fatal("unresolved device function linked successfully")
	}
	if bad == nil || bad.ErrorLog == "" {
		t.Fatal("missing unresolved-symbol diagnostic")
	}
}
