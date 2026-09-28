package cudaprecompiled_test

import (
	"fmt"
	"testing/fstest"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaprecompiled"
)

// A Module without an image compiles its sources when loaded.
// cudaprecompile writes the same declaration with an image and an
// embed.FS in place of the MapFS.
func Example() {
	precompiled := &cudaprecompiled.Module{
		Files: fstest.MapFS{"scale.cu": {Data: []byte(`
extern "C" __global__ void scale(float* x, float a, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) x[i] *= a;
}`)}},
		Sources: []string{"scale.cu"},
	}

	dev, err := cuda.NewDevice(0)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer dev.Close()
	mod, err := precompiled.Load(dev)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer mod.Close()
	scale, err := mod.Function("scale")
	if err != nil {
		fmt.Println(err)
		return
	}
	_ = scale
}

func ExampleModule_SourceHash() {
	m := &cudaprecompiled.Module{
		Files:   fstest.MapFS{"kernel.cu": {Data: []byte("// source")}},
		Sources: []string{"kernel.cu"},
	}
	a, _ := m.SourceHash()
	m.Options = []string{"-DCOUNT=4"}
	b, _ := m.SourceHash()
	fmt.Println(a == b)
	// Output: false
}
