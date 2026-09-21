package cudalineinfo_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudalineinfo"
)

func ExampleUnit_SourceWithLineDirectives() {
	u := cudalineinfo.Unit{
		Name:   "scale",
		Source: "__global__ void scale(float *x, int n) {\n    x[threadIdx.x] *= 2.0f;\n}",
		LineMap: map[int]cudair.SourceLocation{
			2: {File: "model.go", Line: 42},
		},
	}
	fmt.Println(u.SourceWithLineDirectives())
	// Output:
	// __global__ void scale(float *x, int n) {
	// #line 42 "model.go"
	//     x[threadIdx.x] *= 2.0f;
	// #line 3 "scale.cu"
	// }
}

func ExampleAttribute() {
	// A unit whose artifacts are not in hand still attributes its CUDA tier.
	u := cudalineinfo.Unit{
		Name:   "scale",
		Source: "__global__ void scale(float *x, int n) {\n    x[threadIdx.x] *= 2.0f;\n}",
		LineMap: map[int]cudair.SourceLocation{
			2: {File: "model.go", Line: 42},
		},
	}
	a := cudalineinfo.Attribute(u, "", nil)
	for _, r := range a.Rows() {
		fmt.Printf("%s: %d generated line(s)\n", r.Site, len(r.CUDA))
	}
	// Output:
	// model.go:42: 1 generated line(s)
}
