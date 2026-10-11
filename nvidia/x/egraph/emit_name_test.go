package egraph

import "testing"

func TestEmitCUDARefusesGeneratedNameCollisions(t *testing.T) {
	for _, name := range []string{"tvec", "uint8_t", "sqrtf", "expf", "erff", "blockIdx", "threadIdx", "blockDim", "gridDim", "warpSize", "i", "and", "static_cast", "thread_local"} {
		t.Run(name, func(t *testing.T) {
			expr := &Expr{Op: "x"}
			if _, err := EmitCUDA(expr, EmitOptions{KernelName: name, Parameters: []Parameter{{Name: "x", Shape: []int{2}}}}); err == nil {
				t.Fatal("accepted conflicting kernel name")
			}
			expr = &Expr{Op: name}
			if _, err := EmitCUDA(expr, EmitOptions{KernelName: "test_kernel", Parameters: []Parameter{{Name: name, Shape: []int{2}}}}); err == nil {
				t.Fatal("accepted conflicting input name")
			}
		})
	}
	if _, err := EmitCUDA(&Expr{Op: "x"}, EmitOptions{KernelName: "test_kernel", Parameters: []Parameter{{Name: "x", Shape: []int{2}}}}); err != nil {
		t.Fatal(err)
	}
}
