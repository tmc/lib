package cudalowering

import "testing"

func TestDefinitionLineRequiresADefinition(t *testing.T) {
	const src = `// compute(int*) is declared below and called from main.
#include <cuda.h>

__global__ void compute(int *a);   // forward declaration

void host() {
    compute<<<1, 1>>>(nullptr);
}

__global__ void compute(int *a) {
    *a = 1;
}
`
	line, err := DefinitionLine(src, "compute")
	if err != nil {
		t.Fatalf("DefinitionLine: %v", err)
	}
	if line != 10 {
		t.Errorf("line = %d, want 10 (the definition); a comment, a declaration and a call site all name it", line)
	}
}

func TestDefinitionLineReducesMangledSymbol(t *testing.T) {
	const src = `template <int N, int M>
__global__ void down_q2k_d2r_kernel(float *a) {
    *a = 1;
}
`
	// The CUBIN carries the mangled symbol; the source spells the name alone.
	line, err := DefinitionLine(src, "_Z19down_q2k_d2r_kernelILi64ELi64EEvPf")
	if err != nil {
		t.Fatalf("DefinitionLine: %v", err)
	}
	if line != 2 {
		t.Errorf("line = %d, want 2", line)
	}
}

func TestDefinitionLineRejectsPartialIdentifier(t *testing.T) {
	const src = `__global__ void my_compute(int *a) {
    *a = 1;
}
`
	if _, err := DefinitionLine(src, "compute"); err == nil {
		t.Error("my_compute was accepted as a definition of compute")
	}
}

func TestDefinitionLineRefusesAmbiguity(t *testing.T) {
	const src = `__global__ void compute(int *a) { *a = 1; }
__global__ void compute(float *a) { *a = 1; }
`
	_, err := DefinitionLine(src, "compute")
	if err == nil {
		t.Fatal("two definitions resolved to one line")
	}
	if got := err.Error(); got == "" {
		t.Error("empty error")
	}
}
