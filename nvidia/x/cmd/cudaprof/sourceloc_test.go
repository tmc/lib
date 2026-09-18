package main

import (
	"testing"

	"github.com/google/pprof/profile"
	"reflect"
)

func TestFindFunctionLineRejectsNonDefinitions(t *testing.T) {
	const src = `#include "k.cuh"

// saxpy(a, x, y) scales and adds.
extern "C" __global__ void saxpy(float a, float *x, float *y);

void host() {
    saxpy(2.0f, x, y);
}

extern "C" __global__ void saxpy(float a, float *x, float *y) {
    y[0] = a * x[0];
}
`
	// Line 10 is the definition. A comment (line 3), a declaration (line 4)
	// and a call site (line 7) all contain "saxpy(" but none of them is where
	// the kernel's code lives.
	if got := findFunctionLine(src, "saxpy"); got != 10 {
		t.Errorf("findFunctionLine = %d, want 10 (the definition)", got)
	}
}

func TestFindFunctionLineRequiresWholeIdentifier(t *testing.T) {
	const src = `extern "C" __global__ void my_saxpy(float *y) { y[0] = 1.0f; }
`
	// "my_saxpy(" contains "saxpy(", but this file does not define saxpy.
	if got := findFunctionLine(src, "saxpy"); got != 0 {
		t.Errorf("findFunctionLine matched a longer identifier at line %d", got)
	}
}

func TestResolveSourceLocationsAmbiguousStaysUnknown(t *testing.T) {
	const def = "extern \"C\" __global__ void saxpy(float *y) { y[0] = 1.0f; }\n"
	// Two capture files define the same kernel name. Picking either one is a
	// guess, and map iteration order made which one nondeterministic.
	files := map[string][]byte{"a.cu": []byte(def), "b.cu": []byte(def)}
	fn := &profile.Function{Name: "saxpy"}
	p := &profile.Profile{Function: []*profile.Function{fn}}

	resolveSourceLocations(p, files, "")

	if fn.Filename != "" || fn.StartLine != 0 {
		t.Errorf("ambiguous name resolved to %s:%d, want unknown", fn.Filename, fn.StartLine)
	}
}

func TestResolveSourceLocationsUniqueMatchResolves(t *testing.T) {
	files := map[string][]byte{
		"a.cu": []byte("extern \"C\" __global__ void other(float *y) { y[0] = 1.0f; }\n"),
		"b.cu": []byte("\nextern \"C\" __global__ void saxpy(float *y) { y[0] = 1.0f; }\n"),
	}
	fn := &profile.Function{Name: "saxpy"}
	p := &profile.Profile{Function: []*profile.Function{fn}}

	resolveSourceLocations(p, files, "/caps")

	if fn.Filename != "/caps/b.cu" || fn.StartLine != 2 {
		t.Errorf("resolved to %s:%d, want /caps/b.cu:2", fn.Filename, fn.StartLine)
	}
}

func TestInferCodeFilesIsDeterministic(t *testing.T) {
	files := map[string][]byte{
		"a.ptx": nil, "b.ptx": nil, "c.ptx": nil,
		"a.cu": nil, "b.cu": nil,
		"a.sass": nil, "b.sass": nil,
		"a.cubin": nil, "b.cubin": nil,
	}
	// Selection used to depend on map iteration order, so two runs over the
	// same bundle could disassemble one PTX and show another's source.
	want := inferCodeFiles(files)
	for i := 0; i < 50; i++ {
		if got := inferCodeFiles(files); !reflect.DeepEqual(got, want) {
			t.Fatalf("run %d selected %+v, first run selected %+v", i, got, want)
		}
	}
}

func TestInferCodeFilesPrefersCanonicalNames(t *testing.T) {
	files := map[string][]byte{"a.ptx": nil, "kernels.ptx": nil, "z.ptx": nil}
	if got := inferCodeFiles(files).PTX; got != "kernels.ptx" {
		t.Errorf("PTX = %q, want kernels.ptx", got)
	}
}

func TestFindFunctionLineIgnoresTrailingComment(t *testing.T) {
	// The ';' test runs on the raw line, so a declaration annotated with a
	// trailing comment does not end in ';' and was accepted as a definition.
	const content = `#include <cuda.h>

__global__ void saxpy(int n, float a);   // forward declaration

__global__ void saxpy(int n, float a) {
    (void)n;
}
`
	if got := findFunctionLine(content, "saxpy"); got != 5 {
		t.Errorf("findFunctionLine = %d, want 5 (the definition, not the commented declaration)", got)
	}
}
