package clangcuda

import (
	"context"
	"strings"
	"testing"
	"unsafe"
)

func TestUnsavedFiles(t *testing.T) {
	files, names, contents, keep, err := makeUnsavedFiles("kernel.cu", "#include \"types.cuh\"", []File{{Name: "types.cuh", Contents: "struct Pair {};"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(files) != 2 || names[1] != "types.cuh" || contents[1] != "struct Pair {};" {
		t.Fatalf("unsaved files = %#v %#v %#v", files, names, contents)
	}
	if files[0].length != uint64(len("#include \"types.cuh\"")) || files[1].length != uint64(len("struct Pair {};")) {
		t.Fatalf("unsaved lengths = %d, %d", files[0].length, files[1].length)
	}
	if len(keep) != 4 || files[0].filename == nil || files[1].contents == nil {
		t.Fatalf("unsaved storage not retained: %#v", keep)
	}
	if _, _, _, _, err := makeUnsavedFiles("kernel.cu", "", []File{{Name: "kernel.cu", Contents: ""}}); err == nil {
		t.Fatal("duplicate unsaved file accepted")
	}
	if _, _, _, _, err := makeUnsavedFiles("kernel.cu", "", []File{{Name: "bad\x00.cuh"}}); err == nil {
		t.Fatal("NUL unsaved filename accepted")
	}
}

func TestABISizes(t *testing.T) {
	if got := unsafe.Sizeof(cxCursor{}); got != 32 && got != 16 {
		t.Errorf("CXCursor size = %d, want a supported 32/64-bit layout", got)
	}
	if got := unsafe.Sizeof(cxString{}); got != 16 && got != 8 {
		t.Errorf("CXString size = %d, want a supported 32/64-bit layout", got)
	}
}

func TestParseCUDAWhenLibclangAvailable(t *testing.T) {
	index, err := Open()
	if err != nil {
		t.Skipf("libclang unavailable: %v", err)
	}
	defer index.Close()
	unit, err := index.Parse(context.Background(), `
#include "/virtual/types.cuh"
extern "C" __global__ void kernel(Pair *p) { p[0].x = 1; }
`, Options{
		Name:  "kernel.cu",
		Args:  []string{"-nocudainc", "-nocudalib", "--cuda-gpu-arch=sm_70"},
		Files: []File{{Name: "/virtual/types.cuh", Contents: "struct Pair { int x; float y; };"}},
	})
	if err != nil {
		if strings.Contains(err.Error(), "unknown argument") || strings.Contains(err.Error(), "cannot find") || strings.Contains(err.Error(), "unknown type name '__global__'") {
			t.Skipf("libclang cannot parse CUDA on this host: %v", err)
		}
		t.Fatal(err)
	}
	var pair, kernel bool
	for _, declaration := range unit.Declarations {
		pair = pair || declaration.Name == "Pair"
		kernel = kernel || declaration.Name == "kernel"
	}
	if !pair || !kernel {
		t.Fatalf("declarations = %+v", unit.Declarations)
	}
}
