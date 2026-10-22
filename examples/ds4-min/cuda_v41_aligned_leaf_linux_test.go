//go:build linux

package main

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strings"
	"testing"
)

func checkV41AlignedLeaf(source string) error {
	f, err := parser.ParseFile(token.NewFileSet(), "aligned.go", source, 0)
	if err != nil {
		return err
	}
	var failure error
	ast.Inspect(f, func(n ast.Node) bool {
		id, ok := n.(*ast.Ident)
		if !ok {
			return true
		}
		name := id.Name
		if (strings.Contains(strings.ToLower(name), "layer0") && name != "cudaV41Layer0Q8Weight" && name != "cudaV41Layer0FloatWeight") || strings.Contains(strings.ToLower(name), "v41raw") {
			failure = fmt.Errorf("executable layer0/raw coupling: %s", name)
			return false
		}
		return true
	})
	return failure
}

func TestCUDAV41AlignedLeafSource(t *testing.T) {
	for _, name := range []string{"cuda_v41_weight_linux.go", "cuda_v41_compile_linux.go", "cuda_v41_ops_linux.go", "cuda_v41_aligned_q8_linux.go", "cuda_v41_aligned_roles_linux.go", "cuda_v41_aligned_q8_linux_test.go", "cuda_v41_aligned_roles_linux_test.go", "cuda_v41_aligned_literal_linux_test.go", "cuda_v41_aligned_capture_linux_test.go"} {
		b, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		if err := checkV41AlignedLeaf(string(b)); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
	}
	// Inject executable dependencies into the actual aligned constructor source;
	// this checks source coupling, not numerical mutation or a compile failure.
	b, err := os.ReadFile("cuda_v41_aligned_q8_linux.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, symbol := range []string{"newCUDAV41Layer0", "compileCUDAV41RawAttention"} {
		source := string(b)
		const site = "return compileCUDAV41AlignedQ8(r, w, cudaV41AlignedQ8Source)"
		if strings.Count(source, site) != 1 {
			t.Fatal("aligned constructor mutation site not unique")
		}
		source = strings.Replace(source, site, "_ = "+symbol+"; "+site, 1)
		if checkV41AlignedLeaf(source) == nil {
			t.Fatalf("coupling mutant %s survived", symbol)
		}
	}
}
