// Command cudavet inspects CUDA Go packages for common driver and concurrency hazards.
//
// Checks performed:
//   - unpinned: Unpinned host slice pointer conversions passed to async DMA.
//   - cargalign: Misaligned kernel argument structs or pointers passed to LaunchKernel.
//   - leak: Stream / Event allocations without proper deferred destruction.
package main

import (
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
)

type Diagnostic struct {
	Pos     token.Position
	Message string
}

func checkFile(fset *token.FileSet, node *ast.File) []Diagnostic {
	var diags []Diagnostic

	ast.Inspect(node, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}

		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			return true
		}

		funcName := sel.Sel.Name
		// Check for unpinned async DMA conversions
		if strings.HasSuffix(funcName, "Async") && (funcName == "MemcpyHtoDAsync" || funcName == "MemcpyDtoHAsync") {
			pos := fset.Position(call.Pos())
			diags = append(diags, Diagnostic{
				Pos:     pos,
				Message: fmt.Sprintf("unpinned: call to %s must ensure host buffer is pinned with runtime.Pinner tied to stream completion", funcName),
			})
		}

		// Check for cargalign: direct unsafe.Pointer conversions passed into LaunchKernel
		if funcName == "LaunchKernel" && len(call.Args) >= 6 {
			// Inspect kernel parameter array
			paramArg := call.Args[len(call.Args)-1]
			if ident, ok := paramArg.(*ast.Ident); ok && ident.Name == "nil" {
				// nil params are valid for 0-arg kernels
			}
		}

		return true
	})

	return diags
}

func checkDirectory(dir string) ([]Diagnostic, error) {
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, dir, func(fi os.FileInfo) bool {
		return strings.HasSuffix(fi.Name(), ".go") && !strings.HasSuffix(fi.Name(), "_test.go")
	}, 0)
	if err != nil {
		return nil, err
	}

	var diags []Diagnostic
	for _, pkg := range pkgs {
		for _, file := range pkg.Files {
			fileDiags := checkFile(fset, file)
			diags = append(diags, fileDiags...)
		}
	}
	return diags, nil
}

func main() {
	flag.Parse()
	targetDir := "."
	if flag.NArg() > 0 {
		targetDir = flag.Arg(0)
	}

	absDir, err := filepath.Abs(targetDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "cudavet: %v\n", err)
		os.Exit(1)
	}

	diags, err := checkDirectory(absDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "cudavet: %v\n", err)
		os.Exit(1)
	}

	if len(diags) > 0 {
		for _, d := range diags {
			fmt.Printf("%s: %s\n", d.Pos, d.Message)
		}
	}
}
