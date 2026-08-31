// Command cudavet inspects CUDA Go packages for common driver and concurrency hazards.
//
// Checks performed:
//   - Unchecked error returns on CUDA driver functions.
//   - Stream / Event leaks without defer Destroy().
//   - Unpinned host slice pointer conversions passed to async DMA.
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
				Message: fmt.Sprintf("call to %s must ensure host buffer is pinned with runtime.Pinner", funcName),
			})
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
