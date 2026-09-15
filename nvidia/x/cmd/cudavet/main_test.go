package main

import (
	"go/parser"
	"go/token"
	"testing"
)

const sampleHazardCode = `
package sample

func BadCopy(dptr driver.DevicePtr, host []byte, s driver.Stream) {
    dptr.MemcpyHtoDAsync(unsafe.Pointer(&host[0]), 1024, s)
}
`

func TestCheckFile(t *testing.T) {
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, "sample.go", sampleHazardCode, 0)
	if err != nil {
		t.Fatalf("ParseFile failed: %v", err)
	}

	diags := checkFile(fset, node)
	if len(diags) == 0 {
		t.Fatal("expected diagnostic on unpinned MemcpyHtoDAsync call")
	}
}
