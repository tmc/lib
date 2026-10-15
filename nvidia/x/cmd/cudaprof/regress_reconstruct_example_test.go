package main

import (
	"fmt"
	"io"
)

func Example_runRegressReconstructCommand() {
	err := runRegressReconstructCommand(nil, io.Discard)
	fmt.Println(err)
	// Output: usage: cudaprof regress reconstruct -inventory packet.json -inventory-sha256 digest -o new-dir
}
