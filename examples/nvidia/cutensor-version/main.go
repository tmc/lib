// Cutensor-version reports whether cuTENSOR is installed and, if so, its version.
package main

import (
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/cutensor"
)

func main() {
	if err := cutensor.Load(""); err != nil {
		log.Fatal(err)
	}
	defer cutensor.Close()
	fmt.Printf("cuTENSOR %d (CUDA runtime %d)\n", cutensor.GetVersion(), cutensor.GetCudartVersion())
}
