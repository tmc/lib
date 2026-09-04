// CuFile-status opens the GPUDirect Storage driver and prints its version.
package main

import (
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/cufile"
)

func main() {
	if err := cufile.DriverOpen(); err != nil {
		log.Fatal(err)
	}
	defer cufile.DriverClose()
	version, err := cufile.GetVersion()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("cuFile %d\n", version)
}
