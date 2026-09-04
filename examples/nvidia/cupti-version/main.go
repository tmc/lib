// Cupti-version prints the CUPTI API version available on this host.
package main

import (
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/cupti"
)

func main() {
	version, err := cupti.GetVersion()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("CUPTI API %d\n", version)
}
