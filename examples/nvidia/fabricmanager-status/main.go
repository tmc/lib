// Fabricmanager-status initializes and shuts down the Fabric Manager client.
package main

import (
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/fabricmanager"
)

func main() {
	if err := fabricmanager.LibInit(); err != nil {
		log.Fatal(err)
	}
	defer fabricmanager.LibShutdown()
	fmt.Println("Fabric Manager library initialized")
}
