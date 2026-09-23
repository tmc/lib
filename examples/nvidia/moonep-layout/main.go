// Moonep-layout calculates a single-rank expert-parallel buffer layout.
package main

import (
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/x/moonep"
)

func main() {
	l, err := moonep.NewLayout(moonep.Config{S: 8, H: 16, K: 2, E: 4, R: 1}, 64*1024, 64*1024)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("hidden=%d bytes meta=%d bytes\n", l.HiddenChunkBytes(), l.MetaChunkBytes())
}
