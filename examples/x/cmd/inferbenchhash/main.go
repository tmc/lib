// Command inferbenchhash prints identities accepted by inferbench collectors.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/tmc/lib/examples/internal/inferbench"
)

func main() {
	flag.Parse()
	if flag.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "usage: inferbenchhash path...")
		os.Exit(2)
	}
	for _, path := range flag.Args() {
		sum, err := inferbench.HashPath(path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", path, err)
			os.Exit(1)
		}
		fmt.Printf("%s  %s\n", sum, path)
	}
}
