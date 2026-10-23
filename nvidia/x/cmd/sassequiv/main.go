// Command sassequiv reports whether two SASS listings of the same kernels
// compute the same values.
//
// Usage:
//
//	sassequiv before.sass after.sass
//
// The listings are cuobjdump -sass or nvdisasm output. For each function
// the command prints equal, different (with the first expressions that
// differ and their instruction addresses) or unknown (with the reason),
// as package github.com/tmc/lib/nvidia/x/sassequiv decides. It exits 0
// when every function is equal, 1 when any differs, and 3 when none
// differs but some are unknown.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/sassequiv"
)

func main() {
	log.SetFlags(0)
	log.SetPrefix("sassequiv: ")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "usage: sassequiv before.sass after.sass\n")
		os.Exit(2)
	}
	flag.Parse()
	if flag.NArg() != 2 {
		flag.Usage()
	}
	before, err := parse(flag.Arg(0))
	if err != nil {
		log.Fatal(err)
	}
	after, err := parse(flag.Arg(1))
	if err != nil {
		log.Fatal(err)
	}
	code := 0
	for _, r := range sassequiv.CompareAll(before, after) {
		fmt.Println(r)
		switch {
		case r.Verdict == sassequiv.Different:
			code = 1
		case r.Verdict == sassequiv.Unknown && code == 0:
			code = 3
		}
	}
	os.Exit(code)
}

func parse(path string) (*cudasass.Disassembly, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	d, err := cudasass.Parse(f)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return d, nil
}
