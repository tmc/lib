// Command sassequiv reports whether two SASS listings of the same kernels
// compute the same values.
//
// Usage:
//
//	sassequiv [-json] before.sass after.sass
//
// The listings are cuobjdump -sass or nvdisasm output. For each function
// the command prints equal, different (with the first expressions that
// differ and their instruction addresses) or unknown (with the reason),
// as package github.com/tmc/lib/nvidia/x/sassequiv decides. An equal
// that holds only if both compilations find the same values in registers
// they read before writing names them: "equal (assumes identical
// uninitialized R24, R25)". It exits 0 when every function is equal, 1
// when any differs, and 3 when none differs but some are unknown.
//
// The -json flag prints each result as a JSON object with fields
// function, verdict, and, when set, reason, before and after (hex
// instruction addresses) and assumes (register names).
package main

import (
	"encoding/json"
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
		fmt.Fprintf(os.Stderr, "usage: sassequiv [-json] before.sass after.sass\n")
		os.Exit(2)
	}
	asJSON := flag.Bool("json", false, "print results as JSON objects")
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
	enc := json.NewEncoder(os.Stdout)
	enc.SetEscapeHTML(false)
	for _, r := range sassequiv.CompareAll(before, after) {
		if *asJSON {
			if err := enc.Encode(result(r)); err != nil {
				log.Fatal(err)
			}
		} else {
			fmt.Println(r)
		}
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

type jsonResult struct {
	Function string   `json:"function"`
	Verdict  string   `json:"verdict"`
	Reason   string   `json:"reason,omitempty"`
	Before   string   `json:"before,omitempty"`
	After    string   `json:"after,omitempty"`
	Assumes  []string `json:"assumes,omitempty"`
}

func result(r sassequiv.Result) jsonResult {
	j := jsonResult{Function: r.Function, Verdict: r.Verdict.String(), Reason: r.Reason, Assumes: r.Assumes}
	if r.Verdict == sassequiv.Different {
		j.Before, j.After = fmt.Sprintf("%#x", r.Before), fmt.Sprintf("%#x", r.After)
	}
	return j
}
