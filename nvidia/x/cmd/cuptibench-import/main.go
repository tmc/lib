// Command cuptibench-import converts canonical external measurements to Go
// benchmark text. It reads one JSON run from a file argument or standard input.
package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/tmc/lib/nvidia/x/cuptibench/benchfmt"
)

func main() {
	if err := run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(args []string, stdin io.Reader, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("cuptibench-import", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: cuptibench-import [run.json]\n\nConvert canonical measured samples to benchstat input; no workload is executed.")
	}
	if err := fs.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return nil
		}
		return err
	}
	if fs.NArg() > 1 {
		return fmt.Errorf("cuptibench-import: expected at most one JSON file")
	}
	input := stdin
	if fs.NArg() == 1 {
		f, err := os.Open(fs.Arg(0))
		if err != nil {
			return err
		}
		defer f.Close()
		input = f
	}
	measured, err := benchfmt.Decode(input)
	if err != nil {
		return err
	}
	return benchfmt.Write(stdout, measured)
}
