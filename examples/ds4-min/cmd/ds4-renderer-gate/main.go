// Command ds4-renderer-gate compares a registered renderer oracle and
// candidate JSONL corpus.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/tmc/lib/examples/ds4-min/dsv4tools"
)

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("ds4-renderer-gate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	oracle := fs.String("oracle", "", "upstream oracle JSONL corpus")
	candidate := fs.String("candidate", "", "candidate JSONL corpus")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 || *oracle == "" || *candidate == "" {
		fmt.Fprintln(stderr, "ds4-renderer-gate: -oracle and -candidate are required")
		return 2
	}
	receipt, err := dsv4tools.CompareRendererJSONL(*oracle, *candidate)
	if err != nil {
		// The receipt is still emitted so callers retain a machine-readable
		// failure record, while malformed input remains a distinct exit code.
		_ = writeReceipt(stdout, receipt)
		fmt.Fprintf(stderr, "ds4-renderer-gate: %v\n", err)
		return 2
	}
	if err := writeReceipt(stdout, receipt); err != nil {
		fmt.Fprintf(stderr, "ds4-renderer-gate: write receipt: %v\n", err)
		return 2
	}
	if receipt.Status != "PASS" {
		return 1
	}
	return 0
}

func writeReceipt(w io.Writer, receipt dsv4tools.RendererReceipt) error {
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	return enc.Encode(receipt)
}
