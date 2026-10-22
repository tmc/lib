// Command ds4-bench validates and canonicalizes a ds4-bench frontier CSV.
// It consumes existing measurements; it does not run a model or GPU work.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	dsv4bench "github.com/tmc/lib/examples/ds4-min/bench/dsv4bench"
)

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout, stderr io.Writer) int {
	return runWithIO(args, os.Stdin, stdout, stderr)
}

func runWithIO(args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("ds4-bench", flag.ContinueOnError)
	fs.SetOutput(stderr)
	inputName := fs.String("input", "-", "input CSV, or - for stdin")
	outputName := fs.String("output", "-", "output CSV, or comparison JSON, or - for stdout")
	frontier := fs.Bool("frontier", false, "require strictly increasing context frontiers")
	summary := fs.Bool("summary", false, "write a JSON median/min/max summary instead of CSV")
	compareBase := fs.String("compare-base", "", "base benchmark CSV for paired JSON comparison")
	compareTarget := fs.String("compare-target", "", "target benchmark CSV for paired JSON comparison")
	compareFrontierBase := fs.String("compare-frontier-base", "", "base frontier JSONL for F7 comparison")
	compareFrontierTarget := fs.String("compare-frontier-target", "", "target frontier JSONL for F7 comparison")
	frontierAbs := fs.Float64("frontier-abs", 1e-4, "absolute frontier logit tolerance")
	frontierRel := fs.Float64("frontier-rel", 1e-4, "relative frontier logit tolerance")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "ds4-bench: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if (*compareBase != "" || *compareTarget != "") && (*compareFrontierBase != "" || *compareFrontierTarget != "") {
		fmt.Fprintln(stderr, "ds4-bench: benchmark and frontier comparison modes cannot be combined")
		return 2
	}
	if *compareBase != "" || *compareTarget != "" {
		if *compareBase == "" || *compareTarget == "" {
			fmt.Fprintln(stderr, "ds4-bench: -compare-base and -compare-target must be used together")
			return 2
		}
		if *frontier || *summary || flagWasSet(fs, "input") {
			fmt.Fprintln(stderr, "ds4-bench: comparison cannot be combined with -input, -frontier, or -summary")
			return 2
		}
		return runComparison(*compareBase, *compareTarget, *outputName, stdout, stderr)
	}
	if *compareFrontierBase != "" || *compareFrontierTarget != "" {
		if *compareFrontierBase == "" || *compareFrontierTarget == "" {
			fmt.Fprintln(stderr, "ds4-bench: -compare-frontier-base and -compare-frontier-target must be used together")
			return 2
		}
		if *frontier || *summary || flagWasSet(fs, "input") {
			fmt.Fprintln(stderr, "ds4-bench: frontier comparison cannot be combined with -input, -frontier, or -summary")
			return 2
		}
		if *frontierAbs < 0 || *frontierRel < 0 {
			fmt.Fprintln(stderr, "ds4-bench: frontier tolerances must not be negative")
			return 2
		}
		return runFrontierComparison(*compareFrontierBase, *compareFrontierTarget, *outputName, *frontierAbs, *frontierRel, stdout, stderr)
	}
	in, closeIn, err := openInput(*inputName, stdin)
	if err != nil {
		fmt.Fprintln(stderr, "ds4-bench:", err)
		return 1
	}
	rows, err := dsv4bench.ReadCSV(in)
	if closeIn != nil {
		_ = closeIn.Close()
	}
	if err != nil {
		fmt.Fprintln(stderr, "ds4-bench:", err)
		return 1
	}
	if *frontier {
		if err := dsv4bench.ValidateFrontier(rows); err != nil {
			fmt.Fprintln(stderr, "ds4-bench:", err)
			return 1
		}
	}
	if *summary {
		result, err := dsv4bench.Summarize(rows)
		if err != nil {
			fmt.Fprintln(stderr, "ds4-bench:", err)
			return 1
		}
		if err := dsv4bench.WriteSummaryJSON(stdout, result); err != nil {
			fmt.Fprintln(stderr, "ds4-bench:", err)
			return 1
		}
		return 0
	}
	out, closeOut, err := openOutput(*outputName, stdout)
	if err != nil {
		fmt.Fprintln(stderr, "ds4-bench:", err)
		return 1
	}
	err = dsv4bench.WriteCSV(out, rows)
	if closeOut != nil {
		if closeErr := closeOut.Close(); err == nil {
			err = closeErr
		}
	}
	if err != nil {
		fmt.Fprintln(stderr, "ds4-bench:", err)
		return 1
	}
	return 0
}

func runFrontierComparison(baseName, targetName, outputName string, abs, rel float64, stdout, stderr io.Writer) int {
	base, err := readFrontierFile(baseName)
	if err != nil {
		fmt.Fprintln(stderr, "ds4-bench:", err)
		return 1
	}
	target, err := readFrontierFile(targetName)
	if err != nil {
		fmt.Fprintln(stderr, "ds4-bench:", err)
		return 1
	}
	comparison, err := dsv4bench.CompareFrontier(base, target, dsv4bench.FrontierTolerance{Abs: abs, Rel: rel})
	if err != nil {
		fmt.Fprintln(stderr, "ds4-bench:", err)
		return 1
	}
	out, closeOut, err := openOutput(outputName, stdout)
	if err != nil {
		fmt.Fprintln(stderr, "ds4-bench:", err)
		return 1
	}
	err = json.NewEncoder(out).Encode(comparison)
	if closeOut != nil {
		if closeErr := closeOut.Close(); err == nil {
			err = closeErr
		}
	}
	if err != nil {
		fmt.Fprintln(stderr, "ds4-bench: write frontier comparison:", err)
		return 1
	}
	if !comparison.Pass() {
		return 1
	}
	return 0
}

func readFrontierFile(name string) ([]dsv4bench.FrontierLogits, error) {
	if name == "-" {
		return nil, fmt.Errorf("frontier comparison input must be a file, not stdin")
	}
	in, err := os.Open(name)
	if err != nil {
		return nil, fmt.Errorf("open frontier comparison input %q: %w", name, err)
	}
	defer in.Close()
	rows, err := dsv4bench.ReadFrontierJSON(in)
	if err != nil {
		return nil, fmt.Errorf("read frontier comparison input %q: %w", name, err)
	}
	return rows, nil
}

func runComparison(baseName, targetName, outputName string, stdout, stderr io.Writer) int {
	base, err := readCSVFile(baseName)
	if err != nil {
		fmt.Fprintln(stderr, "ds4-bench:", err)
		return 1
	}
	target, err := readCSVFile(targetName)
	if err != nil {
		fmt.Fprintln(stderr, "ds4-bench:", err)
		return 1
	}
	comparison, err := dsv4bench.Compare(base, target)
	if err != nil {
		fmt.Fprintln(stderr, "ds4-bench:", err)
		return 1
	}
	out, closeOut, err := openOutput(outputName, stdout)
	if err != nil {
		fmt.Fprintln(stderr, "ds4-bench:", err)
		return 1
	}
	err = json.NewEncoder(out).Encode(comparison)
	if closeOut != nil {
		if closeErr := closeOut.Close(); err == nil {
			err = closeErr
		}
	}
	if err != nil {
		fmt.Fprintln(stderr, "ds4-bench: write comparison:", err)
		return 1
	}
	return 0
}

func readCSVFile(name string) ([]dsv4bench.Row, error) {
	if name == "-" {
		return nil, fmt.Errorf("comparison input must be a file, not stdin")
	}
	in, err := os.Open(name)
	if err != nil {
		return nil, fmt.Errorf("open comparison input %q: %w", name, err)
	}
	defer in.Close()
	rows, err := dsv4bench.ReadCSV(in)
	if err != nil {
		return nil, fmt.Errorf("read comparison input %q: %w", name, err)
	}
	return rows, nil
}

func flagWasSet(fs *flag.FlagSet, name string) bool {
	set := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == name {
			set = true
		}
	})
	return set
}

func openInput(name string, stdin io.Reader) (io.Reader, io.Closer, error) {
	if name == "-" {
		return stdin, nil, nil
	}
	f, err := os.Open(name)
	if err != nil {
		return nil, nil, fmt.Errorf("open input: %w", err)
	}
	return f, f, nil
}

func openOutput(name string, stdout io.Writer) (io.Writer, io.Closer, error) {
	if name == "-" {
		return stdout, nil, nil
	}
	f, err := os.Create(name)
	if err != nil {
		return nil, nil, fmt.Errorf("create output: %w", err)
	}
	return f, f, nil
}
