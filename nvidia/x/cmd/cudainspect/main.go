package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudainspect"
)

// Exit statuses.
const (
	exitOK       = 0
	exitRequired = 1
	exitRefused  = 2
)

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

const usage = "usage: cudainspect report -symbol NAME [-require a,b] [-optional c] [-sass FILE] [-max-bytes N] FILE.cubin"

// run executes the command line args and returns the exit status.
func run(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 || args[0] != "report" {
		fmt.Fprintln(stderr, usage)
		return exitRefused
	}
	fs := flag.NewFlagSet("report", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() {
		fmt.Fprintln(stderr, usage)
		fs.PrintDefaults()
	}
	symbol := fs.String("symbol", "", "kernel symbol, exactly one .text section of the image")
	require := fs.String("require", "", "comma-separated features whose absence fails the report")
	optional := fs.String("optional", "", "comma-separated features whose absence makes the report partial")
	sassFile := fs.String("sass", "", "cuobjdump -sass or nvdisasm output for the image")
	maxBytes := fs.Int64("max-bytes", 0, "image size bound (default and maximum 64 MiB)")
	if err := fs.Parse(args[1:]); err != nil {
		return exitRefused
	}
	if fs.NArg() != 1 {
		fmt.Fprintln(stderr, usage)
		return exitRefused
	}
	req := cudainspect.Request{
		Symbol:        *symbol,
		Required:      features(*require),
		Optional:      features(*optional),
		MaxInputBytes: *maxBytes,
	}
	r, err := report(fs.Arg(0), *sassFile, req)
	if r == nil {
		fmt.Fprintln(stderr, err)
		return exitRefused
	}
	if err := r.WriteJSON(stdout); err != nil {
		fmt.Fprintf(stderr, "cudainspect: write report: %v\n", err)
		return exitRefused
	}
	if err != nil {
		fmt.Fprintln(stderr, err)
		return exitRequired
	}
	return exitOK
}

// report reads the inputs within their bounds and inspects them. A non-nil
// report with an error means a required feature has no value.
func report(cubinFile, sassFile string, req cudainspect.Request) (*cudainspect.Report, error) {
	limit := req.MaxInputBytes
	if limit <= 0 || limit > cudainspect.DefaultMaxInputBytes {
		limit = cudainspect.DefaultMaxInputBytes // Report refuses an invalid bound itself
	}
	cubin, err := readFile(cubinFile, limit)
	if err != nil {
		return nil, err
	}
	if sassFile != "" {
		sass, err := readFile(sassFile, cudainspect.MaxSASSBytes)
		if err != nil {
			return nil, err
		}
		req.SASS = string(sass)
	}
	i, err := cudainspect.NewInspector(".")
	if err != nil {
		return nil, err
	}
	r, err := i.Report(context.Background(), cubin, req)
	if err != nil && !errors.Is(err, cudainspect.ErrRequiredFeature) {
		return nil, err
	}
	return r, err
}

// readFile reads at most limit bytes of name, refusing a larger file.
func readFile(name string, limit int64) ([]byte, error) {
	f, err := os.Open(name)
	if err != nil {
		return nil, fmt.Errorf("cudainspect: %w", err)
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil {
		return nil, fmt.Errorf("cudainspect: read %s: %w", name, err)
	}
	if int64(len(data)) > limit {
		return nil, fmt.Errorf("%w: %s is over %d bytes", cudainspect.ErrInputTooLarge, name, limit)
	}
	return data, nil
}

// features splits a comma-separated list of feature names, trimming the
// spaces around each one.
func features(list string) []string {
	if list == "" {
		return nil
	}
	names := strings.Split(list, ",")
	for n := range names {
		names[n] = strings.TrimSpace(names[n])
	}
	return names
}
