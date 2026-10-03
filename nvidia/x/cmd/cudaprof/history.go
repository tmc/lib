package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/tmc/lib/nvidia/x/cudaprofilehistory"
)

const historyUsage = "usage: cudaprof history [-max-bytes=N] [-max-records=N] snapshot.ndjson"
const bisectUsage = "usage: cudaprof bisect [-max-bytes=N] request.json"

// These offline handlers deliberately perform no collection or GPU execution.
func runHistoryCommand(args []string, stdout io.Writer) error {
	if len(args) > 0 && args[0] == "compare" {
		return runHistoryCompareCommand(args[1:], stdout)
	}
	fs := flag.NewFlagSet("history", flag.ContinueOnError)
	maxBytes := fs.Int("max-bytes", 8<<20, "input and output byte bound")
	maxRows := fs.Int("max-records", 1000, "maximum observations")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 1 {
		return fmt.Errorf("%s", historyUsage)
	}
	if *maxBytes <= 0 || *maxBytes > 64<<20 {
		return fmt.Errorf("history: invalid byte bound")
	}
	f, err := os.Open(fs.Arg(0))
	if err != nil {
		return err
	}
	defer f.Close()
	s, err := cudaprofilehistory.ReadSnapshot(f, cudaprofilehistory.Limits{MaxRecords: *maxRows, MaxBytes: *maxBytes, MaxRecordBytes: *maxBytes})
	if err != nil {
		return err
	}
	rows, err := s.Records()
	if err != nil {
		return err
	}
	return writeHistoryJSON(stdout, rows, *maxBytes)
}

func runHistoryCompareCommand(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("history compare", flag.ContinueOnError)
	maxBytes := fs.Int("max-bytes", 8<<20, "aggregate input and output byte bound")
	channel := fs.String("timing", "gpu_busy", "explicit timing channel")
	threshold := fs.Float64("threshold", .1, "fractional regression threshold")
	noise := fs.Float64("noise-ns", 0, "absolute timing noise")
	samples := fs.Int("min-samples", 3, "minimum native observations")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 2 {
		return fmt.Errorf("usage: cudaprof history compare [flags] before.json after.json")
	}
	if *maxBytes <= 0 || *maxBytes > 64<<20 {
		return fmt.Errorf("history: invalid byte bound")
	}
	remaining := *maxBytes
	var rows [2]cudaprofilehistory.Record
	for i, path := range fs.Args() {
		f, err := os.Open(path)
		if err != nil {
			return err
		}
		b, err := io.ReadAll(io.LimitReader(f, int64(remaining)+1))
		closeErr := f.Close()
		if err != nil {
			return err
		}
		if closeErr != nil {
			return closeErr
		}
		rows[i], err = cudaprofilehistory.DecodeRecord(b, remaining)
		if err != nil {
			return err
		}
		remaining -= len(b)
	}
	report, err := cudaprofilehistory.Compare(rows[0], rows[1], cudaprofilehistory.Policy{Timing: *channel, RelativeThreshold: *threshold, NoiseNanos: *noise, MinSamples: *samples})
	if err != nil {
		return err
	}
	return writeHistoryJSON(stdout, report, *maxBytes)
}

func runBisectCommand(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("bisect", flag.ContinueOnError)
	maxBytes := fs.Int("max-bytes", 1<<20, "input and output byte bound")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 1 {
		return fmt.Errorf("%s", bisectUsage)
	}
	if *maxBytes <= 0 || *maxBytes > 64<<20 {
		return fmt.Errorf("bisect: invalid byte bound")
	}
	f, err := os.Open(fs.Arg(0))
	if err != nil {
		return err
	}
	defer f.Close()
	b, err := io.ReadAll(io.LimitReader(f, int64(*maxBytes)+1))
	if err != nil {
		return err
	}
	r, err := cudaprofilehistory.DecodeBisectRequest(b, *maxBytes)
	if err != nil {
		return err
	}
	p, err := cudaprofilehistory.PlanBisect(r)
	if err != nil {
		return err
	}
	return writeHistoryJSON(stdout, p, *maxBytes)
}

func writeHistoryJSON(w io.Writer, v any, limit int) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	if len(b)+1 > limit {
		return fmt.Errorf("history: output byte budget exceeded")
	}
	b = append(b, '\n')
	n, err := w.Write(b)
	if err == nil && n != len(b) {
		return io.ErrShortWrite
	}
	return err
}
