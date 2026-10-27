package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

const operatorSummaryUsage = "usage: cudaprof operator-summary -run-id ID [-clock DOMAIN -clock-epoch ID] [-max-bytes N -max-records N -output-bytes N] context.json activities.jsonl"
const operatorCPUSummaryUsage = "usage: cudaprof operator-summary -profile operator-cpu-context-v1 -run-id ID [-max-bytes N -max-records N -output-bytes N] context.json"

func runOperatorSummaryCommand(args []string, out io.Writer) error {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()
	return runOperatorSummary(ctx, args, out)
}

func runOperatorSummary(ctx context.Context, args []string, out io.Writer) error {
	fs := flag.NewFlagSet("operator-summary", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	run := fs.String("run-id", "", "independently supplied application run ID")
	profile := fs.String("profile", "operator-context-v1", "operator-context-v1 or operator-cpu-context-v1")
	clock := fs.String("clock", "", "declared producer clock; omitted means unknown")
	epoch := fs.String("clock-epoch", "", "declared clock epoch; omitted means unknown")
	maxBytes := fs.Int64("max-bytes", 4<<20, "maximum bytes per input (up to 64 MiB)")
	maxRecords := fs.Int("max-records", 10000, "maximum metadata records/member edges (up to 65536)")
	outputBytes := fs.Int64("output-bytes", 8<<20, "maximum JSON output bytes (up to 64 MiB)")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			usage := operatorSummaryUsage
			if *profile == "operator-cpu-context-v1" {
				usage = operatorCPUSummaryUsage
			}
			_, err = fmt.Fprintln(out, usage)
			return err
		}
		return err
	}
	inputs := 2
	if *profile == "operator-cpu-context-v1" {
		inputs = 1
	} else if *profile != "operator-context-v1" {
		return errors.New("unsupported operator summary profile")
	}
	if ctx == nil || out == nil || *run == "" || fs.NArg() != inputs || *maxBytes <= 0 || *maxBytes > 64<<20 || *maxRecords <= 0 || *maxRecords > 65536 || *outputBytes <= 0 || *outputBytes > 64<<20 {
		if *profile == "operator-cpu-context-v1" {
			return errors.New(operatorCPUSummaryUsage)
		}
		return errors.New(operatorSummaryUsage)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	metadata, err := openOperatorInput(fs.Arg(0), *maxBytes)
	if err != nil {
		return err
	}
	defer metadata.Close()
	if *profile == "operator-cpu-context-v1" {
		source := nvtelemetry.WorkloadSource{Provider: "application", Version: *profile, RunID: *run, ClockDomain: *clock, ClockEpochID: *epoch}
		result, err := cudadiag.AnalyzeOperatorCPUJSON(ctx, metadata, source, nvtelemetry.WorkloadLimits{MaxBytes: *maxBytes, MaxMetrics: *maxRecords})
		if err != nil {
			return err
		}
		return writeOperatorSummary(ctx, out, result, *outputBytes)
	}
	activity, err := openOperatorInput(fs.Arg(1), *maxBytes)
	if err != nil {
		return err
	}
	defer activity.Close()
	source := nvtelemetry.WorkloadSource{Provider: "application", Version: "operator-context-v1", RunID: *run, ClockDomain: *clock, ClockEpochID: *epoch}
	raw, err := io.ReadAll(io.LimitReader(activity, *maxBytes+1))
	if err != nil {
		return err
	}
	if int64(len(raw)) > *maxBytes {
		return errors.New("operator activity input byte limit exceeded")
	}
	result, err := cudadiag.AnalyzeOperatorJSONL(ctx, bytes.NewReader(raw), metadata, source, nvtelemetry.WorkloadLimits{MaxBytes: *maxBytes, MaxMetrics: *maxRecords})
	if err != nil {
		return err
	}
	return writeOperatorSummary(ctx, out, result, *outputBytes)
}

func writeOperatorSummary(ctx context.Context, out io.Writer, result any, outputBytes int64) error {
	data, err := json.Marshal(result)
	if err != nil {
		return err
	}
	if int64(len(data))+1 > outputBytes {
		return errors.New("operator summary output byte limit exceeded")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	data = append(data, '\n')
	n, err := out.Write(data)
	if err == nil && n != len(data) {
		return io.ErrShortWrite
	}
	return err
}

func openOperatorInput(name string, maxBytes int64) (*os.File, error) {
	info, err := os.Stat(name)
	if err != nil {
		return nil, fmt.Errorf("operator summary input: %w", err)
	}
	if !info.Mode().IsRegular() || info.Size() > maxBytes {
		return nil, errors.New("operator summary input is nonregular or exceeds byte limit")
	}
	f, err := os.Open(name)
	if err != nil {
		return nil, fmt.Errorf("operator summary input: %w", err)
	}
	info, err = f.Stat()
	if err != nil || !info.Mode().IsRegular() || info.Size() > maxBytes {
		f.Close()
		if err != nil {
			return nil, fmt.Errorf("operator summary input: %w", err)
		}
		return nil, errors.New("operator summary input is nonregular or exceeds byte limit")
	}
	return f, nil
}
