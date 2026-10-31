package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/tmc/lib/nvidia/x/cudaprof/advise"
)

const adviseCountersUsage = "usage: cudaprof advise-counters [-json] [-max-bytes N] actions.json|-\nReads a JSON array of source-bound counter actions; validates structure, not evidence authority or native correctness. No CUDA, profile, ranking or runtime attribution."

const adviseCountersMaxBytes = 16 << 20

func runAdviseCountersCommand(args []string, stdin io.Reader, stdout io.Writer) error {
	fs := flag.NewFlagSet("advise-counters", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	jsonOutput := fs.Bool("json", false, "write the complete counter report as JSON")
	maxBytes := fs.Int("max-bytes", adviseCountersMaxBytes, "maximum input bytes (1..16777216)")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			_, err = fmt.Fprintln(stdout, adviseCountersUsage)
			return err
		}
		return fmt.Errorf("advise-counters: %w", err)
	}
	if fs.NArg() != 1 {
		return errors.New(adviseCountersUsage)
	}
	if *maxBytes <= 0 || *maxBytes > adviseCountersMaxBytes {
		return errors.New("advise-counters: invalid input byte budget")
	}
	input := stdin
	if fs.Arg(0) != "-" {
		file, err := os.Open(fs.Arg(0))
		if err != nil {
			return fmt.Errorf("advise-counters: open input: %w", err)
		}
		defer file.Close()
		input = file
	}
	if input == nil {
		return errors.New("advise-counters: input reader unavailable")
	}
	data, err := io.ReadAll(io.LimitReader(input, int64(*maxBytes)+1))
	if err != nil {
		return fmt.Errorf("advise-counters: read input: %w", err)
	}
	if len(data) > *maxBytes {
		return errors.New("advise-counters: input byte budget exceeded")
	}
	var actions []advise.CounterAction
	if err := decodeStrictJSON(data, &actions); err != nil {
		return fmt.Errorf("advise-counters: decode actions: %w", err)
	}
	if len(actions) == 0 {
		return errors.New("advise-counters: at least one counter action required")
	}
	// encoding/json pads short fixed arrays and drops excess elements. Check
	// the source array length before interpreting a three-dimensional grid.
	var shapes []struct {
		Launch *struct {
			Grid []json.RawMessage `json:"grid"`
		} `json:"launch"`
	}
	if err := json.Unmarshal(data, &shapes); err != nil {
		return fmt.Errorf("advise-counters: decode launch geometry: %w", err)
	}
	for i, shape := range shapes {
		if shape.Launch != nil && len(shape.Launch.Grid) != 3 {
			return fmt.Errorf("advise-counters: action %d: launch grid needs three dimensions", i)
		}
	}
	report, err := advise.AnalyzeCounters(actions)
	if err != nil {
		return fmt.Errorf("advise-counters: analyze actions: %w", err)
	}
	if *jsonOutput {
		encoder := json.NewEncoder(stdout)
		encoder.SetIndent("", "  ")
		if err := encoder.Encode(report); err != nil {
			return fmt.Errorf("advise-counters: write report: %w", err)
		}
		return nil
	}
	if _, err := fmt.Fprintf(stdout, "Counter evidence: %d actions, %d findings\n", len(report.Actions), len(report.Findings)); err != nil {
		return err
	}
	for _, limitation := range report.Limitations {
		if _, err := fmt.Fprintln(stdout, limitation); err != nil {
			return err
		}
	}
	for _, finding := range report.Findings {
		if _, err := fmt.Fprintf(stdout, "\n%q [%s]: %s\n", finding.ActionID, finding.Item.EvidenceLevel, finding.Item.Finding); err != nil {
			return err
		}
		if finding.Item.NextMeasurement != "" {
			if _, err := fmt.Fprintln(stdout, "Next measurement:", finding.Item.NextMeasurement); err != nil {
				return err
			}
		}
	}
	return nil
}
