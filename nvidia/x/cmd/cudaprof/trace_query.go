package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/tmc/lib/nvidia/x/cudaprof/traceanalysis"
)

func runTraceQueryCommand(args []string) error { return runTraceQuery(args, os.Stdout) }
func runTraceQuery(args []string, out io.Writer) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	return runTraceQueryContext(ctx, args, out)
}

func runTraceQueryContext(ctx context.Context, args []string, out io.Writer) error {
	fs := flag.NewFlagSet("trace query", flag.ContinueOnError)
	fs.SetOutput(out)
	resourceLimits := registerTraceViewLimits(fs)
	list := fs.Bool("list", false, "list versioned named-query schemas")
	jsonOutput := fs.Bool("json", false, "write versioned JSON")
	processor := fs.String("perfetto", "", "optional local Trace Processor v58.2 executable for exported trace queries")
	domain := fs.String("domain", "driver", "submission API domain: driver or runtime")
	var goTrace goTraceOptions
	goTrace.register(fs)
	fs.Usage = func() {
		fmt.Fprintln(out, "Usage: cudaprof trace query -list [-json]\n       cudaprof trace query [-json] [-domain driver|runtime] NAME INPUT")
		fmt.Fprintln(out, goTraceOptionsHelp)
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	if err := goTrace.check(); err != nil {
		return fmt.Errorf("trace query: %w", err)
	}
	if err := resourceLimits.validate(); err != nil {
		return err
	}
	out = &limitedExportWriter{Writer: out, remaining: resourceLimits.outputBytes}
	if *list {
		if fs.NArg() != 0 {
			return errors.New("trace query: -list accepts no positional arguments")
		}
		return writeTraceView(out, traceViewResult{Consumer: currentBuild(), Version: 1, Schema: "cudaprof.query-catalog.v1", Backend: "native", State: "available", Evidence: traceanalysis.Catalog()}, *jsonOutput)
	}
	if fs.NArg() != 2 {
		return errors.New("usage: cudaprof trace query [options] NAME INPUT")
	}
	var definition *traceanalysis.Query
	for _, entry := range traceanalysis.Catalog() {
		if entry.Name == fs.Arg(0) {
			copy := entry
			definition = &copy
			break
		}
	}
	if definition == nil {
		return fmt.Errorf("trace query: unknown name %q", fs.Arg(0))
	}
	input, err := loadTraceSessionLimits(fs.Arg(1), *domain, resourceLimits.input)
	if err != nil {
		return err
	}
	result := traceViewResult{Consumer: currentBuild(), Options: map[string]any{"resource_limits": resourceLimits.receipt(), "api_domain": *domain, "query": definition.Name, "go_trace_decoder": goTrace.decoder, "go_trace_scratch": goTrace.scratch}, Version: definition.Version, Schema: definition.Schema, Input: input.Path, SHA256: input.SHA256, Kind: input.Kind, Backend: "native", View: definition.Name, State: "observed", Limits: []string{definition.Aggregation, definition.Required}}
	var failure error
	var backendFailure error
	if input.Capture == nil {
		result.Backend = "trace_processor_v58.2"
		result.State = "unavailable"
		sql, err := traceanalysis.SQLDomain(definition.Name, *domain)
		if err != nil {
			return err
		}
		receipt, backendError := queryTraceProcessorLimits(input.Data, *processor, sql, resourceLimits.backend)
		result.Evidence = receipt
		result.Qualifications = map[string]string{"collector": "unavailable: exported annotations are supplied provenance, not a bound native collector receipt", "correctness": "unverified", "stationarity": "unassessed", "shape": "unqualified", "request_scope": "unavailable"}
		if backendError != nil {
			failure = backendError
			var unavailable *traceBackendUnavailable
			if !errors.As(backendError, &unavailable) {
				backendFailure = backendError
			}
		} else if len(receipt.Rows) <= 1 {
			failure = errors.New("trace query: required versioned role metadata unavailable")
		} else {
			result.State = "observed"
			if definition.Name == "delivery-cadence" {
				result.State = "unavailable"
				failure = errors.New("trace query: required native request/correlation evidence unavailable")
			}
			if definition.Name == "unmatched" || definition.Name == "correlations" {
				if err := validateSubmissionQuery(receipt.Rows, *domain); err != nil {
					result.State = "unavailable"
					failure = err
					backendFailure = err
				}

				for _, row := range receipt.Rows[1:] {
					if len(row) == 0 || row[0] != "available" {
						result.State = "unavailable"
						failure = errors.New("trace query: retained submission receipt coverage unavailable")
					}
				}
			}
			if definition.Name == "qualifications" && len(receipt.Rows[1]) > 1 && receipt.Rows[1][1] == "unavailable" {
				result.State = "unavailable"
				failure = errors.New("trace query: role schema unavailable")
			}
		}

	} else {
		options := traceanalysis.Options{CollectorComplete: input.CollectorError == nil, Domain: *domain}
		if input.CollectorError != nil {
			options.CollectorFailure = input.CollectorError.Error()
			failure = input.CollectorError
			result.State = "unavailable"
		}
		session, err := traceanalysis.New(input.Records, options)
		if err != nil {
			return err
		}
		result.Qualifications = session.Summary().Gates
		if definition.Name == "language-stacks" || definition.Name == "host-stacks" {
			result.Evidence, err = nativeLanguageStackView(ctx, input.Capture, definition.Name, goTrace)
		} else {
			result.Evidence, err = session.Query(definition.Name)
		}
		if err != nil {
			return err
		}
	}
	if failure != nil {
		result.Limits = append(result.Limits, failure.Error())
	}
	if evidence, ok := result.Evidence.(languageStackView); ok {
		result.State = evidence.State
	}
	if result.Qualifications == nil {
		result.Qualifications = map[string]string{}
	}
	result.Qualifications["isolation"] = "unchecked: no bound native observer receipt"
	if input.Capture != nil {
		observed := captureIsolation(input.Capture)
		result.Qualifications["isolation"] = observed.State + ": " + strings.Join(observed.Reasons, "; ")
	}
	if err := writeTraceView(out, result, *jsonOutput); err != nil {
		return err
	}
	return backendFailure
}
