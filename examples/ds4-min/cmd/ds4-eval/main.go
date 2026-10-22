// Command ds4-eval provides model-free case-table and trace tools.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"

	"github.com/tmc/lib/examples/ds4-min/dsv4eval"
)

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("ds4-eval", flag.ContinueOnError)
	fs.SetOutput(stderr)
	list := fs.Bool("list-cases", false, "write the selected case index")
	validate := fs.Bool("validate-cases", false, "validate the embedded case index")
	selfTest := fs.Bool("self-test-extractors", false, "run model-free answer extractor self-tests")
	trace := fs.String("regrade-trace", "", "regrade a ds4-eval trace")
	compareSmoke := fs.String("compare-smoke", "", "compare a hard-smoke trace against LEFT:RIGHT")
	compareSeeded := fs.String("compare-seeded", "", "compare seeded decode JSON traces against LEFT:RIGHT")
	convertSeeded := fs.String("convert-seeded", "", "convert raw step logits in this directory to a seeded decode JSON trace")
	seededLogprobs := fs.String("seeded-logprobs", "", "logprobs JSON paired with -convert-seeded")
	seededBuild := fs.String("seeded-build", "", "build identity for -convert-seeded")
	seededModelSHA := fs.String("seeded-model-sha256", "", "model SHA-256 for -convert-seeded")
	seededContext := fs.Int("seeded-context", 0, "context size for -convert-seeded")
	seededTemperature := fs.Float64("seeded-temperature", 0, "temperature for -convert-seeded")
	seededSeed := fs.Uint64("seeded-seed", 0, "sampling seed for -convert-seeded")
	seededMTPDisabled := fs.Bool("seeded-mtp-disabled", false, "assert MTP was disabled for -convert-seeded")
	seededOutput := fs.String("seeded-output", "", "output JSON path for -convert-seeded (default stdout)")
	g7Smoke := fs.String("g7-smoke", "", "write a JSON G7 receipt for LEFT:RIGHT traces")
	runTrace := fs.String("trace", "", "execute the suite and write its runtime trace")
	model := fs.String("model", "", "GGUF model for runtime evaluation")
	backend := fs.String("backend", "auto", "runtime backend")
	contextSize := fs.Int("ctx", 32768, "runtime context capacity")
	temperature := fs.Float64("temp", 0, "runtime temperature (must be zero)")
	runner := fs.String("runner", "ds4-min", "clean revision-stamped ds4-min executable")
	corpus := fs.String("corpus", "", "external source-generated hard-smoke corpus directory")
	artifacts := fs.String("eval-artifacts-dir", "", "write diagnostic runtime tokens, decision hashes and terminal logits to a new directory")
	importSmoke := fs.String("import-smoke", "", "attach verified provenance to this raw oracle trace")
	importProvenance := fs.String("import-provenance", "", "oracle execution receipt JSON for -import-smoke")
	oracleBinary := fs.String("oracle-binary", "", "retained executable for -import-smoke")
	suiteName := fs.String("suite", "all", "case suite: core, hard, hard-smoke, or all")
	source := fs.String("source", "", "select cases from this source (case-insensitive)")
	domain := fs.String("domain", "", "select cases in this domain (case-insensitive)")
	caseID := fs.String("case-id", "", "select the case with this exact id")
	questions := fs.Int("questions", 0, "select this many matching cases (0 means all)")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "ds4-eval: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *artifacts != "" && *runTrace == "" {
		fmt.Fprintln(stderr, "ds4-eval: -eval-artifacts-dir requires runtime -trace")
		return 2
	}
	operations := 0
	if *list {
		operations++
	}
	if *validate {
		operations++
	}
	if *selfTest {
		operations++
	}
	if *trace != "" {
		operations++
	}
	if *compareSmoke != "" {
		operations++
	}
	if *compareSeeded != "" {
		operations++
	}
	if *convertSeeded != "" {
		operations++
	}
	if *g7Smoke != "" {
		operations++
	}
	if *runTrace != "" {
		operations++
	}
	if *importSmoke != "" {
		operations++
	}
	if operations > 1 {
		fmt.Fprintln(stderr, "ds4-eval: choose one operation")
		return 2
	}
	if *importSmoke != "" {
		if *importProvenance == "" || *oracleBinary == "" || *model == "" || *suiteName != "all" || *source != "" || *domain != "" || *caseID != "" || *questions != 0 {
			fmt.Fprintln(stderr, "ds4-eval: -import-smoke requires -import-provenance, -oracle-binary and -model, without selection flags")
			return 2
		}
		f, err := os.Open(*importProvenance)
		if err != nil {
			fmt.Fprintln(stderr, "ds4-eval:", err)
			return 2
		}
		decoder := json.NewDecoder(f)
		decoder.DisallowUnknownFields()
		var receipt dsv4eval.OracleSmokeProvenance
		err = decoder.Decode(&receipt)
		if err == nil {
			var extra any
			if decoder.Decode(&extra) != io.EOF {
				err = fmt.Errorf("trailing provenance data")
			}
		}
		closeErr := f.Close()
		if err != nil || closeErr != nil {
			fmt.Fprintln(stderr, "ds4-eval: malformed provenance receipt")
			return 2
		}
		if err := dsv4eval.ImportOracleSmokeFiles(stdout, *importSmoke, *oracleBinary, *model, receipt); err != nil {
			fmt.Fprintln(stderr, "ds4-eval:", err)
			return 2
		}
		return 0
	}
	if *runTrace != "" {
		if *suiteName != "hard-smoke" || *model == "" || *corpus == "" || *contextSize <= 0 || *temperature != 0 || *source != "" || *domain != "" || *caseID != "" || *questions != 0 {
			fmt.Fprintln(stderr, "ds4-eval: runtime requires -suite hard-smoke, -model, positive -ctx and -temp 0, without case filters")
			return 2
		}
		cmd := exec.Command(*runner, "-model", *model, "-backend", *backend, "-ctx", fmt.Sprint(*contextSize), "-temp", "0", "-eval-suite", "hard-smoke", "-eval-trace", *runTrace, "-eval-corpus", *corpus)
		if *artifacts != "" {
			cmd.Args = append(cmd.Args, "-eval-artifacts-dir", *artifacts)
		}
		cmd.Stdout, cmd.Stderr = stdout, stderr
		if err := cmd.Run(); err != nil {
			fmt.Fprintln(stderr, "ds4-eval: runtime:", err)
			return 1
		}
		return 0
	}
	selectionSet := *source != "" || *domain != "" || *caseID != "" || *questions != 0 || *suiteName != "all"
	if selectionSet && !*list {
		fmt.Fprintln(stderr, "ds4-eval: case selection flags require -list-cases")
		return 2
	}
	if *questions < 0 {
		fmt.Fprintln(stderr, "ds4-eval: -questions must not be negative")
		return 2
	}
	if *validate {
		if err := dsv4eval.Validate(dsv4eval.Cases()); err != nil {
			fmt.Fprintln(stderr, "ds4-eval:", err)
			return 1
		}
		return 0
	}
	if *selfTest {
		if err := dsv4eval.SelfTestExtractors(); err != nil {
			fmt.Fprintln(stderr, "ds4-eval:", err)
			return 1
		}
		fmt.Fprintln(stdout, "ds4-eval: extractor self-test passed")
		return 0
	}
	if *list {
		suite, err := dsv4eval.SuiteName(*suiteName)
		if err != nil {
			fmt.Fprintln(stderr, "ds4-eval:", err)
			return 2
		}
		cs, err := dsv4eval.SelectN(dsv4eval.Cases(), dsv4eval.Filter{
			Suite: suite, Source: *source, Domain: *domain, ID: *caseID,
		}, *questions)
		if err != nil {
			fmt.Fprintln(stderr, "ds4-eval:", err)
			return 1
		}
		fmt.Fprintln(stdout, "# source domain id answer-kind title")
		if err := dsv4eval.WriteList(stdout, cs); err != nil {
			fmt.Fprintln(stderr, "ds4-eval:", err)
			return 1
		}
		return 0
	}
	if *compareSmoke != "" {
		parts := strings.Split(*compareSmoke, ":")
		if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
			fmt.Fprintln(stderr, "ds4-eval: -compare-smoke requires LEFT:RIGHT")
			return 2
		}
		left, err := os.Open(parts[0])
		if err != nil {
			fmt.Fprintf(stderr, "ds4-eval: open left trace: %v\n", err)
			return 2
		}
		right, err := os.Open(parts[1])
		if err != nil {
			left.Close()
			fmt.Fprintf(stderr, "ds4-eval: open right trace: %v\n", err)
			return 2
		}
		report, compareErr := dsv4eval.CompareSmoke(left, right)
		leftErr, rightErr := left.Close(), right.Close()
		if compareErr != nil {
			fmt.Fprintf(stderr, "ds4-eval: %v\n", compareErr)
			return 2
		}
		if leftErr != nil || rightErr != nil {
			fmt.Fprintln(stderr, "ds4-eval: close trace: close failed")
			return 2
		}
		fmt.Fprintln(stdout, dsv4eval.FormatSmoke(report))
		if !report.Pass() {
			return 1
		}
		return 0
	}
	if *g7Smoke != "" {
		parts := strings.Split(*g7Smoke, ":")
		if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
			fmt.Fprintln(stderr, "ds4-eval: -g7-smoke requires LEFT:RIGHT")
			return 2
		}
		receipt := dsv4eval.CompareSmokeFiles(parts[0], parts[1])
		if err := receipt.WriteJSON(stdout); err != nil {
			fmt.Fprintf(stderr, "ds4-eval: write G7 receipt: %v\n", err)
			return 2
		}
		switch receipt.Status {
		case "PASS":
			return 0
		case "FAIL":
			return 1
		default:
			return 3
		}
	}
	if *compareSeeded != "" {
		parts := strings.Split(*compareSeeded, ":")
		if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
			fmt.Fprintln(stderr, "ds4-eval: -compare-seeded requires LEFT:RIGHT")
			return 2
		}
		leftFile, err := os.Open(parts[0])
		if err != nil {
			fmt.Fprintf(stderr, "ds4-eval: open left seeded trace: %v\n", err)
			return 2
		}
		left, readErr := dsv4eval.ReadSeededDecodeTrace(leftFile)
		closeErr := leftFile.Close()
		if readErr != nil {
			fmt.Fprintf(stderr, "ds4-eval: left seeded trace: %v\n", readErr)
			return 2
		}
		if closeErr != nil {
			fmt.Fprintf(stderr, "ds4-eval: close left seeded trace: %v\n", closeErr)
			return 2
		}
		rightFile, err := os.Open(parts[1])
		if err != nil {
			fmt.Fprintf(stderr, "ds4-eval: open right seeded trace: %v\n", err)
			return 2
		}
		right, readErr := dsv4eval.ReadSeededDecodeTrace(rightFile)
		closeErr = rightFile.Close()
		if readErr != nil {
			fmt.Fprintf(stderr, "ds4-eval: right seeded trace: %v\n", readErr)
			return 2
		}
		if closeErr != nil {
			fmt.Fprintf(stderr, "ds4-eval: close right seeded trace: %v\n", closeErr)
			return 2
		}
		report, compareErr := dsv4eval.CompareSeededDecode(left, right)
		if compareErr != nil {
			fmt.Fprintf(stderr, "ds4-eval: %v\n", compareErr)
			return 2
		}
		data, err := json.MarshalIndent(report, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "ds4-eval: encode seeded report: %v\n", err)
			return 2
		}
		fmt.Fprintln(stdout, string(data))
		if report.Pass() {
			return 0
		}
		return 1
	}
	if *convertSeeded != "" {
		if *seededLogprobs == "" || *seededBuild == "" || *seededModelSHA == "" ||
			*seededContext <= 0 || *seededTemperature <= 0 || *seededSeed == 0 || !*seededMTPDisabled {
			fmt.Fprintln(stderr, "ds4-eval: -convert-seeded requires logprobs, build, model SHA, positive context/temperature/seed, and -seeded-mtp-disabled")
			return 2
		}
		trace, err := dsv4eval.ConvertSeededDecodeArtifacts(dsv4eval.SeededDecodeArtifactOptions{
			LogitsDir: *convertSeeded, LogprobsPath: *seededLogprobs,
			Build: *seededBuild, ModelSHA256: *seededModelSHA,
			Context: *seededContext, Temperature: float32(*seededTemperature),
			Seed: *seededSeed, MTPDisabled: *seededMTPDisabled,
		})
		if err != nil {
			fmt.Fprintln(stderr, "ds4-eval:", err)
			return 2
		}
		if _, err := dsv4eval.CompareSeededDecode(trace, trace); err != nil {
			fmt.Fprintln(stderr, "ds4-eval: converted trace:", err)
			return 2
		}
		data, err := json.MarshalIndent(trace, "", "  ")
		if err != nil {
			fmt.Fprintln(stderr, "ds4-eval: encode seeded trace:", err)
			return 2
		}
		data = append(data, '\n')
		if *seededOutput == "" {
			_, err = stdout.Write(data)
		} else {
			err = os.WriteFile(*seededOutput, data, 0o600)
		}
		if err != nil {
			fmt.Fprintln(stderr, "ds4-eval: write seeded trace:", err)
			return 2
		}
		return 0
	}
	if *trace == "" {
		fmt.Fprintln(stderr, "ds4-eval: one of -list-cases, -validate-cases, -regrade-trace, -compare-smoke, or -compare-seeded is required")
		return 2
	}
	f, err := os.Open(*trace)
	if err != nil {
		fmt.Fprintf(stderr, "ds4-eval: open trace: %v\n", err)
		return 1
	}
	report, regradeErr := dsv4eval.Regrade(f)
	closeErr := f.Close()
	if regradeErr != nil {
		fmt.Fprintf(stderr, "ds4-eval: %v\n", regradeErr)
		return 1
	}
	if closeErr != nil {
		fmt.Fprintf(stderr, "ds4-eval: close trace: %v\n", closeErr)
		return 1
	}
	fmt.Fprintln(stdout, dsv4eval.FormatRegrade(report))
	if report.Unknown != 0 || report.ParseErrors != 0 || report.Cases == 0 {
		return 1
	}
	return 0
}
