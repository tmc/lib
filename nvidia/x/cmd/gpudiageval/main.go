package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"text/tabwriter"

	"github.com/tmc/lib/nvidia/x/cudadiageval"
)

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintf(os.Stderr, "gpudiageval: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string, stdout, stderr io.Writer) error {
	if len(args) == 0 {
		printUsage(stderr)
		return errors.New("command required: run, verify, claim, ledger, or compare")
	}

	switch args[0] {
	case "run":
		return cmdRun(args[1:], stdout, stderr)
	case "verify":
		return cmdVerify(args[1:], stdout, stderr)
	case "compare":
		return cmdCompare(args[1:], stdout, stderr)
	case "ledger":
		return cmdLedger(args[1:], stdout, stderr)
	case "claim":
		return cmdClaim(args[1:], stdout, stderr)
	case "help", "-h", "--help":
		printUsage(stdout)
		return nil
	default:
		printUsage(stderr)
		return fmt.Errorf("unknown command %q", args[0])
	}
}

func printUsage(w io.Writer) {
	fmt.Fprintln(w, "gpudiageval: CUDA workload evaluation and bounded comparative claim qualification")
	fmt.Fprintln(w, "")
	fmt.Fprintln(w, "Usage:")
	fmt.Fprintln(w, "  gpudiageval run [flags]      Execute evaluation protocol across 4 property gates")
	fmt.Fprintln(w, "  gpudiageval verify [flags]   Validate evaluation artifacts and manifest integrity")
	fmt.Fprintln(w, "  gpudiageval claim [flags]    Display bounded comparative claim from evaluation report")
	fmt.Fprintln(w, "  gpudiageval ledger -file path -root dir  Check qualification receipt consistency")
	fmt.Fprintln(w, "  gpudiageval compare -protocol path -candidate path -comparator path")
	fmt.Fprintln(w, "")
	fmt.Fprintln(w, "Flags for 'run':")
	fmt.Fprintln(w, "  -phase string               Evaluation phase: 'pilot' or 'qualification' (default: pilot)")
	fmt.Fprintln(w, "  -out-dir string             Directory to export evaluation report, receipts, and manifest")
	fmt.Fprintln(w, "  -json                       Output complete evaluation report as JSON")
	fmt.Fprintln(w, "  -stress                     Exercise resilience controls (overflow, restart, late attach)")
	fmt.Fprintln(w, "")
	fmt.Fprintln(w, "Flags for 'verify':")
	fmt.Fprintln(w, "  -dir string                 Directory containing evaluation artifacts and manifest.json")
	fmt.Fprintln(w, "  -json                       Output verification status as JSON")
	fmt.Fprintln(w, "")
	fmt.Fprintln(w, "Flags for 'claim':")
	fmt.Fprintln(w, "  -report string              Path to evaluation_report.json (or reads from dir)")
	fmt.Fprintln(w, "  -json                       Output claim as JSON")
}

func cmdRun(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	fs.SetOutput(stderr)

	var (
		phaseStr string
		outDir   string
		asJSON   bool
		stress   bool
	)

	fs.StringVar(&phaseStr, "phase", "pilot", "Evaluation phase: 'pilot' or 'qualification'")
	fs.StringVar(&outDir, "out-dir", "", "Directory to export evaluation artifacts")
	fs.BoolVar(&asJSON, "json", false, "Output report as JSON")
	fs.BoolVar(&stress, "stress", true, "Exercise stress and resilience controls")

	if err := fs.Parse(args); err != nil {
		return err
	}

	phase := cudadiageval.Phase(phaseStr)
	proto := cudadiageval.DefaultProtocol(phase)

	if phase == cudadiageval.PhaseQualification {
		if err := proto.Freeze(); err != nil {
			return fmt.Errorf("freeze protocol: %w", err)
		}
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	cfg := cudadiageval.DefaultEvalConfig()
	cfg.ExerciseStress = stress

	report, err := cudadiageval.RunEvaluation(ctx, proto, cfg)
	if err != nil {
		return fmt.Errorf("run evaluation: %w", err)
	}

	if outDir != "" {
		if _, err := cudadiageval.WriteReportArtifacts(outDir, report); err != nil {
			return fmt.Errorf("write artifacts: %w", err)
		}
	}

	if asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(report)
	}

	printReportSummary(stdout, report, outDir)
	return nil
}

func printReportSummary(w io.Writer, report *cudadiageval.EvaluationReport, outDir string) {
	fmt.Fprintln(w, "================================================================================")
	fmt.Fprintln(w, "            CUDA DIAGNOSTIC EVALUATION REPORT & CLAIM QUALIFICATION             ")
	fmt.Fprintln(w, "================================================================================")
	fmt.Fprintf(w, "Protocol:      %s (Phase: %s)\n", report.Protocol.Name, report.Protocol.Phase)
	fmt.Fprintf(w, "Hardware:      %s (%s), Driver: %s, CUDA: %s\n",
		report.Protocol.Hardware.GPUModel,
		report.Protocol.Hardware.Architecture,
		report.Protocol.Hardware.DriverVersion,
		report.Protocol.Hardware.CUDAVersion)
	fmt.Fprintf(w, "Execution:     Duration: %v, Passed: %t\n", report.Duration, report.Passed)
	fmt.Fprintln(w, "")

	fmt.Fprintln(w, "--- PROPERTY GATE EVALUATION MATRIX ---")
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "PROPERTY\tGATE NAME\tDISPOSITION\tMEASURED\tTHRESHOLD")
	fmt.Fprintln(tw, "--------\t---------\t-----------\t--------\t---------")
	for _, g := range report.Gates {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\n",
			g.Property,
			g.GateName,
			g.Disposition,
			g.MeasuredValue,
			g.Threshold)
	}
	tw.Flush()
	fmt.Fprintln(w, "")

	fmt.Fprintln(w, "--- RELEVANT BASELINES ---")
	for _, b := range report.Baselines {
		ind := "independent baseline"
		if !b.Independent {
			ind = "related project (not independent comparison)"
		}
		fmt.Fprintf(w, "  * %s (%s, %s): %s\n", b.Name, b.Version, b.Category, ind)
	}
	fmt.Fprintln(w, "")

	if report.Claim != nil {
		fmt.Fprintln(w, "--- BOUNDED COMPARATIVE CLAIM ---")
		fmt.Fprint(w, report.Claim.Format())
	}

	if outDir != "" {
		fmt.Fprintln(w, "")
		fmt.Fprintf(w, "Artifacts and manifest exported to: %s\n", outDir)
	}
	fmt.Fprintln(w, "================================================================================")
}

func cmdVerify(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("verify", flag.ContinueOnError)
	fs.SetOutput(stderr)

	var (
		dir    string
		asJSON bool
	)

	fs.StringVar(&dir, "dir", "", "Directory containing evaluation artifacts")
	fs.BoolVar(&asJSON, "json", false, "Output verification as JSON")

	if err := fs.Parse(args); err != nil {
		return err
	}
	if dir == "" {
		return errors.New("missing required -dir flag")
	}

	manifest, err := cudadiageval.VerifyManifest(dir)
	if err != nil {
		return fmt.Errorf("verification failed: %w", err)
	}

	// Also verify report content if present
	repPath := filepath.Join(dir, "evaluation_report.json")
	repData, err := os.ReadFile(repPath)
	if err != nil {
		return fmt.Errorf("read evaluation report: %w", err)
	}

	var report cudadiageval.EvaluationReport
	if err := json.Unmarshal(repData, &report); err != nil {
		return fmt.Errorf("parse evaluation report: %w", err)
	}

	status := struct {
		Verified       bool                        `json:"verified"`
		ManifestDigest string                      `json:"manifest_digest"`
		FilesChecked   int                         `json:"files_checked"`
		ReportPassed   bool                        `json:"report_passed"`
		Gates          []cudadiageval.PropertyGate `json:"gates"`
	}{
		Verified:       true,
		ManifestDigest: manifest.ManifestSHA256,
		FilesChecked:   len(manifest.Files),
		ReportPassed:   report.Passed,
		Gates:          report.Gates,
	}

	if asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(status)
	}

	fmt.Fprintf(stdout, "Artifact verification PASSED for %s\n", dir)
	fmt.Fprintf(stdout, "Manifest SHA256: %s (%d files verified)\n", manifest.ManifestSHA256, len(manifest.Files))
	fmt.Fprintf(stdout, "Evaluation result: Passed=%t, Incorrect Joins=%d\n", report.Passed, report.Attribution.IncorrectJoins)
	return nil
}

func cmdClaim(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("claim", flag.ContinueOnError)
	fs.SetOutput(stderr)

	var (
		repPath string
		asJSON  bool
	)

	fs.StringVar(&repPath, "report", "", "Path to evaluation_report.json or artifact directory")
	fs.BoolVar(&asJSON, "json", false, "Output claim as JSON")

	if err := fs.Parse(args); err != nil {
		return err
	}
	if repPath == "" {
		return errors.New("missing required -report flag")
	}

	// Check if directory
	fi, err := os.Stat(repPath)
	if err != nil {
		return err
	}
	if fi.IsDir() {
		repPath = filepath.Join(repPath, "evaluation_report.json")
	}

	data, err := os.ReadFile(repPath)
	if err != nil {
		return fmt.Errorf("read report: %w", err)
	}

	var report cudadiageval.EvaluationReport
	if err := json.Unmarshal(data, &report); err != nil {
		return fmt.Errorf("unmarshal report: %w", err)
	}

	claim, err := cudadiageval.FormulateClaim(&report)
	if err != nil {
		return fmt.Errorf("formulate claim: %w", err)
	}

	if asJSON {
		b, _ := claim.JSON()
		fmt.Fprintln(stdout, string(b))
		return nil
	}

	fmt.Fprint(stdout, claim.Format())
	return nil
}

// cmdLedger checks receipt consistency; it never executes retained commands.
func cmdLedger(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("ledger", flag.ContinueOnError)
	fs.SetOutput(stderr)
	file := fs.String("file", "", "Bounded qualification ledger JSON")
	root := fs.String("root", "", "Receipt artifact root")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *file == "" || *root == "" || fs.NArg() != 0 {
		return errors.New("ledger requires -file and -root with no positional arguments")
	}
	f, err := os.Open(*file)
	if err != nil {
		return fmt.Errorf("open ledger: %w", err)
	}
	defer f.Close()
	ledger, err := cudadiageval.ReadQualificationLedger(f)
	if err != nil {
		return err
	}
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	results, err := cudadiageval.VerifyQualificationLedger(ctx, *root, *ledger)
	if err != nil {
		return err
	}
	return writeComparisonJSON(stdout, results)
}
