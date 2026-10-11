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
	"time"

	"github.com/tmc/lib/nvidia/x/cudadiagdemo"
	"github.com/tmc/lib/nvidia/x/cudaregress"
)

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintf(os.Stderr, "gpudiagdemo: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string, stdout, stderr io.Writer) error {
	if len(args) == 0 {
		printUsage(stderr)
		return errors.New("command required: run or verify")
	}

	switch args[0] {
	case "run":
		return cmdRun(args[1:], stdout, stderr)
	case "verify":
		return cmdVerify(args[1:], stdout, stderr)
	case "help", "-h", "--help":
		printUsage(stdout)
		return nil
	default:
		printUsage(stderr)
		return fmt.Errorf("unknown command %q", args[0])
	}
}

func printUsage(w io.Writer) {
	fmt.Fprintln(w, "gpudiagdemo: GPU submission-delay diagnosis and regression trial demonstration")
	fmt.Fprintln(w, "")
	fmt.Fprintln(w, "Usage:")
	fmt.Fprintln(w, "  gpudiagdemo run [flags]      Execute end-to-end demonstration")
	fmt.Fprintln(w, "  gpudiagdemo verify [flags]   Validate artifact chain in directory")
	fmt.Fprintln(w, "")
	fmt.Fprintln(w, "Flags for 'run':")
	fmt.Fprintln(w, "  -n, -requests int           Total requests to simulate (default 16)")
	fmt.Fprintln(w, "  -c, -concurrency int        Concurrent worker goroutines (default 4)")
	fmt.Fprintln(w, "  -host-delay duration        Injected host submission delay (default 15ms)")
	fmt.Fprintln(w, "  -device-delay duration      Injected device kernel duration (default 300us)")
	fmt.Fprintln(w, "  -out-dir string             Directory to export complete artifact chain")
	fmt.Fprintln(w, "  -json                       Output full result as JSON")
	fmt.Fprintln(w, "  -native                     Physical CUDA/NVRTC trials, JSON partial-qualification receipt")
	fmt.Fprintln(w, "  -device/-elements/-rounds    Native trial configuration")
	fmt.Fprintln(w, "  -lock-dir                   Canonical GPU lease directory")
	fmt.Fprintln(w, "  -skip-trial                 Skip regression intervention trial replay")
}

func cmdRun(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	fs.SetOutput(stderr)

	var (
		requests                 int
		concurrency              int
		hostDelay                time.Duration
		deviceDelay              time.Duration
		outDir                   string
		asJSON                   bool
		skipTrial                bool
		native                   bool
		device, elements, rounds int
		lockDir                  string
	)

	fs.IntVar(&requests, "requests", 16, "Total requests to simulate")
	fs.IntVar(&requests, "n", 16, "Total requests (shorthand)")
	fs.IntVar(&concurrency, "concurrency", 4, "Concurrent worker goroutines")
	fs.IntVar(&concurrency, "c", 4, "Concurrent worker goroutines (shorthand)")
	fs.DurationVar(&hostDelay, "host-delay", 15*time.Millisecond, "Injected host submission delay")
	fs.DurationVar(&deviceDelay, "device-delay", 300*time.Microsecond, "Injected device kernel duration")
	fs.StringVar(&outDir, "out-dir", "", "Directory to export complete artifact chain")
	fs.BoolVar(&asJSON, "json", false, "Output full result as JSON")
	fs.BoolVar(&skipTrial, "skip-trial", false, "Skip regression intervention trial replay")

	fs.BoolVar(&native, "native", false, "Execute physical CUDA/NVRTC trials under the canonical GPU lease")
	fs.IntVar(&device, "device", 0, "Native device ordinal")
	fs.IntVar(&elements, "elements", 4096, "Native vector elements (maximum 1048576)")
	fs.IntVar(&rounds, "rounds", 3, "Native paired trial rounds (maximum 64)")
	fs.StringVar(&lockDir, "lock-dir", "", "Native canonical lease directory (default cudalock)")

	if err := fs.Parse(args); err != nil {
		return err
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	if native {
		res, runErr := cudadiagdemo.RunNative(ctx, cudadiagdemo.NativeConfig{Device: device, Elements: elements, Rounds: rounds, HostDelay: hostDelay, LockDir: lockDir, OutputDir: outDir})
		if res != nil {
			data, err := json.MarshalIndent(res, "", "  ")
			if err != nil {
				return errors.Join(runErr, err)
			}
			if _, err := fmt.Fprintln(stdout, string(data)); err != nil {
				return errors.Join(runErr, err)
			}
		}
		return runErr
	}

	cfg := cudadiagdemo.DemonstrationConfig{
		NumRequests:       requests,
		Concurrency:       concurrency,
		HostDelay:         hostDelay,
		DeviceDelay:       deviceDelay,
		OutputDir:         outDir,
		SkipIntervention:  skipTrial,
		ExerciseCorruptEv: true,
		ExercisePressure:  true,
	}

	if !asJSON {
		fmt.Fprintf(stdout, "Running GPU submission-delay diagnosis demonstration (%d requests, concurrency %d)...\n",
			cfg.NumRequests, cfg.Concurrency)
	}

	res, err := cudadiagdemo.RunDemonstration(ctx, cfg)
	if err != nil {
		return fmt.Errorf("demonstration failed: %w", err)
	}

	if asJSON {
		data, err := json.MarshalIndent(res, "", "  ")
		if err != nil {
			return fmt.Errorf("failed to marshal JSON result: %w", err)
		}
		fmt.Fprintln(stdout, string(data))
		return nil
	}

	printHumanSummary(stdout, res)
	return nil
}

func printHumanSummary(w io.Writer, res *cudadiagdemo.DemonstrationResult) {
	fmt.Fprintln(w, "")
	fmt.Fprintln(w, "=== Demonstration Summary ===")
	fmt.Fprintf(w, "Attribution Accuracy: %.2f%% (Exact: %d, Associated: %d, Missing: %d, Ambiguous: %d)\n",
		res.AttributionAccuracy*100,
		res.AttributionMetrics.ExactJoins,
		res.AttributionMetrics.AssociatedJoins,
		res.AttributionMetrics.MissingJoins,
		res.AttributionMetrics.AmbiguousJoins,
	)
	fmt.Fprintf(w, "Baseline Diagnosis:   %s (admitted: %t)\n",
		res.BaselineDiagnosis.Mechanism, res.BaselineDiagnosis.Admitted)
	fmt.Fprintf(w, "Host Delay Diagnosis: %s (admitted: %t, rule: %s)\n",
		res.HostDelayDiagnosis.Mechanism, res.HostDelayDiagnosis.Admitted, res.HostDelayDiagnosis.PrimaryFinding.Rule)
	fmt.Fprintf(w, "Device Bottleneck:    %s (admitted: %t)\n",
		res.DeviceBottleneckDiag.Mechanism, res.DeviceBottleneckDiag.Admitted)
	if res.WithheldDiagnosis.Mechanism != "" {
		fmt.Fprintf(w, "Negative Control:     %s (reason: %s)\n",
			res.WithheldDiagnosis.Mechanism, res.WithheldDiagnosis.WithheldReason)
	}

	if res.TrialComparison != nil {
		fmt.Fprintln(w, "")
		fmt.Fprintln(w, "=== Controlled Intervention Trial ===")
		fmt.Fprintf(w, "Trial Plan:           %s\n", res.TrialComparison.PlanID)
		fmt.Fprintf(w, "Verdict:              %s\n", res.TrialComparison.Verdict)
		fmt.Fprintf(w, "Baseline Duration:    %v\n", res.TrialComparison.BaselineDuration)
		fmt.Fprintf(w, "Candidate Duration:   %v\n", res.TrialComparison.CandidateDuration)
		fmt.Fprintf(w, "Speedup Delta:        %v (ratio: %.2fx)\n", res.TrialComparison.Delta, res.TrialComparison.Ratio)
		fmt.Fprintf(w, "Exceeds Noise Floor:  %t (noise floor: %v)\n",
			res.TrialComparison.ExceedsNoiseFloor, res.TrialComparison.NullDriftNoiseFloor)
	}

	fmt.Fprintln(w, "")
	fmt.Fprintln(w, res.ConfusionMatrix.FormatReport())

	if len(res.ArtifactFiles) > 0 {
		fmt.Fprintln(w, "Exported Artifact Chain:")
		tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
		for kind, path := range res.ArtifactFiles {
			fmt.Fprintf(tw, "  %s:\t%s\n", kind, path)
		}
		tw.Flush()
	}
	fmt.Fprintln(w, "")
}

func cmdVerify(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("verify", flag.ContinueOnError)
	fs.SetOutput(stderr)

	var dir string
	fs.StringVar(&dir, "dir", "", "Directory containing exported artifacts")

	if err := fs.Parse(args); err != nil {
		return err
	}
	if dir == "" {
		if fs.NArg() > 0 {
			dir = fs.Arg(0)
		} else {
			return errors.New("verify requires -dir or directory argument")
		}
	}

	requiredFiles := []string{
		"inputs.json",
		"ground_truth.json",
		"diagnosis.json",
		"summary.json",
	}

	for _, name := range requiredFiles {
		p := filepath.Join(dir, name)
		if _, err := os.Stat(p); err != nil {
			return fmt.Errorf("missing required artifact %s: %w", name, err)
		}
	}

	sumData, err := os.ReadFile(filepath.Join(dir, "summary.json"))
	if err != nil {
		return fmt.Errorf("read summary.json: %w", err)
	}
	var res cudadiagdemo.DemonstrationResult
	if err := json.Unmarshal(sumData, &res); err != nil {
		return fmt.Errorf("parse summary.json: %w", err)
	}

	if res.HostDelayDiagnosis.Mechanism != "host_submission_delay" {
		return fmt.Errorf("unexpected host delay diagnosis: got %q, want host_submission_delay",
			res.HostDelayDiagnosis.Mechanism)
	}
	if !res.HostDelayDiagnosis.Admitted {
		return errors.New("host delay diagnosis was not admitted")
	}

	if res.TrialPlan != nil {
		if err := cudaregress.ValidatePlan(res.TrialPlan); err != nil {
			return fmt.Errorf("trial plan validation failed: %w", err)
		}
	}

	fmt.Fprintf(stdout, "Artifact chain in %s verified successfully (attribution accuracy %.2f%%, mechanism %s admitted).\n",
		dir, res.AttributionAccuracy*100, res.HostDelayDiagnosis.Mechanism)
	return nil
}
