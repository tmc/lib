package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaregress"
)

const regressUsage = `usage: cudaprof regress <subcommand> [flags]

Subcommands:
  plan    create an experiment plan from an admitted incident
  check   verify plan artifact closure and environment reconstruction
  export  export a minimal reproducer bundle
  replay  execute a declared replay spec with an observed environment

Run 'cudaprof regress <subcommand> -h' for subcommand details.`

func runRegressCommand(args []string, stdout io.Writer) error {
	if len(args) == 0 {
		return errors.New(regressUsage)
	}

	switch args[0] {
	case "plan":
		return runRegressPlanCommand(args[1:], stdout)
	case "check":
		return runRegressCheckCommand(args[1:], stdout)
	case "export":
		return runRegressExportCommand(args[1:], stdout)
	case "replay":
		return runRegressReplayCommand(args[1:], stdout)
	case "-h", "--help", "help":
		fmt.Fprintln(stdout, regressUsage)
		return nil
	default:
		return fmt.Errorf("unknown regress subcommand %q\n%s", args[0], regressUsage)
	}
}

func runRegressPlanCommand(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("regress plan", flag.ContinueOnError)
	incidentPath := fs.String("incident", "", "path to admitted incident JSON")
	outputPath := fs.String("o", "", "output path for generated experiment plan JSON (default: stdout)")
	rounds := fs.Int("rounds", 5, "number of interleaved trial rounds")
	warmup := fs.Int("warmup", 2, "number of warmup passes")
	if err := fs.Parse(args); err != nil {
		return err
	}

	if *incidentPath == "" {
		return errors.New("usage: cudaprof regress plan -incident file.json [-o plan.json]")
	}

	data, err := os.ReadFile(*incidentPath)
	if err != nil {
		return fmt.Errorf("read incident: %w", err)
	}

	var inc cudaregress.Incident
	if err := json.Unmarshal(data, &inc); err != nil {
		return fmt.Errorf("parse incident JSON: %w", err)
	}

	plan, rec, err := cudaregress.NewPlanFromIncident(inc, cudaregress.PlanOptions{
		Rounds:     *rounds,
		WarmupRuns: *warmup,
	})
	if err != nil {
		return fmt.Errorf("create plan: %w", err)
	}

	type planOutput struct {
		Plan    *cudaregress.ExperimentPlan `json:"plan"`
		Receipt *cudaregress.PlanReceipt    `json:"receipt"`
	}

	outBytes, err := json.MarshalIndent(planOutput{Plan: plan, Receipt: rec}, "", "  ")
	if err != nil {
		return err
	}

	if *outputPath != "" {
		return os.WriteFile(*outputPath, outBytes, 0644)
	}
	_, err = fmt.Fprintln(stdout, string(outBytes))
	return err
}

func runRegressCheckCommand(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("regress check", flag.ContinueOnError)
	planPath := fs.String("plan", "", "path to experiment plan JSON")
	envPath := fs.String("environment", "", "path to separately observed environment JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}

	if *planPath == "" || *envPath == "" {
		return errors.New("usage: cudaprof regress check -plan plan.json -environment observed-environment.json")
	}

	plan, err := readRegressPlanFile(*planPath)
	if err != nil {
		return fmt.Errorf("read plan: %w", err)
	}
	var observed observedEnvironmentFile
	if err := readStrictFile(*envPath, &observed); err != nil {
		return fmt.Errorf("read observed environment: %w", err)
	}
	if observed.Version != 1 || strings.TrimSpace(observed.Provenance) == "" {
		return errors.New("observed environment requires version 1 and provenance")
	}
	if observed.Environment == (cudaregress.Environment{}) {
		return errors.New("observed environment is unavailable")
	}
	if err := cudaregress.ValidateEnvironment(&plan, observed.Environment); err != nil {
		return err
	}

	reconstructable, blockers := cudaregress.CheckReconstruction(&plan, observed.Environment)
	if reconstructable {
		_, err := fmt.Fprintf(stdout, "Plan %s: reconstructable (validated artifacts; caller-supplied environment, provenance %q)\n", plan.PlanID, observed.Provenance)
		return err
	}

	fmt.Fprintf(stdout, "Plan %s: NOT reconstructable (%d blockers):\n", plan.PlanID, len(blockers))
	for _, b := range blockers {
		fmt.Fprintf(stdout, "  - %s\n", b)
	}
	return nil
}

func runRegressExportCommand(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("regress export", flag.ContinueOnError)
	planPath := fs.String("plan", "", "path to experiment plan JSON")
	outputDir := fs.String("o", "", "output directory for minimal reproducer bundle")
	if err := fs.Parse(args); err != nil {
		return err
	}

	if *planPath == "" || *outputDir == "" {
		return errors.New("usage: cudaprof regress export -plan plan.json -o dir")
	}

	plan, err := readRegressPlanFile(*planPath)
	if err != nil {
		return fmt.Errorf("read plan: %w", err)
	}

	if err := cudaregress.ExportMinimalReproducer(&plan, *outputDir); err != nil {
		return err
	}

	fmt.Fprintf(stdout, "Exported minimal reproducer for plan %s to %s\n", plan.PlanID, *outputDir)
	return nil
}

// readRegressPlanFile accepts the plan command's bundle and standalone plans.
// Both schema attempts read the same bounded snapshot. A parsed plan must pass
// ValidatePlan; receipt linkage never substitutes for artifact authentication.
func readRegressPlanFile(path string) (cudaregress.ExperimentPlan, error) {
	var plan cudaregress.ExperimentPlan
	data, err := readBoundedJSONFile(path)
	if err != nil {
		return plan, err
	}
	if err := decodeStrictJSON(data, &plan); err == nil {
		return plan, cudaregress.ValidatePlan(&plan)
	}
	var bundle struct {
		Plan    *cudaregress.ExperimentPlan `json:"plan"`
		Receipt *cudaregress.PlanReceipt    `json:"receipt"`
	}
	if err := decodeStrictJSON(data, &bundle); err != nil {
		return plan, fmt.Errorf("decode plan or plan/receipt bundle: %w", err)
	}
	if bundle.Plan == nil || bundle.Receipt == nil {
		return plan, errors.New("plan bundle requires plan and receipt")
	}
	plan = *bundle.Plan
	if err := cudaregress.ValidatePlan(&plan); err != nil {
		return plan, err
	}
	if bundle.Receipt.PlanID != plan.PlanID || bundle.Receipt.PlanDigest != plan.PlanDigest || bundle.Receipt.IncidentID != plan.IncidentID {
		return plan, errors.New("plan bundle receipt identity does not match plan")
	}
	return plan, nil
}
