package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudadevicetable"
)

// runFeatures assembles the embedded PTX fixtures for each target with ptxas
// and records what it accepted.
//
// This mode needs no device. What a target supports is a property of the
// instruction set and the assembler, so it can be established on any host with
// a toolchain -- which is the only reason the table can hold facts about
// Hopper and GB200 at all, neither of which this project can attach.
func runFeatures(ptxasPath, targetList, update, host string) error {
	targets := strings.Split(targetList, ",")
	for i, t := range targets {
		targets[i] = strings.TrimSpace(t)
		if targets[i] == "" {
			return fmt.Errorf("empty target in -targets")
		}
	}
	if host == "" {
		host, _ = os.Hostname()
	}
	if host == "" {
		host = "unknown"
	}

	ptxVersion, err := cudadevicetable.FixtureVersion()
	if err != nil {
		return err
	}
	probe, err := cudadevicetable.NewFeatureProbe(ptxasPath, host, ptxVersion)
	if err != nil {
		return err
	}
	results, err := cudadevicetable.ProbeTargets(ptxasPath, targets)
	if err != nil {
		return err
	}

	// Report every outcome, including the ones that will not be recorded. A
	// crash or an inconclusive result is the signal that a fixture needs
	// fixing, and silently dropping it would leave a hole in the table that
	// looks exactly like a target nobody asked about.
	for _, r := range results {
		switch r.Outcome {
		case cudadevicetable.Crashed, cudadevicetable.Inconclusive:
			fmt.Fprintf(os.Stderr, "cudadevicetable: %s %s: %s, not recorded: %s\n",
				r.Feature, r.Target, r.Outcome, r.Diagnostic)
		default:
			fmt.Fprintf(os.Stderr, "cudadevicetable: %s %s: %s\n", r.Feature, r.Target, r.Outcome)
		}
	}

	built, err := cudadevicetable.BuildTargets(results, probe)
	if err != nil {
		return err
	}
	if update == "" {
		return writeJSON(os.Stdout, cudadevicetable.Table{Schema: 1, Targets: built})
	}
	return updateFeatureFile(update, built)
}

// updateFeatureFile replaces the probed targets in path, leaving targets this
// run did not cover alone.
func updateFeatureFile(path string, targets map[string]cudadevicetable.Target) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var t cudadevicetable.Table
	if err := json.Unmarshal(b, &t); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	if t.Schema == 0 {
		t.Schema = 1
	}
	if t.Targets == nil {
		t.Targets = map[string]cudadevicetable.Target{}
	}
	for name, tgt := range targets {
		t.Targets[name] = tgt
	}
	if err := t.Validate(); err != nil {
		return fmt.Errorf("%s: refusing to write: %w", path, err)
	}
	f, err := os.CreateTemp(filepath.Dir(path), "devices-*.json")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	if err := writeJSON(f, t); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(f.Name(), path)
}
