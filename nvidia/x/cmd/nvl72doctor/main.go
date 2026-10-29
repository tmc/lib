package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"slices"
	"strings"
	"time"
)

func main() { os.Exit(run(os.Args[1:], os.Stdout, os.Stderr)) }

func run(args []string, out, errout io.Writer) int {
	fs := flag.NewFlagSet("nvl72doctor", flag.ContinueOnError)
	fs.SetOutput(errout)
	profileName := fs.String("profile", "gb300-nvl72", "rack profile: gb300-nvl72 or gb200-nvl72")
	jsonOutput := fs.Bool("json", false, "write the facts and checks as JSON")
	factsFile := fs.String("facts", "", "evaluate saved facts instead of probing this node")
	saveFacts := fs.String("save-facts", "", "write the observed facts to this file")
	doNCCL := fs.Bool("nccl", true, "run the NCCL all-reduce and all-to-all sweeps (uses the GPUs)")
	doCompile := fs.Bool("compile", true, "compile the kernel suite with nvcc")
	archList := fs.String("arch", "", "comma-separated compile targets; default the node's and the profile's sm_XYa")
	nvcc := fs.String("nvcc", "", "nvcc path; default the cudatool search")
	maxBytes := fs.Int("nccl-max-bytes", 256<<20, "largest NCCL message, bytes")
	timeout := fs.Duration("timeout", 5*time.Minute, "limit for each NCCL sweep and each compile")
	root := fs.String("root", "/", "prefix for /dev paths, for tests")
	probe := fs.String("probe", "", "internal child probe")
	ncclOp := fs.String("nccl-op", "", "internal: collective for -probe=nccl")
	ncclGPUs := fs.Int("nccl-gpus", 0, "internal: GPUs for -probe=nccl")
	if err := fs.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	if fs.NArg() != 0 || *timeout <= 0 || *maxBytes < 1<<10 {
		fmt.Fprintln(errout, "nvl72doctor: no arguments, a positive -timeout and -nccl-max-bytes >= 1024 required")
		return 2
	}
	if *probe == "nccl" {
		if err := ncclProbe(*ncclOp, *ncclGPUs, *maxBytes, out); err != nil {
			fmt.Fprintln(errout, "nvl72doctor:", err)
			return 1
		}
		return 0
	} else if *probe != "" {
		fmt.Fprintln(errout, "nvl72doctor: unknown probe", *probe)
		return 2
	}
	p, ok := profiles[*profileName]
	if !ok {
		fmt.Fprintln(errout, "nvl72doctor: unknown profile", *profileName)
		return 2
	}

	var f Facts
	if *factsFile != "" {
		data, err := os.ReadFile(*factsFile)
		if err == nil {
			err = json.Unmarshal(data, &f)
		}
		if err != nil {
			fmt.Fprintln(errout, "nvl72doctor:", err)
			return 2
		}
	} else {
		ctx := context.Background()
		f.Schema = 1
		probeNode(ctx, &f, *nvcc, *root)
		if *doNCCL && f.NVMLError == "" {
			var err error
			f.NCCL, err = runNCCL(ctx, len(f.GPUs), *maxBytes, *timeout)
			if err != nil {
				f.NCCLError = err.Error()
			}
		}
		if *doCompile && f.NVCC.Error == "" {
			f.Compile = compileSuite(ctx, f.NVCC.Path, compileArchs(*archList, f, p), *timeout)
		}
		if *saveFacts != "" {
			data, _ := json.MarshalIndent(f, "", "  ")
			if err := os.WriteFile(*saveFacts, append(data, '\n'), 0o644); err != nil {
				fmt.Fprintln(errout, "nvl72doctor:", err)
				return 1
			}
		}
	}

	checks := Evaluate(f, p)
	if err := writeReport(out, f, p, checks, *jsonOutput); err != nil {
		fmt.Fprintln(errout, "nvl72doctor:", err)
		return 1
	}
	for _, c := range checks {
		if c.Status == "fail" {
			return 1
		}
	}
	return 0
}

// compileArchs returns the -arch list, or the node's GPU architectures
// and the profile's, each with the "a" suffix.
func compileArchs(list string, f Facts, p Profile) []string {
	if list != "" {
		return strings.Split(list, ",")
	}
	var archs []string
	for _, g := range f.GPUs {
		if a := g.Arch(); a != "" && !slices.Contains(archs, a+"a") {
			archs = append(archs, a+"a")
		}
	}
	if !slices.Contains(archs, p.Arch()) {
		archs = append(archs, p.Arch())
	}
	return archs
}

func writeReport(w io.Writer, f Facts, p Profile, checks []Check, jsonOutput bool) error {
	if jsonOutput {
		enc := json.NewEncoder(w)
		enc.SetIndent("", "  ")
		return enc.Encode(struct {
			Profile string  `json:"profile"`
			Facts   Facts   `json:"facts"`
			Checks  []Check `json:"checks"`
		}{p.Name, f, checks})
	}
	if _, err := fmt.Fprintf(w, "nvl72doctor: %s on %s (%s)\n", p.Name, orText(f.Host, "unknown host"), f.OS); err != nil {
		return err
	}
	for _, c := range checks {
		ev := ""
		if c.Evidence != "" {
			ev = " [" + c.Evidence + "]"
		}
		if _, err := fmt.Fprintf(w, "%-4s %s%s\n     %s\n", c.Status, c.Name, ev, c.Detail); err != nil {
			return err
		}
	}
	return nil
}
