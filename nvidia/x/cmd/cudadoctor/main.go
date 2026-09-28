package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"sort"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/internal/cudatool"
)

type check struct {
	Name       string   `json:"name"`
	Status     string   `json:"status"`
	Path       string   `json:"path,omitempty"`
	Version    string   `json:"version,omitempty"`
	Detail     string   `json:"detail,omitempty"`
	Remedy     string   `json:"remedy,omitempty"`
	Candidates []string `json:"candidates,omitempty"`
}

type report struct {
	Schema      int               `json:"schema"`
	Workflow    string            `json:"workflow"`
	Go          string            `json:"go"`
	Platform    string            `json:"platform"`
	Module      *debug.Module     `json:"module,omitempty"`
	Environment map[string]string `json:"environment"`
	Build       map[string]string `json:"build,omitempty"`
	Checks      []check           `json:"checks"`
}

func main() { os.Exit(run(os.Args[1:], os.Stdout, os.Stderr)) }

func run(args []string, out, errout io.Writer) int {
	fs := flag.NewFlagSet("cudadoctor", flag.ContinueOnError)
	fs.SetOutput(errout)
	workflow := fs.String("workflow", "jit", "jit, bind, aot, profile, sanitize, or tileir")
	jsonOutput := fs.Bool("json", false, "write a JSON report")
	smoke := fs.String("smoke", "none", "JIT smoke test: none, compile, load, or run (load/run use GPU)")
	arch := fs.String("arch", "compute_75", "virtual architecture for the JIT smoke test")
	timeout := fs.Duration("timeout", 15*time.Second, "timeout for each probe")
	probe := fs.String("probe", "", "internal subprocess probe")
	if err := fs.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	if *timeout <= 0 || fs.NArg() != 0 {
		fmt.Fprintln(errout, "cudadoctor: positive timeout and no positional arguments required")
		return 2
	}
	if *probe != "" {
		return probeMain(*probe, *arch, out, errout)
	}
	plan, ok := workflows[*workflow]
	if !ok {
		fmt.Fprintln(errout, "cudadoctor: unknown workflow", *workflow)
		return 2
	}
	switch *smoke {
	case "none", "compile", "load", "run":
	default:
		fmt.Fprintln(errout, "cudadoctor: unknown smoke test", *smoke)
		return 2
	}
	if *smoke != "none" && *workflow != "jit" {
		fmt.Fprintln(errout, "cudadoctor: smoke tests require -workflow=jit")
		return 2
	}
	r := report{Schema: 1, Workflow: *workflow, Go: runtime.Version(), Platform: runtime.GOOS + "/" + runtime.GOARCH, Environment: map[string]string{}}
	if info, ok := debug.ReadBuildInfo(); ok {
		r.Build = map[string]string{}
		for _, setting := range info.Settings {
			if strings.HasPrefix(setting.Key, "vcs") || setting.Key == "CGO_ENABLED" {
				r.Build[setting.Key] = setting.Value
			}
		}
		for _, dep := range info.Deps {
			if dep.Path == "github.com/tmc/lib/nvidia" {
				r.Module = dep
			}
		}
		if info.Main.Path == "github.com/tmc/lib/nvidia" {
			r.Module = &info.Main
		}
	}
	for _, key := range []string{"CUDA_HOME", "CUDA_PATH", "CUDA_ROOT", "CUDA_TOOLKIT_PATH", "CUDA_LIB_PATH", "NVRTC_LIB_PATH", "NVJITLINK_LIB_PATH", "CUPTI_LIB_PATH", "CUDABIND_LIBCLANG", "LD_LIBRARY_PATH", "PATH", "CODEGEN_SKIP_BINDINGS"} {
		if v, ok := os.LookupEnv(key); ok {
			r.Environment[key] = v
		}
	}
	for _, name := range plan.tools {
		r.Checks = append(r.Checks, toolCheck(name, *timeout))
	}
	for _, name := range plan.libraries {
		r.Checks = append(r.Checks, childCheck(name, *arch, *timeout))
	}
	if *workflow == "profile" {
		c := toolCheck("cuda-gdb", *timeout)
		c.Name = "cuda-gdb (optional)"
		if c.Status == "fail" {
			c.Status = "warn"
		}
		r.Checks = append(r.Checks, c)
	}
	if *smoke != "none" {
		r.Checks = append(r.Checks, childCheck("smoke-"+*smoke, *arch, *timeout))
	}
	r.Checks = append(r.Checks, check{Name: "execution", Status: "info", Detail: "Library discovery does not initialize a CUDA context or prove kernel execution; only -smoke=load/run uses the GPU."})
	addVersionWarning(&r, *timeout)
	if err := writeReport(out, r, *jsonOutput); err != nil {
		fmt.Fprintln(errout, err)
		return 1
	}
	for _, c := range r.Checks {
		if c.Status == "fail" {
			return 1
		}
	}
	return 0
}

type workflowPlan struct{ tools, libraries []string }

var workflows = map[string]workflowPlan{
	"jit":      {libraries: []string{"nvrtc"}},
	"bind":     {libraries: []string{"libclang"}},
	"aot":      {tools: []string{"nvcc", "ptxas"}},
	"profile":  {tools: []string{"nvdisasm"}, libraries: []string{"driver", "cupti"}},
	"sanitize": {tools: []string{"compute-sanitizer"}, libraries: []string{"driver"}},
	"tileir":   {libraries: []string{"nvrtc", "tileir"}},
}

func childCheck(name, arch string, timeout time.Duration) check {
	c := check{Name: name, Status: "fail"}
	exe, err := os.Executable()
	if err != nil {
		c.Detail = err.Error()
		return c
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, exe, "-probe="+name, "-arch="+arch)
	cmd.WaitDelay = time.Second
	var stderr strings.Builder
	cmd.Stderr = &stderr
	data, err := cmd.Output()
	if ctx.Err() != nil {
		c.Detail = "probe timed out"
		return c
	}
	if err != nil {
		c.Detail = fmt.Sprintf("probe failed: %v: %s", err, stderr.String())
		return c
	}
	if err = json.Unmarshal(data, &c); err != nil {
		c.Detail = "invalid probe response: " + err.Error()
		c.Status = "fail"
	}
	return c
}

func toolCheck(name string, timeout time.Duration) check {
	c := check{Name: name, Status: "fail", Candidates: append(cudatool.SearchPaths(name), "PATH:"+name)}
	path, err := cudatool.LookTool(name, "")
	if err != nil {
		c.Detail = err.Error()
		c.Remedy = "Install the selected workflow's CUDA tool; check CUDA_HOME, CUDA_PATH, CUDA_ROOT and PATH."
		return c
	}
	c.Path = path
	if resolved, err := filepath.EvalSymlinks(path); err == nil {
		c.Path = resolved
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	result, err := cudatool.Run(ctx, path, "", nil, "--version")
	if err != nil {
		c.Detail = err.Error()
		c.Remedy = "Check that the selected executable can run on this host; inspect its dependencies or select another toolkit."
		return c
	}
	c.Status = "pass"
	c.Detail = strings.TrimSpace(string(append(result.Stdout, result.Stderr...)))
	if v, ok := cudatool.ParseVersion(append(result.Stdout, result.Stderr...)); ok {
		c.Version = v.String()
	}
	c.Detail += "; selected by cudatool: toolkit candidates precede PATH"
	return c
}

func addVersionWarning(r *report, timeout time.Duration) {
	// nvcc is optional for JIT. Observe its version without making it required.
	if r.Workflow != "jit" && r.Workflow != "tileir" {
		return
	}
	c := toolCheck("nvcc", timeout)
	if c.Status != "pass" {
		return
	}
	c.Name = "nvcc (optional)"
	r.Checks = append(r.Checks, c)
	for _, v := range r.Checks {
		if v.Name == "nvrtc" && v.Status == "pass" && c.Version != "" && v.Version != c.Version {
			r.Checks = append(r.Checks, check{Name: "compiler versions", Status: "warn", Detail: fmt.Sprintf("NVRTC %s and nvcc %s differ; this is not by itself an incompatibility", v.Version, c.Version), Remedy: "Check NVRTC_LIB_PATH and toolkit selection; preserve both versions in benchmark provenance."})
			break
		}
	}
}

func writeReport(w io.Writer, r report, jsonOutput bool) error {
	if jsonOutput {
		enc := json.NewEncoder(w)
		enc.SetIndent("", "  ")
		return enc.Encode(r)
	}
	if _, err := fmt.Fprintf(w, "cudadoctor: %s (%s, %s)\n", r.Workflow, r.Go, r.Platform); err != nil {
		return err
	}
	keys := make([]string, 0, len(r.Environment))
	for k := range r.Environment {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		if _, err := fmt.Fprintf(w, "%s=%s\n", k, r.Environment[k]); err != nil {
			return err
		}
	}
	if r.Module != nil {
		if _, err := fmt.Fprintf(w, "module: %s %s\n", r.Module.Path, r.Module.Version); err != nil {
			return err
		}
		if r.Module.Replace != nil {
			if _, err := fmt.Fprintf(w, "replace: %s %s\n", r.Module.Replace.Path, r.Module.Replace.Version); err != nil {
				return err
			}
		}
	}
	for _, c := range r.Checks {
		if _, err := fmt.Fprintf(w, "%s %s %s %s\n  %s\n", c.Status, c.Name, c.Version, c.Path, c.Detail); err != nil {
			return err
		}
		if c.Remedy != "" {
			if _, err := fmt.Fprintln(w, "  "+c.Remedy); err != nil {
				return err
			}
		}
	}
	return nil
}
