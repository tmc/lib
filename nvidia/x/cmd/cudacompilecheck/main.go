// Command cudacompilecheck compiles CUDA C++ for one or more architectures
// without a GPU and reports each function's registers, spills, stack,
// barriers and shared memory, as ptxas -v prints them, and optionally a
// SASS instruction census from cuobjdump.
//
// Usage:
//
//	cudacompilecheck [flags] file.cu...
//
// It runs nvcc locally, or through a command prefix given by -exec, such as
// a Lima VM or container that has the toolkit:
//
//	cudacompilecheck -exec 'limactl shell --workdir / okcnvcc --' \
//	    -nvcc /home/me.linux/venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc \
//	    -arch sm_90a,sm_121a -launchers kernels.cu
//
// With -exec, the sources and the work directory, which holds generated
// wrappers, must have the same path on both sides; Lima mounts the macOS
// home directory that way, read-only. The default work directory is then a
// hidden directory beside the first source. Compiler outputs go to a fresh
// directory under /tmp on the compiling host. Both are removed on exit.
//
// Templates are compiled only when instantiated. -keep names functions whose
// addresses a generated wrapper takes, and -launchers adds every
// "inline void name(" defined at the start of a line, so that launchers
// instantiate their kernels.
//
// -max-registers and -no-spill turn the report into a gate: the command
// exits with status 1 if any entry function exceeds the limit or spills.
//
// The command compiles only; it never loads a module or touches a device.
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
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/ptxasinfo"
)

func main() {
	if err := run(context.Background(), os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, "cudacompilecheck:", err)
		os.Exit(1)
	}
}

type listFlag []string

func (l *listFlag) String() string     { return strings.Join(*l, ",") }
func (l *listFlag) Set(s string) error { *l = append(*l, s); return nil }

type config struct {
	nvcc, cuobjdump string
	exec            []string
	archs           []string
	std             string
	includes        listFlag
	defines         listFlag
	extra           []string
	keep            []string
	launchers       bool
	sass            bool
	top             int
	work            string
	maxRegisters    int
	noSpill         bool
	timeout         time.Duration
}

// A Census is the SASS instruction count of one function.
type Census struct {
	Total      int            `json:"total"`
	LocalLoads int            `json:"local_loads"` // LDL
	LocalStore int            `json:"local_stores"`
	Opcodes    map[string]int `json:"opcodes"`
}

type function struct {
	*ptxasinfo.Function
	SASS *Census `json:",omitempty"`
}

type fileReport struct {
	Source    string     `json:"source"`
	Command   []string   `json:"command"`
	Functions []function `json:"functions"`
	Warnings  []string   `json:"warnings,omitempty"`
	Notes     []string   `json:"notes,omitempty"`
	Failures  []string   `json:"failures,omitempty"`
}

func run(ctx context.Context, args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("cudacompilecheck", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var c config
	var execPrefix, archs, extra, keep string
	fs.StringVar(&c.nvcc, "nvcc", "nvcc", "nvcc `path` on the compiling host")
	fs.StringVar(&c.cuobjdump, "cuobjdump", "cuobjdump", "cuobjdump `path` on the compiling host, for -sass")
	fs.StringVar(&execPrefix, "exec", "", "command `prefix` that runs nvcc and cuobjdump elsewhere, split on spaces")
	fs.StringVar(&archs, "arch", "sm_90a", "comma-separated target `architectures`")
	fs.StringVar(&c.std, "std", "c++20", "C++ standard")
	fs.Var(&c.includes, "I", "include `directory` (repeatable)")
	fs.Var(&c.defines, "D", "preprocessor `definition` (repeatable)")
	fs.StringVar(&extra, "flags", "-O3", "extra nvcc `flags`, split on spaces")
	fs.StringVar(&keep, "keep", "", "comma-separated function `names` to instantiate")
	fs.BoolVar(&c.launchers, "launchers", false, "instantiate every inline void function defined at the start of a line")
	fs.BoolVar(&c.sass, "sass", false, "add a SASS instruction census from cuobjdump")
	fs.IntVar(&c.top, "top", 8, "opcodes to print per function with -sass")
	fs.StringVar(&c.work, "work", "", "work `directory`; must be visible to the compiling host")
	fs.IntVar(&c.maxRegisters, "max-registers", 0, "fail if an entry function uses more registers")
	fs.BoolVar(&c.noSpill, "no-spill", false, "fail if a function spills")
	fs.DurationVar(&c.timeout, "timeout", 5*time.Minute, "limit for each tool run")
	asJSON := fs.Bool("json", false, "print JSON")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: cudacompilecheck [flags] file.cu...")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	if fs.NArg() == 0 {
		fs.Usage()
		return errors.New("no source files")
	}
	c.exec = strings.Fields(execPrefix)
	c.extra = strings.Fields(extra)
	for _, a := range strings.Split(archs, ",") {
		if a = strings.TrimSpace(a); a != "" {
			if !archRE.MatchString(a) {
				return fmt.Errorf("bad architecture %q, want sm_NN or sm_NNa", a)
			}
			c.archs = append(c.archs, a)
		}
	}
	if len(c.archs) == 0 {
		return errors.New("no architectures")
	}
	if keep != "" {
		c.keep = strings.Split(keep, ",")
	}

	srcs := make([]string, fs.NArg())
	for i, s := range fs.Args() {
		abs, err := filepath.Abs(s)
		if err != nil {
			return err
		}
		srcs[i] = abs
	}
	work := c.work
	if work == "" {
		var err error
		if len(c.exec) > 0 {
			work, err = os.MkdirTemp(filepath.Dir(srcs[0]), ".cudacompilecheck-")
		} else {
			work, err = os.MkdirTemp("", "cudacompilecheck-")
		}
		if err != nil {
			return err
		}
		defer os.RemoveAll(work)
	} else if work, _ = filepath.Abs(work); work == "" {
		return errors.New("bad work directory")
	}

	// Outputs go where the compiler can write: the work directory when
	// local, a fresh directory under /tmp on the remote side otherwise.
	outRoot := work
	if len(c.exec) > 0 {
		outRoot = fmt.Sprintf("/tmp/cudacompilecheck-%d-%d", os.Getpid(), time.Now().UnixNano())
		defer runTool(context.WithoutCancel(ctx), c.timeout, append(append([]string(nil), c.exec...), "rm", "-rf", outRoot))
	}
	var reports []*fileReport
	failed := false
	for i, src := range srcs {
		rep, err := check(ctx, &c, src, filepath.Join(work, fmt.Sprint(i)), filepath.Join(outRoot, fmt.Sprint(i)))
		if err != nil {
			return fmt.Errorf("%s: %w", src, err)
		}
		if len(rep.Failures) > 0 {
			failed = true
		}
		reports = append(reports, rep)
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(reports); err != nil {
			return err
		}
	} else {
		for _, r := range reports {
			printText(stdout, r, c.top)
		}
	}
	if failed {
		return errors.New("gate failed")
	}
	return nil
}

var (
	archRE     = regexp.MustCompile(`^sm_\d{2,3}[af]?$`)
	launcherRE = regexp.MustCompile(`(?m)^(?:static\s+)?inline\s+void\s+([A-Za-z_]\w*)\s*\(`)
)

// wrapper returns a translation unit that includes src and takes the
// address of each named function, or "" if there is nothing to keep.
func wrapper(src string, text []byte, c *config) string {
	names := append([]string(nil), c.keep...)
	if c.launchers {
		for _, m := range launcherRE.FindAllSubmatch(text, -1) {
			names = append(names, string(m[1]))
		}
	}
	if len(names) == 0 {
		return ""
	}
	var b strings.Builder
	fmt.Fprintf(&b, "#include %q\n", src)
	for i, n := range names {
		fmt.Fprintf(&b, "void* cudacompilecheck_keep_%d = (void*)&%s;\n", i, n)
	}
	return b.String()
}

func check(ctx context.Context, c *config, src, dir, outDir string) (*fileReport, error) {
	text, err := os.ReadFile(src)
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	if out, err := runTool(ctx, c.timeout, append(append([]string(nil), c.exec...), "mkdir", "-p", outDir)); err != nil {
		return nil, fmt.Errorf("make output directory: %v\n%s", err, out)
	}
	input := src
	if w := wrapper(src, text, c); w != "" {
		input = filepath.Join(dir, "wrap.cu")
		if err := os.WriteFile(input, []byte(w), 0o644); err != nil {
			return nil, err
		}
	}
	out := filepath.Join(outDir, "out.fatbin")
	args := []string{c.nvcc, "-std=" + c.std, "-fatbin", "-Xptxas", "-v", "-o", out}
	for _, a := range c.archs {
		args = append(args, fmt.Sprintf("-gencode=arch=compute_%s,code=%s", strings.TrimPrefix(a, "sm_"), a))
	}
	for _, d := range c.includes {
		args = append(args, "-I"+d)
	}
	for _, d := range c.defines {
		args = append(args, "-D"+d)
	}
	args = append(args, c.extra...)
	args = append(args, input)

	cmd := append(append([]string(nil), c.exec...), args...)
	output, err := runTool(ctx, c.timeout, cmd)
	if err != nil {
		return nil, fmt.Errorf("nvcc: %v\n%s", err, output)
	}
	info, err := ptxasinfo.Parse(bytes.NewReader(output))
	if err != nil {
		return nil, err
	}
	rep := &fileReport{Source: src, Command: cmd, Warnings: info.Warnings, Notes: info.Notes}
	census := map[string]map[string]*Census{} // arch -> function -> census
	if c.sass {
		for _, a := range c.archs {
			cmd := append(append([]string(nil), c.exec...), c.cuobjdump, "-sass", "-arch", a, out)
			text, err := runTool(ctx, c.timeout, cmd)
			if err != nil {
				return nil, fmt.Errorf("cuobjdump: %v\n%s", err, text)
			}
			d, err := cudasass.Parse(bytes.NewReader(text))
			if err != nil {
				return nil, fmt.Errorf("cuobjdump -arch %s: %w", a, err)
			}
			census[a] = map[string]*Census{}
			for _, fn := range d.Functions {
				s := fn.Summary()
				census[a][fn.Name] = &Census{
					Total:      s.Total,
					LocalLoads: s.Count("LDL"),
					LocalStore: s.Count("STL"),
					Opcodes:    s.Counts,
				}
			}
		}
	}
	for _, f := range info.Functions() {
		rep.Functions = append(rep.Functions, function{Function: f, SASS: census[f.Arch][f.Name]})
		if f.Entry && c.maxRegisters > 0 && f.Registers > c.maxRegisters {
			rep.Failures = append(rep.Failures, fmt.Sprintf("%s %s: %d registers > %d", f.Arch, f.Name, f.Registers, c.maxRegisters))
		}
		if c.noSpill && f.Spills() {
			rep.Failures = append(rep.Failures, fmt.Sprintf("%s %s: spills %d B stores, %d B loads", f.Arch, f.Name, f.SpillStores, f.SpillLoads))
		}
	}
	if len(rep.Functions) == 0 {
		return nil, fmt.Errorf("nvcc reported no functions; is -Xptxas -v output reaching stderr?\n%s", output)
	}
	return rep, nil
}

func runTool(ctx context.Context, timeout time.Duration, argv []string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, argv[0], argv[1:]...)
	return cmd.CombinedOutput()
}

func printText(w io.Writer, r *fileReport, top int) {
	fmt.Fprintf(w, "%s\n", r.Source)
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "arch\tkind\tregs\tmax\tstack\tspill st\tspill ld\tsmem\tbarriers\tfunction")
	for _, f := range r.Functions {
		kind := "device"
		if f.Entry {
			kind = "entry"
		}
		max := "-"
		if f.MaxRegisters > 0 {
			max = fmt.Sprint(f.MaxRegisters)
		}
		fmt.Fprintf(tw, "%s\t%s\t%d\t%s\t%d\t%d\t%d\t%d\t%d\t%s\n", f.Arch, kind, f.Registers, max,
			f.StackFrame, f.SpillStores, f.SpillLoads, f.SharedMem, f.Barriers, f.Name)
	}
	tw.Flush()
	for _, f := range r.Functions {
		if f.SASS == nil {
			continue
		}
		fmt.Fprintf(w, "  %s %s: %d instructions, %d LDL, %d STL:", f.Arch, f.Name, f.SASS.Total, f.SASS.LocalLoads, f.SASS.LocalStore)
		for _, o := range topOpcodes(f.SASS.Opcodes, top) {
			fmt.Fprintf(w, " %s=%d", o, f.SASS.Opcodes[o])
		}
		fmt.Fprintln(w)
	}
	for _, s := range r.Warnings {
		fmt.Fprintf(w, "  warning: %s\n", s)
	}
	for _, s := range r.Notes {
		fmt.Fprintf(w, "  note: %s\n", s)
	}
	for _, s := range r.Failures {
		fmt.Fprintf(w, "  FAIL: %s\n", s)
	}
}

func topOpcodes(m map[string]int, n int) []string {
	ops := make([]string, 0, len(m))
	for o := range m {
		ops = append(ops, o)
	}
	sort.Slice(ops, func(i, j int) bool {
		if m[ops[i]] != m[ops[j]] {
			return m[ops[i]] > m[ops[j]]
		}
		return ops[i] < ops[j]
	})
	return ops[:min(n, len(ops))]
}
