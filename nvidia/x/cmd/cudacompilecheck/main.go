// Command cudacompilecheck compiles CUDA C++ for one or more architectures
// without a GPU and reports each function's registers, spills, stack,
// barriers and shared memory, as ptxas -v prints them, and optionally a
// SASS instruction census from cuobjdump and a PTX instruction census.
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
// -ptx counts PTX instructions per function, which shows arithmetic
// differences between two versions of a kernel (an extra cvt.rn.bf16, a
// lost fma.rn) before ptxas reorders them. -save-ptx also writes the PTX,
// one file per source and architecture (k.sm_90a.ptx for k.cu), for
// checks the census cannot express.
//
// -max-registers and -no-spill turn the report into a gate: the command
// exits with status 1 if any entry function exceeds the limit or spills.
// -count N and -want-entries are gates on the set of entry functions, per
// file and architecture: -count fails unless there are exactly N, and
// -want-entries fails if any wanted name matches no entry. A wanted name is
// a comma-separated list or a file of names, one per line, with blank lines
// and # comments ignored; each matches an entry whose mangled name equals
// it or matches it as an anchored regular expression. They catch a
// kernel that silently failed to instantiate.
//
// -timeout bounds each tool run. Locally the tool runs in its own process
// group and the timeout kills the group; where process groups are not
// permitted the command says so once and the kill reaches only the tool.
// With -exec the tool also runs under timeout(1) on the compiling host,
// because the local group holds only the prefix.
//
// With -exec, each compile and disassembly also holds an exclusive
// flock(1) on -lock on the compiling host, waiting up to -lock-wait for
// it, so that several users of a small build machine queue instead of
// running out of memory together. Scripts that compile on the same host
// can take the same lock: flock /tmp/cudacompilecheck.lock nvcc ....
//
// A source ending in .py is a submission that builds its CUDA with
// torch.utils.cpp_extension.load_inline. The command reads each
// load_inline call's cuda_sources and extra_cuda_cflags without importing
// the file (with python3, statically: literals, top-level names and +),
// and compiles the translation unit load_inline would: torch's implicit
// includes, defines and nvcc flags, the call's flags in place of -flags,
// and its -gencode architectures unless -arch is given. -torch names the
// site-packages directory, on the compiling host, that holds torch's
// headers; a CPU-only torch wheel is enough.
//
//	cudacompilecheck -exec 'limactl shell --workdir / okcnvcc --' -nvcc ... \
//	    -torch /home/me.linux/venv/lib/python3.12/site-packages submission.py
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
	"io/fs"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"text/tabwriter"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaptx"
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
	ptx             bool
	savePTX         string
	top             int
	work            string
	maxRegisters    int
	noSpill         bool
	count           int
	wantEntries     []string
	timeout         time.Duration
	lock            string
	lockWait        time.Duration
	torch, python   string
}

// A Census is the instruction count of one function. LocalLoads and
// LocalStores count LDL and STL in SASS and ld.local and st.local in PTX.
type Census struct {
	Total      int            `json:"total"`
	LocalLoads int            `json:"local_loads"` // LDL
	LocalStore int            `json:"local_stores"`
	Opcodes    map[string]int `json:"opcodes"`
}

type function struct {
	*ptxasinfo.Function
	SASS *Census `json:",omitempty"`
	PTX  *Census `json:",omitempty"`
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
	fs.BoolVar(&c.ptx, "ptx", false, "add a PTX instruction census")
	fs.StringVar(&c.savePTX, "save-ptx", "", "write each source's PTX to `directory` as NAME.ARCH.ptx; implies -ptx")
	fs.IntVar(&c.top, "top", 8, "opcodes to print per function with -sass or -ptx")
	fs.StringVar(&c.work, "work", "", "work `directory`; must be visible to the compiling host")
	fs.IntVar(&c.maxRegisters, "max-registers", 0, "fail if an entry function uses more registers")
	fs.BoolVar(&c.noSpill, "no-spill", false, "fail if a function spills")
	fs.IntVar(&c.count, "count", -1, "fail unless each architecture has exactly `N` entry functions")
	var wantEntries string
	fs.StringVar(&wantEntries, "want-entries", "", "fail if an entry `name` is missing: a comma-separated list or a file, one per line; exact or regexp")
	fs.DurationVar(&c.timeout, "timeout", 5*time.Minute, "limit for each tool run")
	fs.StringVar(&c.lock, "lock", "/tmp/cudacompilecheck.lock", "with -exec, `file` on the compiling host that compiles hold with flock(1), so concurrent users queue; empty disables")
	fs.DurationVar(&c.lockWait, "lock-wait", 30*time.Minute, "limit on waiting for -lock")
	fs.StringVar(&c.torch, "torch", "", "site-packages `directory` holding torch on the compiling host, for .py sources")
	fs.StringVar(&c.python, "python", "python3", "local `python` that reads load_inline calls from .py sources")
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
	set := map[string]bool{}
	fs.Visit(func(f *flag.Flag) { set[f.Name] = true })
	c.exec = strings.Fields(execPrefix)
	if c.savePTX != "" {
		c.ptx = true
		if err := os.MkdirAll(c.savePTX, 0o755); err != nil {
			return err
		}
	}
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
	if wantEntries != "" {
		var err error
		if c.wantEntries, err = parseWantEntries(wantEntries); err != nil {
			return err
		}
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
		defer c.run(context.WithoutCancel(ctx), "rm", "-rf", outRoot)
	}
	var reports []*fileReport
	failed := false
	var torchFlags []string
	compileErrors := 0
	// fail records a source that did not compile and lets the others run.
	fail := func(name string, err error) {
		fmt.Fprintf(stderr, "cudacompilecheck: %s: %v\n", name, err)
		msg, _, _ := strings.Cut(err.Error(), "\n")
		reports = append(reports, &fileReport{Source: name, Failures: []string{"did not compile: " + msg}})
		compileErrors++
	}
	for i, src := range srcs {
		dir, out := filepath.Join(work, fmt.Sprint(i)), filepath.Join(outRoot, fmt.Sprint(i))
		if !strings.HasSuffix(src, ".py") {
			rep, err := check(ctx, &c, src, dir, out)
			if err != nil {
				fail(src, err)
				continue
			}
			failed = failed || len(rep.Failures) > 0
			reports = append(reports, rep)
			continue
		}
		units, err := loadInlineUnits(ctx, c.python, src)
		if err != nil {
			fail(src, err)
			continue
		}
		if torchFlags == nil {
			if torchFlags, err = c.torchFlags(ctx, work); err != nil {
				return err
			}
		}
		stem := strings.TrimSuffix(filepath.Base(src), ".py")
		for j, u := range units {
			at := fmt.Sprintf("%s:%d", src, u.Line)
			cc, err := unitConfig(&c, u, set["arch"], set["flags"])
			if err != nil {
				fail(at, err)
				continue
			}
			cc.extra = append(cc.extra, torchFlags...)
			name := stem
			if len(units) > 1 {
				name = fmt.Sprintf("%s.%d", stem, j)
			}
			cu, err := writeUnit(filepath.Join(dir, fmt.Sprint(j)), name, u)
			if err != nil {
				return err
			}
			rep, err := check(ctx, cc, cu, filepath.Join(dir, fmt.Sprint(j)), filepath.Join(out, fmt.Sprint(j)))
			if err != nil {
				fail(at, err)
				continue
			}
			rep.Source = at
			failed = failed || len(rep.Failures) > 0
			reports = append(reports, rep)
		}
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
	if compileErrors > 0 {
		return fmt.Errorf("%d of %d sources did not compile", compileErrors, len(reports))
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

// torchFlags returns the include flags for torch's headers on the
// compiling host, adding a stand-in for the one header a CPU-only wheel
// lacks under work, which the compiling host must see.
func (c *config) torchFlags(ctx context.Context, work string) ([]string, error) {
	if c.torch == "" {
		return nil, errors.New("a .py source needs -torch, the site-packages directory holding torch on the compiling host")
	}
	if out, err := c.run(ctx, "test", "-e", c.torch+"/torch/include/torch/extension.h"); err != nil {
		return nil, fmt.Errorf("-torch %s: no torch/include/torch/extension.h there: %v\n%s", c.torch, err, out)
	}
	flags := torchIncludes(c.torch)
	// torch/extension.h reaches Python.h; load_inline adds Python's include
	// directory as sysconfig reports it, so ask the compiling host's python3.
	if out, err := c.run(ctx, "python3", "-c", "import sysconfig; print(sysconfig.get_path('include'))"); err == nil {
		if dir := strings.TrimSpace(string(out)); dir != "" {
			flags = append(flags, "-isystem", dir)
		}
	}
	if _, err := c.run(ctx, "test", "-e", c.torch+"/torch/include/c10/cuda/impl/cuda_cmake_macros.h"); err != nil {
		stub := filepath.Join(work, "torchstub")
		if err := os.MkdirAll(filepath.Join(stub, "c10/cuda/impl"), 0o755); err != nil {
			return nil, err
		}
		if err := os.WriteFile(filepath.Join(stub, "c10/cuda/impl/cuda_cmake_macros.h"), []byte(cudaCMakeMacros), 0o644); err != nil {
			return nil, err
		}
		flags = append(flags, "-isystem", stub)
	}
	return flags, nil
}

// killedHint explains a tool killed by SIGKILL, which on Linux is usually
// the out-of-memory killer: cicc for a large translation unit can need
// gigabytes, and other compiles on a shared host count against the same
// memory.
func killedHint(err error) string {
	var ee *exec.ExitError
	if !errors.As(err, &ee) {
		return ""
	}
	if ee.ExitCode() == 137 || strings.Contains(ee.String(), "signal: killed") {
		return " (SIGKILL: probably out of memory on the compiling host; is another compile running there?)"
	}
	return ""
}

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
	if out, err := c.run(ctx, "mkdir", "-p", outDir); err != nil {
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

	output, err := c.compile(ctx, args...)
	if err != nil {
		return nil, fmt.Errorf("nvcc: %v%s\n%s", err, c.hint(err, c.nvcc, "-nvcc"), output)
	}
	info, err := ptxasinfo.Parse(bytes.NewReader(output))
	if err != nil {
		return nil, err
	}
	rep := &fileReport{Source: src, Command: append(append([]string(nil), c.exec...), args...), Warnings: info.Warnings, Notes: info.Notes}
	census := map[string]map[string]*Census{} // arch -> function -> census
	if c.sass {
		for _, a := range c.archs {
			text, err := c.compile(ctx, c.cuobjdump, "-sass", "-arch", a, out)
			if err != nil {
				return nil, fmt.Errorf("cuobjdump: %v%s\n%s", err, c.hint(err, c.cuobjdump, "-cuobjdump"), text)
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
	ptx := map[string]map[string]*Census{}
	if c.ptx {
		for _, a := range c.archs {
			if ptx[a], err = ptxCensus(ctx, c, src, input, outDir, a); err != nil {
				return nil, err
			}
		}
	}
	for _, f := range info.Functions() {
		rep.Functions = append(rep.Functions, function{Function: f, SASS: census[f.Arch][f.Name], PTX: ptx[f.Arch][f.Name]})
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
	rep.Failures = append(rep.Failures, entryFailures(c, info)...)
	return rep, nil
}

// parseWantEntries reads -want-entries: the name of an existing file of
// names, or a comma-separated list. Every name must compile as a regexp.
func parseWantEntries(v string) ([]string, error) {
	var names []string
	if data, err := os.ReadFile(v); err == nil {
		for _, line := range strings.Split(string(data), "\n") {
			if line = strings.TrimSpace(line); line != "" && !strings.HasPrefix(line, "#") {
				names = append(names, line)
			}
		}
	} else {
		for _, n := range strings.Split(v, ",") {
			if n = strings.TrimSpace(n); n != "" {
				names = append(names, n)
			}
		}
	}
	if len(names) == 0 {
		return nil, fmt.Errorf("-want-entries %q names no entries", v)
	}
	for _, n := range names {
		if _, err := regexp.Compile("^(?:" + n + ")$"); err != nil {
			return nil, fmt.Errorf("-want-entries: %w", err)
		}
	}
	return names, nil
}

// entryFailures applies -count and -want-entries to each architecture.
func entryFailures(c *config, info *ptxasinfo.Report) []string {
	if c.count < 0 && len(c.wantEntries) == 0 {
		return nil
	}
	var failures []string
	for _, a := range c.archs {
		var entries []string
		for _, f := range info.Functions() {
			if f.Entry && f.Arch == a {
				entries = append(entries, f.Name)
			}
		}
		if c.count >= 0 && len(entries) != c.count {
			failures = append(failures, fmt.Sprintf("%s: %d entry functions, want %d", a, len(entries), c.count))
		}
		for _, w := range c.wantEntries {
			re := regexp.MustCompile("^(?:" + w + ")$")
			found := false
			for _, e := range entries {
				if e == w || re.MatchString(e) {
					found = true
					break
				}
			}
			if !found {
				failures = append(failures, fmt.Sprintf("%s: no entry function matches %q", a, w))
			}
		}
	}
	return failures
}

// run runs a tool on the compiling host and returns its combined output.
//
// Locally the tool runs in its own process group, and the timeout kills the
// group, so compiler subprocesses (cicc, ptxas) die with nvcc. Where a
// process group cannot be created, run says so once on stderr and the
// timeout reaches only the direct child.
//
// With -exec the local group holds only the prefix (limactl, ssh, docker),
// not the tool on the far side, so the tool also runs under timeout(1)
// there with the same limit; the compiling host must provide it.
func (c *config) run(ctx context.Context, args ...string) ([]byte, error) {
	return c.runOn(ctx, false, args)
}

// compile is run for the heavy tools, which with -exec hold -lock.
func (c *config) compile(ctx context.Context, args ...string) ([]byte, error) {
	return c.runOn(ctx, true, args)
}

// lockBusy is flock's exit status when -lock-wait expires.
const lockBusy = 75

func (c *config) runOn(ctx context.Context, lock bool, args []string) ([]byte, error) {
	argv, limit := args, c.timeout
	if len(c.exec) > 0 {
		secs := int(math.Ceil(c.timeout.Seconds()))
		argv = append(append([]string(nil), c.exec...), "timeout", "-k", "5", fmt.Sprintf("%ds", secs))
		if lock && c.lock != "" {
			wait := int(math.Ceil(c.lockWait.Seconds()))
			argv = append(append(append([]string(nil), c.exec...), "flock", "-E", fmt.Sprint(lockBusy), "-w", fmt.Sprint(wait), c.lock), argv[len(c.exec):]...)
			limit += c.lockWait
		}
		argv = append(argv, args...)
	}
	return runTool(ctx, limit, argv)
}

// hint explains a tool failure that has a likely cause: a tool missing
// from the compiling host, a compiler killed for memory, or -lock held
// past -lock-wait.
func (c *config) hint(err error, tool, flagName string) string {
	var ee *exec.ExitError
	switch {
	case errors.Is(err, exec.ErrNotFound), errors.Is(err, fs.ErrNotExist), errors.As(err, &ee) && ee.ExitCode() == 127 && len(c.exec) > 0:
		return fmt.Sprintf(" (%s not found on the compiling host; %s gives its path)", tool, flagName)
	case errors.As(err, &ee) && ee.ExitCode() == lockBusy && len(c.exec) > 0 && c.lock != "":
		return fmt.Sprintf(" (waited %v for %s on the compiling host; another compile holds it)", c.lockWait, c.lock)
	}
	return killedHint(err)
}

// groupWarning receives the one-time notice that process groups are
// unavailable.
var (
	groupWarning     io.Writer = os.Stderr
	groupWarningOnce sync.Once
)

func runTool(ctx context.Context, timeout time.Duration, argv []string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	command := func(group bool) *exec.Cmd {
		cmd := exec.CommandContext(ctx, argv[0], argv[1:]...)
		// Grandchildren that keep the output pipe open must not hold
		// Wait past the timeout.
		cmd.WaitDelay = 5 * time.Second
		if group {
			setProcessGroup(cmd)
		}
		return cmd
	}
	cmd := command(true)
	out, err := cmd.CombinedOutput()
	grouped := true
	if cmd.Process == nil && isGroupError(err) {
		groupWarningOnce.Do(func() {
			fmt.Fprintf(groupWarning, "cudacompilecheck: cannot create a process group (%v); a timeout kills only %s, not its children\n", err, filepath.Base(argv[0]))
		})
		cmd = command(false)
		out, err = cmd.CombinedOutput()
		grouped = false
	}
	if err != nil && ctx.Err() == context.DeadlineExceeded {
		reach := "killed its process group"
		if !grouped {
			reach = "killed " + filepath.Base(argv[0]) + " only; its children may still be running"
		}
		return out, fmt.Errorf("timed out after %v (%s): %w", timeout, reach, err)
	}
	return out, err
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
		printCensus(w, "SASS", f.Arch, f.Name, f.SASS, top)
		printCensus(w, "PTX", f.Arch, f.Name, f.PTX, top)
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

func printCensus(w io.Writer, kind, arch, name string, c *Census, top int) {
	if c == nil {
		return
	}
	fmt.Fprintf(w, "  %s %s %s: %d instructions, %d local loads, %d local stores:", kind, arch, name, c.Total, c.LocalLoads, c.LocalStore)
	for _, o := range topOpcodes(c.Opcodes, top) {
		fmt.Fprintf(w, " %s=%d", o, c.Opcodes[o])
	}
	fmt.Fprintln(w)
}

// ptxCensus compiles input, built from src, to PTX for arch and counts
// each function's instructions.
func ptxCensus(ctx context.Context, c *config, src, input, outDir, arch string) (map[string]*Census, error) {
	out := filepath.Join(outDir, arch+".ptx")
	args := []string{c.nvcc, "-std=" + c.std, "-ptx", "-arch=compute_" + strings.TrimPrefix(arch, "sm_"), "-o", out}
	for _, d := range c.includes {
		args = append(args, "-I"+d)
	}
	for _, d := range c.defines {
		args = append(args, "-D"+d)
	}
	args = append(append(args, c.extra...), input)
	if text, err := c.compile(ctx, args...); err != nil {
		return nil, fmt.Errorf("nvcc -ptx: %v%s\n%s", err, c.hint(err, c.nvcc, "-nvcc"), text)
	}
	// The PTX is on the compiling host, which may be another machine.
	text, err := c.run(ctx, "cat", out)
	if err != nil {
		return nil, fmt.Errorf("read ptx: %v\n%s", err, text)
	}
	if c.savePTX != "" {
		name := strings.TrimSuffix(filepath.Base(src), filepath.Ext(src)) + "." + arch + ".ptx"
		if err := os.WriteFile(filepath.Join(c.savePTX, name), text, 0o644); err != nil {
			return nil, err
		}
	}
	m, err := cudaptx.ParseBytes(text)
	if err != nil {
		return nil, fmt.Errorf("parse ptx for %s: %w", arch, err)
	}
	census := map[string]*Census{}
	for _, k := range m.Kernels {
		if !k.Body {
			continue
		}
		s := k.Summary()
		census[k.Name] = &Census{
			Total:      s.Total,
			LocalLoads: s.Count("ld.local"),
			LocalStore: s.Count("st.local"),
			Opcodes:    s.Counts,
		}
	}
	return census, nil
}
