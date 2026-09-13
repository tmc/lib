package cudatool

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"
)

var (
	update  = flag.Bool("update", false, "rewrite optiondata from the toolkits installed on this host")
	helpDir = flag.String("helpdir", "", "distill --help output captured in this directory instead of running the tools")
)

// corpusTools are the tools the corpus covers. A tool is worth recording once
// anything in the tree might run it, because the record costs a few kilobytes
// and answers what a command line may contain without a toolkit installed.
var corpusTools = []string{"nvcc", "ptxas", "nvdisasm", "cuobjdump", "nvlink", "nvprune", "compute-sanitizer"}

// TestCorpusMatchesInstalledToolkits compares the shipped option sets against
// the toolkits installed on this host, and with -update rewrites them.
//
// It is the gate on a toolkit upgrade: when /usr/local/cuda moves to a version
// the corpus records, every option and every arity must still match, and a
// difference is reported here rather than discovered in a caller's command line.
// A version the corpus does not record is reported too, as a record worth
// adding, since until it is added [Capabilities] must run --help for it.
//
// With -update, the option sets are rewritten from the installed toolkits, or
// from a directory of captured --help output named <tool>-<version>.txt given
// with -helpdir. Capturing help output on one host and distilling it on another
// is the only way to record a toolkit this machine cannot run.
func TestCorpusMatchesInstalledToolkits(t *testing.T) {
	found := installedToolkits(t)
	if *helpDir != "" {
		found = capturedHelp(t, *helpDir)
	}
	if len(found) == 0 {
		t.Skip("no CUDA toolkit found: nothing to compare the recorded option sets against")
	}
	for _, f := range found {
		t.Run(fmt.Sprintf("%s-%s", f.tool, f.version), func(t *testing.T) {
			live, err := ParseHelp(f.help)
			if err != nil {
				t.Fatalf("parsing %s --help: %v", f.path, err)
			}
			path := filepath.Join("optiondata", fmt.Sprintf("%s-%s.txt", f.tool, f.version))
			if *update {
				if err := os.WriteFile(path, []byte(live.Text()), 0o644); err != nil {
					t.Fatal(err)
				}
				t.Logf("wrote %s: %d options", path, len(live))
				return
			}
			var v Version
			if _, err := fmt.Sscanf(f.version, "%d.%d", &v.Major, &v.Minor); err != nil {
				t.Fatalf("%s does not name a major.minor version: %v", f.version, err)
			}
			if known := KnownVersions(f.tool); !containsVersion(known, f.version) {
				t.Errorf("%s %s is installed but not recorded (records: %v); capture it with -update", f.tool, f.version, known)
				return
			}
			recorded, _, ok := KnownOptions(f.tool, v)
			if !ok {
				t.Errorf("no option set recorded for %s %s; capture it with -update", f.tool, f.version)
				return
			}
			diffOptions(t, recorded, live)
		})
	}
}

// diffOptions reports every difference between the recorded and live option
// sets, rather than stopping at the first, so one run names the whole change.
func diffOptions(t *testing.T, recorded, live Options) {
	t.Helper()
	for _, opt := range live {
		was, ok := recorded.Find(opt.Name)
		if !ok {
			t.Errorf("%s is new: the tool accepts it and the record does not", opt.Name)
			continue
		}
		if was.Arity != opt.Arity {
			t.Errorf("%s arity is %v, recorded as %v", opt.Name, opt.Arity, was.Arity)
		}
		if was.Default != opt.Default {
			t.Errorf("%s default is %q, recorded as %q", opt.Name, opt.Default, was.Default)
		}
		if strings.Join(was.Values, ",") != strings.Join(opt.Values, ",") {
			t.Errorf("%s values are %q, recorded as %q", opt.Name, opt.Values, was.Values)
		}
	}
	for _, opt := range recorded {
		if _, ok := live.Find(opt.Name); !ok {
			t.Errorf("%s is gone: the record has it and the tool does not", opt.Name)
		}
	}
}

// toolHelp is one tool's help output, with the toolkit version it came from.
type toolHelp struct {
	tool    string
	version string
	path    string
	help    []byte
}

// installedToolkits runs --help for every corpus tool in every toolkit prefix
// on this host.
func installedToolkits(t *testing.T) []toolHelp {
	t.Helper()
	prefixes, err := filepath.Glob("/usr/local/cuda-*")
	if err != nil || len(prefixes) == 0 {
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	var found []toolHelp
	for _, prefix := range prefixes {
		version := strings.TrimPrefix(filepath.Base(prefix), "cuda-")
		// A multi-version install also carries major-only links such as
		// cuda-13 beside the real prefixes. They name no version of their own
		// and would record a duplicate under a name that cannot be parsed.
		var v Version
		if _, err := fmt.Sscanf(version, "%d.%d", &v.Major, &v.Minor); err != nil {
			continue
		}
		for _, tool := range corpusTools {
			for _, sub := range toolkitSubdirs {
				path := filepath.Join(prefix, sub, tool)
				if !IsExecutableFile(path) {
					continue
				}
				out, err := Run(ctx, path, "", nil, "--help")
				help := append(out.Stdout, out.Stderr...)
				if len(help) == 0 {
					t.Logf("%s --help produced nothing: %v", path, err)
					break
				}
				found = append(found, toolHelp{tool: tool, version: version, path: path, help: help})
				break
			}
		}
	}
	sort.Slice(found, func(i, j int) bool {
		if found[i].tool != found[j].tool {
			return found[i].tool < found[j].tool
		}
		return found[i].version < found[j].version
	})
	return found
}

// capturedHelp reads help output saved as <tool>-<version>.txt.
func capturedHelp(t *testing.T, dir string) []toolHelp {
	t.Helper()
	files, err := filepath.Glob(filepath.Join(dir, "*.txt"))
	if err != nil {
		t.Fatal(err)
	}
	var found []toolHelp
	for _, file := range files {
		name := strings.TrimSuffix(filepath.Base(file), ".txt")
		i := strings.LastIndex(name, "-")
		if i < 0 {
			t.Fatalf("%s is not named <tool>-<version>.txt", file)
		}
		b, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		found = append(found, toolHelp{tool: name[:i], version: name[i+1:], path: file, help: b})
	}
	return found
}

func containsVersion(versions []Version, text string) bool {
	var v Version
	if _, err := fmt.Sscanf(text, "%d.%d", &v.Major, &v.Minor); err != nil {
		return false
	}
	for _, known := range versions {
		if known.Compare(v) == 0 {
			return true
		}
	}
	return false
}
