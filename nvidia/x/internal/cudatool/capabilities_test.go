package cudatool

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"
)

func TestCorpusLoads(t *testing.T) {
	tools := Tools()
	want := []string{"compute-sanitizer", "cuobjdump", "nvcc", "nvdisasm", "nvlink", "nvprune", "ptxas"}
	if !slices.Equal(tools, want) {
		t.Fatalf("Tools() = %q, want %q", tools, want)
	}
	for _, tool := range tools {
		versions := KnownVersions(tool)
		if len(versions) == 0 {
			t.Errorf("%s: no versions recorded", tool)
		}
		for _, v := range versions {
			opts, source, ok := KnownOptions(tool, v)
			if !ok || source.Compare(v) != 0 {
				t.Errorf("KnownOptions(%q, %v) = _, %v, %v; want the exact version", tool, v, source, ok)
			}
			if len(opts) == 0 {
				t.Errorf("%s %v: no options recorded", tool, v)
			}
		}
	}
}

// TestKnownOptionsVersionSelection pins the rule for a version the corpus does
// not record: a version between two records is described by the lower one, a
// version below everything by the oldest record, and a version above everything
// by nothing at all, which is what sends Capabilities to the tool itself.
func TestKnownOptionsVersionSelection(t *testing.T) {
	tests := []struct {
		name       string
		ask        Version
		wantSource Version
		wantOK     bool
	}{
		{"exact", V(13, 0), V(13, 0), true},
		{"between records", V(13, 2), V(13, 0), true},
		{"below everything", V(11, 8), V(12, 6), true},
		{"above everything", V(14, 0), Version{}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, source, ok := KnownOptions("ptxas", tt.ask)
			if ok != tt.wantOK || source.Compare(tt.wantSource) != 0 {
				t.Errorf("KnownOptions(ptxas, %v) = _, %v, %v; want %v, %v", tt.ask, source, ok, tt.wantSource, tt.wantOK)
			}
		})
	}
	if _, _, ok := KnownOptions("no-such-tool", V(13, 0)); ok {
		t.Error("KnownOptions reported an answer for a tool it has no record of")
	}
}

// TestMeasuredVersionDifferences records the flag differences measured across
// the toolkits installed on the GB10 host. They are the reason this corpus
// exists: each row is a command line that works under one toolkit and fails
// under another, and a toolkit upgrade that changes one of them should fail here
// rather than in a caller.
func TestMeasuredVersionDifferences(t *testing.T) {
	tests := []struct {
		tool    string
		option  string
		present []Version
		absent  []Version
		arity   Arity
	}{
		// nvcc introduced the sanitizer instrumentation option in 13.0 and
		// renamed it in 13.4, so exactly one version accepts each spelling.
		{tool: "nvcc", option: "--sanitize", present: []Version{V(13, 0)}, absent: []Version{V(12, 6), V(13, 4)}, arity: Value},
		{tool: "nvcc", option: "--fdevice-sanitize", present: []Version{V(13, 4)}, absent: []Version{V(12, 6), V(13, 0)}, arity: Value},
		// ptxas gained a way to ask which architectures it accepts.
		{tool: "ptxas", option: "--list-arch", present: []Version{V(13, 4)}, absent: []Version{V(12, 6), V(13, 0)}, arity: Bare},
		{tool: "ptxas", option: "--jobserver", present: []Version{V(13, 4)}, absent: []Version{V(12, 6), V(13, 0)}, arity: Bare},
		{tool: "nvlink", option: "--link-as-tools-patch", present: []Version{V(13, 4)}, absent: []Version{V(12, 6), V(13, 0)}, arity: Bare},
		// These two compute-sanitizer options take no argument on every
		// version installed: passing one is rejected with "extra yes detected
		// on the input command line" and no report is written.
		{tool: "compute-sanitizer", option: "--kill", present: []Version{V(12, 6), V(13, 0), V(13, 4)}, arity: Bare},
		{tool: "compute-sanitizer", option: "--track-unused-memory", present: []Version{V(12, 6), V(13, 0), V(13, 4)}, arity: Bare},
		{tool: "compute-sanitizer", option: "--leak-check", present: []Version{V(12, 6), V(13, 0), V(13, 4)}, arity: Value},
		// Nothing in any installed version emits CSV, though the tool's docs
		// have been read that way before.
		{tool: "compute-sanitizer", option: "--csv", absent: []Version{V(12, 6), V(13, 0), V(13, 4)}},
		// Of the two tools this tree drives, cuobjdump's option names and
		// arities are identical across all three versions, while nvdisasm
		// gained exactly one option in 13.0: --emit-json, which cudasass
		// emits for JSON mode and must therefore refuse under 12.6.
		{tool: "nvdisasm", option: "--print-code", present: []Version{V(12, 6), V(13, 0), V(13, 4)}, arity: Bare},
		{tool: "nvdisasm", option: "--emit-json", present: []Version{V(13, 0), V(13, 4)}, absent: []Version{V(12, 6)}, arity: Bare},
		{tool: "cuobjdump", option: "--dump-sass", present: []Version{V(12, 6), V(13, 0), V(13, 4)}, arity: Bare},
	}
	for _, tt := range tests {
		t.Run(tt.tool+tt.option, func(t *testing.T) {
			for _, v := range tt.present {
				opts, _, ok := KnownOptions(tt.tool, v)
				if !ok {
					t.Fatalf("no record for %s %v", tt.tool, v)
				}
				opt, ok := opts.Find(tt.option)
				if !ok {
					t.Errorf("%s %v does not accept %s, want it to", tt.tool, v, tt.option)
					continue
				}
				if opt.Arity != tt.arity {
					t.Errorf("%s %v %s arity = %v, want %v", tt.tool, v, tt.option, opt.Arity, tt.arity)
				}
			}
			for _, v := range tt.absent {
				opts, _, ok := KnownOptions(tt.tool, v)
				if !ok {
					t.Fatalf("no record for %s %v", tt.tool, v)
				}
				if _, ok := opts.Find(tt.option); ok {
					t.Errorf("%s %v accepts %s, want it not to", tt.tool, v, tt.option)
				}
			}
		})
	}
}

func TestSearchPaths(t *testing.T) {
	env := func(m map[string]string) func(string) string {
		return func(k string) string { return m[k] }
	}
	tests := []struct {
		name  string
		tool  string
		goos  string
		env   map[string]string
		want  []string
		empty bool
	}{
		{
			name: "default prefixes",
			tool: "nvdisasm",
			goos: "linux",
			want: []string{
				"/usr/local/cuda/bin/nvdisasm",
				"/usr/local/cuda/compute-sanitizer/nvdisasm",
				"/usr/local/cuda-13/bin/nvdisasm",
			},
		},
		{
			name: "CUDA_HOME first",
			tool: "compute-sanitizer",
			goos: "linux",
			env:  map[string]string{"CUDA_HOME": "/usr/local/cuda-13.4"},
			want: []string{
				"/usr/local/cuda-13.4/bin/compute-sanitizer",
				"/usr/local/cuda-13.4/compute-sanitizer/compute-sanitizer",
				"/usr/local/cuda/bin/compute-sanitizer",
			},
		},
		{
			name: "env naming a default prefix is not repeated",
			tool: "ptxas",
			goos: "linux",
			env:  map[string]string{"CUDA_PATH": "/usr/local/cuda"},
			want: []string{
				"/usr/local/cuda/bin/ptxas",
				"/usr/local/cuda/compute-sanitizer/ptxas",
				"/usr/local/cuda-13/bin/ptxas",
			},
		},
		{name: "darwin has no toolkit layout", tool: "ptxas", goos: "darwin", empty: true},
		{name: "no tool named", tool: "", goos: "linux", empty: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := searchPaths(tt.tool, tt.goos, env(tt.env))
			if tt.empty {
				if len(got) != 0 {
					t.Fatalf("searchPaths = %q, want none", got)
				}
				return
			}
			if len(got) < len(tt.want) {
				t.Fatalf("searchPaths = %q, want it to start with %q", got, tt.want)
			}
			for i, want := range tt.want {
				if got[i] != want {
					t.Fatalf("searchPaths[%d] = %q, want %q\ngot %q", i, got[i], want, got)
				}
			}
			seen := map[string]bool{}
			for _, p := range got {
				if seen[p] {
					t.Errorf("searchPaths = %q, want no duplicate %q", got, p)
				}
				seen[p] = true
			}
		})
	}
}

func TestIsExecutableFile(t *testing.T) {
	dir := t.TempDir()
	exe := filepath.Join(dir, "exe")
	if err := os.WriteFile(exe, nil, 0o755); err != nil {
		t.Fatal(err)
	}
	plain := filepath.Join(dir, "plain")
	if err := os.WriteFile(plain, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name string
		path string
		want bool
	}{
		{"executable", exe, true},
		{"not executable", plain, false},
		{"directory", dir, false},
		{"absent", filepath.Join(dir, "absent"), false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsExecutableFile(tt.path); got != tt.want {
				t.Errorf("IsExecutableFile(%q) = %v, want %v", tt.path, got, tt.want)
			}
		})
	}
}

func TestParseVersion(t *testing.T) {
	tests := []struct {
		name string
		out  string
		want Version
		ok   bool
	}{
		{
			name: "nvcc 13.4",
			out:  "nvcc: NVIDIA (R) Cuda compiler driver\nCuda compilation tools, release 13.4, V13.4.59\n",
			want: V(13, 4), ok: true,
		},
		{
			name: "nvcc 12.6",
			out:  "Cuda compilation tools, release 12.6, V12.6.85\n",
			want: V(12, 6), ok: true,
		},
		{
			name: "compute-sanitizer reports its own product version",
			out:  "NVIDIA (R) Compute Sanitizer\nVersion 2026.3.0.0 (build 38637409) (public-release)\n",
			ok:   false,
		},
		{name: "empty", out: "", ok: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := ParseVersion([]byte(tt.out))
			if ok != tt.ok || got.Compare(tt.want) != 0 {
				t.Errorf("ParseVersion = %v, %v; want %v, %v", got, ok, tt.want, tt.ok)
			}
		})
	}
}

func TestVersionFromPath(t *testing.T) {
	tests := []struct {
		path string
		want Version
		ok   bool
	}{
		{path: "/usr/local/cuda-13.4/bin/compute-sanitizer", want: V(13, 4), ok: true},
		{path: "/usr/local/cuda-12.6/compute-sanitizer/compute-sanitizer", want: V(12, 6), ok: true},
		// An unversioned prefix that is not a link to a versioned one.
		{path: "/usr/local/cuda/bin/compute-sanitizer", ok: false},
		{path: "/usr/bin/compute-sanitizer", ok: false},
	}
	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			got, ok := VersionFromPath(tt.path)
			if ok != tt.ok || got.Compare(tt.want) != 0 {
				t.Errorf("VersionFromPath(%q) = %v, %v; want %v, %v", tt.path, got, ok, tt.want, tt.ok)
			}
		})
	}
}

// TestVersionFromPathFollowsLinks covers the usual install shape: the tool is
// reached through an unversioned link whose target names the version.
func TestVersionFromPathFollowsLinks(t *testing.T) {
	dir := t.TempDir()
	real := filepath.Join(dir, "cuda-13.4", "bin")
	if err := os.MkdirAll(real, 0o755); err != nil {
		t.Fatal(err)
	}
	exe := filepath.Join(real, "compute-sanitizer")
	if err := os.WriteFile(exe, nil, 0o755); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(dir, "cuda")
	if err := os.Symlink(filepath.Join(dir, "cuda-13.4"), link); err != nil {
		t.Fatal(err)
	}
	through := filepath.Join(link, "bin", "compute-sanitizer")
	got, ok := VersionFromPath(through)
	if !ok || got.Compare(V(13, 4)) != 0 {
		t.Errorf("VersionFromPath(%q) = %v, %v; want 13.4, true", through, got, ok)
	}
}

func TestVersionCompare(t *testing.T) {
	tests := []struct {
		a, b Version
		want int
	}{
		{V(13, 0), V(13, 0), 0},
		{V(12, 6), V(13, 0), -1},
		{V(13, 4), V(13, 0), 1},
		{V(13, 0), V(13, 4), -1},
		{Version{}, V(12, 6), -1},
	}
	for _, tt := range tests {
		if got := tt.a.Compare(tt.b); got != tt.want {
			t.Errorf("%v.Compare(%v) = %d, want %d", tt.a, tt.b, got, tt.want)
		}
	}
	if V(0, 0).Valid() {
		t.Error("the zero Version reports itself valid")
	}
}

// fakeTool writes a script that answers --version and --help, so both the
// corpus path and the probe path of Capabilities run on any host.
func fakeTool(t *testing.T, name, version, help string) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("the fake tool is a shell script")
	}
	dir := t.TempDir()
	path := filepath.Join(dir, name)
	script := "#!/bin/sh\ncase \"$1\" in\n--version) cat <<'VER'\n" + version + "\nVER\n;;\n--help) cat <<'HELP'\n" + help + "\nHELP\n;;\nesac\n"
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestCapabilitiesUsesCorpusThenProbe exercises both answers: a toolkit the
// corpus covers is answered without running --help at all, and one newer than
// anything measured is answered by the tool itself.
func TestCapabilitiesUsesCorpusThenProbe(t *testing.T) {
	const probeHelp = `
Usage  : nvdisasm [options] <image file>

Options
=======

--brand-new-option <value>                     (-bno)    
        An option no measured toolkit had.
        Default value:  7.

--print-code                                   (-c)      
        Print the code.
`
	tests := []struct {
		name       string
		version    string
		wantVer    Version
		wantProbed bool
		wantOption string
		absent     string
	}{
		{
			name:       "version the corpus covers",
			version:    "Cuda compilation tools, release 13.0, V13.0.88",
			wantVer:    V(13, 0),
			wantProbed: false,
			wantOption: "--print-code",
			absent:     "--brand-new-option",
		},
		{
			name:       "toolkit newer than anything measured",
			version:    "Cuda compilation tools, release 99.9, V99.9.0",
			wantVer:    V(99, 9),
			wantProbed: true,
			wantOption: "--brand-new-option",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := fakeTool(t, "nvdisasm", tt.version, probeHelp)
			caps, err := Capabilities(context.Background(), "nvdisasm", path)
			if err != nil {
				t.Fatalf("Capabilities: %v", err)
			}
			if caps.Version.Compare(tt.wantVer) != 0 {
				t.Errorf("Version = %v, want %v", caps.Version, tt.wantVer)
			}
			if caps.Probed != tt.wantProbed {
				t.Errorf("Probed = %v, want %v", caps.Probed, tt.wantProbed)
			}
			if !caps.Accepts(tt.wantOption) {
				t.Errorf("Accepts(%q) = false, want true", tt.wantOption)
			}
			if tt.absent != "" && caps.Accepts(tt.absent) {
				t.Errorf("Accepts(%q) = true, want false: the corpus answer must not come from --help", tt.absent)
			}
			if arity, ok := caps.Arity("--print-code"); !ok || arity != Bare {
				t.Errorf("Arity(--print-code) = %v, %v; want bare, true", arity, ok)
			}
			if _, ok := caps.Arity("--no-such-option"); ok {
				t.Error("Arity reported an option the tool does not accept")
			}
		})
	}
}

// TestCapabilitiesRecoversVersionFromPrefix covers compute-sanitizer, which
// reports a product version rather than a toolkit one.
func TestCapabilitiesRecoversVersionFromPrefix(t *testing.T) {
	dir := t.TempDir()
	prefix := filepath.Join(dir, "cuda-13.0", "bin")
	if err := os.MkdirAll(prefix, 0o755); err != nil {
		t.Fatal(err)
	}
	src := fakeTool(t, "compute-sanitizer", "NVIDIA (R) Compute Sanitizer\nVersion 2025.3.1.0 (build 36400806)", "")
	b, err := os.ReadFile(src)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(prefix, "compute-sanitizer")
	if err := os.WriteFile(path, b, 0o755); err != nil {
		t.Fatal(err)
	}
	caps, err := Capabilities(context.Background(), "compute-sanitizer", path)
	if err != nil {
		t.Fatalf("Capabilities: %v", err)
	}
	if caps.Version.Compare(V(13, 0)) != 0 {
		t.Errorf("Version = %v, want 13.0 recovered from the install prefix", caps.Version)
	}
	if caps.Probed {
		t.Error("Probed = true, want the corpus to answer for 13.0")
	}
	if arity, ok := caps.Arity("--kill"); !ok || arity != Bare {
		t.Errorf("Arity(--kill) = %v, %v; want bare, true", arity, ok)
	}
}
