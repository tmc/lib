package cudadevicetable

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
)

// TestResolverAnswersATargetTheTableLacks runs a real assembler against a
// target that is deliberately absent from devices.json. It is the property
// that keeps the checked-in table from acting as an allowlist: a target
// nobody recorded still gets a measured answer, with provenance.
func TestResolverAnswersATargetTheTableLacks(t *testing.T) {
	ptxas, err := FindPTXAS()
	if err != nil {
		t.Skipf("no ptxas available to resolve with: %v", err)
	}
	skipUnlessFixtureISA(t, ptxas)
	// sm_86 is old enough to be uncontroversial and is not in the table.
	const target = "sm_86"
	if _, ok := LookupTarget(target); ok {
		t.Skipf("%s is now in the table; pick a target that is not", target)
	}

	r := NewResolver(ptxas, "test")
	a := r.Ask(target, "tma")
	if !a.Measured {
		t.Fatalf("Ask(%s, tma) came back unmeasured: %s", target, a.Why())
	}
	if a.Source == "" {
		t.Error("resolved answer carries no source")
	}
	if !strings.Contains(a.Source, "ptxas") {
		t.Errorf("source %q does not name the assembler", a.Source)
	}
	// TMA is a Hopper construct; an Ampere target must reject it. If this
	// ever passes, the fixture stopped testing what it claims to.
	if a.Supported {
		t.Errorf("%s accepted tma: %s", target, a.Why())
	}
	if a.Diagnostic == "" {
		t.Error("a rejection with no diagnostic is not a measurement")
	}

	// The second Ask must not launch another assembler. Resolved() is the
	// observable proxy: one target resolved, not two.
	r.Ask(target, "mbarrier")
	if got := r.Resolved(); len(got) != 1 {
		t.Errorf("Resolved() has %d targets after two Asks, want 1", len(got))
	}
}

// TestResolverDegradesToUnmeasured checks the failure direction. A resolver
// that cannot run anything must produce silence; producing a negative would
// be worse than having no resolver at all.
func TestResolverDegradesToUnmeasured(t *testing.T) {
	r := NewResolver("/nonexistent/ptxas", "test")
	a := r.Ask("sm_130a", "wgmma")
	if a.Measured {
		t.Fatalf("a resolver with no assembler claims a measurement: %s", a.Why())
	}
	if a.Supported {
		t.Error("unmeasured answer is supported")
	}
	if got := a.Or(AssumeSupported); !got {
		t.Error("policy not applied to a failed resolution")
	}
	// A target already in the table must still answer without the assembler.
	if a := r.Ask("sm_90a", "wgmma"); !a.Measured || !a.Supported {
		t.Errorf("table fast path broken when the assembler is missing: %s", a.Why())
	}
}

// TestResolverRejectsNonTargets keeps the probe from shelling out for a typo.
func TestResolverRejectsNonTargets(t *testing.T) {
	r := NewResolver("/nonexistent/ptxas", "test")
	for _, bad := range []string{"hopper", "compute_90", "sm_", ""} {
		if _, err := r.Resolve(bad); err == nil {
			t.Errorf("Resolve(%q) succeeded", bad)
		}
	}
}

// TestFindPTXASPrefersTheToolkit builds a machine with two assemblers -- an
// older one first on $PATH and the toolkit's own under $CUDA_HOME -- and
// requires FindPTXAS to answer with the toolkit's.
//
// That is the arrangement a distribution ptxas package creates, and it is not
// hypothetical: on the GPU host /usr/bin/ptxas is 12.0 while /usr/local/cuda
// is 13.4. The consequence is specific to this package. CUDA 12.0's ptxas
// does not know sm_121a, so a Resolver running it would record "unsupported"
// for a target that the machine's real toolkit supports -- a fabricated
// answer wearing the provenance of a measurement, which is the one outcome
// FindPTXAS's doc comment promises not to produce.
//
// The toolkit search is Linux-only because the CUDA toolkit ships only there;
// on other platforms $PATH is the only source and there is nothing to order.
func TestFindPTXASPrefersTheToolkit(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skipf("toolkit search paths are empty on %s: nothing to prefer over $PATH", runtime.GOOS)
	}
	dir := t.TempDir()
	pathDir := filepath.Join(dir, "usr", "bin")
	toolkit := filepath.Join(dir, "toolkit")
	toolkitBin := filepath.Join(toolkit, "bin")
	for _, d := range []string{pathDir, toolkitBin} {
		if err := os.MkdirAll(d, 0o777); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(d, "ptxas"), []byte("#!/bin/sh\nexit 0\n"), 0o777); err != nil {
			t.Fatal(err)
		}
	}
	t.Setenv("PTXAS", "")
	t.Setenv("CUDA_HOME", toolkit)
	t.Setenv("PATH", pathDir)

	got, err := FindPTXAS()
	if err != nil {
		t.Fatal(err)
	}
	if want := filepath.Join(toolkitBin, "ptxas"); got != want {
		t.Errorf("FindPTXAS() = %q, want %q (the $PATH copy at %q shadowed the toolkit)", got, want, filepath.Join(pathDir, "ptxas"))
	}
}

// TestFindPTXASNamedAssemblerWins checks that $PTXAS is used as given, with
// a toolkit and a $PATH copy both present to be passed over.
//
// There is deliberately no arm for "no assembler anywhere": the toolkit roots
// include the unconditional /usr/local/cuda, which no environment variable
// can hide, so a machine with a real toolkit cannot be made to have none.
// That case belongs to cudatool.LookTool, where the search paths are
// injectable.
func TestFindPTXASNamedAssemblerWins(t *testing.T) {
	dir := t.TempDir()
	named := filepath.Join(dir, "myptxas")
	if err := os.WriteFile(named, []byte("#!/bin/sh\nexit 0\n"), 0o777); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "ptxas"), []byte("#!/bin/sh\nexit 0\n"), 0o777); err != nil {
		t.Fatal(err)
	}
	t.Setenv("CUDA_HOME", dir)
	t.Setenv("PATH", dir)
	t.Setenv("PTXAS", named)
	if got, err := FindPTXAS(); err != nil || got != named {
		t.Errorf("FindPTXAS() with $PTXAS = (%q, %v), want (%q, nil)", got, err, named)
	}
}

// TestResolverResolvesEveryRecordedTargetAlone is the population-size control.
//
// The rules a resolved record is held to are the generator's, and one of them
// -- that a probed feature must be accepted by some target -- is a property of
// a corpus. A Resolver probes one target, so its corpus has one member, and
// "supported somewhere" collapses into "supported here": every feature the
// target lacks reads as a fixture that assembles nowhere. Measured before the
// fix, resolving each recorded target on its own failed for all of them; the
// only reason the package looked healthy is that its other resolver test picks
// sm_86, and the blamed feature is chosen by map iteration order, so the
// failure named a different feature each run and read as noise.
//
// Every recorded target must resolve alone, including the ones the table
// already covers: the table's answer is not the path under test here.
func TestResolverResolvesEveryRecordedTargetAlone(t *testing.T) {
	ptxas, err := FindPTXAS()
	if err != nil {
		t.Skipf("no ptxas available to resolve with: %v", err)
	}
	skipUnlessFixtureISA(t, ptxas)
	for _, target := range Targets() {
		t.Run(target, func(t *testing.T) {
			// A fresh Resolver per target is the point: a shared one would
			// carry the previous target's results and quietly restore the
			// population this test is trying to reduce to one.
			r := NewResolver(ptxas, "test")
			if _, err := r.Resolve(target); err != nil {
				t.Errorf("Resolve(%s) alone: %v", target, err)
			}
		})
	}
}

// skipUnlessFixtureISA skips t when ptxas cannot read the PTX ISA the
// embedded fixtures declare. An older ptxas makes every probe inconclusive,
// which the resolver reports correctly; it is not a measurement to test.
func skipUnlessFixtureISA(t *testing.T, ptxas string) {
	t.Helper()
	src, err := fixtureFS.ReadFile("ptxprobe/cluster.ptx")
	if err != nil {
		t.Fatal(err)
	}
	isa := regexp.MustCompile(`(?m)^\.version (\S+)$`).FindSubmatch(src)
	if isa == nil {
		t.Fatal("ptxprobe/cluster.ptx declares no .version")
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "empty.ptx")
	body := fmt.Sprintf(".version %s\n.target sm_80\n.address_size 64\n.visible .entry empty() { ret; }\n", isa[1])
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command(ptxas, "-arch=sm_80", path, "-o", filepath.Join(dir, "out.cubin")).CombinedOutput(); err != nil {
		t.Skipf("%s cannot assemble PTX %s, the fixtures' ISA: %s", ptxas, isa[1], bytes.TrimSpace(out))
	}
}
