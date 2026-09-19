package cudadevicetable

import (
	"embed"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
)

//go:embed ptxprobe/*.ptx
var fixtureFS embed.FS

// targetPlaceholder is what each fixture writes in its .target directive. The
// probe substitutes it, so one fixture serves every target and no fixture can
// disagree with another about what it is testing.
const targetPlaceholder = "TARGET"

// Fixtures returns the embedded probe fixtures keyed by feature name, which is
// the fixture's base name.
func Fixtures() map[string]string {
	entries, err := fixtureFS.ReadDir("ptxprobe")
	if err != nil {
		panic("cudadevicetable: embedded ptxprobe: " + err.Error())
	}
	out := make(map[string]string, len(entries))
	for _, e := range entries {
		b, err := fixtureFS.ReadFile("ptxprobe/" + e.Name())
		if err != nil {
			panic("cudadevicetable: embedded ptxprobe: " + err.Error())
		}
		out[strings.TrimSuffix(e.Name(), ".ptx")] = string(b)
	}
	return out
}

// Outcome is what the assembler did with a fixture.
type Outcome int

const (
	// Accepted means ptxas assembled the fixture for the target.
	Accepted Outcome = iota

	// Rejected means ptxas refused the fixture and said the construct is not
	// supported for the target. This is the only negative that is a fact
	// about the target.
	Rejected

	// Crashed means ptxas died on a signal. A crash says something about the
	// assembler and nothing about the target.
	Crashed

	// Inconclusive means ptxas failed for some other reason -- a malformed
	// fixture, a PTX version too old for the target, a missing file. It is
	// not evidence either way.
	Inconclusive
)

func (o Outcome) String() string {
	switch o {
	case Accepted:
		return "accepted"
	case Rejected:
		return "rejected"
	case Crashed:
		return "crashed"
	default:
		return "inconclusive"
	}
}

// rejectionPattern matches the diagnostics ptxas emits when a construct exists
// but is not available for the requested target. Anything else that fails is
// inconclusive rather than a rejection: the difference between "this target
// says no" and "this fixture is broken" is the difference between a
// measurement and an invented one.
var rejectionPattern = regexp.MustCompile(`(?i)not supported on \.target|Feature '[^']*' not supported|requires \.target`)

// Result is one fixture assembled for one target.
type Result struct {
	Feature    string
	Target     string
	Outcome    Outcome
	Diagnostic string
	Command    string
}

// ProbeFeature assembles one fixture for one target with the ptxas at
// ptxasPath and classifies the result.
//
// ptxas signals failure with exit status 255, not 1, and dies with a signal on
// some malformed input. Reading any non-zero status as "unsupported" turns
// both of those into confident negatives, which is how a broken fixture
// becomes a row of hardware facts.
func ProbeFeature(ptxasPath, feature, source, target string) (Result, error) {
	dir, err := os.MkdirTemp("", "cudadevicetable-probe")
	if err != nil {
		return Result{}, err
	}
	defer os.RemoveAll(dir)

	src := filepath.Join(dir, feature+".ptx")
	body := strings.ReplaceAll(source, targetPlaceholder, target)
	if body == source {
		return Result{}, fmt.Errorf("fixture %q contains no %s placeholder", feature, targetPlaceholder)
	}
	if err := os.WriteFile(src, []byte(body), 0o644); err != nil {
		return Result{}, err
	}

	cmd := exec.Command(ptxasPath, "-arch="+target, src, "-o", filepath.Join(dir, "out.cubin"))
	out, err := cmd.CombinedOutput()
	res := Result{
		Feature: feature,
		Target:  target,
		Command: fmt.Sprintf("ptxas -arch=%s %s.ptx", target, feature),
	}
	switch {
	case err == nil:
		res.Outcome = Accepted
	case cmd.ProcessState != nil && signaled(cmd.ProcessState.ExitCode()):
		res.Outcome = Crashed
		res.Diagnostic = fmt.Sprintf("ptxas died on a signal (exit %d)", cmd.ProcessState.ExitCode())
	case rejectionPattern.Match(out):
		res.Outcome = Rejected
		res.Diagnostic = firstErrorLine(out, src, feature)
	default:
		res.Outcome = Inconclusive
		res.Diagnostic = firstErrorLine(out, src, feature)
	}
	return res, nil
}

// signaled reports whether an exit code is a shell-style signal status. Go
// reports a signalled child as -1 through ExitCode; a shell reports 128+signal.
func signaled(code int) bool {
	return code < 0 || (code > 128 && code < 192)
}

// firstErrorLine returns ptxas's first error line with the temporary source
// path rewritten to the fixture name. The temporary directory is unique per
// run, so leaving it in would make a recorded diagnostic differ from itself on
// every regeneration and bury a real change in the noise.
func firstErrorLine(out []byte, src, feature string) string {
	text := strings.ReplaceAll(string(out), src, feature+".ptx")
	for _, line := range strings.Split(text, "\n") {
		if strings.Contains(line, "error") {
			return strings.TrimSpace(line)
		}
	}
	return strings.TrimSpace(strings.SplitN(text, "\n", 2)[0])
}

// ProbeTargets assembles every embedded fixture for every target and returns
// the results sorted by feature then target.
func ProbeTargets(ptxasPath string, targets []string) ([]Result, error) {
	fixtures := Fixtures()
	names := make([]string, 0, len(fixtures))
	for name := range fixtures {
		names = append(names, name)
	}
	sort.Strings(names)

	var results []Result
	for _, name := range names {
		for _, target := range targets {
			r, err := ProbeFeature(ptxasPath, name, fixtures[name], target)
			if err != nil {
				return nil, fmt.Errorf("probe %s for %s: %w", name, target, err)
			}
			results = append(results, r)
		}
	}
	return results, nil
}

// BuildTargets turns probe results into table records. A feature accepted
// nowhere is reported as an error rather than recorded: see the positive
// control rule in validateTargets.
func BuildTargets(results []Result, probe FeatureProbe) (map[string]Target, error) {
	out, err := buildTargets(results, probe)
	if err != nil {
		return nil, err
	}
	if err := (Table{Schema: 1, Targets: out}).validateTargets(); err != nil {
		return nil, err
	}
	return out, nil
}

// buildTargets records the conclusive results without applying the rules that
// range over the whole table. The positive control asks whether a feature is
// supported *somewhere*, and a caller probing a single target has a population
// of one, in which "somewhere" collapses to "here" -- so every feature that
// target lacks reads as a fixture that assembles nowhere. Such a caller
// validates the union of its results with a corpus instead; see
// [Resolver.Resolve].
func buildTargets(results []Result, probe FeatureProbe) (map[string]Target, error) {
	out := map[string]Target{}
	for _, r := range results {
		switch r.Outcome {
		case Crashed, Inconclusive:
			// Not evidence. Recording it as unsupported is the mistake this
			// whole package exists to prevent, and recording it as supported
			// is worse.
			continue
		}
		t, ok := out[r.Target]
		if !ok {
			p := probe
			t = Target{Target: r.Target, Probe: &p, Features: map[string]Feature{}}
		}
		f := Feature{
			Supported:  r.Outcome == Accepted,
			Source:     Probed,
			Instrument: r.Command,
			Fixture:    "ptxprobe/" + r.Feature + ".ptx",
		}
		if !f.Supported {
			f.Diagnostic = r.Diagnostic
		}
		t.Features[r.Feature] = f
		out[r.Target] = t
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no conclusive probe results")
	}
	return out, nil
}

// NewFeatureProbe records the assembler identity for a probe run.
func NewFeatureProbe(ptxasPath, host, ptxVersion string) (FeatureProbe, error) {
	out, err := exec.Command(ptxasPath, "--version").Output()
	if err != nil {
		return FeatureProbe{}, fmt.Errorf("ptxas --version: %w", err)
	}
	version := ""
	for _, line := range strings.Split(string(out), "\n") {
		if i := strings.Index(line, ", V"); i >= 0 {
			version = strings.TrimSpace(line[i+3:])
		}
	}
	if version == "" {
		return FeatureProbe{}, fmt.Errorf("ptxas --version: no version in %q", out)
	}
	return FeatureProbe{
		Tool:        "ptxas",
		ToolVersion: version,
		PTXVersion:  ptxVersion,
		Host:        host,
		ProbedAt:    time.Now().UTC().Format("2006-01-02"),
	}, nil
}

// FixtureVersion reports the PTX .version the fixtures declare. It is read
// from the fixtures rather than written down beside them, so the version
// recorded in a probe stamp cannot drift from the one actually assembled.
//
// Fixtures that disagreed would make a single stamp false for some of the
// answers it covers, so disagreement is an error rather than a first match.
func FixtureVersion() (string, error) {
	version, from := "", ""
	for name, body := range Fixtures() {
		found := ""
		for _, line := range strings.Split(body, "\n") {
			if v, ok := strings.CutPrefix(strings.TrimSpace(line), ".version "); ok {
				found = strings.TrimSpace(v)
				break
			}
		}
		if found == "" {
			return "", fmt.Errorf("cudadevicetable: fixture %s declares no .version", name)
		}
		if version == "" {
			version, from = found, name
			continue
		}
		if found != version {
			return "", fmt.Errorf("cudadevicetable: fixtures disagree on .version: %s says %s, %s says %s", from, version, name, found)
		}
	}
	if version == "" {
		return "", fmt.Errorf("cudadevicetable: no fixtures")
	}
	return version, nil
}
