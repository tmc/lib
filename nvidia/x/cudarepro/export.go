package cudarepro

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"go/format"
	"os"
	"path/filepath"
	"reflect"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaschedule"
)

// ExportOptions configures standalone Go test package export.
type ExportOptions struct {
	// RunnerPath names a source-reviewed executable implementing the request and
	// response protocol in the exported reproducer_test.go. It receives preserved
	// Case and kernel bytes on stdin; no shell or inferred CUDA ABI is used.
	// Its receipt assertions do not establish native qualification.
	RunnerPath     string
	RunnerSHA256   string        // SHA-256 of the exact caller-supplied executable.
	AttemptTimeout time.Duration // Per-process deadline; zero uses 30 seconds.

	PackageName          string  // Package name for generated test files (defaults to "reproducer").
	Repeats              int     // Declared baseline replay count R.
	DetectionProbability float64 // Target detection probability for nondeterministic findings (e.g. 0.99).
	MaxAttempts          int     // Bound on test attempts.
	OracleRequired       bool    // Require lane 52 memory/access violation oracle.
	OraclePatch          string  // Oracle identifier or patch name (e.g. "lane52-memcheck").
}

// ArtifactOptions bounds a preserved exporter artifact package. Detection
// assumes independent attempts with probability equal to the measured rate;
// that model is explicit and is not a statistical confidence guarantee.
type ArtifactOptions struct {
	Repeats              int
	DetectionProbability float64
	MaxAttempts          int
}

// ExportPackage emits a standalone, runnable Go test package into a caller-named directory.
// The directory must be absolute, canonical, and not exist prior to export.
// TestReproduce refuses execution without a pinned caller-supplied runner. Each
// attempt starts a fresh process and retains bounded raw output in receipts.
// Candidate and oracle metadata from the runner remain assertions; they do not
// independently establish CUDA execution, custody, or native qualification.
// The emitted package contains:
//   - go.mod: module file for the standalone test package;
//   - testdata/source.ptx: original pristine kernel source;
//   - testdata/candidate.ptx: replayed/injected artifact to execute;
//   - testdata/finding.json: serialized campaign seed result;
//   - testdata/case.json: reduced workload inputs, shape, and launch geometry;
//   - testdata/policy.json: failure identity, measurement, and attempt policy;
//   - testdata/digests.json: SHA-256 digests of all preserved artifacts;
//   - reproducer_test.go: clean, runnable Go test verifying artifact integrity and
//     reproducing the recorded defect with the required oracle and bounded attempts.
func ExportPackage(dir string, finding cudaschedule.SeedResult, reduction Reduction, opts ExportOptions) error {
	if !filepath.IsAbs(dir) || filepath.Clean(dir) != dir {
		return fmt.Errorf("export package: directory must be absolute and canonical")
	}
	for p := filepath.Dir(dir); ; p = filepath.Dir(p) {
		info, err := os.Lstat(p)
		if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
			return fmt.Errorf("export package: invalid directory ancestor %q", p)
		}
		if p == filepath.Dir(p) {
			break
		}
	}
	if opts.PackageName == "" {
		opts.PackageName = "reproducer"
	}
	if opts.DetectionProbability == 0 {
		opts.DetectionProbability = 0.99
	}
	files, err := exportFiles(finding, reduction, opts)
	if err != nil {
		return err
	}
	if err := os.Mkdir(dir, 0700); err != nil {
		return fmt.Errorf("export package: %w", err)
	}
	if err := os.Mkdir(filepath.Join(dir, "testdata"), 0700); err != nil {
		return fmt.Errorf("export package: %w", err)
	}
	for name, body := range files {
		f, err := os.OpenFile(filepath.Join(dir, name), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
		if err != nil {
			return fmt.Errorf("write file %s: %w", name, err)
		}
		_, writeErr := f.Write(body)
		closeErr := f.Close()
		if writeErr != nil {
			return fmt.Errorf("write file %s: %w", name, writeErr)
		}
		if closeErr != nil {
			return fmt.Errorf("close file %s: %w", name, closeErr)
		}
	}
	return nil
}

// WriteArtifactPackage writes an artifact directory containing preserved testdata
// and an integrity test. Callers should prefer [ExportPackage], which additionally
// emits a runnable failure reproducer test.
func WriteArtifactPackage(dir string, finding cudaschedule.SeedResult, reduction Reduction, opts ArtifactOptions) error {
	exportOpts := ExportOptions{
		PackageName:          "reproducer",
		Repeats:              opts.Repeats,
		DetectionProbability: opts.DetectionProbability,
		MaxAttempts:          opts.MaxAttempts,
	}
	return ExportPackage(dir, finding, reduction, exportOpts)
}

func exportFiles(finding cudaschedule.SeedResult, reduction Reduction, opts ExportOptions) (map[string][]byte, error) {
	if opts.AttemptTimeout == 0 {
		opts.AttemptTimeout = 30 * time.Second
	}
	if opts.AttemptTimeout < 0 || opts.AttemptTimeout > 10*time.Minute {
		return nil, fmt.Errorf("export package: invalid attempt timeout")
	}
	if opts.RunnerPath != "" || opts.RunnerSHA256 != "" {
		if !filepath.IsAbs(opts.RunnerPath) || len(opts.RunnerSHA256) != 64 {
			return nil, fmt.Errorf("export package: absolute runner and SHA-256 required")
		}
		for _, c := range opts.RunnerSHA256 {
			if c < '0' || c > '9' {
				if c < 'a' || c > 'f' {
					return nil, fmt.Errorf("export package: invalid runner SHA-256")
				}
			}
		}
	}
	if opts.OracleRequired && opts.OraclePatch == "" {
		return nil, fmt.Errorf("export package: required oracle identity missing")
	}

	if reduction.Original.Source == "" || reduction.Original.Source != reduction.Reduced.Source {
		return nil, fmt.Errorf("export package: source changed")
	}
	var raw []byte
	var err error
	if finding.Run != nil {
		raw, err = json.Marshal(finding)
		if err != nil {
			return nil, err
		}
		if _, err := ReadFinding(bytes.NewReader(raw), []byte(reduction.Original.Source)); err != nil {
			return nil, err
		}
	}
	var identity Identity
	if finding.Run != nil {
		identity, err = Identify(*finding.Run)
		if err != nil {
			return nil, err
		}
	} else {
		identity = Identity{Index: 0, Policy: "exact-uint64"}
	}

	measurement := reduction.Baseline
	measuredCase := reduction.Original
	for _, a := range reduction.Attempts {
		if a.Accepted {
			measurement = a.Measurement
			measuredCase = a.Case
		}
	}
	if !reflect.DeepEqual(measuredCase, reduction.Reduced) {
		return nil, fmt.Errorf("export package: reduced case differs from measured case")
	}
	for i := range reduction.Reduced.Grid {
		if reduction.Reduced.Grid[i] == 0 || reduction.Reduced.Block[i] == 0 {
			return nil, fmt.Errorf("export package: zero launch extent")
		}
	}
	for _, extent := range reduction.Reduced.Shape {
		if extent == 0 {
			return nil, fmt.Errorf("export package: zero shape extent")
		}
	}
	if ok, err := AcceptRate(identity, reduction.Baseline, measurement, true); err != nil || !ok {
		return nil, fmt.Errorf("export package: invalid reduced reproduction rate")
	}
	attempts, err := DetectionAttempts(identity, measurement, opts.Repeats, opts.DetectionProbability, opts.MaxAttempts)
	if err != nil {
		return nil, err
	}
	ptx, err := reduction.Reduced.Artifact()
	if err != nil {
		return nil, err
	}
	for _, run := range measurement.Runs {
		if err := checkScheduleRun(reduction.Reduced, run); err != nil {
			return nil, err
		}
	}

	files := map[string][]byte{
		"testdata/source.ptx":    []byte(reduction.Reduced.Source),
		"testdata/candidate.ptx": []byte(ptx),
	}
	if len(raw) > 0 {
		files["testdata/finding.json"] = raw
	}
	caseBody, err := json.Marshal(reduction.Reduced)
	if err != nil {
		return nil, err
	}
	files["testdata/case.json"] = caseBody

	candidateDigest := fmt.Sprintf("%x", sha256.Sum256([]byte(ptx)))
	policy, err := json.Marshal(struct {
		Identity        Identity
		Measurement     Measurement
		Attempts        int
		Options         ExportOptions
		CandidateDigest string
		Model           string
		OracleRequired  bool
		OraclePatch     string
		NativeQualified bool
	}{
		Identity:        identity,
		Measurement:     measurement,
		Attempts:        attempts,
		Options:         opts,
		CandidateDigest: candidateDigest,
		Model:           "iid probability equals empirical rate; not a confidence bound",
		OracleRequired:  opts.OracleRequired,
		OraclePatch:     opts.OraclePatch,
		NativeQualified: false,
	})
	if err != nil {
		return nil, err
	}
	files["testdata/policy.json"] = policy

	digests := make(map[string]string)
	var total int
	for name, b := range files {
		if len(b) > 4<<20 || total > (16<<20)-len(b) {
			return nil, fmt.Errorf("export package: body budget exceeded")
		}
		total += len(b)
		digests[name] = fmt.Sprintf("%x", sha256.Sum256(b))
	}
	manifest, err := json.Marshal(digests)
	if err != nil {
		return nil, err
	}
	files["testdata/digests.json"] = manifest

	goModContent := fmt.Sprintf("module %s\n\ngo 1.22\n", opts.PackageName)
	files["go.mod"] = []byte(goModContent)

	reproducerSource := fmt.Sprintf(`package %s

import (
	"bytes"
 "context"
 "crypto/sha256"
 "encoding/json"
 "fmt"
 "io"
 "os"
 "os/exec"
 "path/filepath"
 "testing"
 "time"
)

type policyData struct {
 Options struct {
 RunnerPath string
 RunnerSHA256 string
 AttemptTimeout int64
 }

	Identity struct {
		Index  uint64
		Policy string
	}
	Measurement struct {
		Reproductions int
	}
	Attempts        int
	CandidateDigest string
	OracleRequired  bool
	OraclePatch     string
	NativeQualified bool
}

func readPolicy(t *testing.T) policyData {
	t.Helper()
	b, err := os.ReadFile("testdata/policy.json")
	if err != nil {
		t.Fatalf("read policy: %%v", err)
	}
	var p policyData
	if err := json.Unmarshal(b, &p); err != nil {
		t.Fatalf("unmarshal policy: %%v", err)
	}
	return p
}

func checkIntegrity(t *testing.T) {
	t.Helper()
	b, err := os.ReadFile("testdata/digests.json")
	if err != nil {
		t.Fatalf("read digests: %%v", err)
	}
	var want map[string]string
	if err := json.Unmarshal(b, &want); err != nil {
		t.Fatalf("unmarshal digests: %%v", err)
	}
	for path, digest := range want {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read artifact %%s: %%v", path, err)
		}
		got := fmt.Sprintf("%%x", sha256.Sum256(data))
		if got != digest {
			t.Fatalf("artifact changed: %%s (got %%s, want %%s)", path, got, digest)
		}
	}
}

// TestArtifacts checks that preserved testdata and kernel files match recorded digests.
func TestArtifacts(t *testing.T) {
	checkIntegrity(t)
}

// TestReproduce executes the reproducer and asserts the defect reproduces.
func TestReproduce(t *testing.T) {
	checkIntegrity(t)
	policy := readPolicy(t)

	kernelPath := "testdata/candidate.ptx"
	if p := os.Getenv("REPRO_KERNEL_PATH"); p != "" {
		kernelPath = p
	}
	kernelBytes, err := readKernel(kernelPath)
	if err != nil {
		t.Fatalf("read kernel %%s: %%v", kernelPath, err)
	}

	caseBytes, err := os.ReadFile("testdata/case.json")
	if err != nil {
		t.Fatalf("read case: %%v", err)
	}

	attempts := policy.Attempts
	if attempts <= 0 {
		attempts = 1
	}

	for n := 1; n <= attempts; n++ {
		reproduced, drift, err := runCandidateAttempt(n, kernelBytes, caseBytes, policy)
		if err != nil {
			t.Fatalf("attempt %%d failed: %%v", n, err)
		}
		if drift != "" {
			t.Fatalf("attempt %%d rejected for identity drift: %%s", n, drift)
		}
		if reproduced {
			t.Fatalf("defect reproduced on attempt %%d/%%d: mismatch at index %%d under policy %%s",
				n, attempts, policy.Identity.Index, policy.Identity.Policy)
		}
	}
}

`, opts.PackageName)

	formattedSource, err := format.Source([]byte(reproducerSource + exportRunnerSource))
	if err != nil {
		return nil, fmt.Errorf("format reproducer test: %w", err)
	}
	files["reproducer_test.go"] = formattedSource

	return files, nil
}
