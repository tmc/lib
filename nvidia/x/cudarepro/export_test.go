package cudarepro

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaschedule"
)

func sampleExportSetup() (cudaschedule.SeedResult, Reduction, ExportOptions) {
	finding, source := findingRecord()
	c := Case{
		Source: string(source),
		Inputs: []uint64{0, 0, 0},
		Grid:   [3]uint32{1, 1, 1},
		Block:  [3]uint32{64, 1, 1},
	}
	r := Reduction{
		Original: c,
		Reduced:  c,
		Baseline: Measurement{
			Reproductions: 2,
			Runs:          finding.Confirmation.Runs,
		},
	}
	opts := ExportOptions{
		PackageName:          "reproducer",
		Repeats:              2,
		DetectionProbability: 0.99,
		MaxAttempts:          5,
	}
	return finding, r, opts
}

func testTempDir(t *testing.T) string {
	t.Helper()
	dir, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	return dir
}

func runGoTest(dir string, envExtra ...string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "go", "test", "-count=1", "-v")
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GOWORK=off")
	cmd.Env = append(cmd.Env, envExtra...)
	cmd.WaitDelay = time.Second
	return cmd.CombinedOutput()
}

func TestExportExecutedCPUResult(t *testing.T) {
	finding, r, opts := sampleExportSetup()
	setExportRunner(t, &opts)
	dir := filepath.Join(testTempDir(t), "reproducer")
	if err := ExportPackage(dir, finding, r, opts); err != nil {
		t.Fatal(err)
	}
	out, err := runGoTest(dir, "EXPORT_CPU_MODE=")
	if err == nil || !strings.Contains(string(out), "defect reproduced on attempt") {
		t.Fatalf("bug result: %v\n%s", err, out)
	}
	// Different digest, same executed defect: digest changes never establish health.
	changed := filepath.Join(dir, "testdata", "changed.ptx")
	artifact, _ := r.Reduced.Artifact()
	if err := os.WriteFile(changed, []byte(artifact+"\n// different digest"), 0600); err != nil {
		t.Fatal(err)
	}
	out, err = runGoTest(dir, "REPRO_KERNEL_PATH="+changed, "EXPORT_CPU_MODE=")
	if err == nil || !strings.Contains(string(out), "defect reproduced on attempt") {
		t.Fatalf("changed bug result: %v\n%s", err, out)
	}
	fixed := filepath.Join(dir, "testdata", "fixed.ptx")
	if err := os.WriteFile(fixed, []byte(artifact+"\n// FIXED"), 0600); err != nil {
		t.Fatal(err)
	}
	out, err = runGoTest(dir, "REPRO_KERNEL_PATH="+fixed, "EXPORT_CPU_MODE=")
	if err != nil {
		t.Fatalf("fixed executed result: %v\n%s", err, out)
	}
	entries, err := os.ReadDir(filepath.Join(dir, "receipts"))
	if err != nil || len(entries) == 0 {
		t.Fatalf("raw receipts missing: %v", err)
	}
	for _, entry := range entries {
		raw, err := os.ReadFile(filepath.Join(dir, "receipts", entry.Name(), "receipt.json"))
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("executed CPU protocol receipt: %s", raw)
	}
}

func TestExportMissingRunnerRefuses(t *testing.T) {
	finding, r, opts := sampleExportSetup()
	dir := filepath.Join(testTempDir(t), "reproducer")
	if err := ExportPackage(dir, finding, r, opts); err != nil {
		t.Fatal(err)
	}
	out, err := runGoTest(dir, "REPRO_DEVICE_AVAILABLE=1", "REPRO_SIMULATED=1", "REPRO_SIMULATED_REPRODUCE=pass")
	if err == nil || !strings.Contains(string(out), "reviewed runner and executable SHA-256 required") {
		t.Fatalf("missing runner result: %v\n%s", err, out)
	}
}

func TestExportRunnerRefusals(t *testing.T) {
	finding, r, opts := sampleExportSetup()
	setExportRunner(t, &opts)
	opts.OracleRequired = true
	opts.OraclePatch = "declared-oracle"
	opts.AttemptTimeout = 2 * time.Second
	dir := filepath.Join(testTempDir(t), "reproducer")
	if err := ExportPackage(dir, finding, r, opts); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ mode, want string }{
		{"pid", "process identity mismatch"}, {"digest", "artifact identity mismatch"},
		{"oracle", "lacks process-bound evidence"}, {"arrays", "missing matching output/reference arrays"},
		{"drift", "identity drift"}, {"trailing", "trailing data"}, {"overflow", "runner failed"}, {"timeout", "runner failed"},
	} {
		t.Run(tc.mode, func(t *testing.T) {
			out, err := runGoTest(dir, "EXPORT_CPU_MODE="+tc.mode, "REPRO_ORACLE_ACTIVE=1")
			if err == nil || !strings.Contains(string(out), tc.want) {
				t.Fatalf("result: %v\n%s", err, out)
			}
		})
	}
	if err := os.WriteFile(opts.RunnerPath, []byte("changed"), 0700); err != nil {
		t.Fatal(err)
	}
	out, err := runGoTest(dir, "EXPORT_CPU_MODE=")
	if err == nil || !strings.Contains(string(out), "runner executable changed") {
		t.Fatalf("changed runner: %v\n%s", err, out)
	}
}

// TestExport_Refusals checks invalid export inputs and invariants.
func TestExport_Refusals(t *testing.T) {
	for _, name := range []string{"source", "measurement", "shape", "geometry", "rate", "artifact", "runner", "deadline", "oracle"} {
		t.Run(name, func(t *testing.T) {
			finding, r, opts := sampleExportSetup()
			switch name {
			case "source":
				r.Reduced.Source += "different"
			case "measurement":
				r.Reduced.Inputs = []uint64{1}
			case "shape":
				r.Original.Shape = []uint64{0}
				r.Reduced.Shape = []uint64{0}
			case "geometry":
				r.Original.Grid[0] = 0
				r.Reduced.Grid[0] = 0
			case "rate":
				r.Baseline.Reproductions = 1
			case "runner":
				opts.RunnerPath = "relative-runner"
				opts.RunnerSHA256 = strings.Repeat("0", 64)
			case "deadline":
				opts.AttemptTimeout = -time.Second
			case "oracle":
				opts.OracleRequired = true
			case "artifact":
				r.Baseline.Runs[1].PTXDigest = strings.Repeat("0", 64)
			}
			dir := filepath.Join(testTempDir(t), "output")
			if err := ExportPackage(dir, finding, r, opts); err == nil {
				t.Fatalf("invalid export %s was admitted", name)
			}
		})
	}
}

// TestExport_AncestorRefusal tests that symlink ancestors are rejected.
func TestExport_AncestorRefusal(t *testing.T) {
	root := testTempDir(t)
	alias := filepath.Join(root, "alias")
	if err := os.Symlink(root, alias); err != nil {
		t.Fatal(err)
	}
	_, r, opts := sampleExportSetup()
	if err := ExportPackage(filepath.Join(alias, "output"), cudaschedule.SeedResult{}, r, opts); err == nil {
		t.Fatal("symlink ancestor admitted")
	}
}

// TestExport_NonCanonicalDirRefusal tests relative and non-canonical directories are rejected.
func TestExport_NonCanonicalDirRefusal(t *testing.T) {
	_, r, opts := sampleExportSetup()
	if err := ExportPackage("relative/dir", cudaschedule.SeedResult{}, r, opts); err == nil {
		t.Fatal("relative path admitted")
	}
	root := testTempDir(t)
	if err := ExportPackage(root+"/./extra", cudaschedule.SeedResult{}, r, opts); err == nil {
		t.Fatal("non-canonical path admitted")
	}
}

// TestWriteArtifactPackage_Compatibility tests backward compatibility with WriteArtifactPackage.
func TestWriteArtifactPackage_Compatibility(t *testing.T) {
	finding, r, _ := sampleExportSetup()
	dir := filepath.Join(testTempDir(t), "artifact_compat")
	opts := ArtifactOptions{
		Repeats:              2,
		DetectionProbability: 0.99,
		MaxAttempts:          5,
	}
	if err := WriteArtifactPackage(dir, finding, r, opts); err != nil {
		t.Fatal(err)
	}
	out, err := runGoTest(dir, "REPRO_DEVICE_AVAILABLE=1")
	if err == nil {
		t.Fatalf("expected test failure on bug, got:\n%s", out)
	}
}
