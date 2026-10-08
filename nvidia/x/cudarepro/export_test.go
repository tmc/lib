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

// TestExport_PackageBuildsAndFailsOnBug tests that the emitted Go test package
// compiles cleanly and fails on the recorded buggy kernel defect when run under
// simulated/native device execution.
func TestExport_PackageBuildsAndFailsOnBug(t *testing.T) {
	finding, r, opts := sampleExportSetup()
	dir := filepath.Join(testTempDir(t), "reproducer_bug")
	if err := ExportPackage(dir, finding, r, opts); err != nil {
		t.Fatal(err)
	}

	// Run with device available: should fail because the bug reproduced.
	out, err := runGoTest(dir, "REPRO_DEVICE_AVAILABLE=1")
	if err == nil {
		t.Fatalf("expected test failure on buggy kernel, but passed:\n%s", out)
	}
	if !strings.Contains(string(out), "defect reproduced on attempt") {
		t.Fatalf("output does not report reproduced defect:\n%s", out)
	}
}

// TestExport_PackagePassesOnFixedKernel tests that the emitted test package
// passes cleanly when pointed at a fixed kernel that does not exhibit the defect.
func TestExport_PackagePassesOnFixedKernel(t *testing.T) {
	finding, r, opts := sampleExportSetup()
	dir := filepath.Join(testTempDir(t), "reproducer_fixed")
	if err := ExportPackage(dir, finding, r, opts); err != nil {
		t.Fatal(err)
	}

	// Write a fixed kernel variant in testdata.
	fixedPath := filepath.Join(dir, "testdata/fixed.ptx")
	fixedContent := "// FIXED: synchronization bug resolved\n.version 8.0\n.target sm_80\n.address_size 64\n.visible .entry fixture() { ret; }\n"
	if err := os.WriteFile(fixedPath, []byte(fixedContent), 0600); err != nil {
		t.Fatal(err)
	}

	// Run test pointing to fixed kernel: should pass cleanly!
	out, err := runGoTest(dir, "REPRO_DEVICE_AVAILABLE=1", "REPRO_KERNEL_PATH="+fixedPath)
	if err != nil {
		t.Fatalf("expected test to pass on fixed kernel, but failed: %v\nOutput:\n%s", err, out)
	}
	if !strings.Contains(string(out), "PASS") {
		t.Fatalf("output does not contain PASS:\n%s", out)
	}
}

// TestExport_PendingWithoutDevice asserts that without a device or native environment,
// the exported test returns the shared pending sentinel and NEVER reports a pass.
func TestExport_PendingWithoutDevice(t *testing.T) {
	finding, r, opts := sampleExportSetup()
	dir := filepath.Join(testTempDir(t), "reproducer_pending")
	if err := ExportPackage(dir, finding, r, opts); err != nil {
		t.Fatal(err)
	}

	// Filter out device-enabling environment variables.
	cleanEnv := []string{
		"CUDA_VISIBLE_DEVICES=",
		"REPRO_DEVICE_AVAILABLE=",
		"GPU_LOCK_HELD=",
		"REPRO_SIMULATED=",
	}
	out, err := runGoTest(dir, cleanEnv...)
	if err != nil {
		t.Fatalf("expected exit 0 for skip, got: %v\nOutput:\n%s", err, out)
	}
	outStr := string(out)
	if !strings.Contains(outStr, "native qualification pending:") {
		t.Fatalf("expected pending sentinel in output, got:\n%s", outStr)
	}
	if !strings.Contains(outStr, "SKIP: TestReproduce") {
		t.Fatalf("expected TestReproduce to SKIP, got:\n%s", outStr)
	}
	// Verify that TestReproduce was NOT passed.
	if strings.Contains(outStr, "PASS: TestReproduce") {
		t.Fatalf("TestReproduce reported pass without a device; must report pending sentinel:\n%s", outStr)
	}
}

// TestExport_OracleDroppedRefusal asserts that when an oracle is required (e.g. lane 52 patch),
// running without the oracle active fails and refuses execution.
func TestExport_OracleDroppedRefusal(t *testing.T) {
	finding, r, opts := sampleExportSetup()
	opts.OracleRequired = true
	opts.OraclePatch = "lane52-memcheck"
	dir := filepath.Join(testTempDir(t), "reproducer_oracle")
	if err := ExportPackage(dir, finding, r, opts); err != nil {
		t.Fatal(err)
	}

	// Run with device available but oracle inactive.
	out, err := runGoTest(dir, "REPRO_DEVICE_AVAILABLE=1", "REPRO_ORACLE_ACTIVE=", "CUDASANITIZER_ACTIVE=")
	if err == nil {
		t.Fatalf("expected test failure when required oracle is inactive, but passed:\n%s", out)
	}
	if !strings.Contains(string(out), "required oracle \"lane52-memcheck\" is not active") {
		t.Fatalf("expected oracle inactive error message, got:\n%s", out)
	}

	// Now run with oracle active: should proceed to defect detection.
	outActive, errActive := runGoTest(dir, "REPRO_DEVICE_AVAILABLE=1", "REPRO_ORACLE_ACTIVE=1")
	if errActive == nil {
		t.Fatalf("expected failure on bug with active oracle, but passed:\n%s", outActive)
	}
	if !strings.Contains(string(outActive), "defect reproduced on attempt") {
		t.Fatalf("expected defect reproduced output with active oracle, got:\n%s", outActive)
	}
}

// TestExport_RejectsIdentityDrift asserts that if execution experiences identity drift,
// the test fails and never passes.
func TestExport_RejectsIdentityDrift(t *testing.T) {
	finding, r, opts := sampleExportSetup()
	dir := filepath.Join(testTempDir(t), "reproducer_drift")
	if err := ExportPackage(dir, finding, r, opts); err != nil {
		t.Fatal(err)
	}

	out, err := runGoTest(dir, "REPRO_DEVICE_AVAILABLE=1", "REPRO_SIMULATED_REPRODUCE=drift")
	if err == nil {
		t.Fatalf("expected test failure on identity drift, but passed:\n%s", out)
	}
	if !strings.Contains(string(out), "rejected for identity drift") {
		t.Fatalf("expected drift message, got:\n%s", out)
	}
}

// TestExport_Refusals checks invalid export inputs and invariants.
func TestExport_Refusals(t *testing.T) {
	for _, name := range []string{"source", "measurement", "shape", "geometry", "rate", "artifact"} {
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
