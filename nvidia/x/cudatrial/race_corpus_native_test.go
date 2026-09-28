//go:build native

package cudatrial

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalock"
)

const cuHadronCommit = "0a6c6d5"

func TestNativeRunExternalDetectorExecControl(t *testing.T) {
	if os.Getenv("CUDATRIAL_NATIVE") != "1" {
		t.Skip("native detector qualification requires CUDATRIAL_NATIVE=1")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatal("CUDATRIAL_NATIVE=1 requires gpu-lock: ", err)
	}
	arch := os.Getenv("CUDATRIAL_NATIVE_ARCH")
	if arch == "" {
		arch = "sm_121a"
	}
	if arch != "sm_121a" {
		t.Fatalf("exec-control qualification requires GB10 target sm_121a, got %q", arch)
	}
	if err := requireGB10(); err != nil {
		t.Fatal(err)
	}
	root := prepareCuHadronBinaries(t, arch)
	t.Setenv("CUHADRON_BINARY_ROOT", root)

	sanitizer := os.Getenv("COMPUTE_SANITIZER")
	if sanitizer == "" {
		sanitizer = "compute-sanitizer"
	}
	sanitizer, err := exec.LookPath(sanitizer)
	if err != nil {
		t.Fatal("compute-sanitizer is required: ", err)
	}
	direct := shellQuote(sanitizer) + " --tool racecheck --error-exitcode 1"
	corpus := NewRaceCorpus()
	directResults, err := corpus.RunExternal(context.Background(), direct, false, arch)
	if err != nil {
		t.Fatalf("direct compute-sanitizer run: %v", err)
	}
	if err := requireDetectedPlantedControl(directResults); err != nil {
		t.Fatal("direct arm: ", err)
	}

	wrapper := filepath.Join(t.TempDir(), "exec-wrapper")
	if err := os.WriteFile(wrapper, []byte("#!/bin/sh\nexec \"$@\"\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	wrapped := shellQuote(sanitizer) + " --tool racecheck --error-exitcode 1 " + shellQuote(wrapper)
	wrappedResults, err := corpus.RunExternal(context.Background(), wrapped, false, arch)
	if !errors.Is(err, ErrDetectorNeverReached) {
		t.Fatalf("exec-wrapper arm error = %v, want ErrDetectorNeverReached", err)
	}
	if err := requireDetectedPlantedControl(wrappedResults); err == nil {
		t.Fatal("exec-wrapper arm unexpectedly detected its planted race")
	}
}

func requireDetectedPlantedControl(results []EvaluationResult) error {
	for _, result := range results {
		if result.TestCase == "cudatrial/planted-race-control" {
			if result.Passed() && result.HasRace {
				return nil
			}
			return fmt.Errorf("planted race control was not detected: %+v", result)
		}
	}
	return fmt.Errorf("planted race control result is missing")
}

func requireGB10() error {
	tool, err := exec.LookPath("nvidia-smi")
	if err != nil {
		return fmt.Errorf("nvidia-smi is required to identify the native device: %w", err)
	}
	out, err := exec.Command(tool, "--query-gpu=name,compute_cap", "--format=csv,noheader").CombinedOutput()
	if err != nil {
		return fmt.Errorf("query GB10 device: %w: %s", err, out)
	}
	if !strings.Contains(strings.ToLower(string(out)), "gb10") || !strings.Contains(string(out), "12.1") {
		return fmt.Errorf("native exec-control gate requires GB10 (compute capability 12.1), got %q", out)
	}
	return nil
}

func prepareCuHadronBinaries(t *testing.T, arch string) string {
	t.Helper()
	root := os.Getenv("CUHADRON_BINARY_ROOT")
	if root != "" {
		data, err := os.ReadFile(filepath.Join(root, "SOURCE-COMMIT"))
		if err != nil || strings.TrimSpace(string(data)) != cuHadronCommit {
			t.Fatalf("CUHADRON_BINARY_ROOT must contain SOURCE-COMMIT=%s", cuHadronCommit)
		}
		return root
	}
	source := os.Getenv("CUHADRON_SOURCE_ROOT")
	if source == "" {
		t.Fatal("set CUHADRON_SOURCE_ROOT to the cuHadron checkout at commit " + cuHadronCommit + " or provide a manifested CUHADRON_BINARY_ROOT")
	}
	commitOut, err := exec.Command("git", "-C", source, "rev-parse", "HEAD").CombinedOutput()
	if err != nil || !strings.HasPrefix(strings.TrimSpace(string(commitOut)), cuHadronCommit) {
		t.Fatalf("cuHadron source must be at %s; got %q (%v)", cuHadronCommit, commitOut, err)
	}

	buildSource := filepath.Join(t.TempDir(), "cuHadron")
	if err := os.MkdirAll(buildSource, 0o755); err != nil {
		t.Fatal(err)
	}
	archive, err := exec.Command("git", "-C", source, "archive", "--format=tar", "HEAD").Output()
	if err != nil {
		t.Fatal("archive cuHadron source: ", err)
	}
	extract := exec.Command("tar", "-xf", "-", "-C", buildSource)
	extract.Stdin = bytes.NewReader(archive)
	if out, err := extract.CombinedOutput(); err != nil {
		t.Fatalf("extract cuHadron source: %v: %s", err, out)
	}

	root = filepath.Join(t.TempDir(), "binaries")
	suffix := binaryArchSuffix(arch)
	for _, fixed := range []string{"0", "1"} {
		clean := exec.Command("make", "-C", buildSource, "clean")
		if out, err := clean.CombinedOutput(); err != nil {
			t.Fatalf("clean cuHadron FIXED=%s: %v: %s", fixed, err, out)
		}
		build := exec.Command("make", "-C", buildSource, "FIXED="+fixed, "ARCH="+arch)
		if out, err := build.CombinedOutput(); err != nil {
			t.Fatalf("build cuHadron FIXED=%s ARCH=%s: %v: %s", fixed, arch, err, out)
		}
		fixedRoot := filepath.Join(root, "FIXED="+fixed)
		if err := filepath.WalkDir(buildSource, func(path string, entry os.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			if entry.IsDir() || !strings.HasSuffix(entry.Name(), "."+suffix+".out") {
				return nil
			}
			relative, err := filepath.Rel(buildSource, path)
			if err != nil {
				return err
			}
			destination := filepath.Join(fixedRoot, relative)
			if err := os.MkdirAll(filepath.Dir(destination), 0o755); err != nil {
				return err
			}
			sourceFile, err := os.Open(path)
			if err != nil {
				return err
			}
			defer sourceFile.Close()
			destinationFile, err := os.Create(destination)
			if err != nil {
				return err
			}
			if _, err := io.Copy(destinationFile, sourceFile); err != nil {
				destinationFile.Close()
				return err
			}
			return destinationFile.Close()
		}); err != nil {
			t.Fatal("copy cuHadron binaries: ", err)
		}
	}
	if err := os.WriteFile(filepath.Join(root, "SOURCE-COMMIT"), []byte(cuHadronCommit+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	return root
}

func shellQuote(s string) string {
	return "'" + strings.ReplaceAll(s, "'", "'\\''") + "'"
}
