package cuptibench

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
)

// The normal package test process owns no process profile. Child processes
// exercise exactly the TestMain lifecycle documented for users.
func TestMain(m *testing.M) {
	enabled := os.Getenv("CUPTIBENCH_EXPORT_TEST") == "1"
	if enabled {
		Init()
	}
	code := m.Run()
	if enabled {
		if err := Finish(); err != nil {
			fmt.Fprintln(os.Stderr, err)
			code = 1
		}
	}
	os.Exit(code)
}

func BenchmarkCuptiExportLifecycle(b *testing.B) {
	if os.Getenv("CUPTIBENCH_EXPORT_TEST") != "1" {
		b.Skip("synthetic export control")
	}
	for _, name := range []string{"full", "isolated/kernel"} {
		b.Run(name, func(b *testing.B) {
			p, w, _, _, _ := benchmarkFixture()
			p.additive = []bool{os.Getenv("CUPTIBENCH_EXPORT_RATIO") != "1"}
			r, err := benchmark(b, p, w, func(*Plan, Workload) (benchmarkTimer, error) { return &syntheticTimer{}, nil })
			if err != nil {
				b.Fatal(err)
			}
			if err := p.Close(); err != nil {
				b.Fatal(err)
			}
			if err := recordBenchmarkResult(r, p); err != nil {
				b.Fatal(err)
			}
		})
	}
}

func TestExportTestMainLifecycle(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	pkg, err := packagePath()
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, pattern string
		results       int
	}{
		{"all", "^BenchmarkCuptiExportLifecycle$", 4},
		{"isolated", "^BenchmarkCuptiExportLifecycle$/^isolated$/^kernel$", 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			template := filepath.Join(exportDir(t), "{package}.pprof")
			path, err := expandProfilePath(template, pkg)
			if err != nil {
				t.Fatal(err)
			}
			run := func(extra ...string) ([]byte, error) {
				cmd := exec.Command(exe, "-test.run=^$", "-test.bench="+tc.pattern, "-test.benchtime=2x", "-test.count=2", "-cuptiprofile="+template)
				cmd.Env = append(os.Environ(), "CUPTIBENCH_EXPORT_TEST=1")
				cmd.Env = append(cmd.Env, extra...)
				return cmd.CombinedOutput()
			}
			out, err := run()
			if err != nil {
				t.Fatalf("TestMain runner: %v\n%s", err, out)
			}
			data, err := os.ReadFile(path + ".json")
			if err != nil {
				t.Fatal(err)
			}
			var manifest exportManifest
			if err := json.Unmarshal(data, &manifest); err != nil {
				t.Fatal(err)
			}
			if len(manifest.Results) != tc.results {
				t.Fatalf("results=%d want %d", len(manifest.Results), tc.results)
			}
			ids := map[string]bool{}
			for _, r := range manifest.Results {
				if ids[r.Metadata["run_id"]] {
					t.Fatal("repeated run identity")
				}
				ids[r.Metadata["run_id"]] = true
				if r.Operations != 3 || r.Timing.Operations != 6 {
					t.Fatalf("logical operation counts: %+v", r)
				}
				if tc.name == "isolated" && !strings.HasSuffix(r.Metadata["benchmark"], "/isolated/kernel") {
					t.Fatal("filtered predecessor executed")
				}
			}
			profData, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			prof, err := profile.ParseData(profData)
			if err != nil {
				t.Fatal(err)
			}
			if len(prof.Sample) != tc.results {
				t.Fatalf("samples=%d want %d", len(prof.Sample), tc.results)
			}
			for _, s := range prof.Sample {
				if s.Value[0] != 42 {
					t.Fatal("profile counted replay or divided by benchmark iterations")
				}
			}
			out, err = run()
			if err == nil || !strings.Contains(string(out), "already exists") {
				t.Fatalf("collision did not fail Finish: %v\n%s", err, out)
			}
			after, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(after, profData) {
				t.Fatal("collision changed prior profile")
			}
		})
	}
}

func TestExportTestMainRejectsRatioOnlyProfile(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(exportDir(t), "ratio.pprof")
	cmd := exec.Command(exe, "-test.run=^$", "-test.bench=^BenchmarkCuptiExportLifecycle$/^full$", "-test.benchtime=1x", "-cuptiprofile="+path)
	cmd.Env = append(os.Environ(), "CUPTIBENCH_EXPORT_TEST=1", "CUPTIBENCH_EXPORT_RATIO=1")
	out, err := cmd.CombinedOutput()
	if err == nil || !strings.Contains(string(out), "no verified additive metrics") {
		t.Fatalf("ratio export should fail Finish: %v\n%s", err, out)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("inadmissible profile published: %v", err)
	}
}

type reviewCleanupCollector struct {
	*benchmarkCollector
	fail   bool
	closes int
}

func (c *reviewCleanupCollector) Close() error {
	c.closes++
	if c.fail {
		return fmt.Errorf("injected override cleanup failure")
	}
	return nil
}

func TestOverrideCleanupRetainsChildUntilRetry(t *testing.T) {
	owner, _, _, _, _ := benchmarkFixture()
	child, _, native, _, _ := benchmarkFixture()
	backend := &reviewCleanupCollector{benchmarkCollector: native, fail: true}
	child.backend = backend
	// benchmarkPlan holds the caller plan until the temporary owner is closed.
	if err := owner.begin(); err != nil {
		t.Fatal(err)
	}
	if err := closeOverride(owner, child); err == nil {
		t.Fatal("accepted cleanup failure")
	}
	if owner.pendingOverride != child || child.backend != backend || child.context == nil {
		t.Fatal("lost temporary plan resources")
	}
	owner.end()
	if err := owner.begin(); err == nil {
		owner.end()
		t.Fatal("new capture admitted with failed override cleanup")
	}
	if err := owner.Close(); err == nil {
		t.Fatal("accepted repeat child failure")
	}
	if owner.backend == nil || owner.context == nil || owner.pendingOverride != child {
		t.Fatal("released caller resources before child cleanup")
	}
	backend.fail = false
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
	if backend.closes != 3 || owner.pendingOverride != nil || !child.closed || !owner.closed {
		t.Fatalf("cleanup not retained/retried: closes=%d child=%+v owner=%+v", backend.closes, child, owner)
	}
}

func BenchmarkCuptiExportCleanupFailure(b *testing.B) {
	if os.Getenv("CUPTIBENCH_EXPORT_TEST") != "1" {
		b.Skip("synthetic cleanup control")
	}
	p, w, _, _, _ := benchmarkFixture()
	p.additive = []bool{true}
	timer := &syntheticTimer{fail: true}
	r, err := benchmark(b, p, w, func(*Plan, Workload) (benchmarkTimer, error) { return timer, nil })
	if err != nil {
		if r != nil || p.pendingTimer != timer {
			b.Fatal("cleanup failure did not invalidate result and retain timer")
		}
		timer.fail = false
		if cleanup := p.Close(); cleanup != nil {
			b.Fatal(cleanup)
		}
		b.Fatal(err)
	}
	if err := recordBenchmarkResult(r, nil); err != nil {
		b.Fatal(err)
	}
}

func TestCleanupFailureDoesNotPublishProfile(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(exportDir(t), "cleanup.pprof")
	cmd := exec.Command(exe, "-test.run=^$", "-test.bench=^BenchmarkCuptiExportCleanupFailure$", "-test.benchtime=1x", "-cuptiprofile="+path)
	cmd.Env = append(os.Environ(), "CUPTIBENCH_EXPORT_TEST=1")
	out, err := cmd.CombinedOutput()
	if err == nil || !strings.Contains(string(out), "timer cleanup") {
		t.Fatalf("expected cleanup failure: %v\n%s", err, out)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("cleanup failure published profile: %v", err)
	}
	if _, err := os.Stat(path + ".json"); !os.IsNotExist(err) {
		t.Fatalf("cleanup failure published manifest: %v", err)
	}
}

func BenchmarkCuptiExportOverrideFailure(b *testing.B) {
	if os.Getenv("CUPTIBENCH_EXPORT_TEST") != "1" {
		b.Skip("synthetic override cleanup control")
	}
	owner, _, _, _, _ := benchmarkFixture()
	child, w, native, _, _ := benchmarkFixture()
	child.additive = []bool{true}
	backend := &reviewCleanupCollector{benchmarkCollector: native, fail: true}
	child.backend = backend
	if err := owner.begin(); err != nil {
		b.Fatal(err)
	}
	r, err := benchmark(b, child, w, func(*Plan, Workload) (benchmarkTimer, error) { return &syntheticTimer{}, nil })
	if err != nil {
		owner.end()
		b.Fatal(err)
	}
	// benchmark has called ReportMetric. A subsequent owner cleanup failure must
	// still cause the Go runner to suppress that failed benchmark's result row.
	err = closeOverride(owner, child)
	owner.end()
	if err != nil {
		if owner.pendingOverride != child {
			b.Fatal("lost failed override")
		}
		backend.fail = false
		if cleanup := owner.Close(); cleanup != nil {
			b.Fatal(cleanup)
		}
		b.Fatal(err)
	}
	if err := recordBenchmarkResult(r, nil); err != nil {
		b.Fatal(err)
	}
}

func TestOverrideFailureAfterReportSuppressesOutput(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(exportDir(t), "override.pprof")
	cmd := exec.Command(exe, "-test.run=^$", "-test.bench=^BenchmarkCuptiExportOverrideFailure$", "-test.benchtime=1x", "-cuptiprofile="+path)
	cmd.Env = append(os.Environ(), "CUPTIBENCH_EXPORT_TEST=1")
	out, err := cmd.CombinedOutput()
	if err == nil || !strings.Contains(string(out), "injected override cleanup failure") {
		t.Fatalf("expected override cleanup failure: %v\n%s", err, out)
	}
	if strings.Contains(string(out), "gpu-ns/op") {
		t.Fatalf("failed benchmark printed metric row:\n%s", out)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("failed override published profile: %v", err)
	}
	if _, err := os.Stat(path + ".json"); !os.IsNotExist(err) {
		t.Fatalf("failed override published manifest: %v", err)
	}
}

func BenchmarkCuptiOwnerLifecycle(b *testing.B) {
	if os.Getenv("CUPTIBENCH_EXPORT_TEST") != "1" {
		b.Skip("synthetic caller ownership control")
	}
	mode := os.Getenv("CUPTIBENCH_OWNER_CLEANUP")
	p, w, native, _, _ := benchmarkFixture()
	backend := &reviewCleanupCollector{benchmarkCollector: native, fail: mode == "fail" || mode == "retry"}
	p.backend = backend
	p.additive = []bool{true}
	if mode != "open" {
		b.Cleanup(func() {
			if err := p.Close(); err != nil {
				b.Error(err)
				if mode == "retry" {
					backend.fail = false
					if err := p.Close(); err != nil {
						b.Error(err)
					}
				}
			}
		})
	}
	r, err := benchmark(b, p, w, func(*Plan, Workload) (benchmarkTimer, error) { return &syntheticTimer{}, nil })
	if err != nil {
		b.Fatal(err)
	}
	if err := recordBenchmarkResult(r, p); err != nil {
		b.Fatal(err)
	}
}

func TestExportRequiresSuccessfulCallerCleanup(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	for _, mode := range []string{"healthy", "fail", "retry", "open"} {
		t.Run(mode, func(t *testing.T) {
			path := filepath.Join(exportDir(t), "owner.pprof")
			cmd := exec.Command(exe, "-test.run=^$", "-test.bench=^BenchmarkCuptiOwnerLifecycle$", "-test.benchtime=1x", "-cuptiprofile="+path)
			cmd.Env = append(os.Environ(), "CUPTIBENCH_EXPORT_TEST=1", "CUPTIBENCH_OWNER_CLEANUP="+mode)
			out, err := cmd.CombinedOutput()
			if mode == "healthy" {
				if err != nil {
					t.Fatalf("healthy ownership: %v\n%s", err, out)
				}
				data, err := os.ReadFile(path)
				if err != nil {
					t.Fatal(err)
				}
				p, err := profile.ParseData(data)
				if err != nil || len(p.Sample) != 1 {
					t.Fatalf("healthy export: %v %+v", err, p)
				}
				return
			}
			diagnostic := "profile owner cleanup failed"
			if mode == "open" {
				diagnostic = "contributing plan is not closed"
			}
			if err == nil || !strings.Contains(string(out), diagnostic) {
				t.Fatalf("mode=%s error=%v output:\n%s", mode, err, out)
			}
			for _, p := range []string{path, path + ".json"} {
				if _, err := os.Stat(p); !os.IsNotExist(err) {
					t.Fatalf("unqualified owner published %s: %v", p, err)
				}
			}
		})
	}
}
