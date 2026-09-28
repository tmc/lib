package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"rsc.io/script"
	"rsc.io/script/scripttest"
)

func TestArguments(t *testing.T) {
	for _, args := range [][]string{{"-workflow=unknown"}, {"-smoke=unknown"}, {"-timeout=0"}, {"extra"}, {"-workflow=aot", "-smoke=run"}} {
		var out, stderr bytes.Buffer
		if code := run(args, &out, &stderr); code != 2 {
			t.Errorf("%q: exit %d, want 2", args, code)
		}
	}
}

func TestReport(t *testing.T) {
	r := report{Schema: 1, Workflow: "jit", Environment: map[string]string{}, Checks: []check{{Name: "nvrtc", Status: "fail", Remedy: "set NVRTC_LIB_PATH"}}}
	var out bytes.Buffer
	if err := writeReport(&out, r, true); err != nil {
		t.Fatal(err)
	}
	var got report
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != 1 || len(got.Checks) != 1 || got.Checks[0].Status != "fail" {
		t.Fatalf("report: %+v", got)
	}
	out.Reset()
	if err := writeReport(&out, r, false); err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(out.Bytes(), []byte("set NVRTC_LIB_PATH")) {
		t.Fatal("missing remedy")
	}
}

func TestScript(t *testing.T) {
	bin := t.TempDir()
	cmd := exec.Command("go", "build", "-o", filepath.Join(bin, "cudadoctor"), ".")
	cmd.Env = append(os.Environ(), "GOWORK=off")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, out)
	}
	engine := &script.Engine{Cmds: scripttest.DefaultCmds(), Conds: scripttest.DefaultConds()}
	env := []string{"PATH=" + bin + string(os.PathListSeparator) + os.Getenv("PATH"), "HOME=" + os.Getenv("HOME"), "TMPDIR=" + os.Getenv("TMPDIR")}
	scripttest.Test(t, context.Background(), engine, env, "testdata/*.txt")
}

func TestToolSelection(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("cudatool searches toolkit roots only on linux")
	}
	dir := t.TempDir()
	bin := filepath.Join(dir, "bin")
	if err := os.Mkdir(bin, 0700); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(bin, "nvcc")
	t.Setenv("CUDA_HOME", dir)
	for _, tt := range []struct{ name, body, want string }{
		{"version", "echo 'Cuda compilation tools, release 13.4, V13.4.59'", "pass"},
		{"broken", "echo broken >&2; exit 9", "fail"},
		{"timeout", "exec sleep 10", "fail"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if err := os.WriteFile(path, []byte("#!/bin/sh\n"+tt.body+"\n"), 0700); err != nil {
				t.Fatal(err)
			}
			c := toolCheck("nvcc", 100*time.Millisecond)
			if c.Status != tt.want || c.Path != path {
				t.Fatalf("check = %+v", c)
			}
			if tt.name == "version" {
				if c.Version != "13.4" {
					t.Fatalf("version = %q", c.Version)
				}
				r := report{Workflow: "jit", Checks: []check{{Name: "nvrtc", Status: "pass", Version: "13.0"}}}
				addVersionWarning(&r, time.Second)
				if r.Checks[len(r.Checks)-1].Status != "warn" {
					t.Fatalf("missing mismatch warning: %+v", r.Checks)
				}
			}
		})
	}
}

func TestBindProbeMissingLibrary(t *testing.T) {
	t.Setenv("CUDABIND_LIBCLANG", filepath.Join(t.TempDir(), "missing-libclang.so"))
	var out, errout bytes.Buffer
	if code := probeMain("libclang", "compute_75", &out, &errout); code != 0 {
		t.Fatalf("probe exit %d: %s", code, &errout)
	}
	var c check
	if err := json.Unmarshal(out.Bytes(), &c); err != nil {
		t.Fatal(err)
	}
	if c.Status != "fail" || !strings.Contains(c.Remedy, "CUDABIND_LIBCLANG") {
		t.Fatalf("check = %+v", c)
	}
}
