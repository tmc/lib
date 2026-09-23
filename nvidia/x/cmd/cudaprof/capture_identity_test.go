package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// buildHelper builds a Go program in a fresh git repository and returns the
// binary, its directory, and the commit it was built from.
func buildHelper(t *testing.T) (bin, dir, rev string) {
	t.Helper()
	dir = t.TempDir()
	write := func(name, text string) {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(text), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module example.com/capturehelper\n\ngo 1.22\n")
	write("main.go", "package main\n\nfunc main() {}\n")
	write(".gitignore", "capturehelper\n")
	run := func(name string, args ...string) string {
		t.Helper()
		cmd := exec.Command(name, args...)
		cmd.Dir = dir
		cmd.Env = append(os.Environ(), "GOWORK=off", "GOFLAGS=-buildvcs=true",
			"GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@example.com", "GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@example.com")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("%s %s: %v\n%s", name, strings.Join(args, " "), err, out)
		}
		return strings.TrimSpace(string(out))
	}
	run("git", "init", "-q")
	run("git", "add", ".")
	run("git", "-c", "commit.gpgsign=false", "commit", "-q", "-m", "helper")
	rev = run("git", "rev-parse", "HEAD")
	run("go", "build", "-o", "capturehelper", ".")
	bin, err := filepath.EvalSymlinks(filepath.Join(dir, "capturehelper"))
	if err != nil {
		t.Fatal(err)
	}
	return bin, dir, rev
}

// buildEmptyLibrary builds a shared library the dynamic loader accepts and
// that does nothing, to stand in for the agent.
func buildEmptyLibrary(t *testing.T) string {
	t.Helper()
	cc, err := exec.LookPath("cc")
	if err != nil {
		t.Skip("no C compiler")
	}
	dir := t.TempDir()
	src := filepath.Join(dir, "empty.c")
	if err := os.WriteFile(src, []byte("int cudaprof_test_agent;\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	lib := filepath.Join(dir, "agent.so")
	if out, err := exec.Command(cc, "-shared", "-fPIC", "-o", lib, src).CombinedOutput(); err != nil {
		t.Fatalf("cc: %v\n%s", err, out)
	}
	lib, err = filepath.EvalSymlinks(lib)
	if err != nil {
		t.Fatal(err)
	}
	return lib
}

func sha256Of(t *testing.T, path string) string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func TestCaptureRecordsTargetIdentity(t *testing.T) {
	bin, dir, rev := buildHelper(t)
	agent := buildEmptyLibrary(t)
	pathEnv := "PATH=" + os.Getenv("PATH")
	tests := []struct {
		name    string
		command []string
		env     []string
		wrapped bool
	}{
		{name: "direct", command: []string{bin}, env: []string{pathEnv}},
		{name: "relative", command: []string{"./capturehelper"}, env: []string{pathEnv}},
		{name: "env assignments", command: []string{"env", "A=1", "B=2", bin, "-x"}, env: []string{pathEnv}, wrapped: true},
		{name: "env PATH lookup", command: []string{"env", "A=1", "capturehelper"}, env: []string{"PATH=" + dir + string(filepath.ListSeparator) + os.Getenv("PATH")}, wrapped: true},
		{name: "env sets PATH", command: []string{"env", "-u", "X", "PATH=" + dir, "capturehelper"}, env: []string{pathEnv}, wrapped: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// The helper preloads an empty library: it runs no CUDA, but
			// the capture still names the agent it injected.
			result, err := runCapture(context.Background(), captureOptions{command: tt.command, Dir: dir, Env: tt.env, Inject: agent, DrainTimeout: 1})
			if err != nil {
				t.Fatal(err)
			}
			if result.TargetExitCode != 0 {
				t.Fatalf("target exit %d: %s", result.TargetExitCode, result.CaptureError)
			}
			exe := result.Target.Executable
			if exe == nil || exe.Error != "" {
				t.Fatalf("executable = %+v", exe)
			}
			if exe.Path != bin || exe.SHA256 != sha256Of(t, bin) {
				t.Errorf("executable %s %s, want %s %s", exe.Path, exe.SHA256, bin, sha256Of(t, bin))
			}
			if fi, err := os.Stat(bin); err != nil || exe.Size != fi.Size() {
				t.Errorf("size = %d, want %d", exe.Size, fi.Size())
			}
			if (exe.Wrapper != "") != tt.wrapped || tt.wrapped && filepath.Base(exe.Wrapper) != "env" {
				t.Errorf("wrapper = %q, wrapped %v", exe.Wrapper, tt.wrapped)
			}
			g := exe.Go
			if g == nil {
				t.Fatal("no Go build information")
			}
			if g.MainModule != "example.com/capturehelper" || g.VCSRevision != rev || g.VCSModified != "false" || g.VCSTime == "" {
				t.Errorf("go build = %+v, want module example.com/capturehelper revision %s unmodified", g, rev)
			}
			a := result.Target.Agent
			if a == nil || a.Path != agent || a.SHA256 != sha256Of(t, agent) || a.Size == 0 || a.Go != nil {
				t.Errorf("agent = %+v", a)
			}
		})
	}
}

func TestIdentifyTargetRefusesToGuess(t *testing.T) {
	tests := []struct {
		name    string
		command []string
		want    string
	}{
		{"unknown env option", []string{"env", "-S", "prog"}, "not understood"},
		{"env without program", []string{"env", "A=1"}, "names no program"},
		{"missing program", []string{"env", "no-such-program-here"}, "not found"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			id := identifyTarget("/usr/bin/env", tt.command, []string{"PATH=" + t.TempDir()}, "")
			if !strings.Contains(id.Error, tt.want) || id.SHA256 != "" || id.Size != 0 {
				t.Errorf("identity = %+v, want an error containing %q and no digest", id, tt.want)
			}
		})
	}
	// A non-Go file is identified by its bytes and carries no Go block.
	path := filepath.Join(t.TempDir(), "script")
	if err := os.WriteFile(path, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	if id := identifyExecutable(path); id.Error != "" || id.Go != nil || id.Size != 10 {
		t.Errorf("identity = %+v", id)
	}
}
