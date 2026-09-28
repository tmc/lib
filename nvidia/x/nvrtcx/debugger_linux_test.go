//go:build linux

package nvrtcx_test

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// TestDeviceDebugger requires an exclusive GPU lane and the actual cuda-gdb.
// It tests device debugging of a Go-launched kernel, not Go host debugging.
func TestDeviceDebugger(t *testing.T) {
	if os.Getenv("NVRTCX_DEBUGGER_TEST") != "1" {
		t.Skip("NVRTCX_DEBUGGER_TEST=1 requires an exclusive GPU and cuda-gdb")
	}
	debugger, err := exec.LookPath("cuda-gdb")
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	binary := filepath.Join(dir, "probe")
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	build := exec.CommandContext(ctx, "go", "build", "-o", binary, "./testdata/debugger")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, out)
	}
	source, err := filepath.Abs("testdata/debugger/kernel.cu")
	if err != nil {
		t.Fatal(err)
	}
	script := filepath.Join(dir, "commands.gdb")
	commands := "set pagination off\nset breakpoint pending on\nbreak " + strconv.Quote(source) + ":3\nrun\ninfo cuda threads\nprint value\nprint index\ncontinue\n"
	if err := os.WriteFile(script, []byte(commands), 0600); err != nil {
		t.Fatal(err)
	}
	cmd := exec.CommandContext(ctx, debugger, "--batch", "-x", script, "--args", binary, source)
	out, err := cmd.CombinedOutput()
	t.Logf("%s", out)
	if err != nil {
		t.Fatalf("cuda-gdb: %v", err)
	}
	for _, want := range []string{"CUDA thread hit Breakpoint", "kernel.cu:3", "ThreadIdx", "$1 = 37", "$2 = 0", "debug output: [37 38 39 40]", "exited normally"} {
		if !strings.Contains(string(out), want) {
			t.Errorf("cuda-gdb output missing %q", want)
		}
	}
}
