package tileirdisasm_test

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/tmc/lib/nvidia/x/tileirdisasm"
)

func TestDisassemblyRetainsOutput(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX subprocess fixture")
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "tileirdisasm")
	scratch := t.TempDir()
	for _, fail := range []bool{false, true} {
		body := "#!/bin/sh\ntest \"$1\" = -g || exit 9\ntest -f \"$2\" || exit 8\nprintf 'entry @fixture() { return }\\n'\nprintf 'fixture diagnostic' >&2\n"
		if fail {
			body += "exit 3\n"
		}
		if err := os.WriteFile(path, []byte(body), 0700); err != nil {
			t.Fatal(err)
		}
		m, out, err := tileirdisasm.DisassembleWithOutput(context.Background(), []byte("synthetic bytecode"), tileirdisasm.WithPath(path), tileirdisasm.WithTempDir(scratch))
		if (err != nil) != fail || (!fail && m == nil) {
			t.Fatalf("module=%v err=%v", m, err)
		}
		if out.Path != path || len(out.Args) != 2 || string(out.Stderr) != "fixture diagnostic" || len(out.Stdout) == 0 {
			t.Fatalf("output lost: %+v", out)
		}
		if filepath.Dir(filepath.Dir(out.Args[1])) != scratch {
			t.Fatalf("wrong scratch: %+v", out)
		}
		if _, err := os.Stat(out.Args[1]); !os.IsNotExist(err) {
			t.Fatal("scratch input retained")
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := tileirdisasm.DisassembleWithOutput(ctx, []byte("input"), tileirdisasm.WithTempDir(scratch)); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation: %v", err)
	}
	entries, err := os.ReadDir(scratch)
	if err != nil || len(entries) != 0 {
		t.Fatal("scratch leaked")
	}
}

func ExampleDisassembleWithOutput() {
	_, output, err := tileirdisasm.DisassembleWithOutput(context.Background(), nil)
	fmt.Println(err != nil, len(output.Args))
	// Output: true 0
}

func ExampleWithTempDir() {
	_, _, err := tileirdisasm.DisassembleWithOutput(context.Background(), nil, tileirdisasm.WithTempDir("."))
	fmt.Println(err != nil)
	// Output: true
}
