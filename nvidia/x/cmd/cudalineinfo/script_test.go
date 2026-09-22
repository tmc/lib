package main

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"rsc.io/script"
	"rsc.io/script/scripttest"
)

// TestScript runs the command the way a reader does: as a program, on files,
// with its output compared as text. The fixtures are the ones the package's
// own tests use, so a script and a Go test disagreeing about the same
// artifacts is a real disagreement.
//
// Every script runs offline. Compiling needs NVRTC and a GPU, which the
// scripts that ask for it guard with a condition.
func TestScript(t *testing.T) {
	bin := t.TempDir()
	build := exec.Command("go", "build", "-o", filepath.Join(bin, "cudalineinfo"), ".")
	build.Env = append(os.Environ(), "GOWORK=off")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build cudalineinfo: %v\n%s", err, out)
	}
	fixtures, err := filepath.Abs(testdata)
	if err != nil {
		t.Fatal(err)
	}
	engine := &script.Engine{
		Cmds:  scripttest.DefaultCmds(),
		Conds: scripttest.DefaultConds(),
	}
	pkg, err := filepath.Abs(filepath.Join("..", "..", "examples", "source-wiring"))
	if err != nil {
		t.Fatal(err)
	}
	// HOME is the real one: a script that runs a Go tool needs the module
	// and build caches under it, and no script writes there.
	env := []string{
		"PATH=" + bin + string(os.PathListSeparator) + os.Getenv("PATH"),
		"HOME=" + os.Getenv("HOME"),
		"GOWORK=off",
		"TESTDATA=" + fixtures,
		"PKGDIR=" + pkg,
	}
	scripttest.Test(t, context.Background(), engine, env, "testdata/script/*.txt")
}
