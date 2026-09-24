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

// TestScript runs trainstat as a program on files and compares its output
// as text. The scripts also build benchstat, to check that trainstat's
// benchfmt output is benchstat's input, and run go test in a scratch module
// that uses the trainstat package from this module.
func TestScript(t *testing.T) {
	if testing.Short() {
		t.Skip("builds binaries")
	}
	bin := t.TempDir()
	for _, pkg := range []string{".", "golang.org/x/perf/cmd/benchstat"} {
		build := exec.Command("go", "build", "-o", bin, pkg)
		build.Env = append(os.Environ(), "GOWORK=off")
		if out, err := build.CombinedOutput(); err != nil {
			t.Fatalf("build %s: %v\n%s", pkg, err, out)
		}
	}
	fixtures, err := filepath.Abs("testdata")
	if err != nil {
		t.Fatal(err)
	}
	root, err := filepath.Abs("../../..")
	if err != nil {
		t.Fatal(err)
	}
	engine := &script.Engine{
		Cmds:  scripttest.DefaultCmds(),
		Conds: scripttest.DefaultConds(),
	}
	// HOME is the real one: the gotest script runs go test, which needs
	// the module and build caches under it.
	env := []string{
		"PATH=" + bin + string(os.PathListSeparator) + os.Getenv("PATH"),
		"HOME=" + os.Getenv("HOME"),
		"GOWORK=off",
		"TESTDATA=" + fixtures,
		"NVIDIA=" + root,
	}
	scripttest.Test(t, context.Background(), engine, env, "testdata/script/*.txt")
}
