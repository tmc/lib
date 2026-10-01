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

// TestLanguageStackScript exercises the CLI without injection or GPU calls.
func TestLanguageStackScript(t *testing.T) {
	if testing.Short() {
		t.Skip("builds the CLI")
	}
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 is unavailable")
	}
	bin := t.TempDir()
	build := exec.Command("go", "build", "-o", filepath.Join(bin, "cudaprof"), ".")
	build.Env = append(os.Environ(), "GOWORK=off")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build cudaprof: %v\n%s", err, out)
	}
	engine := &script.Engine{
		Cmds:  scripttest.DefaultCmds(),
		Conds: scripttest.DefaultConds(),
	}
	env := []string{
		"PATH=" + bin + string(os.PathListSeparator) + os.Getenv("PATH"),
		"HOME=" + os.Getenv("HOME"),
		"PYTHONDONTWRITEBYTECODE=1",
		"GOWORK=off",
	}
	scripttest.Test(t, context.Background(), engine, env, "testdata/script/*.txt")
}
