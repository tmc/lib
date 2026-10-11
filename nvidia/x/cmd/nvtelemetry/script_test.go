package main

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"rsc.io/script"
	"rsc.io/script/scripttest"
)

func TestScript(t *testing.T) {
	if testing.Short() {
		t.Skip("builds the CLI")
	}
	bin := t.TempDir()
	build := exec.Command("go", "build", "-mod=readonly", "-o", filepath.Join(bin, "nvtelemetry"), ".")
	build.Env = append(os.Environ(), "GOWORK=off")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build nvtelemetry: %v\n%s", err, out)
	}
	engine := &script.Engine{Cmds: scripttest.DefaultCmds(), Conds: scripttest.DefaultConds()}
	env := []string{"PATH=" + bin + string(os.PathListSeparator) + os.Getenv("PATH"), "GOWORK=off"}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	t.Cleanup(cancel)
	scripttest.Test(t, ctx, engine, env, "testdata/script/*.txt")
}
