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

func TestScript(t *testing.T) {
	bin := t.TempDir()
	build := exec.Command("go", "build", "-o", filepath.Join(bin, "pairedtiming"), ".")
	build.Env = append(os.Environ(), "GOWORK=off")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build pairedtiming: %v\n%s", err, out)
	}
	engine := &script.Engine{
		Cmds:  scripttest.DefaultCmds(),
		Conds: scripttest.DefaultConds(),
	}
	env := []string{"PATH=" + bin + string(os.PathListSeparator) + os.Getenv("PATH")}
	scripttest.Test(t, context.Background(), engine, env, "testdata/script/*.txt")
}
