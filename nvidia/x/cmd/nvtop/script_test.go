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

// TestScript runs nvtop as a program and checks what it prints. The
// scripts that need a GPU are guarded by the nvml condition.
func TestScript(t *testing.T) {
	bin := t.TempDir()
	build := exec.Command("go", "build", "-o", filepath.Join(bin, "nvtop"), ".")
	build.Env = append(os.Environ(), "GOWORK=off")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build nvtop: %v\n%s", err, out)
	}
	conds := scripttest.DefaultConds()
	conds["nvml"] = script.BoolCondition("NVML reports a device", haveDevice())
	engine := &script.Engine{
		Cmds:  scripttest.DefaultCmds(),
		Conds: conds,
	}
	env := []string{
		"PATH=" + bin + string(os.PathListSeparator) + os.Getenv("PATH"),
		"HOME=" + os.Getenv("HOME"),
	}
	scripttest.Test(t, context.Background(), engine, env, "testdata/script/*.txt")
}

// haveDevice reports whether NVML loads and reports a device.
func haveDevice() bool {
	src, err := openNVML(newHost(os.DirFS("/proc")))
	if err != nil {
		return false
	}
	defer src.close()
	return len(src.devices()) > 0
}
