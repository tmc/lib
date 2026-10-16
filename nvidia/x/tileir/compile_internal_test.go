package tileir

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cudair"
)

// CheckTileCompiler lets external tests share the NVRTC version boundary.
var CheckTileCompiler = checkTileCompiler

// NVRTC 13.0, 13.1 and 13.2 fail with "unrecognized option --tile-only";
// 13.3 and 13.4 compile the semantic source (measured on GB10).
func TestCheckTileCompiler(t *testing.T) {
	tests := []struct {
		major, minor int
		want         error
	}{
		{12, 9, ErrTileCompilerUnsupported},
		{13, 0, ErrTileCompilerUnsupported},
		{13, 2, ErrTileCompilerUnsupported},
		{13, 3, nil},
		{13, 4, nil},
		{14, 0, nil},
	}
	for _, tt := range tests {
		err := checkTileCompiler(tt.major, tt.minor)
		if !errors.Is(err, tt.want) || (err == nil) != (tt.want == nil) {
			t.Errorf("checkTileCompiler(%d, %d) = %v, want %v", tt.major, tt.minor, err, tt.want)
		}
	}
	const want = "tileir: NVRTC does not support tile-only compilation: NVRTC 13.2, need 13.3 or later"
	if err := checkTileCompiler(13, 2); err == nil || err.Error() != want {
		t.Errorf("checkTileCompiler(13, 2) = %v, want %q", err, want)
	}
}

const oldNVRTCChildEnv = "TILEIR_OLD_NVRTC_CHILD"

// TestCompileSemanticTileRefusesOldNVRTC loads an NVRTC older than 13.3 in a
// child process, where NVRTC_LIB_PATH selects the library before first use,
// and requires CompileSemanticTile to refuse it before compiling.
// CUDA_TILE_OLD_NVRTC_LIB overrides the library search.
func TestCompileSemanticTileRefusesOldNVRTC(t *testing.T) {
	if os.Getenv(oldNVRTCChildEnv) == "1" {
		compileWithOldNVRTC(t)
		return
	}
	if os.Getenv("CUDA_TILE_NVRTC_TEST") != "1" {
		t.Skip("actual NVRTC semantic artifact gate disabled")
	}
	lib := os.Getenv("CUDA_TILE_OLD_NVRTC_LIB")
	if lib == "" {
		for _, v := range []string{"13.2", "13.1", "13.0"} {
			path := filepath.Join("/usr/local/cuda-"+v, "lib64", "libnvrtc.so")
			if _, err := os.Stat(path); err == nil {
				lib = path
				break
			}
		}
	}
	if lib == "" {
		t.Skip("no NVRTC older than 13.3 found; set CUDA_TILE_OLD_NVRTC_LIB")
	}
	cmd := exec.Command(os.Args[0], "-test.run=^TestCompileSemanticTileRefusesOldNVRTC$", "-test.v", "-test.count=1")
	cmd.Env = append(os.Environ(), oldNVRTCChildEnv+"=1", "NVRTC_LIB_PATH="+lib)
	out, err := cmd.CombinedOutput()
	if err != nil || !strings.Contains(string(out), "--- PASS: TestCompileSemanticTileRefusesOldNVRTC") {
		t.Fatalf("child with NVRTC_LIB_PATH=%s: %v\n%s", lib, err, out)
	}
	t.Logf("child with NVRTC_LIB_PATH=%s:\n%s", lib, out)
}

func compileWithOldNVRTC(t *testing.T) {
	major, minor, err := nvrtc.Version()
	if err != nil {
		t.Fatalf("nvrtc.Version: %v", err)
	}
	if checkTileCompiler(int(major), int(minor)) == nil {
		t.Fatalf("NVRTC_LIB_PATH=%s loaded NVRTC %d.%d, want older than 13.3", os.Getenv("NVRTC_LIB_PATH"), major, minor)
	}
	g := cudair.NewGraph()
	a := g.AddInput("a", []int{2, 3}, cudair.Float32)
	g.AddReduce("sum", a, 1)
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	artifact, log, err := CompileSemanticTile(ctx, snap, LowerOptions{KernelName: "sum_rows"}, 9, 0)
	if !errors.Is(err, ErrTileCompilerUnsupported) {
		t.Fatalf("CompileSemanticTile with NVRTC %d.%d = %v, want ErrTileCompilerUnsupported", major, minor, err)
	}
	if artifact != nil || log != "" {
		t.Fatalf("refused compile returned artifact %v, log %q", artifact, log)
	}
	if !strings.Contains(err.Error(), nvrtc.Path()) {
		t.Fatalf("refusal %q does not name library %s", err, nvrtc.Path())
	}
	t.Logf("NVRTC %d.%d (%s): %v", major, minor, nvrtc.Path(), err)
}
