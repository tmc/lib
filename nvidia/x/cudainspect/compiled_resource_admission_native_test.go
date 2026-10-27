//go:build native

package cudainspect

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudalock"
)

// TestCompiledResourceNativeAdmissionBoundary calls the actual native fixture
// with an invalid holder and no packet/output paths. Even with its admission
// omitted, the fixture must stop at the empty packet-path guard, before source
// emission, image reads, inspection, RequireDevice, or any vendor entry.
func TestCompiledResourceNativeAdmissionBoundary(t *testing.T) {
	if os.Getenv("COMPILED_RESOURCE_ADMISSION_CHILD") == "1" {
		if os.Getenv("GPU_LOCK_HELD") != "1" || os.Getenv("COMPILED_RESOURCE_PACKET_DIR") != "" || os.Getenv("COMPILED_RESOURCE_OUTPUT_DIR") != "" || os.Getenv("GPU_LOCKDIR") == "" {
			t.Fatal("unsafe admission-boundary child environment")
		}
		runCompiledMLP(t, 0)
		t.Fatal("compiled-resource fixture unexpectedly returned from early refusal")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestCompiledResourceNativeAdmissionBoundary$", "-test.v")
	cmd.WaitDelay = 2 * time.Second
	cmd.Env = append(os.Environ(),
		"COMPILED_RESOURCE_ADMISSION_CHILD=1",
		"GPU_LOCK_HELD=1",
		"GPU_LOCKDIR="+filepath.Join(t.TempDir(), "unknown-lock"),
		"GPU_LOCK_DISABLED=",
		"CUDATEST_NO_LOCK=",
		"COMPILED_RESOURCE_PACKET_DIR=",
		"COMPILED_RESOURCE_OUTPUT_DIR=",
	)
	out, err := cmd.CombinedOutput()
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) || exitErr.ExitCode() != 1 {
		t.Fatalf("admission-boundary child exit = %v; want1\n%s", err, out)
	}
	text := string(out)
	if !strings.Contains(text, cudalock.ErrNotInherited.Error()) {
		t.Fatalf("missing canonical early refusal:\n%s", text)
	}
	if strings.Contains(text, "COMPILED_RESOURCE_PACKET_DIR required") || strings.Contains(text, "COMPILED_RESOURCE_OUTPUT_DIR required") {
		t.Fatalf("packet guards reached before custody refusal:\n%s", text)
	}
}
