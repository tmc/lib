package cudaprof

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
)

type resilienceShortFile struct {
	externalStackSidecarFile
	writes *int
}

func (f resilienceShortFile) Write(data []byte) (int, error) {
	*f.writes++
	return f.externalStackSidecarFile.Write(data[:len(data)/2])
}

func TestCaptureResilience_PartialWritePoisonsWriter(t *testing.T) {
	ResetExternalStacks()
	t.Setenv(ExternalStacksEnv, filepath.Join(t.TempDir(), "stacks"))
	old := openExternalStackSidecarFile
	writes := 0
	openExternalStackSidecarFile = func(path string) (externalStackSidecarFile, error) {
		f, e := old(path)
		return resilienceShortFile{f, &writes}, e
	}
	t.Cleanup(func() { ResetExternalStacks(); openExternalStackSidecarFile = old })
	for id := uint64(1); id <= 3; id++ {
		if RecordExternalStackKind(cupti.ExternalCorrelationKindCustom0, id, []string{"frame"}) {
			t.Fatal("short write retained")
		}
	}
	if writes != 1 || ExternalStackLosses() != 3 {
		t.Fatalf("poisoned writer appended: writes=%d losses=%d", writes, ExternalStackLosses())
	}
}

func TestCaptureResilience_Complete64FramesAcrossFile(t *testing.T) {
	if os.Getenv("CUDAPROF_RESILIENCE_64") == "child" {
		pc := resilienceCallerPC()
		pcs := make([]uintptr, 64)
		for i := range pcs {
			pcs[i] = pc
		}
		if !RecordExternalStackKind(cupti.ExternalCorrelationKindCustom0, 1, formatCallstack(pcs)) {
			os.Exit(2)
		}
		os.Exit(0)
	}
	base := filepath.Join(t.TempDir(), "stacks")
	cmd := exec.Command(os.Args[0], "-test.run=^TestCaptureResilience_Complete64FramesAcrossFile$")
	cmd.Env = append(os.Environ(), "CUDAPROF_RESILIENCE_64=child", ExternalStacksEnv+"="+base, "CUDAPROF_SESSION=epoch64")
	if out, e := cmd.CombinedOutput(); e != nil {
		t.Fatalf("writer: %v %s", e, out)
	}
	f, e := os.Open(fmt.Sprintf("%s.%d", base, cmd.Process.Pid))
	if e != nil {
		t.Fatal(e)
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	var rows int
	for s.Scan() {
		var row externalStackSidecarLine
		if e := json.Unmarshal(s.Bytes(), &row); e != nil {
			t.Fatal(e)
		}
		if row.Kind != "stack" {
			continue
		}
		rows++
		if row.Process != uint64(cmd.Process.Pid) || row.Epoch != "epoch64" || len(row.Frames) != 64 || strings.Contains(strings.Join(row.Frames, ";"), "[truncated]") {
			t.Fatalf("64-frame target row: %+v", row)
		}
	}
	if s.Err() != nil || rows != 1 {
		t.Fatalf("sidecar rows=%d: %v", rows, s.Err())
	}
}

//go:noinline
func resilienceCallerPC() uintptr {
	var pc [1]uintptr
	runtime.Callers(1, pc[:])
	return pc[0]
}

func TestCaptureResilience_FrameBudgetBeforeMarshal(t *testing.T) {
	ResetExternalStacks()
	t.Cleanup(ResetExternalStacks)
	if RecordExternalStackKind(cupti.ExternalCorrelationKindCustom0, 1, []string{strings.Repeat("x", 8193)}) {
		t.Fatal("oversized frame retained")
	}
	if ExternalStackLosses() != 1 {
		t.Fatal("oversize not counted")
	}
	if _, e := marshalExternalStackSidecarLine(externalStackSidecarLine{Kind: "stack", Frames: []string{strings.Repeat("x", 8193)}}); e == nil {
		t.Fatal("marshal accepted unbounded frame")
	}
}
