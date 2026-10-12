package cudaprof

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/cupti"
)

//go:noinline
func hostSleepCallSite(f func()) uint64 {
	return DoID(context.Background(), Labels(), func(context.Context) { f() })
}

// TestFoldedHostSleepIsNotKernelTime runs two Do regions at one call site,
// each sleeping 10ms on the host before a 500ns kernel. The stacks are joined
// to the kernels by the external IDs read back from the sidecar. The folded
// weight must be the 1000ns of kernel time, not the host time between and
// around the kernels.
func TestFoldedHostSleepIsNotKernelTime(t *testing.T) {
	const (
		sleep    = 10 * time.Millisecond
		kernelNs = 500
	)
	resetCorrelation(t)
	base := filepath.Join(t.TempDir(), "external-stacks.json")
	t.Setenv(ExternalStacksEnv, base)
	withFakeCorrelation(t, func(cupti.ExternalCorrelationKind, uint64) error { return nil })

	origin := time.Now()
	launchNs := make(map[uint64]uint64)
	for range 2 {
		var start uint64
		regionStart := time.Now()
		id := hostSleepCallSite(func() {
			time.Sleep(sleep)
			start = uint64(time.Since(origin))
		})
		if region := time.Since(regionStart); region < sleep {
			t.Fatalf("Do region took %v, want at least %v", region, sleep)
		}
		if id == 0 {
			t.Fatal("DoID returned no external ID with a succeeding push")
		}
		launchNs[id] = start
	}

	stacks := readSidecarStacks(t, fmt.Sprintf("%s.%d", base, os.Getpid()))
	var execs []KernelExecution
	var last uint64
	for id, start := range launchNs {
		stack, ok := stacks[id]
		if !ok {
			t.Fatalf("sidecar has no stack for external ID %d", id)
		}
		if mem, _ := ExternalStack(id); !slices.Equal(mem, stack) {
			t.Fatalf("sidecar stack %q differs from in-process stack %q", stack, mem)
		}
		execs = append(execs, KernelExecution{KernelName: "short", Callstack: stack, StartNs: start, EndNs: start + kernelNs})
		last = max(last, start+kernelNs)
	}
	if first := min(execs[0].StartNs, execs[1].StartNs); last-first < uint64(sleep) {
		t.Fatalf("kernels span %dns, want at least the %v host sleep between them", last-first, sleep)
	}

	var buf bytes.Buffer
	if err := ExportFoldedStacks(&buf, execs); err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSuffix(buf.String(), "\n"), "\n")
	if len(lines) != 1 {
		t.Fatalf("folded output has %d lines, want 1:\n%s", len(lines), buf.String())
	}
	if !strings.Contains(lines[0], "hostSleepCallSite;[GPU]short ") {
		t.Errorf("folded line %q does not attribute the kernel to hostSleepCallSite", lines[0])
	}
	if want := fmt.Sprintf(" %d", 2*kernelNs); !strings.HasSuffix(lines[0], want) {
		t.Errorf("folded line %q, want weight%s", lines[0], want)
	}
}

func readSidecarStacks(t *testing.T, path string) map[uint64][]string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	stacks := make(map[uint64][]string)
	sc := bufio.NewScanner(bytes.NewReader(data))
	sc.Buffer(nil, 1<<20)
	for sc.Scan() {
		var line externalStackSidecarLine
		if err := json.Unmarshal(sc.Bytes(), &line); err != nil {
			t.Fatalf("sidecar line %q: %v", sc.Bytes(), err)
		}
		if line.Kind == "stack" && line.ExternalKind == uint32(cupti.ExternalCorrelationKindCustom0) {
			stacks[line.ExternalID] = line.Frames
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	return stacks
}
