//go:build linux

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudaprof"
)

type externalStackContract struct {
	PID          int    `json:"pid"`
	ExternalID   uint64 `json:"external_id"`
	ExternalKind uint32 `json:"external_kind"`
	UUID         string `json:"uuid"`
	Target       string `json:"target"`
	Artifact     string `json:"artifact"`
	SleepNs      int64  `json:"sleep_ns"`
	Output       uint32 `json:"output"`
}

// This test requires a separately admitted prebuilt target and agent, an exact
// device UUID/ordinal/SM, an inherited canonical GPU hold, and an external finite
// watchdog. It never builds a tool, invents a device activity, or acquires a lock.
func TestNativeExternalStackDeviceWeight(t *testing.T) {
	if os.Getenv("CUDAPROF_EXTERNAL_STACK_NATIVE") != "1" {
		t.Skip("requires separately authorized physical external-stack capture")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatal(err)
	}
	if !filepath.IsAbs(os.Getenv("GPU_LOCKDIR")) || os.Getenv("CODEGEN_SKIP_BINDINGS") == "1" {
		t.Fatal("explicit canonical GPU_LOCKDIR and admitted native bindings required")
	}
	if os.Getenv("CUDAPROF_CORRELATE") != "0" || os.Getenv("CUDAPROF_GO_STACKS") != "0" {
		t.Fatal("startup correlation and trace hooks must be disabled; target enables correlation after inherited admission")
	}
	target := checkedExternalStackFile(t, "CUDAPROF_EXTERNAL_STACK_TARGET")
	agent := checkedExternalStackFile(t, "CUDAPROF_EXTERNAL_STACK_AGENT")
	uuid, err := hex.DecodeString(os.Getenv("CUDAPROF_EXTERNAL_STACK_UUID"))
	if err != nil || len(uuid) != 16 || hex.EncodeToString(uuid) != os.Getenv("CUDAPROF_EXTERNAL_STACK_UUID") {
		t.Fatal("explicit 32-hex device UUID required")
	}
	if n, err := strconv.Atoi(os.Getenv("CUDAPROF_EXTERNAL_STACK_DEVICE")); err != nil || n < 0 {
		t.Fatal("explicit device ordinal required")
	}
	sm := os.Getenv("CUDAPROF_EXTERNAL_STACK_SM")
	if sm != "sm_100a" && sm != "sm_121a" {
		t.Fatal("explicit sm_100a or sm_121a target required")
	}
	root := os.Getenv("CUDAPROF_EXTERNAL_STACK_EVIDENCE")
	info, err := os.Stat(root)
	if !filepath.IsAbs(root) || err != nil || !info.IsDir() || info.Mode().Perm()&0077 != 0 {
		t.Fatal("existing private absolute evidence directory required")
	}
	dir, err := os.MkdirTemp(root, "external-stack-")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("retained physical attempt evidence: %s", dir)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	var stdout, stderr bytes.Buffer
	// captureOptions.Env is the complete child environment, not an overlay.
	// Carry verified ancestor custody without fabricating a held flag. Keep
	// pre-main correlation hooks disabled; main enables them after admission.
	env := []string{"PATH=/usr/bin:/bin", "TMPDIR=" + dir, "CUDAPROF_CORRELATE=0", "CUDAPROF_GO_STACKS=0"}
	for _, key := range []string{"GPU_LOCKDIR", "GPU_LOCK_HELD", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "CUDAPROF_EXTERNAL_STACK_NATIVE", "CUDAPROF_EXTERNAL_STACK_DEVICE", "CUDAPROF_EXTERNAL_STACK_UUID", "CUDAPROF_EXTERNAL_STACK_SM"} {
		if value, ok := os.LookupEnv(key); ok {
			env = append(env, key+"="+value)
		}
	}
	r, err := runCapture(ctx, captureOptions{
		command: []string{target}, Inject: agent, RequireComplete: true,
		APIActivity: true, apiActivitySet: true, HostSampleHz: 0, hostSampleSet: true,
		Timeout: 30 * time.Second, DrainTimeout: 5 * time.Second,
		MaxJournalRecords: 4096, CaptureFileBytes: 4 << 20, CaptureTotalBytes: 16 << 20,
		Env:    env,
		Stdout: &stdout, Stderr: &stderr,
	})
	// The real controller owns its target lifecycle. Retain failed attempts and
	// scratch, rather than claiming process cleanup or deleting missing evidence.
	for name, data := range map[string][]byte{"target.stdout": stdout.Bytes(), "target.stderr": stderr.Bytes()} {
		if e := os.WriteFile(filepath.Join(dir, name), data, 0600); e != nil {
			t.Fatal(e)
		}
	}
	status, marshalErr := json.MarshalIndent(r, "", "  ")
	if marshalErr != nil {
		t.Fatal(marshalErr)
	}
	if e := os.WriteFile(filepath.Join(dir, "controller.json"), status, 0600); e != nil {
		t.Fatal(e)
	}
	if err != nil || !r.CaptureSuccess || r.TargetExitCode != 0 || !r.CollectorFinalized {
		t.Fatalf("physical capture unqualified: %v; target=%d scratch=%s", err, r.TargetExitCode, r.ScratchDir)
	}
	var c externalStackContract
	d := json.NewDecoder(bytes.NewReader(stdout.Bytes()))
	d.DisallowUnknownFields()
	if err := d.Decode(&c); err != nil {
		t.Fatal(err)
	}
	if err := d.Decode(new(any)); err != io.EOF {
		t.Fatal("trailing target contract")
	}
	if c.PID != r.TargetPID || c.ExternalID == 0 || c.ExternalKind != 3 || c.UUID != os.Getenv("CUDAPROF_EXTERNAL_STACK_UUID") || c.Target != sm || c.SleepNs != int64(10*time.Millisecond) || c.Output != 42 {
		t.Fatalf("target contract mismatch: %+v", c)
	}
	if digest, err := hex.DecodeString(c.Artifact); err != nil || len(digest) != 32 {
		t.Fatal("missing real target artifact identity")
	}
	path := filepath.Join(dir, "capture.cudaprof")
	if _, err := writeCapture(path, r, captureArtifactBudget{FileBytes: 4 << 20, TotalBytes: 16 << 20}); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	cap, readErr := cudacapture.Read(f)
	closeErr := f.Close()
	if readErr != nil || closeErr != nil {
		t.Fatalf("read capture: %v; close: %v", readErr, closeErr)
	}
	stackData := cap.Files[cudacapture.ExternalStacksName]
	if len(stackData) == 0 {
		t.Fatal("real target sidecar was not attached")
	}
	if cap.Profile == nil {
		t.Fatal("capture has no activity profile")
	}
	// This exercises the actual controller's existing tracker projection too.
	matchedProfile := 0
	for _, sample := range cap.Profile.Sample {
		stack := strings.Join(sample.Label["cupti.external.stack"], "")
		if strings.Contains(stack, "sleptKernel") {
			if strings.Join(sample.Label["cupti.external.runtime_status"], "") != "matched" || strings.Join(sample.Label["cupti.external.stack_basis"], "") != "runtime_api" {
				t.Fatal("controller did not project a real runtime/API stack join")
			}
			matchedProfile++
		}
	}
	if matchedProfile != 1 {
		t.Fatalf("controller matched stack samples=%d, want 1", matchedProfile)
	}
	joins := externalStackJoins(t, r, stackData, r.Activities, c)
	if len(joins) != 1 || !joins[0].HasExternalID || joins[0].ExternalID != c.ExternalID || !strings.Contains(strings.Join(joins[0].Callstack, ";"), "sleptKernel") {
		t.Fatalf("real ID/stack join missing: %+v", joins)
	}
	j := joins[0]
	if j.KernelName != "external_stack_short" || j.DeviceEnd <= j.DeviceStart {
		t.Fatalf("actual short kernel interval missing: %+v", j)
	}
	// There is exactly one real kernel, so its per-device interval union is
	// independently End-Start. No host clock is used to choose or join it.
	want := j.DeviceEnd - j.DeviceStart
	if want >= uint64(c.SleepNs) {
		t.Fatal("short-kernel discrimination unproven: device interval is not shorter than host sleep")
	}
	var folded bytes.Buffer
	if err := cudacapture.ExportFoldedStacks(&folded, joins); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(folded.String(), "sleptKernel") || !strings.HasSuffix(folded.String(), fmt.Sprintf(";[GPU]external_stack_short %d\n", want)) || bytes.Count(folded.Bytes(), []byte{'\n'}) != 1 {
		t.Fatalf("folded output does not use real kernel union %d: %s", want, &folded)
	}
	if err := os.WriteFile(filepath.Join(dir, "folded.txt"), folded.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	for _, mode := range []string{"missing_external_link", "scrambled_external_id"} {
		t.Run(mode, func(t *testing.T) {
			derived := make([]cudacapture.ActivityRecord, 0, len(r.Activities))
			changed := 0
			for _, a := range r.Activities {
				if a.Subtype == "external_correlation" && a.Process == uint64(c.PID) && a.ExternalKind == c.ExternalKind && a.ExternalID == c.ExternalID {
					changed++
					if mode == "missing_external_link" {
						continue
					}
					// Only an observed link is changed; all device intervals,
					// runtime API rows and actual sidecar stacks are untouched.
					a.ExternalID = absentExternalStackID(t, stackData)
				}
				derived = append(derived, a)
			}
			if changed == 0 {
				t.Fatal("no real link changed")
			}
			got := externalStackJoins(t, r, stackData, derived, c)
			if len(got) != 1 || len(got[0].Callstack) != 0 {
				t.Fatalf("timestamps or wrong ID supplied stack: %+v", got)
			}
			if got[0].DeviceStart != j.DeviceStart || got[0].DeviceEnd != j.DeviceEnd {
				t.Fatal("control altered real device interval")
			}
		})
	}
}

func checkedExternalStackFile(t *testing.T, key string) string {
	t.Helper()
	path, digest := os.Getenv(key), os.Getenv(key+"_SHA256")
	if !filepath.IsAbs(path) || len(digest) != 64 {
		t.Fatalf("absolute %s and exact SHA256 required", key)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	info, err := f.Stat()
	if err != nil || !info.Mode().IsRegular() || info.Size() > 64<<20 {
		f.Close()
		t.Fatal("tool must be a bounded regular file")
	}
	h := sha256.New()
	n, copyErr := io.Copy(h, io.LimitReader(f, (64<<20)+1))
	after, statErr := f.Stat()
	closeErr := f.Close()
	if copyErr != nil || statErr != nil || closeErr != nil || n != info.Size() || !os.SameFile(info, after) || !info.ModTime().Equal(after.ModTime()) || hex.EncodeToString(h.Sum(nil)) != digest {
		t.Fatalf("tool custody/hash mismatch: %s", key)
	}
	return path
}

func externalStackJoins(t *testing.T, r captureResult, data []byte, activities []cudacapture.ActivityRecord, c externalStackContract) []cudacapture.CorrelationJoin {
	t.Helper()
	tracker := cudacapture.NewCorrelationTracker()
	if err := tracker.ReadExternalStackTable(bytes.NewReader(data)); err != nil {
		t.Fatal(err)
	}
	kernels := 0
	var kernel cudacapture.ActivityRecord
	for _, a := range activities {
		if a.Process != uint64(c.PID) {
			continue
		}
		if a.Subtype == "kernel" && a.Valid {
			kernels++
			kernel = a
		}
		if a.Subtype == "external_correlation" || a.Subtype == "kernel" && a.Valid {
			if err := tracker.RecordActivity(a); err != nil {
				t.Fatal(err)
			}
		}
	}
	if kernels != 1 {
		t.Fatalf("actual target kernel records=%d, want 1", kernels)
	}
	corr := kernel.Correlation
	if corr == 0 {
		corr = kernel.RuntimeCorrelation
	}
	if corr == 0 {
		t.Fatal("real kernel has no correlation ID")
	}
	for _, api := range r.APIRecords {
		if api.ProcessID == uint32(c.PID) && api.Domain == "driver" && api.Correlation == corr {
			if api.ReturnValue != 0 {
				t.Fatal("associated driver call did not succeed")
			}
			// The name is the real activity's symbol for this exact ID;
			// host thread and launch presence come only from the API row.
			if err := tracker.RecordRuntime(cudacapture.RuntimeLaunch{CorrelationID: uint64(api.Correlation), KernelName: kernel.Name, HostThreadID: uint64(api.ThreadID)}); err != nil {
				t.Fatal(err)
			}
		}
	}
	joins, ambiguous, _, missing := tracker.JoinKind(c.ExternalKind)
	if len(ambiguous) != 0 || len(missing) != 0 {
		t.Fatalf("actual join unresolved: ambiguous=%+v missing=%v", ambiguous, missing)
	}
	return joins
}

func absentExternalStackID(t *testing.T, data []byte) uint64 {
	t.Helper()
	var table cudaprof.ExternalStackTable
	if err := json.Unmarshal(data, &table); err != nil {
		t.Fatal(err)
	}
	used := make(map[uint64]bool)
	for _, row := range table.Stacks {
		used[row.ExternalID] = true
	}
	for id := uint64(1); id <= uint64(len(used))+1; id++ {
		if !used[id] {
			return id
		}
	}
	t.Fatal("no bounded unused external ID")
	return 0
}
