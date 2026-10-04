package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof"
)

// These host targets publish genuine target-side stacks. Device activities in
// the host projection test below are explicitly synthetic join controls.
func TestCaptureResilience_Target(t *testing.T) {
	mode := os.Getenv("CUDAPROF_RESILIENCE_TARGET")
	if mode == "" {
		return
	}
	if mode == "pair" {
		var children []*exec.Cmd
		for _, role := range []string{"bad", "good"} {
			cmd := exec.Command(os.Args[0], "-test.run=^TestCaptureResilience_Target$")
			cmd.Env = setCaptureEnv(os.Environ(), "CUDAPROF_RESILIENCE_TARGET", role)
			if err := cmd.Start(); err != nil {
				t.Fatal(err)
			}
			children = append(children, cmd)
		}
		for i, cmd := range children {
			role := []string{"bad", "good"}[i]
			if err := waitResilienceFile(filepath.Join(os.Getenv("CUDAPROF_TRANSPORT"), role+".ready")); err != nil {
				t.Fatal(err)
			}
			if role == "bad" {
				if err := cmd.Process.Kill(); err != nil {
					t.Fatal(err)
				}
			}
		}
		badErr := children[0].Wait()
		if badErr == nil {
			t.Fatal("planted child did not fail")
		}
		if err := children[1].Wait(); err != nil {
			t.Fatal(err)
		}
		fmt.Printf("%d %d\n", children[0].Process.Pid, children[1].Process.Pid)
		fmt.Printf("failed_process=%d cause=%q\n", children[0].Process.Pid, badErr)
		return
	}
	if !cudaprof.RecordExternalStack(cupti.ExternalCorrelationKindCustom0, 71, []string{"main", mode}) {
		t.Fatal("target stack not retained")
	}
	if os.Getenv("CUDAPROF_RESILIENCE_GPU") == "1" {
		if resilienceNativeWork == nil {
			t.Fatal("native target implementation unavailable")
		}
		if err := resilienceNativeWork(); err != nil {
			t.Fatal(err)
		}
	}
	if mode == "bad" {
		path := fmt.Sprintf("%s.%d", os.Getenv(cudaprof.ExternalStacksEnv), os.Getpid())
		f, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0600)
		if err != nil {
			t.Fatal(err)
		}
		if _, err = f.WriteString(`{"version":1,"kind":"stack","external_id":72`); err != nil {
			t.Fatal(err)
		}
		f.Close()
	}
	if err := os.WriteFile(filepath.Join(os.Getenv("CUDAPROF_TRANSPORT"), mode+".ready"), []byte("ready"), 0600); err != nil {
		t.Fatal(err)
	}
	if mode == "cancel" {
		fmt.Println("stack-published")
	}
	if mode == "bad" || mode == "cancel" {
		for {
			time.Sleep(time.Second)
		}
	}
}

var resilienceNativeWork func() error

func waitResilienceFile(path string) error {
	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) {
		if _, err := os.Stat(path); err == nil {
			return nil
		}
		time.Sleep(10 * time.Millisecond)
	}
	return fmt.Errorf("target readiness timeout: %s", path)
}

func resilienceCapture(t *testing.T, mode string, ctx context.Context, stdout io.Writer) (captureResult, error) {
	t.Helper()
	r, err := runCapture(ctx, captureOptions{command: []string{os.Args[0], "-test.run=^TestCaptureResilience_Target$"}, Env: setCaptureEnv(os.Environ(), "CUDAPROF_RESILIENCE_TARGET", mode), NoInject: true, FollowChildren: true, Stdout: stdout, Stderr: stdout, Timeout: 30 * time.Second, DrainTimeout: 100 * time.Millisecond})
	if r.ScratchDir != "" {
		t.Cleanup(func() { os.RemoveAll(r.ScratchDir) })
	}
	return r, err
}

func resilienceTable(t *testing.T, r captureResult) (cudaprof.ExternalStackTable, stackSidecarRead) {
	t.Helper()
	got, err := readExternalStackSidecar(filepath.Join(r.ScratchDir, cudacapture.ExternalStacksName), uint64(r.TargetPID), r.Session)
	if err != nil {
		t.Fatal(err)
	}
	var table cudaprof.ExternalStackTable
	if err := json.Unmarshal(got.data, &table); err != nil {
		t.Fatal(err)
	}
	return table, got
}

func TestCaptureResilience_ReexecTwoProcessesAndKilledWriter(t *testing.T) {
	var stdout bytes.Buffer
	r, err := resilienceCapture(t, "pair", context.Background(), &stdout)
	if err != nil {
		t.Fatal(err)
	}
	var bad, good uint64
	if _, err := fmt.Sscanf(stdout.String(), "%d %d", &bad, &good); err != nil {
		t.Fatalf("child receipt %q: %v", stdout.String(), err)
	}
	if bad == good || bad == 0 || good == 0 {
		t.Fatal("child identities collapsed")
	}
	table, sidecar := resilienceTable(t, r)
	if len(table.Stacks) != 2 || table.Losses != 0 || table.DiscardedRecords != 1 || !table.LossesLowerBound || !sidecar.damaged {
		t.Fatalf("survival/loss receipt: %+v damaged=%v", table, sidecar.damaged)
	}
	for _, row := range table.Stacks {
		want := "good"
		if row.Process == bad {
			want = "bad"
		} else if row.Process != good {
			t.Fatalf("unexpected PID %d", row.Process)
		}
		if row.ExternalID != 71 || strings.Join(row.Frames, ";") != "main;"+want {
			t.Fatalf("cross-process stack: %+v", row)
		}
	}
	if !strings.Contains(strings.Join(sidecar.damageFiles, ","), "."+strconv.FormatUint(bad, 10)) {
		t.Fatal("failed writer not named")
	}
	// Host-only synthetic activity verifies the actual capture-file join;
	// it is not evidence of CUPTI or a successful native capture.
	for i, pid := range []uint64{bad, good} {
		corr := uint32(171 + i)
		r.Activities = append(r.Activities, cudacapture.ActivityRecord{Process: pid, Subtype: "external_correlation", ExternalKind: 3, ExternalID: 71, Correlation: corr}, cudacapture.ActivityRecord{Process: pid, Subtype: "kernel", Name: []string{"bad", "good"}[i], Correlation: corr, Start: 10, End: 20, Valid: true})
	}
	path := filepath.Join(t.TempDir(), "host-control.cudaprof")
	if _, err := writeCapture(path, r, captureArtifactBudget{FileBytes: 1 << 20, TotalBytes: 4 << 20}); err != nil {
		t.Fatal(err)
	}
	cap, err := readCapturePath(path)
	if err != nil {
		t.Fatal(err)
	}
	samples := samplesByName(cap)
	assertStack(t, samples["bad"], "matched", "main;bad")
	assertStack(t, samples["good"], "matched", "main;good")
}

func TestCaptureResilience_CancellationRetainsFailureAndEvidence(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX group termination; Windows parser/build controls remain separate")
	}
	cause := errors.New("requested resilience stop")
	ctx, cancel := context.WithCancelCause(context.Background())
	defer cancel(cause)
	// Cancel only after the real target sidecar exists. This avoids a
	// scheduling deadline deciding whether evidence was ever produced.
	stdout := &resilienceReadyWriter{ready: make(chan struct{}, 1)}
	ready := make(chan captureResult, 1)
	go func() {
		r, err := resilienceCapture(t, "cancel", ctx, stdout)
		if !errors.Is(err, cause) {
			t.Errorf("original cause lost: %v", err)
		}
		ready <- r
	}()
	// The child acknowledges target-side publication on stdout below.
	deadline := time.After(20 * time.Second)
	select {
	case <-deadline:
		t.Fatal("target publication timeout")
	case <-stdout.ready:
	}
	cancel(cause)
	r := <-ready
	if !r.Cancelled || r.TargetSignal == "" {
		t.Fatalf("termination identity lost: %+v", r)
	}
	table, _ := resilienceTable(t, r)
	if len(table.Stacks) != 1 || table.Stacks[0].Process != uint64(r.TargetPID) {
		t.Fatal("cancelled target evidence lost")
	}
}

type resilienceReadyWriter struct {
	data  bytes.Buffer
	ready chan struct{}
}

func (w *resilienceReadyWriter) Write(p []byte) (int, error) {
	n, err := w.data.Write(p)
	if strings.Contains(w.data.String(), "stack-published") {
		select {
		case w.ready <- struct{}{}:
		default:
		}
	}
	return n, err
}

func TestCaptureResilience_FreshEpochHasNoOldStacks(t *testing.T) {
	var aout, bout bytes.Buffer
	a, err := resilienceCapture(t, "old", context.Background(), &aout)
	if err != nil {
		t.Fatal(err)
	}
	b, err := resilienceCapture(t, "fresh", context.Background(), &bout)
	if err != nil {
		t.Fatal(err)
	}
	if a.Session == b.Session || a.ScratchDir == b.ScratchDir || a.TargetPID == b.TargetPID {
		t.Fatal("fresh target reused epoch or PID")
	}
	table, _ := resilienceTable(t, b)
	if len(table.Stacks) != 1 || strings.Join(table.Stacks[0].Frames, ";") != "main;fresh" {
		t.Fatal("stale target stack inherited")
	}
	oldPath := fmt.Sprintf("%s.%d", filepath.Join(a.ScratchDir, cudacapture.ExternalStacksName), a.TargetPID)
	data, err := os.ReadFile(oldPath)
	if err != nil {
		t.Fatal(err)
	}
	// Copy an old epoch into a new session under the same old PID: only the
	// epoch check can reject it; filename and record PID deliberately agree.
	newPath := fmt.Sprintf("%s.%d", filepath.Join(b.ScratchDir, cudacapture.ExternalStacksName), a.TargetPID)
	if err := os.WriteFile(newPath, data, 0600); err != nil {
		t.Fatal(err)
	}
	table, got := resilienceTable(t, b)
	if len(table.Stacks) != 1 || !got.damaged {
		t.Fatalf("copied stale epoch admitted: %+v", table)
	}
}

func TestCaptureResilience_TornTailPreservesCompleteRows(t *testing.T) {
	var out bytes.Buffer
	r, err := resilienceCapture(t, "good", context.Background(), &out)
	if err != nil {
		t.Fatal(err)
	}
	path := fmt.Sprintf("%s.%d", filepath.Join(r.ScratchDir, cudacapture.ExternalStacksName), r.TargetPID)
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		t.Fatal(err)
	}
	_, err = f.WriteString(`{"version":1,"kind":"stack","external_id":2`)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	table, got := resilienceTable(t, r)
	if len(table.Stacks) != 1 || table.Losses != 0 || table.DiscardedRecords != 1 || !table.LossesLowerBound || !got.damaged {
		t.Fatalf("torn target table: %+v", table)
	}
}

func TestCaptureResilience_CorruptProcessDoesNotDropGoodProcess(t *testing.T) {
	var out bytes.Buffer
	r, err := resilienceCapture(t, "pair", context.Background(), &out)
	if err != nil {
		t.Fatal(err)
	}
	var bad, good uint64
	fmt.Sscanf(out.String(), "%d %d", &bad, &good)
	path := fmt.Sprintf("%s.%d", filepath.Join(r.ScratchDir, cudacapture.ExternalStacksName), bad)
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		t.Fatal(err)
	}
	// Finish the torn row as corrupt framing, then place another valid row
	// after it. The next complete row is independently recoverable.
	f.WriteString("\n{\"version\":1,\"kind\":\"stack\",\"external_id\":0,\"frames\":[\"invalid\"]}\n")
	row := stackSidecarLine{Version: 1, Kind: "stack", Process: bad, Epoch: r.Session, ExternalKind: 3, ExternalID: 72, Frames: []string{"main", "after-corruption"}}
	b, _ := json.Marshal(row)
	f.Write(append(b, '\n'))
	f.Close()
	table, got := resilienceTable(t, r)
	if len(table.Stacks) != 3 || table.Losses != 0 || table.DiscardedRecords != 2 || !got.damaged {
		t.Fatalf("recoverable rows lost: %+v", table)
	}
	var goodRows int
	for _, s := range table.Stacks {
		if s.Process == good {
			goodRows++
			if strings.Join(s.Frames, ";") != "main;good" {
				t.Fatal("wrong PID stack inherited")
			}
		}
	}
	if goodRows != 1 {
		t.Fatal("intact process dropped")
	}
}

func TestCaptureResilience_FilenamePIDCannotBorrowAnotherStack(t *testing.T) {
	var out bytes.Buffer
	r, err := resilienceCapture(t, "pair", context.Background(), &out)
	if err != nil {
		t.Fatal(err)
	}
	var bad, good uint64
	fmt.Sscanf(out.String(), "%d %d", &bad, &good)
	path := fmt.Sprintf("%s.%d", filepath.Join(r.ScratchDir, cudacapture.ExternalStacksName), bad)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	data = bytes.ReplaceAll(data, []byte(fmt.Sprintf(`"process":%d`, bad)), []byte(fmt.Sprintf(`"process":%d`, good)))
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	table, got := resilienceTable(t, r)
	if !got.damaged || len(table.Stacks) != 1 || table.Stacks[0].Process != good || strings.Join(table.Stacks[0].Frames, ";") != "main;good" {
		t.Fatalf("PID borrowing admitted: %+v", table)
	}
}

func TestCaptureResilience_OversizedRowsAreBoundedAndCounted(t *testing.T) {
	var out bytes.Buffer
	r, err := resilienceCapture(t, "good", context.Background(), &out)
	if err != nil {
		t.Fatal(err)
	}
	base := filepath.Join(r.ScratchDir, cudacapture.ExternalStacksName)
	path := fmt.Sprintf("%s.%d", base, r.TargetPID)
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		t.Fatal(err)
	}
	f.WriteString(strings.Repeat("x", maxStackSidecarRowBytes+1) + "\n")
	f.Close()
	table, got := resilienceTable(t, r)
	if len(table.Stacks) != 1 || table.Losses != 0 || table.DiscardedRecords != 1 || !got.damaged {
		t.Fatalf("oversized row: %+v", table)
	}
	// Sparse oversized process file must never be read into memory.
	f, err = os.Create(base + ".999999999")
	if err != nil {
		t.Fatal(err)
	}
	if err = f.Truncate(maxStackSidecarBytes + 1); err != nil {
		t.Fatal(err)
	}
	f.Close()
	table, got = resilienceTable(t, r)
	if len(table.Stacks) != 1 || table.Losses != 0 || table.DiscardedRecords != 1 || !table.DiscardedRecordsLowerBound || !table.LossesLowerBound || !got.damaged {
		t.Fatalf("oversized file: %+v", table)
	}
}

func TestCaptureResilience_LargeCorruptProcessKeepsSmallGoodAttachment(t *testing.T) {
	var out bytes.Buffer
	r, err := resilienceCapture(t, "good", context.Background(), &out)
	if err != nil {
		t.Fatal(err)
	}
	base := filepath.Join(r.ScratchDir, cudacapture.ExternalStacksName)
	if err := os.WriteFile(base+".999999999", []byte(strings.Repeat("x", maxStackSidecarRowBytes+1)+"\n"), 0600); err != nil {
		t.Fatal(err)
	}
	pid := uint64(r.TargetPID)
	// Synthetic activity tests only optional-file finalization, not CUPTI.
	r.Activities = []cudacapture.ActivityRecord{{Process: pid, Subtype: "external_correlation", ExternalKind: 3, ExternalID: 71, Correlation: 171}, {Process: pid, Subtype: "kernel", Name: "good", Correlation: 171, Start: 10, End: 20, Valid: true}}
	path := filepath.Join(t.TempDir(), "capture.cudaprof")
	if _, err := writeCapture(path, r, captureArtifactBudget{FileBytes: 4096, TotalBytes: 1 << 20}); err != nil {
		t.Fatal(err)
	}
	cap, err := readCapturePath(path)
	if err != nil {
		t.Fatal(err)
	}
	assertStack(t, samplesByName(cap)["good"], "matched", "main;good")
	if len(cap.Files[cudacapture.ExternalStacksName]) == 0 {
		t.Fatal("corrupt raw file removed intact normalized attachment")
	}
	if !strings.Contains(strings.Join(cap.Profile.Comments, "\n"), "discarded sidecar records exact: 1") {
		t.Fatalf("discard not counted: %v", cap.Profile.Comments)
	}
}

func TestCaptureResilience_DiscardUnitsDoNotInventStackLosses(t *testing.T) {
	var out bytes.Buffer
	r, err := resilienceCapture(t, "good", context.Background(), &out)
	if err != nil {
		t.Fatal(err)
	}
	path := fmt.Sprintf("%s.%d", filepath.Join(r.ScratchDir, cudacapture.ExternalStacksName), r.TargetPID)
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		t.Fatal(err)
	}
	checkpoint := stackSidecarLine{Version: 1, Kind: "losses", Process: uint64(r.TargetPID), Epoch: r.Session, Losses: "5", LowerBound: true}
	b, _ := json.Marshal(checkpoint)
	f.Write(append(b, '\n'))
	f.WriteString("{\"version\":7,\"kind\":\"header\"}\n")
	f.Close()
	table, got := resilienceTable(t, r)
	if len(table.Stacks) != 1 || table.Losses != 5 || table.DiscardedRecords != 1 || table.DiscardedRecordsLowerBound || !got.damaged {
		t.Fatalf("mixed units or discarded retained row: %+v", table)
	}
}

func TestCaptureResilience_InvalidMetadataCannotCreateEmptySuccess(t *testing.T) {
	for _, losses := range []string{"invalid", "18446744073709551616"} {
		t.Run(losses, func(t *testing.T) {
			data := []byte(fmt.Sprintf("{\"version\":1,\"kind\":\"header\",\"losses\":%q}\n", losses))
			table, damaged, err := decodeStackSidecar(data, 0)
			if err == nil || !damaged || table.Losses != 0 || table.DiscardedRecords != 1 || len(table.Stacks) != 0 {
				t.Fatalf("invalid metadata admitted: table=%+v damaged=%v err=%v", table, damaged, err)
			}
		})
	}
}

func TestCaptureResilience_UnreadFileHasUnknownRecordCount(t *testing.T) {
	base := filepath.Join(t.TempDir(), cudacapture.ExternalStacksName)
	if err := os.Mkdir(base+".9", 0700); err != nil {
		t.Fatal(err)
	}
	got, err := readExternalStackSidecar(base, 9)
	if err != nil {
		t.Fatal(err)
	}
	var table cudaprof.ExternalStackTable
	if err := json.Unmarshal(got.data, &table); err != nil {
		t.Fatal(err)
	}
	if !got.damaged || len(got.damageFiles) != 1 || table.DiscardedRecords != 0 || !table.DiscardedRecordsLowerBound || table.Losses != 0 {
		t.Fatalf("unknown file invented counts: %+v damage=%+v", table, got)
	}
}

func TestCaptureResilience_AllReadsConsumeAggregateBudget(t *testing.T) {
	for _, mode := range []string{"partial-error", "post-stat-growth"} {
		t.Run(mode, func(t *testing.T) {
			base := filepath.Join(t.TempDir(), cudacapture.ExternalStacksName)
			good := "{\"version\":1,\"kind\":\"stack\",\"process\":1,\"external_id\":1,\"frames\":[\"good\"]}\n"
			if err := os.WriteFile(base, []byte(good), 0600); err != nil {
				t.Fatal(err)
			}
			for i := 1; i <= 8; i++ {
				if err := os.WriteFile(fmt.Sprintf("%s.%d", base, i), []byte("x"), 0600); err != nil {
					t.Fatal(err)
				}
			}
			oldRead := readExternalStackFile
			t.Cleanup(func() { readExternalStackFile = oldRead })
			payload := make([]byte, maxStackSidecarBytes+1)
			var consumed int64
			var reads int
			readExternalStackFile = func(path string, limit int64) ([]byte, error) {
				if path == base {
					data, err := oldRead(path, limit)
					consumed += int64(len(data))
					return data, err
				}
				reads++
				if limit <= 0 || limit > int64(maxStackSidecarBytes+1) || consumed+limit > maxStackSidecarReadBytes {
					t.Fatalf("unbounded read allowance: consumed=%d limit=%d", consumed, limit)
				}
				consumed += limit
				if mode == "partial-error" {
					return payload[:limit], io.ErrUnexpectedEOF
				}
				// Grow only after the controller's size check, exercising the
				// real reader's cap rather than its pre-read stat guard.
				if err := os.Truncate(path, maxStackSidecarBytes+2); err != nil {
					t.Fatal(err)
				}
				return oldRead(path, limit)
			}
			got, err := readExternalStackSidecar(base, 1)
			if err != nil {
				t.Fatal(err)
			}
			var table cudaprof.ExternalStackTable
			if err := json.Unmarshal(got.data, &table); err != nil {
				t.Fatal(err)
			}
			if consumed != maxStackSidecarReadBytes || reads != 4 || !got.damaged || len(got.damageFiles) != 8 || len(table.Stacks) != 1 || table.Stacks[0].Frames[0] != "good" || table.DiscardedRecords != 0 || !table.DiscardedRecordsLowerBound {
				t.Fatalf("budget or intact process lost: consumed=%d reads=%d table=%+v damage=%v", consumed, reads, table, got.damageFiles)
			}
		})
	}
}
