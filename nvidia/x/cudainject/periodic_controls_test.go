package cudainject_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudainject"
)

// TestPeriodicCPUChild produces explicitly synthetic protocol rows using its
// actual child PID. No CUDA, native agent, driver or physical receipt is used.
func TestPeriodicCPUChild(t *testing.T) {
	mode := os.Getenv("CUDAINJECT_CPU_CHILD")
	if mode == "" {
		return
	}
	path := os.Getenv("CUDAINJECT_CPU_JOURNAL")
	file, err := os.Create(path)
	if err != nil {
		os.Exit(2)
	}
	identity := cudainject.Identity{Process: uint64(os.Getpid()), Generation: 1}
	for i := range identity.Session {
		identity.Session[i] = 1
	}
	if mode == "foreign-pid" {
		identity.Process++
	}
	sequence := uint64(0)
	write := func(kind uint16, payload string) {
		sequence++
		if cudainject.WriteRecord(file, cudainject.Record{Kind: kind, Identity: identity, Sequence: sequence, Payload: []byte(payload)}, cudainject.Limits{}) != nil {
			os.Exit(2)
		}
	}
	write(cudainject.KindBootstrap, `{}`)
	write(cudainject.KindCapability, `{"flush_interval_ms":5,"driver_api":true,"api_records":true}`)
	kernel := periodicKernel
	if mode == "wrong-kernel" {
		kernel = "foreign"
	}
	for i := 1; i <= 32; i++ {
		write(cudainject.KindAPI, fmt.Sprintf(`{"domain":"driver","thread_id":17,"callback_id":1,"correlation":%d}`, i))
		write(cudainject.KindActivity, fmt.Sprintf(`{"subtype":"kernel","name":%q,"valid":true,"start":1,"end":2,"correlation":%d}`, kernel, i))
	}
	if file.Close() != nil {
		os.Exit(2)
	}
	fmt.Print(periodicHeader(os.Getpid()))
	if mode == "exit-before-release" {
		os.Exit(0)
	}
	var release [1]byte
	if _, err := io.ReadFull(os.Stdin, release[:]); err != nil || release[0] != 'x' {
		os.Exit(3)
	}
	if mode != "missing-ack" {
		fmt.Println("released")
	}
	file, err = os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		os.Exit(2)
	}
	write(cudainject.KindFinalized, `{"state":"collected","dropped":0,"native_dropped":0,"queue_dropped":0}`)
	if file.Close() != nil {
		os.Exit(2)
	}
	os.Exit(0)
}

func TestPeriodicActualCPUChildProof(t *testing.T) {
	for _, mode := range []string{"positive", "foreign-pid", "wrong-kernel", "exit-before-release", "missing-ack", "retention-fail"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()
			executable, err := os.Executable()
			if err != nil {
				t.Fatal(err)
			}
			journal := filepath.Join(t.TempDir(), "journal")
			command := exec.CommandContext(ctx, executable, "-test.run=^TestPeriodicCPUChild$")
			command.Env = append(os.Environ(), "CUDAINJECT_CPU_CHILD="+mode, "CUDAINJECT_CPU_JOURNAL="+journal)
			output, stderr := &periodicOutput{remaining: 64 << 10}, &periodicOutput{remaining: 64 << 10}
			command.Stdout, command.Stderr = output, stderr
			command.WaitDelay = time.Second
			input, err := command.StdinPipe()
			if err != nil {
				t.Fatal(err)
			}
			if err := command.Start(); err != nil {
				t.Fatal(err)
			}
			pid := command.Process.Pid
			found := false
			deadline := time.Now().Add(time.Second)
			for time.Now().Before(deadline) {
				r, err := cudainject.ReadJournal(journal, cudainject.Limits{})
				_, outputErr := periodicWorkerOutput(output.Bytes(), pid, false)
				if err == nil && outputErr == nil && periodicEvidence(r, pid, false) == nil {
					found = true
					break
				}
				time.Sleep(2 * time.Millisecond)
			}
			_, finishErr := finishPeriodicWorker(found, pid, input, cancel, command.Wait, output, stderr, func(name string, b []byte) error {
				if mode == "retention-fail" {
					return errors.New("declared CPU retention failure")
				}
				return os.WriteFile(filepath.Join(filepath.Dir(journal), name), b, 0600)
			})
			r, readErr := cudainject.ReadJournal(journal, cudainject.Limits{})
			accepted := errors.Join(finishErr, readErr) == nil && periodicEvidence(r, pid, true) == nil
			if accepted != (mode == "positive") {
				t.Fatalf("accepted=%v found=%v finish=%v stdout=%q", accepted, found, finishErr, output.Bytes())
			}
			if command.ProcessState == nil || !command.ProcessState.Exited() {
				t.Fatal("CPU child not waited/reaped")
			}
		})
	}
}

type periodicRelease struct {
	writeErr, closeErr error
	short              bool
}

func (p periodicRelease) Write([]byte) (int, error) {
	if p.short {
		return 0, nil
	}
	if p.writeErr != nil {
		return 0, p.writeErr
	}
	return 1, nil
}
func (p periodicRelease) Close() error { return p.closeErr }

func TestPeriodicReleaseFailures(t *testing.T) {
	for _, p := range []periodicRelease{{writeErr: io.ErrClosedPipe}, {closeErr: errors.New("close refused")}, {short: true}} {
		cancelled, waited := false, false
		err := releasePeriodicWorker(p, func() { cancelled = true }, func() error { waited = true; return nil })
		if err == nil || !cancelled || !waited {
			t.Fatalf("release failure lost: %v cancelled=%v waited=%v", err, cancelled, waited)
		}
	}
}

func TestPeriodicExitedChildReleaseRefused(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	command := exec.CommandContext(ctx, executable, "-test.run=^TestPeriodicCPUChild$")
	command.Env = append(os.Environ(), "CUDAINJECT_CPU_CHILD=exit-before-release", "CUDAINJECT_CPU_JOURNAL="+filepath.Join(t.TempDir(), "journal"))
	output := &periodicOutput{remaining: 64 << 10}
	command.Stdout, command.Stderr = output, output
	input, err := command.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := command.Start(); err != nil {
		t.Fatal(err)
	}
	pid := command.Process.Pid
	workerErr := command.Wait()
	if workerErr != nil || command.ProcessState == nil || !command.ProcessState.Exited() {
		t.Fatalf("actual exit0 child: %v", workerErr)
	}
	// An already completed process can have Wait=nil despite failed release.
	// Preserve that actual status while exercising the same release helper.
	err = releasePeriodicWorker(input, cancel, func() error { return workerErr })
	if err == nil {
		t.Fatal("already-exited child admitted after failed release")
	}
	t.Logf("CPU child pid=%d exited/reaped=true wait=<nil> release=%v", pid, err)
}

func TestPeriodicRetentionAndOutputBounds(t *testing.T) {
	for _, failed := range []string{"worker.stdout", "worker.stderr"} {
		err := retainPeriodicOutput(func(name string, _ []byte) error {
			if name == failed {
				return errors.New("retention refused")
			}
			return nil
		}, nil, nil)
		if err == nil {
			t.Fatal("retention failure lost")
		}
	}
	output := &periodicOutput{remaining: 3}
	if _, err := output.Write([]byte("four")); err == nil || len(output.Bytes()) > 3 {
		t.Fatal("output bound lost")
	}
	for _, raw := range []string{periodicHeader(17), strings.Replace(periodicHeader(17), "pid=17", "pid=18", 1), strings.Replace(periodicHeader(17), "01010101010101010101010101010101", "00000000000000000000000000000000", 1)} {
		_, err := periodicWorkerOutput([]byte(raw), 17, false)
		if (raw == periodicHeader(17)) != (err == nil) {
			t.Fatalf("identity accepted incorrectly: %q %v", raw, err)
		}
	}
}

func TestPeriodicEvidenceRefusals(t *testing.T) {
	// Source-level predicates are also tested with decoded, framed CPU rows by
	// the reexec fixture above. These extra controls isolate loss/capability.
	if err := periodicEvidence(cudainject.Recovery{}, 17, false); err == nil {
		t.Fatal("empty evidence accepted")
	}
}

func TestPeriodicWorkerSourceContract(t *testing.T) {
	for _, literal := range []string{`.visible .entry lane87_periodic()`, `i<32`, `puts("waiting-before-exit")`, `release!='x'`, `puts("released")`} {
		if !strings.Contains(string(periodicWorkerSource), literal) {
			t.Fatalf("fixed source lost %q", literal)
		}
	}
}
