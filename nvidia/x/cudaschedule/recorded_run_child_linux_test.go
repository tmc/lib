//go:build linux

package cudaschedule

import (
	"context"
	"os"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestRecordedSupervisorOwnsWorkerStatus(t *testing.T) {
	worker, err := recordedStartWorker("/bin/sh", []string{"-c", "sleep .02; printf literal-worker; exit 9"}, os.Environ(), t.TempDir(), 1024)
	if worker != nil {
		t.Cleanup(func() {
			if err := worker.signal(syscall.SIGKILL); err != nil {
				t.Errorf("worker cleanup: %v", err)
			}
			if err := worker.closePipes(); err != nil {
				t.Errorf("worker pipes: %v", err)
			}
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	if worker.identity.PID == os.Getpid() || worker.identity.Parent != os.Getpid() || worker.identity.Group != syscall.Getpgrp() {
		t.Fatalf("worker/supervisor identity conflated: %+v", worker.identity)
	}
	deadline := time.Now().Add(2 * time.Second)
	for (!worker.reaped || !worker.outEOF || !worker.errEOF) && time.Now().Before(deadline) {
		if err := worker.poll(); err != nil {
			t.Fatal(err)
		}
		time.Sleep(time.Millisecond)
	}
	if !worker.reaped || !worker.outEOF || !worker.errEOF || !worker.status.Exited() || worker.status.ExitStatus() != 9 {
		t.Fatalf("missing literal direct-child status/drain: reaped=%v EOF=%v/%v status=%v", worker.reaped, worker.outEOF, worker.errEOF, worker.status)
	}
	if output, overflow := worker.out.snapshot(); output != "literal-worker" || overflow {
		t.Fatalf("actual worker output %q %v", output, overflow)
	}
}

func TestRecordedChildStatusAndDrain(t *testing.T) {
	for _, tt := range []struct {
		name, script string
		code         int
		out, errOut  string
	}{
		{"nonzero", "printf actual-worker; printf actual-error >&2; exit 7", 7, "actual-worker", "actual-error"},
		{"silent", "sleep .05", 0, "", ""},
	} {
		t.Run(tt.name, func(t *testing.T) {
			child, err := recordedStartChild("/bin/sh", []string{"-c", tt.script}, os.Environ(), t.TempDir(), 1024)
			if child != nil {
				t.Cleanup(func() {
					ctx, cancel := context.WithTimeout(context.Background(), time.Second)
					defer cancel()
					if err := recordedRetire(ctx, []int{child.identity.Group}); err != nil {
						t.Errorf("child cleanup: %v", err)
					}
					if err := child.closePipes(); err != nil {
						t.Errorf("child pipes: %v", err)
					}
				})
			}
			if err != nil {
				t.Fatal(err)
			}
			for _, fd := range []int{child.stdoutFD, child.stderrFD} {
				flags, _, errno := syscall.Syscall(syscall.SYS_FCNTL, uintptr(fd), syscall.F_GETFL, 0)
				if errno != 0 || flags&syscall.O_NONBLOCK == 0 {
					t.Fatalf("blocking owned output descriptor: flags=%x errno=%v", flags, errno)
				}
			}
			deadline := time.Now().Add(3 * time.Second)
			for !child.reaped && time.Now().Before(deadline) {
				start := time.Now()
				if err := child.poll(); err != nil {
					t.Fatal(err)
				}
				if time.Since(start) > 100*time.Millisecond {
					t.Fatal("poll blocked on silent output")
				}
				time.Sleep(time.Millisecond)
			}
			if !child.reaped || !child.status.Exited() || child.status.ExitStatus() != tt.code {
				t.Fatalf("actual child status reaped=%v status=%v", child.reaped, child.status)
			}
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			if err := recordedRetire(ctx, []int{child.identity.Group}); err != nil {
				t.Fatal(err)
			}
			if err := child.poll(); err != nil {
				t.Fatal(err)
			}
			out, overflow := child.out.snapshot()
			errOut, errOverflow := child.errOut.snapshot()
			if out != tt.out || errOut != tt.errOut || overflow || errOverflow {
				t.Fatalf("lost actual output %q %q overflow=%v/%v", out, errOut, overflow, errOverflow)
			}
		})
	}
}

func TestRecordedChildFloodBoundAndRetirement(t *testing.T) {
	child, err := recordedStartChild("/bin/sh", []string{"-c", `trap '' TERM; while :; do printf 012345678901234567890123456789; done`}, os.Environ(), t.TempDir(), 64)
	if child != nil {
		t.Cleanup(func() {
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			if err := recordedRetire(ctx, []int{child.identity.Group}); err != nil {
				t.Errorf("child cleanup: %v", err)
			}
			if err := child.closePipes(); err != nil {
				t.Errorf("child pipes: %v", err)
			}
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if err := child.poll(); err != nil {
			t.Fatal(err)
		}
		if _, overflow := child.out.snapshot(); overflow {
			break
		}
		time.Sleep(time.Millisecond)
	}
	data, overflow := child.out.snapshot()
	if !overflow || len(data) != 64 || !strings.HasPrefix(data, "0123456789") {
		t.Fatalf("unbounded/lost output: %q %v", data, overflow)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := recordedRetire(ctx, []int{child.identity.Group}); err != nil {
		t.Fatal(err)
	}
	for !child.reaped && time.Now().Before(deadline) {
		if err := child.poll(); err != nil {
			t.Fatal(err)
		}
		time.Sleep(time.Millisecond)
	}
	if !child.reaped || !child.status.Signaled() || child.status.Signal() != syscall.SIGKILL {
		t.Fatalf("failed actual KILL/wait: %v %v", child.reaped, child.status)
	}
}
