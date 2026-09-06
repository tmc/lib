package cudasanitizer

import (
	"context"
	"errors"
	"os/exec"
	"testing"
	"time"
)

type failedCleanup struct{ closeErr error }

func (g failedCleanup) kill() error  { return nil }
func (g failedCleanup) close() error { return g.closeErr }

func TestProcessCleanupFailure(t *testing.T) {
	fault := errors.New("injected cleanup failure")
	cmd := exec.Command(fixtureTool(t))
	cmd.Env = fixtureEnv()
	cmd.WaitDelay = time.Second
	err := runContextWithStart(context.Background(), cmd, func(cmd *exec.Cmd) (processCleanup, error) {
		if err := cmd.Start(); err != nil {
			return nil, err
		}
		return failedCleanup{fault}, nil
	})
	result := &Result{Log: ParseLog(cleanReport), ExitCode: 0, ErrorExitCode: 66}
	result.classify(context.Background(), err)
	if result.Outcome != OutcomeToolFailure || !errors.Is(result.Err, fault) {
		t.Fatalf("cleanup failure classified as %v, error %v", result.Outcome, result.Err)
	}
}

func TestCleanupFailurePreservesCause(t *testing.T) {
	fault := errors.New("injected cleanup failure")
	exit := &exec.ExitError{}
	for _, tt := range []struct {
		name  string
		cause error
		want  error
	}{
		{"exit", exit, exit},
		{"deadline", context.DeadlineExceeded, ErrTimeout},
		{"cancel", context.Canceled, context.Canceled},
	} {
		t.Run(tt.name, func(t *testing.T) {
			result := &Result{Log: ParseLog(cleanReport), ExitCode: 1, ErrorExitCode: 66}
			result.classify(context.Background(), errors.Join(tt.cause, &cleanupError{fault}))
			if result.Outcome != OutcomeToolFailure || !errors.Is(result.Err, fault) || !errors.Is(result.Err, tt.want) || !errors.Is(result.Err, tt.cause) {
				t.Fatalf("outcome %v error %v; want cleanup failure and %v", result.Outcome, result.Err, tt.name)
			}
		})
	}
}

type failedKill struct {
	processCleanup
	fault error
}

func (g failedKill) kill() error { return errors.Join(g.processCleanup.kill(), g.fault) }

func TestProcessKillFailure(t *testing.T) {
	fault := errors.New("injected kill failure")
	cmd := exec.Command(fixtureTool(t))
	cmd.Env = fixtureEnv(fakeSleep + "=5s")
	cmd.WaitDelay = time.Second
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	err := runContextWithStart(ctx, cmd, func(cmd *exec.Cmd) (processCleanup, error) {
		group, err := startGroup(cmd)
		if err != nil {
			return nil, err
		}
		cancel()
		return failedKill{group, fault}, nil
	})
	if !errors.Is(err, fault) || !errors.Is(err, context.Canceled) {
		t.Fatalf("error %v, want kill failure and cancellation", err)
	}
	if cmd.ProcessState == nil {
		t.Fatal("process was not reaped")
	}
}
