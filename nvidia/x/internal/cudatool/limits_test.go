package cudatool

import (
	"context"
	"errors"
	"runtime"
	"testing"
	"time"
)

func TestOutputLimitCancelsAndRetainsPrefix(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX subprocess fixture")
	}
	for _, script := range []string{
		"while :; do printf 0123456789; done",
		"while :; do printf 0123456789 >&2; done",
		"while :; do printf 0123456789; printf abcdefghij >&2; done",
	} {
		out, err := RunWithLimits(context.Background(), Limits{MaxOutputBytes: 127, Timeout: 5 * time.Second}, "sh", "", nil, "-c", script)
		if !errors.Is(err, ErrOutputLimit) || !out.Truncated || len(out.Stdout)+len(out.Stderr) != 127 {
			t.Fatalf("output=%+v error=%v", out, err)
		}
		if out.Duration <= 0 || out.MaxOutputBytes != 127 {
			t.Fatal("missing execution receipt")
		}
	}
}

func TestOutputLimitBoundaryAndExitStatus(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX subprocess fixture")
	}
	out, err := RunWithLimits(context.Background(), Limits{MaxOutputBytes: 5}, "sh", "", nil, "-c", "printf abc; printf de >&2")
	if err != nil || out.Truncated || out.ExitCode != 0 || string(out.Stdout) != "abc" || string(out.Stderr) != "de" {
		t.Fatalf("boundary: %+v %v", out, err)
	}
	out, err = RunWithLimits(context.Background(), Limits{MaxOutputBytes: 5}, "sh", "", nil, "-c", "printf bad >&2; exit 7")
	if err == nil || out.ExitCode != 7 || out.Truncated || string(out.Stderr) != "bad" {
		t.Fatalf("failed process: %+v %v", out, err)
	}
}

func TestToolTimeoutAndCancellation(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX subprocess fixture")
	}
	for _, callerDeadline := range []bool{false, true} {
		ctx := context.Background()
		limit := 20 * time.Millisecond
		if callerDeadline {
			var cancel context.CancelFunc
			ctx, cancel = context.WithTimeout(ctx, limit)
			defer cancel()
			limit = time.Minute
		}
		out, err := RunWithLimits(ctx, Limits{Timeout: limit}, "sh", "", nil, "-c", "while :; do :; done")
		if !errors.Is(err, context.DeadlineExceeded) || out.Truncated {
			t.Fatalf("deadline: %+v %v", out, err)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	out, err := Run(ctx, "missing", "", nil)
	if !errors.Is(err, context.Canceled) || out.ExitCode != -1 {
		t.Fatalf("pre-cancel: %+v %v", out, err)
	}
	for _, limits := range []Limits{{MaxOutputBytes: -1}, {Timeout: -1}} {
		if _, err := RunWithLimits(context.Background(), limits, "missing", "", nil); err == nil {
			t.Fatal("accepted negative limit")
		}
	}
}
