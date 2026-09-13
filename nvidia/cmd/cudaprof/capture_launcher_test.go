package main

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestParseCaptureArgsPreservesTargetArgv(t *testing.T) {
	o, err := parseCaptureArgs([]string{"--deep", "--collect", "runtime", "--", "sh", "-c", "printf '%s\\n' \"$1\"", "sh", "-value with spaces"})
	if err != nil {
		t.Fatal(err)
	}
	if !o.Deep || !o.Trace || !reflect.DeepEqual(o.Collect, []string{"runtime"}) {
		t.Fatalf("options = %+v", o)
	}
	if !reflect.DeepEqual(o.command, []string{"sh", "-c", "printf '%s\\n' \"$1\"", "sh", "-value with spaces"}) {
		t.Fatalf("command = %#v", o.command)
	}
}

func TestParseCaptureArgsMaxRecords(t *testing.T) {
	for _, tc := range []struct {
		args []string
		want int64
	}{
		{[]string{"--max-records=5000000", "--", "true"}, 5000000},
		{[]string{"--max-records", "10000000", "--", "true"}, 10000000},
		{[]string{"-max-records=2000000", "--", "true"}, 2000000},
	} {
		o, err := parseCaptureArgs(tc.args)
		if err != nil {
			t.Fatalf("parseCaptureArgs(%v): %v", tc.args, err)
		}
		if o.MaxJournalRecords != tc.want {
			t.Errorf("parseCaptureArgs(%v) MaxJournalRecords = %d, want %d", tc.args, o.MaxJournalRecords, tc.want)
		}
	}
}

func TestParseCaptureArgsRejectsInvalidPlanBeforeSpawn(t *testing.T) {
	for _, args := range [][]string{{"--unknown", "--", "true"}, {"--timeout=0s", "--", "true"}} {
		o, err := parseCaptureArgs(args)
		_ = o
		if err == nil {
			t.Fatalf("parseCaptureArgs(%q) accepted", args)
		}
	}
}

func TestRunCapturePreservesCWDEnvAndStdio(t *testing.T) {
	dir := t.TempDir()
	var stdout bytes.Buffer
	o := captureOptions{command: []string{os.Args[0], "-test.run=TestCaptureHelper", "--", "-value"}, Dir: dir, Env: []string{"CUDAPROF_HELPER=1"}, Stdout: &stdout, Stderr: &stdout}
	result, err := runCapture(context.Background(), o)
	if err != nil {
		t.Fatal(err)
	}
	if result.TargetExitCode != 0 || result.TargetSignal != "" || result.Outcome != captureOutcomeNoCUDA || result.TransportReady || result.CaptureSuccess {
		t.Fatalf("result = %+v", result)
	}
	realDir, _ := filepath.EvalSymlinks(dir)
	if got := strings.TrimSpace(stdout.String()); !strings.HasPrefix(got, realDir+"|1|-value") {
		t.Fatalf("child output = %q", got)
	}
}

func TestRunCaptureSuccessfulChildWithoutBootstrapIsIncomplete(t *testing.T) {
	result, err := runCapture(context.Background(), captureOptions{command: []string{"true"}})
	if err != nil {
		t.Fatal(err)
	}
	if result.TargetExitCode != 0 || result.Outcome != captureOutcomeNoCUDA || result.CaptureSuccess || result.BootstrapSeen {
		t.Fatalf("result = %+v", result)
	}
}

func TestRunCaptureRejectsUnimplementedPolicyBeforeSpawn(t *testing.T) {
	for _, name := range []string{"deep", "exhaustive", "follow"} {
		t.Run(name, func(t *testing.T) {
			o := captureOptions{command: []string{"/path/that/must/not/spawn"}}
			switch name {
			case "deep":
				o.Deep = true
			case "exhaustive":
				o.Exhaustive = true
			case "follow":
				o.FollowChildren = true
			}
			if _, err := runCapture(context.Background(), o); err == nil {
				t.Fatal("accepted unimplemented policy")
			}
		})
	}
}

func TestRunCaptureInjectionWithoutBootstrapIsLoaderFailure(t *testing.T) {
	lib := filepath.Join(t.TempDir(), "libcapture.so")
	if err := os.WriteFile(lib, []byte("test fixture"), 0600); err != nil {
		t.Fatal(err)
	}
	result, err := runCapture(context.Background(), captureOptions{command: []string{"true"}, Inject: lib})
	if err != nil {
		t.Fatal(err)
	}
	if result.TargetExitCode != 0 || result.Outcome != captureOutcomeInjection || result.CaptureSuccess {
		t.Fatalf("result = %+v", result)
	}
}

func TestRunCaptureSeparatesTargetExitAndSignal(t *testing.T) {
	for _, tt := range []struct {
		name    string
		command []string
		exit    int
		signal  bool
	}{
		{"exit", []string{os.Args[0], "-test.run=TestCaptureHelper"}, 7, false},
		{"signal", []string{os.Args[0], "-test.run=TestCaptureHelper"}, -1, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			result, err := runCapture(context.Background(), captureOptions{command: tt.command, Env: []string{"CUDAPROF_HELPER_MODE=" + tt.name}, DrainTimeout: time.Second})
			if err != nil {
				t.Fatal(err)
			}
			if result.TargetExitCode != tt.exit || (result.TargetSignal == "terminated") != tt.signal {
				t.Fatalf("result = %+v", result)
			}
		})
	}
}

func TestRunCaptureCancellationKillsOwnedProcess(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	resultCh := make(chan struct {
		r   captureResult
		err error
	}, 1)
	go func() {
		r, err := runCapture(ctx, captureOptions{command: []string{os.Args[0], "-test.run=TestCaptureHelper"}, Env: []string{"CUDAPROF_HELPER_MODE=sleep"}, DrainTimeout: 100 * time.Millisecond})
		resultCh <- struct {
			r   captureResult
			err error
		}{r, err}
	}()
	time.Sleep(50 * time.Millisecond)
	cancel()
	got := <-resultCh
	if !got.r.Cancelled || got.err == nil {
		t.Fatalf("result=%+v err=%v", got.r, got.err)
	}
}

func TestCaptureHelper(t *testing.T) {
	if os.Getenv("CUDAPROF_PROTOCOL") == "" {
		return
	}
	switch os.Getenv("CUDAPROF_HELPER_MODE") {
	case "sleep":
		time.Sleep(10 * time.Second)
	case "signal":
		signalCaptureTestProcess()
	case "exit":
		os.Exit(7)
	default:
		cwd, _ := os.Getwd()
		fmt.Fprintf(os.Stdout, "%s|%s|%s", cwd, os.Getenv("CUDAPROF_HELPER"), os.Args[len(os.Args)-1])
	}
}
