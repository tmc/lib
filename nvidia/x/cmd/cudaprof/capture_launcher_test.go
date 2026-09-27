package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

func TestParseCaptureArgsPreservesTargetArgv(t *testing.T) {
	// The subject here is argv preservation past --, so the leading options
	// are ones this command implements: an option it does not is refused
	// during parsing, and a rejected parse would say nothing about argv.
	o, err := parseCaptureArgs([]string{"--trace", "--max-records=5000000", "--", "sh", "-c", "printf '%s\\n' \"$1\"", "sh", "-value with spaces"})
	if err != nil {
		t.Fatal(err)
	}
	if !o.Trace || o.MaxJournalRecords != 5000000 {
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
	o := captureOptions{command: []string{os.Args[0], "-test.run=TestCaptureHelper", "--", "-value"}, Dir: dir, Env: []string{"CUDAPROF_HELPER=1"}, Stdout: &stdout, Stderr: &stdout, NoInject: true}
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
	result, err := runCapture(context.Background(), captureOptions{command: []string{"true"}, NoInject: true})
	if err != nil {
		t.Fatal(err)
	}
	if result.TargetExitCode != 0 || result.Outcome != captureOutcomeNoCUDA || result.CaptureSuccess || result.BootstrapSeen {
		t.Fatalf("result = %+v", result)
	}
}

func TestRunCaptureFailsClosedWithoutSpawningWhenInjectMissing(t *testing.T) {
	sentinel := filepath.Join(t.TempDir(), "should_not_exist")
	result, err := runCapture(context.Background(), captureOptions{command: []string{"touch", sentinel}})
	if err != nil {
		t.Fatal(err)
	}
	if result.CaptureSuccess {
		t.Fatalf("expected failure, got success: %+v", result)
	}
	if !strings.Contains(result.CaptureError, "no injection library configured") {
		t.Fatalf("unexpected error message: %q", result.CaptureError)
	}
	if _, err := os.Stat(sentinel); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("target was spawned despite missing injection library: %v", err)
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
			result, err := runCapture(context.Background(), captureOptions{command: tt.command, Env: []string{"CUDAPROF_HELPER_MODE=" + tt.name}, DrainTimeout: time.Second, NoInject: true})
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
		r, err := runCapture(ctx, captureOptions{command: []string{os.Args[0], "-test.run=TestCaptureHelper"}, Env: []string{"CUDAPROF_HELPER_MODE=sleep"}, DrainTimeout: 100 * time.Millisecond, NoInject: true})
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

func TestParseCaptureArgsNoInjectAndJSON(t *testing.T) {
	o, err := parseCaptureArgs([]string{"--no-inject", "--json", "--", "true"})
	if err != nil {
		t.Fatal(err)
	}
	if !o.NoInject {
		t.Error("expected NoInject to be true")
	}
	if !o.JSON {
		t.Error("expected JSON to be true")
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

// TestParseCaptureArgsCollectionPolicy covers the three flags that configure
// the agent rather than the launcher. Both the separated and the =-joined form
// are exercised, as --max-records already is: an option the neighbours accept
// in one spelling and not the other is a trap.
func TestParseCaptureArgsCollectionPolicy(t *testing.T) {
	for _, tc := range []struct {
		name  string
		args  []string
		want  captureOptions
		wantE string
	}{
		{name: "range joined", args: []string{"--capture-range=cudaProfilerApi", "--", "true"}, want: captureOptions{CaptureRange: "cudaProfilerApi"}},
		{name: "range separated", args: []string{"--capture-range", "none", "--", "true"}, want: captureOptions{CaptureRange: "none"}},
		{name: "range single dash", args: []string{"-capture-range=cudaProfilerApi", "--", "true"}, want: captureOptions{CaptureRange: "cudaProfilerApi"}},
		{name: "finalize", args: []string{"--capture-range=cudaProfilerApi", "--finalize=profiler-stop", "--", "true"}, want: captureOptions{CaptureRange: "cudaProfilerApi", Finalize: "profiler-stop"}},
		{name: "finalize exit", args: []string{"--finalize=exit", "--", "true"}, want: captureOptions{Finalize: "exit"}},
		{name: "flush interval", args: []string{"--flush-interval=250ms", "--", "true"}, want: captureOptions{FlushInterval: 250 * time.Millisecond}},
		{name: "flush interval separated", args: []string{"--flush-interval", "1s", "--", "true"}, want: captureOptions{FlushInterval: time.Second}},
		{name: "recover", args: []string{"--recover-unfinalized", "--", "true"}, want: captureOptions{RecoverUnfinalized: true}},
		{name: "bad range", args: []string{"--capture-range=nvtx", "--", "true"}, wantE: "invalid capture range"},
		{name: "bad finalize", args: []string{"--finalize=sigterm", "--", "true"}, wantE: "invalid finalize trigger"},
		{name: "bad flush", args: []string{"--flush-interval=0", "--", "true"}, wantE: "invalid --flush-interval"},
		// Below the agent's resolution the interval rounds to zero, which is
		// indistinguishable from not asking for a periodic flush at all.
		{name: "sub-millisecond flush", args: []string{"--flush-interval=100us", "--", "true"}, wantE: "below the agent's 1ms resolution"},
		// The two profiler-API options are independent requests. A target that
		// calls only cuProfilerStop, to hand the agent the flush its exit path
		// will never run, wants the trigger without the gating, so this is a
		// supported configuration and not a mistake to refuse.
		{name: "finalize without range", args: []string{"--finalize=profiler-stop", "--", "true"}, want: captureOptions{Finalize: "profiler-stop"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			o, err := parseCaptureArgs(tc.args)
			if tc.wantE != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantE) {
					t.Fatalf("parseCaptureArgs(%v) error = %v, want one naming %q", tc.args, err, tc.wantE)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseCaptureArgs(%v): %v", tc.args, err)
			}
			if o.CaptureRange != tc.want.CaptureRange || o.Finalize != tc.want.Finalize || o.FlushInterval != tc.want.FlushInterval || o.RecoverUnfinalized != tc.want.RecoverUnfinalized {
				t.Errorf("parseCaptureArgs(%v) = range %q finalize %q flush %v recover %v, want range %q finalize %q flush %v recover %v",
					tc.args, o.CaptureRange, o.Finalize, o.FlushInterval, o.RecoverUnfinalized,
					tc.want.CaptureRange, tc.want.Finalize, tc.want.FlushInterval, tc.want.RecoverUnfinalized)
			}
		})
	}
}

// TestCaptureEnvironmentExportsCollectionPolicy is the half that matters: the
// flags configure the agent, and the agent only ever learns about them through
// these three variables. A flag parsed and not exported does nothing, which is
// the state CUDAPROF_CAPTURE_RANGE support was in -- the agent has read it
// since it was written and no command line could set it.
func TestCaptureEnvironmentExportsCollectionPolicy(t *testing.T) {
	o := captureOptions{Env: []string{}, CaptureRange: "cudaProfilerApi", Finalize: "profiler-stop", FlushInterval: 250 * time.Millisecond}
	env := captureEnvironment(o, "session", "dir")
	for _, tc := range []struct{ key, want string }{
		{"CUDAPROF_CAPTURE_RANGE", "cudaProfilerApi"},
		{"CUDAPROF_FINALIZE", "profiler-stop"},
		{"CUDAPROF_FLUSH_INTERVAL_MS", "250"},
		{"CUDAPROF_CORRELATE", "1"},
		{"CUDAPROF_EXTERNAL_STACKS", filepath.Join("dir", cudacapture.ExternalStacksName)},
	} {
		if got := lookupEnv(env, tc.key); got != tc.want {
			t.Errorf("%s = %q, want %q", tc.key, got, tc.want)
		}
	}
}

func TestParseCaptureArgsCrashTrace(t *testing.T) {
	tests := []struct {
		args    []string
		want    int
		wantErr bool
	}{
		{args: []string{"--", "true"}, want: 0},
		{args: []string{"--crash-trace=64", "--", "true"}, want: 64},
		{args: []string{"--crash-trace", "4096", "--", "true"}, want: 4096},
		{args: []string{"--crash-trace=0", "--", "true"}, wantErr: true},
		{args: []string{"--crash-trace=4097", "--", "true"}, wantErr: true},
		{args: []string{"--crash-trace=x", "--", "true"}, wantErr: true},
	}
	for _, tt := range tests {
		o, err := parseCaptureArgs(tt.args)
		if (err != nil) != tt.wantErr {
			t.Errorf("parseCaptureArgs(%q) error = %v, want error %v", tt.args, err, tt.wantErr)
			continue
		}
		if err != nil {
			continue
		}
		if o.CrashTrace != tt.want {
			t.Errorf("parseCaptureArgs(%q).CrashTrace = %d, want %d", tt.args, o.CrashTrace, tt.want)
		}
		env := captureEnvironment(o, "session", "dir")
		want := ""
		if tt.want > 0 {
			want = strconv.Itoa(tt.want)
		}
		if got := lookupEnv(env, "CUDAPROF_CRASH_TRACE"); got != want {
			t.Errorf("parseCaptureArgs(%q): CUDAPROF_CRASH_TRACE = %q, want %q", tt.args, got, want)
		}
	}
}

// TestCaptureEnvironmentOmitsUnsetCollectionPolicy: an unset flag leaves the
// caller's own environment alone, so a value exported by a wrapper script
// survives, and --capture-range=none is how it is turned back off. Exporting an
// empty value instead would reach the agent as a mode name it does not know.
func TestCaptureEnvironmentOmitsUnsetCollectionPolicy(t *testing.T) {
	env := captureEnvironment(captureOptions{Env: []string{"CUDAPROF_CAPTURE_RANGE=cudaProfilerApi"}}, "session", "dir")
	if got := lookupEnv(env, "CUDAPROF_CAPTURE_RANGE"); got != "cudaProfilerApi" {
		t.Errorf("CUDAPROF_CAPTURE_RANGE = %q, want the caller's value", got)
	}
	for _, key := range []string{"CUDAPROF_FINALIZE", "CUDAPROF_FLUSH_INTERVAL_MS"} {
		if got := lookupEnv(env, key); got != "" {
			t.Errorf("%s = %q, want unset", key, got)
		}
	}
	env = captureEnvironment(captureOptions{Env: []string{"CUDAPROF_CAPTURE_RANGE=cudaProfilerApi"}, CaptureRange: captureRangeNone}, "session", "dir")
	if got := lookupEnv(env, "CUDAPROF_CAPTURE_RANGE"); got != captureRangeNone {
		t.Errorf("CUDAPROF_CAPTURE_RANGE = %q, want none to override the environment", got)
	}
}

func TestParseCaptureArgsRequireComplete(t *testing.T) {
	o, err := parseCaptureArgs([]string{"--require-complete", "--", "true"})
	if err != nil {
		t.Fatal(err)
	}
	if !o.RequireComplete {
		t.Fatal("RequireComplete was not set")
	}
}

// TestTargetEnvironmentDiscriminatesArmsWithoutDisclosingThem covers the
// stamp that answers "which arm was this".
//
// Two arms of one program can differ only in an environment variable, produce
// different output, and pass every gate the capture applies. Nothing in the
// capture said which arm it profiled, and a reader lost a day to it. The
// digest is what discriminates -- including on a variable nobody thought to
// name in advance, which is the case that bites -- and it must do that
// without putting any value in an artifact people attach to bug reports.
func TestTargetEnvironmentDiscriminatesArmsWithoutDisclosingThem(t *testing.T) {
	base := []string{"PATH=/usr/bin", "SECRET_TOKEN=hunter2", "DS4_PHASE=decode"}
	other := []string{"PATH=/usr/bin", "SECRET_TOKEN=hunter2", "DS4_PHASE=prefill"}

	a := stampTargetEnvironment([]string{"./ds4", "-n", "1"}, "/work", base, nil)
	b := stampTargetEnvironment([]string{"./ds4", "-n", "1"}, "/work", other, nil)
	if a.EnvDigest == "" {
		t.Fatal("no digest recorded")
	}
	if a.EnvDigest == b.EnvDigest {
		t.Error("two arms differing in one variable share a digest, so the digest cannot tell them apart")
	}
	// Order is not part of the environment's identity.
	shuffled := []string{"DS4_PHASE=decode", "PATH=/usr/bin", "SECRET_TOKEN=hunter2"}
	if c := stampTargetEnvironment([]string{"./ds4", "-n", "1"}, "/work", shuffled, nil); c.EnvDigest != a.EnvDigest {
		t.Error("the same environment in a different order produced a different digest")
	}

	// Nothing is disclosed unless it was named.
	raw, err := json.Marshal(a)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(raw, []byte("hunter2")) || bytes.Contains(raw, []byte("decode")) {
		t.Errorf("an unnamed environment value was written into the capture: %s", raw)
	}
	if !bytes.Contains(raw, []byte(`"./ds4"`)) {
		t.Errorf("the command was not recorded: %s", raw)
	}

	// Naming a variable discloses it, and only it.
	named := stampTargetEnvironment([]string{"./ds4"}, "/work", base, []string{"DS4_*", "NOT_SET_HERE"})
	if named.Env["DS4_PHASE"] != "decode" {
		t.Errorf("named variable not recorded: %+v", named.Env)
	}
	if _, ok := named.Env["SECRET_TOKEN"]; ok {
		t.Errorf("an unnamed variable was disclosed: %+v", named.Env)
	}
	// A variable that was not set is absent, not empty: "" is a value a
	// program can branch on differently than unset, so the two must not read
	// the same in the capture.
	if _, ok := named.Env["NOT_SET_HERE"]; ok {
		t.Errorf("an unset variable was recorded as a value: %+v", named.Env)
	}
	if !slices.Contains(named.EnvUnset, "NOT_SET_HERE") {
		t.Errorf("an unset named variable was not reported as unset: %+v", named.EnvUnset)
	}
}

// TestParseCaptureArgsProjectionPolicy covers the two flags whose default is
// on. A boolean that is only ever spelled bare cannot be turned off, which is
// the state --follow-children was in, so both spellings are exercised here
// along with the single-dash form the neighbouring options accept.
func TestParseCaptureArgsProjectionPolicy(t *testing.T) {
	for _, tc := range []struct {
		name    string
		args    []string
		follow  bool
		api     bool
		apiSet  bool
		exclude []string
		wantE   string
	}{
		{name: "defaults", args: []string{"--", "true"}, follow: true, api: true},
		{name: "follow bare", args: []string{"--follow-children", "--", "true"}, follow: true, api: true},
		{name: "follow off", args: []string{"--follow-children=false", "--", "true"}, api: true},
		{name: "follow single dash", args: []string{"-follow-children=0", "--", "true"}, api: true},
		{name: "api bare", args: []string{"--api-activity", "--", "true"}, follow: true, api: true, apiSet: true},
		{name: "api off", args: []string{"--api-activity=false", "--", "true"}, follow: true, apiSet: true},
		{name: "api single dash", args: []string{"-api-activity=0", "--", "true"}, follow: true, apiSet: true},
		{name: "exclude repeats", args: []string{"--exclude=sgemm", "--exclude", "elementwise", "--", "true"}, follow: true, api: true, exclude: []string{"sgemm", "elementwise"}},
		{name: "bad bool", args: []string{"--follow-children=sometimes", "--", "true"}, wantE: `invalid --follow-children "sometimes"`},
		{name: "empty exclude", args: []string{"--exclude=", "--", "true"}, wantE: "non-empty value"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			o, err := parseCaptureArgs(tc.args)
			if tc.wantE != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantE) {
					t.Fatalf("parseCaptureArgs(%v) error = %v, want one naming %q", tc.args, err, tc.wantE)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseCaptureArgs(%v): %v", tc.args, err)
			}
			if o.FollowChildren != tc.follow || o.APIActivity != tc.api || o.apiActivitySet != tc.apiSet {
				t.Errorf("parseCaptureArgs(%v) = follow %v api %v set %v, want follow %v api %v set %v",
					tc.args, o.FollowChildren, o.APIActivity, o.apiActivitySet, tc.follow, tc.api, tc.apiSet)
			}
			if !reflect.DeepEqual(o.Exclude, tc.exclude) {
				t.Errorf("parseCaptureArgs(%v) exclude = %#v, want %#v", tc.args, o.Exclude, tc.exclude)
			}
		})
	}
}

// TestCaptureEnvironmentExportsAPIActivity is the half that matters for
// --api-activity: the agent learns of it only through CUDAPROF_API_ACTIVITY,
// and it reads that variable as "off only when it is exactly 0". An untouched
// flag must leave the caller's own value alone, and an explicit --api-activity
// must be able to override a 0 the surrounding environment set.
func TestCaptureEnvironmentExportsAPIActivity(t *testing.T) {
	for _, tc := range []struct {
		name string
		o    captureOptions
		want string
	}{
		{"unset", captureOptions{Env: []string{}, APIActivity: true}, ""},
		{"off", captureOptions{Env: []string{}, apiActivitySet: true}, "0"},
		{"on", captureOptions{Env: []string{}, APIActivity: true, apiActivitySet: true}, "1"},
		{"inherits", captureOptions{Env: []string{"CUDAPROF_API_ACTIVITY=0"}, APIActivity: true}, "0"},
		{"overrides", captureOptions{Env: []string{"CUDAPROF_API_ACTIVITY=0"}, APIActivity: true, apiActivitySet: true}, "1"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := lookupEnv(captureEnvironment(tc.o, "session", "dir"), "CUDAPROF_API_ACTIVITY"); got != tc.want {
				t.Errorf("CUDAPROF_API_ACTIVITY = %q, want %q", got, tc.want)
			}
		})
	}
}

// TestCaptureEnvironmentExportsMemsets pins memsets as opt-in: the launcher
// always exports CUDAPROF_MEMSETS, 1 only for --memsets, so a value the
// caller's environment carries cannot turn the kind on or off.
func TestCaptureEnvironmentExportsMemsets(t *testing.T) {
	for _, tc := range []struct {
		name string
		args []string
		env  []string
		want string
	}{
		{"default", nil, nil, "0"},
		{"off", []string{"--memsets=false"}, nil, "0"},
		{"on", []string{"--memsets"}, nil, "1"},
		{"single dash", []string{"-memsets=true"}, nil, "1"},
		{"inherited on", nil, []string{"CUDAPROF_MEMSETS=1"}, "0"},
		{"inherited off", []string{"--memsets"}, []string{"CUDAPROF_MEMSETS=0"}, "1"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			o, err := parseCaptureArgs(append(append([]string(nil), tc.args...), "--", "true"))
			if err != nil {
				t.Fatalf("parseCaptureArgs(%q): %v", tc.args, err)
			}
			o.Env = append([]string{}, tc.env...)
			if got := lookupEnv(captureEnvironment(o, "session", "dir"), "CUDAPROF_MEMSETS"); got != tc.want {
				t.Errorf("CUDAPROF_MEMSETS = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestParseCaptureArgsInjectMode(t *testing.T) {
	for _, tc := range []struct {
		args []string
		want string
		bad  bool
	}{
		{args: []string{"--", "true"}},
		{args: []string{"--inject-mode=cuda-injection", "--", "true"}, want: injectModeCUDA},
		{args: []string{"-inject-mode", "preload", "--", "true"}, want: injectModePreload},
		{args: []string{"--inject-mode=ld-preload", "--", "true"}, bad: true},
	} {
		o, err := parseCaptureArgs(tc.args)
		if tc.bad {
			if err == nil {
				t.Errorf("parseCaptureArgs(%q) accepted", tc.args)
			}
			continue
		}
		if err != nil {
			t.Fatalf("parseCaptureArgs(%q): %v", tc.args, err)
		}
		if o.InjectMode != tc.want {
			t.Errorf("parseCaptureArgs(%q) InjectMode = %q, want %q", tc.args, o.InjectMode, tc.want)
		}
	}
}

func TestCaptureEnvironmentCUDAInjection(t *testing.T) {
	base := []string{"LD_PRELOAD=libother.so", "DYLD_INSERT_LIBRARIES=libother.dylib"}
	env := captureEnvironment(captureOptions{Env: base, Inject: "agent.so", InjectMode: injectModeCUDA}, "session", "dir")
	path := lookupEnv(env, "CUDA_INJECTION64_PATH")
	if !filepath.IsAbs(path) || filepath.Base(path) != "agent.so" {
		t.Errorf("CUDA_INJECTION64_PATH = %q, want an absolute path to agent.so", path)
	}
	for _, key := range []string{"LD_PRELOAD", "DYLD_INSERT_LIBRARIES"} {
		if got, want := lookupEnv(env, key), lookupEnv(base, key); got != want {
			t.Errorf("%s = %q, want the caller's %q untouched", key, got, want)
		}
	}
	env = captureEnvironment(captureOptions{Env: []string{}, Inject: "agent.so", InjectMode: injectModePreload}, "session", "dir")
	if v := lookupEnv(env, "CUDA_INJECTION64_PATH"); v != "" {
		t.Errorf("preload set CUDA_INJECTION64_PATH = %q", v)
	}
}

func TestRunCaptureCUDAInjection(t *testing.T) {
	lib := filepath.Join(t.TempDir(), "libcapture.so")
	if err := os.WriteFile(lib, []byte("test fixture"), 0600); err != nil {
		t.Fatal(err)
	}

	// A target that never calls cuInit never loads the agent.
	result, err := runCapture(context.Background(), captureOptions{command: []string{"true"}, Inject: lib, InjectMode: injectModeCUDA})
	if err != nil {
		t.Fatal(err)
	}
	if result.InjectMode != injectModeCUDA || result.Outcome != captureOutcomeNoCUDA || result.CaptureSuccess || !strings.Contains(result.CaptureError, "cuInit") {
		t.Fatalf("result = %+v", result)
	}

	// The driver loads one injection library, so one already set is refused
	// before the target runs.
	sentinel := filepath.Join(t.TempDir(), "should_not_exist")
	env := append(os.Environ(), "CUDA_INJECTION64_PATH=/opt/other/libtool.so")
	_, err = runCapture(context.Background(), captureOptions{command: []string{"touch", sentinel}, Env: env, Inject: lib, InjectMode: injectModeCUDA})
	if err == nil || !strings.Contains(err.Error(), "/opt/other/libtool.so") {
		t.Fatalf("runCapture error = %v, want one naming the existing CUDA_INJECTION64_PATH", err)
	}
	if _, err := os.Stat(sentinel); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("target ran despite the refusal: %v", err)
	}
}
