package main

import (
	"bytes"
	"errors"
	"io"
	"reflect"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaprof/iteration"
)

// iterationRecs is n iterations of 1000ns: a marker kernel, a gemm and a
// memcpy, with the device idle for the last 500ns of each.
func iterationRecs(n int) []rec {
	var recs []rec
	for i := range uint64(n) {
		t := 1000 * (i + 1)
		recs = append(recs,
			rec{subtype: "kernel", name: "Embed", start: t, end: t + 100},
			rec{subtype: "kernel", name: "gemm_tf32", start: t + 100, end: t + 400},
			rec{subtype: "memcpy", start: t + 400, end: t + 500, bytes: 64},
		)
	}
	return recs
}

func TestMeasureIterations(t *testing.T) {
	cap := compareCapture(iterationRecs(5), completeStatus())
	rules := mustRules(t, "gemm gemm\nEmbed embed\n")
	res, err := measureIterations(cap, regexp.MustCompile(`^Embed$`), rules, iterationsOptions{skip: 1, expect: 5}, nil)
	if err != nil {
		t.Fatal(err)
	}
	s := res.Report.Steady
	if s.Iterations != 3 || s.WindowNanos != 3000 || s.BusyNanos != 1500 || s.IdleNanos != 1500 {
		t.Errorf("steady = %+v, want 3 iterations, window 3000, busy 1500, idle 1500", s)
	}
	var out bytes.Buffer
	if err := res.write(&out, 10); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"5 markers, 4 iterations, skip 1, 3 steady",
		"gate     pass",
		"0.001 ms",
		"gemm",
		"[idle] the device is idle 50.0% of each iteration (0.001 ms of 0.001 ms)",
	} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("output lacks %q:\n%s", want, out.String())
		}
	}
}

func TestMeasureIterationsRefuses(t *testing.T) {
	status := completeStatus()
	status["run_completeness"] = "partial"
	cap := compareCapture(iterationRecs(5), status)
	_, err := measureIterations(cap, regexp.MustCompile(`^Embed$`), nil, iterationsOptions{skip: 1}, nil)
	if err == nil || !strings.Contains(err.Error(), `run_completeness is "partial"`) {
		t.Errorf("err = %v, want a refusal of the partial run", err)
	}
}

func TestParseIterationsArgs(t *testing.T) {
	tests := []struct {
		name    string
		args    []string
		want    iterationsOptions
		wantErr string
	}{
		{"defaults", []string{"-iteration-marker=^Embed$", "c.pprof"}, iterationsOptions{marker: "^Embed$", skip: 1, kernels: 15}, ""},
		{"all", []string{"-iteration-marker", "step", "-nvtx", "-skip-iterations=2", "-expect-markers=11", "-json", "c.pprof"}, iterationsOptions{marker: "step", nvtx: true, skip: 2, expect: 11, json: true, kernels: 15}, ""},
		{"perf", []string{"-iteration-marker=x", "-perf=h.txt", "c.pprof"}, iterationsOptions{marker: "x", skip: 1, kernels: 15, perfPath: "h.txt", gaps: time.Millisecond}, ""},
		{"gaps", []string{"-iteration-marker=x", "-gaps=2us", "-perf=h.txt", "c.pprof"}, iterationsOptions{marker: "x", skip: 1, kernels: 15, perfPath: "h.txt", gaps: 2 * time.Microsecond}, ""},
		{"negative gaps", []string{"-iteration-marker=x", "-gaps=-1ms", "c.pprof"}, iterationsOptions{}, "invalid -gaps"},
		{"no marker", []string{"c.pprof"}, iterationsOptions{}, "-iteration-marker is required"},
		{"negative skip", []string{"-iteration-marker=x", "-skip-iterations=-1", "c.pprof"}, iterationsOptions{}, "invalid -skip-iterations -1"},
		{"two captures", []string{"-iteration-marker=x", "a.pprof", "b.pprof"}, iterationsOptions{}, "want one capture"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, path, err := parseIterationsArgs(tt.args, io.Discard)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("err = %v, want %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			got.maxRecords, got.journalBytes = 0, 0
			if got != tt.want || path != "c.pprof" {
				t.Errorf("got %+v %q, want %+v c.pprof", got, path, tt.want)
			}
		})
	}
}

func TestCaptureIterationGate(t *testing.T) {
	tests := []struct {
		name   string
		opts   iterationsOptions
		pass   bool
		stderr string
	}{
		{"pass", iterationsOptions{marker: `^Embed$`, skip: 1, expect: 5}, true,
			"iteration gate pass: 5 markers, 3 steady iterations; per iteration window 0.001 ms"},
		{"expect", iterationsOptions{marker: `^Embed$`, skip: 1, expect: 6}, false,
			"iteration gate FAIL: marker fired 5 times, want 6"},
		{"no marker", iterationsOptions{marker: `^Nope$`, skip: 1}, false, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cap := compareCapture(iterationRecs(5), completeStatus())
			var out bytes.Buffer
			err := captureIterationGate(&out, cap, tt.opts)
			if tt.pass != (err == nil) {
				t.Fatalf("err = %v, want pass %v", err, tt.pass)
			}
			if err != nil {
				var code *exitCodeError
				if !errors.As(err, &code) || code.Code != 2 {
					t.Errorf("err = %v, want exit status 2", err)
				}
			}
			if !strings.Contains(out.String(), tt.stderr) {
				t.Errorf("output = %q, want %q", out.String(), tt.stderr)
			}
		})
	}
}

func TestParseCaptureIterations(t *testing.T) {
	tests := []struct {
		name    string
		args    []string
		want    *iterationsOptions
		wantErr string
	}{
		{"none", []string{"-o", "c.pprof", "--", "t"}, nil, ""},
		{"marker", []string{"-o", "c.pprof", "--iteration-marker", "^gather", "--", "t"},
			&iterationsOptions{marker: "^gather", skip: 1}, ""},
		{"all", []string{"-o", "c.pprof", "--iteration-marker=^g", "--skip-iterations", "2", "--expect-markers", "10", "--", "t"},
			&iterationsOptions{marker: "^g", skip: 2, expect: 10}, ""},
		{"no output", []string{"--iteration-marker", "^g", "--", "t"}, nil, "needs -o"},
		{"no marker", []string{"-o", "c.pprof", "--skip-iterations", "2", "--", "t"}, nil, "need --iteration-marker"},
		{"bad regexp", []string{"-o", "c.pprof", "--iteration-marker", "(", "--", "t"}, nil, "invalid --iteration-marker"},
		{"bad skip", []string{"-o", "c.pprof", "--iteration-marker", "g", "--skip-iterations", "-1", "--", "t"}, nil, "invalid --skip-iterations"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			o, err := parseCaptureArgs(tt.args)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("err = %v, want %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(o.Iterations, tt.want) {
				t.Errorf("Iterations = %+v, want %+v", o.Iterations, tt.want)
			}
		})
	}
}

func TestIterationsGaps(t *testing.T) {
	var host []iteration.HostSample
	for i := range int64(5) {
		t := 1000*(i+1) + 600
		host = append(host,
			iteration.HostSample{Time: t, Thread: 1, Stack: []string{"tokenize", "main.run", "main.main"}},
			iteration.HostSample{Time: t + 1, Thread: 1, Stack: []string{"tokenize", "main.run", "main.main"}},
			iteration.HostSample{Time: t + 2, Thread: 2, Stack: []string{"runtime.gcBgMarkWorker"}},
		)
	}
	cap := compareCapture(iterationRecs(5), completeStatus())
	res, err := measureIterations(cap, regexp.MustCompile(`^Embed$`), nil, iterationsOptions{skip: 1, gaps: 100}, host)
	if err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := res.write(&out, 0); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"idle gaps (per steady iteration): 1.0 gaps, 0.001 ms of 0.001 ms idle, 0.0 host API calls begun in them, 3.0 host samples taken in them",
		"66.7%    3     tokenize < main.run < main.main",
		"33.3%    3     runtime.gcBgMarkWorker",
		"longest idle gaps (top 3 of 3)",
	} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("output lacks %q:\n%s", want, out.String())
		}
	}
}

func TestIterationsMemsetNote(t *testing.T) {
	tests := []struct {
		name       string
		capability any
		want       string
	}{
		{"collected", map[string]any{"memsets": true}, ""},
		{"no capability", nil, ""},
		{"not collected", map[string]any{"memsets": false}, "memset activity was not collected (capture --memsets=false), so the device time of every memset counts as idle: each steady iteration made 0.0 memset calls"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			status := completeStatus()
			if tt.capability != nil {
				status["capability"] = tt.capability
			}
			cap := compareCapture(iterationRecs(5), status)
			res, err := measureIterations(cap, regexp.MustCompile(`^Embed$`), nil, iterationsOptions{skip: 1}, nil)
			if err != nil {
				t.Fatal(err)
			}
			got := strings.Join(res.Notes, "\n")
			if tt.want == "" && got != "" || !strings.Contains(got, tt.want) {
				t.Errorf("notes = %q, want %q", got, tt.want)
			}
		})
	}
}
