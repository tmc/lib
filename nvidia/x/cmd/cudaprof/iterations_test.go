package main

import (
	"bytes"
	"io"
	"regexp"
	"strings"
	"testing"
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
	res, err := measureIterations(cap, regexp.MustCompile(`^Embed$`), rules, iterationsOptions{skip: 1, expect: 5})
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
	_, err := measureIterations(cap, regexp.MustCompile(`^Embed$`), nil, iterationsOptions{skip: 1})
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
