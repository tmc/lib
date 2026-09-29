package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

func TestParseABArgs(t *testing.T) {
	base := []string{"-o", "d", "-base", "b", "-target", "t"}
	tests := []struct {
		name    string
		args    []string
		order   []string
		control bool
		capture []string
		compare []string
		err     string
	}{
		{name: "default", args: base, order: []string{"base", "target"}},
		{name: "control", args: append(base, "-control"), order: []string{"base", "target", "control"}, control: true},
		{name: "order", args: append(base, "-order=control,target,base"), order: []string{"control", "target", "base"}, control: true},
		{
			name:    "capture limits stay with capture",
			args:    append(base, "-compare", "-warmup-ms=5 -max-records=9", "--", "--inject", "a.so", "--max-records", "8388608", "--capture-file-bytes=1073741824"),
			order:   []string{"base", "target"},
			capture: []string{"--inject", "a.so", "--max-records", "8388608", "--capture-file-bytes=1073741824"},
			compare: []string{"-warmup-ms=5", "-max-records=9"},
		},
		{name: "no target", args: []string{"-o", "d", "-base", "b"}, err: "usage"},
		{name: "unknown arm", args: append(base, "-order=base,new"), err: `unknown arm "new"`},
		{name: "repeated arm", args: append(base, "-order=base,target,base"), err: "base appears twice"},
		{name: "order lacks target", args: append(base, "-order=base,control"), err: "must name base and target"},
		{name: "control not ordered", args: append(base, "-control", "-order=target,base"), err: "no control"},
		{name: "output in capture flags", args: append(base, "--", "-o", "x"), err: "set by ab"},
		{name: "bad compare flag", args: append(base, "-compare", "-nope"), err: "-compare"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts, err := parseABArgs(tt.args, io.Discard)
			if tt.err != "" {
				if err == nil || !strings.Contains(err.Error(), tt.err) {
					t.Fatalf("err = %v, want %q", err, tt.err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(opts.order, tt.order) || opts.control != tt.control {
				t.Errorf("order %v control %v, want %v %v", opts.order, opts.control, tt.order, tt.control)
			}
			if len(opts.captureFlags)+len(tt.capture) > 0 && !reflect.DeepEqual(opts.captureFlags, tt.capture) {
				t.Errorf("capture flags %q, want %q", opts.captureFlags, tt.capture)
			}
			if len(opts.compareArgs)+len(tt.compare) > 0 && !reflect.DeepEqual(opts.compareArgs, tt.compare) {
				t.Errorf("compare args %q, want %q", opts.compareArgs, tt.compare)
			}
		})
	}
}

// fakeABCapture writes a synthetic capture for each command: the command is
// "spin N SM TEMP [CUPTI]", one kernel of N ns in a range whose edges read SM
// MHz and TEMP C, captured by an agent that loaded CUPTI version CUPTI.
func fakeABCapture(t *testing.T, seen *[]string) func(flags []string, out, log string, command []string) error {
	return func(flags []string, out, log string, command []string) error {
		*seen = append(*seen, filepath.Base(out))
		var n, sm, temp, cupti uint64
		if _, err := fmt.Sscanf(strings.Join(command, " ")+" 0", "spin %d %d %d %d", &n, &sm, &temp, &cupti); err != nil {
			t.Fatal(err)
		}
		sample := map[string]any{"sampled_at": 1, "lag_nanos": 1000, "devices": []any{
			map[string]any{"index": 0, "sm_clock_mhz": sm, "mem_clock_mhz": 8533, "temperature_c": temp, "power_mw": 90000},
		}}
		status := phaseStatus(1000, 1000+n+100)
		status["phase_receipt"].(map[string]any)["edges"] = []any{
			map[string]any{"sequence": 1, "timestamp": 1000, "edge": "start", "api": "cuProfilerStart", "gpu": sample},
			map[string]any{"sequence": 2, "timestamp": 1000 + n + 100, "edge": "stop", "api": "cuProfilerStop", "gpu": sample},
		}
		if cupti != 0 {
			status["capability"] = map[string]any{"runtime_cupti": cupti, "compile_cupti": 130400, "library": fmt.Sprintf("/cupti-%d/libcupti.so.13", cupti)}
		}
		c := compareCapture([]rec{{subtype: "kernel", name: "k", start: 1050, end: 1050 + n}}, status)
		f, err := os.Create(out)
		if err != nil {
			return err
		}
		defer f.Close()
		if err := cudacapture.Write(f, c); err != nil {
			return err
		}
		return os.WriteFile(log, []byte("ok\n"), 0o644)
	}
}

func TestRunAB(t *testing.T) {
	dir := t.TempDir()
	var seen []string
	opts, err := parseABArgs([]string{"-o", dir, "-base", "spin 1000000 2522 50", "-target", "spin 400000 2400 60", "-control",
		"-quiet", "echo quiet"}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	// The control's own command is the base's; give it a different
	// duration here so its delta is not zero.
	capture := fakeABCapture(t, &seen)
	opts.capture = func(flags []string, out, log string, command []string) error {
		if filepath.Base(out) == "control.pprof" {
			command = []string{"spin", "1020000", "2522", "51"}
		}
		return capture(flags, out, log, command)
	}
	var stdout, stderr bytes.Buffer
	if err := runAB(opts, &stdout, &stderr); err != nil {
		t.Fatalf("%v\n%s", err, stderr.String())
	}
	if want := []string{"base.pprof", "target.pprof", "control.pprof"}; !reflect.DeepEqual(seen, want) {
		t.Errorf("captured %v, want %v", seen, want)
	}
	out := squash(stdout.String())
	for _, want := range []string{
		"base 1.000 1.000 1.000 0.000 2522/2522 8533/8533 50/50 90/90",
		"base -> target -0.600 -0.600 -0.600 +0.000",
		"base -> control +0.020 +0.020 +0.020 +0.000 noise floor",
		"|Δwindow| is 30.0x the control's",
		"warning: SM clock at the start edge was 2522 MHz in base and 2400 MHz in target, 4.8% apart (tolerance 2%)",
		"warning: temperature at the stop edge was 50 C in base and 60 C in target, 10 C apart (tolerance 5 C)",
	} {
		if !strings.Contains(out, squash(want)) {
			t.Errorf("output lacks %q:\n%s", want, stdout.String())
		}
	}
	if strings.Contains(out, "in control") {
		t.Errorf("control within tolerance was warned about:\n%s", stdout.String())
	}
	for _, name := range []string{"ab.json", "ab.txt", "base-target.txt", "base-target.json", "base-control.txt", "base-control.json"} {
		if _, err := os.Stat(filepath.Join(dir, name)); err != nil {
			t.Error(err)
		}
	}
	quiet, err := os.ReadFile(filepath.Join(dir, "quiet.log"))
	if err != nil {
		t.Fatal(err)
	}
	if n := strings.Count(string(quiet), "\nquiet\n"); n != 3 {
		t.Errorf("quiet ran %d times, want 3:\n%s", n, quiet)
	}
	var rep abReport
	data, err := os.ReadFile(filepath.Join(dir, "ab.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &rep); err != nil {
		t.Fatal(err)
	}
	if rep.NoiseRatio == nil || *rep.NoiseRatio != 30 {
		t.Errorf("noise ratio %v, want 30", rep.NoiseRatio)
	}

	// A second run into the same directory is refused before it captures.
	seen = nil
	if err := runAB(opts, io.Discard, io.Discard); err == nil || !strings.Contains(err.Error(), "exists") {
		t.Errorf("rerun: err = %v, want exists", err)
	}
	if seen != nil {
		t.Errorf("rerun captured %v", seen)
	}
}

func TestRunABCUPTI(t *testing.T) {
	tests := []struct {
		name         string
		base, target string
		want         []string // lines of ab.txt
		wantNot      string
	}{
		{
			name: "same CUPTI",
			base: "spin 1000 2522 50 130400", target: "spin 900 2522 50 130400",
			want:    []string{"base 0.001 0.001 0.001 0.000 2522/2522 8533/8533 50/50 90/90", "13.4.0"},
			wantNot: "warning:",
		},
		{
			name: "different CUPTI",
			base: "spin 1000 2522 50 130400", target: "spin 900 2522 50 130001",
			want: []string{
				"13.0.1",
				"warning: CUPTI DIFFERS: base loaded CUPTI 13.4.0 (/cupti-130400/libcupti.so.13) and target loaded CUPTI 13.0.1 (/cupti-130001/libcupti.so.13); the arms were not measured by the same CUPTI",
				"base -> target",
			},
		},
		{
			name: "not recorded",
			base: "spin 1000 2522 50 130400", target: "spin 900 2522 50",
			want: []string{"warning: target: CUPTI version not recorded"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			var seen []string
			opts, err := parseABArgs([]string{"-o", dir, "-base", tt.base, "-target", tt.target}, io.Discard)
			if err != nil {
				t.Fatal(err)
			}
			opts.capture = fakeABCapture(t, &seen)
			var stdout bytes.Buffer
			if err := runAB(opts, &stdout, io.Discard); err != nil {
				t.Fatalf("a CUPTI difference must not refuse the comparison: %v", err)
			}
			out := squash(stdout.String())
			for _, want := range tt.want {
				if !strings.Contains(out, squash(want)) {
					t.Errorf("output lacks %q:\n%s", want, stdout.String())
				}
			}
			if tt.wantNot != "" && strings.Contains(out, tt.wantNot) {
				t.Errorf("output mentions %q:\n%s", tt.wantNot, stdout.String())
			}
			var rep abReport
			data, err := os.ReadFile(filepath.Join(dir, "ab.json"))
			if err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(data, &rep); err != nil {
				t.Fatal(err)
			}
			if c := rep.Runs[0].CUPTI; c == nil || c.Runtime != 130400 || c.Library != "/cupti-130400/libcupti.so.13" {
				t.Errorf("base cupti = %+v, want runtime 130400", c)
			}
		})
	}
}

func TestRunABQuietStops(t *testing.T) {
	dir := t.TempDir()
	var seen []string
	opts, err := parseABArgs([]string{"-o", dir, "-base", "spin 1 1 1", "-target", "spin 1 1 1", "-quiet", "exit 3"}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	opts.capture = fakeABCapture(t, &seen)
	err = runAB(opts, io.Discard, io.Discard)
	if err == nil || !strings.Contains(err.Error(), "quiet check before base") {
		t.Errorf("err = %v, want quiet check failure", err)
	}
	if seen != nil {
		t.Errorf("captured %v after a failed quiet check", seen)
	}
}

func TestSplitWords(t *testing.T) {
	tests := []struct {
		in   string
		want []string
		err  string
	}{
		{in: "", want: nil},
		{in: "  spin  stream 1 ", want: []string{"spin", "stream", "1"}},
		{in: `env A='x y' B="p \"q\" \\ \n" C=a\ b ''`, want: []string{"env", "A=x y", `B=p "q" \ \n`, "C=a b", ""}},
		{in: "a'b'c", want: []string{"abc"}},
		{in: "'x", err: "single quote"},
		{in: `"x`, err: "double quote"},
		{in: `x\`, err: "backslash"},
	}
	for _, tt := range tests {
		got, err := splitWords(tt.in)
		if tt.err != "" {
			if err == nil || !strings.Contains(err.Error(), tt.err) {
				t.Errorf("splitWords(%q) err = %v, want %q", tt.in, err, tt.err)
			}
			continue
		}
		if err != nil || !reflect.DeepEqual(got, tt.want) {
			t.Errorf("splitWords(%q) = %q, %v, want %q", tt.in, got, err, tt.want)
		}
	}
}
