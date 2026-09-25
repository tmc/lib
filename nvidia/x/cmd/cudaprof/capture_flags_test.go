package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestNoDocumentedFlagIsAcceptedThenRefused is the §6.3 control, inverted.
// Five collector-policy flags were parsed, documented in help, and then
// refused -- first while parsing and again inside runCapture, after the target
// had been spawned and after runCaptureCommand had written an artifact from
// the zero-valued result it got back. Each of the five is now either
// implemented or gone, and this pins that: every option the help text
// documents must parse, and no parse may fail with "not implemented".
func TestNoDocumentedFlagIsAcceptedThenRefused(t *testing.T) {
	for _, args := range [][]string{
		{"--api-activity=false"},
		{"--api-activity"},
		{"--memsets=false"},
		{"--follow-children"},
		{"--follow-children=false"},
		{"--exclude", "sgemm"},
		{"--trace"},
		{"--require-complete"},
		{"--recover-unfinalized"},
	} {
		name := strings.TrimPrefix(args[0], "--")
		t.Run(name, func(t *testing.T) {
			if !strings.Contains(captureHelp(), strings.SplitN(args[0], "=", 2)[0]) {
				t.Fatalf("capture help does not document %s", args[0])
			}
			if _, err := parseCaptureArgs(append(append([]string(nil), args...), "--", "true")); err != nil {
				t.Fatalf("parseCaptureArgs(%v): %v", args, err)
			}
		})
	}
}

// TestRetiredFlagsAreUnknownAndSayWhatReplacedThem covers the other three.
// --deep and --exhaustive named a mode this agent has no referent for -- API
// attribution is collected by default -- and --collect asked for hardware
// counters, which need the CUPTI Profiling API with replay and exist only
// in-process. They are gone rather than deprecated: the refusal happens while
// the flag is still an argument, before anything runs and before anything is
// written, and it names what to reach for instead.
func TestRetiredFlagsAreUnknownAndSayWhatReplacedThem(t *testing.T) {
	for _, tc := range []struct {
		flag string
		want string
	}{
		{"--deep", "--api-activity=false"},
		{"--exhaustive", "--api-activity=false"},
		{"--collect", "cuptirange"},
	} {
		t.Run(strings.TrimPrefix(tc.flag, "--"), func(t *testing.T) {
			if strings.Contains(captureHelp(), tc.flag+" ") {
				t.Errorf("capture help still documents the retired %s", tc.flag)
			}
			out := filepath.Join(t.TempDir(), "capture.pprof")
			args := []string{tc.flag, "--no-inject", "-o", out, "--", "/bin/echo", "hello"}

			_, err := parseCaptureArgs(args)
			if err == nil {
				t.Fatalf("parseCaptureArgs accepted the retired %s", tc.flag)
			}
			if !strings.Contains(err.Error(), tc.flag) || !strings.Contains(err.Error(), tc.want) {
				t.Errorf("rejection of %s does not name the flag and %q: %v", tc.flag, tc.want, err)
			}
			if err := runCaptureCommand(args); err == nil {
				t.Fatalf("runCaptureCommand(%s) returned nil error", tc.flag)
			}
			if _, err := os.Stat(out); err == nil {
				t.Errorf("runCaptureCommand(%s) wrote %s despite failing; a retired flag must not produce an artifact", tc.flag, out)
			}
		})
	}
}
