package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestUnimplementedPolicyRejectedBeforeSpawn is the §6.3 control. The
// collector policy flags are parsed and accepted, then rejected deep inside
// runCapture -- after the target has been spawned and after runCaptureCommand
// has already written an artifact from the zero-valued result it got back.
// A flag the tool does not implement must be refused while it is still an
// argument, before anything runs and before anything is written.
func TestUnimplementedPolicyRejectedBeforeSpawn(t *testing.T) {
	for _, flagArg := range []string{"--follow-children", "--deep", "--exhaustive"} {
		t.Run(strings.TrimPrefix(flagArg, "--"), func(t *testing.T) {
			out := filepath.Join(t.TempDir(), "capture.pprof")
			args := []string{flagArg, "--no-inject", "-o", out, "--", "/bin/echo", "hello"}

			if _, err := parseCaptureArgs(args); err == nil {
				t.Errorf("parseCaptureArgs accepted %s, which is not implemented", flagArg)
			} else if !strings.Contains(err.Error(), flagArg) {
				t.Errorf("rejection of %s does not name the flag: %v", flagArg, err)
			}

			err := runCaptureCommand(args)
			if err == nil {
				t.Fatalf("runCaptureCommand(%s) returned nil error", flagArg)
			}
			if _, statErr := os.Stat(out); statErr == nil {
				t.Errorf("runCaptureCommand(%s) wrote %s despite failing; an unimplemented flag must not produce an artifact", flagArg, out)
			}
		})
	}
}
