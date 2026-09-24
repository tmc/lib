package cudacmp

import (
	"context"
	"regexp"
	"runtime"
	"testing"
)

func TestCommandRunner(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("needs /bin/sh")
	}
	// The script prints a result line, the phase as it saw it both ways, and
	// a line that changes on every run.
	const script = `echo "result 42"; echo "arg $0 env $CUDACMP_PHASE"; echo "tok/s $$"`
	tests := []struct {
		name    string
		cmd     Command
		same    bool // two runs digest alike
		wantErr bool
	}{
		{"whole output varies", Command{Path: "/bin/sh", Args: []string{"-c", script}}, false, false},
		{"digest keeps result lines", Command{Path: "/bin/sh", Args: []string{"-c", script}, Digest: regexp.MustCompile(`^result `)}, true, false},
		{"phase in the argument", Command{Path: "/bin/sh", Args: []string{"-c", script}, Digest: regexp.MustCompile(`^arg decode env decode$`)}, true, false},
		{"phase only in the environment", Command{Path: "/bin/sh", Args: []string{"-c", script, "noarg"}, NoPhaseArg: true, Digest: regexp.MustCompile(`^arg noarg env decode$`)}, true, false},
		{"digest matches nothing", Command{Path: "/bin/sh", Args: []string{"-c", script}, Digest: regexp.MustCompile(`^absent`)}, false, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			run := tt.cmd.Runner()
			a, err := run(context.Background(), PhaseDecode)
			if tt.wantErr {
				if err == nil || a.Complete {
					t.Fatalf("run = %+v, %v; want an error", a, err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			b, err := run(context.Background(), PhaseDecode)
			if err != nil {
				t.Fatal(err)
			}
			if same := a.Digest == b.Digest; same != tt.same {
				t.Errorf("digests %s and %s: same = %v, want %v", a.Digest, b.Digest, same, tt.same)
			}
		})
	}
}

// TestCommandArmRunnerKeepsArgs checks that the appended phase does not leak
// into the caller's argument slice from one run to the next.
func TestCommandArmRunnerKeepsArgs(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("needs /bin/sh")
	}
	args := make([]string, 2, 8)
	args[0], args[1] = "-c", `echo "$0 $#"`
	run := CommandArmRunner("/bin/sh", args...)
	a, err := run(context.Background(), PhasePrefill)
	if err != nil {
		t.Fatal(err)
	}
	b, err := run(context.Background(), PhasePrefill)
	if err != nil {
		t.Fatal(err)
	}
	if a.Digest != b.Digest {
		t.Error("two runs of one phase digest differently")
	}
	if spare := args[:3][2]; spare != "" {
		t.Errorf("run wrote %q into the caller's spare capacity", spare)
	}
}
