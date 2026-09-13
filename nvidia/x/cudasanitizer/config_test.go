package cudasanitizer

import (
	"errors"
	"strings"
	"testing"
	"time"
)

func TestConfigValidate(t *testing.T) {
	tests := []struct {
		name string
		cfg  Config
		want string // substring of the expected error, empty for success
	}{
		{
			name: "ok",
			cfg:  Config{Tool: Memcheck, MissingTool: MissingToolFail},
		},
		{
			name: "no tool",
			cfg:  Config{MissingTool: MissingToolFail},
			want: "Tool",
		},
		{
			name: "unknown tool",
			cfg:  Config{Tool: Tool("leakcheck"), MissingTool: MissingToolFail},
			want: "Tool",
		},
		{
			// The zero policy is rejected on purpose: an absent
			// compute-sanitizer must never default to a quiet pass.
			name: "missing-tool policy unset",
			cfg:  Config{Tool: Memcheck},
			want: "MissingTool",
		},
		{
			name: "error exit code out of range",
			cfg:  Config{Tool: Memcheck, MissingTool: MissingToolFail, ErrorExitCode: 300},
			want: "ErrorExitCode",
		},
		{
			name: "bad racecheck report",
			cfg:  Config{Tool: Racecheck, MissingTool: MissingToolFail, RacecheckReport: "everything"},
			want: "RacecheckReport",
		},
		{
			name: "bad target processes",
			cfg:  Config{Tool: Memcheck, MissingTool: MissingToolFail, TargetProcesses: "some"},
			want: "TargetProcesses",
		},
		{
			name: "extra args duplicate a managed option",
			cfg: Config{Tool: Memcheck, MissingTool: MissingToolFail,
				ExtraArgs: []string{"--error-exitcode=0"}},
			want: "--error-exitcode",
		},
		{
			name: "extra args are otherwise allowed",
			cfg: Config{Tool: Memcheck, MissingTool: MissingToolFail,
				ExtraArgs: []string{"--padding", "8"}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.cfg.validate()
			if tt.want == "" {
				if err != nil {
					t.Fatalf("validate() = %v, want nil", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("validate() = nil, want an error mentioning %q", tt.want)
			}
			if !errors.Is(err, ErrInvalidConfig) {
				t.Errorf("validate() = %v, want it to wrap ErrInvalidConfig", err)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("validate() = %v, want it to mention %q", err, tt.want)
			}
		})
	}
}

// TestErrorExitCodeIsNeverZero pins the central defence: whatever the caller
// configures, the argument vector carries a nonzero --error-exitcode, because
// the tool's own default of 0 hides detected errors.
func TestErrorExitCodeIsNeverZero(t *testing.T) {
	for _, cfg := range []Config{
		{Tool: Memcheck, MissingTool: MissingToolFail},
		{Tool: Memcheck, MissingTool: MissingToolFail, ErrorExitCode: 0},
		{Tool: Racecheck, MissingTool: MissingToolSkip, ErrorExitCode: 42},
	} {
		args, err := cfg.Args("", "/app")
		if err != nil {
			t.Fatalf("Args: %v", err)
		}
		i := indexOf(args, "--error-exitcode")
		if i < 0 || i+1 >= len(args) {
			t.Fatalf("Args() = %q, want --error-exitcode present", args)
		}
		if args[i+1] == "0" {
			t.Fatalf("Args() = %q, --error-exitcode must never be 0", args)
		}
	}
}

func TestArgsOrder(t *testing.T) {
	cfg := Config{
		Tool:              Initcheck,
		MissingTool:       MissingToolFail,
		ErrorExitCode:     42,
		RequireCUDAInit:   ToggleYes,
		TrackUnusedMemory: ToggleYes,
		LeakCheck:         ToggleNo,
		TargetProcesses:   "all",
		PrintLimit:        7,
		LaunchTimeout:     90 * time.Second,
		ExtraArgs:         []string{"--padding", "8"},
	}
	got, err := cfg.Args("/tmp/report.log", "/bin/app", "-run", "TestX")
	if err != nil {
		t.Fatalf("Args: %v", err)
	}
	want := []string{
		"--tool", "initcheck",
		"--error-exitcode", "42",
		"--log-file", "/tmp/report.log",
		"--require-cuda-init", "yes",
		"--leak-check", "no",
		"--track-unused-memory",
		"--target-processes", "all",
		"--print-limit", "7",
		"--launch-timeout", "90",
		"--padding", "8",
		"/bin/app", "-run", "TestX",
	}
	if len(got) != len(want) {
		t.Fatalf("Args() = %q\nwant %q", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Args()[%d] = %q, want %q\ngot  %q\nwant %q", i, got[i], want[i], got, want)
		}
	}
}

func TestArgsToggleDefaultEmitsNothing(t *testing.T) {
	cfg := Config{Tool: Memcheck, MissingTool: MissingToolFail}
	args, err := cfg.Args("", "/bin/app")
	if err != nil {
		t.Fatalf("Args: %v", err)
	}
	for _, unwanted := range []string{"--require-cuda-init", "--leak-check", "--log-file", "--kill", "--nvtx"} {
		if indexOf(args, unwanted) >= 0 {
			t.Errorf("Args() = %q, want no %s when it was not configured", args, unwanted)
		}
	}
}

// TestArgsBareFlags checks the options compute-sanitizer spells without an
// argument. Passing one is a hard error ("extra yes detected on the input
// command line"), so ToggleYes must emit the option alone and ToggleNo must
// emit nothing rather than an "off" spelling the tool does not accept.
func TestArgsBareFlags(t *testing.T) {
	for _, tt := range []struct {
		name string
		cfg  Config
		flag string
		want bool
	}{
		{"kill yes", Config{Tool: Memcheck, MissingTool: MissingToolFail, Kill: ToggleYes}, "--kill", true},
		{"kill no", Config{Tool: Memcheck, MissingTool: MissingToolFail, Kill: ToggleNo}, "--kill", false},
		{"track-unused-memory yes", Config{Tool: Initcheck, MissingTool: MissingToolFail, TrackUnusedMemory: ToggleYes}, "--track-unused-memory", true},
		{"track-unused-memory no", Config{Tool: Initcheck, MissingTool: MissingToolFail, TrackUnusedMemory: ToggleNo}, "--track-unused-memory", false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			const prog = "/bin/app"
			args, err := tt.cfg.Args("", prog)
			if err != nil {
				t.Fatalf("Args: %v", err)
			}
			i := indexOf(args, tt.flag)
			if (i >= 0) != tt.want {
				t.Fatalf("Args() = %q, want %s present=%v", args, tt.flag, tt.want)
			}
			if i < 0 {
				return
			}
			next := i + 1
			if next < len(args) && args[next] != prog && !strings.HasPrefix(args[next], "-") {
				t.Fatalf("Args() = %q, want %s with no argument, got %q after it", args, tt.flag, args[next])
			}
		})
	}
}

func indexOf(s []string, v string) int {
	for i, x := range s {
		if x == v {
			return i
		}
	}
	return -1
}
