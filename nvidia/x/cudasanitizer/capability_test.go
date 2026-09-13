package cudasanitizer

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/internal/cudatool"
)

// recordedCaps returns what the shipped option corpus records for
// compute-sanitizer at a toolkit version, so the check [Run] makes can be
// exercised for toolkits this host does not have.
func recordedCaps(t *testing.T, v cudatool.Version) cudatool.Caps {
	t.Helper()
	opts, source, ok := cudatool.KnownOptions(DefaultToolName, v)
	if !ok {
		t.Fatalf("no recorded option set for %s %v", DefaultToolName, v)
	}
	return cudatool.Caps{Tool: DefaultToolName, Version: v, Source: source, Options: opts}
}

// TestConfigFlagsAgainstRecordedToolkits checks the command lines this package
// builds against every compute-sanitizer the option corpus records.
//
// Every option this package emits exists, with the same arity, on 12.6, 13.0 and
// 13.4. The value of the check is forward-looking: a release that renames one of
// these options or changes its arity is reported here, and by [Run] as a Go
// error, rather than as a command line the tool rejects while writing no report.
func TestConfigFlagsAgainstRecordedToolkits(t *testing.T) {
	tests := []struct {
		name string
		cfg  Config
	}{
		{"minimal", Config{Tool: Memcheck, MissingTool: MissingToolFail}},
		{
			name: "memcheck with every modelled option",
			cfg: Config{
				Tool:                 Memcheck,
				MissingTool:          MissingToolFail,
				ErrorExitCode:        66,
				RequireCUDAInit:      ToggleYes,
				LeakCheck:            ToggleYes,
				CheckAPIMemoryAccess: ToggleYes,
				Kill:                 ToggleYes,
				NVTX:                 ToggleYes,
				TargetProcesses:      "all",
				Demangle:             "full",
				DestroyOnDeviceError: "context",
				PrintLimit:           10,
				LaunchTimeout:        30 * time.Second,
			},
		},
		{
			name: "negated toggles",
			cfg: Config{
				Tool:                 Memcheck,
				MissingTool:          MissingToolFail,
				RequireCUDAInit:      ToggleNo,
				LeakCheck:            ToggleNo,
				CheckAPIMemoryAccess: ToggleNo,
				NVTX:                 ToggleNo,
			},
		},
		{
			name: "racecheck",
			cfg:  Config{Tool: Racecheck, MissingTool: MissingToolFail, RacecheckReport: "all"},
		},
		{
			name: "initcheck",
			cfg:  Config{Tool: Initcheck, MissingTool: MissingToolFail, TrackUnusedMemory: ToggleYes},
		},
	}
	versions := cudatool.KnownVersions(DefaultToolName)
	if len(versions) < 2 {
		t.Fatalf("recorded versions for %s are %v; the comparison needs at least two", DefaultToolName, versions)
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			flags := tt.cfg.flags("compute-sanitizer.log")
			for _, v := range versions {
				if err := recordedCaps(t, v).Check(flags...); err != nil {
					t.Errorf("%v: %v", v, err)
				}
			}
		})
	}
}

// TestMeasuredArities states the arities this package's command lines depend on,
// separately from the command lines themselves.
//
// --kill and --track-unused-memory are the ones that matter: compute-sanitizer
// spells both without an argument on every installed version, and passing one
// is rejected outright with "extra yes detected on the input command line" and
// no report written. [Config] therefore emits them bare, and a record claiming
// they take a value would make Run refuse a command line the tool accepts.
func TestMeasuredArities(t *testing.T) {
	tests := []struct {
		option string
		arity  cudatool.Arity
	}{
		{"--kill", cudatool.Bare},
		{"--track-unused-memory", cudatool.Bare},
		{"--leak-check", cudatool.Value},
		{"--require-cuda-init", cudatool.Value},
		{"--check-api-memory-access", cudatool.Value},
		{"--nvtx", cudatool.Value},
		{"--tool", cudatool.Value},
		{"--error-exitcode", cudatool.Value},
		{"--log-file", cudatool.Value},
		{"--racecheck-report", cudatool.Value},
		{"--target-processes", cudatool.Value},
		{"--demangle", cudatool.Value},
		{"--destroy-on-device-error", cudatool.Value},
		{"--print-limit", cudatool.Value},
		{"--launch-timeout", cudatool.Value},
	}
	for _, tt := range tests {
		t.Run(tt.option, func(t *testing.T) {
			for _, v := range cudatool.KnownVersions(DefaultToolName) {
				caps := recordedCaps(t, v)
				arity, ok := caps.Arity(tt.option)
				if !ok {
					t.Errorf("%v does not accept %s", v, tt.option)
					continue
				}
				if arity != tt.arity {
					t.Errorf("%v %s arity = %v, want %v", v, tt.option, arity, tt.arity)
				}
			}
		})
	}
}

// TestCheckReportsWrongArity is the other direction: a command line that gives
// an argument to an option the tool spells bare must be reported, because the
// tool rejects the whole command line rather than ignoring the extra word.
func TestCheckReportsWrongArity(t *testing.T) {
	caps := recordedCaps(t, cudatool.V(13, 4))
	tests := []struct {
		name string
		flag cudatool.Flag
		want bool
	}{
		{"kill bare", cudatool.Flag{Name: "--kill"}, false},
		{"kill with an argument", cudatool.Flag{Name: "--kill", Value: "yes"}, true},
		{"track-unused-memory with an argument", cudatool.Flag{Name: "--track-unused-memory", Value: "yes"}, true},
		{"leak-check bare", cudatool.Flag{Name: "--leak-check"}, true},
		{"leak-check with an argument", cudatool.Flag{Name: "--leak-check", Value: "full"}, false},
		{"option no version has", cudatool.Flag{Name: "--csv"}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := caps.Check(tt.flag)
			if (err != nil) != tt.want {
				t.Fatalf("Check(%+v) = %v, want an error: %v", tt.flag, err, tt.want)
			}
			if err != nil && !errors.Is(err, cudatool.ErrUnsupportedOption) {
				t.Errorf("error %v does not wrap ErrUnsupportedOption", err)
			}
		})
	}
}

// fakeSanitizer writes an executable that answers --version the way
// compute-sanitizer does, under an install prefix naming a toolkit version. The
// tool reports a product version rather than a toolkit one, so the prefix is
// where the version comes from.
func fakeSanitizer(t *testing.T, version string) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("the fake tool is a shell script")
	}
	dir := filepath.Join(t.TempDir(), "cuda-"+version, "bin")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, DefaultToolName)
	script := "#!/bin/sh\ncase \"$1\" in\n--version) echo 'NVIDIA (R) Compute Sanitizer'; echo 'Version 2025.3.1.0 (build 36400806)';;\nesac\n"
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestCheckOptions covers the resolved-tool path end to end, without a toolkit:
// the executable reports a product version, the install prefix names the
// toolkit, and the recorded option set for that toolkit answers.
func TestCheckOptions(t *testing.T) {
	tests := []struct {
		name    string
		extra   []string
		wantErr string
	}{
		{name: "modelled options only"},
		{name: "extra option the toolkit has", extra: []string{"--padding", "16"}},
		{name: "extra option no toolkit has", extra: []string{"--no-such-option"}, wantErr: "--no-such-option is not accepted"},
		{name: "extra option with an attached value", extra: []string{"--no-such-option=3"}, wantErr: "--no-such-option is not accepted"},
		{name: "extra option added after 13.0", extra: []string{"--check-tensor-ops", "yes"}, wantErr: "--check-tensor-ops is not accepted"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := Config{
				Tool:        Memcheck,
				MissingTool: MissingToolFail,
				Path:        fakeSanitizer(t, "13.0"),
				Kill:        ToggleYes,
				LeakCheck:   ToggleYes,
				ExtraArgs:   tt.extra,
			}
			err := cfg.CheckOptions(context.Background())
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("CheckOptions: %v", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("CheckOptions accepted %q", tt.extra)
			}
			if !errors.Is(err, cudatool.ErrUnsupportedOption) {
				t.Errorf("error %v does not wrap ErrUnsupportedOption", err)
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("error %v does not mention %q", err, tt.wantErr)
			}
			if !strings.Contains(err.Error(), "13.0") {
				t.Errorf("error %v does not name the toolkit version", err)
			}
		})
	}
}

// TestCheckOptionsWithUnknowableTool pins the one case where no check is made: an
// executable that reports no version and whose --help cannot be read supports no
// claim about what it accepts, so CheckOptions passes and the run proceeds to be
// diagnosed by the tool itself.
func TestCheckOptionsWithUnknowableTool(t *testing.T) {
	cfg := Config{Tool: Memcheck, MissingTool: MissingToolFail, Path: fixtureTool(t), Kill: ToggleYes}
	if err := cfg.CheckOptions(context.Background()); err != nil {
		t.Fatalf("CheckOptions: %v", err)
	}
}

// TestRunReportsUnsupportedOption shows the check reaching a caller through Run:
// the outcome is a tool failure with the unsupported option named, and no
// subprocess ran the rejected command line.
func TestRunReportsUnsupportedOption(t *testing.T) {
	cfg := Config{
		Tool:        Memcheck,
		MissingTool: MissingToolFail,
		Path:        fakeSanitizer(t, "12.6"),
		ExtraArgs:   []string{"--backtrace-short", "yes"}, // arrived in 13.0
	}
	res, err := Run(context.Background(), cfg, "/bin/true")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res.Outcome != OutcomeToolFailure {
		t.Errorf("Outcome = %v, want %v", res.Outcome, OutcomeToolFailure)
	}
	if !errors.Is(res.Err, ErrUnsupportedOption) {
		t.Fatalf("Err = %v, want it to wrap ErrUnsupportedOption", res.Err)
	}
	if !strings.Contains(res.Err.Error(), "--backtrace-short") {
		t.Errorf("Err = %v, want it to name --backtrace-short", res.Err)
	}
}
