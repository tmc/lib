package nvidiasmi

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// TestMain makes the test binary stand in for nvidia-smi, so no test
// can run the real one. When fakeEnv is set, the binary records its
// arguments in $NVIDIASMI_FAKE_ARGS, writes $NVIDIASMI_FAKE_STDOUT and
// $NVIDIASMI_FAKE_STDERR, and exits with $NVIDIASMI_FAKE_EXIT.
func TestMain(m *testing.M) {
	if os.Getenv(fakeEnv) != "" {
		fakeNVIDIASMI()
	}
	exe, err := os.Executable()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	command = exe
	os.Setenv(fakeEnv, "1")
	os.Exit(m.Run())
}

const fakeEnv = "NVIDIASMI_FAKE"

func fakeNVIDIASMI() {
	if f := os.Getenv("NVIDIASMI_FAKE_ARGS"); f != "" {
		if err := os.WriteFile(f, []byte(strings.Join(os.Args[1:], "\n")), 0o666); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(254)
		}
	}
	os.Stdout.WriteString(os.Getenv("NVIDIASMI_FAKE_STDOUT"))
	os.Stderr.WriteString(os.Getenv("NVIDIASMI_FAKE_STDERR"))
	code, _ := strconv.Atoi(os.Getenv("NVIDIASMI_FAKE_EXIT"))
	os.Exit(code)
}

// fake configures the fake nvidia-smi for one test and returns a
// function that reports the arguments it was run with, or nil if it
// was not run.
func fake(t *testing.T, stdout, stderr string, code int) func() []string {
	t.Helper()
	args := filepath.Join(t.TempDir(), "args")
	t.Setenv("NVIDIASMI_FAKE_ARGS", args)
	t.Setenv("NVIDIASMI_FAKE_STDOUT", stdout)
	t.Setenv("NVIDIASMI_FAKE_STDERR", stderr)
	t.Setenv("NVIDIASMI_FAKE_EXIT", strconv.Itoa(code))
	return func() []string {
		b, err := os.ReadFile(args)
		if errors.Is(err, fs.ErrNotExist) {
			return nil
		}
		if err != nil {
			t.Fatal(err)
		}
		return strings.Split(string(b), "\n")
	}
}

func TestReset(t *testing.T) {
	tests := []struct {
		ids  []string
		want string // -i argument
	}{
		{[]string{"0"}, "0"},
		{[]string{"15"}, "15"},
		{[]string{"GPU-25e04bdf-6a27-0a2d-f989-a9228a03f373"}, "GPU-25e04bdf-6a27-0a2d-f989-a9228a03f373"},
		{[]string{"GPU-25E04BDF-6A27-0A2D-F989-A9228A03F373"}, "GPU-25E04BDF-6A27-0A2D-F989-A9228A03F373"},
		{[]string{"00000000:29:00.0"}, "00000000:29:00.0"},
		{[]string{"0000:3b:00.0"}, "0000:3b:00.0"},
		{[]string{"0", "1", "00000000:29:00.0"}, "0,1,00000000:29:00.0"},
	}
	for _, tt := range tests {
		t.Run(strings.Join(tt.ids, ","), func(t *testing.T) {
			args := fake(t, "GPU 00000000:29:00.0 was successfully reset.\n", "", 0)
			if err := Reset(context.Background(), tt.ids...); err != nil {
				t.Fatalf("Reset: %v", err)
			}
			got := strings.Join(args(), " ")
			if want := "--gpu-reset -i " + tt.want; got != want {
				t.Errorf("ran nvidia-smi %s, want %s", got, want)
			}
		})
	}
}

func TestResetRejects(t *testing.T) {
	tests := []struct {
		name string
		ids  []string
	}{
		{"none", nil},
		{"empty", []string{""}},
		{"all", []string{"all"}},
		{"serial", []string{"1564720004631"}},
		{"four digits", []string{"1000"}},
		{"negative", []string{"-1"}},
		{"comma", []string{"0,1"}},
		{"space", []string{" 0"}},
		{"flag", []string{"-r"}},
		{"mig uuid", []string{"MIG-25e04bdf-6a27-0a2d-f989-a9228a03f373"}},
		{"short uuid", []string{"GPU-25e04bdf-6a27-0a2d-f989-a9228a03f37"}},
		{"uuid groups", []string{"GPU-25e04bdf6a27-0a2d-f989-a9228a03f373"}},
		{"uuid not hex", []string{"GPU-25e04bdf-6a27-0a2d-f989-a9228a03f37g"}},
		{"no domain", []string{"29:00.0"}},
		{"domain 6", []string{"000000:29:00.0"}},
		{"bus 3", []string{"00000000:029:00.0"}},
		{"no function", []string{"00000000:29:00"}},
		{"function 2", []string{"00000000:29:00.00"}},
		{"pci not hex", []string{"0000000g:29:00.0"}},
		{"one bad", []string{"0", "all"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			args := fake(t, "", "", 0)
			err := Reset(context.Background(), tt.ids...)
			if err == nil {
				t.Fatal("Reset succeeded")
			}
			if !strings.HasPrefix(err.Error(), "reset gpu: ") {
				t.Errorf("err = %q, want prefix %q", err, "reset gpu: ")
			}
			if a := args(); a != nil {
				t.Errorf("ran nvidia-smi %q", a)
			}
		})
	}
}

func TestResetError(t *testing.T) {
	// The "In use by another client" line follows NVIDIA forum reports
	// of nvidia-smi -r. The other texts, the exit statuses for them,
	// and the stream each is written to are made up for the test.
	const inUse = "The following GPUs could not be reset:\n  GPU 00000000:29:00.0: In use by another client\n"
	tests := []struct {
		name           string
		stdout, stderr string
		code           int
		is             error
		msg            string
	}{
		{"not supported", "", "GPU reset is not supported.\n", 3, ErrNotSupported, "exit status 3: GPU reset is not supported."},
		{"permission", "", "Insufficient Permissions\n", 4, fs.ErrPermission, "exit status 4: Insufficient Permissions"},
		{"in use on stdout", inUse, "", 255, ErrUnknown, "exit status 255: The following GPUs could not be reset: GPU 00000000:29:00.0: In use by another client"},
		{"invalid", "", "", 2, ErrInvalidArgument, "exit status 2"},
		{"not found", "", "", 6, ErrNotFound, "exit status 6"},
		{"no driver", "", "", 9, ErrDriverNotLoaded, "exit status 9"},
		{"lost", "", "", 15, ErrGPULost, "exit status 15"},
	}
	all := []error{ErrInvalidArgument, ErrNotSupported, fs.ErrPermission, ErrNotFound, ErrDriverNotLoaded, ErrGPULost, ErrUnknown}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fake(t, tt.stdout, tt.stderr, tt.code)
			err := Reset(context.Background(), "00000000:29:00.0")
			var e *Error
			if !errors.As(err, &e) {
				t.Fatalf("err = %v (%T), want *Error", err, err)
			}
			if e.Code != tt.code || string(e.Stdout) != tt.stdout || string(e.Stderr) != tt.stderr {
				t.Errorf("Error = %+v", e)
			}
			if want := "nvidia-smi --gpu-reset -i 00000000:29:00.0: " + tt.msg; err.Error() != want {
				t.Errorf("err = %q, want %q", err, want)
			}
			for _, target := range all {
				if got := errors.Is(err, target); got != (target == tt.is) {
					t.Errorf("errors.Is(err, %v) = %v", target, got)
				}
			}
		})
	}
}

func TestQueryFake(t *testing.T) {
	data, err := os.ReadFile("testdata/tesla-t4-515.xml")
	if err != nil {
		t.Fatal(err)
	}
	args := fake(t, string(data), "", 0)
	l, err := Query(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(args(), " "); got != "-q -x" {
		t.Errorf("ran nvidia-smi %s, want -q -x", got)
	}
	if len(l.GPUs) != 1 || l.GPUs[0].ProductName != "Tesla T4" {
		t.Errorf("Query = %+v", l)
	}

	fake(t, "", "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.\n", 9)
	if _, err := Query(context.Background()); !errors.Is(err, ErrDriverNotLoaded) {
		t.Errorf("Query err = %v, want ErrDriverNotLoaded", err)
	}
}

func TestRunMissing(t *testing.T) {
	saved := command
	defer func() { command = saved }()
	command = filepath.Join(t.TempDir(), "nvidia-smi")
	_, err := QueryTopology(context.Background())
	var e *Error
	if err == nil || errors.As(err, &e) || !strings.HasPrefix(err.Error(), "run nvidia-smi: ") {
		t.Errorf("err = %v, want a run error that is not *Error", err)
	}
}
