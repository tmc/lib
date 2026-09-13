package cudatool

import (
	"errors"
	"slices"
	"strings"
	"testing"
)

func TestArgs(t *testing.T) {
	tests := []struct {
		name  string
		flags []Flag
		want  []string
	}{
		{name: "none"},
		{name: "bare", flags: []Flag{{Name: "--kill"}}, want: []string{"--kill"}},
		{
			name:  "value",
			flags: []Flag{{Name: "--tool", Value: "memcheck"}},
			want:  []string{"--tool", "memcheck"},
		},
		{
			name:  "mixed",
			flags: []Flag{{Name: "--tool", Value: "initcheck"}, {Name: "--track-unused-memory"}},
			want:  []string{"--tool", "initcheck", "--track-unused-memory"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Args(tt.flags...); !slices.Equal(got, tt.want) {
				t.Errorf("Args = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestFlagArity(t *testing.T) {
	if got := (Flag{Name: "--kill"}).Arity(); got != Bare {
		t.Errorf("bare flag arity = %v, want bare", got)
	}
	if got := (Flag{Name: "--tool", Value: "memcheck"}).Arity(); got != Value {
		t.Errorf("value flag arity = %v, want value", got)
	}
}

// TestCapsCheck covers both ways a command line can be wrong about a tool: an
// option that toolkit does not have, and one whose arity differs.
func TestCapsCheck(t *testing.T) {
	caps := func(t *testing.T, tool string, v Version) Caps {
		t.Helper()
		opts, source, ok := KnownOptions(tool, v)
		if !ok {
			t.Fatalf("no record for %s %v", tool, v)
		}
		return Caps{Tool: tool, Version: v, Source: source, Options: opts}
	}
	tests := []struct {
		name    string
		tool    string
		version Version
		flags   []Flag
		want    string // substring of the error, empty when the flags are accepted
	}{
		{
			name: "accepted", tool: "nvdisasm", version: V(13, 4),
			flags: []Flag{{Name: "--print-code"}, {Name: "--life-range-mode", Value: "count"}},
		},
		{
			name: "option the toolkit lacks", tool: "nvdisasm", version: V(12, 6),
			flags: []Flag{{Name: "--emit-json"}},
			want:  "--emit-json is not accepted",
		},
		{
			name: "argument given to a bare option", tool: "compute-sanitizer", version: V(13, 4),
			flags: []Flag{{Name: "--kill", Value: "yes"}},
			want:  `--kill takes no argument, the command line gives it "yes"`,
		},
		{
			name: "argument omitted from a value option", tool: "compute-sanitizer", version: V(13, 4),
			flags: []Flag{{Name: "--leak-check"}},
			want:  "--leak-check takes an argument, the command line emits it bare",
		},
		{
			name: "every fault is named", tool: "nvdisasm", version: V(12, 6),
			flags: []Flag{{Name: "--emit-json"}, {Name: "--no-such-option"}},
			want:  "--emit-json is not accepted; --no-such-option is not accepted",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := caps(t, tt.tool, tt.version).Check(tt.flags...)
			if tt.want == "" {
				if err != nil {
					t.Fatalf("Check: %v", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("Check accepted %q", Args(tt.flags...))
			}
			if !errors.Is(err, ErrUnsupportedOption) {
				t.Errorf("error %v does not wrap ErrUnsupportedOption", err)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("error %v does not contain %q", err, tt.want)
			}
			if !strings.Contains(err.Error(), tt.tool) || !strings.Contains(err.Error(), tt.version.String()) {
				t.Errorf("error %v does not name the tool and its version", err)
			}
		})
	}
}

// TestCapsCheckNames covers the pass-through form: names are checked, arity is
// not, and an argument that is not an option is skipped.
func TestCapsCheckNames(t *testing.T) {
	opts, source, ok := KnownOptions("compute-sanitizer", V(13, 0))
	if !ok {
		t.Fatal("no record for compute-sanitizer 13.0")
	}
	caps := Caps{Tool: "compute-sanitizer", Version: V(13, 0), Source: source, Options: opts}
	tests := []struct {
		name string
		args []string
		want string
	}{
		{name: "accepted", args: []string{"--padding", "16"}},
		{name: "attached value", args: []string{"--padding=16"}},
		{name: "bare option spelled with a value", args: []string{"--kill"}},
		{name: "unknown name", args: []string{"--check-tensor-ops", "yes"}, want: "--check-tensor-ops is not accepted"},
		{name: "unknown attached value", args: []string{"--check-tensor-ops=yes"}, want: "--check-tensor-ops is not accepted"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := caps.CheckNames(tt.args...)
			switch {
			case tt.want == "" && err != nil:
				t.Fatalf("CheckNames(%q): %v", tt.args, err)
			case tt.want != "" && err == nil:
				t.Fatalf("CheckNames(%q) accepted it", tt.args)
			case tt.want != "" && !strings.Contains(err.Error(), tt.want):
				t.Errorf("error %v does not contain %q", err, tt.want)
			}
		})
	}
}

// TestCapsOrigin pins what an error says about where the option set came from,
// since that is what tells a reader whether to trust it.
func TestCapsOrigin(t *testing.T) {
	tests := []struct {
		name string
		caps Caps
		want string
	}{
		{name: "exact record", caps: Caps{Version: V(13, 4), Source: V(13, 4)}, want: "recorded"},
		{name: "nearest record", caps: Caps{Version: V(13, 2), Source: V(13, 0)}, want: "recorded for 13.0"},
		{name: "probed", caps: Caps{Version: V(14, 0), Source: V(14, 0), Probed: true}, want: "from --help"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.caps.origin(); got != tt.want {
				t.Errorf("origin = %q, want %q", got, tt.want)
			}
		})
	}
}
