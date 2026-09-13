package cudatool

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func readFixture(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// TestParseHelpGrammars checks both help grammars against real output: the
// column layout the binary utilities print and the program_options layout
// compute-sanitizer prints. The options asserted here are the ones whose shape
// a command line depends on, including the two compute-sanitizer spells without
// an argument.
func TestParseHelpGrammars(t *testing.T) {
	tests := []struct {
		fixture string
		want    []Option
		absent  []string
	}{
		{
			fixture: "help/nvdisasm-13.4.txt",
			want: []Option{
				{Name: "--base-address", Short: "-base", Arity: Value, Default: "0x0"},
				{Name: "--emit-json", Short: "-json", Arity: Bare},
				{Name: "--print-life-ranges", Short: "-plr", Arity: Bare},
			},
		},
		{
			fixture: "help/nvcc-13.4.txt",
			want: []Option{
				// An enumerated argument spec, which is not in angle brackets.
				{Name: "--std", Short: "-std", Arity: Value, Values: []string{"c++03", "c++11", "c++14", "c++17", "c++20", "c++23"}},
				{Name: "--compress-mode", Short: "-compress-mode", Arity: Value, Values: []string{"default", "size", "speed", "balance", "none"}},
				// A list argument spec with no brackets at all.
				{Name: "--entries", Short: "-e", Arity: Value},
				{Name: "--cubin", Short: "-cubin", Arity: Bare},
				// Renamed in 13.4; the old spelling is gone.
				{Name: "--fdevice-sanitize", Short: "-fdevice-sanitize", Arity: Value},
			},
			absent: []string{"--sanitize"},
		},
		{
			fixture: "help/compute-sanitizer-13.4.txt",
			want: []Option{
				// Measured: passing an argument to either is rejected outright.
				{Name: "--kill", Arity: Bare},
				{Name: "--track-unused-memory", Arity: Bare},
				{Name: "--leak-check", Arity: Value, Default: "no"},
				{Name: "--nvtx", Arity: Value, Default: "yes"},
				{Name: "--tool", Arity: Value, Default: "memcheck", Values: []string{"memcheck", "racecheck", "synccheck", "initcheck"}},
				{Name: "--help", Short: "-h", Arity: Bare},
			},
			absent: []string{"--csv"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.fixture, func(t *testing.T) {
			opts, err := ParseHelp(readFixture(t, tt.fixture))
			if err != nil {
				t.Fatalf("ParseHelp: %v", err)
			}
			if !slices.IsSortedFunc(opts, func(a, b Option) int { return cmpString(a.Name, b.Name) }) {
				t.Error("ParseHelp returned an unsorted option set")
			}
			for _, want := range tt.want {
				got, ok := opts.Find(want.Name)
				if !ok {
					t.Errorf("option %s missing", want.Name)
					continue
				}
				if got.Arity != want.Arity {
					t.Errorf("%s arity = %v, want %v", want.Name, got.Arity, want.Arity)
				}
				if got.Short != want.Short {
					t.Errorf("%s short = %q, want %q", want.Name, got.Short, want.Short)
				}
				if got.Default != want.Default {
					t.Errorf("%s default = %q, want %q", want.Name, got.Default, want.Default)
				}
				if want.Values != nil && !slices.Equal(got.Values, want.Values) {
					t.Errorf("%s values = %q, want %q", want.Name, got.Values, want.Values)
				}
			}
			for _, name := range tt.absent {
				if _, ok := opts.Find(name); ok {
					t.Errorf("option %s present, want it absent", name)
				}
			}
		})
	}
}

func cmpString(a, b string) int {
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	}
	return 0
}

func TestOptionsTextRoundTrip(t *testing.T) {
	for _, fixture := range []string{"help/nvdisasm-13.4.txt", "help/nvcc-13.4.txt", "help/compute-sanitizer-13.4.txt"} {
		t.Run(fixture, func(t *testing.T) {
			opts, err := ParseHelp(readFixture(t, fixture))
			if err != nil {
				t.Fatal(err)
			}
			back, err := ParseText(opts.Text())
			if err != nil {
				t.Fatalf("ParseText: %v", err)
			}
			if len(back) != len(opts) {
				t.Fatalf("round trip yielded %d options, want %d", len(back), len(opts))
			}
			for i := range opts {
				if opts[i].Name != back[i].Name || opts[i].Arity != back[i].Arity ||
					opts[i].Short != back[i].Short || opts[i].Default != back[i].Default ||
					!slices.Equal(opts[i].Values, back[i].Values) {
					t.Errorf("round trip changed %+v into %+v", opts[i], back[i])
				}
			}
		})
	}
}

func TestParseTextErrors(t *testing.T) {
	tests := []struct {
		name, text string
	}{
		{"no arity", "--kill\n"},
		{"bad arity", "--kill maybe\n"},
		{"not an option", "kill bare\n"},
		{"bare key", "--kill bare short\n"},
		{"unknown key", "--kill bare color=red\n"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := ParseText(tt.text); err == nil {
				t.Errorf("ParseText(%q) succeeded, want an error", tt.text)
			}
		})
	}
}

func TestParseHelpRejectsOptionlessOutput(t *testing.T) {
	if _, err := ParseHelp([]byte("ptxas fatal   : Unknown option '-help'\n")); err == nil {
		t.Error("ParseHelp succeeded on output with no options, want an error")
	}
}
