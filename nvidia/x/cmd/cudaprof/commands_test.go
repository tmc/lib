package main

import (
	"regexp"
	"strings"
	"testing"
)

// TestUsageListsEveryCommand fails when run dispatches a command that the
// top-level usage does not show.
func TestUsageListsEveryCommand(t *testing.T) {
	for _, c := range commands() {
		re := regexp.MustCompile(`(?m)^  cudaprof ` + regexp.QuoteMeta(c.name) + `(\s|$)`)
		if !re.MatchString(usage) {
			t.Errorf("usage does not list %q", c.name)
		}
	}
}

func TestLookupCommand(t *testing.T) {
	tests := []struct {
		arg  string
		want string // "" when arg names no command
	}{
		{"capture", "capture"},
		{"verify-capture", "verify-capture"},
		{"qualify", "qualify"},
		{"export", "export"},
		{"version", "version"},
		{"help", "help"},
		{"cfg", "cfg"},
		{"-cfg", "cfg"},
		{"-cfg=kernel", "cfg"},
		{"--cfg=kernel", "cfg"},
		{"bbcfg", "bbcfg"},
		{"-bbcfg=kernel", "bbcfg"},
		{"-pressure=kernel", "pressure"},
		{"view", "view"},
		{"web", "view"},
		{"-web", "view"},
		{"--web", "view"},
		{"-extract", "-extract"},
		{"-pack", "-pack"},
		{"capture.pprof", ""},
		{"-top", ""},
		{"-http=:8087", ""},
	}
	for _, tt := range tests {
		c, ok := lookupCommand(tt.arg)
		if ok != (tt.want != "") || c.name != tt.want {
			t.Errorf("lookupCommand(%q) = %q, %v, want %q", tt.arg, c.name, ok, tt.want)
		}
	}
}

func TestHelpCommand(t *testing.T) {
	out := captureStdout(t, func() error { return runHelpCommand([]string{"help"}) })
	if out != usage {
		t.Errorf("help printed %q, want the top-level usage", out)
	}
	for _, c := range commands() {
		out := captureStdout(t, func() error { return runHelpCommand([]string{"help", c.name}) })
		if !strings.Contains(out, "cudaprof "+c.name) {
			t.Errorf("help %s printed %q, want its usage", c.name, out)
		}
	}
	if err := runHelpCommand([]string{"help", "capture.pprof"}); err == nil {
		t.Error("help of an unknown command succeeded")
	}
}
