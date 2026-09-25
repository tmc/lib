package main

import (
	"strings"
	"testing"
	"time"
)

func TestParseFlags(t *testing.T) {
	tests := []struct {
		args   []string
		want   options
		status int
		ok     bool
		stdout string
		stderr string
	}{
		{args: nil, want: options{encodeHide: 30 * time.Second}, ok: true},
		{args: []string{"-d", "5", "-p", "-C"}, want: options{delay: 500 * time.Millisecond, noPlot: true, noColor: true, encodeHide: 30 * time.Second}, ok: true},
		{args: []string{"--delay=0"}, want: options{delay: 100 * time.Millisecond, encodeHide: 30 * time.Second}, ok: true},
		{args: []string{"-d", "5000"}, want: options{delay: 99900 * time.Millisecond, encodeHide: 30 * time.Second}, ok: true},
		{args: []string{"-E", "-1", "-r", "-P", "-f"}, want: options{encodeHide: -time.Second, reverse: true, noProcs: true, fahrenheit: true}, ok: true},
		{args: []string{"-i"}, want: options{infoBar: true, encodeHide: 30 * time.Second}, ok: true},
		{args: []string{"--gpu-info"}, want: options{infoBar: true, encodeHide: 30 * time.Second}, ok: true},
		{args: []string{"-s", "--loop"}, want: options{snapshot: true, loop: true, encodeHide: 30 * time.Second}, ok: true},
		{args: []string{"-v"}, stdout: "nvtop version " + version},
		{args: []string{"-h"}, stdout: "Available options:"},
		{args: []string{"-d", "-1"}, status: 2, stderr: "time machine"},
		{args: []string{"-d", "x"}, status: 2, stderr: "tenths of seconds"},
		{args: []string{"-E", "x"}, status: 2, stderr: "encode/decode hide time"},
		{args: []string{"-x"}, status: 2, stderr: "Available options:"},
		{args: []string{"extra"}, status: 2, stderr: "unexpected argument"},
	}
	for _, tt := range tests {
		t.Run(strings.Join(tt.args, " "), func(t *testing.T) {
			var stdout, stderr strings.Builder
			opts, status, ok := parseFlags(tt.args, &stdout, &stderr)
			if ok != tt.ok || status != tt.status {
				t.Fatalf("parseFlags = status %d, ok %v; want %d, %v\nstderr: %s", status, ok, tt.status, tt.ok, stderr.String())
			}
			if ok && opts != tt.want {
				t.Errorf("options = %+v, want %+v", opts, tt.want)
			}
			if !strings.Contains(stdout.String(), tt.stdout) {
				t.Errorf("stdout = %q, want it to contain %q", stdout.String(), tt.stdout)
			}
			if !strings.Contains(stderr.String(), tt.stderr) {
				t.Errorf("stderr = %q, want it to contain %q", stderr.String(), tt.stderr)
			}
		})
	}
}
