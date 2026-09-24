package main

import (
	"runtime"
	"runtime/debug"
	"strings"
	"testing"
)

func TestStampOf(t *testing.T) {
	tests := []struct {
		name string
		info *debug.BuildInfo
		want buildStamp
	}{
		{
			name: "no build info",
			want: buildStamp{Module: "unknown", Revision: "unknown", Time: "unknown", Modified: "unknown", GoVersion: runtime.Version()},
		},
		{
			name: "no vcs",
			info: &debug.BuildInfo{GoVersion: "go1.99", Main: debug.Module{Path: "example.com/m"}},
			want: buildStamp{Module: "example.com/m", Revision: "unknown", Time: "unknown", Modified: "unknown", GoVersion: "go1.99"},
		},
		{
			name: "vcs",
			info: &debug.BuildInfo{
				GoVersion: "go1.99",
				Main:      debug.Module{Path: "example.com/m"},
				Settings: []debug.BuildSetting{
					{Key: "vcs", Value: "git"},
					{Key: "vcs.revision", Value: "0123abcd"},
					{Key: "vcs.time", Value: "2026-01-02T03:04:05Z"},
					{Key: "vcs.modified", Value: "true"},
				},
			},
			want: buildStamp{Module: "example.com/m", Revision: "0123abcd", Time: "2026-01-02T03:04:05Z", Modified: "true", GoVersion: "go1.99"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := stampOf(tt.info); got != tt.want {
				t.Errorf("stampOf = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestBuildStampWrite(t *testing.T) {
	b := buildStamp{Module: "example.com/m", Revision: "0123abcd", Time: "2026-01-02T03:04:05Z", Modified: "false", GoVersion: "go1.99"}
	var out strings.Builder
	if err := b.write(&out); err != nil {
		t.Fatal(err)
	}
	want := "module=example.com/m\nrevision=0123abcd\ntime=2026-01-02T03:04:05Z\nmodified=false\ngo=go1.99\n"
	if out.String() != want {
		t.Errorf("write = %q, want %q", out.String(), want)
	}
}
