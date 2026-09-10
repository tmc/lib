package main

import "testing"

func TestEngineOutputPath(t *testing.T) {
	tests := []struct {
		path string
		n    int
		want string
	}{
		{"", 0, ""},
		{"", 3, ""},
		{"-", 1, "-"},
		{"report.txt", 0, "report.txt"},
		{"report.txt", 1, "report-2.txt"},
		{"report.txt", 2, "report-3.txt"},
		{"out/trace.perfetto-trace", 1, "out/trace-2.perfetto-trace"},
		{"noext", 1, "noext-2"},
		{"dir.d/file", 1, "dir.d/file-2"},
	}
	for _, tt := range tests {
		if got := engineOutputPath(tt.path, tt.n); got != tt.want {
			t.Errorf("engineOutputPath(%q, %d) = %q, want %q", tt.path, tt.n, got, tt.want)
		}
	}
}
