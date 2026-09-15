package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestOfflineArtifacts(t *testing.T) {
	dir := t.TempDir()
	input := filepath.Join(dir, "kernel.sass")
	output := filepath.Join(dir, "report.html")
	if err := os.WriteFile(input, []byte("Function : k\n/*0000*/ EXIT;\n"), 0600); err != nil {
		t.Fatal(err)
	}
	var b bytes.Buffer
	if err := run([]string{"-html=" + output, input}, &b); err != nil {
		t.Fatal(err)
	}
	html, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(html), "EXIT;") {
		t.Fatal("missing artifact")
	}
	if strings.Contains(string(html), "/api/") {
		t.Fatal("offline artifact page contains server route")
	}
}
