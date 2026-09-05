package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestInventory(t *testing.T) {
	root := t.TempDir()
	for _, name := range []string{"internal", "testdata"} {
		if err := os.Mkdir(filepath.Join(root, name), 0755); err != nil {
			t.Fatal(err)
		}
	}
	dir := filepath.Join(root, "example")
	if err := os.Mkdir(dir, 0755); err != nil {
		t.Fatal(err)
	}
	source := "// Package example handles left | right\n// across wrapped lines. More details.\npackage example\n"
	if err := os.WriteFile(filepath.Join(dir, "doc.go"), []byte(source), 0644); err != nil {
		t.Fatal(err)
	}
	if err := run(root, false); err != nil {
		t.Fatal(err)
	}
	if err := run(root, true); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, "README.md")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "[example](example/) | Package example handles left \\| right across wrapped lines.") || strings.Contains(string(data), "More details") {
		t.Fatalf("unexpected inventory:\n%s", data)
	}
	if err := os.WriteFile(path, append(data, '\n'), 0644); err != nil {
		t.Fatal(err)
	}
	if err := run(root, true); err == nil {
		t.Fatal("check accepted stale output")
	}
	if err := os.Remove(filepath.Join(dir, "doc.go")); err != nil {
		t.Fatal(err)
	}
	if err := run(root, false); err == nil {
		t.Fatal("accepted undocumented package")
	}
}

func TestSnapshot(t *testing.T) {
	if err := run("../../../x", true); err != nil {
		t.Fatal(err)
	}
}
