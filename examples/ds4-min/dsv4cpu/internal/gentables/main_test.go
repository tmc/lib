package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestRejectSource(t *testing.T) {
	dir := t.TempDir()
	out := filepath.Join(dir, "tables.go")
	if err := run("", out); err == nil {
		t.Fatal("missing source accepted")
	}
	bad := filepath.Join(dir, "bad.c")
	if err := os.WriteFile(bad, []byte("not pinned source"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := run(bad, out); err == nil {
		t.Fatal("malformed source accepted")
	}
	if _, err := os.Stat(out); !os.IsNotExist(err) {
		t.Fatalf("rejected source published output: %v", err)
	}
}

func TestDeterministicRegeneration(t *testing.T) {
	source := os.Getenv("DS4_CPU_ORACLE_SOURCE")
	if source == "" {
		t.Skip("DS4_CPU_ORACLE_SOURCE not set")
	}
	dir := t.TempDir()
	a, b := filepath.Join(dir, "a.go"), filepath.Join(dir, "b.go")
	if err := run(source, a); err != nil {
		t.Fatal(err)
	}
	if err := run(source, b); err != nil {
		t.Fatal(err)
	}
	x, err := os.ReadFile(a)
	if err != nil {
		t.Fatal(err)
	}
	y, err := os.ReadFile(b)
	if err != nil {
		t.Fatal(err)
	}
	tracked, err := os.ReadFile("../../iq2_tables.go")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(x, y) || !bytes.Equal(x, tracked) {
		t.Fatal("regeneration differs")
	}
}
