//go:build linux

package main

import (
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLiteralMXFP4QuantizerHashRefusal(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "cuda", "mmq")
	if err := os.MkdirAll(path, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(path, "quantize.cu"), []byte("changed upstream arithmetic"), 0600); err != nil {
		t.Fatal(err)
	}
	source, err := literalMXFP4Quantizer(dir)
	if source != "" || err == nil || !strings.Contains(err.Error(), "pinned C quantize.cu hash mismatch") {
		t.Fatalf("got source %q error %v", source, err)
	}
}

func TestLiteralMXFP4Read(t *testing.T) {
	for _, name := range []string{"quantize.cu", "common.cuh"} {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "cuda", "mmq")
			if err := os.MkdirAll(path, 0700); err != nil {
				t.Fatal(err)
			}
			data := []byte("pinned source\n")
			want := fmt.Sprintf("%x", sha256.Sum256(data))
			file := filepath.Join(path, name)
			if _, err := literalMXFP4Read(dir, name, want); !os.IsNotExist(err) {
				t.Fatalf("missing file: %v", err)
			}
			if err := os.WriteFile(file, data, 0600); err != nil {
				t.Fatal(err)
			}
			got, err := literalMXFP4Read(dir, name, want)
			if err != nil || got != string(data) {
				t.Fatalf("read: %q %v", got, err)
			}
			if err := os.WriteFile(file, append(data, ' '), 0600); err != nil {
				t.Fatal(err)
			}
			got, err = literalMXFP4Read(dir, name, want)
			if got != "" || err == nil || !strings.Contains(err.Error(), "pinned C "+name+" hash mismatch") {
				t.Fatalf("mutation accepted: %q %v", got, err)
			}
		})
	}
}

func TestLiteralMXFP4Part(t *testing.T) {
	for _, tc := range []struct{ name, source, want, errorText string }{
		{"exact", "before START arithmetic END after", "START arithmetic ", ""},
		{"missing-start", "arithmetic END", "", "missing literal C start"},
		{"missing-end", "START arithmetic", "", "missing literal C end"},
		{"end-before-start", "END START arithmetic", "", "missing literal C end"},
		{"first-end", "START arithmetic END END", "START arithmetic ", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := literalMXFP4Part(tc.source, "START", "END")
			if tc.errorText != "" {
				if got != "" || err == nil || !strings.Contains(err.Error(), tc.errorText) {
					t.Fatalf("got %q error %v", got, err)
				}
			} else if err != nil || got != tc.want {
				t.Fatalf("got %q error %v want %q", got, err, tc.want)
			}
		})
	}
}
