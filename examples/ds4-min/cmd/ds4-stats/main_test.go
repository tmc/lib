package main

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunVectorCases(t *testing.T) {
	tests := []struct {
		name string
		data []byte
		rc   int
		want string
	}{
		{name: "valid", data: floats(1, -2, 3), want: "sample: min=-2 max=3 rms=2.1602469"},
		{name: "empty", rc: 1},
		{name: "short", data: []byte{1}, rc: 1},
		{name: "nonfinite", data: floats(float32(math.NaN())), rc: 1},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "vector.bin")
			if err := os.WriteFile(path, test.data, 0o600); err != nil {
				t.Fatal(err)
			}
			var out, errOut bytes.Buffer
			rc := run([]string{"-name", "sample", path}, &out, &errOut)
			if rc != test.rc {
				t.Fatalf("rc=%d, want %d, stderr=%q", rc, test.rc, errOut.String())
			}
			if test.rc == 0 && !strings.Contains(out.String(), test.want) {
				t.Fatalf("output=%q, want substring %q", out.String(), test.want)
			}
		})
	}
}

func TestRunRejectsOversizedVector(t *testing.T) {
	path := filepath.Join(t.TempDir(), "large.bin")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Truncate(int64(maxValues*4 + 4)); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	if rc := run([]string{path}, &bytes.Buffer{}, &bytes.Buffer{}); rc == 0 {
		t.Fatal("oversized vector unexpectedly accepted")
	}
}

func TestRunRejectsUsage(t *testing.T) {
	if rc := run(nil, &bytes.Buffer{}, &bytes.Buffer{}); rc != 2 {
		t.Fatalf("rc=%d, want 2", rc)
	}
}

func floats(values ...float32) []byte {
	data := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(value))
	}
	return data
}
