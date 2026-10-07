package ncclras

import (
	"bytes"
	"errors"
	"strings"
	"testing"
)

func TestReadText(t *testing.T) {
	b := legacyFixture(t)
	s, err := ReadText(bytes.NewReader(b), Limits{})
	if err != nil {
		t.Fatal(err)
	}
	if s.Version != "2.24.3" || s.GPUs != 2 || len(s.Groups) != 1 || !bytes.Equal(s.Raw, b) {
		t.Fatalf("unexpected observation: %+v", s)
	}
	for _, tc := range []struct{ name, old, replacement string }{
		{"version", "2.24.3", "2.25.1"},
		{"counts", "      1          2            1          2        2", "      1          2            1          3        2"},
		{"status", "RUNNING", "INVENTED"},
		{"truncated", "Warnings\n========\n", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ReadText(strings.NewReader(strings.Replace(string(b), tc.old, tc.replacement, 1)), Limits{})
			if err == nil {
				t.Fatal("accepted unsupported or malformed response")
			}
		})
	}
	_, err = ReadText(strings.NewReader(strings.Replace(string(b), "2.24.3", "2.25.1", 1)), Limits{})
	var unsupported *UnsupportedError
	if !errors.As(err, &unsupported) || unsupported.Version != "2.25.1" {
		t.Fatalf("reported version lost: %v", err)
	}
	_, err = ReadText(strings.NewReader("{}\n"), Limits{})
	if !errors.As(err, &unsupported) || unsupported.Version != "" {
		t.Fatalf("invented version: %v", err)
	}
}
