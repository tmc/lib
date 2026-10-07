package dcgmjson

import (
	"bytes"
	"errors"
	"os"
	"strings"
	"testing"
)

func TestRead(t *testing.T) {
	b, err := os.ReadFile("testdata/constructed-4.6.1.json")
	if err != nil {
		t.Fatal(err)
	}
	r, err := Read(bytes.NewReader(b), Limits{})
	if err != nil {
		t.Fatal(err)
	}
	if r.Version != "4.6.1" || !bytes.Equal(r.Raw, b) || len(r.Categories) != 1 {
		t.Fatalf("lost facts: %+v", r)
	}
	tests := r.Categories[0].Tests
	if tests[0].Results[0].Status != "Fail" || uint32(tests[0].Results[0].Warnings[0].Code) != 44 || tests[1].ResultsAvailable || tests[1].Summary != nil {
		t.Fatal("status/code/availability differs")
	}
	for _, tc := range []struct{ name, old, new string }{
		{"version", "4.6.1", "4.5.0"}, {"null status", `"status":"Fail"`, `"status":null`},
		{"code", `"error_id":44`, `"error_id":4294967296`},
		{"duplicate", `"version":"4.6.1"`, `"version":"4.6.1","version":"4.6.1"`},
		{"unreported version", `"version":"4.6.1",`, ``},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := Read(strings.NewReader(strings.Replace(string(b), tc.old, tc.new, 1)), Limits{}); err == nil {
				t.Fatal("accepted malformed/unsupported")
			}
		})
	}
	_, err = Read(strings.NewReader(`{"DCGM Diagnostic":{}}`), Limits{})
	var unsupported *UnsupportedError
	if !errors.As(err, &unsupported) || unsupported.Version != "" {
		t.Fatalf("version invented: %v", err)
	}
}
