package nativereceipt

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestNilAndInvalidIdentity(t *testing.T) {
	var s *Session
	if err := s.Capture("stage", "", ""); err == nil {
		t.Fatal("nil capture")
	}
	if err := s.Write("/unused"); err == nil {
		t.Fatal("nil write")
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if err := (Receipt{}).Check(Process{}, "", ""); err == nil {
		t.Fatal("empty proof")
	}
	if _, err := Read("relative"); err == nil {
		t.Fatal("relative receipt")
	}
}
func TestUniqueJSON(t *testing.T) {
	for _, tt := range []struct {
		input string
		bad   bool
	}{
		{`{"a":1,"b":{"a":2}}`, false},
		{`{"a":1,"a":2}`, true},
		{`{"b":{"a":1,"a":2}}`, true},
		{strings.Repeat("[", 34) + "0" + strings.Repeat("]", 34), true},
	} {
		err := uniqueJSON(json.NewDecoder(strings.NewReader(tt.input)), 0)
		if (err != nil) != tt.bad {
			t.Fatalf("input %q: %v", tt.input, err)
		}
	}
}
