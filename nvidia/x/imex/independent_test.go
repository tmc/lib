package imex

import (
	"bytes"
	"errors"
	"io"
	"strings"
	"testing"
)

// These constructed inputs exercise the reader, not daemon compatibility.
func TestIndependentProfileAndReportedVersion(t *testing.T) {
	b := bytes.ReplaceAll(spec(t), []byte(`"version":"580.30"`), []byte(`"version":"future"`))
	s, err := Read(bytes.NewReader(b), DCGMSpec58030, Limits{})
	if err != nil || s.Nodes["3"].Version == nil || *s.Nodes["3"].Version != "future" {
		t.Fatalf("explicit profile did not preserve reported version: %+v, %v", s, err)
	}
	for _, profile := range []Profile{"", "580.30", "future"} {
		_, err := Read(bytes.NewReader(spec(t)), profile, Limits{})
		var unsupported *UnsupportedProfileError
		if !errors.As(err, &unsupported) || unsupported.Profile != profile {
			t.Fatalf("unsupported profile %q: %v", profile, err)
		}
	}
}

func TestIndependentExactBoundaries(t *testing.T) {
	b := spec(t)
	s, err := Read(bytes.NewReader(b), DCGMSpec58030, Limits{Bytes: len(b), Nodes: 2, Connections: 4})
	if err != nil || !bytes.Equal(s.Raw, b) {
		t.Fatalf("exact boundary: %v", err)
	}
	for _, limit := range []Limits{{Bytes: len(b) - 1}, {Nodes: 1}, {Connections: 3}} {
		if _, err := Read(bytes.NewReader(b), DCGMSpec58030, limit); err == nil {
			t.Fatalf("accepted exceeded limit %+v", limit)
		}
	}
}

func TestIndependentEmptyAndNestedUnknownFacts(t *testing.T) {
	b := string(spec(t))
	b = strings.ReplaceAll(b, `"version":"580.30"`, `"version":""`)
	b = strings.Replace(b, `"status":"CONNECTED"`, `"status":"FUTURE_LINK","new":{"number":18446744073709551615}`, 1)
	s, err := Read(strings.NewReader(b), DCGMSpec58030, Limits{})
	if err != nil {
		t.Fatal(err)
	}
	n := s.Nodes["3"]
	if n.Version == nil || *n.Version != "" || n.Hostname == nil || *n.Hostname != "" {
		t.Fatal("reported empty strings became absent")
	}
	var found bool
	for _, node := range s.Nodes {
		for _, connection := range node.Connections {
			if connection.Status == "FUTURE_LINK" {
				found = bytes.Equal(connection.Unknown["new"], []byte(`{"number":18446744073709551615}`))
			}
		}
	}
	if !found || string(s.Raw) != b {
		t.Fatal("unknown connection fact lost")
	}
}

func TestIndependentMalformedJSON(t *testing.T) {
	b := string(spec(t))
	for _, raw := range []string{
		b + "null",
		strings.Replace(b, `"status":"DEGRADED"`, `"status":"DEGRADED","stat\u0075s":"UP"`, 1),
		strings.Replace(b, `"changed":true`, `"changed":null`, 1),
		strings.Replace(b, `"hostName":""`, `"hostName":[]`, 1),
	} {
		if _, err := Read(strings.NewReader(raw), DCGMSpec58030, Limits{}); err == nil {
			t.Fatal("malformed JSON accepted")
		}
	}
	_, err := Read(io.MultiReader(strings.NewReader(b), errorReader{}), DCGMSpec58030, Limits{})
	if !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("trailing reader error lost: %v", err)
	}
}
