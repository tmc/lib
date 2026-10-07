package imex

import (
	"bytes"
	"errors"
	"io"
	"os"
	"strings"
	"testing"
)

func spec(t testing.TB) []byte {
	t.Helper()
	b, err := os.ReadFile("testdata/dcgm-spec-580.30.json")
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestSpecificationFacts(t *testing.T) {
	b := spec(t)
	s, err := Read(bytes.NewReader(b), DCGMSpec58030, Limits{})
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(s.Raw, b) || s.Status != "DEGRADED" || s.Timestamp != "5/22/2025 19:46:40.891" || len(s.Nodes) != 2 {
		t.Fatalf("snapshot %+v", s)
	}
	n := s.Nodes["3"]
	if n.Version == nil || *n.Version != "580.30" || n.Hostname == nil || *n.Hostname != "" || n.Host != "10.76.186.255" || n.Status != "READY" || !n.Changed || n.Connections["6"].Status != "CONNECTED" {
		t.Fatalf("node %+v", n)
	}
	if _, ok := s.Nodes["6"]; ok {
		t.Fatal("invented missing member")
	}
	if _, ok := s.Nodes["1"].Connections["3"]; ok {
		t.Fatal("invented reciprocal connection")
	}
}

// Constructed variants check reader contracts, not daemon measurements.
func TestAvailabilityAndUnknownFacts(t *testing.T) {
	b := string(spec(t))
	b = strings.ReplaceAll(b, `"version":"580.30",`, "")
	b = strings.ReplaceAll(b, `"hostName":""`, `"future":{"counter":18446744073709551615}`)
	b = strings.Replace(b, `"status":"DEGRADED"`, `"status":"FUTURE_STATE","extra":[1,2]`, 1)
	s, err := Read(strings.NewReader(b), DCGMSpec58030, Limits{})
	if err != nil {
		t.Fatal(err)
	}
	if s.Nodes["3"].Version != nil || s.Nodes["3"].Hostname != nil || s.Status != "FUTURE_STATE" || string(s.Unknown["extra"]) != "[1,2]" || !bytes.Contains(s.Nodes["3"].Unknown["future"], []byte("18446744073709551615")) || string(s.Raw) != b {
		t.Fatal("lost unavailable or unknown facts")
	}
}

func TestRefusals(t *testing.T) {
	b := string(spec(t))
	for _, tc := range []struct {
		name, body string
		limits     Limits
	}{
		{"empty", "", Limits{}}, {"null", "null", Limits{}}, {"array", "[]", Limits{}},
		{"trailing", b + "{}", Limits{}}, {"duplicate", strings.Replace(b, `"status":"DEGRADED"`, `"status":"DEGRADED","status":"UP"`, 1), Limits{}},
		{"alias", strings.Replace(b, `"status":"DEGRADED"`, `"status":"DEGRADED","Status":"UP"`, 1), Limits{}},
		{"null_flag", strings.Replace(b, `"changed":true`, `"changed":null`, 1), Limits{}},
		{"wrong_flag", strings.Replace(b, `"changed":true`, `"changed":1`, 1), Limits{}},
		{"null_version", strings.Replace(b, `"version":"580.30"`, `"version":null`, 1), Limits{}},
		{"wrong_node", strings.Replace(b, `"3": {`, `"03": {`, 1), Limits{}},
		{"null_connections", strings.Replace(b, `"connections":{`, `"connections":null,"other":{`, 1), Limits{}},
		{"bytes", b, Limits{Bytes: len(b) - 1}}, {"nodes", b, Limits{Nodes: 1}}, {"connections", b, Limits{Connections: 3}},
		{"depth", b, Limits{Depth: 2}}, {"invalid_limit", b, Limits{Nodes: -1}},
		{"missing_status", strings.Replace(b, `"status":"DEGRADED"`, `"other":"DEGRADED"`, 1), Limits{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := Read(strings.NewReader(tc.body), DCGMSpec58030, tc.limits); err == nil {
				t.Fatal("accepted malformed snapshot")
			}
		})
	}
	var unsupported *UnsupportedProfileError
	if _, err := Read(strings.NewReader(b), Profile("580.30"), Limits{}); !errors.As(err, &unsupported) {
		t.Fatalf("unsupported type %v", err)
	}
	if _, err := Read(nil, DCGMSpec58030, Limits{}); err == nil {
		t.Fatal("nil reader")
	}
}

type errorReader struct{}

func (errorReader) Read([]byte) (int, error) { return 0, io.ErrUnexpectedEOF }
func TestReaderError(t *testing.T) {
	if _, err := Read(errorReader{}, DCGMSpec58030, Limits{}); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("reader error %v", err)
	}
}

func FuzzRead(f *testing.F) {
	f.Add(spec(f))
	f.Add([]byte("{}"))
	f.Fuzz(func(t *testing.T, b []byte) {
		s, err := Read(bytes.NewReader(b), DCGMSpec58030, Limits{Bytes: 65536, Nodes: 64, Connections: 128})
		if err == nil && !bytes.Equal(s.Raw, b) {
			t.Fatal("raw drift")
		}
	})
}
