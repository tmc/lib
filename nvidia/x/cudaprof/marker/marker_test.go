package marker

import (
	"errors"
	"testing"
)

func TestParse(t *testing.T) {
	for _, tt := range []struct {
		name, input     string
		valid, ordinary bool
	}{
		{"missing", "", false, true},
		{"inferred", "arg_reduce", false, true},
		{"unsupported", `cudaprof:v2 {"phase":"decode"}`, false, false},
		{"phase", `cudaprof:v1 {"phase":"decode"}`, true, false},
		{"optional", `cudaprof:v1 {"phase":"decode","token":0,"layer":0,"lookahead":false,"materialization":true}`, true, false},
		{"no phase", `cudaprof:v1 {}`, false, false},
		{"bad phase", `cudaprof:v1 {"phase":"other"}`, false, false},
		{"negative", `cudaprof:v1 {"phase":"decode","token":-1}`, false, false},
		{"fractional", `cudaprof:v1 {"phase":"decode","layer":1.2}`, false, false},
		{"case alias", `cudaprof:v1 {"phase":"decode","PHASE":"prefill"}`, false, false},
		{"case field", `cudaprof:v1 {"Phase":"decode"}`, false, false},
		{"unknown", `cudaprof:v1 {"phase":"decode","unknown":"float32"}`, false, false},
		{"empty dtype", `cudaprof:v1 {"phase":"decode","dtype":""}`, false, false},
		{"negative cache", `cudaprof:v1 {"phase":"decode","cache_length":-1}`, false, false},
		{"shape", `cudaprof:v1 {"phase":"decode","dtype":"float32","cache_length":10}`, true, false},
		{"duplicate", `cudaprof:v1 {"phase":"decode","phase":"prefill"}`, false, false},
		{"null", `cudaprof:v1 {"phase":"decode","token":null}`, false, false},
		{"malformed", `cudaprof:v1 {`, false, false},
		{"trailing", `cudaprof:v1 {"phase":"decode"} {}`, false, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			m, err := Parse(tt.input)
			if (err == nil) != tt.valid {
				t.Fatalf("Parse = %+v, %v", m, err)
			}
			if errors.Is(err, ErrNotMarker) != tt.ordinary {
				t.Fatalf("ordinary error = %v", err)
			}
			if tt.valid {
				text, err := Encode(m)
				if err != nil {
					t.Fatal(err)
				}
				if _, err := Parse(text); err != nil {
					t.Fatal(err)
				}
			}
		})
	}
}

func TestParseRange(t *testing.T) {
	for _, name := range []string{`cudaprof:v1 {"phase":"decode"}`, `inference/cudaprof:v1 {"phase":"decode"}`, `domain/subdomain/cudaprof:v1 {"phase":"decode"}`} {
		if _, err := ParseRange(name); err != nil {
			t.Errorf("ParseRange(%q): %v", name, err)
		}
	}
	if _, err := ParseRange(`other cudaprof:v1 {"phase":"decode"}`); !errors.Is(err, ErrNotMarker) {
		t.Fatal(err)
	}
}

func TestParseRangeUnsupported(t *testing.T) {
	_, err := ParseRange(`domain/cudaprof:v2 {"phase":"decode"}`)
	if err == nil || errors.Is(err, ErrNotMarker) {
		t.Fatalf("unsupported version error = %v", err)
	}
}

func TestDeliveryEvents(t *testing.T) {
	token := 3
	for _, event := range []string{"delivered", "eos", "token_limit"} {
		message, err := Encode(Metadata{Phase: "decode", Token: &token, Event: event})
		if err != nil {
			t.Fatal(err)
		}
		m, err := Parse(message)
		if err != nil || m.Event != event {
			t.Fatalf("Parse(%q) = %+v, %v", message, m, err)
		}
	}
	for _, m := range []Metadata{{Phase: "decode", Event: "delivered"}, {Phase: "decode", Token: &token, Event: "unknown"}} {
		if _, err := Encode(m); err == nil {
			t.Fatalf("Encode(%+v) succeeded", m)
		}
	}
}

func TestRequestMetadata(t *testing.T) {
	for _, tt := range []struct {
		name, input string
		valid       bool
	}{
		{"request", `cudaprof:v2 {"phase":"cleanup","event":"request_end","request_id":"r","role":"measured","boundary":"completion","execution_unit":"logical_forward","executions":32}`, true},
		{"legacy request", `cudaprof:v1 {"phase":"decode","request_id":"r"}`, false},
		{"legacy cleanup", `cudaprof:v1 {"phase":"cleanup"}`, false},
		{"duplicate request", `cudaprof:v2 {"phase":"decode","request_id":"r","request_id":"s","role":"measured"}`, false},
		{"missing role", `cudaprof:v2 {"phase":"decode","request_id":"r"}`, false},
		{"bad boundary", `cudaprof:v2 {"phase":"decode","request_id":"r","role":"measured","boundary":"inferred"}`, false},
		{"negative executions", `cudaprof:v2 {"phase":"decode","request_id":"r","role":"measured","execution_unit":"logical_forward","executions":-1}`, false},
		{"missing optional observations", `cudaprof:v2 {"phase":"decode","request_id":"r","role":"warmup"}`, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			m, err := Parse(tt.input)
			if (err == nil) != tt.valid {
				t.Fatalf("Parse = %+v, %v", m, err)
			}
			if tt.valid {
				text, err := Encode(m)
				if err != nil {
					t.Fatal(err)
				}
				if _, err := Parse(text); err != nil {
					t.Fatal(err)
				}
			}
		})
	}
}
