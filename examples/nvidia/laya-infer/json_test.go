package main

import (
	"strings"
	"testing"
)

// TestPyFloat checks the repr(float) formatting json.dumps uses. The want
// values come from CPython.
func TestPyFloat(t *testing.T) {
	tests := []struct {
		in   float64
		want string
	}{
		{0.0, "0.0"},
		{1.0, "1.0"},
		{0.5, "0.5"},
		{-0.5, "-0.5"},
		{129.95, "129.95"},
		{3.0, "3.0"},
		{1e-07, "1e-07"},
		{1.2345678901234567e+20, "1.2345678901234567e+20"},
		{1e16, "1e+16"},
		{1e15, "1000000000000000.0"},
		{0.0001, "0.0001"},
		{0.00001, "1e-05"},
		{123456789012345.6, "123456789012345.6"},
		{2.5e-10, "2.5e-10"},
	}
	for _, tt := range tests {
		if got := pyFloat(tt.in); got != tt.want {
			t.Errorf("pyFloat(%v) = %q, want %q", tt.in, got, tt.want)
		}
	}
	if got := pyFloat(negZero()); got != "-0.0" {
		t.Errorf("pyFloat(-0.0) = %q, want %q", got, "-0.0")
	}
}

func negZero() float64 { z := 0.0; return -z }

// TestDumps checks that a parsed document round-trips the way json.dumps
// prints it, preserving key order and integer literals.
func TestDumps(t *testing.T) {
	tests := []struct {
		name         string
		in           string
		ascii, plain string
	}{
		{
			name:  "order and separators",
			in:    `{"z":1,"a":2,"m":[1,2,3]}`,
			ascii: `{"z": 1, "a": 2, "m": [1, 2, 3]}`,
			plain: `{"z": 1, "a": 2, "m": [1, 2, 3]}`,
		},
		{
			name:  "int stays int, float stays float",
			in:    `{"i":3,"f":3.0,"e":1e-7,"n":-0}`,
			ascii: `{"i": 3, "f": 3.0, "e": 1e-07, "n": 0}`,
			plain: `{"i": 3, "f": 3.0, "e": 1e-07, "n": 0}`,
		},
		{
			name:  "literals",
			in:    `{"t":true,"f":false,"z":null}`,
			ascii: `{"t": true, "f": false, "z": null}`,
			plain: `{"t": true, "f": false, "z": null}`,
		},
		{
			name:  "escapes and non-ASCII",
			in:    `{"s":"a\"b\\c\nd\te\u0001f","u":"東京 café"}`,
			ascii: `{"s": "a\"b\\c\nd\te\u0001f", "u": "\u6771\u4eac caf\u00e9"}`,
			plain: `{"s": "a\"b\\c\nd\te\u0001f", "u": "東京 café"}`,
		},
		{
			name:  "surrogate pair",
			in:    `{"e":"😀"}`,
			ascii: `{"e": "\ud83d\ude00"}`,
			plain: `{"e": "😀"}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v, err := parseJSON([]byte(tt.in))
			if err != nil {
				t.Fatalf("parseJSON: %v", err)
			}
			if got := v.dumpString(true); got != tt.ascii {
				t.Errorf("dumps(ascii) = %s, want %s", got, tt.ascii)
			}
			if got := v.dumpString(false); got != tt.plain {
				t.Errorf("dumps(plain) = %s, want %s", got, tt.plain)
			}
		})
	}
}

// TestPutNum checks the round(x, 4) formatting used for every number in a reply.
func TestPutNum(t *testing.T) {
	tests := []struct {
		in   float64
		want string
	}{
		{1.0, "1.0"},
		{0.0, "0.0"},
		{0.97284, "0.9728"},
		{0.011, "0.011"},
		{0.0072, "0.0072"},
		{1.43961, "1.4396"},
		{1e-9, "0.0"},
		{-1e-9, "0.0"},
		{-0.25, "-0.25"},
	}
	for _, tt := range tests {
		if got := putNum(tt.in); got != tt.want {
			t.Errorf("putNum(%v) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

// TestParseQuestion covers the option rendering of each question type.
func TestParseQuestion(t *testing.T) {
	tests := []struct {
		name   string
		in     string
		typ    int
		opts   []string
		labels []string // choice only; nil skips the check
	}{
		{
			name: "choice object",
			in:   `{"type":"choice","instructions":"pick","criteria":{"a":"first","b":null,"c":{"x":1},"d":""}}`,
			typ:  0,
			opts: []string{"a: first", "b", `c: {"x": 1}`, "d"},
		},
		{
			name: "choice list",
			in:   `{"type":"choice","instructions":"pick","criteria":["x","y"]}`,
			typ:  0,
			opts: []string{"x", "y"},
		},
		{
			name:   "choice label objects",
			in:     `{"type":"choice","instructions":"pick","criteria":[{"label":"a","description":"first"},{"label":"b"},{"label":"c","description":{"x":"é"}},{"label":"d","description":null},"e"]}`,
			typ:    0,
			opts:   []string{"a: first", "b", `c: {"x": "é"}`, "d", "e"},
			labels: []string{"a", "b", "c", "d", "e"},
		},
		{
			name: "score",
			in:   `{"type":"score","instructions":"rate","criteria":["none",{"l":2}]}`,
			typ:  1,
			opts: []string{"level 0: none", `level 1: {"l": 2}`},
		},
		{
			name: "noul defaults",
			in:   `{"type":"noul","instructions":"claim"}`,
			typ:  2,
			opts: []string{"false: no, the statement does not hold", "true: yes, the statement holds"},
		},
		{
			name: "noul criteria",
			in:   `{"type":"noul","instructions":"claim","criteria":{"true":"yep","false":""}}`,
			typ:  2,
			opts: []string{"false: no, the statement does not hold", "true: yep"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v, err := parseJSON([]byte(tt.in))
			if err != nil {
				t.Fatalf("parseJSON: %v", err)
			}
			q, err := parseQuestion("q", v)
			if err != nil {
				t.Fatalf("parseQuestion: %v", err)
			}
			if q.typ != tt.typ {
				t.Errorf("typ = %d, want %d", q.typ, tt.typ)
			}
			if strings.Join(q.opts, "|") != strings.Join(tt.opts, "|") {
				t.Errorf("opts = %q, want %q", q.opts, tt.opts)
			}
			if tt.labels != nil && strings.Join(q.labels, "|") != strings.Join(tt.labels, "|") {
				t.Errorf("labels = %q, want %q", q.labels, tt.labels)
			}
		})
	}
}

// TestParseQuestionErrors checks the requests that must be rejected.
func TestParseQuestionErrors(t *testing.T) {
	for _, in := range []string{
		`{"instructions":"x","criteria":["a"]}`,
		`{"type":"nope","instructions":"x","criteria":["a"]}`,
		`{"type":"choice","criteria":["a"]}`,
		`{"type":"choice","instructions":"x"}`,
		`{"type":"choice","instructions":"x","criteria":[]}`,
		`{"type":"choice","instructions":"x","criteria":[1,2]}`,
		`{"type":"score","instructions":"x","criteria":{"a":1}}`,
		`{"type":"noul","instructions":"x","criteria":["a"]}`,
		`{"type":"choice","instructions":"x","criteria":["a","a"]}`,
		`{"type":"choice","instructions":"x","criteria":["a",""]}`,
		`{"type":"choice","instructions":"x","criteria":{"":"empty"}}`,
		`{"type":"choice","instructions":"x","criteria":[{"label":"a"},{"label":"a","description":"again"}]}`,
		`{"type":"choice","instructions":"x","criteria":[{"label":"a"},"a"]}`,
		`{"type":"choice","instructions":"x","criteria":[{"label":""}]}`,
		`{"type":"choice","instructions":"x","criteria":[{"description":"no label"}]}`,
		`{"type":"choice","instructions":"x","criteria":[{"label":3}]}`,
	} {
		v, err := parseJSON([]byte(in))
		if err != nil {
			t.Fatalf("parseJSON(%s): %v", in, err)
		}
		if _, err := parseQuestion("q", v); err == nil {
			t.Errorf("parseQuestion(%s) succeeded, want error", in)
		}
	}
}

// TestInstructionsJSON checks that non-string instructions are rendered with
// json.dumps' ensure_ascii=True, as the reference implementation does.
func TestInstructionsJSON(t *testing.T) {
	v, err := parseJSON([]byte(`{"type":"noul","instructions":{"ask":"東京?","n":2}}`))
	if err != nil {
		t.Fatal(err)
	}
	q, err := parseQuestion("q", v)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"ask": "\u6771\u4eac?", "n": 2}`
	if q.ins != want {
		t.Errorf("ins = %s, want %s", q.ins, want)
	}
}

// TestParseRequest checks both spellings of the question set, and that each
// keeps the request's order.
func TestParseRequest(t *testing.T) {
	const q = `{"type":"noul","instructions":"claim"}`
	tests := []struct {
		name  string
		in    string
		state string
		ids   []string // nil means the request must be rejected
	}{
		{
			name:  "object",
			in:    `{"state":"s","questions":{"z":` + q + `,"a":` + q + `}}`,
			state: "s",
			ids:   []string{"z", "a"},
		},
		{
			name:  "array",
			in:    `{"state":{"k":1},"questions":[{"id":"z","type":"noul","instructions":"claim"},{"type":"noul","id":"a","instructions":"claim"}]}`,
			state: `{"k": 1}`,
			ids:   []string{"z", "a"},
		},
		{name: "object duplicate", in: `{"state":"s","questions":{"a":` + q + `,"a":` + q + `}}`},
		{name: "array duplicate", in: `{"state":"s","questions":[{"id":"a","type":"noul","instructions":"x"},{"id":"a","type":"noul","instructions":"y"}]}`},
		{name: "array missing id", in: `{"state":"s","questions":[` + q + `]}`},
		{name: "array empty id", in: `{"state":"s","questions":[{"id":"","type":"noul","instructions":"x"}]}`},
		{name: "array numeric id", in: `{"state":"s","questions":[{"id":1,"type":"noul","instructions":"x"}]}`},
		{name: "array of strings", in: `{"state":"s","questions":["a"]}`},
		{name: "questions string", in: `{"state":"s","questions":"a"}`},
		{name: "no state", in: `{"questions":{"a":` + q + `}}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r, err := parseRequest([]byte(tt.in))
			if tt.ids == nil {
				if err == nil {
					t.Fatalf("parseRequest succeeded, want error")
				}
				return
			}
			if err != nil {
				t.Fatalf("parseRequest: %v", err)
			}
			if r.state != tt.state {
				t.Errorf("state = %s, want %s", r.state, tt.state)
			}
			if strings.Join(r.ids, "|") != strings.Join(tt.ids, "|") {
				t.Errorf("ids = %q, want %q", r.ids, tt.ids)
			}
			if len(r.questions) != len(r.ids) {
				t.Errorf("%d questions for %d ids", len(r.questions), len(r.ids))
			}
		})
	}
}
