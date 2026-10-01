package stack

import (
	"bytes"
	"fmt"
	"strings"
	"testing"
)

func TestJournal(t *testing.T) {
	r := Record{Version: Version, Process: "1:boot:start", PID: 1, Language: "go", Attribution: "scope_entry", ExternalKind: 5, ExternalID: 7, Status: "attached", Frames: []Frame{{Function: "leaf"}, {Function: "root"}}}
	var b bytes.Buffer
	j := NewJournal(&b, Limits{Records: 1})
	if err := j.Record(r); err != nil {
		t.Fatal(err)
	}
	if err := j.Record(r); err == nil {
		t.Fatal("record limit accepted")
	}
	got, err := Read(&b, Limits{})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Frames[0].Function != "leaf" {
		t.Fatalf("got %v", got)
	}
}

func TestJournalWithoutWriter(t *testing.T) {
	var journal Journal
	if err := journal.Record(Record{}); err == nil {
		t.Fatal("accepted missing writer")
	}
}
func TestReadRejects(t *testing.T) {
	r := Record{Version: 1, Process: "p", PID: 1, Language: "python", Attribution: "submission", ExternalID: 1, Status: "attached", Frames: []Frame{{Function: "f"}}}
	var b bytes.Buffer
	NewJournal(&b, Limits{}).Record(r)
	valid := b.String()
	for _, tc := range []struct {
		name, input string
		limits      Limits
	}{
		{"duplicate field", strings.Replace(valid, `"version":1`, `"version":1,"version":1`, 1), Limits{}},
		{"unfinished", strings.TrimSuffix(valid, "\n"), Limits{}},
		{"unknown", strings.Replace(valid, `"version":1`, `"unknown":1,"version":1`, 1), Limits{}},
		{"duplicate", valid + valid, Limits{}},
		{"byte", valid, Limits{Bytes: 10}},
		{"version", strings.Replace(valid, `"version":1`, `"version":2`, 1), Limits{}},
		{"attribution", strings.Replace(valid, "submission", "guessed", 1), Limits{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := Read(strings.NewReader(tc.input), tc.limits); err == nil {
				t.Fatal("accepted invalid input")
			}
		})
	}
}
func ExampleJournal_Record() {
	var b bytes.Buffer
	j := NewJournal(&b, Limits{})
	j.Record(Record{Version: Version, Process: "generation", PID: 1, Language: "go", Attribution: "scope_entry", ExternalID: 1, Status: "attached", Frames: []Frame{{Function: "main.work"}}})
	records, _ := Read(&b, Limits{})
	fmt.Println(records[0].Frames[0].Function)
	// Output: main.work
}

func TestCoverage(t *testing.T) {
	r := Record{Version: Version, Process: "p", PID: 1, Session: "s", Generation: 2, Language: "go", Attribution: "submission", ExternalID: 1, Status: "attached", Frames: []Frame{{Function: "f"}}}
	var b bytes.Buffer
	NewJournal(&b, Limits{}).Record(r)
	c := r
	c.Attempted = 1
	c.Recorded = 1
	if err := WriteCoverage(&b, c); err != nil {
		t.Fatal(err)
	}
	got, err := Read(bytes.NewReader(b.Bytes()), Limits{})
	if err != nil || len(got) != 2 {
		t.Fatalf("%v %v", got, err)
	}
	malformed := strings.Replace(b.String(), `"final":true`, `"final":false`, 1)
	if _, err := Read(strings.NewReader(malformed), Limits{}); err == nil {
		t.Fatal("accepted nonfinal coverage")
	}
	c.Generation = 3
	var wrong bytes.Buffer
	NewJournal(&wrong, Limits{}).Record(r)
	WriteCoverage(&wrong, c)
	if _, err := Read(&wrong, Limits{}); err == nil {
		t.Fatal("accepted mismatched generation")
	}
}
