package main

import (
	"bytes"
	"encoding/json"
	"slices"
	"strings"
	"testing"
)

// TestStampProducerNamesTheBinaryThatReadTheRecord covers the stamp itself.
//
// The interesting arm is the unstamped one. A producer block carrying no
// identity is worse than a missing block, because a reader treats any present
// field as an answer, so a build that cannot say who it is must say nothing
// rather than say it emptily.
func TestStampProducerNamesTheBinaryThatReadTheRecord(t *testing.T) {
	const record = `{"kind":"capability","layout_check":"unavailable","memsets":true}`

	known := producerIdentity{Revision: "2b1a9c4", Modified: true, Time: "2026-09-18T00:00:00Z"}
	got := stampProducer([]byte(record), known)
	var m map[string]json.RawMessage
	if err := json.Unmarshal(got, &m); err != nil {
		t.Fatalf("stamped record does not parse: %v: %s", err, got)
	}
	var p producerIdentity
	if raw, ok := m["producer"]; !ok {
		t.Fatalf("no producer block in %s", got)
	} else if err := json.Unmarshal(raw, &p); err != nil {
		t.Fatalf("producer block does not parse: %v", err)
	}
	if p != known {
		t.Errorf("producer = %+v, want %+v", p, known)
	}
	// The stamp must not cost the record anything it already said.
	if string(m["layout_check"]) != `"unavailable"` || string(m["memsets"]) != "true" {
		t.Errorf("stamping altered the record: %s", got)
	}

	if out := stampProducer([]byte(record), producerIdentity{}); string(out) != record {
		t.Errorf("a build with no identity stamped anyway:\n got %s\nwant %s", out, record)
	}

	// A record this cannot parse keeps its bytes. Trading a record that does
	// not say who wrote it for no record at all is the worse of the two.
	const broken = `{"kind":"capability",`
	if out := stampProducer([]byte(broken), known); string(out) != broken {
		t.Errorf("unparseable record was not returned unchanged: %s", out)
	}
	if out := stampProducer(nil, known); out != nil {
		t.Errorf("nil record became %q", out)
	}
}

// TestProducerIdentityOmitsWhatTheBuildDidNotSay pins the absent-not-empty
// rendering, which is the whole reason the fields are omitempty.
func TestProducerIdentityOmitsWhatTheBuildDidNotSay(t *testing.T) {
	b, err := json.Marshal(producerIdentity{Revision: "2b1a9c4"})
	if err != nil {
		t.Fatal(err)
	}
	if got := string(b); got != `{"revision":"2b1a9c4"}` {
		t.Errorf("marshal = %s, want only the revision: a false modified:false claims the tree was clean", got)
	}
	for _, k := range []string{"modified", "time", "version"} {
		if strings.Contains(string(b), k) {
			t.Errorf("%q present for a build that did not supply it: %s", k, b)
		}
	}
}

// TestStampProducerScopeStatesTheProducersItDoesNotDescribe covers the
// multi-process half of the identity.
//
// A capture that spans processes merges their activity records and publishes
// one capability record, taken from the target's journal. When every process
// ran the same agent that record describes the whole capture. When they did
// not it describes part of it, in exactly the same words, and a reader cannot
// tell which -- the same defect agent_revision was added to fix, one level up.
//
// The quiet arm is the one that keeps the loud arm worth reading. A scope
// block written on every capture would be a caveat a reader learns to skip,
// and then it is not a warning.
func TestStampProducerScopeStatesTheProducersItDoesNotDescribe(t *testing.T) {
	const record = `{"kind":"capability","agent_revision":"c0ffee1","memsets":true}`

	for _, tc := range []struct {
		name      string
		described string
		others    []contributingProducer
	}{
		{"a single producer", "c0ffee1", nil},
		{"every process on the same agent", "c0ffee1", []contributingProducer{
			{process: 42, revision: "c0ffee1", records: 10},
			{process: 43, revision: "c0ffee1", records: 20},
		}},
		{"no agent named itself", "", []contributingProducer{{process: 42, records: 10}}},
	} {
		if got := string(stampProducerScope([]byte(record), tc.described, tc.others)); got != record {
			t.Errorf("%s: record was annotated anyway:\n got %s\nwant %s", tc.name, got, record)
		}
	}

	others := []contributingProducer{
		{process: 43, revision: "deadbee", records: 20},
		{process: 42, revision: "c0ffee1", records: 10},
		{process: 44, records: 5},
	}
	got := stampProducerScope([]byte(record), "c0ffee1", others)
	var m map[string]json.RawMessage
	if err := json.Unmarshal(got, &m); err != nil {
		t.Fatalf("annotated record does not parse: %v: %s", err, got)
	}
	if string(m["memsets"]) != "true" || string(m["agent_revision"]) != `"c0ffee1"` {
		t.Errorf("annotating altered the record: %s", got)
	}
	var scope producerScope
	if raw, ok := m["producer_scope"]; !ok {
		t.Fatalf("no producer_scope in %s", got)
	} else if err := json.Unmarshal(raw, &scope); err != nil {
		t.Fatalf("producer_scope does not parse: %v", err)
	}
	if scope.Described != "c0ffee1" {
		t.Errorf("described = %q, want %q", scope.Described, "c0ffee1")
	}
	// The process that ran the same agent is described by the record and is
	// not an omission; listing it would inflate every count in the note.
	want := []otherProducer{
		{Process: 43, Revision: "deadbee", Records: 20},
		{Process: 44, Records: 5},
	}
	if !slices.Equal(scope.Others, want) {
		t.Errorf("others = %+v, want %+v", scope.Others, want)
	}
	// The note is what a reader who is not querying fields will see, so it has
	// to carry the same facts: how many, which revisions, how many records.
	for _, frag := range []string{"c0ffee1", "deadbee", "2 other process(es)", "25 record(s)", "1 of them an agent that names no revision"} {
		if !strings.Contains(scope.Note, frag) {
			t.Errorf("note does not state %q: %s", frag, scope.Note)
		}
	}

	// An agent that named no revision is still one side of a disagreement.
	// Rendering it as a revision would put "" where a reader expects a value,
	// so the described field is omitted and the note says it in words.
	unnamed := stampProducerScope([]byte(record), "", []contributingProducer{{process: 42, revision: "deadbee", records: 7}})
	if bytes.Contains(unnamed, []byte(`"described"`)) {
		t.Errorf("described written for an agent that named nothing: %s", unnamed)
	}
	if !bytes.Contains(unnamed, []byte("this record describes an agent that names no revision")) {
		t.Errorf("note does not say the described agent is unnamed: %s", unnamed)
	}

	// A record this cannot parse keeps its bytes, for the same reason
	// stampProducer does: no record is worse than an undescribed one.
	const broken = `{"kind":"capability",`
	if out := stampProducerScope([]byte(broken), "c0ffee1", others); string(out) != broken {
		t.Errorf("unparseable record was not returned unchanged: %s", out)
	}
}

// TestAgentRevisionReadsAbsenceAsAbsence pins the reader half of the
// absent-not-empty rule: a record that names no agent and a record whose agent
// named itself must not read the same.
func TestAgentRevisionReadsAbsenceAsAbsence(t *testing.T) {
	if got := agentRevision([]byte(`{"kind":"capability","agent_revision":"c0ffee1"}`)); got != "c0ffee1" {
		t.Errorf("agentRevision = %q, want %q", got, "c0ffee1")
	}
	for _, in := range []string{`{"kind":"capability"}`, `{"kind":"capability",`, ``} {
		if got := agentRevision([]byte(in)); got != "" {
			t.Errorf("agentRevision(%q) = %q, want \"\"", in, got)
		}
	}
}
