package main

import (
	"encoding/json"
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
