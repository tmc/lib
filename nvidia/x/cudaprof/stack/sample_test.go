package stack

import (
	"bytes"
	"encoding/json"
	"testing"
)

func TestReadSamples(t *testing.T) {
	r := Sample{Version: 1, Type: "sample", Process: "g", PID: 1, Session: "s", Generation: 2, Language: "python", Time: 100, Clock: "monotonic", Status: "clock_unavailable", Provenance: "wall_stack_snapshot", Frames: []Frame{{Function: "f"}}}
	var b bytes.Buffer
	json.NewEncoder(&b).Encode(r)
	c := r
	c.Type = "coverage"
	c.Frames = nil
	c.Final = true
	c.Attempted = 1
	c.Recorded = 1
	json.NewEncoder(&b).Encode(c)
	got, err := ReadSamples(bytes.NewReader(b.Bytes()), Limits{})
	if err != nil || len(got) != 2 {
		t.Fatalf("%v %v", got, err)
	}
	r.Generation = 3
	json.NewEncoder(&b).Encode(r)
	if _, err := ReadSamples(&b, Limits{}); err == nil {
		t.Fatal("accepted sample after coverage")
	}
}
