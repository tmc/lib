package fuse

import (
	"bytes"
	"testing"
)

func TestFrameRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	want := InHeader{Opcode: OpcodeGetattr, Unique: 99, NodeID: RootID, UID: 1, GID: 2, PID: 3}
	if err := WriteRequest(&buf, want, []byte("abc")); err != nil {
		t.Fatal(err)
	}
	got, err := ReadFrame(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if got.Header.Opcode != want.Opcode || got.Header.Unique != want.Unique || got.Header.NodeID != want.NodeID {
		t.Fatalf("header = %#v, want %#v", got.Header, want)
	}
	if string(got.Data) != "abc" {
		t.Fatalf("data = %q, want abc", got.Data)
	}
}

func TestResponseRoundTrip(t *testing.T) {
	data := EncodeResponse(7, 0, []byte("ok"))
	h, body, err := ReadResponse(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	if h.Unique != 7 || h.Error != 0 || string(body) != "ok" {
		t.Fatalf("response = %#v %q", h, body)
	}
}
