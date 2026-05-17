package nbd

import (
	"bytes"
	"encoding/binary"
	"testing"
)

func TestRequestRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	want := Request{Flags: 1, Type: CommandWrite, Handle: 42, Offset: 4096, Length: 7}
	if err := writeRequest(&buf, want); err != nil {
		t.Fatal(err)
	}
	got, err := readRequest(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("request = %#v, want %#v", got, want)
	}
}

func TestMagicConstants(t *testing.T) {
	var buf bytes.Buffer
	if err := binary.Write(&buf, binary.BigEndian, oldMagic); err != nil {
		t.Fatal(err)
	}
	if got := buf.String(); got != "NBDMAGIC" {
		t.Fatalf("old magic bytes = %q", got)
	}
}
