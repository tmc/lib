package virtiofs

import (
	"bytes"
	"context"
	"testing"

	"github.com/tmc/lib/fuse"
)

func TestServerHandleLookup(t *testing.T) {
	fs := fuse.NewMemoryFS()
	if _, err := fs.AddFile(fuse.RootID, "hello.txt", []byte("hello")); err != nil {
		t.Fatal(err)
	}
	server, err := NewServer(fs)
	if err != nil {
		t.Fatal(err)
	}
	var req bytes.Buffer
	if err := fuse.WriteRequest(&req, fuse.InHeader{Opcode: fuse.OpcodeLookup, Unique: 22, NodeID: fuse.RootID}, []byte("hello.txt\x00")); err != nil {
		t.Fatal(err)
	}
	resp, err := server.Handle(context.Background(), req.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	h, data, err := fuse.ReadResponse(bytes.NewReader(resp))
	if err != nil {
		t.Fatal(err)
	}
	if h.Unique != 22 || h.Error != 0 || len(data) == 0 {
		t.Fatalf("response = %#v len=%d", h, len(data))
	}
}

func TestMemoryTransportClose(t *testing.T) {
	tp, err := NewMemoryTransport(fuse.NewMemoryFS())
	if err != nil {
		t.Fatal(err)
	}
	if err := tp.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := tp.RoundTrip(context.Background(), nil); err != ErrClosed {
		t.Fatalf("RoundTrip error = %v, want ErrClosed", err)
	}
}
