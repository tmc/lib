package fuse

import (
	"bytes"
	"context"
	"encoding/binary"
	"net"
	"testing"
)

func TestMemoryFSLookupReadWrite(t *testing.T) {
	fs := NewMemoryFS()
	id, err := fs.AddFile(RootID, "hello.txt", []byte("hello"))
	if err != nil {
		t.Fatal(err)
	}
	attr, err := fs.Getattr(context.Background(), id)
	if err != nil {
		t.Fatal(err)
	}
	if attr.Size != 5 {
		t.Fatalf("size = %d, want 5", attr.Size)
	}
	data, err := fs.Read(context.Background(), id, HandleID(id), 0, 5)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "hello" {
		t.Fatalf("read = %q, want hello", data)
	}
	if _, err := fs.Write(context.Background(), id, HandleID(id), 5, []byte("!")); err != nil {
		t.Fatal(err)
	}
	data, _ = fs.Read(context.Background(), id, HandleID(id), 0, 6)
	if string(data) != "hello!" {
		t.Fatalf("read = %q, want hello!", data)
	}
}

func TestMemoryFSRejectsHugeGrowth(t *testing.T) {
	fs := NewMemoryFS()
	id, err := fs.AddFile(RootID, "hello.txt", []byte("hello"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := fs.Write(context.Background(), id, HandleID(id), 1<<62, []byte("x")); err != ErrInvalidArgument {
		t.Fatalf("Write error = %v, want ErrInvalidArgument", err)
	}
	if _, err := fs.Setattr(context.Background(), id, Setattr{Valid: 8, Size: 1 << 62}); err != ErrInvalidArgument {
		t.Fatalf("Setattr error = %v, want ErrInvalidArgument", err)
	}
}

func TestServeMemoryFS(t *testing.T) {
	server, client := net.Pipe()
	defer server.Close()
	defer client.Close()
	fs := NewMemoryFS()
	id, err := fs.AddFile(RootID, "a", []byte("abc"))
	if err != nil {
		t.Fatal(err)
	}
	errc := make(chan error, 1)
	go func() { errc <- Serve(context.Background(), server, fs) }()

	if err := WriteRequest(client, InHeader{Opcode: OpcodeRead, Unique: 1, NodeID: id}, readBody(HandleID(id), 0, 3)); err != nil {
		t.Fatal(err)
	}
	h, data, err := ReadResponse(client)
	if err != nil {
		t.Fatal(err)
	}
	if h.Error != 0 || string(data) != "abc" {
		t.Fatalf("read response = %#v %q", h, data)
	}
	_ = client.Close()
	<-errc
}

func TestHandleFrameLookup(t *testing.T) {
	fs := NewMemoryFS()
	if _, err := fs.AddFile(RootID, "a", []byte("abc")); err != nil {
		t.Fatal(err)
	}
	resp, err := HandleFrame(context.Background(), fs, Frame{Header: InHeader{Opcode: OpcodeLookup, Unique: 2, NodeID: RootID}, Data: []byte("a\x00")})
	if err != nil {
		t.Fatal(err)
	}
	h, body, err := ReadResponse(bytes.NewReader(resp))
	if err != nil {
		t.Fatal(err)
	}
	if h.Error != 0 || len(body) == 0 {
		t.Fatalf("lookup response = %#v len=%d", h, len(body))
	}
}

func readBody(fh HandleID, off uint64, size uint32) []byte {
	var buf bytes.Buffer
	_ = binary.Write(&buf, binary.LittleEndian, uint64(fh))
	_ = binary.Write(&buf, binary.LittleEndian, off)
	_ = binary.Write(&buf, binary.LittleEndian, size)
	_ = binary.Write(&buf, binary.LittleEndian, uint32(0))
	_ = binary.Write(&buf, binary.LittleEndian, uint64(0))
	_ = binary.Write(&buf, binary.LittleEndian, uint32(0))
	_ = binary.Write(&buf, binary.LittleEndian, uint32(0))
	return buf.Bytes()
}
