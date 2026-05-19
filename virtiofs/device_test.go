package virtiofs

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"testing"

	"github.com/tmc/lib/fuse"
)

func TestDeviceRoundTripRead(t *testing.T) {
	fs := fuse.NewMemoryFS()
	node, err := fs.AddFile(fuse.RootID, "hello.txt", []byte("hello"))
	if err != nil {
		t.Fatal(err)
	}
	transport, err := NewMemoryTransport(fs)
	if err != nil {
		t.Fatal(err)
	}
	dev, err := NewDevice(DefaultConfig("host"), transport)
	if err != nil {
		t.Fatal(err)
	}
	h, data, err := dev.RoundTrip(context.Background(), fuse.Frame{
		Header: fuse.InHeader{Opcode: fuse.OpcodeRead, Unique: 11, NodeID: node},
		Data:   readBody(fuse.HandleID(node), 0, 5),
	})
	if err != nil {
		t.Fatal(err)
	}
	if h.Unique != 11 || string(data) != "hello" {
		t.Fatalf("response = %#v %q", h, data)
	}
}

func TestDeviceRoundTripResponseError(t *testing.T) {
	fs := fuse.NewMemoryFS()
	transport, err := NewMemoryTransport(fs)
	if err != nil {
		t.Fatal(err)
	}
	dev, err := NewDevice(DefaultConfig("host"), transport)
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = dev.RoundTrip(context.Background(), fuse.Frame{
		Header: fuse.InHeader{Opcode: fuse.OpcodeLookup, Unique: 12, NodeID: fuse.RootID},
		Data:   []byte("missing\x00"),
	})
	var respErr ResponseError
	if !errors.As(err, &respErr) || respErr.Code == 0 {
		t.Fatalf("RoundTrip error = %v, want ResponseError", err)
	}
}

func readBody(fh fuse.HandleID, off uint64, size uint32) []byte {
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
