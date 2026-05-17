package nbd

import (
	"bytes"
	"context"
	"encoding/binary"
	"io"
	"net"
	"sync"
	"testing"
)

func TestServeDialMemoryDevice(t *testing.T) {
	server, client := net.Pipe()
	defer server.Close()
	defer client.Close()

	dev := NewMemoryDevice(64)
	errc := make(chan error, 1)
	go func() {
		errc <- Serve(context.Background(), server, Export{Name: "disk", Device: dev})
	}()

	c, err := Dial(context.Background(), client, "disk")
	if err != nil {
		t.Fatal(err)
	}
	if c.Size() != 64 {
		t.Fatalf("size = %d, want 64", c.Size())
	}
	if _, err := c.WriteAt([]byte("hello"), 8); err != nil {
		t.Fatal(err)
	}
	got := make([]byte, 5)
	if _, err := c.ReadAt(got, 8); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, []byte("hello")) {
		t.Fatalf("read = %q, want hello", got)
	}
	if err := c.Trim(8, 5); err != nil {
		t.Fatal(err)
	}
	got = make([]byte, 5)
	if _, err := c.ReadAt(got, 8); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, make([]byte, 5)) {
		t.Fatalf("trimmed read = %x, want zeroes", got)
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}
	if err := <-errc; err != nil {
		t.Fatal(err)
	}
}

func TestFlagNoZeroesCompliance(t *testing.T) {
	server, client := net.Pipe()
	defer server.Close()
	defer client.Close()

	errc := make(chan error, 1)
	go func() {
		errc <- Serve(context.Background(), server, Export{Name: "disk", Device: NewMemoryDevice(1)})
	}()

	var old uint64
	var opt uint64
	var hsFlags uint16
	for _, v := range []any{&old, &opt, &hsFlags} {
		if err := binary.Read(client, binary.BigEndian, v); err != nil {
			t.Fatal(err)
		}
	}
	if old != oldMagic || opt != optionMagic || hsFlags&flagNoZeroes == 0 {
		t.Fatalf("handshake = %#x %#x %#x", old, opt, hsFlags)
	}
	if err := binary.Write(client, binary.BigEndian, uint32(flagNoZeroes)); err != nil {
		t.Fatal(err)
	}
	for _, v := range []any{optionMagic, optionExportName, uint32(4)} {
		if err := binary.Write(client, binary.BigEndian, v); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := io.WriteString(client, "disk"); err != nil {
		t.Fatal(err)
	}
	var size uint64
	var flags uint16
	if err := binary.Read(client, binary.BigEndian, &size); err != nil {
		t.Fatal(err)
	}
	if err := binary.Read(client, binary.BigEndian, &flags); err != nil {
		t.Fatal(err)
	}
	if size != 1 {
		t.Fatalf("size = %d, want 1", size)
	}
	if err := writeRequest(client, Request{Type: CommandDisc, Handle: 1}); err != nil {
		t.Fatal(err)
	}
	if err := <-errc; err != nil {
		t.Fatal(err)
	}
}

func TestServerRejectsOversizedRead(t *testing.T) {
	server, client := net.Pipe()
	defer server.Close()
	defer client.Close()

	errc := make(chan error, 1)
	go func() {
		errc <- Serve(context.Background(), server, Export{Name: "disk", Device: NewMemoryDevice(1)})
	}()
	c, err := Dial(context.Background(), client, "disk")
	if err != nil {
		t.Fatal(err)
	}
	handle := c.nextID.Add(1)
	if err := writeRequest(client, Request{Type: CommandRead, Handle: handle, Length: 0xffffffff}); err != nil {
		t.Fatal(err)
	}
	if err := readReply(client, handle, nil); err == nil {
		t.Fatal("oversized read succeeded")
	}
	_ = c.Close()
	if err := <-errc; err != nil {
		t.Fatal(err)
	}
}

func TestClientChunksLargeIO(t *testing.T) {
	server, client := net.Pipe()
	defer server.Close()
	defer client.Close()

	size := uint64(maxPayload + 8)
	dev := NewMemoryDevice(size)
	errc := make(chan error, 1)
	go func() {
		errc <- Serve(context.Background(), server, Export{Name: "disk", Device: dev})
	}()
	c, err := Dial(context.Background(), client, "disk")
	if err != nil {
		t.Fatal(err)
	}
	data := bytes.Repeat([]byte{0x5a}, int(size))
	n, err := c.WriteAt(data, 0)
	if err != nil {
		t.Fatal(err)
	}
	if n != len(data) {
		t.Fatalf("WriteAt n = %d, want %d", n, len(data))
	}
	got := make([]byte, len(data))
	n, err = c.ReadAt(got, 0)
	if err != nil {
		t.Fatal(err)
	}
	if n != len(got) {
		t.Fatalf("ReadAt n = %d, want %d", n, len(got))
	}
	if !bytes.Equal(got, data) {
		t.Fatal("large read/write data mismatch")
	}
	_ = c.Close()
	if err := <-errc; err != nil {
		t.Fatal(err)
	}
}

func TestServerRejectsOutOfBoundsOffset(t *testing.T) {
	server, client := net.Pipe()
	defer server.Close()
	defer client.Close()

	errc := make(chan error, 1)
	go func() {
		errc <- Serve(context.Background(), server, Export{Name: "disk", Device: NewMemoryDevice(1)})
	}()
	c, err := Dial(context.Background(), client, "disk")
	if err != nil {
		t.Fatal(err)
	}
	handle := c.nextID.Add(1)
	if err := writeRequest(client, Request{Type: CommandRead, Handle: handle, Offset: maxInt64 + 1, Length: 1}); err != nil {
		t.Fatal(err)
	}
	if err := readReply(client, handle, nil); err == nil {
		t.Fatal("out-of-bounds read succeeded")
	}
	_ = c.Close()
	if err := <-errc; err != nil {
		t.Fatal(err)
	}
}

func TestConcurrentClientReadAt(t *testing.T) {
	server, client := net.Pipe()
	defer server.Close()
	defer client.Close()

	dev := NewMemoryDevice(128)
	if _, err := dev.WriteAt([]byte("abcdefghijklmnopqrstuvwxyz"), 0); err != nil {
		t.Fatal(err)
	}
	errc := make(chan error, 1)
	go func() {
		errc <- Serve(context.Background(), server, Export{Name: "disk", Device: dev})
	}()
	c, err := Dial(context.Background(), client, "disk")
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			buf := make([]byte, 1)
			if _, err := c.ReadAt(buf, int64(i)); err != nil {
				t.Errorf("ReadAt(%d): %v", i, err)
			}
			if want := byte('a' + i); buf[0] != want {
				t.Errorf("ReadAt(%d) = %q, want %q", i, buf[0], want)
			}
		}(i)
	}
	wg.Wait()
	_ = c.Close()
	if err := <-errc; err != nil {
		t.Fatal(err)
	}
}

func TestServeContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	server, client := net.Pipe()
	defer client.Close()
	errc := make(chan error, 1)
	go func() {
		errc <- Serve(ctx, server, Export{Name: "disk", Device: NewMemoryDevice(1)})
	}()
	cancel()
	if err := <-errc; err == nil {
		t.Fatal("Serve returned nil after cancellation")
	}
}
