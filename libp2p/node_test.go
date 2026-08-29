package libp2p

import (
	"context"
	"io"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/protocol"
)

func TestLoopbackStream(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	a, err := New(ctx, DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	b, err := New(ctx, DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()

	const proto = protocol.ID("/tmc/lib/libp2p/test/1.0.0")
	if err := b.Handle(proto, func(s Stream) {
		defer s.Close()
		buf := make([]byte, 5)
		if _, err := io.ReadFull(s, buf); err != nil {
			t.Errorf("read request: %v", err)
			return
		}
		if string(buf) != "hello" {
			t.Errorf("request = %q", buf)
			return
		}
		if _, err := s.Write([]byte("ack")); err != nil {
			t.Errorf("write response: %v", err)
		}
	}); err != nil {
		t.Fatal(err)
	}

	if err := a.Connect(ctx, b.AddrInfo()); err != nil {
		t.Fatal(err)
	}
	stream, err := a.OpenStream(ctx, b.ID(), proto)
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	if _, err := stream.Write([]byte("hello")); err != nil {
		t.Fatal(err)
	}
	reply := make([]byte, 3)
	if _, err := io.ReadFull(stream, reply); err != nil {
		t.Fatal(err)
	}
	if string(reply) != "ack" {
		t.Fatalf("reply = %q", reply)
	}
}

func TestNodeClosed(t *testing.T) {
	var n *Node
	if err := n.Connect(context.Background(), peer.AddrInfo{}); err != ErrClosed {
		t.Fatalf("nil Connect err = %v, want ErrClosed", err)
	}
}
