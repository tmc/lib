package tcpboot

import (
	"fmt"
	"net"
	"testing"
)

// freeAddr reserves an ephemeral localhost port and returns its address.
func freeAddr(t *testing.T) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := ln.Addr().String()
	ln.Close()
	return addr
}

func TestCollectives(t *testing.T) {
	const world = 4
	addr := freeAddr(t)
	errs := make(chan error, world)
	for r := range world {
		go func(rank int) {
			errs <- func() error {
				b, err := New(rank, world, addr)
				if err != nil {
					return err
				}
				defer b.Close()

				if b.Rank() != rank || b.WorldSize() != world {
					return fmt.Errorf("rank/world mismatch")
				}

				// Broadcast from rank 0.
				in0 := ""
				if rank == 0 {
					in0 = "hello from 0"
				}
				got, err := b.BroadcastString(in0, 0)
				if err != nil {
					return err
				}
				if got != "hello from 0" {
					return fmt.Errorf("rank %d: broadcast(0) = %q", rank, got)
				}

				// Barrier round-trips.
				for range 3 {
					if err := b.Barrier(); err != nil {
						return err
					}
				}

				// Broadcast from a non-zero root, with a payload that
				// exercises base64 framing.
				payload := "multi word\npayload"
				in := ""
				if rank == 2 {
					in = payload
				}
				got, err = b.BroadcastString(in, 2)
				if err != nil {
					return err
				}
				if got != payload {
					return fmt.Errorf("rank %d: broadcast(2) = %q", rank, got)
				}
				return nil
			}()
		}(r)
	}
	for range world {
		if err := <-errs; err != nil {
			t.Fatal(err)
		}
	}
}

func TestNewValidatesRank(t *testing.T) {
	if _, err := New(3, 2, "127.0.0.1:0"); err == nil {
		t.Error("rank out of range: want error")
	}
}
