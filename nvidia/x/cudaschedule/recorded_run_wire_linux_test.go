//go:build linux

package cudaschedule

import (
	"context"
	"errors"
	"net"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestRecordedWire(t *testing.T) {
	listener, err := net.ListenUnix("unixpacket", &net.UnixAddr{Name: "@recorded-wire-" + strconv.Itoa(os.Getpid()) + "-" + strconv.FormatInt(time.Now().UnixNano(), 10), Net: "unixpacket"})
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	client, err := net.DialUnix("unixpacket", nil, listener.Addr().(*net.UnixAddr))
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	server, err := recordedAccept(ctx, listener)
	if err != nil {
		t.Fatal(err)
	}
	defer server.Close()
	nonce := strings.Repeat("a", 64)
	msg := recordedMessage{Version: 1, Nonce: nonce, Kind: "ready"}
	if err := recordedSend(ctx, client, msg); err != nil {
		t.Fatal(err)
	}
	got, err := recordedReceive(ctx, server, nonce)
	if err != nil || got != msg {
		t.Fatalf("packet substitution: %+v %v", got, err)
	}
	short, shortCancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer shortCancel()
	start := time.Now()
	if _, err := recordedReceive(short, server, nonce); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("missing receive bound: %v", err)
	}
	if time.Since(start) > 300*time.Millisecond {
		t.Fatal("receive did not obey deadline")
	}
	if _, err := recordedReceive(context.Background(), server, nonce); err == nil {
		t.Fatal("accepted unbounded receive")
	}
	if err := client.SetWriteDeadline(time.Now().Add(time.Second)); err != nil {
		t.Fatal(err)
	}
	if _, err := client.Write([]byte(strings.Repeat("x", recordedMessageLimit+1))); err != nil {
		t.Fatal(err)
	}
	if _, err := recordedReceive(ctx, server, nonce); err == nil {
		t.Fatal("accepted oversized packet")
	}
}
