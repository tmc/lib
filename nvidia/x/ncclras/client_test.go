package ncclras

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"strings"
	"testing"
	"time"
)

func serve(t *testing.T, fn func(net.Conn)) string {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan struct{})
	go func() {
		defer close(done)
		c, err := listener.Accept()
		if err != nil {
			return
		}
		defer c.Close()
		c.SetDeadline(time.Now().Add(2 * time.Second))
		fn(c)
	}()
	t.Cleanup(func() {
		listener.Close()
		select {
		case <-done:
		case <-time.After(3 * time.Second):
			t.Error("loopback worker did not retire")
		}
	})
	return listener.Addr().String()
}

func TestClientFraming(t *testing.T) {
	b := statusFixture(t)
	frames := make(chan []string, 1)
	addr := serve(t, func(c net.Conn) {
		r := bufio.NewReader(c)
		var commands []string
		for _, reply := range []string{"SERVER PROTOCOL 2\n", "OK\n", "OK\n"} {
			line, err := r.ReadString('\n')
			if err != nil {
				return
			}
			commands = append(commands, line)
			io.WriteString(c, reply)
		}
		line, _ := r.ReadString('\n')
		commands = append(commands, line)
		frames <- commands
		c.Write(b)
	})
	s, err := (Client{Address: addr, Timeout: time.Second}).Status(context.Background())
	if err != nil || !bytes.Equal(s.Raw, b) {
		t.Fatalf("status: %+v %v", s, err)
	}
	if got := strings.Join(<-frames, ""); got != "CLIENT PROTOCOL 2\nTIMEOUT 1\nSET FORMAT json\nSTATUS\n" {
		t.Fatalf("wire commands %q", got)
	}
}

func TestClientRefusals(t *testing.T) {
	for _, reply := range []string{"SERVER PROTOCOL 1\n", "SERVER PROTOCOL 3\n", "SERVER PROTOCOL 2 extra\n", "ERROR\n", strings.Repeat("x", 5000) + "\n", "SERVER PROTOCOL 2"} {
		t.Run(fmt.Sprintf("reply_%d_%d", len(reply), reply[0]), func(t *testing.T) {
			addr := serve(t, func(c net.Conn) { bufio.NewReader(c).ReadString('\n'); io.WriteString(c, reply) })
			if _, err := (Client{Address: addr, Timeout: time.Second}).Status(context.Background()); err == nil {
				t.Fatal("accepted unsupported or incomplete handshake")
			}
		})
	}
	for stage := 1; stage <= 2; stage++ {
		t.Run(fmt.Sprintf("ack_%d", stage), func(t *testing.T) {
			addr := serve(t, func(c net.Conn) {
				r := bufio.NewReader(c)
				r.ReadString('\n')
				io.WriteString(c, "SERVER PROTOCOL 2\n")
				for i := 1; i <= stage; i++ {
					r.ReadString('\n')
					if i == stage {
						io.WriteString(c, "ERROR: unsupported\n")
					} else {
						io.WriteString(c, "OK\n")
					}
				}
			})
			if _, err := (Client{Address: addr, Timeout: time.Second}).Status(context.Background()); err == nil {
				t.Fatal("accepted failed protocol command")
			}
		})
	}
	for _, c := range []Client{{}, {Address: "invalid", Timeout: time.Second}, {Address: "127.0.0.1:1", Timeout: 0}, {Address: "127.0.0.1:1", Timeout: time.Minute + 1}, {Address: "127.0.0.1:1", Timeout: time.Second, Limits: Limits{Bytes: -1}}} {
		if _, err := c.Status(context.Background()); err == nil {
			t.Fatal("invalid client")
		}
	}
	if _, err := (Client{Address: "127.0.0.1:1", Timeout: time.Second}).Status(nil); err == nil {
		t.Fatal("nil context")
	}
}

func TestClientCancellationAndTimeout(t *testing.T) {
	for _, cancelEarly := range []bool{false, true} {
		t.Run(fmt.Sprint(cancelEarly), func(t *testing.T) {
			connected := make(chan struct{})
			addr := serve(t, func(c net.Conn) { close(connected); io.Copy(io.Discard, c) })
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			done := make(chan error, 1)
			go func() { _, err := (Client{Address: addr, Timeout: 50 * time.Millisecond}).Status(ctx); done <- err }()
			select {
			case <-connected:
			case <-time.After(time.Second):
				t.Fatal("loopback connection did not arrive")
			}
			if cancelEarly {
				cancel()
			}
			select {
			case err := <-done:
				if err == nil {
					t.Fatal("timeout/cancellation accepted")
				}
				if cancelEarly && !errors.Is(err, context.Canceled) {
					t.Fatalf("lost cancellation: %v", err)
				}
			case <-time.After(time.Second):
				t.Fatal("client failed to stop")
			}
		})
	}
}
