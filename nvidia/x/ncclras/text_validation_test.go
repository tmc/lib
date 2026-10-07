package ncclras

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"io"
	"net"
	"os"
	"strings"
	"testing"
	"time"
)

func legacyFixture(t testing.TB) []byte {
	t.Helper()
	b, err := os.ReadFile("testdata/legacy-2.24.3.txt")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.HasSuffix(b, []byte("Warnings\n========\n")) || bytes.HasSuffix(b, []byte("\n\n")) {
		t.Fatal("fixture storage must end with one newline after warning separator")
	}
	// The wire response ends with an empty line, omitted from text storage.
	return append(b, '\n')
}
func TestLegacyExactAggregateBudget(t *testing.T) {
	b := legacyFixture(t)
	if _, err := ReadText(bytes.NewReader(b), Limits{Records: 1}); err != nil {
		t.Fatalf("one group/no diagnostics %v", err)
	}
	raw := strings.Replace(string(b), "Warnings\n========\n\n", "  detail error\nWarnings\n========\n\n  detail warning\n", 1)
	if _, err := ReadText(strings.NewReader(raw), Limits{Records: 3}); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadText(strings.NewReader(raw), Limits{Records: 2}); err == nil {
		t.Fatal("aggregate group+diagnostic limit omitted")
	}
}
func TestLegacyWireAndCancellation(t *testing.T) {
	frames := make(chan string, 1)
	b := legacyFixture(t)
	addr := serve(t, func(c net.Conn) {
		r := bufio.NewReader(c)
		var got string
		for _, reply := range []string{"SERVER PROTOCOL 2\n", "OK\n"} {
			s, err := r.ReadString('\n')
			if err != nil {
				return
			}
			got += s
			io.WriteString(c, reply)
		}
		s, err := r.ReadString('\n')
		if err != nil {
			return
		}
		got += s
		frames <- got
		c.Write(b)
	})
	s, err := (Client{Address: addr, Timeout: 1500 * time.Millisecond}).StatusText(context.Background())
	if err != nil || !bytes.Equal(s.Raw, b) {
		t.Fatalf("%v", err)
	}
	select {
	case got := <-frames:
		if got != "CLIENT PROTOCOL 2\nTIMEOUT 2\nSTATUS\n" {
			t.Fatalf("wire %q", got)
		}
	case <-time.After(time.Second):
		t.Fatal("wire missing")
	}
	ready := make(chan struct{})
	closed := make(chan error, 1)
	addr = serve(t, func(c net.Conn) {
		r := bufio.NewReader(c)
		for _, reply := range []string{"SERVER PROTOCOL 2\n", "OK\n"} {
			r.ReadString('\n')
			io.WriteString(c, reply)
		}
		r.ReadString('\n')
		close(ready)
		_, err := r.ReadByte()
		closed <- err
	})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { _, err := (Client{Address: addr, Timeout: time.Second}).StatusText(ctx); done <- err }()
	select {
	case <-ready:
	case <-time.After(time.Second):
		t.Fatal("server not ready")
	}
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("cancel not bounded")
	}
	select {
	case err := <-closed:
		if err == nil {
			t.Fatal("connection still live")
		}
	case <-time.After(time.Second):
		t.Fatal("connection not closed")
	}
}
func TestLegacyVersionAndShape(t *testing.T) {
	b := legacyFixture(t)
	for _, version := range []string{"2.25.1", "2.32.3"} {
		_, err := ReadText(strings.NewReader(strings.Replace(string(b), "2.24.3", version, 1)), Limits{})
		var u *UnsupportedError
		if !errors.As(err, &u) || u.Version != version {
			t.Fatalf("typed version %v", err)
		}
	}
	if _, err := ReadText(bytes.NewReader(b[:len(b)-1]), Limits{}); err == nil {
		t.Fatal("partial accepted")
	}
	if _, err := ReadText(bytes.NewReader(b), Limits{Bytes: len(b)}); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadText(bytes.NewReader(b), Limits{Bytes: len(b) - 1}); err == nil {
		t.Fatal("byte cap omitted")
	}
}
func FuzzReadText(f *testing.F) {
	b := legacyFixture(f)
	f.Add(string(b))
	f.Fuzz(func(t *testing.T, s string) {
		if len(s) > 4096 {
			return
		}
		ReadText(strings.NewReader(s), Limits{Bytes: 4096, Line: 1024, Records: 64})
	})
}

func TestLegacyFixtureWireEnding(t *testing.T) {
	b := legacyFixture(t)
	if !bytes.HasSuffix(b, []byte("Warnings\n========\n\n")) || bytes.HasSuffix(b, []byte("\n\n\n")) {
		t.Fatal("fixture wire ending changed")
	}
}
