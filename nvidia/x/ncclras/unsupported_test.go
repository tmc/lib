package ncclras

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"io"
	"net"
	"strings"
	"testing"
	"time"
)

func TestUnsupportedJSONVersion(t *testing.T) {
	b := bytes.Replace(statusFixture(t), []byte(`"2.29.1"`), []byte(`"2.30.1"`), 1)
	_, err := Read(bytes.NewReader(b), Limits{})
	var unsupported *UnsupportedError
	if !errors.As(err, &unsupported) || unsupported.Format != "json" || unsupported.Version != "2.30.1" {
		t.Fatalf("unsupported version: %v", err)
	}
}

func TestUnsupportedJSONTextFormat(t *testing.T) {
	for _, tc := range []struct{ input, version string }{
		{string(legacyFixture(t)), "2.24.3"},
		{"NCCL version 2.25.1+cuda12.4 compiled with CUDA 12.4\n", "2.25.1+cuda12.4"},
	} {
		_, err := Read(strings.NewReader(tc.input), Limits{})
		var unsupported *UnsupportedError
		if !errors.As(err, &unsupported) || unsupported.Format != "json" || unsupported.Version != tc.version {
			t.Fatalf("unsupported text: %v", err)
		}
	}
}

func TestUnsupportedClientFormat(t *testing.T) {
	addr := serve(t, func(c net.Conn) {
		r := bufio.NewReader(c)
		for _, reply := range []string{"SERVER PROTOCOL 2\n", "OK\n", "ERROR: unknown command\n"} {
			if _, err := r.ReadString('\n'); err != nil {
				return
			}
			io.WriteString(c, reply)
		}
	})
	_, err := (Client{Address: addr, Timeout: time.Second}).Status(context.Background())
	var unsupported *UnsupportedError
	if !errors.As(err, &unsupported) || unsupported.Format != "json" || unsupported.Version != "" {
		t.Fatalf("invented version or untyped refusal: %v", err)
	}
}
