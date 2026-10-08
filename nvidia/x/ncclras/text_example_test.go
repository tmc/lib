package ncclras_test

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/ncclras"
)

func ExampleUnsupportedError() {
	_, err := ncclras.ReadText(strings.NewReader("unknown format\n"), ncclras.Limits{})
	var unsupported *ncclras.UnsupportedError
	fmt.Println(errors.As(err, &unsupported), unsupported.Version == "")
	// Output: true true
}

func ExampleReadText() {
	// This fixture is constructed from the pinned producer, not a live capture.
	b, err := os.ReadFile("testdata/legacy-2.24.3.txt")
	if err != nil {
		panic(err)
	}
	s, err := ncclras.ReadText(strings.NewReader(string(b)+"\n"), ncclras.Limits{})
	if err != nil {
		panic(err)
	}
	fmt.Println(s.Version, s.Processes, s.GPUs, len(s.Groups))
	// Output: 2.24.3 2 2 1
}

func ExampleClient_StatusText() {
	// A loopback protocol double supplies the explicitly constructed fixture.
	b, err := os.ReadFile("testdata/legacy-2.24.3.txt")
	if err != nil {
		panic(err)
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		panic(err)
	}
	defer listener.Close()
	done := make(chan error, 1)
	go func() {
		conn, err := listener.Accept()
		if err != nil {
			done <- err
			return
		}
		defer conn.Close()
		conn.SetDeadline(time.Now().Add(time.Second))
		r := bufio.NewReader(conn)
		for _, frame := range []struct{ command, reply string }{
			{"CLIENT PROTOCOL 2\n", "SERVER PROTOCOL 2\n"},
			{"TIMEOUT 1\n", "OK\n"},
			{"STATUS\n", string(b) + "\n"},
		} {
			command, err := r.ReadString('\n')
			if err != nil || command != frame.command {
				done <- fmt.Errorf("unexpected command %q: %v", command, err)
				return
			}
			if _, err := io.WriteString(conn, frame.reply); err != nil {
				done <- err
				return
			}
		}
		done <- nil
	}()
	client := ncclras.Client{Address: listener.Addr().String(), Timeout: time.Second}
	s, err := client.StatusText(context.Background())
	if err != nil {
		panic(err)
	}
	if err := <-done; err != nil {
		panic(err)
	}
	fmt.Println(s.Version, s.GPUs)
	// Output: 2.24.3 2
}
