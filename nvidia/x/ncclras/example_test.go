package ncclras_test

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net"
	"os"
	"time"

	"github.com/tmc/lib/nvidia/x/ncclras"
)

func ExampleRead() {
	f, err := os.Open("testdata/status.json")
	if err != nil {
		panic(err)
	}
	defer f.Close()
	s, err := ncclras.Read(f, ncclras.Limits{})
	if err != nil {
		panic(err)
	}
	fmt.Println(s.Version, s.Communicators[0].MissingRanksCount)
	// Output: 2.29.1 1
}

func ExampleClient() {
	c := ncclras.Client{Address: "127.0.0.1:28028"}
	// Merely constructing a client opens no connection. Set a positive Timeout
	// before calling Status with the address explicitly chosen by the caller.
	fmt.Println(c.Address)
	// Output: 127.0.0.1:28028
}

func ExampleClient_Status() {
	// A loopback protocol fixture, not a running NCCL job.
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		panic(err)
	}
	defer listener.Close()
	body, err := os.ReadFile("testdata/status.json")
	if err != nil {
		panic(err)
	}
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
		for _, reply := range []string{"SERVER PROTOCOL 2\n", "OK\n", "OK\n"} {
			if _, err := r.ReadString('\n'); err != nil {
				done <- err
				return
			}
			if _, err := io.WriteString(conn, reply); err != nil {
				done <- err
				return
			}
		}
		if _, err := r.ReadString('\n'); err != nil {
			done <- err
			return
		}
		_, err = conn.Write(body)
		done <- err
	}()
	c := ncclras.Client{Address: listener.Addr().String(), Timeout: time.Second}
	s, err := c.Status(context.Background())
	if err != nil {
		panic(err)
	}
	if err := <-done; err != nil {
		panic(err)
	}
	fmt.Println(s.Version)
	// Output: 2.29.1
}

func ExampleLimits() {
	l := ncclras.Limits{Bytes: 1 << 20, Line: 16 << 10, Records: 1024}
	fmt.Println(l.Records)
	// Output: 1024
}

func ExampleStatus() {
	s := ncclras.Status{Version: "2.29.1"}
	fmt.Println(s.Version)
	// Output: 2.29.1
}

func ExampleCommunicator() {
	c := ncclras.Communicator{Size: 2, RanksCount: 1, MissingRanksCount: 1}
	fmt.Println(c.Size, c.MissingRanksCount)
	// Output: 2 1
}

func ExampleRank() {
	r := ncclras.Rank{Rank: 0, Counts: map[string]uint64{"AllReduce": 3}}
	fmt.Println(r.Rank, r.Counts["AllReduce"])
	// Output: 0 3
}

func ExampleRankStatus() {
	s := ncclras.RankStatus{AsyncError: 6}
	fmt.Println(s.AsyncError)
	// Output: 6
}

func ExampleMissingRank() {
	r := ncclras.MissingRank{Rank: 1, Status: ncclras.MissingStatus{Unresponsive: true}}
	fmt.Println(r.Rank, r.Status.Unresponsive)
	// Output: 1 true
}

func ExampleMissingStatus() {
	s := ncclras.MissingStatus{Unresponsive: true, ConsideredDead: false}
	fmt.Println(s.Unresponsive, s.ConsideredDead)
	// Output: true false
}

func ExampleCollection() {
	c := ncclras.Collection{Timeouts: 2}
	fmt.Println(c.Timeouts)
	// Output: 2
}
