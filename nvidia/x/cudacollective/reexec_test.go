//go:build !windows

package cudacollective_test

import (
	"fmt"
	"net"
	"os"
	"os/exec"
	"strconv"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/nvlshm/tcpboot"
)

const (
	envWorkerRank  = "CUDACOLLECTIVE_TEST_WORKER_RANK"
	envWorkerWorld = "CUDACOLLECTIVE_TEST_WORKER_WORLD"
	envWorkerAddr  = "CUDACOLLECTIVE_TEST_WORKER_ADDR"
	envWorkerMode  = "CUDACOLLECTIVE_TEST_WORKER_MODE"
)

func TestMain(m *testing.M) {
	if os.Getenv(envWorkerRank) != "" {
		if err := runReExecWorker(); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		os.Exit(0)
	}
	os.Exit(m.Run())
}

func runReExecWorker() error {
	rank, err := strconv.Atoi(os.Getenv(envWorkerRank))
	if err != nil {
		return err
	}
	world, err := strconv.Atoi(os.Getenv(envWorkerWorld))
	if err != nil {
		return err
	}
	addr := os.Getenv(envWorkerAddr)
	mode := os.Getenv(envWorkerMode)

	b, err := tcpboot.New(rank, world, addr)
	if err != nil {
		return err
	}
	defer b.Close()
	b.RoundTimeout = 5 * time.Second

	switch mode {
	case "stall":
		time.Sleep(30 * time.Second)
		return nil
	case "exit":
		_ = b.Barrier()
		os.Exit(0)
	case "ok":
		// Normal participation
		if _, err := b.BroadcastString(fmt.Sprintf("rank-%d", rank), 0); err != nil {
			return err
		}
		return b.Barrier()
	}
	return nil
}

// TestMultiProcessCollectiveHarness exercises multi-process execution
// across parent and child processes using the re-exec pattern.
func TestMultiProcessCollectiveHarness(t *testing.T) {
	const world = 3
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := ln.Addr().String()
	ln.Close()

	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}

	var workers []*exec.Cmd
	for r := 1; r < world; r++ {
		cmd := exec.Command(exe, "-test.run=TestNothingRunsInWorker")
		cmd.Env = append(os.Environ(),
			envWorkerRank+"="+strconv.Itoa(r),
			envWorkerWorld+"="+strconv.Itoa(world),
			envWorkerAddr+"="+addr,
			envWorkerMode+"=ok",
		)
		cmd.Stderr = os.Stderr
		if err := cmd.Start(); err != nil {
			t.Fatal(err)
		}
		workers = append(workers, cmd)
	}
	t.Cleanup(func() {
		for _, w := range workers {
			if w.ProcessState == nil {
				_ = w.Process.Kill()
				_ = w.Wait()
			}
		}
	})

	// Rank 0 acts as hub and participant
	b, err := tcpboot.New(0, world, addr)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	b.RoundTimeout = 5 * time.Second

	// Exercise broadcast
	res, err := b.BroadcastString("rank-0", 0)
	if err != nil {
		t.Fatalf("broadcast failed: %v", err)
	}
	if res != "rank-0" {
		t.Fatalf("broadcast payload mismatch: got %q, want rank-0", res)
	}

	// Exercise barrier
	if err := b.Barrier(); err != nil {
		t.Fatalf("barrier failed: %v", err)
	}

	for _, w := range workers {
		if err := w.Wait(); err != nil {
			t.Fatalf("worker failed: %v", err)
		}
	}
}

// TestNothingRunsInWorker is an empty target for worker sub-processes.
func TestNothingRunsInWorker(t *testing.T) {}
