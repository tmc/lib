package tcpboot

import (
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
	"time"
)

// The failure-isolation tests need ranks in separate processes: a goroutine
// rank cannot be killed, and a rank that stalls inside the test binary would
// stall the test itself. TestMain re-runs this binary as a worker rank when
// the environment names one.

const (
	envRank  = "TCPBOOT_TEST_RANK"
	envWorld = "TCPBOOT_TEST_WORLD"
	envAddr  = "TCPBOOT_TEST_ADDR"
	envMode  = "TCPBOOT_TEST_MODE"
)

const workerTimeout = 5 * time.Second

func TestMain(m *testing.M) {
	if os.Getenv(envRank) == "" {
		os.Exit(m.Run())
	}
	if err := runWorker(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	os.Exit(0)
}

// runWorker joins the world as one rank and behaves per envMode: "ok"
// completes the collectives, "exit" leaves after the first barrier, and
// "stall" connects and then never contributes a round.
func runWorker() error {
	rank, err := strconv.Atoi(os.Getenv(envRank))
	if err != nil {
		return err
	}
	world, err := strconv.Atoi(os.Getenv(envWorld))
	if err != nil {
		return err
	}
	b, err := New(rank, world, os.Getenv(envAddr))
	if err != nil {
		return err
	}
	b.RoundTimeout = workerTimeout
	defer b.Close()

	switch os.Getenv(envMode) {
	case "stall":
		time.Sleep(time.Minute)
		return nil
	case "exit":
		if err := b.Barrier(); err != nil {
			return err
		}
		os.Exit(0)
	}
	got, err := b.BroadcastString("", 0)
	if err != nil {
		return err
	}
	if got != "from rank 0" {
		return fmt.Errorf("rank %d: broadcast = %q", rank, got)
	}
	return b.Barrier()
}

// startWorker launches rank in its own process.
func startWorker(t *testing.T, rank, world int, addr, mode string) *exec.Cmd {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(exe, "-test.run=TestNothingRunsInWorker")
	cmd.Env = append(os.Environ(),
		envRank+"="+strconv.Itoa(rank),
		envWorld+"="+strconv.Itoa(world),
		envAddr+"="+addr,
		envMode+"="+mode,
	)
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if cmd.ProcessState == nil {
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
		}
	})
	return cmd
}

func TestMultiProcessCollectives(t *testing.T) {
	const world = 3
	addr := freeAddr(t)
	var workers []*exec.Cmd
	for rank := 1; rank < world; rank++ {
		workers = append(workers, startWorker(t, rank, world, addr, "ok"))
	}
	b, err := New(0, world, addr)
	if err != nil {
		t.Fatal(err)
	}
	b.RoundTimeout = workerTimeout
	defer b.Close()

	got, err := b.BroadcastString("from rank 0", 0)
	if err != nil {
		t.Fatal(err)
	}
	if got != "from rank 0" {
		t.Fatalf("broadcast = %q", got)
	}
	if err := b.Barrier(); err != nil {
		t.Fatal(err)
	}
	for _, cmd := range workers {
		if err := cmd.Wait(); err != nil {
			t.Fatalf("worker: %v", err)
		}
	}
}

// TestStalledRankDoesNotHangPeers is the isolation case that motivates
// RoundTimeout: a rank that connects and then contributes nothing must not
// hold its peers past the timeout.
func TestStalledRankDoesNotHangPeers(t *testing.T) {
	const world = 3
	addr := freeAddr(t)
	startWorker(t, 1, world, addr, "ok")
	startWorker(t, 2, world, addr, "stall")

	b, err := New(0, world, addr)
	if err != nil {
		t.Fatal(err)
	}
	b.RoundTimeout = time.Second
	defer b.Close()

	start := time.Now()
	_, err = b.BroadcastString("from rank 0", 0)
	elapsed := time.Since(start)
	if err == nil {
		t.Fatal("broadcast with a stalled rank: want error")
	}
	if !strings.Contains(err.Error(), "receive round") {
		t.Errorf("err = %v, want a receive-round failure", err)
	}
	if elapsed > 5*time.Second {
		t.Errorf("broadcast returned after %v, want about the 1s timeout", elapsed)
	}
	// Lockstep is broken; later collectives must fail fast rather than read
	// the reply they missed.
	start = time.Now()
	if got := b.Barrier(); got != err {
		t.Errorf("barrier after a failed round = %v, want original error %v", got, err)
	}
	if elapsed := time.Since(start); elapsed > 100*time.Millisecond {
		t.Errorf("failed barrier took %v, want immediate failure", elapsed)
	}
	t.Logf("stalled rank released peer after %v with %v", elapsed, err)
}

// TestDepartedRankFailsPeers checks that a rank leaving mid-job surfaces as an
// error on the survivors rather than a hang.
func TestDepartedRankFailsPeers(t *testing.T) {
	const world = 2
	addr := freeAddr(t)
	startWorker(t, 1, world, addr, "exit")

	b, err := New(0, world, addr)
	if err != nil {
		t.Fatal(err)
	}
	b.RoundTimeout = workerTimeout
	defer b.Close()

	if err := b.Barrier(); err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	err = b.Barrier()
	if err == nil {
		t.Fatal("barrier after a rank left: want error")
	}
	t.Logf("departed rank surfaced after %v with %v", time.Since(start), err)
}

// TestNothingRunsInWorker exists so the worker process can be started with a
// -test.run pattern that matches no real test.
func TestNothingRunsInWorker(t *testing.T) {}
