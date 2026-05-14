package nvlshm

import (
	"fmt"
	"os"
	"testing"

	"golang.org/x/sys/unix"
)

// runRanks runs fn as world concurrent ranks and fails the test on any error.
func runRanks(t *testing.T, world int, fn func(rank int, b Bootstrap) error) {
	t.Helper()
	boots := newLocalBoots(world)
	errs := make(chan error, world)
	for r := 0; r < world; r++ {
		go func(r int) {
			errs <- fn(r, boots[r])
		}(r)
	}
	for i := 0; i < world; i++ {
		if err := <-errs; err != nil {
			t.Fatal(err)
		}
	}
}

// exportFile creates an unlinked temp file containing msg and returns its fd.
// A file (not a pipe) because SCM_RIGHTS shares the file description: every
// receiver holds a dup of the SAME description, so Pread at offset 0 is the
// only read that all receivers can perform independently.
func exportFile(t *testing.T, msg string) int {
	t.Helper()
	f, err := os.CreateTemp(t.TempDir(), "fdxchg")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteString(msg); err != nil {
		t.Fatal(err)
	}
	fd, err := unix.Dup(int(f.Fd()))
	if err != nil {
		t.Fatal(err)
	}
	f.Close()
	return fd
}

// readFD reads fd's contents at offset 0 and closes it (receiver-owned).
func readFD(fd int) (string, error) {
	defer unix.Close(fd)
	buf := make([]byte, 64)
	n, err := unix.Pread(fd, buf, 0)
	if err != nil {
		return "", err
	}
	return string(buf[:n]), nil
}

func TestExchangeFDsAllToAll(t *testing.T) {
	const world = 4
	runRanks(t, world, func(rank int, b Bootstrap) error {
		// Each rank exports a file fd whose contents name the sender.
		localFD := exportFile(t, fmt.Sprintf("from-%d", rank))

		senders := make([]int, world)
		for i := range senders {
			senders[i] = i
		}
		fds, err := exchangeFDs(b, localFD, senders)
		unix.Close(localFD)
		if err != nil {
			return err
		}
		if len(fds) != world {
			return fmt.Errorf("rank %d: got %d fds, want %d", rank, len(fds), world)
		}
		for sender, fd := range fds {
			got, err := readFD(fd)
			if err != nil {
				return fmt.Errorf("rank %d read fd from %d: %w", rank, sender, err)
			}
			want := fmt.Sprintf("from-%d", sender)
			if got != want {
				return fmt.Errorf("rank %d: fd from %d reads %q, want %q", rank, sender, got, want)
			}
		}
		return nil
	})
}

func TestExchangeFDsSingleSender(t *testing.T) {
	const world, owner = 3, 1
	runRanks(t, world, func(rank int, b Bootstrap) error {
		localFD := -1
		if rank == owner {
			localFD = exportFile(t, "owner-data")
		}
		fds, err := exchangeFDs(b, localFD, []int{owner})
		if localFD >= 0 {
			unix.Close(localFD)
		}
		if err != nil {
			return err
		}
		if len(fds) != 1 {
			return fmt.Errorf("rank %d: got %d fds, want 1", rank, len(fds))
		}
		got, err := readFD(fds[owner])
		if err != nil {
			return err
		}
		if got != "owner-data" {
			return fmt.Errorf("rank %d: read %q", rank, got)
		}
		return nil
	})
}

func TestExchangeFDsValidation(t *testing.T) {
	boots := newLocalBoots(1)
	if _, err := exchangeFDs(boots[0], -1, []int{0}); err == nil {
		t.Error("sender rank without localFD: want error")
	}
	if _, err := exchangeFDs(boots[0], -1, []int{5}); err == nil {
		t.Error("out-of-range sender: want error")
	}
}
