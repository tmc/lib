package cudahotswap

import (
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"path/filepath"
	"sort"
	"time"

	"github.com/tmc/lib/nvidia/x/cudahotswap/internal/fswatch"
	"github.com/tmc/lib/nvidia/x/cudalock"
)

// maxOwnerBytes bounds one read of the lock owner file. A gpu-lock owner
// line is a few hundred bytes.
const maxOwnerBytes = 4 << 10

// LockWait describes one service request blocked in GPU lock acquisition.
//
// Waited is host monotonic time since the request entered acquisition,
// measured when Waiting took its snapshot. It is neither the holder's age
// nor GPU time.
//
// Holder is an advisory, unverified reading of the lock directory's owner
// file. It is not a lease and establishes neither liveness nor custody; the
// holder may have released, or not yet published, by the time it is read.
// When HolderUnavailable is nonempty it says why no owner was read, and
// Holder is the zero Owner.
type LockWait struct {
	Request           uint64 // service-local serial, in acquisition order
	Name              string // kernel name, or a request kind such as "release" or "close"
	Label             string // lock label the service requests for Name
	Waited            time.Duration
	Holder            cudalock.Owner
	HolderUnavailable string
}

// acquireWaits records requests inside acquisition. Service.mu guards it.
type acquireWaits struct {
	serial  uint64
	pending map[uint64]acquireWait
	now     func() time.Time
	observe func() (cudalock.Owner, string)
}

type acquireWait struct {
	name    string
	entered time.Time
}

func newAcquireWaits() acquireWaits {
	return acquireWaits{
		pending: make(map[uint64]acquireWait),
		now:     time.Now,
		observe: func() (cudalock.Owner, string) { return observeOwner(cudalock.DefaultDir()) },
	}
}

func lockLabel(name string) string { return "cudahotswap:" + name }

// acquireWaiting publishes the request for Waiting until rawAcquire returns,
// whatever its outcome, and returns rawAcquire's result unchanged.
func (s *Service) acquireWaiting(ctx context.Context, name string) (io.Closer, error) {
	id := s.enterWait(name)
	defer s.leaveWait(id)
	return s.rawAcquire(ctx, name)
}

func (s *Service) enterWait(name string) uint64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.waits.serial++
	s.waits.pending[s.waits.serial] = acquireWait{name: name, entered: s.waits.now()}
	return s.waits.serial
}

func (s *Service) leaveWait(id uint64) {
	s.mu.Lock()
	delete(s.waits.pending, id)
	s.mu.Unlock()
}

// Waiting returns the requests currently blocked in GPU lock acquisition, in
// acquisition order, or nil if there are none. The result belongs to the
// caller.
//
// Waiting copies the pending requests under the service mutex and then, with
// no service lock held, reads the canonical cudalock owner file once. The read
// is bounded in size and refuses missing, symlinked, nonregular, oversized and
// malformed files, and every file on platforms without a nonblocking no-follow
// open, but it is an ordinary filesystem read with no deadline. Nothing is
// polled or watched between calls.
//
// The result is a snapshot: a listed request may have acquired, failed or been
// canceled, and the holder may have changed, before Waiting returns. Pending
// requests are not registered kernels and do not appear in List or Status.
func (s *Service) Waiting() []LockWait {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	if len(s.waits.pending) == 0 {
		s.mu.Unlock()
		return nil
	}
	now := s.waits.now()
	waits := make([]LockWait, 0, len(s.waits.pending))
	for id, w := range s.waits.pending {
		waits = append(waits, LockWait{Request: id, Name: w.name, Label: lockLabel(w.name), Waited: now.Sub(w.entered)})
	}
	observe := s.waits.observe
	s.mu.Unlock()

	sort.Slice(waits, func(i, j int) bool { return waits[i].Request < waits[j].Request })
	holder, unavailable := observe()
	for i := range waits {
		waits[i].Holder = holder
		waits[i].HolderUnavailable = unavailable
	}
	return waits
}

// observeOwner reads dir's owner file without verifying the process it names.
// It returns the zero Owner and a reason when no owner can be read.
func observeOwner(dir string) (cudalock.Owner, string) {
	data, err := fswatch.ReadFile(filepath.Join(dir, "owner"), maxOwnerBytes)
	if errors.Is(err, fs.ErrNotExist) {
		return cudalock.Owner{}, "owner file missing"
	}
	if err != nil {
		return cudalock.Owner{}, fmt.Sprintf("owner file unreadable: %v", err)
	}
	owner, err := cudalock.ParseOwner(data)
	if err != nil {
		return cudalock.Owner{}, fmt.Sprintf("owner file malformed: %v", err)
	}
	return owner, ""
}
