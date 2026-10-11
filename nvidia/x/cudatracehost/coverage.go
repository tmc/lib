package cudatracehost

import (
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"time"

	"github.com/tmc/lib/ebpfhost"
)

// ThreadSnapshot identifies a discovered task by its Linux starttime ticks.
// Generation is the /proc stat token, not the kernel BPF event's nanosecond
// birth field. The two values must not be compared directly.
type ThreadSnapshot struct {
	TID        uint32
	Generation uint64
}

// SnapshotCoverage records the BPF attachment snapshot. Offset fields are
// elapsed monotonic nanoseconds from discovery start, not event timestamps.
// Successful verification establishes matching task generations before and
// after attachment, not continuous coverage or an atomic process snapshot.
// Future threads and calls during discovery/attachment remain unobserved.
// Tracefs coverage is unavailable here: that backend enumerates separately.
// The zero value (including legacy manifests omitting this field) means unknown.
type SnapshotCoverage struct {
	Status                                                                            string
	Clock                                                                             string
	DiscoveryEndNanos, AttachmentStartNanos, AttachmentEndNanos, VerificationEndNanos int64
	Threads                                                                           []ThreadSnapshot
	FutureThreads                                                                     string
	BlindInterval                                                                     string
}

func discoveredThreads(ids []int, generation func(int) (uint64, error), limit int) ([]ThreadSnapshot, error) {
	if limit < 1 || len(ids) == 0 || len(ids) > limit {
		return nil, fmt.Errorf("cudatracehost: thread snapshot exceeds attachment bound or is empty")
	}
	ids = append([]int(nil), ids...)
	sort.Ints(ids)
	out := make([]ThreadSnapshot, 0, len(ids))
	for i, id := range ids {
		if id <= 0 || uint64(id) > uint64(^uint32(0)) || i > 0 && id == ids[i-1] {
			return nil, fmt.Errorf("cudatracehost: invalid or duplicate snapshot thread")
		}
		g, err := generation(id)
		if err != nil {
			return nil, fmt.Errorf("cudatracehost: thread %d discovery: %w", id, err)
		}
		if g == 0 {
			return nil, fmt.Errorf("cudatracehost: thread %d generation unavailable", id)
		}
		out = append(out, ThreadSnapshot{uint32(id), g})
	}
	return out, nil
}

func verifyThreads(threads []ThreadSnapshot, generation func(int) (uint64, error)) error {
	for _, t := range threads {
		g, err := generation(int(t.TID))
		if err != nil {
			return fmt.Errorf("cudatracehost: thread %d verification: %w", t.TID, err)
		}
		if g != t.Generation {
			return fmt.Errorf("cudatracehost: thread %d lifetime changed during attachment", t.TID)
		}
	}
	return nil
}

func taskGeneration(pid, tid int) (uint64, error) {
	path := fmt.Sprintf("/proc/%d/task/%d", pid, tid)
	if _, err := os.Stat(path); err != nil {
		return 0, err
	}
	g, err := ebpfhost.ProcessStartTime(tid)
	if err != nil {
		return 0, err
	}
	if _, err := os.Stat(path); err != nil {
		return 0, err
	}
	return g, nil
}

func discoverCoverage(pid, probeCount int) (SnapshotCoverage, time.Time, error) {
	start := time.Now()
	c := SnapshotCoverage{Status: "unverified", Clock: "elapsed_since_discovery", FutureThreads: "unobserved", BlindInterval: "discovery_through_attachment"}
	if probeCount < 1 || probeCount > 256 {
		return c, start, fmt.Errorf("cudatracehost: invalid probe count for snapshot")
	}
	dir, err := os.Open(fmt.Sprintf("/proc/%d/task", pid))
	if err != nil {
		return c, start, err
	}
	defer dir.Close()
	entries, err := dir.ReadDir(256/probeCount + 1)
	if err != nil && err != io.EOF {
		return c, start, err
	}
	if len(entries) > 256/probeCount {
		return c, start, fmt.Errorf("cudatracehost: attachment snapshot exceeds 256 probes")
	}
	ids := make([]int, len(entries))
	for i, e := range entries {
		ids[i], err = strconv.Atoi(e.Name())
		if err != nil {
			return c, start, err
		}
	}
	c.Threads, err = discoveredThreads(ids, func(tid int) (uint64, error) { return taskGeneration(pid, tid) }, 256/probeCount)
	c.DiscoveryEndNanos = time.Since(start).Nanoseconds()
	return c, start, err
}
