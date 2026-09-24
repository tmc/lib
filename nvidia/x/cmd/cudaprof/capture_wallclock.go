package main

import "time"

// A captureWallClock is a reading of CLOCK_MONOTONIC and of the wall clock
// taken together. The agent stamps every record it originates, the
// capture-range edges among them, on CLOCK_MONOTONIC, which has no relation
// to wall time; the pair is what joins those timestamps to a log kept by
// something outside the capture, such as a watcher recording which other
// processes loaded the host.
//
// MonotonicNanos is the midpoint of two monotonic reads that bracket the wall
// read, and ReadNanos is the time between them, so the pairing is uncertain
// by at most half of ReadNanos. The mapping also assumes the wall clock is
// not stepped while the capture runs; NTP slewing moves it by at most 500
// ppm.
type captureWallClock struct {
	MonotonicNanos uint64 `json:"monotonic_nanos"`
	UnixNanos      int64  `json:"unix_nanos"`
	ReadNanos      uint64 `json:"read_nanos"`
}

// readWallClock returns the current pairing, or nil where there is no
// CLOCK_MONOTONIC to pair.
func readWallClock() *captureWallClock {
	m0, ok := monotonicNanos()
	if !ok {
		return nil
	}
	wall := time.Now().UnixNano()
	m1, ok := monotonicNanos()
	if !ok {
		return nil
	}
	return &captureWallClock{MonotonicNanos: m0 + (m1-m0)/2, UnixNanos: wall, ReadNanos: m1 - m0}
}

// at returns the wall time of the monotonic timestamp t.
func (c *captureWallClock) at(t uint64) time.Time {
	// The unsigned difference wraps to the right signed one for a t before
	// the reading.
	return time.Unix(0, c.UnixNanos+int64(t-c.MonotonicNanos))
}
