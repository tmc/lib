package nvtelemetry

import (
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

// Clock provides monotonic and wall-clock times.
type Clock interface {
	NowMonotonic() int64
	NowUnix() int64
}

// ClockMetadata provides clock domain, epoch, and synchronization metadata.
type ClockMetadata interface {
	MonotonicDomain() string
	MonotonicEpochID() string
	WallDomain() string
	Provenance() string
}

// CalibratedClock provides clock calibration bounds and mapping validity.
type CalibratedClock interface {
	Clock
	ClockMetadata
	Calibration() (validFrom, validUntil, bounds int64, ok bool)
}

// processStart defines the process-local epoch. In Go, time.Now() includes a monotonic clock
// reading; subsequent calls to time.Since(processStart) read runtime monotonic time and are
// immune to NTP wall-clock adjustments.
var (
	processStart = time.Now()
	processEpoch = fmt.Sprintf("proc-%d-%d", os.Getpid(), processStart.UnixNano())
)

// ProcessEpochID returns the unique process-local monotonic epoch ID.
func ProcessEpochID() string {
	return processEpoch
}

// RealClock uses host system time with decoupled monotonic and wall-clock sources.
type RealClock struct{}

// NowMonotonic returns monotonic time in nanoseconds elapsed since process start.
// It uses Go runtime monotonic clocking and never steps with NTP adjustments.
func (RealClock) NowMonotonic() int64 {
	return time.Since(processStart).Nanoseconds()
}

// NowUnix returns host Unix time in nanoseconds (wall clock).
func (RealClock) NowUnix() int64 {
	return time.Now().UnixNano()
}

// MonotonicDomain returns the monotonic domain of RealClock.
func (RealClock) MonotonicDomain() string {
	return "process_local_go_monotonic"
}

// MonotonicEpochID returns the unique process-local epoch ID.
func (RealClock) MonotonicEpochID() string {
	return ProcessEpochID()
}

// WallDomain returns the wall clock domain.
func (RealClock) WallDomain() string {
	return "unix_utc"
}

// Provenance returns the synchronization authority.
func (RealClock) Provenance() string {
	return "system_kernel"
}

var fakeEpochSeq uint64

// FakeClock provides deterministic time control for reproducible testing.
type FakeClock struct {
	mu        sync.Mutex
	monotonic int64
	unix      int64
	epochID   string
}

// NewFakeClock creates a FakeClock with initialized timestamps and a distinct epoch ID.
func NewFakeClock(startMono, startUnix int64) *FakeClock {
	seq := atomic.AddUint64(&fakeEpochSeq, 1)
	return &FakeClock{
		monotonic: startMono,
		unix:      startUnix,
		epochID:   fmt.Sprintf("fake-%s-%d", ProcessEpochID(), seq),
	}
}

// SetEpochID configures the monotonic epoch ID explicitly (e.g. for coordinated test fixtures).
func (f *FakeClock) SetEpochID(id string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.epochID = id
}

// NowMonotonic returns the current fake monotonic time.
func (f *FakeClock) NowMonotonic() int64 {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.monotonic
}

// NowUnix returns the current fake Unix time.
func (f *FakeClock) NowUnix() int64 {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.unix
}

// Advance steps both monotonic and wall clocks forward by d.
func (f *FakeClock) Advance(d time.Duration) {
	f.mu.Lock()
	defer f.mu.Unlock()
	nanos := d.Nanoseconds()
	f.monotonic += nanos
	f.unix += nanos
}

// StepWall simulates a wall-clock jump (e.g. NTP adjustment) without changing monotonic time.
func (f *FakeClock) StepWall(d time.Duration) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.unix += d.Nanoseconds()
}

// MonotonicDomain returns the synthetic domain of FakeClock.
func (f *FakeClock) MonotonicDomain() string {
	return "synthetic_fake_clock"
}

// MonotonicEpochID returns the synthetic epoch ID of FakeClock.
// If unconfigured or zero-valued, it returns an empty string (unqualified identity).
func (f *FakeClock) MonotonicEpochID() string {
	if f == nil {
		return ""
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.epochID
}

// WallDomain returns the synthetic wall domain.
func (f *FakeClock) WallDomain() string {
	return "synthetic_unix_utc"
}

// Provenance returns the synthetic provenance.
func (f *FakeClock) Provenance() string {
	return "injected_fake"
}
