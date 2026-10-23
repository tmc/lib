package nvml

import (
	"errors"
	"fmt"
	"math"
	"runtime"
)

// ComputeRunningProcessesInto stores information about processes with a
// compute context on d in buf and returns the number of entries stored.
// The caller owns buf; entries buf[:n] are valid after a nil error.
//
// It makes exactly one call to nvmlDeviceGetComputeRunningProcesses_v3 with
// capacity len(buf). Unlike [Device.ComputeRunningProcesses], it never
// retries, allocates no query buffer, and never sizes storage from a count
// NVML reports: buf is the only storage the call fills. This is not a promise
// of no heap allocation at all; a returned error, for one, is allocated.
// A zero length, or one that does not fit in a uint32, is refused before the
// call and before the NVML library is loaded. If the processes do not fit,
// it returns an error wrapping [ErrInsufficientSize] and stores
// nothing usable; it does not truncate the list. A returned count beyond
// len(buf) is likewise refused. Errors from the call wrap its [Result], so
// errors.Is reports the NVML error.
//
// One synchronous native call is not a hard wall-time bound: the driver
// decides how long the call takes, and it cannot be interrupted.
func (d Device) ComputeRunningProcessesInto(buf []ProcessInfo) (int, error) {
	return runningProcessesInto("compute", buf, func(n *uint32, p *ProcessInfo) error {
		return deviceGetComputeRunningProcesses(d, n, p)
	})
}

// GraphicsRunningProcessesInto stores information about processes with a
// graphics context on d in buf and returns the number of entries stored.
// It makes exactly one call to nvmlDeviceGetGraphicsRunningProcesses_v3 and
// has the bounds and errors of [Device.ComputeRunningProcessesInto].
// One synchronous native call is not a hard wall-time bound.
func (d Device) GraphicsRunningProcessesInto(buf []ProcessInfo) (int, error) {
	return runningProcessesInto("graphics", buf, func(n *uint32, p *ProcessInfo) error {
		return deviceGetGraphicsRunningProcesses(d, n, p)
	})
}

// MPSComputeRunningProcessesInto stores information about processes with a
// Multi-Process Service compute context on d in buf and returns the number of
// entries stored. It makes exactly one call to
// nvmlDeviceGetMPSComputeRunningProcesses_v3 and has the bounds and errors of
// [Device.ComputeRunningProcessesInto]. An entry does not identify an MPS
// server or client relationship. One synchronous native call is not a hard
// wall-time bound.
func (d Device) MPSComputeRunningProcessesInto(buf []ProcessInfo) (int, error) {
	return runningProcessesInto("mps compute", buf, func(n *uint32, p *ProcessInfo) error {
		return deviceGetMPSComputeRunningProcesses(d, n, p)
	})
}

// processCapacity returns n as a call capacity, refusing zero and lengths
// that do not fit in a uint32.
func processCapacity(n int) (uint32, error) {
	if n <= 0 {
		return 0, errors.New("empty buffer")
	}
	if uint64(n) > math.MaxUint32 {
		return 0, fmt.Errorf("buffer length %d exceeds %d", n, uint32(math.MaxUint32))
	}
	return uint32(n), nil
}

// runningProcessesInto makes the single raw query call with capacity len(buf).
func runningProcessesInto(kind string, buf []ProcessInfo, call func(*uint32, *ProcessInfo) error) (int, error) {
	capacity, err := processCapacity(len(buf))
	if err != nil {
		return 0, fmt.Errorf("query %s processes: %v", kind, err)
	}
	n := capacity
	err = call(&n, &buf[0])
	runtime.KeepAlive(buf)
	if errors.Is(err, ErrInsufficientSize) {
		return 0, fmt.Errorf("query %s processes: %d entries do not fit buffer %d: %w", kind, n, capacity, err)
	}
	if err != nil {
		return 0, fmt.Errorf("query %s processes: %w", kind, err)
	}
	if n > capacity {
		return 0, fmt.Errorf("query %s processes: count %d exceeds buffer %d", kind, n, capacity)
	}
	return int(n), nil
}
