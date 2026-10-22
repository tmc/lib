package main

import (
	"time"

	"golang.org/x/sys/unix"
)

// uptimeNS reads the clock gputrace calls cpu_uptime_ns. On Apple silicon
// a Metal command buffer's CPU timestamps come from mach_absolute_time,
// which CLOCK_UPTIME_RAW reports in nanoseconds, so a host span and a GPU
// timing sample are directly comparable.
func uptimeNS() uint64 {
	var ts unix.Timespec
	if err := unix.ClockGettime(unix.CLOCK_UPTIME_RAW, &ts); err != nil {
		return 0
	}
	return uint64(ts.Sec)*1e9 + uint64(ts.Nsec)
}

// unixNS reads CLOCK_REALTIME, the domain gpuevent spans declare.
func unixNS() uint64 { return uint64(time.Now().UnixNano()) }
