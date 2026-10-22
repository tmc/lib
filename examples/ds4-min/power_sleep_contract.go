package main

import "time"

func oneShotPowerSleep(d time.Duration, nanosleep func(sec, nsec int64) int32) {
	if d <= 0 {
		return
	}
	// Ignore the result, including EINTR, exactly once. Do not retry a
	// throttling sleep after cancellation signals.
	_ = nanosleep(int64(d/time.Second), int64(d%time.Second))
}
