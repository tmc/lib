//go:build linux

package main

import (
	"time"

	"golang.org/x/sys/unix"
)

func powerSleep(d time.Duration) {
	oneShotPowerSleep(d, func(sec, nsec int64) int32 {
		request := unix.Timespec{Sec: sec, Nsec: nsec}
		if unix.Nanosleep(&request, nil) != nil {
			return -1
		}
		return 0
	})
}

func powerSleepSupported() error { return nil }
