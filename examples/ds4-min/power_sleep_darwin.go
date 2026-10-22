//go:build darwin

package main

import (
	"fmt"
	"sync"
	"time"

	"github.com/ebitengine/purego"
)

type powerTimespec struct{ sec, nsec int64 }

var darwinPowerSleep struct {
	once sync.Once
	call func(*powerTimespec, *powerTimespec) int32
	err  error
}

func powerSleepSupported() error {
	darwinPowerSleep.once.Do(func() {
		// RTLD_DEFAULT borrows libSystem already loaded by the process. There
		// is no acquired library handle or native owner to close.
		address, err := purego.Dlsym(purego.RTLD_DEFAULT, "nanosleep")
		if err != nil {
			darwinPowerSleep.err = fmt.Errorf("resolve power nanosleep: %w", err)
			return
		}
		purego.RegisterFunc(&darwinPowerSleep.call, address)
	})
	return darwinPowerSleep.err
}

func powerSleep(d time.Duration) {
	// Admission resolves this function before any request construction.
	oneShotPowerSleep(d, func(sec, nsec int64) int32 {
		request := powerTimespec{sec: sec, nsec: nsec}
		return darwinPowerSleep.call(&request, nil)
	})
}
