//go:build !linux && !darwin

package main

import "time"

// No GPU backend is admitted on these platforms.
func powerSleep(d time.Duration) { time.Sleep(d) }

func powerSleepSupported() error { return nil }
