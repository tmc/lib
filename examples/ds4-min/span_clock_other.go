//go:build !darwin && !linux

package main

import "time"

// uptimeNS has no uptime clock to read on this platform, so spans carry a
// realtime stamp in both fields. Nothing here binds them to GPU evidence.
func uptimeNS() uint64 { return uint64(time.Now().UnixNano()) }

// unixNS reads CLOCK_REALTIME, the domain gpuevent spans declare.
func unixNS() uint64 { return uint64(time.Now().UnixNano()) }
