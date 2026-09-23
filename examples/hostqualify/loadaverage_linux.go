package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

// loadAverage reads the 1-minute load average, the first field of
// /proc/loadavg.
func loadAverage() (float64, error) {
	data, err := os.ReadFile("/proc/loadavg")
	if err != nil {
		return 0, fmt.Errorf("read /proc/loadavg: %w", err)
	}
	fields := strings.Fields(string(data))
	if len(fields) == 0 {
		return 0, fmt.Errorf("read /proc/loadavg: empty")
	}
	load, err := strconv.ParseFloat(fields[0], 64)
	if err != nil {
		return 0, fmt.Errorf("read /proc/loadavg: %w", err)
	}
	return load, nil
}
