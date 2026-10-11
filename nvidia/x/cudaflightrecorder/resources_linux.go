//go:build linux

package cudaflightrecorder

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
	"syscall"
	"time"
)

func processCPUTime() (time.Duration, error) {
	var usage syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &usage); err != nil {
		return 0, err
	}
	return time.Duration(usage.Utime.Sec+usage.Stime.Sec)*time.Second + time.Duration(usage.Utime.Usec+usage.Stime.Usec)*time.Microsecond, nil
}

func sampleHostUsage(ctx context.Context, window time.Duration) (float64, uint64, error) {
	if err := ctx.Err(); err != nil {
		return 0, 0, err
	}
	before, err := processCPUTime()
	if err != nil {
		return 0, 0, fmt.Errorf("sample process CPU: %w", err)
	}
	start := time.Now()
	timer := time.NewTimer(window)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return 0, 0, ctx.Err()
	case <-timer.C:
	}
	after, err := processCPUTime()
	if err != nil {
		return 0, 0, fmt.Errorf("sample process CPU: %w", err)
	}
	elapsed := time.Since(start)
	data, err := os.ReadFile("/proc/self/statm")
	if err != nil {
		return 0, 0, fmt.Errorf("sample process RSS: %w", err)
	}
	fields := strings.Fields(string(data))
	if len(fields) < 2 {
		return 0, 0, fmt.Errorf("sample process RSS: incomplete statm")
	}
	pages, err := strconv.ParseUint(fields[1], 10, 64)
	if err != nil {
		return 0, 0, fmt.Errorf("sample process RSS: %w", err)
	}
	pageSize := uint64(os.Getpagesize())
	if pages > ^uint64(0)/pageSize {
		return 0, 0, fmt.Errorf("sample process RSS: size overflow")
	}
	return float64(after-before) / float64(elapsed) * 100, pages * pageSize, nil
}
