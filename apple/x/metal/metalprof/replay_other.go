//go:build !darwin

package metalprof

import (
	"context"
	"errors"
	"time"
)

// ReplayHardwareProfile aggregates hardware-measured GPU kernel execution timings and shader statistics.
type ReplayHardwareProfile struct {
	TotalGPUActiveTime time.Duration
	PeakRegisterUsage  int
}

// ReplayCapture returns an error on non-darwin platforms.
func ReplayCapture(ctx context.Context, capturePath string) (*ReplayHardwareProfile, error) {
	return nil, errors.New("metalprof: replay profiling only supported on darwin")
}
