//go:build darwin

package metalprof

import (
	"context"
	"fmt"
	"time"

	"github.com/tmc/apple/x/gpureplay"
)

// ReplayHardwareProfile aggregates hardware-measured GPU kernel execution timings and shader statistics.
type ReplayHardwareProfile struct {
	TotalGPUActiveTime time.Duration
	PeakRegisterUsage  int
	KernelTimings      []gpureplay.KernelTiming
	ShaderMetrics      []gpureplay.ShaderMetric
}

// ReplayCapture replays a captured .gputrace bundle via MTLReplayer and returns detailed hardware metrics.
func ReplayCapture(ctx context.Context, capturePath string) (*ReplayHardwareProfile, error) {
	if err := gpureplay.Available(); err != nil {
		return nil, fmt.Errorf("metalprof: %w", err)
	}

	report, err := gpureplay.Replay(ctx, capturePath, gpureplay.Options{
		Timeout: 45 * time.Second,
	})
	if err != nil {
		return nil, fmt.Errorf("metalprof replay: %w", err)
	}

	return &ReplayHardwareProfile{
		TotalGPUActiveTime: report.TotalGPUActiveTime,
		PeakRegisterUsage:  report.PeakRegisterUsage,
		KernelTimings:      report.KernelTimings,
		ShaderMetrics:      report.ShaderMetrics,
	}, nil
}
