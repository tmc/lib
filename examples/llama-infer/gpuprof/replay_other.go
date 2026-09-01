//go:build !darwin

package gpuprof

import (
	"context"
	"errors"
)

// HardwareReplayData summarizes GPU hardware performance telemetry parsed from a replay session.
type HardwareReplayData struct {
	ActiveTimeNS int64
	PeakGPRs     int
	KernelCount  int
}

// ProfileReplay returns an unsupported error on non-darwin platforms.
func ProfileReplay(ctx context.Context, capturePath string) (*HardwareReplayData, error) {
	return nil, errors.New("gpuprof: replay profiling only supported on darwin")
}
