//go:build darwin

package gpuprof

import (
	"context"
	"fmt"
	"time"

	"github.com/tmc/apple/x/gpureplay"
)

// HardwareReplayData summarizes GPU hardware performance telemetry parsed from a replay session.
type HardwareReplayData struct {
	ActiveTimeNS       int64
	PeakGPRs           int
	KernelCount        int
	Report             *gpureplay.ReplayReport
}

// ProfileReplay runs MTLReplayer on the provided trace bundle and extracts hardware stats.
func ProfileReplay(ctx context.Context, capturePath string) (*HardwareReplayData, error) {
	if err := gpureplay.Available(); err != nil {
		return nil, err
	}

	rep, err := gpureplay.Replay(ctx, capturePath, gpureplay.Options{
		Timeout: 30 * time.Second,
	})
	if err != nil {
		return nil, fmt.Errorf("gpuprof replay: %w", err)
	}

	return &HardwareReplayData{
		ActiveTimeNS: rep.TotalGPUActiveTime.Nanoseconds(),
		PeakGPRs:     rep.PeakRegisterUsage,
		KernelCount:  len(rep.KernelTimings),
		Report:       rep,
	}, nil
}
