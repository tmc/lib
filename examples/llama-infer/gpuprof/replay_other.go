//go:build !darwin

package gpuprof

import (
	"context"
)

// HardwareReplayData summarizes GPU hardware performance telemetry parsed from a replay session.
//
// On non-darwin platforms the telemetry comes from a cudaprof capture rather
// than from a Metal replay, so PeakGPRs is always zero: a capture records
// activity records and code artifacts, and register usage is not among them.
type HardwareReplayData struct {
	ActiveTimeNS int64
	PeakGPRs     int
	KernelCount  int
	Capture      *CaptureSummary
}

// ProfileReplay summarizes the cudaprof capture at capturePath.
//
// Unlike the darwin implementation this replays nothing: a CUDA capture is
// already the recorded result of a run, so the reported activity is what that
// run retained. A capture whose producer dropped records reports the survivors,
// which is why the underlying CaptureSummary is returned alongside the totals.
func ProfileReplay(ctx context.Context, capturePath string) (*HardwareReplayData, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	sum, err := ReadCaptureSummary(capturePath)
	if err != nil {
		return nil, err
	}
	return &HardwareReplayData{
		ActiveTimeNS: sum.KernelTimeNS,
		KernelCount:  sum.KernelLaunches,
		Capture:      sum,
	}, nil
}
