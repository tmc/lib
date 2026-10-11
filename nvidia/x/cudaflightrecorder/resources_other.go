//go:build !linux

package cudaflightrecorder

import (
	"context"
	"fmt"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
)

func sampleHostUsage(ctx context.Context, window time.Duration) (float64, uint64, error) {
	if err := ctx.Err(); err != nil {
		return 0, 0, err
	}
	return 0, 0, fmt.Errorf("sample host resources: %w", cudaerrors.ErrUnsupported)
}
