//go:build !linux && !darwin

package dsv4bench

import (
	"fmt"
	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

func peakRSSBytes() (int64, error) {
	return 0, fmt.Errorf("peak RSS: %w", dsv4.ErrUnimplemented)
}
