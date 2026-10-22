package dsv4bench

import (
	"fmt"
	"io"
	"math"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

func observeMemory(state dsv4.State, snapshot dsv4.Snapshot, row *Row) error {
	r, ok := state.(dsv4.MemoryReporter)
	if !ok {
		return fmt.Errorf("benchmark state memory: %w", dsv4.ErrUnimplemented)
	}
	m, err := r.MemoryStats()
	if err != nil {
		return fmt.Errorf("benchmark state memory: %w", err)
	}
	total, err := m.Total()
	if err != nil {
		return err
	}
	var size int64
	if snapshot != nil {
		sizer, ok := snapshot.(dsv4.SnapshotSizer)
		if !ok {
			return fmt.Errorf("benchmark snapshot size: %w", dsv4.ErrUnimplemented)
		}
		size, err = sizer.SnapshotBytes()
		if err != nil || size < 0 {
			return fmt.Errorf("benchmark snapshot size: invalid size %d: %w", size, err)
		}
	}
	rss, err := peakRSSBytes()
	if err != nil {
		return err
	}
	row.ContextRawBytes, row.ContextCompressedBytes, row.ContextScratchBytes = m.Raw, m.Compressed, m.Scratch
	row.ContextTotalBytes, row.KVCacheBytes, row.PeakRSSBytes = total, size, rss
	return nil
}

func normalizePeakRSS(value int64, kib bool) (int64, error) {
	if value < 0 || kib && value > math.MaxInt64/1024 {
		return 0, fmt.Errorf("peak RSS: invalid byte count")
	}
	if kib {
		value *= 1024
	}
	return value, nil
}

// WriteMeasuredCSV writes only completely observed benchmark rows.
func WriteMeasuredCSV(w io.Writer, result RunResult) error {
	rows := make([]Row, len(result.Measurements))
	if len(rows) == 0 {
		return fmt.Errorf("benchmark CSV: no measurements")
	}
	for i, m := range result.Measurements {
		if !m.MemoryObserved {
			return fmt.Errorf("benchmark CSV: row %d memory unavailable", i)
		}
		rows[i] = m.Row
	}
	return WriteCSV(w, rows)
}
