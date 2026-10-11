package ncclras_test

import (
	"bytes"
	"context"
	"fmt"
	"time"

	"github.com/tmc/lib/nvidia/x/ncclinspector"
)

const diagnosticCaptureBytes = 4 << 20

// waitDiagnosticCapture samples the growing file with bounded reads. Only the
// unfinished last line is deferred; malformed complete lines and mismatched
// identities fail immediately. Acceptance requires a fully terminated snapshot.
// These limits bound the observer, not the upstream writer's disk usage.
func waitDiagnosticCapture(ctx context.Context, read func() ([]byte, error), host string, pid int, hash string, interval time.Duration) ([]byte, error) {
	total := 0
	for attempt := 0; attempt < 80; attempt++ {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("Inspector capture incomplete: %w", err)
		}
		b, err := read()
		if err != nil {
			return nil, fmt.Errorf("Inspector capture: %w", err)
		}
		total += len(b)
		if len(b) > diagnosticCaptureBytes || total > 32<<20 {
			return nil, fmt.Errorf("Inspector capture byte budget exceeded")
		}
		end := bytes.LastIndexByte(b, '\n') + 1
		if end > 0 {
			records, err := ncclinspector.Read(bytes.NewReader(b[:end]), ncclinspector.Limits{})
			if err != nil {
				return nil, err
			}
			var counts, announced [2]uint64
			var stats [2]bool
			seen := [2]map[uint64]bool{{}, {}}
			for _, r := range records {
				if r.Metadata.PID != pid || r.Metadata.Hostname != host || r.Header.Ranks != 2 || r.Header.Nodes != 1 || r.Header.ID != hash {
					return nil, fmt.Errorf("Inspector process/membership differs")
				}
				if r.Stats != nil {
					if r.Stats.CollectiveDroppedTotal != 0 || r.Stats.PointToPointDroppedTotal != 0 || r.Stats.CollectiveDropped != 0 || r.Stats.PointToPointDropped != 0 {
						return nil, fmt.Errorf("dropped Inspector operations")
					}
					if counts[r.Header.Rank] != announced[r.Header.Rank] {
						return nil, fmt.Errorf("Inspector dump block count differs")
					}
					stats[r.Header.Rank] = true
					announced[r.Header.Rank] += r.Stats.Collectives
					if r.Stats.PointToPoints != 0 {
						return nil, fmt.Errorf("Inspector point-to-point records differ")
					}
				}
				if r.Collective != nil {
					if r.Collective.Name != "AllReduce" || r.Collective.Bytes != 65536 {
						return nil, fmt.Errorf("Inspector operation differs")
					}
					if seen[r.Header.Rank][r.Collective.Sequence] {
						return nil, fmt.Errorf("duplicate Inspector collective sequence")
					}
					seen[r.Header.Rank][r.Collective.Sequence] = true
					counts[r.Header.Rank]++
					if counts[r.Header.Rank] > announced[r.Header.Rank] || counts[r.Header.Rank] > 4 {
						return nil, fmt.Errorf("Inspector collective count differs")
					}
				}
				if r.PointToPoint != nil {
					return nil, fmt.Errorf("Inspector point-to-point operation differs")
				}
			}
			if end == len(b) && counts == [2]uint64{4, 4} && counts == announced && stats[0] && stats[1] {
				return append([]byte(nil), b...), nil
			}
		}
		timer := time.NewTimer(interval)
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil, fmt.Errorf("Inspector capture incomplete: %w", ctx.Err())
		case <-timer.C:
		}
	}
	return nil, fmt.Errorf("Inspector capture observation budget exhausted")
}
