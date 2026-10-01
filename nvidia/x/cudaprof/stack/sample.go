package stack

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
)

// Sample is a statistical language stack snapshot, never submission evidence.
// Python wall snapshots include blocked threads. Time is an unaligned source
// clock until a consumer verifies the corresponding native clock capability.
type Sample struct {
	Version        int     `json:"version"`
	Type           string  `json:"type"`
	Process        string  `json:"process"`
	PID            int     `json:"pid"`
	Session        string  `json:"session"`
	Generation     uint64  `json:"generation"`
	Language       string  `json:"language"`
	TID            uint64  `json:"tid"`
	Goroutine      uint64  `json:"goroutine,omitempty"`
	TraceTime      uint64  `json:"trace_time,omitempty"`
	Time           uint64  `json:"time"`
	Clock          string  `json:"clock"`
	Status         string  `json:"status"`
	Provenance     string  `json:"provenance"`
	Frames         []Frame `json:"frames,omitempty"`
	Truncated      bool    `json:"truncated,omitempty"`
	Attempted      uint64  `json:"attempted,omitempty"`
	Recorded       uint64  `json:"recorded,omitempty"`
	Dropped        uint64  `json:"dropped,omitempty"`
	TruncatedCount uint64  `json:"truncated_count,omitempty"`
	Final          bool    `json:"final,omitempty"`
}

// ReadSamples reads bounded sample JSONL, retaining terminal coverage records.
// It rejects mixed identities and unsupported clocks or sampling provenance.
// Zero native generation and thread IDs are retained as unavailable evidence.
func ReadSamples(rd io.Reader, limits Limits) ([]Sample, error) {
	l := limits.defaults()
	s := bufio.NewScanner(io.LimitReader(rd, l.Bytes+1))
	s.Buffer(make([]byte, 4096), 1<<20)
	s.Split(splitLines)
	var out []Sample
	var size int64
	var count int
	var identity Sample
	var final bool
	for s.Scan() {
		size += int64(len(s.Bytes()) + 1)
		if size > l.Bytes {
			return nil, fmt.Errorf("stack sample byte limit exceeded")
		}
		if final {
			return nil, fmt.Errorf("stack sample after final coverage")
		}
		if err := checkJSON(s.Bytes()); err != nil {
			return nil, err
		}
		d := json.NewDecoder(bytes.NewReader(s.Bytes()))
		d.DisallowUnknownFields()
		var r Sample
		if err := d.Decode(&r); err != nil {
			return nil, fmt.Errorf("read stack sample: %w", err)
		}
		if r.Version != Version || r.PID <= 0 || r.Process == "" || r.Language != "python" || r.Provenance != "wall_stack_snapshot" || r.Clock != "monotonic" {
			return nil, fmt.Errorf("unsupported stack sample identity or provenance")
		}
		if len(out) == 0 {
			identity = r
		} else if r.PID != identity.PID || r.Session != identity.Session || r.Generation != identity.Generation || r.Process != identity.Process {
			return nil, fmt.Errorf("mixed stack sample identity")
		}
		switch r.Type {
		case "sample":
			count++
			if count > l.Records {
				return nil, fmt.Errorf("stack sample record limit exceeded")
			}
			if r.Time == 0 || r.Status != "clock_unavailable" || len(r.Frames) == 0 || len(r.Frames) > l.Frames {
				return nil, fmt.Errorf("invalid stack sample")
			}
			for _, f := range r.Frames {
				if f.Function == "" {
					return nil, fmt.Errorf("stack sample frame has no function")
				}
			}
		case "coverage":
			if !r.Final || r.Recorded > r.Attempted || r.Dropped > r.Attempted-r.Recorded || r.Recorded != uint64(count) {
				return nil, fmt.Errorf("invalid stack sample coverage")
			}
			final = true
		default:
			return nil, fmt.Errorf("unsupported stack sample record type")
		}
		out = append(out, r)
	}
	if err := s.Err(); err != nil {
		return nil, fmt.Errorf("read stack samples: %w", err)
	}
	return out, nil
}
