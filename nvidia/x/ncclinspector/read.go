package ncclinspector

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
)

// Limits bounds total bytes, line length and record count. Zero fields use
// 4 MiB, 64 KiB and 65536; maxima are 16 MiB, 1 MiB and 262144.
type Limits struct{ Bytes, Line, Records int }

// Header identifies a communicator and rank. Name can be absent in older
// v4.1 examples; ID, rank membership and node count are required.
type Header struct {
	ID    string `json:"id"`
	Name  string `json:"comm_name"`
	Rank  int    `json:"rank"`
	Ranks int    `json:"n_ranks"`
	Nodes uint64 `json:"nnodes"`
}

// Metadata preserves the producer version, build, process and dump timestamp.
type Metadata struct {
	Version         string `json:"inspector_output_format_version"`
	GitRevision     string `json:"git_rev"`
	Mechanism       string `json:"rec_mechanism"`
	TimestampMicros uint64 `json:"dump_timestamp_us"`
	Hostname        string `json:"hostname"`
	PID             int    `json:"pid"`
}

// Collective is a collective operation observation. TimingSource can be absent
// in older examples; no timing origin is guessed when it is absent.
type Collective struct {
	Name               string  `json:"coll"`
	Sequence           uint64  `json:"coll_sn"`
	Bytes              uint64  `json:"coll_msg_size_bytes"`
	Micros             uint64  `json:"coll_exec_time_us"`
	TimingSource       string  `json:"coll_timing_source"`
	AlgorithmBandwidth float64 `json:"coll_algobw_gbs"`
	BusBandwidth       float64 `json:"coll_busbw_gbs"`
}

// PointToPoint is one send/receive observation. Its sequence domain is not
// interchangeable with collective sequences or other communicators/ranks.
type PointToPoint struct {
	Name               string  `json:"p2p"`
	Sequence           uint64  `json:"p2p_sn"`
	Peer               int     `json:"p2p_peer"`
	Bytes              uint64  `json:"p2p_msg_size_bytes"`
	Micros             uint64  `json:"p2p_exec_time_us"`
	TimingSource       string  `json:"p2p_timing_source"`
	AlgorithmBandwidth float64 `json:"p2p_algobw_gbs"`
	BusBandwidth       float64 `json:"p2p_busbw_gbs"`
}

// DumpStats preserves reported ring-buffer counts, including cumulative and
// per-dump drops. Nonzero drops do not become a complete sample.
type DumpStats struct {
	Collectives              uint64 `json:"coll_records"`
	CollectiveDroppedTotal   uint64 `json:"coll_dropped_total"`
	CollectiveDropped        uint64 `json:"coll_dropped_since_last_dump"`
	PointToPoints            uint64 `json:"p2p_records"`
	PointToPointDroppedTotal uint64 `json:"p2p_dropped_total"`
	PointToPointDropped      uint64 `json:"p2p_dropped_since_last_dump"`
}

// Record contains exactly one operation or dump-statistics body. Raw is the
// exact input line including its newline. Nil Stats means unavailable for this
// record, not zero dropped operations or complete session coverage.
type Record struct {
	Header       Header        `json:"header"`
	Metadata     Metadata      `json:"metadata"`
	Collective   *Collective   `json:"coll_perf,omitempty"`
	PointToPoint *PointToPoint `json:"p2p_perf,omitempty"`
	Stats        *DumpStats    `json:"dump_stats,omitempty"`
	Raw          []byte        `json:"-"`
}

// Read reads a finite JSON Lines capture. Each nonempty line must end in a
// newline and contain one complete object. It rejects empty, partial, malformed
// and unsupported records, but does not infer session completion, reconcile
// ring counts or impose ordering across independently dumped sequence domains.
// The caller must arrange cancellation for a reader that can block.
func Read(r io.Reader, limits Limits) ([]Record, error) {
	l, err := limits.checked()
	if err != nil {
		return nil, err
	}
	if r == nil {
		return nil, errors.New("inspector: nil reader")
	}
	b, err := io.ReadAll(io.LimitReader(r, int64(l.Bytes)+1))
	if err != nil {
		return nil, fmt.Errorf("inspector read: %w", err)
	}
	if len(b) == 0 || b[len(b)-1] != '\n' {
		return nil, errors.New("inspector: empty or incomplete JSON Lines capture")
	}
	if len(b) > l.Bytes {
		return nil, errors.New("inspector: byte limit exceeded")
	}
	var records []Record
	for len(b) > 0 {
		i := bytes.IndexByte(b, '\n')
		line := b[:i+1]
		b = b[i+1:]
		if len(line)-1 > l.Line {
			return nil, errors.New("inspector: line limit exceeded")
		}
		if len(records) >= l.Records {
			return nil, errors.New("inspector: record limit exceeded")
		}
		record, err := readRecord(line)
		if err != nil {
			return nil, fmt.Errorf("inspector record %d: %w", len(records), err)
		}
		records = append(records, record)
	}
	return records, nil
}

func (l Limits) checked() (Limits, error) {
	if l.Bytes == 0 {
		l.Bytes = 4 << 20
	}
	if l.Line == 0 {
		l.Line = 64 << 10
	}
	if l.Records == 0 {
		l.Records = 65536
	}
	if l.Bytes < 1 || l.Bytes > 16<<20 || l.Line < 1 || l.Line > 1<<20 || l.Records < 1 || l.Records > 262144 {
		return l, errors.New("inspector: invalid limits")
	}
	return l, nil
}

func readRecord(line []byte) (Record, error) {
	var r Record
	if err := uniqueJSON(line); err != nil {
		return r, err
	}
	m, err := requiredObject(line, "header", "metadata")
	if err != nil {
		return r, err
	}
	if _, err = requiredObject(m["header"], "id", "rank", "n_ranks", "nnodes"); err != nil {
		return r, err
	}
	if _, err = requiredObject(m["metadata"], "inspector_output_format_version", "git_rev", "rec_mechanism", "dump_timestamp_us", "hostname", "pid"); err != nil {
		return r, err
	}
	if err = json.Unmarshal(line, &r); err != nil {
		return r, err
	}
	if r.Metadata.Version != "v4.1" {
		return r, fmt.Errorf("unsupported format version %q", r.Metadata.Version)
	}
	if r.Header.ID == "" || r.Header.Ranks < 1 || r.Header.Rank < 0 || r.Header.Rank >= r.Header.Ranks || r.Header.Nodes < 1 || r.Header.Nodes > uint64(r.Header.Ranks) || r.Metadata.PID < 1 || r.Metadata.Hostname == "" || r.Metadata.TimestampMicros == 0 || r.Metadata.Mechanism == "" {
		return r, errors.New("invalid communicator or process identity")
	}
	bodies := 0
	if body, ok := m["coll_perf"]; ok {
		bodies++
		if _, err = requiredObject(body, "coll", "coll_sn", "coll_msg_size_bytes", "coll_exec_time_us", "coll_algobw_gbs", "coll_busbw_gbs"); err != nil {
			return r, err
		}
		c := r.Collective
		if c == nil || !collectiveName(c.Name) || !bandwidth(c.AlgorithmBandwidth) || !bandwidth(c.BusBandwidth) {
			return r, errors.New("invalid collective observation")
		}
	}
	if body, ok := m["p2p_perf"]; ok {
		bodies++
		if _, err = requiredObject(body, "p2p", "p2p_sn", "p2p_peer", "p2p_msg_size_bytes", "p2p_exec_time_us", "p2p_algobw_gbs", "p2p_busbw_gbs"); err != nil {
			return r, err
		}
		c := r.PointToPoint
		if c == nil || (c.Name != "Send" && c.Name != "Recv") || c.Peer < 0 || c.Peer >= r.Header.Ranks || !bandwidth(c.AlgorithmBandwidth) || !bandwidth(c.BusBandwidth) {
			return r, errors.New("invalid point-to-point observation")
		}
	}
	if body, ok := m["dump_stats"]; ok {
		bodies++
		if _, err = requiredObject(body, "coll_records", "coll_dropped_total", "coll_dropped_since_last_dump", "p2p_records", "p2p_dropped_total", "p2p_dropped_since_last_dump"); err != nil {
			return r, err
		}
		c := r.Stats
		if c == nil || c.CollectiveDropped > c.CollectiveDroppedTotal || c.PointToPointDropped > c.PointToPointDroppedTotal {
			return r, errors.New("invalid drop counters")
		}
	}
	if bodies != 1 {
		return r, errors.New("exactly one operation or statistics body is required")
	}
	r.Raw = append([]byte(nil), line...)
	return r, nil
}

func bandwidth(f float64) bool { return !math.IsNaN(f) && !math.IsInf(f, 0) && f >= 0 }
func collectiveName(s string) bool {
	switch s {
	case "Broadcast", "Reduce", "AllGather", "ReduceScatter", "AllReduce", "All2All", "AllGatherV":
		return true
	}
	return false
}
