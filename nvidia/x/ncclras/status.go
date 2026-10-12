package ncclras

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strings"
	"unicode"
)

// Limits bounds input bytes, physical line length and total communicator/rank
// records. Zero fields select defaults of 4 MiB, 64 KiB and 65536 records.
// Maxima are 16 MiB, 1 MiB and 262144 records.
type Limits struct {
	Bytes   int
	Line    int
	Records int
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
		return l, errors.New("ras: invalid limits")
	}
	return l, nil
}

// Status is one complete RAS JSON response. Raw preserves its exact bytes,
// including unrecognized fields; it does not imply every rank responded.
type Status struct {
	Version            string         `json:"nccl_version"`
	CUDARuntime        int            `json:"cuda_runtime_version"`
	CUDADriver         int            `json:"cuda_driver_version"`
	Timestamp          string         `json:"timestamp"`
	CommunicatorsCount int            `json:"communicators_count"`
	Communicators      []Communicator `json:"communicators"`
	Collection         Collection     `json:"ras"`
	Raw                []byte         `json:"-"`
}

// Communicator retains rank membership, including explicitly missing ranks.
type Communicator struct {
	Hash              string        `json:"hash"`
	SecondaryHash     string        `json:"secondary_hash"`
	Size              int           `json:"size"`
	RanksCount        int           `json:"ranks_count"`
	MissingRanksCount int           `json:"missing_ranks_count"`
	Ranks             []Rank        `json:"ranks"`
	MissingRanks      []MissingRank `json:"missing_ranks"`
}

// Rank is one responding rank. Counts are cumulative collective counters,
// not measurements of completion, throughput or ordering between ranks.
type Rank struct {
	Rank       int               `json:"rank"`
	Host       string            `json:"host"`
	PID        int               `json:"pid"`
	CUDADevice int               `json:"cuda_dev"`
	NVMLDevice int               `json:"nvml_dev"`
	Status     RankStatus        `json:"status"`
	Counts     map[string]uint64 `json:"collective_counts"`
}

// RankStatus preserves the reported NCCL status values without interpreting
// unavailable or failed initialization as a healthy state.
type RankStatus struct {
	InitState      int  `json:"init_state"`
	AsyncError     int  `json:"async_error"`
	FinalizeCalled bool `json:"finalize_called"`
	Destroy        bool `json:"destroy_flag"`
	Abort          bool `json:"abort_flag"`
}

// MissingRank is the reported identity and liveness state of an absent rank.
type MissingRank struct {
	Rank       int           `json:"rank"`
	Host       string        `json:"host"`
	PID        int           `json:"pid"`
	CUDADevice int           `json:"cuda_dev"`
	NVMLDevice int           `json:"nvml_dev"`
	Status     MissingStatus `json:"status"`
}

// MissingStatus is reported by RAS, not independently probed by this reader.
type MissingStatus struct {
	Unresponsive   bool `json:"unresponsive"`
	ConsideredDead bool `json:"considered_dead"`
}

// Collection records the server collection duration and timeout count.
type Collection struct {
	Seconds  float64 `json:"collection_time_sec"`
	Timeouts int     `json:"timeouts_count"`
}

// Read reads a bounded, complete status JSON document. It rejects unsupported
// versions, omitted required fields, duplicate keys, contradictory rank counts
// and trailing documents. Unknown fields are retained in Status.Raw.
// The caller must arrange cancellation for a reader that can block.
func Read(r io.Reader, limits Limits) (*Status, error) {
	l, err := limits.checked()
	if err != nil {
		return nil, err
	}
	if r == nil {
		return nil, errors.New("ras: nil reader")
	}
	b, err := io.ReadAll(io.LimitReader(r, int64(l.Bytes)+1))
	if err != nil {
		return nil, fmt.Errorf("ras read: %w", err)
	}
	if len(b) > l.Bytes {
		return nil, errors.New("ras: byte limit exceeded")
	}
	for _, line := range bytes.Split(b, []byte{'\n'}) {
		if len(line) > l.Line {
			return nil, errors.New("ras: line limit exceeded")
		}
	}
	// A legacy version preamble is producer evidence even when this reader
	// cannot decode its text format. Do not infer a version from protocol 2.
	first, _, _ := bytes.Cut(b, []byte{'\n'})
	if match := textVersion.FindStringSubmatch(string(first)); match != nil {
		return nil, &UnsupportedError{Format: "json", Version: match[1]}
	}
	if err := uniqueJSON(b); err != nil {
		return nil, fmt.Errorf("ras JSON: %w", err)
	}
	root, err := requiredObject(b, "nccl_version", "cuda_runtime_version", "cuda_driver_version", "timestamp", "communicators_count", "communicators", "ras")
	if err != nil {
		return nil, err
	}
	var s Status
	if err := json.Unmarshal(b, &s); err != nil {
		return nil, fmt.Errorf("ras status: %w", err)
	}
	// 2.32.3 is admitted from its emitter source (testdata/README.md).
	if s.Version != "2.29.1" && s.Version != "2.32.3" {
		return nil, &UnsupportedError{Format: "json", Version: s.Version}
	}
	if s.CUDARuntime <= 0 || s.CUDADriver <= 0 || s.Timestamp == "" || s.CommunicatorsCount < 0 || s.CommunicatorsCount != len(s.Communicators) || !array(root["communicators"]) {
		return nil, errors.New("ras: invalid status identity or communicator population")
	}
	if _, err := requiredObject(root["ras"], "collection_time_sec", "timeouts_count"); err != nil {
		return nil, err
	}
	if math.IsNaN(s.Collection.Seconds) || math.IsInf(s.Collection.Seconds, 0) || s.Collection.Seconds < 0 || s.Collection.Timeouts < 0 {
		return nil, errors.New("ras: invalid collection metadata")
	}
	var rawComms []json.RawMessage
	if err := json.Unmarshal(root["communicators"], &rawComms); err != nil {
		return nil, err
	}
	remaining := l.Records
	seen := make(map[string]bool)
	for i, c := range s.Communicators {
		if remaining < 1 {
			return nil, errors.New("ras: record limit exceeded")
		}
		remaining--
		m, err := requiredObject(rawComms[i], "hash", "secondary_hash", "size", "ranks_count", "missing_ranks_count", "ranks", "missing_ranks")
		if err != nil {
			return nil, err
		}
		identity := c.Hash + "/" + c.SecondaryHash
		if c.Hash == "" || c.SecondaryHash == "" || seen[identity] || c.Size < 1 || c.RanksCount < 0 || c.MissingRanksCount < 0 || c.RanksCount != len(c.Ranks) || c.MissingRanksCount != len(c.MissingRanks) || c.RanksCount > c.Size || c.MissingRanksCount != c.Size-c.RanksCount || !array(m["ranks"]) || !array(m["missing_ranks"]) {
			return nil, errors.New("ras: invalid communicator membership")
		}
		seen[identity] = true
		ids := make(map[int]bool)
		for _, key := range []string{"ranks", "missing_ranks"} {
			var ranks []json.RawMessage
			if err := json.Unmarshal(m[key], &ranks); err != nil {
				return nil, err
			}
			for j, body := range ranks {
				if remaining < 1 {
					return nil, errors.New("ras: record limit exceeded")
				}
				remaining--
				rm, err := requiredObject(body, "rank", "host", "pid", "cuda_dev", "nvml_dev", "status")
				if err != nil {
					return nil, err
				}
				var rank, pid, cuda, nvml int
				var host string
				if err := json.Unmarshal(rm["rank"], &rank); err != nil {
					return nil, err
				}
				if key == "ranks" {
					r := c.Ranks[j]
					pid, host, cuda, nvml = r.PID, r.Host, r.CUDADevice, r.NVMLDevice
					if _, err := requiredObject(rm["status"], "init_state", "async_error", "finalize_called", "destroy_flag", "abort_flag"); err != nil {
						return nil, err
					}
					counts, err := requiredObject(rm["collective_counts"], "Broadcast", "Reduce", "AllGather", "ReduceScatter", "AllReduce")
					if err != nil {
						return nil, err
					}
					for _, v := range counts {
						var n uint64
						if err := json.Unmarshal(v, &n); err != nil || bytes.Equal(bytes.TrimSpace(v), []byte("null")) {
							return nil, errors.New("ras: invalid collective count")
						}
					}
				} else {
					r := c.MissingRanks[j]
					pid, host, cuda, nvml = r.PID, r.Host, r.CUDADevice, r.NVMLDevice
					if _, err := requiredObject(rm["status"], "unresponsive", "considered_dead"); err != nil {
						return nil, err
					}
				}
				if rank < 0 || rank >= c.Size || ids[rank] || pid <= 0 || host == "" || cuda < 0 || nvml < 0 {
					return nil, errors.New("ras: invalid or duplicate rank identity")
				}
				ids[rank] = true
			}
		}
	}
	s.Raw = b
	return &s, nil
}

func array(b []byte) bool { b = bytes.TrimSpace(b); return len(b) > 0 && b[0] == '[' }

func requiredObject(b []byte, keys ...string) (map[string]json.RawMessage, error) {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(b, &m); err != nil || m == nil {
		return nil, errors.New("ras: required object is missing or malformed")
	}
	for _, k := range keys {
		if v, ok := m[k]; !ok || bytes.Equal(bytes.TrimSpace(v), []byte("null")) {
			return nil, fmt.Errorf("ras: missing %s", k)
		}
	}
	return m, nil
}

// JSON decoding otherwise accepts duplicate and case-aliased struct fields.
func uniqueJSON(b []byte) error {
	d := json.NewDecoder(bytes.NewReader(b))
	d.UseNumber()
	if err := jsonValue(d, 0); err != nil {
		return err
	}
	if _, err := d.Token(); err != io.EOF {
		return errors.New("trailing or malformed JSON")
	}
	return nil
}

func jsonValue(d *json.Decoder, depth int) error {
	if depth > 32 {
		return errors.New("JSON nesting limit exceeded")
	}
	t, err := d.Token()
	if err != nil {
		return err
	}
	delim, ok := t.(json.Delim)
	if !ok {
		return nil
	}
	if delim != '{' && delim != '[' {
		return errors.New("unexpected JSON delimiter")
	}
	seen := make(map[string]bool)
	for d.More() {
		if delim == '{' {
			t, err := d.Token()
			if err != nil {
				return err
			}
			k, ok := t.(string)
			if !ok {
				return errors.New("invalid JSON key")
			}
			k = fold(k)
			if seen[k] {
				return errors.New("duplicate or case-aliased JSON key")
			}
			seen[k] = true
		}
		if err := jsonValue(d, depth+1); err != nil {
			return err
		}
	}
	_, err = d.Token()
	return err
}

func fold(s string) string {
	return strings.Map(func(r rune) rune {
		m := r
		for n := unicode.SimpleFold(r); n != r; n = unicode.SimpleFold(n) {
			if n < m {
				m = n
			}
		}
		return m
	}, s)
}
