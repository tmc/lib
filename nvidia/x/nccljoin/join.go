package nccljoin

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"sort"

	"github.com/tmc/lib/nvidia/x/ncclinspector"
)

// Scope identifies the caller-selected capture, job and communicator epoch.
// These strings are asserted by the caller, not verified or inferred by Join.
type Scope struct{ Capture, Job, Epoch string }

// Capture holds one decoded input and its caller-reported partial provenance.
// ID is unique within Scope. Partial and Truncated do not identify which
// operation was lost. Records can be out of order or repeat an earlier dump.
type Capture struct {
	ID                 string
	Partial, Truncated bool
	Records            []ncclinspector.Record
}

// Options bounds inputs and supplies optional caller-verified membership by
// communicator ID. Otherwise Header.Ranks defines ranks [0, Ranks). Zero limits
// use 65536 records, 4096 ranks, 16384 groups and 16 MiB of encoded/raw input.
// Maxima are 262144 records, 65536 ranks, 65536 groups and 64 MiB.
// Total expected membership entries are also bounded by MaxRecords.
type Options struct {
	Scope                                     Scope
	Expected                                  map[string][]int
	MaxRecords, MaxRanks, MaxGroups, MaxBytes int
}

// Observation retains one rank's duration, timing source and dump provenance.
// Repeated observations remain explicit even if their operation facts agree.
type Observation struct {
	Capture    string
	Header     ncclinspector.Header
	Metadata   ncclinspector.Metadata
	Collective ncclinspector.Collective
	Raw        []byte
}

// Collective aligns one communicator/sequence pair. Observations are sorted by
// rank, capture and dump timestamp. Unobserved uses explicit expected membership.
// A conflicting same-rank operation or collective name/size returns an error.
type Collective struct {
	Communicator                  string
	Sequence                      uint64
	Name                          string
	Bytes                         uint64
	Expected, Present, Unobserved []int
	Observations                  []Observation
}

// CaptureFacts preserves partial flags and raw dump-statistics observations.
// Statistics are not summed across repeated dumps or assigned to missing ranks.
type CaptureFacts struct {
	ID                 string
	Partial, Truncated bool
	Statistics         []ncclinspector.Record
}

// Result contains deterministic groups and independent capture provenance.
type Result struct {
	Scope       Scope
	Collectives []Collective
	Captures    []CaptureFacts
}

// Join validates bounded caller-created records and aligns collective facts.
// It refuses point-to-point records, unknown formats and inconsistent membership.
// It copies all retained slices and operation bodies; input order is irrelevant.
func Join(captures []Capture, opts Options) (*Result, error) {
	limits := []*int{&opts.MaxRecords, &opts.MaxRanks, &opts.MaxGroups, &opts.MaxBytes}
	defaults := []int{65536, 4096, 16384, 16 << 20}
	maxima := []int{262144, 65536, 65536, 64 << 20}
	for i, p := range limits {
		if *p == 0 {
			*p = defaults[i]
		}
		if *p < 1 || *p > maxima[i] {
			return nil, errors.New("nccljoin: invalid limits")
		}
	}
	validText := func(s string) bool { return s != "" && len(s) <= 4096 }
	if !validText(opts.Scope.Capture) || !validText(opts.Scope.Job) || !validText(opts.Scope.Epoch) {
		return nil, errors.New("nccljoin: explicit scope required")
	}
	if len(captures) > opts.MaxRecords || len(opts.Expected) > opts.MaxGroups {
		return nil, errors.New("nccljoin: input limit exceeded")
	}
	membership := make(map[string][]int)
	members := 0
	for id, ranks := range opts.Expected {
		if !validText(id) || len(ranks) == 0 || len(ranks) > opts.MaxRanks {
			return nil, errors.New("nccljoin: invalid expected membership")
		}
		members += len(ranks)
		if members > opts.MaxRecords {
			return nil, errors.New("nccljoin: membership limit exceeded")
		}
		ranks = append([]int(nil), ranks...)
		sort.Ints(ranks)
		for i, r := range ranks {
			if r < 0 || r >= opts.MaxRanks || (i > 0 && r == ranks[i-1]) {
				return nil, errors.New("nccljoin: invalid expected rank")
			}
		}
		membership[id] = ranks
	}
	type key struct {
		id  string
		seq uint64
	}
	groups := make(map[key]*Collective)
	type rankKey struct {
		group key
		rank  int
	}
	first := make(map[rankKey]Observation)
	counts := make(map[string]int)
	nodes := make(map[string]uint64)
	names := make(map[string]string)
	seen := make(map[string]bool)
	records, totalBytes := 0, 0
	result := &Result{Scope: opts.Scope}
	for _, capture := range captures {
		if !validText(capture.ID) || seen[capture.ID] {
			return nil, errors.New("nccljoin: unique capture IDs required")
		}
		seen[capture.ID] = true
		facts := CaptureFacts{ID: capture.ID, Partial: capture.Partial, Truncated: capture.Truncated}
		for _, r := range capture.Records {
			records++
			if records > opts.MaxRecords {
				return nil, errors.New("nccljoin: record limit exceeded")
			}
			if r.Header.Ranks > opts.MaxRanks || !validText(r.Header.ID) || len(r.Header.Name) > 4096 || !validText(r.Metadata.Hostname) || !validText(r.Metadata.Mechanism) || len(r.Metadata.GitRevision) > 4096 || len(r.Metadata.Version) > 4096 || len(r.Raw) > opts.MaxBytes {
				return nil, errors.New("nccljoin: bounded record identity required")
			}
			if r.PointToPoint != nil {
				return nil, errors.New("nccljoin: point-to-point observations unsupported")
			}
			if r.Collective != nil && (len(r.Collective.Name) > 4096 || len(r.Collective.TimingSource) > 4096) {
				return nil, errors.New("nccljoin: operation string limit exceeded")
			}
			b, err := json.Marshal(r)
			if err != nil {
				return nil, fmt.Errorf("nccljoin encode: %w", err)
			}
			totalBytes += len(b) + len(r.Raw)
			if totalBytes > opts.MaxBytes {
				return nil, errors.New("nccljoin: byte limit exceeded")
			}
			parsed, err := ncclinspector.Read(bytes.NewReader(append(b, '\n')), ncclinspector.Limits{Bytes: 16 << 20, Line: 1 << 20, Records: 1})
			if err != nil {
				return nil, fmt.Errorf("nccljoin record: %w", err)
			}
			clean := parsed[0]
			clean.Raw = append([]byte(nil), r.Raw...)
			if len(r.Raw) > 0 {
				raw, err := ncclinspector.Read(bytes.NewReader(r.Raw), ncclinspector.Limits{Bytes: 16 << 20, Line: 1 << 20, Records: 1})
				if err != nil || len(raw) != 1 {
					return nil, errors.New("nccljoin: invalid raw record")
				}
				raw[0].Raw = nil
				compare := clean
				compare.Raw = nil
				if !reflect.DeepEqual(raw[0], compare) {
					return nil, errors.New("nccljoin: raw record differs from decoded facts")
				}
			}
			id := r.Header.ID
			if n, ok := counts[id]; ok && (n != r.Header.Ranks || nodes[id] != r.Header.Nodes || names[id] != r.Header.Name) {
				return nil, errors.New("nccljoin: conflicting communicator membership")
			}
			counts[id] = r.Header.Ranks
			nodes[id] = r.Header.Nodes
			names[id] = r.Header.Name
			if ranks, ok := membership[id]; ok {
				if len(ranks) != r.Header.Ranks {
					return nil, errors.New("nccljoin: expected membership differs")
				}
				for i, v := range ranks {
					if i != v {
						return nil, errors.New("nccljoin: expected ranks must match communicator rank range")
					}
				}
			} else {
				members += r.Header.Ranks
				if members > opts.MaxRecords {
					return nil, errors.New("nccljoin: membership limit exceeded")
				}
				ranks := make([]int, r.Header.Ranks)
				for i := range ranks {
					ranks[i] = i
				}
				membership[id] = ranks
			}
			if clean.Stats != nil {
				facts.Statistics = append(facts.Statistics, clean)
				continue
			}
			c := *clean.Collective
			k := key{id, c.Sequence}
			g := groups[k]
			if g == nil {
				if len(groups) >= opts.MaxGroups {
					return nil, errors.New("nccljoin: group limit exceeded")
				}
				g = &Collective{Communicator: id, Sequence: c.Sequence, Name: c.Name, Bytes: c.Bytes, Expected: append([]int(nil), membership[id]...)}
				groups[k] = g
			}
			if g.Name != c.Name || g.Bytes != c.Bytes {
				return nil, errors.New("nccljoin: conflicting collective identity")
			}
			rk := rankKey{k, r.Header.Rank}
			if old, ok := first[rk]; ok && (old.Collective != c || old.Metadata.Hostname != r.Metadata.Hostname || old.Metadata.PID != r.Metadata.PID || old.Metadata.Version != r.Metadata.Version || old.Metadata.GitRevision != r.Metadata.GitRevision || old.Metadata.Mechanism != r.Metadata.Mechanism) {
				return nil, errors.New("nccljoin: conflicting duplicate observation")
			}
			first[rk] = Observation{capture.ID, clean.Header, clean.Metadata, c, nil}
			g.Observations = append(g.Observations, Observation{capture.ID, clean.Header, clean.Metadata, c, clean.Raw})
		}
		sort.Slice(facts.Statistics, func(i, j int) bool {
			a, b := facts.Statistics[i], facts.Statistics[j]
			if a.Header.ID != b.Header.ID {
				return a.Header.ID < b.Header.ID
			}
			if a.Header.Rank != b.Header.Rank {
				return a.Header.Rank < b.Header.Rank
			}
			return a.Metadata.TimestampMicros < b.Metadata.TimestampMicros
		})
		result.Captures = append(result.Captures, facts)
	}
	for _, g := range groups {
		sort.Slice(g.Observations, func(i, j int) bool {
			a, b := g.Observations[i], g.Observations[j]
			if a.Header.Rank != b.Header.Rank {
				return a.Header.Rank < b.Header.Rank
			}
			if a.Capture != b.Capture {
				return a.Capture < b.Capture
			}
			if a.Metadata.TimestampMicros != b.Metadata.TimestampMicros {
				return a.Metadata.TimestampMicros < b.Metadata.TimestampMicros
			}
			return bytes.Compare(a.Raw, b.Raw) < 0
		})
		present := make(map[int]bool)
		for _, o := range g.Observations {
			present[o.Header.Rank] = true
		}
		for _, r := range g.Expected {
			if present[r] {
				g.Present = append(g.Present, r)
			} else {
				g.Unobserved = append(g.Unobserved, r)
			}
		}
		result.Collectives = append(result.Collectives, *g)
	}
	sort.Slice(result.Collectives, func(i, j int) bool {
		a, b := result.Collectives[i], result.Collectives[j]
		if a.Communicator != b.Communicator {
			return a.Communicator < b.Communicator
		}
		return a.Sequence < b.Sequence
	})
	sort.Slice(result.Captures, func(i, j int) bool { return result.Captures[i].ID < result.Captures[j].ID })
	return result, nil
}
