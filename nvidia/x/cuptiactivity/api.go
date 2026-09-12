package cuptiactivity

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/cupti"
)

// APIMatch links Records[Index] to a native API record at APIIndex. Status is
// matched, not_observed, or ambiguous. APIIndex is -1 unless matched. The link
// identifies an API invocation, not a Python operator, stack, or successful
// operation. Inspect the API record's timestamps and return value separately.
type APIMatch struct {
	Index    int
	APIIndex int
	Status   string
}

// MatchAPI joins kernel, memcpy and memset records to API invocations by exact
// session, run, process generation and native correlation identity. kind must
// be ActivityKindDriver or ActivityKindRuntime; the two domains are separate.
// It returns one result per device activity in input order, without modifying
// records. Delivery order, names and timestamp proximity never establish links.
//
// Kernels use CorrelationID in either selected domain. Memcpy uses CorrelationID
// for driver APIs and RuntimeCorrelationID for runtime APIs. Memset has only a
// driver link in the retained layout; runtime matches stay not_observed. Zero
// IDs never match. Identical duplicate API records use the first input index;
// conflicting records for an identity are ambiguous, even if one is incomplete.
// One API may link multiple activities, such as a graph launch: do not add its
// duration once per linked activity. A match does not establish loss-free capture
// or qualify subtraction between API and device timestamps.
func MatchAPI(records []ScopedRecord, kind cupti.ActivityKind) ([]APIMatch, error) {
	if kind != cupti.ActivityKindDriver && kind != cupti.ActivityKindRuntime {
		return nil, errors.New("cuptiactivity: invalid API activity kind")
	}
	type key struct {
		session, run, process string
		correlation           uint32
	}
	type link struct {
		index     int
		ambiguous bool
	}
	links := make(map[key]link)
	for i, s := range records {
		if s.Session == "" || s.Run == "" || s.Process == "" {
			return nil, fmt.Errorf("cuptiactivity: record %d lacks collection identity", i)
		}
		r := s.Record
		if r.Kind != kind || r.CorrelationID == 0 {
			continue
		}
		k := key{s.Session, s.Run, s.Process, r.CorrelationID}
		if old, ok := links[k]; ok {
			if records[old.index].Record != r {
				old.ambiguous = true
				links[k] = old
			}
		} else {
			links[k] = link{index: i}
		}
	}
	var out []APIMatch
	for i, s := range records {
		r := s.Record
		id := r.CorrelationID
		switch r.Kind {
		case cupti.ActivityKindConcurrentKernel:
		case cupti.ActivityKindMemcpy:
			if kind == cupti.ActivityKindRuntime {
				id = r.RuntimeCorrelationID
			}
		case cupti.ActivityKindMemset:
			if kind == cupti.ActivityKindRuntime {
				id = 0
			}
		default:
			continue
		}
		m := APIMatch{Index: i, APIIndex: -1, Status: "not_observed"}
		if l, ok := links[key{s.Session, s.Run, s.Process, id}]; ok && id != 0 {
			if l.ambiguous {
				m.Status = "ambiguous"
			} else {
				m.Status, m.APIIndex = "matched", l.index
			}
		}
		out = append(out, m)
	}
	return out, nil
}
