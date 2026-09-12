package cuptiactivity

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/cupti"
)

// ScopedRecord identifies the collection that owns a record. Process identifies
// a process generation, not just an OS PID. IDs must come from retained producer
// identity; this type does not infer them from clocks, devices or names.
type ScopedRecord struct {
	Session string
	Run     string
	Process string
	Record  Record
}

// ExternalMatch describes the external ID associated with Records[Index].
// Status is matched, not_observed, or ambiguous. ExternalID is meaningful only
// when Status is matched; zero is a valid external ID. A matched ID establishes
// only the native external-correlation link, not the identity of a Python frame.
type ExternalMatch struct {
	Index      int
	ExternalID uint64
	Status     string
}

// MatchExternal joins native external-correlation records to API and device
// activities by session, run, process generation and native correlation ID.
// It returns one result per supported API/kernel/copy/memset/memory/synchronization
// record, in input order. Missing mappings stay unmatched. Multiple distinct external IDs for
// one key are ambiguous; no timing or naming heuristic selects between them.
// Memcpy also checks its separately retained RuntimeCorrelationID. Either native
// ID may supply an external link; differing external IDs or an ambiguous link
// on either path make the result ambiguous. Missing IDs are never inferred.
//
// kind selects one external ID namespace. Collection losses remain the caller's
// responsibility: a match is not a claim that collection was complete.
func MatchExternal(records []ScopedRecord, kind cupti.ExternalCorrelationKind) ([]ExternalMatch, error) {
	if kind <= cupti.ExternalCorrelationKindInvalid || kind >= cupti.ExternalCorrelationKindSize {
		return nil, errors.New("cuptiactivity: invalid external correlation kind")
	}
	type key struct {
		session, run, process string
		correlation           uint32
	}
	type mapping struct {
		id        uint64
		ambiguous bool
	}
	links := make(map[key]mapping)
	for i, r := range records {
		if r.Session == "" || r.Run == "" || r.Process == "" {
			return nil, fmt.Errorf("cuptiactivity: record %d lacks collection identity", i)
		}
		a := r.Record
		if a.Kind != cupti.ActivityKindExternalCorrelation || a.ExternalKind != kind || a.CorrelationID == 0 {
			continue
		}
		k := key{r.Session, r.Run, r.Process, a.CorrelationID}
		if old, ok := links[k]; ok {
			if old.id != a.ExternalID {
				old.ambiguous = true
				links[k] = old
			}
		} else {
			links[k] = mapping{id: a.ExternalID}
		}
	}
	var matches []ExternalMatch
	for i, r := range records {
		a := r.Record
		switch a.Kind {
		case cupti.ActivityKindDriver, cupti.ActivityKindRuntime, cupti.ActivityKindConcurrentKernel, cupti.ActivityKindMemcpy, cupti.ActivityKindMemset, cupti.ActivityKindMemory2, cupti.ActivityKindSynchronization:
		default:
			continue
		}
		m := ExternalMatch{Index: i, Status: "not_observed"}
		ids := [2]uint32{a.CorrelationID}
		if a.Kind == cupti.ActivityKindMemcpy {
			ids[1] = a.RuntimeCorrelationID
		}
		for _, id := range ids {
			link, ok := links[key{r.Session, r.Run, r.Process, id}]
			if !ok || id == 0 {
				continue
			}
			if link.ambiguous || m.Status == "matched" && m.ExternalID != link.id {
				m.Status = "ambiguous"
				m.ExternalID = 0
				break
			} else {
				m.Status = "matched"
				m.ExternalID = link.id
			}
		}
		matches = append(matches, m)
	}
	return matches, nil
}
