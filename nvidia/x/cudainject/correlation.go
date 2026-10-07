package cudainject

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// API retains the producer's API domain, callback and host-thread attribution.
// CallbackID is a CUPTI domain-specific ID, not a portable function name.
type API struct {
	Domain        string `json:"domain"`
	CallbackID    uint32 `json:"callback_id"`
	ThreadID      uint64 `json:"thread_id"`
	CorrelationID uint64 `json:"correlation"`
}

// Correlations describes exact joins in one agent identity. Missing and
// ambiguous joins are retained; there is no timestamp-based fallback.
type Correlations struct {
	Identity        Identity
	APIs            []API
	Matched         []cudacapture.CorrelationJoin
	Ambiguous       []cudacapture.AmbiguousCorrelation
	MissingAPI      []uint64
	MissingActivity []uint64
}

// JoinCorrelations joins the existing API and kernel payloads in records from
// exactly one session/process/generation. It accepts at most 4096 records,
// 16 MiB per payload and 64 MiB of payloads in total.
// API records are not assumed to be launches: unmatched calls remain explicit.
// Multiple API records for one ID (including different domains) are ambiguous.
func JoinCorrelations(records []Record) (Correlations, error) {
	var result Correlations
	if len(records) > 4096 {
		return result, ErrRecordLimit
	}
	tracker := cudacapture.NewCorrelationTracker()
	var total uint64
	for i, record := range records {
		total += uint64(len(record.Payload))
		if len(record.Payload) > 16<<20 || total > 64<<20 {
			return result, ErrOversize
		}
		if i == 0 {
			result.Identity = record.Identity
		}
		if record.Identity != result.Identity {
			return result, errors.New("cudainject: mixed correlation identities")
		}
		switch record.Kind {
		case KindAPI:
			var api API
			if err := correlationPayload(record.Payload, &api, "domain", "callback_id", "thread_id", "correlation"); err != nil {
				return result, fmt.Errorf("cudainject: API payload: %w", err)
			}
			if api.Domain != "runtime" && api.Domain != "driver" {
				return result, errors.New("cudainject: unsupported API correlation domain")
			}
			result.APIs = append(result.APIs, api)
			// Zero denotes an uncorrelated API and must not join a kernel.
			if api.CorrelationID == 0 {
				continue
			}
			if err := tracker.RecordRuntime(cudacapture.RuntimeLaunch{CorrelationID: api.CorrelationID, HostThreadID: api.ThreadID}); err != nil {
				return result, err
			}
		case KindActivity:
			var activity cudacapture.ActivityRecord
			if err := correlationPayload(record.Payload, &activity, "subtype"); err != nil {
				return result, fmt.Errorf("cudainject: activity payload: %w", err)
			}
			if activity.Subtype != "kernel" {
				continue
			}
			if err := correlationPayload(record.Payload, &activity, "name", "valid", "start", "end", "correlation"); err != nil {
				return result, err
			}
			if !activity.Valid || activity.Correlation == 0 || activity.End < activity.Start {
				return result, errors.New("cudainject: invalid kernel correlation")
			}
			if err := tracker.RecordActivity(activity); err != nil {
				return result, err
			}
		case KindExternal:
			var external struct {
				Kind        uint32 `json:"external_kind"`
				ID          uint64 `json:"external_id"`
				Correlation uint64 `json:"correlation"`
			}
			if err := correlationPayload(record.Payload, &external, "external_kind", "external_id", "correlation"); err != nil {
				return result, fmt.Errorf("cudainject: external correlation payload: %w", err)
			}
			if err := tracker.RecordExternalCorrelation(external.Kind, external.ID, external.Correlation); err != nil {
				return result, err
			}
		}
	}
	result.Matched, result.Ambiguous, result.MissingActivity, result.MissingAPI = tracker.Join()
	return result, nil
}
