package nvtelemetry

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"unicode/utf8"

	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

// ServingAttemptRef names a version 1 serving attempt by its full identity.
// A reference with a missing or unmatched field is retained as supplied; a
// consumer cannot attach it to any attempt.
type ServingAttemptRef struct {
	RequestID   string `json:"request_id"`
	Attempt     string `json:"attempt"`
	Worker      string `json:"worker"`
	WorkerEpoch string `json:"worker_epoch"`
}

// ServingRoute is a supplied routing decision delivering one attempt from a
// frontend to its worker. Decision is routed, retried, rejected or unknown.
//
// Basis, here and in the other version 2 edges, is native_id when the
// producer reported the identities natively, annotation when the application
// associated them, or unknown. Evidence names the supporting artifact and is
// required unless the basis is unknown; for native_id it is the artifact's
// SHA-256.
type ServingRoute struct {
	Attempt  ServingAttemptRef `json:"attempt"`
	Frontend string            `json:"frontend"`
	Decision string            `json:"decision"`
	Basis    string            `json:"basis"`
	Evidence string            `json:"evidence,omitempty"`
}

// ServingBatch is a supplied batch executed by one worker restart epoch.
// Members and OpIDs are many-to-many: an attempt may appear in several
// batches and a batch's operations are shared work, not per-request shares.
// Process scopes OpIDs to the worker process that issued them.
type ServingBatch struct {
	ID          string                     `json:"id"`
	Worker      string                     `json:"worker"`
	WorkerEpoch string                     `json:"worker_epoch"`
	Role        string                     `json:"role"` // prefill, decode, combined, unknown
	Members     []ServingAttemptRef        `json:"members"`
	OpIDs       []cudaworkload.OpID        `json:"op_ids,omitempty"`
	Process     *cudaworkload.ProcessGenID `json:"process,omitempty"`
	Basis       string                     `json:"basis"`
	Evidence    string                     `json:"evidence,omitempty"`
}

// ServingCache declares one cache namespace generation owned by a worker
// restart epoch. A cache block is not general device memory.
type ServingCache struct {
	Namespace   string `json:"namespace"`
	Generation  string `json:"generation"`
	Worker      string `json:"worker"`
	WorkerEpoch string `json:"worker_epoch"`
	Evidence    string `json:"evidence"`
}

// ServingCacheUse is a supplied cache event for one attempt. Event is hit,
// miss, store, evict or unknown. A record is identified by its attempt,
// namespace, generation and event: it states that the event occurred, not
// how often, so a repeated event of one kind is a single record.
type ServingCacheUse struct {
	Attempt    ServingAttemptRef `json:"attempt"`
	Namespace  string            `json:"namespace"`
	Generation string            `json:"generation"`
	Event      string            `json:"event"`
	Basis      string            `json:"basis"`
	Evidence   string            `json:"evidence,omitempty"`
}

// ServingTransferRef references a MovementTransfer by ID from the movement
// context of the same run. It is not a second transfer record.
type ServingTransferRef struct {
	Attempt    ServingAttemptRef `json:"attempt"`
	TransferID string            `json:"transfer_id"`
	Basis      string            `json:"basis"`
	Evidence   string            `json:"evidence,omitempty"`
}

// ServingContextV2 is the version 2 serving interchange. Attempts keep the
// version 1 meaning; routes, batches, caches and transfer references extend
// them. ReadServingContext refuses version 2, and ReadServingContextV2
// refuses every other version.
type ServingContextV2 struct {
	Version   int                  `json:"version"`
	RunID     string               `json:"run_id"`
	Attempts  []ServingAttempt     `json:"attempts"`
	Routes    []ServingRoute       `json:"routes"`
	Batches   []ServingBatch       `json:"batches"`
	Caches    []ServingCache       `json:"caches"`
	CacheUses []ServingCacheUse    `json:"cache_uses"`
	Transfers []ServingTransferRef `json:"transfers"`
}

// V1 returns the version 1 projection of the attempts. The attempt slice is
// copied; a nil slice stays nil so the version 1 validator still refuses it.
func (c ServingContextV2) V1() ServingContext {
	v1 := ServingContext{Version: 1, RunID: c.RunID}
	if c.Attempts != nil {
		v1.Attempts = make([]ServingAttempt, len(c.Attempts))
		copy(v1.Attempts, c.Attempts)
	}
	return v1
}

// ReadServingContextV2 reads a bounded version 2 application interchange with
// its original bytes and digest. Attempts are validated as in version 1.
// Duplicate keys, unknown fields, other versions, unsupported enumerations,
// duplicate batch, cache or edge records and unsupported bases are refused.
// Route records are identified by attempt and frontend, cache uses as
// ServingCacheUse describes, and transfer references by attempt and
// transfer ID; basis and evidence are not part of a record's identity.
// Edge references are retained as supplied, including missing identities.
func ReadServingContextV2(r io.Reader, s WorkloadSource, l WorkloadLimits) (WorkloadExport, ServingContextV2, error) {
	if s.Provider != "application" {
		return WorkloadExport{}, ServingContextV2{}, errors.New("serving context requires application source")
	}
	e, l, err := readWorkload(r, s, l)
	if err != nil {
		return e, ServingContextV2{}, err
	}
	var c ServingContextV2
	if err := decodeWorkloadJSON(e.Raw, &c); err != nil {
		return e, c, err
	}
	d := json.NewDecoder(bytes.NewReader(e.Raw))
	d.DisallowUnknownFields()
	if err := d.Decode(&c); err != nil {
		return e, c, fmt.Errorf("decode serving context: %w", err)
	}
	if err := validateServingContextV2(c, s.RunID, l); err != nil {
		return e, c, err
	}
	return e, c, nil
}

// WriteServingContextV2 validates and writes the version 2 interchange,
// checking the encoded byte budget one record at a time.
func WriteServingContextV2(w io.Writer, c ServingContextV2, l WorkloadLimits) error {
	l, err := workloadLimits(l)
	if err != nil {
		return err
	}
	if w == nil {
		return errors.New("missing serving context writer")
	}
	if err := validateServingContextV2(c, c.RunID, l); err != nil {
		return err
	}
	name, _ := json.Marshal(c.RunID)
	raw := append([]byte(`{"version":2,"run_id":`), name...)
	lists := []struct {
		name    string
		records []any
	}{
		{"attempts", anySlice(c.Attempts)},
		{"routes", anySlice(c.Routes)},
		{"batches", anySlice(c.Batches)},
		{"caches", anySlice(c.Caches)},
		{"cache_uses", anySlice(c.CacheUses)},
		{"transfers", anySlice(c.Transfers)},
	}
	for _, list := range lists {
		raw = append(raw, []byte(`,"`+list.name+`":[`)...)
		for i, record := range list.records {
			encoded, err := json.Marshal(record)
			if err != nil {
				return err
			}
			if i > 0 {
				raw = append(raw, ',')
			}
			if int64(len(raw))+int64(len(encoded))+2 > l.MaxBytes {
				return errors.New("serving context byte limit exceeded")
			}
			raw = append(raw, encoded...)
		}
		raw = append(raw, ']')
	}
	raw = append(raw, '}')
	if int64(len(raw)) > l.MaxBytes {
		return errors.New("serving context byte limit exceeded")
	}
	n, err := w.Write(raw)
	if err == nil && n != len(raw) {
		return io.ErrShortWrite
	}
	return err
}

func anySlice[T any](s []T) []any {
	out := make([]any, len(s))
	for i := range s {
		out[i] = s[i]
	}
	return out
}

func validateServingContextV2(c ServingContextV2, run string, l WorkloadLimits) error {
	if c.Version != 2 {
		return errors.New("invalid serving context version")
	}
	if c.Routes == nil || c.Batches == nil || c.Caches == nil || c.CacheUses == nil || c.Transfers == nil {
		return errors.New("missing serving context extension list")
	}
	if err := validateServingContext(c.V1(), run, l); err != nil {
		return err
	}
	n := len(c.Attempts) + len(c.Routes) + len(c.Caches) + len(c.CacheUses) + len(c.Transfers)
	for _, b := range c.Batches {
		n += 1 + len(b.Members) + len(b.OpIDs)
	}
	if n > l.MaxMetrics {
		return errors.New("serving context record limit exceeded")
	}
	long := func(s ...string) bool {
		for _, v := range s {
			if len(v) > l.MaxStringBytes || !utf8.ValidString(v) {
				return true
			}
		}
		return false
	}
	ref := func(a ServingAttemptRef) bool {
		return long(a.RequestID, a.Attempt, a.Worker, a.WorkerEpoch)
	}
	basis := func(b, evidence string) bool {
		switch b {
		case "native_id":
			return streamHash(evidence)
		case "annotation":
			return evidence != ""
		case "unknown":
			return evidence == ""
		}
		return false
	}
	type routeKey struct {
		ServingAttemptRef
		frontend string
	}
	routes := make(map[routeKey]bool)
	for _, r := range c.Routes {
		if ref(r.Attempt) || long(r.Frontend, r.Evidence) {
			return errors.New("serving context string limit exceeded or invalid")
		}
		if r.Frontend == "" || !basis(r.Basis, r.Evidence) {
			return errors.New("invalid serving route")
		}
		switch r.Decision {
		case "routed", "retried", "rejected", "unknown":
		default:
			return errors.New("unsupported serving route decision")
		}
		k := routeKey{r.Attempt, r.Frontend}
		if routes[k] {
			return errors.New("duplicate serving route")
		}
		routes[k] = true
	}
	batches := make(map[string]bool)
	for _, b := range c.Batches {
		if long(b.ID, b.Worker, b.WorkerEpoch, b.Evidence) {
			return errors.New("serving context string limit exceeded or invalid")
		}
		if b.ID == "" || b.Members == nil || !basis(b.Basis, b.Evidence) || batches[b.ID] {
			return errors.New("invalid or duplicate serving batch")
		}
		switch b.Role {
		case "prefill", "decode", "combined", "unknown":
		default:
			return errors.New("unsupported serving batch role")
		}
		batches[b.ID] = true
		members := make(map[ServingAttemptRef]bool)
		for _, m := range b.Members {
			if ref(m) {
				return errors.New("serving context string limit exceeded or invalid")
			}
			if members[m] {
				return errors.New("duplicate serving batch member")
			}
			members[m] = true
		}
		if b.Process != nil && (b.Process.PID == 0 || b.Process.Generation == 0) {
			return errors.New("invalid serving batch process")
		}
		ops := make(map[cudaworkload.OpID]bool)
		for _, op := range b.OpIDs {
			if op == 0 || ops[op] {
				return errors.New("invalid or duplicate serving batch operation")
			}
			ops[op] = true
		}
	}
	type cacheKey struct{ namespace, generation string }
	caches := make(map[cacheKey]bool)
	for _, cache := range c.Caches {
		if long(cache.Namespace, cache.Generation, cache.Worker, cache.WorkerEpoch, cache.Evidence) {
			return errors.New("serving context string limit exceeded or invalid")
		}
		k := cacheKey{cache.Namespace, cache.Generation}
		if cache.Namespace == "" || cache.Generation == "" || cache.Worker == "" || cache.WorkerEpoch == "" || cache.Evidence == "" || caches[k] {
			return errors.New("invalid or duplicate serving cache generation")
		}
		caches[k] = true
	}
	type cacheUseKey struct {
		ServingAttemptRef
		namespace, generation, event string
	}
	uses := make(map[cacheUseKey]bool)
	for _, u := range c.CacheUses {
		if ref(u.Attempt) || long(u.Namespace, u.Generation, u.Evidence) {
			return errors.New("serving context string limit exceeded or invalid")
		}
		if !basis(u.Basis, u.Evidence) {
			return errors.New("invalid serving cache use")
		}
		switch u.Event {
		case "hit", "miss", "store", "evict", "unknown":
		default:
			return errors.New("unsupported serving cache event")
		}
		k := cacheUseKey{u.Attempt, u.Namespace, u.Generation, u.Event}
		if uses[k] {
			return errors.New("duplicate serving cache use")
		}
		uses[k] = true
	}
	type transferKey struct {
		ServingAttemptRef
		id string
	}
	transfers := make(map[transferKey]bool)
	for _, t := range c.Transfers {
		if ref(t.Attempt) || long(t.TransferID, t.Evidence) {
			return errors.New("serving context string limit exceeded or invalid")
		}
		k := transferKey{t.Attempt, t.TransferID}
		if t.TransferID == "" || !basis(t.Basis, t.Evidence) || transfers[k] {
			return errors.New("invalid or duplicate serving transfer reference")
		}
		transfers[k] = true
	}
	return nil
}
