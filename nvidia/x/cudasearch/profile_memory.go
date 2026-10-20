package cudasearch

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"reflect"

	"github.com/tmc/lib/egraphcore"
	"github.com/tmc/lib/nvidia/x/cudaprof"
)

// profileFallback is an admission refusal that selects the unmodified action
// order. It is distinct from a storage refusal, which fails the search.
type profileFallback string

func (f profileFallback) Error() string { return string(f) }

// jsonBound accumulates an upper bound on a JSON encoding. It walks every
// value once without a visit limit and calls no marshaler.
type jsonBound struct {
	bytes, entries int64
	keyText        int64 // integer map keys, each converted to text while sorting
	err            error
}

func (b *jsonBound) add(n int64) {
	if b.err == nil && (n < 0 || n > math.MaxInt64/4-b.bytes) {
		b.err = ErrMemoryBudgetExceeded
		return
	}
	b.bytes += n
}

// str adds a quoted string: at most six output bytes per input byte.
func (b *jsonBound) str(s string) {
	if int64(len(s)) > (math.MaxInt64/4)/6 {
		b.err = ErrMemoryBudgetExceeded
		return
	}
	b.add(6*int64(len(s)) + 2)
}

// counters adds a map of named counts; each entry also needs a sort slot.
func (b *jsonBound) counters(m map[string]int64) {
	b.entries += int64(len(m))
	b.add(2)
	for k := range m {
		b.str(k)
		b.add(32 + 2)
	}
}

// jsonSortSlot is the key sorting slot json.Marshal makes for each map entry.
type jsonSortSlot struct {
	key, value reflect.Value
	name       string
}

// maxKeyText is the longest decimal uint64 key.
const maxKeyText = 20

// reserve charges scratch for encoding the bounded value: three times the
// output, since json.Marshal returns a copy of its pooled buffer, whose
// capacity may reach twice the output; the sort slots; and the text of each
// integer map key.
func (b *jsonBound) reserve(scratch *egraphcore.MemoryScope) error {
	if b.err != nil {
		return b.err
	}
	if err := bookkeepingSlots[jsonSortSlot](scratch, int(b.entries)); err != nil {
		return err
	}
	if b.keyText > (math.MaxInt64-3*b.bytes)/maxKeyText {
		return ErrMemoryBudgetExceeded
	}
	return scratch.Reserve(3*b.bytes + maxKeyText*b.keyText)
}

// snapshotBound bounds json.Marshal of s in closed form, so profiles of any
// size are admissible under a sufficient cap.
func snapshotBound(s *cudaprof.ProfileSnapshot) *jsonBound {
	b := new(jsonBound)
	b.add(1024) // field names, numbers and time
	for _, v := range []string{s.CodeDigest, s.Region, s.Workload, s.TokensHash, s.Device, s.CollectorMode} {
		b.str(v)
	}
	if p := s.Sampling; p != nil {
		b.add(512)
		b.str(p.RunID)
		b.str(p.KernelName)
	}
	b.counters(s.Metrics)
	b.entries += int64(len(s.MetricUnits))
	b.add(2)
	for k, v := range s.MetricUnits {
		b.str(k)
		b.str(v)
		b.add(2)
	}
	b.entries += int64(len(s.PCStalls))
	b.keyText += int64(len(s.PCStalls))
	b.add(2)
	for _, counters := range s.PCStalls {
		b.add(32 + 2) // quoted decimal key
		b.counters(counters)
	}
	return b
}

// decisionBound bounds json.Marshal of d in closed form.
func decisionBound(d *ProfileDecision) *jsonBound {
	b := new(jsonBound)
	b.add(1024)
	for _, v := range []string{d.Target.CodeDigest, d.Target.Region, d.Target.Workload, d.Target.TokensHash, d.Target.Device, d.Target.CollectorMode, d.MinCoverage, d.ID, d.SnapshotSHA256, d.SnapshotError, d.Fallback, d.IdentityError} {
		b.str(v)
	}
	for _, p := range d.Priors {
		b.add(256) // field names and minimum
		b.str(p.Metric)
		b.str(p.Unit)
		b.str(p.Rule)
	}
	for _, r := range d.ActiveRules {
		b.str(r)
		b.add(1)
	}
	return b
}

// reserveDecision charges result for the storage the decision retains before
// it is built: the record, its prior copy and the fixed-size digests and
// coverage text. Strings borrowed from config are not charged.
func reserveDecision(config *ProfileActions, result *egraphcore.MemoryScope) error {
	if err := result.Reserve(int64(reflect.TypeFor[ProfileDecision]().Size())); err != nil {
		return err
	}
	if err := bookkeepingSlots[MetricRulePrior](result, len(config.Priors)); err != nil {
		return err
	}
	// MinCoverage text, SnapshotSHA256 and ID.
	return result.Reserve(32 + 2*64)
}

// snapshotDigest returns the identity of the encoded snapshot or the encoding
// error text. With scopes, the encoding is reserved in scratch before it is
// built, and retained error text in result.
func snapshotDigest(snap *cudaprof.ProfileSnapshot, scratch, result *egraphcore.MemoryScope) (digest, encodeErr string, err error) {
	if scratch != nil {
		if err := snapshotBound(snap).reserve(scratch); err != nil {
			return "", "", err
		}
		// Digest array and its hexadecimal text.
		if err := scratch.Reserve(32 + 64); err != nil {
			return "", "", err
		}
	}
	data, err := json.Marshal(snap)
	if err != nil {
		return "", err.Error(), result.Reserve(int64(len(err.Error())))
	}
	return fmt.Sprintf("%x", sha256.Sum256(data)), "", nil
}

// identifyDecision sets the ID or IdentityError of d, which must have both
// empty. With scopes, the encoding is reserved in scratch before it is built.
func identifyDecision(d *ProfileDecision, scratch, result *egraphcore.MemoryScope) error {
	if scratch != nil {
		if err := decisionBound(d).reserve(scratch); err != nil {
			return err
		}
		if err := scratch.Reserve(32 + 64); err != nil {
			return err
		}
	}
	data, err := json.Marshal(d)
	if err != nil {
		d.IdentityError = err.Error()
		return result.Reserve(int64(len(d.IdentityError)))
	}
	d.ID = fmt.Sprintf("%x", sha256.Sum256(data))
	return nil
}
