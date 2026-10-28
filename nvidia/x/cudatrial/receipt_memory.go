package cudatrial

import (
	"fmt"
	"math"
	"strings"
	"time"
	"unsafe"
)

// Clone returns a receipt whose maps and coverage counters are detached. Error
// values remain shared so that errors.Is and producer error identities survive.
func (r Receipt) Clone() Receipt {
	out, _ := r.CloneWithMemory(nil)
	return out
}

// CloneWithMemory reserves conservative logical storage before copying receipt
// metadata. reserve may be nil for unbounded copies. The caller owns successful
// reservations, including those made before an error, and releases them when
// the copy is no longer used. It does not measure producer-internal allocations.
func (r Receipt) CloneWithMemory(reserve func(int64) error) (Receipt, error) {
	var n int64 = int64(unsafe.Sizeof(r))
	add := func(v int64) error {
		if v < 0 || n > math.MaxInt64-v {
			return fmt.Errorf("cudatrial: receipt memory size overflow")
		}
		n += v
		return nil
	}
	str := func(s string) error { return add(int64(len(s))) }
	for _, s := range []string{r.Name, r.Digest, string(r.TimingKind), r.Before.Device, r.After.Device} {
		if err := str(s); err != nil {
			return Receipt{}, err
		}
	}
	// Include map headers, buckets and conservative growth slack. The mutable
	// populations are known before any destination collection is allocated.
	for _, count := range []int{len(r.Timings), len(r.Provenance), len(r.Before.Values), len(r.After.Values)} {
		if int64(count) > (math.MaxInt64-64)/128 {
			return Receipt{}, fmt.Errorf("cudatrial: receipt memory size overflow")
		}
		if err := add(64 + 128*int64(count)); err != nil {
			return Receipt{}, err
		}
	}
	for k := range r.Timings {
		if err := str(string(k)); err != nil {
			return Receipt{}, err
		}
	}
	for k, v := range r.Provenance {
		if err := str(k); err != nil {
			return Receipt{}, err
		}
		if err := str(v); err != nil {
			return Receipt{}, err
		}
	}
	for k := range r.Before.Values {
		if err := str(string(k)); err != nil {
			return Receipt{}, err
		}
	}
	for k := range r.After.Values {
		if err := str(string(k)); err != nil {
			return Receipt{}, err
		}
	}
	if r.Coverage != nil {
		if err := add(int64(unsafe.Sizeof(EdgeBitmap{})) + int64(len(r.Coverage.counts))); err != nil {
			return Receipt{}, err
		}
	}
	if reserve != nil {
		if err := reserve(n); err != nil {
			return Receipt{}, err
		}
	}
	out := r
	out.Name = strings.Clone(r.Name)
	out.Digest = strings.Clone(r.Digest)
	out.TimingKind = TimingKind(strings.Clone(string(r.TimingKind)))
	if r.Timings != nil {
		out.Timings = make(map[TimingKind]time.Duration, len(r.Timings))
		for k, v := range r.Timings {
			out.Timings[TimingKind(strings.Clone(string(k)))] = v
		}
	}
	if r.Provenance != nil {
		out.Provenance = make(map[string]string, len(r.Provenance))
		for k, v := range r.Provenance {
			out.Provenance[strings.Clone(k)] = strings.Clone(v)
		}
	}
	out.Before = cloneConditions(r.Before)
	out.After = cloneConditions(r.After)
	if r.Coverage != nil {
		out.Coverage = &EdgeBitmap{counts: make([]byte, len(r.Coverage.counts))}
		copy(out.Coverage.counts, r.Coverage.counts)
	}
	return out, nil
}

func cloneConditions(c Conditions) Conditions {
	c.Device = strings.Clone(c.Device)
	if c.Values != nil {
		values := make(map[ConditionKind]int64, len(c.Values))
		for k, v := range c.Values {
			values[ConditionKind(strings.Clone(string(k)))] = v
		}
		c.Values = values
	}
	return c
}
