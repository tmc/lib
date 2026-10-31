package autotune

import (
	"cmp"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"slices"
)

// Fingerprint returns a stable identifier for t: the hex SHA-256 of its
// JSON encoding, truncated to 16 bytes. Twins that differ in any field
// have different fingerprints.
func (t HardwareTwin) Fingerprint() string {
	b, err := json.Marshal(t)
	if err != nil {
		panic(err) // HardwareTwin has only plain fields
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:16])
}

// A CacheEntry records the tuning outcome for one workload on one device.
type CacheEntry struct {
	Fingerprint string            `json:"fingerprint"`
	ShapeKey    string            `json:"shape_key"`
	Weights     ScheduleWeights   `json:"weights"`
	Winner      ScheduleCandidate `json:"winner"`
	WinnerUS    float64           `json:"winner_us"`
}

// A TuneCache persists [CacheEntry] values so later runs can skip warmup
// exploration. Its JSON encoding is deterministic: entries sorted by
// fingerprint and shape key.
type TuneCache struct {
	Entries []CacheEntry `json:"entries"`
}

// Put adds or replaces the entry for e's fingerprint and shape key.
func (c *TuneCache) Put(e CacheEntry) {
	i, ok := slices.BinarySearchFunc(c.Entries, e, compareEntries)
	if ok {
		c.Entries[i] = e
		return
	}
	c.Entries = slices.Insert(c.Entries, i, e)
}

// Get returns the entry for the fingerprint and shape key.
func (c *TuneCache) Get(fingerprint, shapeKey string) (CacheEntry, bool) {
	i, ok := slices.BinarySearchFunc(c.Entries, CacheEntry{Fingerprint: fingerprint, ShapeKey: shapeKey}, compareEntries)
	if !ok {
		return CacheEntry{}, false
	}
	return c.Entries[i], true
}

func compareEntries(a, b CacheEntry) int {
	if c := cmp.Compare(a.Fingerprint, b.Fingerprint); c != 0 {
		return c
	}
	return cmp.Compare(a.ShapeKey, b.ShapeKey)
}

// MarshalJSON encodes c with its entries sorted.
func (c *TuneCache) MarshalJSON() ([]byte, error) {
	es := slices.Clone(c.Entries)
	slices.SortFunc(es, compareEntries)
	return json.Marshal(struct {
		Entries []CacheEntry `json:"entries"`
	}{es})
}

// UnmarshalJSON decodes c and sorts its entries, rejecting duplicates.
func (c *TuneCache) UnmarshalJSON(b []byte) error {
	var v struct {
		Entries []CacheEntry `json:"entries"`
	}
	if err := json.Unmarshal(b, &v); err != nil {
		return fmt.Errorf("unmarshal tune cache: %w", err)
	}
	slices.SortFunc(v.Entries, compareEntries)
	for i := 1; i < len(v.Entries); i++ {
		if compareEntries(v.Entries[i-1], v.Entries[i]) == 0 {
			return fmt.Errorf("unmarshal tune cache: duplicate entry %s/%s", v.Entries[i].Fingerprint, v.Entries[i].ShapeKey)
		}
	}
	c.Entries = v.Entries
	return nil
}

// CacheEntry returns the daemon's current tuning outcome, or false if no
// candidate is serving yet.
func (d *Daemon) CacheEntry() (CacheEntry, bool) {
	d.mu.Lock()
	defer d.mu.Unlock()
	m := d.measured[d.serving]
	if m == nil {
		return CacheEntry{}, false
	}
	return CacheEntry{
		Fingerprint: d.cfg.Twin.Fingerprint(),
		ShapeKey:    d.cfg.ShapeKey,
		Weights:     d.weights,
		Winner:      m.variant.Candidate,
		WinnerUS:    m.obs.MedianUS,
	}, true
}

// Warm seeds the daemon with a cached outcome for its twin and shape: the
// weights are restored and the cached winner is sampled right after the
// analytical baseline. Warm returns an error if e belongs to another device or
// workload.
func (d *Daemon) Warm(e CacheEntry) error {
	if fp := d.cfg.Twin.Fingerprint(); e.Fingerprint != fp || e.ShapeKey != d.cfg.ShapeKey {
		return fmt.Errorf("warm daemon: entry %s/%s does not match %s/%s", e.Fingerprint, e.ShapeKey, fp, d.cfg.ShapeKey)
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	d.weights = e.Weights
	d.warm = e.Winner.Name
	return nil
}
