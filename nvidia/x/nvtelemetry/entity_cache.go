package nvtelemetry

import (
	"fmt"
	"sync"
)

// EntityCache is a bounded relationship cache keyed by boot, process birth,
// namespace and exec generation. The zero value is not usable; use NewEntityCache.
// Entries expire at the lesser of their interval end and a caller supplied TTL.
// It neither discovers nor authenticates assignments.
type EntityCache struct {
	mu        sync.Mutex
	limit     int
	ttl       int64
	entries   map[entityKey]entityCacheEntry
	completed map[EntityOwner]bool
}
type entityKey struct {
	boot, node, namespace, domain, epoch, device, migEpoch string
	pid                                                    uint32
	birth, exec                                            uint64
}
type entityCacheEntry struct {
	assignments []EntityAssignment
	expires     int64
	inserted    int64
}

// NewEntityCache creates a cache with a positive capacity and TTL in nanoseconds.
func NewEntityCache(capacity int, ttlNanos int64) (*EntityCache, error) {
	if capacity <= 0 || capacity > entityMaxRecords || ttlNanos <= 0 {
		return nil, fmt.Errorf("entity cache: invalid capacity or ttl")
	}
	return &EntityCache{limit: capacity, ttl: ttlNanos, entries: make(map[entityKey]entityCacheEntry), completed: make(map[EntityOwner]bool)}, nil
}
func assignmentKey(a EntityAssignment) (entityKey, error) {
	p := a.Process
	if p.BootID == "" || p.NodeID == "" || p.PIDNamespace == "" || p.Process.PID == 0 || p.Process.Generation == 0 || !p.HasExecGeneration || a.Device.UUID == "" {
		return entityKey{}, fmt.Errorf("entity cache: incomplete process or device generation")
	}
	return entityKey{p.BootID, p.NodeID, p.PIDNamespace, a.ClockDomain, a.ClockEpoch, a.Device.UUID, a.MIGConfigurationEpoch, p.Process.PID, p.Process.Generation, p.ExecGeneration}, nil
}

// Put replaces relationships for a single exact cache key at now in the
// assignment clock. Contradictions and unknown state are retained, not promoted.
// Reusing a PID without the same birth and exec generation cannot hit this entry.
func (c *EntityCache) Put(assignments []EntityAssignment, now int64) error {
	if c == nil {
		return fmt.Errorf("entity cache: nil cache")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.entries == nil {
		return fmt.Errorf("entity cache: uninitialized cache")
	}
	if len(assignments) == 0 || len(assignments) > entityMaxRecords {
		return fmt.Errorf("entity cache: invalid assignment count")
	}
	var key entityKey
	expires := int64(^uint64(0) >> 1)
	copy := make([]EntityAssignment, len(assignments))
	for i, a := range assignments {
		if err := a.Validate(); err != nil {
			return err
		}
		if c.completed[a.Owner] {
			return fmt.Errorf("entity cache: job completed")
		}
		k, err := assignmentKey(a)
		if err != nil {
			return err
		}
		if i == 0 {
			key = k
		} else if k != key {
			return fmt.Errorf("entity cache: mixed generation keys")
		}
		if now < a.StartNanos || now >= a.EndNanos {
			return fmt.Errorf("entity cache: assignment not current")
		}
		if a.EndNanos < expires {
			expires = a.EndNanos
		}
		copy[i] = cloneAssignment(a)
	}
	if now <= int64(^uint64(0)>>1)-c.ttl && now+c.ttl < expires {
		expires = now + c.ttl
	}
	for k, e := range c.entries {
		if k.domain == key.domain && k.epoch == key.epoch && now >= e.expires {
			delete(c.entries, k)
		}
	}
	if _, ok := c.entries[key]; !ok && len(c.entries) >= c.limit {
		return fmt.Errorf("entity cache: capacity exceeded")
	}
	c.entries[key] = entityCacheEntry{assignments: copy, expires: expires, inserted: now}
	return nil
}

// Get returns a cloned relationship set for the exact key of query. Misses,
// incomplete generations and expired entries yield no relationships.
func (c *EntityCache) Get(query EntityAssignment, now int64) []EntityAssignment {
	if c == nil {
		return nil
	}
	key, err := assignmentKey(query)
	if err != nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.entries[key]
	if !ok {
		return nil
	}
	if now < e.inserted {
		return nil
	}
	if now >= e.expires {
		delete(c.entries, key)
		return nil
	}
	out := make([]EntityAssignment, len(e.assignments))
	for i, a := range e.assignments {
		if now < a.StartNanos {
			return nil
		}
		out[i] = cloneAssignment(a)
	}
	return out
}

// Clear removes all relationships and job completion markers, for example after
// a metadata authority restart.
func (c *EntityCache) Clear() {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	clear(c.entries)
	clear(c.completed)
}

// CompleteJob removes every relationship set containing the exact owner and
// prevents that owner from being reinserted until Clear. JobID must be present;
// callers must supply the original stable allocation and attempt fields too.
// Completion markers are bounded by the cache capacity and persisted by SaveFile.
func (c *EntityCache) CompleteJob(owner EntityOwner) error {
	if c == nil || owner.JobID == "" {
		return fmt.Errorf("entity cache: missing cache or job identity")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.entries == nil || c.completed == nil {
		return fmt.Errorf("entity cache: uninitialized cache")
	}
	if !c.completed[owner] && len(c.completed) >= c.limit {
		return fmt.Errorf("entity cache: completion capacity exceeded")
	}
	c.completed[owner] = true
	for key, entry := range c.entries {
		for _, a := range entry.assignments {
			if a.Owner == owner {
				delete(c.entries, key)
				break
			}
		}
	}
	return nil
}
