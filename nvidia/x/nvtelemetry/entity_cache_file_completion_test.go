package nvtelemetry

import (
	"os"
	"testing"
)

func TestEntityCacheCompletionPersistence(t *testing.T) {
	path := entityCacheTestPath(t)
	a := entityAssignment()
	a.Owner.JobID = "42"
	c := newEntityFileCache(t, 2, 10)
	if err := c.Put([]EntityAssignment{a}, 0); err != nil {
		t.Fatal(err)
	}
	if err := c.SaveFile(path); err != nil {
		t.Fatal(err)
	}
	stale, _ := os.ReadFile(path)
	if err := c.CompleteJob(a.Owner); err != nil {
		t.Fatal(err)
	}
	if len(c.Get(a, 1)) != 0 {
		t.Fatal("completed job remains cached")
	}
	if err := c.Put([]EntityAssignment{a}, 1); err == nil {
		t.Fatal("completed owner reinserted")
	}
	// Loading an older snapshot into this live cache cannot undo completion.
	if discarded, err := c.LoadFile(path, entityCacheTestScope(a), 1); err != nil || discarded || len(c.Get(a, 1)) != 0 {
		t.Fatal("old file revived completed job", discarded, err)
	}
	if err := c.SaveFile(path); err != nil {
		t.Fatal(err)
	}
	saved := cacheFileData(t, path)
	if len(saved.Entries) != 0 || len(saved.Completed) != 1 || saved.Completed[0] != a.Owner {
		t.Fatal("completion marker not persisted", saved)
	}
	loaded := newEntityFileCache(t, 2, 10)
	if discarded, err := loaded.LoadFile(path, entityCacheTestScope(a), 1); err != nil || discarded {
		t.Fatal(discarded, err)
	}
	if err := loaded.Put([]EntityAssignment{a}, 1); err == nil {
		t.Fatal("restart lost completion marker")
	}
	// A different allocation attempt is a distinct typed owner.
	b := a
	b.Owner.Attempt = "next-attempt"
	b.Process.Process.Generation++
	if err := loaded.Put([]EntityAssignment{b}, 1); err != nil {
		t.Fatal(err)
	}
	if len(loaded.Get(b, 1)) != 1 {
		t.Fatal("unrelated allocation was evicted")
	}
	loaded.Clear()
	if err := loaded.Put([]EntityAssignment{a}, 1); err != nil {
		t.Fatal("explicit reset retained completion", err)
	}
	// Fresh processes cannot detect malicious rollback; checksum is not authority.
	// The public contract therefore requires exclusive file ownership, not trust
	// in a stale snapshot supplied by another party.
	if len(stale) == 0 {
		t.Fatal("missing historical snapshot fixture")
	}
}

func TestEntityCacheCompletionBounds(t *testing.T) {
	a := entityAssignment()
	a.Owner.JobID = "42"
	c := newEntityFileCache(t, 1, 10)
	if err := c.Put([]EntityAssignment{a}, 0); err != nil {
		t.Fatal(err)
	}
	if err := c.CompleteJob(EntityOwner{}); err == nil || len(c.Get(a, 1)) != 1 {
		t.Fatal("empty completion mutated cache")
	}
	if err := c.CompleteJob(a.Owner); err != nil {
		t.Fatal(err)
	}
	if err := c.CompleteJob(a.Owner); err != nil {
		t.Fatal("idempotent completion refused", err)
	}
	other := a.Owner
	other.JobID = "other-job"
	if err := c.CompleteJob(other); err == nil {
		t.Fatal("unbounded completion markers")
	}
	if err := c.Put([]EntityAssignment{a}, 1); err == nil {
		t.Fatal("failed admission lost existing completion")
	}
	if err := (*EntityCache)(nil).CompleteJob(a.Owner); err == nil {
		t.Fatal("nil cache accepted")
	}
	if err := (&EntityCache{}).CompleteJob(a.Owner); err == nil {
		t.Fatal("zero cache accepted")
	}
}

func TestEntityCacheCompletionContradictions(t *testing.T) {
	a := entityAssignment()
	a.Owner.JobID = "42"
	b := a
	b.Owner.JobID = "other-job"
	b.State = "contradictory"
	a.State = "contradictory"
	c := newEntityFileCache(t, 2, 10)
	if err := c.Put([]EntityAssignment{a, b}, 0); err != nil {
		t.Fatal(err)
	}
	if err := c.CompleteJob(a.Owner); err != nil {
		t.Fatal(err)
	}
	if len(c.Get(b, 1)) != 0 {
		t.Fatal("partial contradiction promoted after completion")
	}
}
