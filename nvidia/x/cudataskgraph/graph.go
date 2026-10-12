package cudataskgraph

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"hash"
	"slices"
	"sort"
	"sync"
)

type Scope uint8

const (
	ScopeCTA Scope = iota + 1
	ScopeCluster
	ScopeDevice
	ScopeSystem
)

type Task struct {
	ID   string
	Deps []string

	// Inputs names the dependencies whose outputs the task reads, in operand
	// order. Each must also appear in Deps, at most once; a dependency not
	// named here orders the task without supplying it data.
	//
	// An empty Inputs declares nothing. Consumers that predate this field
	// read such a task's first dependency, if any, as its only data input.
	Inputs []string

	Rank int

	// Bytes is the task's dynamic shared-memory claim, in bytes. It is not
	// the size of the task's output and not a cost: a scheduler that reads
	// it as either will rank tasks by a quantity that says nothing about
	// them. The megakernel executor budgets it against the shared memory the
	// compiled worker admits (cudamega.Progress); apart from that it is only
	// hashed into the graph fingerprint.
	Bytes uint64

	Scope Scope

	// Extent is the number of elements the task produces. Zero means the
	// extent is undeclared, not that the task is empty. New records Extent
	// in the identity but checks it against nothing; whoever runs the task
	// decides whether the elements it is given match.
	Extent uint64
}
type Graph struct {
	tasks map[string]Task
	order []string
}

func New(tasks []Task) (Graph, error) {
	g := Graph{tasks: make(map[string]Task)}
	for _, t := range tasks {
		if t.ID == "" || t.Rank < 0 || t.Scope < ScopeCTA || t.Scope > ScopeSystem {
			return Graph{}, fmt.Errorf("invalid task %q", t.ID)
		}
		if _, ok := g.tasks[t.ID]; ok {
			return Graph{}, fmt.Errorf("duplicate task %q", t.ID)
		}
		t.Deps = append([]string(nil), t.Deps...)
		t.Inputs = append([]string(nil), t.Inputs...)
		g.tasks[t.ID] = t
	}
	for _, t := range tasks {
		for _, d := range t.Deps {
			if _, ok := g.tasks[d]; !ok {
				return Graph{}, fmt.Errorf("task %q has unknown dependency %q", t.ID, d)
			}
			if g.tasks[d].Rank >= t.Rank {
				return Graph{}, fmt.Errorf("dependency rank does not precede task %q", t.ID)
			}
		}
		for i, in := range t.Inputs {
			if !slices.Contains(t.Deps, in) {
				return Graph{}, fmt.Errorf("task %q has input %q that is not a dependency", t.ID, in)
			}
			if slices.Contains(t.Inputs[:i], in) {
				return Graph{}, fmt.Errorf("task %q names input %q twice", t.ID, in)
			}
		}
	}
	vis := map[string]uint8{}
	var visit func(string) error
	visit = func(id string) error {
		if vis[id] == 1 {
			return fmt.Errorf("cycle at %q", id)
		}
		if vis[id] == 2 {
			return nil
		}
		vis[id] = 1
		for _, d := range g.tasks[id].Deps {
			if err := visit(d); err != nil {
				return err
			}
		}
		vis[id] = 2
		g.order = append(g.order, id)
		return nil
	}
	for id := range g.tasks {
		if err := visit(id); err != nil {
			return Graph{}, err
		}
	}
	sort.Slice(g.order, func(i, j int) bool {
		a, b := g.tasks[g.order[i]], g.tasks[g.order[j]]
		if a.Rank != b.Rank {
			return a.Rank < b.Rank
		}
		return a.ID < b.ID
	})
	return g, nil
}

// Clone returns an independent graph snapshot, including dependency slices.
func (g Graph) Clone() Graph {
	r := Graph{tasks: make(map[string]Task, len(g.tasks)), order: append([]string(nil), g.order...)}
	for id, task := range g.tasks {
		task.Deps = append([]string(nil), task.Deps...)
		task.Inputs = append([]string(nil), task.Inputs...)
		r.tasks[id] = task
	}
	return r
}

func (g Graph) Tasks() []Task {
	r := make([]Task, 0, len(g.order))
	for _, id := range g.order {
		t := g.tasks[id]
		t.Deps = append([]string(nil), t.Deps...)
		t.Inputs = append([]string(nil), t.Inputs...)
		r = append(r, t)
	}
	return r
}

// Identity returns a stable identity for the validated graph: every task's
// ID, rank, shared-memory bytes, scope, extent, dependencies and data inputs,
// in task order.
//
// Each string and list is written with its length before it, so no
// arrangement of bytes inside an ID can be read as a field of another task.
func (g Graph) Identity() string {
	h := sha256.New()
	tasks := g.Tasks()
	writeUint(h, uint64(len(tasks)))
	for _, t := range tasks {
		writeString(h, t.ID)
		writeUint(h, uint64(t.Rank))
		writeUint(h, t.Bytes)
		writeUint(h, uint64(t.Scope))
		writeUint(h, t.Extent)
		writeStrings(h, t.Deps)
		writeStrings(h, t.Inputs)
	}
	return fmt.Sprintf("%x", h.Sum(nil))
}

func writeUint(h hash.Hash, v uint64) {
	h.Write(binary.BigEndian.AppendUint64(nil, v))
}

func writeString(h hash.Hash, s string) {
	writeUint(h, uint64(len(s)))
	h.Write([]byte(s))
}

func writeStrings(h hash.Hash, ss []string) {
	writeUint(h, uint64(len(ss)))
	for _, s := range ss {
		writeString(h, s)
	}
}

type Epoch struct {
	mu        sync.Mutex
	Number    uint64
	published map[string]bool
	owned     map[string]bool
	aborted   bool
	drained   bool
}

func NewEpoch(n uint64) *Epoch {
	return &Epoch{Number: n, published: map[string]bool{}, owned: map[string]bool{}}
}
func (e *Epoch) Publish(id string, g Graph) error {
	if e == nil {
		return fmt.Errorf("nil epoch")
	}
	return e.PublishEpoch(e.Number, id, g)
}

// PublishEpoch publishes a completion only for the epoch that owns it.
func (e *Epoch) PublishEpoch(epoch uint64, id string, g Graph) error {
	if e == nil {
		return fmt.Errorf("nil epoch")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.aborted {
		return fmt.Errorf("epoch is aborted")
	}
	if e.Number == 0 {
		return fmt.Errorf("epoch number is zero")
	}
	if epoch != e.Number {
		return fmt.Errorf("stale completion epoch %d", epoch)
	}
	t, ok := g.tasks[id]
	if !ok {
		return fmt.Errorf("unknown task %q", id)
	}
	for _, d := range t.Deps {
		if !e.published[d] {
			return fmt.Errorf("dependency %q is unpublished", d)
		}
	}
	if e.published == nil {
		e.published = make(map[string]bool)
	}
	if e.published[id] {
		return fmt.Errorf("task %q already published", id)
	}
	e.published[id] = true
	return nil
}
func (e *Epoch) Published(id string) bool {
	if e == nil {
		return false
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.published[id]
}
func (e *Epoch) Abort() {
	if e != nil {
		e.mu.Lock()
		defer e.mu.Unlock()
		e.aborted = true
	}
}
func (e *Epoch) Aborted() bool {
	if e == nil {
		return true
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.aborted
}

// Ownership of task output storage, and what an abort does to it.
//
// A task's output buffer is owned by the device from the moment the host
// claims it for a launch until the host has both observed the launch finish
// and released it. An abort does not end that ownership: the abort word is
// something the worker reads at its next poll, not a fence it has already
// passed, so a buffer freed on abort can be freed while a store is in flight.
// [Epoch.Release] therefore refuses until [Epoch.Drain] records that the host
// joined the launch, and [Epoch.Retained] names what may not be freed yet.

// Claim records that the device owns task id's output storage for this epoch.
func (e *Epoch) Claim(id string, g Graph) error {
	if e == nil {
		return fmt.Errorf("nil epoch")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if _, ok := g.tasks[id]; !ok {
		return fmt.Errorf("unknown task %q", id)
	}
	if e.owned == nil {
		e.owned = make(map[string]bool)
	}
	if e.owned[id] {
		return fmt.Errorf("task %q output storage is already claimed", id)
	}
	e.owned[id] = true
	return nil
}

// Drain records that the host has observed the launch finish, so nothing is
// still storing into the claimed buffers.
func (e *Epoch) Drain() {
	if e != nil {
		e.mu.Lock()
		defer e.mu.Unlock()
		e.drained = true
	}
}

// Drained reports whether the launch has been observed to finish.
func (e *Epoch) Drained() bool {
	if e == nil {
		return false
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.drained
}

// Release relinquishes ownership of task id's output storage.
//
// It refuses while an abort is outstanding and the launch has not been
// drained. That refusal is the point: an aborted launch is the case where
// freeing looks safest and is not.
func (e *Epoch) Release(id string) error {
	if e == nil {
		return fmt.Errorf("nil epoch")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if !e.owned[id] {
		return fmt.Errorf("task %q output storage is not claimed", id)
	}
	if e.aborted && !e.drained {
		return fmt.Errorf("task %q output storage may not be released: an abort is outstanding and the launch has not been drained", id)
	}
	delete(e.owned, id)
	return nil
}

// Retained returns the tasks whose output storage is still owned, sorted.
func (e *Epoch) Retained() []string {
	if e == nil {
		return nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	out := make([]string, 0, len(e.owned))
	for id := range e.owned {
		out = append(out, id)
	}
	sort.Strings(out)
	return out
}
