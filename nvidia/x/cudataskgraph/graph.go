package cudataskgraph

import (
	"crypto/sha256"
	"fmt"
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
	ID    string
	Deps  []string
	Rank  int
	Bytes uint64
	Scope Scope
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
		r.tasks[id] = task
	}
	return r
}

func (g Graph) Tasks() []Task {
	r := make([]Task, 0, len(g.order))
	for _, id := range g.order {
		t := g.tasks[id]
		t.Deps = append([]string(nil), t.Deps...)
		r = append(r, t)
	}
	return r
}

// Identity returns a stable identity for the validated graph topology.
func (g Graph) Identity() string {
	h := sha256.New()
	for _, t := range g.Tasks() {
		fmt.Fprintf(h, "%s\x00%d\x00%d\x00%d\x00", t.ID, t.Rank, t.Bytes, t.Scope)
		for _, d := range t.Deps {
			fmt.Fprintf(h, "%s\x00", d)
		}
	}
	return fmt.Sprintf("%x", h.Sum(nil))
}

type Epoch struct {
	mu        sync.Mutex
	Number    uint64
	published map[string]bool
	aborted   bool
}

func NewEpoch(n uint64) *Epoch { return &Epoch{Number: n, published: map[string]bool{}} }
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
