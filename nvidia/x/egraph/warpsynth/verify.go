package warpsynth

import (
	"fmt"
	"strings"
)

// A ProtocolErrorKind classifies a ring-protocol failure.
type ProtocolErrorKind int

const (
	Deadlock    ProtocolErrorKind = iota // no role can step before every tile finishes
	Race                                 // a stage is written before its previous tile is released
	StaleRead                            // a stage is read while it holds another tile
	ParityAlias                          // a wait passed on parity while the phase count differs
)

func (k ProtocolErrorKind) String() string {
	return [...]string{"deadlock", "race", "stale read", "parity alias"}[k]
}

// A ProtocolError reports the first failure found and the scheduler or
// search that found it.
type ProtocolError struct {
	Kind   ProtocolErrorKind
	Search string // "exhaustive", "producer-first", "consumer-first" or "round-robin"
	Role   WarpRoleKind
	Tile   int
	Ring   int
	Stage  int
	Detail string
}

func (e *ProtocolError) Error() string {
	return fmt.Sprintf("%s (%s search): role %s tile %d ring %d stage %d: %s", e.Kind, e.Search, e.Role, e.Tile, e.Ring, e.Stage, e.Detail)
}

// protocol holds the knobs the emitters and the verifier share. The zero
// value is the correct protocol; the fields exist so tests can inject the
// mutations the verifier must catch.
type protocol struct {
	parityByIter  bool // wait on k&1 rather than (k/D)&1
	skipEmptyWait bool // the producer does not wait for a stage's release
}

// ring is one producer→consumer ring of the protocol.
type ring struct {
	depth    int
	releaser int // role index whose step releases a stage
}

func rings(s Schedule) []ring {
	var out []ring
	for i, d := range s.RingDepths {
		rel := i + 1
		if i == 0 && !s.UseEarlySMEMRelease && len(s.Roles) > 2 {
			rel = 2 // the epilogue releases the stage after it finishes the tile
		}
		out = append(out, ring{depth: d, releaser: rel})
	}
	return out
}

func (p protocol) fullParity(k, d int) int {
	if p.parityByIter {
		return k & 1
	}
	return k / d & 1
}

func (p protocol) emptyParity(k, d int) int { return p.fullParity(k, d) ^ 1 }

// state is the explicit protocol state: each role's next tile, the
// completed-phase count of every full and empty barrier, and which tile each
// stage holds (-1 when released).
type state struct {
	next  []int
	full  [][]int
	empty [][]int
	slot  [][]int
}

func newState(s Schedule) *state {
	st := &state{next: make([]int, len(s.Roles))}
	for _, d := range s.RingDepths {
		st.full = append(st.full, make([]int, d))
		st.empty = append(st.empty, make([]int, d))
		sl := make([]int, d)
		for i := range sl {
			sl[i] = -1
		}
		st.slot = append(st.slot, sl)
	}
	return st
}

func (st *state) clone() *state {
	c := &state{next: append([]int(nil), st.next...)}
	for i := range st.full {
		c.full = append(c.full, append([]int(nil), st.full[i]...))
		c.empty = append(c.empty, append([]int(nil), st.empty[i]...))
		c.slot = append(c.slot, append([]int(nil), st.slot[i]...))
	}
	return c
}

func (st *state) key() string {
	var b strings.Builder
	fmt.Fprint(&b, st.next, st.full, st.empty, st.slot)
	return b.String()
}

type model struct {
	s     Schedule
	p     protocol
	rings []ring
	tiles int
}

// mbarrier.try_wait.parity P succeeds when the barrier's current phase has
// parity other than P, that is, when the phase with parity P has completed.
func passes(count, parity int) bool { return count&1 != parity }

// enabled reports whether role r can take its next step. A step whose waits
// pass on parity but not on count is enabled, and its execution reports the
// alias.
func (m *model) enabled(st *state, r int) bool {
	k := st.next[r]
	if k >= m.tiles {
		return false
	}
	if r > 0 {
		d := m.rings[r-1].depth
		if !passes(st.full[r-1][k%d], m.p.fullParity(k, d)) {
			return false
		}
	}
	if r < len(m.rings) && !(r == 0 && m.p.skipEmptyWait) {
		d := m.rings[r].depth
		if !passes(st.empty[r][k%d], m.p.emptyParity(k, d)) {
			return false
		}
	}
	return true
}

// step executes role r's next tile on st.
func (m *model) step(st *state, r int, search string) *ProtocolError {
	k := st.next[r]
	fail := func(kind ProtocolErrorKind, ring, stage int, format string, args ...any) *ProtocolError {
		return &ProtocolError{Kind: kind, Search: search, Role: m.s.Roles[r].Kind, Tile: k, Ring: ring, Stage: stage, Detail: fmt.Sprintf(format, args...)}
	}
	if r > 0 {
		i := r - 1
		d := m.rings[i].depth
		s := k % d
		if want := k/d + 1; st.full[i][s] != want {
			return fail(ParityAlias, i, s, "full barrier has completed %d phases, the wait assumed %d", st.full[i][s], want)
		}
		if st.slot[i][s] != k {
			return fail(StaleRead, i, s, "stage holds tile %d", st.slot[i][s])
		}
	}
	if r < len(m.rings) {
		d := m.rings[r].depth
		s := k % d
		if !(r == 0 && m.p.skipEmptyWait) {
			if want := k / d; st.empty[r][s] != want {
				return fail(ParityAlias, r, s, "empty barrier has completed %d phases, the wait assumed %d", st.empty[r][s], want)
			}
		}
		if st.slot[r][s] != -1 {
			return fail(Race, r, s, "stage still holds unreleased tile %d", st.slot[r][s])
		}
		st.slot[r][s] = k
		st.full[r][s]++
	}
	for i, rg := range m.rings {
		if rg.releaser != r {
			continue
		}
		s := k % rg.depth
		st.slot[i][s] = -1
		st.empty[i][s]++
	}
	st.next[r]++
	return nil
}

func (m *model) done(st *state) bool {
	for _, k := range st.next {
		if k < m.tiles {
			return false
		}
	}
	return true
}

func (m *model) deadlock(st *state, search string) *ProtocolError {
	r := 0
	for i, k := range st.next {
		if k < st.next[r] {
			r = i
		}
	}
	return &ProtocolError{Kind: Deadlock, Search: search, Role: m.s.Roles[r].Kind, Tile: st.next[r], Detail: fmt.Sprintf("roles stopped at tiles %v", st.next)}
}

// maxStates bounds the exhaustive search. A correct ring protocol's
// reachable states are bounded by tiles·Π depths, far below it.
const maxStates = 1 << 21

func (m *model) exhaustive() error {
	start := newState(m.s)
	seen := map[string]bool{start.key(): true}
	queue := []*state{start}
	for len(queue) > 0 {
		st := queue[0]
		queue = queue[1:]
		moved := false
		for r := range st.next {
			if !m.enabled(st, r) {
				continue
			}
			moved = true
			nx := st.clone()
			if err := m.step(nx, r, "exhaustive"); err != nil {
				return err
			}
			if k := nx.key(); !seen[k] {
				if len(seen) >= maxStates {
					return fmt.Errorf("exhaustive search exceeded %d states", maxStates)
				}
				seen[k] = true
				queue = append(queue, nx)
			}
		}
		if !moved && !m.done(st) {
			return m.deadlock(st, "exhaustive")
		}
	}
	return nil
}

func (m *model) scheduled(name string, pick func(st *state, turn int) int) error {
	st := newState(m.s)
	for turn := 0; !m.done(st); turn++ {
		r := pick(st, turn)
		if r < 0 {
			return m.deadlock(st, name)
		}
		if err := m.step(st, r, name); err != nil {
			return err
		}
	}
	return nil
}

// VerifyDeadlockFreedom checks the ring protocol s implies over numTiles
// tiles, as the package documentation describes, and returns a
// [*ProtocolError] for the first failure.
func VerifyDeadlockFreedom(s Schedule, numTiles int) error {
	return verify(s, numTiles, protocol{})
}

func verify(s Schedule, numTiles int, p protocol) error {
	if err := s.Check(); err != nil {
		return fmt.Errorf("verify: %w", err)
	}
	if numTiles <= 0 {
		return fmt.Errorf("verify: %d tiles", numTiles)
	}
	m := &model{s: s, p: p, rings: rings(s), tiles: numTiles}
	n := len(s.Roles)
	first := func(order func(i int) int) func(*state, int) int {
		return func(st *state, _ int) int {
			for i := range n {
				if r := order(i); m.enabled(st, r) {
					return r
				}
			}
			return -1
		}
	}
	runs := []struct {
		name string
		pick func(*state, int) int
	}{
		{"producer-first", first(func(i int) int { return i })},
		{"consumer-first", first(func(i int) int { return n - 1 - i })},
		{"round-robin", func(st *state, turn int) int {
			for i := range n {
				if r := (turn + i) % n; m.enabled(st, r) {
					return r
				}
			}
			return -1
		}},
	}
	for _, run := range runs {
		if err := m.scheduled(run.name, run.pick); err != nil {
			return err
		}
	}
	return m.exhaustive()
}
