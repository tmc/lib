package nvlshm

import "sync"

// localBoot is an in-process Bootstrap for tests: world goroutines
// coordinate through shared state.
type localBoot struct {
	rank int
	st   *localState
}

type localState struct {
	world int

	mu   sync.Mutex
	cond *sync.Cond
	gen  int
	n    int

	slot string
}

func newLocalBoots(world int) []*localBoot {
	st := &localState{world: world}
	st.cond = sync.NewCond(&st.mu)
	boots := make([]*localBoot, world)
	for i := range boots {
		boots[i] = &localBoot{rank: i, st: st}
	}
	return boots
}

func (b *localBoot) Rank() int      { return b.rank }
func (b *localBoot) WorldSize() int { return b.st.world }

func (b *localBoot) Barrier() error {
	st := b.st
	st.mu.Lock()
	defer st.mu.Unlock()
	gen := st.gen
	st.n++
	if st.n == st.world {
		st.n = 0
		st.gen++
		st.cond.Broadcast()
		return nil
	}
	for st.gen == gen {
		st.cond.Wait()
	}
	return nil
}

func (b *localBoot) BroadcastString(s string, root int) (string, error) {
	if b.rank == root {
		b.st.mu.Lock()
		b.st.slot = s
		b.st.mu.Unlock()
	}
	if err := b.Barrier(); err != nil {
		return "", err
	}
	b.st.mu.Lock()
	v := b.st.slot
	b.st.mu.Unlock()
	if err := b.Barrier(); err != nil {
		return "", err
	}
	return v, nil
}
