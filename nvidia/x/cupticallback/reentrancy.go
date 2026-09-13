package cupticallback

import "sync"

type reentrancyTracker struct {
	mu     sync.Mutex
	active map[uint64]int
}

func (r *reentrancyTracker) enter() (depth int, tid uint64) {
	tid = currentThreadID()
	r.mu.Lock()
	r.active[tid]++
	d := r.active[tid]
	r.mu.Unlock()
	return d, tid
}

func (r *reentrancyTracker) exit(tid uint64) {
	r.mu.Lock()
	r.active[tid]--
	if r.active[tid] <= 0 {
		delete(r.active, tid)
	}
	r.mu.Unlock()
}
