package cudahotswap

// State returns the generation and settings retained when the pin was acquired.
// The snapshot remains available after Release; it does not imply the function
// is still retained. A nil pin returns the zero State.
func (p *Pin) State() State {
	if p == nil {
		return State{}
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.state
}

// State returns the generation and settings retained for the submission attempt.
// It does not establish successful enqueue or completion. The snapshot remains
// available after Wait. A nil pending launch returns the zero State.
func (p *PendingLaunch) State() State {
	if p == nil {
		return State{}
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.state
}

// States returns a detached, name-sorted snapshot of generations and settings
// retained for a graph submission attempt. It does not establish enqueue or
// completion. A nil launch, or a request that retained no graph work, returns nil.
// The snapshot remains available after Wait.
func (p *GraphLaunch) States() []State {
	if p == nil {
		return nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]State(nil), p.states...)
}
