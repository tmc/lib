package bpfprobe

import "errors"

// Group shares one ordered kernel ring across bounded entry/return and task
// attachments. Tags identify each configured observation. Attachments are
// installed sequentially; callers must hold their workload until setup returns
// to obtain a complete start boundary. Configuration changes require a new group.
type Group struct {
	*Probe
	probes []*Probe
}

// Disable stops all attachments while keeping the ring and final counters
// readable. It must precede a final drain and completeness receipt.
func (g *Group) Disable() error {
	if g == nil {
		return nil
	}
	var errs []error
	for _, p := range g.probes {
		errs = append(errs, p.Disable())
	}
	return errors.Join(errs...)
}

// Close closes all perf attachments before closing their shared ring and maps.
func (g *Group) Close() error {
	if g == nil {
		return nil
	}
	var errs []error
	for _, p := range g.probes {
		errs = append(errs, p.Close())
	}
	errs = append(errs, g.Probe.Close())
	return errors.Join(errs...)
}

// Disable stops capture while retaining the event ring and counters.
func (p *Probe) Disable() error {
	if p == nil {
		return nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return errors.New("bpf probe: closed")
	}
	if p.perf == nil {
		return nil
	}
	return disablePerf(p.perf)
}
