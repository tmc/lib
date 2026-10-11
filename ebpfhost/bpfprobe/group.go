package bpfprobe

import (
	"errors"
	"fmt"
	"sync"
)

// Group shares one ordered kernel ring across bounded entry/return and task
// attachments. Tags identify each configured observation. Attachments are
// installed sequentially; callers must hold their workload until setup returns
// to obtain a complete start boundary. Add installs new attachments without
// replacing shared maps; callers must retain explicit coverage boundaries.
type Group struct {
	*Probe
	probes              []*Probe
	groupMu             sync.Mutex
	capacity            int
	disabled, closed    bool
	updateErr, closeErr error
	attach              func(Config, *Probe) error
}

// Disable stops all attachments while keeping the ring and final counters
// readable. It must precede a final drain and completeness receipt.
func (g *Group) Disable() error {
	if g == nil {
		return nil
	}
	g.groupMu.Lock()
	defer g.groupMu.Unlock()
	if g.closed {
		return errors.New("bpf probe: group closed")
	}
	g.disabled = true
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
	g.groupMu.Lock()
	defer g.groupMu.Unlock()
	if g.closed {
		return g.closeErr
	}
	g.closed = true
	var errs []error
	for _, p := range g.probes {
		errs = append(errs, p.Close())
	}
	errs = append(errs, g.Probe.Close())
	g.closeErr = errors.Join(errs...)
	return g.closeErr
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

// Add installs additional task attachments into the same ordered event ring.
// At most 256 attachments may be installed over a group's lifetime. Existing
// attachments continue observing during setup; new attachments start sequentially.
// Capacity must match the original group. Stack/output acquisition requires maps
// provisioned by the original configurations; Add never creates replacement maps.
//
// verify must validate process/task birth identities before and after attachment.
// It runs under the group mutation lock and must not call group methods. A nil
// verifier is refused. Verification failure is terminal for future additions.
//
// On an attachment or verification failure, Add closes only new attachments
// and retains the original group. Events submitted before rollback may remain
// in the shared ring.
// Further additions are refused after failure, Disable or Close. The owner must
// stop, drain and record incomplete coverage; Add does not retry kernel refusals.
func (g *Group) Add(configs []Config, verify func() error) error {
	if g == nil {
		return errors.New("bpf probe: nil group")
	}
	g.groupMu.Lock()
	defer g.groupMu.Unlock()
	if g.closed {
		return errors.New("bpf probe: group closed")
	}
	if g.disabled {
		return errors.New("bpf probe: group disabled")
	}
	if g.updateErr != nil {
		return fmt.Errorf("bpf probe: previous addition failed: %w", g.updateErr)
	}
	if g.Probe == nil || g.capacity <= 0 {
		return errors.New("bpf probe: uninitialized group")
	}
	if len(configs) == 0 || len(configs) > 256-len(g.probes) {
		return errors.New("bpf probe: lifetime attachment count outside 1..256")
	}
	if verify == nil {
		return errors.New("bpf probe: task verification required")
	}
	validated := make([]Config, len(configs))
	for i, c := range configs {
		var err error
		c, err = validateConfig(c)
		if err != nil {
			return err
		}
		if c.Capacity != g.capacity {
			return errors.New("bpf probe: inconsistent group ring capacity")
		}
		if c.UserStack && g.stacks == nil {
			return errors.New("bpf probe: group has no stack map")
		}
		if c.OutputPointer && len(g.extras) != 2 {
			return errors.New("bpf probe: group has no output maps")
		}
		if (c.generationOffset != 0 && c.generationOffset != g.TaskStartOffset) || (c.execOffset != 0 && c.execOffset != g.TaskExecOffset) || (c.execSize != 0 && c.execSize != g.TaskExecSize) {
			return errors.New("bpf probe: inconsistent group task layout")
		}
		c.generationOffset, c.execOffset, c.execSize = g.TaskStartOffset, g.TaskExecOffset, g.TaskExecSize
		validated[i] = c
	}
	if err := verify(); err != nil {
		g.updateErr = fmt.Errorf("bpf probe: pre-attachment verification: %w", err)
		return g.updateErr
	}
	added := make([]*Probe, 0, len(validated))
	rollback := func(err error) error {
		errs := []error{err}
		for _, p := range added {
			errs = append(errs, p.Close())
		}
		g.updateErr = errors.Join(errs...)
		return g.updateErr
	}
	attach := g.attach
	if attach == nil {
		attach = func(c Config, p *Probe) error { return attachGroupMember(g, c, p) }
	}
	for _, c := range validated {
		p := &Probe{}
		added = append(added, p)
		if err := attach(c, p); err != nil {
			return rollback(err)
		}
	}
	if err := verify(); err != nil {
		return rollback(fmt.Errorf("bpf probe: post-attachment verification: %w", err))
	}
	g.probes = append(g.probes, added...)
	return nil
}
