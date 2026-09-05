package cuptiactivity

import (
	"errors"
	"fmt"
	"sync"

	"github.com/tmc/lib/nvidia/cupti"
)

// activityOwner serializes teardown without holding callback locks. A failed
// disable or flush leaves ownership intact so the caller can retry. In
// particular, buffers must never be unpinned merely because Close was called.
type activityOwner struct {
	mu      sync.Mutex
	kinds   []cupti.ActivityKind
	pool    *bufferPool
	disable func(cupti.ActivityKind) error
	flush   func() error
	release func()
	closed  bool
	quiesce func() error
}

// startActivities takes ownership before enabling kinds. If rollback fails,
// the returned owner must be closed again; dropping it would orphan callbacks
// and their pinned buffers. Validation precedes every native operation.
func startActivities(kinds []cupti.ActivityKind, layouts map[cupti.ActivityKind]uint, pool *bufferPool,
	enable, disable func(cupti.ActivityKind) error, flush func() error, release func(), quiesce ...func() error) (*activityOwner, error) {
	if pool == nil || enable == nil || disable == nil || flush == nil {
		return nil, errors.New("cuptiactivity: missing activity ownership dependency")
	}
	if err := validateKinds(kinds, layouts); err != nil {
		return nil, err
	}
	o := &activityOwner{pool: pool, disable: disable, flush: flush, release: release}
	if len(quiesce) != 0 {
		o.quiesce = quiesce[0]
	}
	for _, kind := range kinds {
		if err := enable(kind); err != nil {
			startErr := fmt.Errorf("cuptiactivity: enable %s: %w", kind, err)
			if err := o.close(); err != nil {
				return o, errors.Join(startErr, err)
			}
			return nil, startErr
		}
		o.kinds = append(o.kinds, kind)
	}
	return o, nil
}

func validateKinds(kinds []cupti.ActivityKind, layouts map[cupti.ActivityKind]uint) error {
	if len(kinds) == 0 {
		return errors.New("cuptiactivity: no activity kinds requested")
	}
	seen := make(map[cupti.ActivityKind]bool)
	for _, kind := range kinds {
		if layouts[kind] == 0 {
			return fmt.Errorf("cuptiactivity: unsupported activity kind %s", kind)
		}
		if seen[kind] {
			return fmt.Errorf("cuptiactivity: duplicate activity kind %s", kind)
		}
		seen[kind] = true
	}
	return nil
}

func (o *activityOwner) close() error {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.closed {
		return nil
	}
	var errs []error
	remaining := o.kinds[:0]
	for _, kind := range o.kinds {
		if err := o.disable(kind); err != nil {
			remaining = append(remaining, kind)
			errs = append(errs, fmt.Errorf("cuptiactivity: disable %s: %w", kind, err))
		}
	}
	o.kinds = remaining
	// Flush even after a disable failure to bound outstanding buffers. Flush
	// does not synchronize application GPU work or prove timestamps complete.
	if err := o.flush(); err != nil {
		errs = append(errs, fmt.Errorf("cuptiactivity: flush: %w", err))
	}
	if pending, _ := o.pool.status(); pending != 0 {
		errs = append(errs, fmt.Errorf("cuptiactivity: %d activity buffers still owned", pending))
	}
	if err := errors.Join(errs...); err != nil {
		return err
	}
	if o.quiesce != nil {
		if err := o.quiesce(); err != nil {
			return err
		}
	}
	o.closed = true
	if o.release != nil {
		o.release()
	}
	return nil
}
