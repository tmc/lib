package rankgroup

import (
	"errors"
	"fmt"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagreen"
)

// A Mode is how ranks on one device share it.
type Mode uint8

const (
	Serial  Mode = iota // one stream, ranks in order
	Streams             // one stream per rank
	Green               // one green context and stream per rank
)

func (m Mode) String() string {
	switch m {
	case Serial:
		return "serial"
	case Streams:
		return "streams"
	case Green:
		return "green"
	}
	return fmt.Sprintf("Mode(%d)", uint8(m))
}

// ParseMode parses a [Mode] by name.
func ParseMode(s string) (Mode, error) {
	for _, m := range []Mode{Serial, Streams, Green} {
		if s == m.String() {
			return m, nil
		}
	}
	return 0, fmt.Errorf("rankgroup: unknown mode %q (want serial, streams or green)", s)
}

// A Group is the ranks of a tensor-parallel group, each with a device and
// a stream. Work for rank r is enqueued on Stream(r) with Launch.
//
// A Group's methods that touch the driver must run inside [Group.Do],
// which binds a device context to the calling OS thread. A Group is not
// safe for concurrent use.
type Group struct {
	mode    Mode
	devs    []*cuda.Device
	ctxs    []*cuda.Context // by rank; primary contexts
	streams []driver.Stream
	events  []driver.Event
	owned   []driver.Stream // streams to destroy
	green   *cudagreen.Partitions
	sms     []int // by rank
}

// MaxRanks is the largest group size.
const MaxRanks = 8

// OnDevice returns n ranks on dev, sharing it as mode says.
func OnDevice(dev *cuda.Device, n int, mode Mode) (_ *Group, err error) {
	if n < 1 || n > MaxRanks {
		return nil, fmt.Errorf("rankgroup: %d ranks outside 1..%d", n, MaxRanks)
	}
	g := &Group{mode: mode}
	for range n {
		g.devs = append(g.devs, dev)
	}
	defer func() {
		if err != nil {
			err = errors.Join(err, g.Close())
		}
	}()
	ctx, err := dev.PrimaryContext()
	if err != nil {
		return nil, err
	}
	for range n {
		g.ctxs = append(g.ctxs, ctx)
	}
	sms, err := dev.MultiprocessorCount()
	if err != nil {
		return nil, err
	}
	err = ctx.Run(func() error {
		switch mode {
		case Serial:
			s, err := driver.StreamCreateWithPriority(uint32(driver.StreamNonBlocking), 0)
			if err != nil {
				return fmt.Errorf("rankgroup: stream: %w", err)
			}
			g.owned = append(g.owned, s)
			for range n {
				g.streams = append(g.streams, s)
				g.sms = append(g.sms, sms)
			}
		case Streams:
			for range n {
				s, err := driver.StreamCreateWithPriority(uint32(driver.StreamNonBlocking), 0)
				if err != nil {
					return fmt.Errorf("rankgroup: stream: %w", err)
				}
				g.owned = append(g.owned, s)
				g.streams = append(g.streams, s)
				g.sms = append(g.sms, sms)
			}
		case Green:
			if err := g.openGreen(dev.Raw(), n); err != nil {
				return err
			}
		default:
			return fmt.Errorf("rankgroup: bad mode %v", mode)
		}
		return g.newEvents()
	})
	if err != nil {
		return nil, err
	}
	return g, nil
}

// openGreen splits dev's SMs into n equal green contexts.
func (g *Group) openGreen(dev driver.Device, n int) error {
	total, gran, err := cudagreen.DeviceGranularity(dev)
	if err != nil {
		return err
	}
	if gran, err = cudagreen.DeviceSplit(dev, gran); err != nil {
		return err
	}
	if gran.Groups < n {
		return fmt.Errorf("rankgroup: device splits into %d groups of %d SMs, need %d ranks", gran.Groups, gran.Granule(), n)
	}
	major, err := dev.Attribute(driver.DevAttrComputeCapabilityMajor)
	if err != nil {
		return err
	}
	minor, err := dev.Attribute(driver.DevAttrComputeCapabilityMinor)
	if err != nil {
		return err
	}
	// Equal shares of the groups the driver makes.
	per := gran.Groups / n * gran.Granule()
	var specs []cudagreen.Spec
	for range n {
		specs = append(specs, cudagreen.Spec{Role: cudagreen.RoleCompute, MinSMs: per, Fraction: float64(per) / float64(total)})
	}
	plan, err := cudagreen.ComputePlan(fmt.Sprintf("sm_%d%d", major, minor), total, gran, specs)
	if err != nil {
		return fmt.Errorf("rankgroup: green plan: %w", err)
	}
	if g.green, err = cudagreen.Open(dev, plan); err != nil {
		return err
	}
	for i, q := range plan.Partitions {
		g.streams = append(g.streams, g.green.Stream(i))
		g.sms = append(g.sms, q.SMs)
	}
	return nil
}

// newEvents makes one event per rank in the context of the rank's stream.
func (g *Group) newEvents() error {
	for r := range g.streams {
		mk := func() error {
			e, err := driver.NewEvent(driver.EventDisableTiming)
			if err != nil {
				return fmt.Errorf("rankgroup: rank %d event: %w", r, err)
			}
			g.events = append(g.events, e)
			return nil
		}
		if g.green == nil {
			if err := mk(); err != nil {
				return err
			}
			continue
		}
		ctx, err := driver.CtxFromGreenCtx(g.green.GreenCtx(r))
		if err != nil {
			return fmt.Errorf("rankgroup: rank %d green context: %w", r, err)
		}
		if err := ctx.PushCurrent(); err != nil {
			return fmt.Errorf("rankgroup: rank %d green context: %w", r, err)
		}
		err = mk()
		if _, perr := driver.CtxPopCurrent(); err == nil && perr != nil {
			err = fmt.Errorf("rankgroup: rank %d green context: %w", r, perr)
		}
		if err != nil {
			return err
		}
	}
	return nil
}

// OnDevices returns one rank on each of devs, each with its own stream.
func OnDevices(devs []*cuda.Device) (_ *Group, err error) {
	if len(devs) < 1 || len(devs) > MaxRanks {
		return nil, fmt.Errorf("rankgroup: %d ranks outside 1..%d", len(devs), MaxRanks)
	}
	g := &Group{mode: Streams, devs: append([]*cuda.Device(nil), devs...)}
	defer func() {
		if err != nil {
			err = errors.Join(err, g.Close())
		}
	}()
	for r, dev := range devs {
		ctx, err := dev.PrimaryContext()
		if err != nil {
			return nil, err
		}
		g.ctxs = append(g.ctxs, ctx)
		sms, err := dev.MultiprocessorCount()
		if err != nil {
			return nil, err
		}
		g.sms = append(g.sms, sms)
		err = ctx.Run(func() error {
			s, err := driver.StreamCreateWithPriority(uint32(driver.StreamNonBlocking), 0)
			if err != nil {
				return fmt.Errorf("rankgroup: rank %d stream: %w", r, err)
			}
			g.owned = append(g.owned, s)
			g.streams = append(g.streams, s)
			e, err := driver.NewEvent(driver.EventDisableTiming)
			if err != nil {
				return fmt.Errorf("rankgroup: rank %d event: %w", r, err)
			}
			g.events = append(g.events, e)
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	return g, nil
}

// Size returns the number of ranks.
func (g *Group) Size() int { return len(g.streams) }

// Mode returns the group's mode; a group made by [OnDevices] reports
// [Streams].
func (g *Group) Mode() Mode { return g.mode }

// Device returns rank r's device.
func (g *Group) Device(r int) *cuda.Device { return g.devs[r] }

// SMs returns the number of SMs rank r's work may use.
func (g *Group) SMs(r int) int { return g.sms[r] }

// Stream returns rank r's stream.
func (g *Group) Stream(r int) driver.Stream { return g.streams[r] }

// SingleDevice reports whether every rank is on one device.
func (g *Group) SingleDevice() bool {
	for _, d := range g.devs {
		if d != g.devs[0] {
			return false
		}
	}
	return true
}

// Do runs fn with rank 0's device context current on a locked OS thread.
// Launch switches to another rank's context when the ranks span devices.
func (g *Group) Do(fn func() error) error { return g.ctxs[0].Run(fn) }

// use makes rank r's device context current, if the ranks span devices.
func (g *Group) use(r int) error {
	if g.SingleDevice() {
		return nil
	}
	return g.ctxs[r].Raw().SetCurrent()
}

// Launch enqueues fn on rank r's stream. Arguments may be
// [driver.DevicePtr], int32, uint32, float32 or [Pointers]. It must run
// inside [Group.Do].
func (g *Group) Launch(r int, fn driver.Function, grid, block [3]uint32, smem uint32, args ...any) error {
	if err := g.use(r); err != nil {
		return err
	}
	params := make([]unsafe.Pointer, len(args))
	for i, a := range args {
		switch v := a.(type) {
		case driver.DevicePtr:
			params[i] = unsafe.Pointer(&v)
		case int32:
			params[i] = unsafe.Pointer(&v)
		case uint32:
			params[i] = unsafe.Pointer(&v)
		case float32:
			params[i] = unsafe.Pointer(&v)
		case Pointers:
			params[i] = unsafe.Pointer(&v)
		default:
			return fmt.Errorf("rankgroup: launch argument %d has type %T", i, a)
		}
	}
	err := fn.LaunchKernel(driver.Dim3{X: grid[0], Y: grid[1], Z: grid[2]}, driver.Dim3{X: block[0], Y: block[1], Z: block[2]}, smem, g.streams[r], params)
	if err != nil {
		return fmt.Errorf("rankgroup: rank %d launch: %w", r, err)
	}
	return nil
}

// Pointers is a kernel argument holding one device address per rank, as
// a struct of MaxRanks pointers.
type Pointers [MaxRanks]driver.DevicePtr

// Barrier makes each rank's later work wait for all ranks' earlier work.
// In [Serial] mode stream order already provides it. It must run inside
// [Group.Do].
func (g *Group) Barrier() error {
	if g.mode == Serial && g.SingleDevice() {
		return nil
	}
	for r, s := range g.streams {
		if err := g.use(r); err != nil {
			return err
		}
		if err := g.events[r].Record(s); err != nil {
			return fmt.Errorf("rankgroup: rank %d record: %w", r, err)
		}
	}
	for r, s := range g.streams {
		if err := g.use(r); err != nil {
			return err
		}
		for q, e := range g.events {
			if q == r {
				continue
			}
			if err := s.WaitEvent(e, 0); err != nil {
				return fmt.Errorf("rankgroup: rank %d wait on rank %d: %w", r, q, err)
			}
		}
	}
	return nil
}

// Synchronize waits for every rank's stream. It must run inside
// [Group.Do].
func (g *Group) Synchronize() error {
	for r, s := range g.streams {
		if err := g.use(r); err != nil {
			return err
		}
		if err := s.Synchronize(); err != nil {
			return fmt.Errorf("rankgroup: rank %d synchronize: %w", r, err)
		}
	}
	return nil
}

// Close destroys the group's streams, events and green contexts.
func (g *Group) Close() error {
	if len(g.ctxs) == 0 {
		return nil
	}
	var errs []error
	errs = append(errs, g.Do(func() error {
		var errs []error
		for r, e := range g.events {
			errs = append(errs, g.use(r), e.Close())
		}
		for _, s := range g.owned {
			errs = append(errs, s.Close())
		}
		if g.green != nil {
			errs = append(errs, g.green.Close())
		}
		return errors.Join(errs...)
	}))
	released := make(map[*cuda.Context]bool)
	for _, c := range g.ctxs {
		if !released[c] {
			released[c] = true
			errs = append(errs, c.Release())
		}
	}
	g.ctxs, g.events, g.owned, g.green = nil, nil, nil, nil
	return errors.Join(errs...)
}
