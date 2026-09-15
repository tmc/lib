package cuptirange

import (
	"errors"
	"fmt"
	"sync"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/internal/cuptiowner"
	"github.com/tmc/lib/nvidia/x/cuda"
)

// Limits bounds counter storage and caller-controlled replay.
type Limits struct {
	MaxRanges         uint
	MaxRangeTreeNodes uint32
	MaxCounterBytes   int
	MaxPasses         uint
}

// CollectionQuality retains native collection progress independently of capture
// success. Pass counts count successful native Start/Stop calls, not verified
// workload outputs or CUPTI internal kernel replays. AllPassesSubmitted means a successful Stop reported 1.
// DroppedRanges is meaningful only when Decoded is true; an unattempted or failed
// decode must not be reported as zero loss. Cleanup failures remain separate.
type CollectionQuality struct {
	PassesStarted, PassesStopped uint
	AllPassesSubmitted           bool
	Decoded                      bool
	DroppedRanges                uint64
}

// Capture retains counter data, collection quality and any pending cleanup.
// Image and Ranges must not be used if Capture returned an error.
type Capture struct {
	// RangeInfo is populated by CaptureReplay, in native evaluation index order.
	RangeInfo []RangeInfo
	Image     []byte
	Ranges    uint
	Quality   CollectionQuality
	// DiagnosticImage is a copy of the counter buffer after a failed collection.
	// It may contain only initialization or partial data. Retain it for diagnostics,
	// never evaluate it as a completed image. Its size is bounded by MaxCounterBytes.
	DiagnosticImage []byte
	lease           *cuptiowner.Lease
	mu              sync.Mutex
	context         captureContext
	store           *counterStore
	kernelNames     *kernelNames
}

type captureContext interface {
	Run(func() error) error
	Release() error
}

// Close retries cleanup with the retained CUDA context current. It is nil-safe
// and safe for concurrent calls. Successful capture already released resources.
func (c *Capture) Close() error {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.context == nil {
		return nil
	}
	if c.store != nil {
		if err := c.context.Run(c.store.close); err != nil {
			return err
		}
		c.store = nil
	}
	if c.kernelNames != nil {
		if err := c.kernelNames.Close(); err != nil {
			return err
		}
		c.kernelNames = nil
	}
	if err := c.context.Release(); err != nil {
		return err
	}
	c.context = nil
	c.lease.Release()
	c.lease = nil
	return nil
}

// Capture collects automatic kernel ranges using bounded user replay. run is
// called with the device's retained primary context current; it must restore
// identical workload state on every pass. Each pass synchronizes before Stop.
// The caller must exclusively own profiling on this context and arrange device
// isolation externally. Profiling changes execution and is not a timing baseline.
// Any nonnil result accompanying an error must be closed to retry cleanup.
// If the configuration's required pass count exceeds MaxPasses, Capture fails
// before acquiring a CUDA context or invoking run. The count is a planning
// bound, not proof of completion: native completion is still checked after runs.
func (p *Plan) Capture(dev *cuda.Device, limits Limits, run func(uint) error) (result *Capture, err error) {
	if run == nil {
		return nil, errors.New("cuptirange: nil callback")
	}
	return p.captureReplay(dev, limits, CaptureOptions{Mode: Kernels}, Replay{Run: func(pass uint, _ *Scope) error { return run(pass) }, Synchronize: driver.CtxSynchronize}, false)
}

// CaptureReplay collects bounded user replay with preparation outside profiling.
// Regions verifies the named region trace across passes. CUPTI user replay
// still requires the caller to repeat identical kernel launches and state;
// automatic range inventory does not independently prove replay equivalence.
// MaxDepth is a configured nesting bound, and native completion may require
// more passes than Configuration reports for one nesting level.
// Any nonnil result accompanying an error must be closed to retry cleanup.
func (p *Plan) CaptureReplay(dev *cuda.Device, limits Limits, options CaptureOptions, replay Replay) (*Capture, error) {
	return p.captureReplay(dev, limits, options, replay, true)
}

func (p *Plan) captureReplay(dev *cuda.Device, limits Limits, options CaptureOptions, replay Replay, identities bool) (result *Capture, err error) {
	if options.Mode < Whole || options.Mode > Kernels || replay.Run == nil || replay.Synchronize == nil {
		return nil, errors.New("cuptirange: invalid replay options")
	}
	if options.ReplayMode != UserReplay && options.ReplayMode != KernelReplay {
		return nil, errors.New("cuptirange: invalid replay mode")
	}
	if options.ReplayMode == KernelReplay && options.Mode != Kernels {
		return nil, errors.New("cuptirange: kernel replay requires automatic kernel ranges")
	}
	if options.MaxDepth == 0 {
		options.MaxDepth = 1
		if options.Mode == Regions {
			options.MaxDepth = 16
		}
	}
	if options.Mode != Regions && options.MaxDepth != 1 {
		return nil, errors.New("cuptirange: nesting requires region mode")
	}
	if p == nil || dev == nil || limits.MaxPasses == 0 || limits.MaxRanges == 0 || uint64(limits.MaxRangeTreeNodes) < uint64(limits.MaxRanges) || limits.MaxCounterBytes <= 0 {
		return nil, errors.New("cuptirange: invalid capture arguments")
	}
	p.mu.Lock()
	if p.active {
		p.mu.Unlock()
		return nil, errors.New("cuptirange: capture active")
	}
	if p.pending != nil {
		if err := p.pending.Close(); err != nil {
			p.mu.Unlock()
			return nil, err
		}
		p.pending = nil
	}
	if p.host == 0 || p.names == nil {
		p.mu.Unlock()
		return nil, errors.New("cuptirange: closed plan")
	}
	if p.passes > limits.MaxPasses {
		p.mu.Unlock()
		return nil, fmt.Errorf("cuptirange: configuration requires %d passes, limit is %d", p.passes, limits.MaxPasses)
	}
	if err := ReplayModeSupport(p.version, options.ReplayMode); err != nil {
		p.mu.Unlock()
		return nil, err
	}
	if err := CachePolicySupport(p.version, options.CachePolicy); err != nil {
		p.mu.Unlock()
		return nil, err
	}
	p.active = true
	defer func() {
		p.mu.Lock()
		p.active = false
		// Retain the cleanup owner even if a workload panic prevents returning it.
		p.pending = result
		p.mu.Unlock()
	}()
	config := append([]byte(nil), p.config...)
	metrics := make([]string, len(p.names.data))
	for i, name := range p.names.data {
		metrics[i] = string(name[:len(name)-1])
	}
	p.mu.Unlock()
	ctx, err := dev.PrimaryContext()
	if err != nil {
		return nil, err
	}
	result = &Capture{context: ctx}
	result.lease, err = cuptiowner.AcquireDevice(uint32(dev.Ordinal()))
	if err != nil {
		return result, errors.Join(err, result.Close())
	}
	defer func() { err = errors.Join(err, result.Close()) }()
	err = ctx.Run(func() error {
		store, err := newCounterStore(cupti.Context(ctx.Raw()), metrics, limits.MaxRanges, limits.MaxRangeTreeNodes, limits.MaxCounterBytes,
			counterAPI{cupti.RangeProfilerEnable, cupti.RangeProfilerDisable, cupti.RangeProfilerGetCounterDataSize, cupti.RangeProfilerCounterDataImageInitialize})
		result.store = store
		if err != nil {
			return err
		}
		if options.Mode == Kernels {
			names, err := newKernelNames(cupti.Context(ctx.Raw()), limits.MaxRanges)
			result.kernelNames = names
			if err != nil {
				return err
			}
			if err := names.configureGraphs(options.Graphs, options.ReplayMode == KernelReplay); err != nil {
				return err
			}
			store.kernelNames = names
		}
		ranges, info, err := store.captureReplay(config, limits, options, replay, identities)
		result.RangeInfo = info
		result.retainCollection(store, ranges, err)
		if err != nil {
			return err
		}
		return nil
	})
	return result, err
}

func (c *Capture) retainCollection(store *counterStore, ranges uint, err error) {
	c.Quality = store.quality
	if err != nil {
		c.DiagnosticImage = append([]byte(nil), store.image...)
		return
	}
	c.Image = append([]byte(nil), store.image...)
	c.Ranges = ranges
}
