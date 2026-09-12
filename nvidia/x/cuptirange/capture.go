package cuptirange

import (
	"errors"
	"fmt"
	"sync"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
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
// workload outputs. AllPassesSubmitted means a successful Stop reported 1.
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
	Image   []byte
	Ranges  uint
	Quality CollectionQuality
	// DiagnosticImage is a copy of the counter buffer after a failed collection.
	// It may contain only initialization or partial data. Retain it for diagnostics,
	// never evaluate it as a completed image. Its size is bounded by MaxCounterBytes.
	DiagnosticImage []byte
	mu              sync.Mutex
	context         captureContext
	store           *counterStore
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
	if err := c.context.Release(); err != nil {
		return err
	}
	c.context = nil
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
	if p == nil || dev == nil || run == nil || limits.MaxPasses == 0 || limits.MaxRanges == 0 || limits.MaxRangeTreeNodes == 0 || limits.MaxCounterBytes <= 0 {
		return nil, errors.New("cuptirange: invalid capture arguments")
	}
	p.mu.Lock()
	if p.host == 0 || p.names == nil {
		p.mu.Unlock()
		return nil, errors.New("cuptirange: closed plan")
	}
	if p.passes > limits.MaxPasses {
		p.mu.Unlock()
		return nil, fmt.Errorf("cuptirange: configuration requires %d passes, limit is %d", p.passes, limits.MaxPasses)
	}
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
	defer func() { err = errors.Join(err, result.Close()) }()
	err = ctx.Run(func() error {
		store, err := newCounterStore(cupti.Context(ctx.Raw()), metrics, limits.MaxRanges, limits.MaxRangeTreeNodes, limits.MaxCounterBytes,
			counterAPI{cupti.RangeProfilerEnable, cupti.RangeProfilerDisable, cupti.RangeProfilerGetCounterDataSize, cupti.RangeProfilerCounterDataImageInitialize})
		result.store = store
		if err != nil {
			return err
		}
		ranges, err := store.collect(config, limits.MaxRanges, limits.MaxPasses, func(pass uint) (err error) {
			defer func() { err = errors.Join(err, driver.CtxSynchronize()) }()
			return run(pass)
		}, collectionAPI{cupti.RangeProfilerSetConfig, cupti.RangeProfilerStart, cupti.RangeProfilerStop, cupti.RangeProfilerDecodeData, cupti.RangeProfilerGetCounterDataInfo})
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
