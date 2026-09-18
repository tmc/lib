package cuptibench

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"unicode"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cuptirange"
	"github.com/tmc/lib/nvidia/x/cuptisass"
)

// Mode selects the ranges collected during a workload sequence.
type Mode int

const (
	// Whole collects one inclusive operation range.
	Whole Mode = iota
	// Regions collects the operation and its named, inclusive regions.
	Regions
	// Kernels collects automatic kernel ranges.
	//
	// Kernels is qualified least. QUALIFICATION.md records it failing on the
	// DS4 workload, where it "remains selectable, fails on this workload, and
	// never falls back", and records kernel capture failing in the bounded
	// prepared-replay row. Neither result generalizes to every workload, so
	// this is not refused here; it is stated so a caller choosing Kernels from
	// the doc comment knows the package has seen it fail, and can treat a
	// failure as expected rather than as a defect to chase.
	//
	// Callers that need a mode known to work on an arbitrary workload should
	// use Regions, which is what DS4 defaults to.
	Kernels
)

// Config bounds a profiling sequence. All limits and Operations must be positive.
type Config struct {
	// ReplayMode defaults to application-controlled user replay. KernelReplay
	// is explicit and only valid for Kernels; native internal replay can change
	// scheduling and cache state relative to the separately timed execution.
	ReplayMode cuptirange.ReplayMode
	// CachePolicy controls CUPTI cache resets, not application preparation.
	// Default and Preserve request no native reset. Flush policies require a
	// qualified reset ABI and currently return an error before workload execution.
	CachePolicy    cuptirange.CachePolicy
	Metrics        []string
	Mode           Mode
	Operations     int
	Limits         cuptirange.Limits
	MaxConfigBytes int
	// MaxDepth bounds nested Regions, including the operation root; zero uses 16.
	MaxDepth uint16
	// AllowWarmCacheTraffic acknowledges collecting cache-state-dependent
	// counters (L1/L2/DRAM sectors) without a CUPTI cache reset. New refuses
	// that combination by default, because a traffic measurement taken against
	// a warm cache understates the traffic the workload performs in situ and
	// nothing downstream can detect it.
	//
	// Setting this does not make the measurement comparable; it records that
	// the caller knows it is not. The measurement is labelled warm in Result
	// metadata and in benchmark output wherever the counters appear.
	//
	// There is currently no way to avoid this acknowledgement: CachePolicySupport
	// rejects CacheFlushL2 and CacheFlushAll on every version it admits, so no
	// flush policy is reachable and cold traffic must be arranged by the
	// application, by evicting between passes with its own work.
	AllowWarmCacheTraffic bool
}

// warmCacheTraffic reports the cache-state-dependent metrics in c that are
// being collected without a cache reset, in configuration order. A flush
// policy, once one is reachable, removes them all.
func (c Config) warmCacheTraffic() []string {
	if c.CachePolicy != cuptirange.CacheDefault && c.CachePolicy != cuptirange.CachePreserve {
		return nil
	}
	var names []string
	for _, name := range c.Metrics {
		if trafficMetric(name) {
			names = append(names, name)
		}
	}
	return names
}

func (c Config) validate() error {
	if c.ReplayMode < cuptirange.UserReplay || c.ReplayMode > cuptirange.KernelReplay || (c.ReplayMode == cuptirange.KernelReplay && c.Mode != Kernels) {
		return errors.New("cuptibench: invalid replay mode for collection mode")
	}
	if c.CachePolicy < cuptirange.CacheDefault || c.CachePolicy > cuptirange.CacheFlushAll {
		return errors.New("cuptibench: invalid cache policy")
	}
	if c.Mode < Whole || c.Mode > Kernels || c.Operations <= 0 || c.MaxConfigBytes <= 0 || c.Limits.MaxRanges == 0 || c.Limits.MaxRangeTreeNodes == 0 || c.Limits.MaxCounterBytes <= 0 || c.Limits.MaxPasses == 0 {
		return errors.New("cuptibench: invalid mode, operation count, or limits")
	}
	if uint64(c.Limits.MaxRangeTreeNodes) < uint64(c.Limits.MaxRanges) || (c.Mode != Regions && c.MaxDepth > 1) {
		return errors.New("cuptibench: invalid range tree or nesting bounds")
	}
	if len(c.Metrics) == 0 {
		return errors.New("cuptibench: no metrics")
	}
	seen := make(map[string]bool)
	for _, name := range c.Metrics {
		if name == "" || strings.ContainsRune(name, 0) || strings.IndexFunc(name, unicode.IsSpace) >= 0 || seen[name] {
			return fmt.Errorf("cuptibench: invalid or duplicate metric %q", name)
		}
		seen[name] = true
	}
	return nil
}

type planContext interface {
	Run(func() error) error
	Release() error
}
type collector interface {
	CaptureReplay(*cuda.Device, cuptirange.Limits, cuptirange.CaptureOptions, cuptirange.Replay) (*cuptirange.Capture, error)
	Evaluate([]byte, uint) ([]float64, error)
	Close() error
}

// Plan owns a retained CUDA context and metric configuration. Its zero value is
// not usable. Calls must be sequential; concurrent calls fail immediately.
type Plan struct {
	dev                 *cuda.Device
	config              Config
	configurationPasses uint
	context             planContext
	checkContext        func() error
	backend             collector
	units               []string
	additive            []bool
	metadata            map[string]string
	busy                atomic.Bool
	exportStatus        atomic.Uint32
	closed              bool
	pending             *cuptirange.Capture
	pendingTimer        interface{ Close() error }
	pendingOverride     *Plan
}

// New configures collection on dev's primary context. The device and workload
// resources must remain open until Close succeeds. A nonnil plan returned with
// an error retains cleanup ownership and must be closed.
func New(dev *cuda.Device, c Config) (p *Plan, err error) {
	if err = c.validate(); err != nil {
		return nil, err
	}
	if dev == nil {
		return nil, errors.New("cuptibench: nil device")
	}
	c.Metrics = append([]string(nil), c.Metrics...)
	p = &Plan{dev: dev, config: c, metadata: map[string]string{"metric_selection": "config"}, units: make([]string, len(c.Metrics)), additive: make([]bool, len(c.Metrics))}
	defer func() {
		if err != nil {
			if cleanup := p.Close(); cleanup != nil {
				err = errors.Join(err, cleanup)
			} else {
				p = nil
			}
		}
	}()
	// WithContext rejects a closed device before retaining another reference.
	err = dev.WithContext(func() error { var e error; p.context, e = dev.PrimaryContext(); return e })
	if err != nil {
		return p, err
	}
	ctx := p.context.(*cuda.Context)
	p.checkContext = func() error {
		current, e := driver.NewContextGetCurrent()
		if e != nil {
			return e
		}
		if current != ctx.Raw() {
			return errors.Join(errors.New("cuptibench: callback changed CUDA context"), ctx.Raw().SetCurrent())
		}
		return nil
	}
	err = p.context.Run(func() error {
		chip, e := cuptisass.DeviceChipName(uint32(dev.Ordinal()))
		if e != nil {
			return e
		}
		availability, e := cuptirange.CounterAvailability(dev, c.MaxConfigBytes)
		if e != nil {
			return e
		}
		p.metadata["counter_availability_sha256"] = fmt.Sprintf("%x", sha256.Sum256(availability))
		backend, e := cuptirange.NewPlan(chip, availability, c.Metrics, c.MaxConfigBytes)
		p.backend = backend
		if e != nil {
			return e
		}
		_, passes := backend.Configuration()
		p.configurationPasses = passes
		p.metadata["configuration_passes"] = fmt.Sprint(passes)
		if err := c.checkPassBudget(passes); err != nil {
			return err
		}
		p.metadata["chip"] = chip
		uuid, e := dev.Raw().GetUuid()
		if e != nil {
			return e
		}
		p.metadata["device_uuid"] = fmt.Sprintf("%x", uuid)
		version, e := driver.DriverGetVersion()
		if e != nil {
			return e
		}
		p.metadata["cuda_driver_version"] = fmt.Sprint(version)
		cv, e := cupti.GetVersion()
		if e != nil {
			return e
		}
		p.metadata["cupti_version"] = fmt.Sprint(cv)
		if e := cuptirange.ReplayModeSupport(cv, c.ReplayMode); e != nil {
			return e
		}
		if e := cuptirange.CachePolicySupport(cv, c.CachePolicy); e != nil {
			return e
		}
		p.metadata["cache_control_requested"] = c.CachePolicy.String()
		p.metadata["cache_control_effective"] = "none"
		p.metadata["cache_control_basis"] = "CUPTI documented default and pre-13.4 behavior; not a cache-content measurement"
		p.metadata["cache_control_reference"] = "https://docs.nvidia.com/cupti/13.4.0/main/main.html#cache-state-reset"
		if warm := c.warmCacheTraffic(); len(warm) > 0 {
			if !c.AllowWarmCacheTraffic {
				return fmt.Errorf("cuptibench: cache policy %s collects %s against an uncontrolled cache: "+
					"a traffic counter measured on a warm cache understates traffic and reads identically to a cold one. "+
					"No flush policy is reachable (CachePolicySupport rejects flush-l2 and flush-all on CUPTI %d), so either "+
					"evict in the workload's own preparation or set Config.AllowWarmCacheTraffic to record that the "+
					"measurement is warm",
					c.CachePolicy, strings.Join(warm, ", "), cv)
			}
			p.metadata["traffic_cache_state"] = "warm, acknowledged by caller"
			p.metadata["traffic_cache_metrics"] = strings.Join(warm, ",")
		}

		for i, name := range c.Metrics {
			prop, e := backend.Properties(name, 1<<20)
			if e != nil {
				return e
			}
			p.units[i] = prop.Dimension
			p.additive[i] = additiveMetric(name, prop.Dimension, prop.Type)
			if p.units[i] == "" {
				p.units[i] = "unknown"
			}
		}
		return nil
	})
	return p, err
}

func (p *Plan) begin() error {
	if p == nil {
		return errors.New("cuptibench: nil plan")
	}
	if !p.busy.CompareAndSwap(false, true) {
		return errors.New("cuptibench: plan is in use")
	}
	if p.closed || p.context == nil || p.backend == nil {
		p.end()
		return errors.New("cuptibench: closed or uninitialized plan")
	}
	if p.pending != nil || p.pendingTimer != nil || p.pendingOverride != nil {
		p.end()
		return errors.New("cuptibench: pending collection cleanup; call Close")
	}
	return nil
}
func (p *Plan) end() { p.busy.Store(false) }

// Close releases resources. It rejects active use and retains failed cleanup
// for retry. Close is nil-safe and repeated successful calls are harmless.
func (p *Plan) Close() (err error) {
	if p == nil {
		return nil
	}
	if !p.busy.CompareAndSwap(false, true) {
		return errors.New("cuptibench: plan is in use")
	}
	defer p.end()
	defer func() {
		if err != nil {
			p.exportStatus.Or(exportOwnerFailed)
		} else {
			p.exportStatus.Or(exportOwnerClosed)
		}
	}()
	if p.pendingOverride != nil {
		if err := p.pendingOverride.Close(); err != nil {
			return err
		}
		p.pendingOverride = nil
	}
	if p.pendingTimer != nil {
		if err := p.pendingTimer.Close(); err != nil {
			return err
		}
		p.pendingTimer = nil
	}
	if p.pending != nil {
		if err := p.pending.Close(); err != nil {
			return err
		}
		p.pending = nil
	}
	if p.backend != nil {
		if err := p.backend.Close(); err != nil {
			return err
		}
		p.backend = nil
	}
	if p.context != nil {
		if err := p.context.Release(); err != nil {
			return err
		}
		p.context = nil
	}
	p.closed = true
	return nil
}

// call checks callback context ownership before native collection continues.
func (p *Plan) call(fn func() error) (err error) {
	if p.checkContext != nil {
		if err = p.checkContext(); err != nil {
			return err
		}
		defer func() { err = errors.Join(err, p.checkContext()) }()
	}
	return fn()
}

// checkPassBudget rejects a known insufficient host replay budget. CUPTI handles
// kernel replay internally. For user replay the configuration count is only a
// lower bound: nested ranges can require additional host passes, whose exact
// count is not exposed by ProfilerHostGetNumOfPasses.
func (c Config) checkPassBudget(passes uint) error {
	if c.ReplayMode != cuptirange.KernelReplay && passes > c.Limits.MaxPasses {
		return fmt.Errorf("cuptibench: configuration requires %d passes, limit is %d", passes, c.Limits.MaxPasses)
	}
	return nil
}
