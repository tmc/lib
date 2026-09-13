package cuptipcsampling

import (
	"errors"
	"fmt"
	"sync"

	"github.com/tmc/lib/nvidia/cupti"
)

type samplingControlAPI struct {
	version  func() (uint32, error)
	set, get func(*cupti.PCSamplingConfigurationInfoParams) error
	enable   func(*cupti.PCSamplingEnableParams) error
	start    func(*cupti.PCSamplingStartParams) error
	stop     func(*cupti.PCSamplingStopParams) error
	disable  func(*cupti.PCSamplingDisableParams) error
}

func nativeSamplingControlAPI() samplingControlAPI {
	return samplingControlAPI{
		version: cupti.GetVersion,
		set:     cupti.PCSamplingSetConfigurationAttribute,
		get:     cupti.PCSamplingGetConfigurationAttribute,
		enable:  cupti.PCSamplingEnable,
		start:   cupti.PCSamplingStart,
		stop:    cupti.PCSamplingStop,
		disable: cupti.PCSamplingDisable,
	}
}

// Enable turns on PC sampling for ctx and returns a function that turns it off.
//
// CUPTI rejects every PC sampling query on a context where sampling is not
// enabled, including the reason inventory that a [Config] needs to name its
// reasons. A caller that must build that inventory before [Start] therefore
// enables the context itself; [Start] enables the context it is given, so a
// caller with no inventory of its own never calls this.
//
// The returned function disables sampling once and reports the native error;
// later calls report the same result without calling CUPTI again.
func Enable(ctx cupti.Context) (disable func() error, err error) {
	return enableSampling(ctx, nativeSamplingControlAPI())
}

func enableSampling(ctx cupti.Context, api samplingControlAPI) (func() error, error) {
	if ctx == 0 || api.enable == nil || api.disable == nil {
		return nil, errors.New("cuptipcsampling: invalid enable request")
	}
	p := cupti.PCSamplingEnableParams{Size: enableParamsSize(), Ctx: ctx}
	if err := api.enable(&p); err != nil {
		return nil, fmt.Errorf("cuptipcsampling: enable sampling: %w", err)
	}
	if p.Size != 24 || p.PPriv != 0 || p.Ctx != ctx {
		return nil, errors.New("cuptipcsampling: enable changed input storage")
	}
	var (
		once sync.Once
		err  error
	)
	return func() error {
		once.Do(func() {
			d := cupti.PCSamplingDisableParams{Size: disableParamsSize(), Ctx: ctx}
			if e := api.disable(&d); e != nil {
				err = fmt.Errorf("cuptipcsampling: disable sampling: %w", e)
			}
		})
		return err
	}, nil
}

// samplingControl owns profiling state, never the workload or CUDA context.
// Its owner serializes calls and keeps the context alive. Native errors can
// leave state partly changed, so touched state requires successful disable
// before any pinned buffers or shared names may be released.
type samplingControl struct {
	ctx                                                          cupti.Context
	config                                                       *samplingConfig
	api                                                          samplingControlAPI
	prepared, touched, enabled, startAttempted, failed, disabled bool
	version                                                      uint32
	starts, stops                                                uint64
}

func (s *samplingControl) prepare() error {
	a := s.api
	if s.prepared || s.ctx == 0 || s.config == nil || a.version == nil || a.set == nil || a.get == nil || a.enable == nil || a.start == nil || a.stop == nil || a.disable == nil {
		return errors.New("cuptipcsampling: invalid sampling control")
	}
	if s.config.attempted || s.config.released || s.config.output.released {
		return errors.New("cuptipcsampling: sampling configuration is not fresh")
	}
	s.prepared = true
	v, err := a.version()
	if err != nil {
		return fmt.Errorf("cuptipcsampling: query sampling version: %w", err)
	}
	if !qualifiedSamplingVersion(v) {
		return fmt.Errorf("cuptipcsampling: unqualified sampling control version %d", v)
	}
	s.version = v
	s.touched = true
	if err := s.config.apply(s.ctx, a.set); err != nil {
		s.failed = true
		return err
	}
	if err := s.config.verify(s.ctx, a.get); err != nil {
		s.failed = true
		return err
	}
	// A context enabled before the configuration was applied is already
	// enabled here: the reason inventory the configuration names cannot be
	// queried on a context where sampling is off.
	if !s.enabled {
		p := cupti.PCSamplingEnableParams{Size: enableParamsSize(), Ctx: s.ctx}
		if err := a.enable(&p); err != nil {
			s.failed = true
			return fmt.Errorf("cuptipcsampling: enable sampling: %w", err)
		}
		if p.Size != 24 || p.PPriv != 0 || p.Ctx != s.ctx {
			s.failed = true
			return errors.New("cuptipcsampling: enable changed input storage")
		}
		s.enabled = true
	}
	return nil
}

func (s *samplingControl) start() error {
	if !s.enabled || s.disabled || s.failed || s.startAttempted {
		return errors.New("cuptipcsampling: sampling cannot start")
	}
	s.startAttempted = true
	p := cupti.PCSamplingStartParams{Size: startParamsSize(), Ctx: s.ctx}
	if err := s.api.start(&p); err != nil {
		s.failed = true
		return fmt.Errorf("cuptipcsampling: start sampling: %w", err)
	}
	if p.Size != 24 || p.PPriv != 0 || p.Ctx != s.ctx {
		s.failed = true
		return errors.New("cuptipcsampling: start changed input storage")
	}
	s.starts++
	return nil
}

func (s *samplingControl) stop() error {
	if !s.enabled || s.disabled || !s.startAttempted {
		return errors.New("cuptipcsampling: sampling cannot stop")
	}
	p := cupti.PCSamplingStopParams{Size: stopParamsSize(), Ctx: s.ctx}
	if err := s.api.stop(&p); err != nil {
		s.failed = true
		return fmt.Errorf("cuptipcsampling: stop sampling: %w", err)
	}
	if p.Size != 24 || p.PPriv != 0 || p.Ctx != s.ctx {
		s.failed = true
		return errors.New("cuptipcsampling: stop changed input storage")
	}
	s.startAttempted = false
	s.stops++
	return nil
}

// shutdown requires the owner's declared workload boundary. It makes at most
// one stop and one disable attempt, never synchronizes or replays work, and
// never frees buffers. Disable may fill the configured output buffer, which
// must still be inspected after success. A failed disable retains ownership;
// the owner may explicitly retry shutdown without re-enabling the collector.
func (s *samplingControl) shutdown() error {
	if s.disabled || !s.touched {
		return nil
	}
	var stopErr error
	if s.startAttempted {
		stopErr = s.stop()
	}
	p := cupti.PCSamplingDisableParams{Size: disableParamsSize(), Ctx: s.ctx}
	err := s.api.disable(&p)
	if err == nil && (p.Size != 24 || p.PPriv != 0 || p.Ctx != s.ctx) {
		err = errors.New("disable changed input storage")
	}
	if err != nil {
		s.failed = true
		return errors.Join(stopErr, fmt.Errorf("cuptipcsampling: disable sampling: %w", err))
	}
	s.disabled = true
	s.enabled = false
	s.startAttempted = false
	return stopErr
}

func (s *samplingControl) canRelease() bool { return !s.touched || s.disabled }
