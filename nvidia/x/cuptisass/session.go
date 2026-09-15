package cuptisass

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/internal/cuptiowner"
)

type sessionAPI struct {
	version    func() error
	initialize func() error
	device     func(uintptr) (uint32, error)
	config     func(*cupti.SassMetricsSetConfigParams) error
	unset      func(*cupti.SassMetricsUnsetConfigParams) error
	enable     func(*cupti.SassMetricsEnableParams) error
	disable    func(*cupti.SassMetricsDisableParams) error
}

// Session manages a CUPTI SASS metrics data collection lifecycle on a CUDA context. Its zero value is not usable.
// Do not discard the session until Close succeeds.
type Session struct {
	api        sessionAPI
	lease      *cuptiowner.Lease
	closing    bool
	mu         sync.Mutex
	ctx        uintptr
	device     uint32
	cfg        Config
	configured bool
	enabled    bool
	closed     bool
}

// NewSession reserves the device and configures SASS collection on ctx. The
// context must belong to cfg.DeviceIndex and stay live until Close succeeds.
// Cooperating range and PM collectors are excluded until Close succeeds. A
// nonnil session returned with an error retains cleanup ownership; retry Close.
func NewSession(ctx uintptr, cfg Config) (*Session, error) {
	return newSession(ctx, cfg, sessionAPI{CheckVersion, initProfiler, contextDevice, cupti.SassMetricsSetConfig, cupti.SassMetricsUnsetConfig, cupti.SassMetricsEnable, cupti.SassMetricsDisable})
}
func newSession(ctx uintptr, cfg Config, api sessionAPI) (result *Session, err error) {
	if ctx == 0 {
		return nil, errors.New("cuptisass: nil context")
	}
	if len(cfg.Metrics) == 0 {
		return nil, errors.New("cuptisass: at least one metric must be configured")
	}
	for i, m := range cfg.Metrics {
		if m.Granularity > GranularitySMSP {
			return nil, fmt.Errorf("cuptisass: invalid output granularity %d for metric index %d", m.Granularity, i)
		}
	}

	if err := api.version(); err != nil {
		return nil, err
	}
	device, err := api.device(ctx)
	if err != nil {
		return nil, err
	}
	if device != cfg.DeviceIndex {
		return nil, errors.New("cuptisass: context belongs to a different device")
	}
	lease, err := cuptiowner.AcquireDevice(device)
	if err != nil {
		return nil, err
	}
	cfg.Metrics = append([]MetricConfig(nil), cfg.Metrics...)
	sess := &Session{api: api, lease: lease, ctx: ctx, device: device, cfg: cfg}
	defer func() {
		if err != nil {
			cleanup := sess.Close()
			if cleanup != nil {
				result = sess
				err = errors.Join(err, cleanup)
			}
		}
	}()
	if err := api.initialize(); err != nil {
		return nil, err
	}

	pConfigs := make([]cupti.SassMetricsConfig, len(cfg.Metrics))
	for i, m := range cfg.Metrics {
		pConfigs[i] = cupti.SassMetricsConfig{
			MetricId:          m.MetricID,
			OutputGranularity: uint8(m.Granularity),
		}
	}

	var configPin runtime.Pinner
	configPin.Pin(&pConfigs[0])
	defer configPin.Unpin()
	var setParams cupti.SassMetricsSetConfigParams
	setParams.StructSize = uint(unsafe.Offsetof(setParams.DeviceIndex) + unsafe.Sizeof(setParams.DeviceIndex))
	setParams.NumOfMetricConfig = uint(len(pConfigs))
	setParams.PConfigs = uintptr(unsafe.Pointer(&pConfigs[0]))
	setParams.DeviceIndex = cfg.DeviceIndex

	if err := api.config(&setParams); err != nil {
		runtime.KeepAlive(pConfigs)
		return nil, fmt.Errorf("cuptisass: set config: %w", err)
	}
	runtime.KeepAlive(pConfigs)

	sess.configured = true
	return sess, nil
}

// Enable starts SASS metric profiling. Subsequent kernel launches on the context
// will be patched and profiled.
func (s *Session) Enable() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed || s.closing {
		return errors.New("cuptisass: session is closed")
	}
	if !s.configured {
		return errors.New("cuptisass: session is not configured")
	}
	if s.enabled {
		return errors.New("cuptisass: session is already enabled")
	}

	var enableParams cupti.SassMetricsEnableParams
	enableParams.StructSize = uint(unsafe.Offsetof(enableParams.EnableLazyPatching) + unsafe.Sizeof(enableParams.EnableLazyPatching))
	enableParams.Ctx = cupti.Context(s.ctx)
	if s.cfg.LazyPatching {
		enableParams.EnableLazyPatching = 1
	}

	if err := s.api.enable(&enableParams); err != nil {
		s.closing = true
		return fmt.Errorf("cuptisass: enable: %w", err)
	}
	s.enabled = true
	return nil
}

// Disable stops SASS metric profiling. It returns the number of dropped instruction records
// if any kernels were executed whose data was not flushed prior to Disable.
func (s *Session) Disable() (uint, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.enabled {
		return 0, nil
	}

	var disableParams cupti.SassMetricsDisableParams
	disableParams.StructSize = uint(unsafe.Offsetof(disableParams.NumOfDroppedRecords) + unsafe.Sizeof(disableParams.NumOfDroppedRecords))
	disableParams.Ctx = cupti.Context(s.ctx)

	if err := s.api.disable(&disableParams); err != nil {
		return 0, fmt.Errorf("cuptisass: disable: %w", err)
	}
	s.enabled = false
	return disableParams.NumOfDroppedRecords, nil
}

// DataProperties returns the count of patched instruction records and metric instances collected so far.
func (s *Session) DataProperties() (numPatchedRecords, numInstances uint, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed || s.closing {
		return 0, 0, errors.New("cuptisass: session is closed")
	}

	var props cupti.SassMetricsGetDataPropertiesParams
	props.StructSize = uint(unsafe.Offsetof(props.NumOfInstances) + unsafe.Sizeof(props.NumOfInstances))
	props.Ctx = cupti.Context(s.ctx)

	if err := cupti.SassMetricsGetDataProperties(&props); err != nil {
		return 0, 0, fmt.Errorf("cuptisass: get data properties: %w", err)
	}
	return props.NumOfPatchedInstructionRecords, props.NumOfInstances, nil
}

// Flush retrieves all collected SASS metric records from CUPTI's internal buffers and clears them.
func (s *Session) Flush() ([]Record, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed || s.closing {
		return nil, errors.New("cuptisass: session is closed")
	}

	var props cupti.SassMetricsGetDataPropertiesParams
	props.StructSize = uint(unsafe.Offsetof(props.NumOfInstances) + unsafe.Sizeof(props.NumOfInstances))
	props.Ctx = cupti.Context(s.ctx)

	if err := cupti.SassMetricsGetDataProperties(&props); err != nil {
		return nil, fmt.Errorf("cuptisass: flush query properties: %w", err)
	}

	if props.NumOfPatchedInstructionRecords == 0 || props.NumOfInstances == 0 {
		return nil, nil
	}

	numRecords := props.NumOfPatchedInstructionRecords
	numInstances := props.NumOfInstances

	rawRecords := make([]cupti.SassMetricsData, numRecords)
	rawValues := make([][]cupti.SassMetricsInstanceValue, numRecords)

	var pin runtime.Pinner
	pin.Pin(&rawRecords[0])
	for i := uint(0); i < numRecords; i++ {
		rawValues[i] = make([]cupti.SassMetricsInstanceValue, numInstances)
		pin.Pin(&rawValues[i][0])
		rawRecords[i].StructSize = uint(unsafe.Sizeof(cupti.SassMetricsData{}))
		rawRecords[i].PInstanceValues = uintptr(unsafe.Pointer(&rawValues[i][0]))
	}

	var flushParams cupti.SassMetricsFlushDataParams
	flushParams.StructSize = uint(unsafe.Offsetof(flushParams.NumOfInstances) + unsafe.Sizeof(flushParams.NumOfInstances))
	flushParams.Ctx = cupti.Context(s.ctx)
	flushParams.NumOfPatchedInstructionRecords = numRecords
	flushParams.NumOfInstances = numInstances
	flushParams.PMetricsData = uintptr(unsafe.Pointer(&rawRecords[0]))

	err := cupti.SassMetricsFlushData(&flushParams)
	if err != nil {
		pin.Unpin()
		return nil, fmt.Errorf("cuptisass: flush data: %w", err)
	}

	records := make([]Record, numRecords)
	for i := uint(0); i < numRecords; i++ {
		name := readCString(rawRecords[i].FunctionName)
		if rawRecords[i].FunctionName != 0 {
			nativeFree(rawRecords[i].FunctionName)
		}
		records[i] = Record{
			CubinCRC:      rawRecords[i].CubinCrc,
			FunctionIndex: rawRecords[i].FunctionIndex,
			FunctionName:  name,
			PCOffset:      rawRecords[i].PcOffset,
			Values:        make([]InstanceValue, numInstances),
		}
		for j := uint(0); j < numInstances; j++ {
			records[i].Values[j] = InstanceValue{
				MetricID: rawValues[i][j].MetricId,
				Value:    rawValues[i][j].Value,
			}
		}
	}
	pin.Unpin()
	return records, nil
}

// Close disables SASS metrics collection if active, resets the device configuration,
// and releases the device lease. Failed disable or unset retains ownership for
// retry; after a Close attempt only cleanup methods may be called.
func (s *Session) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closing = true
	if s.enabled {
		p := cupti.SassMetricsDisableParams{Ctx: cupti.Context(s.ctx)}
		p.StructSize = uint(unsafe.Offsetof(p.NumOfDroppedRecords) + unsafe.Sizeof(p.NumOfDroppedRecords))
		if err := s.api.disable(&p); err != nil {
			return fmt.Errorf("cuptisass: close disable: %w", err)
		}
		s.enabled = false
	}
	if s.configured {
		p := cupti.SassMetricsUnsetConfigParams{DeviceIndex: s.device}
		p.StructSize = uint(unsafe.Offsetof(p.DeviceIndex) + unsafe.Sizeof(p.DeviceIndex))
		if err := s.api.unset(&p); err != nil {
			return fmt.Errorf("cuptisass: close unset config: %w", err)
		}
		s.configured = false
	}
	s.lease.Release()
	s.lease = nil
	s.closed = true
	return nil
}

// contextDevice verifies the configured ordinal against the actual context,
// using the longstanding current-context API without requiring cuCtxGetDevice_v2.
func contextDevice(ctx uintptr) (device uint32, err error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	previous, err := driver.NewContextGetCurrent()
	if err != nil {
		return 0, err
	}
	if err := driver.Context(ctx).SetCurrent(); err != nil {
		return 0, err
	}
	defer func() { err = errors.Join(err, previous.SetCurrent()) }()
	raw, err := driver.CtxGetDevice()
	if err != nil {
		return 0, err
	}
	return uint32(raw), nil
}
