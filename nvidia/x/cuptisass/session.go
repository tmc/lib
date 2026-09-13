package cuptisass

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

// Session manages a CUPTI SASS metrics data collection lifecycle on a CUDA context.
type Session struct {
	mu         sync.Mutex
	ctx        uintptr
	device     uint32
	cfg        Config
	configured bool
	enabled    bool
	closed     bool
}

// NewSession configures and creates a new SASS metrics collection session for the device and context.
func NewSession(ctx uintptr, cfg Config) (*Session, error) {
	if len(cfg.Metrics) == 0 {
		return nil, errors.New("cuptisass: at least one metric must be configured")
	}
	for i, m := range cfg.Metrics {
		if m.Granularity > GranularitySMSP {
			return nil, fmt.Errorf("cuptisass: invalid output granularity %d for metric index %d", m.Granularity, i)
		}
	}

	if err := CheckVersion(); err != nil {
		return nil, err
	}
	if err := initProfiler(); err != nil {
		return nil, err
	}

	pConfigs := make([]cupti.SassMetricsConfig, len(cfg.Metrics))
	for i, m := range cfg.Metrics {
		pConfigs[i] = cupti.SassMetricsConfig{
			MetricId:          m.MetricID,
			OutputGranularity: uint8(m.Granularity),
		}
	}

	var setParams cupti.SassMetricsSetConfigParams
	setParams.StructSize = uint(unsafe.Offsetof(setParams.DeviceIndex) + unsafe.Sizeof(setParams.DeviceIndex))
	setParams.NumOfMetricConfig = uint(len(pConfigs))
	setParams.PConfigs = uintptr(unsafe.Pointer(&pConfigs[0]))
	setParams.DeviceIndex = cfg.DeviceIndex

	if err := cupti.SassMetricsSetConfig(&setParams); err != nil {
		runtime.KeepAlive(pConfigs)
		return nil, fmt.Errorf("cuptisass: set config: %w", err)
	}
	runtime.KeepAlive(pConfigs)

	return &Session{
		ctx:        ctx,
		device:     cfg.DeviceIndex,
		cfg:        cfg,
		configured: true,
	}, nil
}

// Enable starts SASS metric profiling. Subsequent kernel launches on the context
// will be patched and profiled.
func (s *Session) Enable() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
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

	if err := cupti.SassMetricsEnable(&enableParams); err != nil {
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

	if err := cupti.SassMetricsDisable(&disableParams); err != nil {
		return 0, fmt.Errorf("cuptisass: disable: %w", err)
	}
	s.enabled = false
	return disableParams.NumOfDroppedRecords, nil
}

// DataProperties returns the count of patched instruction records and metric instances collected so far.
func (s *Session) DataProperties() (numPatchedRecords, numInstances uint, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
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

	if s.closed {
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
// and releases all associated session state.
func (s *Session) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return nil
	}
	s.closed = true

	var firstErr error
	if s.enabled {
		var disableParams cupti.SassMetricsDisableParams
		disableParams.StructSize = uint(unsafe.Offsetof(disableParams.NumOfDroppedRecords) + unsafe.Sizeof(disableParams.NumOfDroppedRecords))
		disableParams.Ctx = cupti.Context(s.ctx)
		if err := cupti.SassMetricsDisable(&disableParams); err != nil {
			firstErr = fmt.Errorf("cuptisass: close disable: %w", err)
		}
		s.enabled = false
	}

	if s.configured {
		var unsetParams cupti.SassMetricsUnsetConfigParams
		unsetParams.StructSize = uint(unsafe.Offsetof(unsetParams.DeviceIndex) + unsafe.Sizeof(unsetParams.DeviceIndex))
		unsetParams.DeviceIndex = s.device
		if err := cupti.SassMetricsUnsetConfig(&unsetParams); err != nil && firstErr == nil {
			firstErr = fmt.Errorf("cuptisass: close unset config: %w", err)
		}
		s.configured = false
	}

	return firstErr
}
