package cuptipmsampling

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

// Session manages a CUPTI PM sampling lifecycle on a CUDA device.
type Session struct {
	mu          sync.Mutex
	handle      uintptr
	device      uint32
	version     uint32
	cfg         Config
	counterData []byte
	pin         runtime.Pinner
	started     bool
	closed      bool
}

// NewSession creates, enables, and configures a new PM sampling session on the device.
func NewSession(cfg Config) (*Session, error) {
	if cfg.SamplingInterval == 0 {
		return nil, errors.New("cuptipmsampling: sampling interval must be greater than 0")
	}
	if cfg.MaxSamples == 0 {
		return nil, errors.New("cuptipmsampling: max samples must be greater than 0")
	}
	if len(cfg.ConfigImage) == 0 {
		return nil, errors.New("cuptipmsampling: config image cannot be empty")
	}
	if len(cfg.MetricNames) == 0 {
		return nil, errors.New("cuptipmsampling: at least one metric name must be specified")
	}

	v, err := CheckVersion()
	if err != nil {
		return nil, err
	}
	if err := initProfiler(); err != nil {
		return nil, err
	}

	// 1. Enable PM sampling on device
	var enableParams cupti.PmSamplingEnableParams
	enableParams.StructSize = uint(unsafe.Offsetof(enableParams.PPmSamplingObject) + unsafe.Sizeof(enableParams.PPmSamplingObject))
	enableParams.DeviceIndex = uint(cfg.DeviceIndex)

	if err := cupti.PmSamplingEnable(&enableParams); err != nil {
		return nil, fmt.Errorf("cuptipmsampling: enable: %w", err)
	}
	handle := enableParams.PPmSamplingObject

	cleanupDisable := func() {
		var disableParams cupti.PmSamplingDisableParams
		disableParams.StructSize = uint(unsafe.Offsetof(disableParams.PPmSamplingObject) + unsafe.Sizeof(disableParams.PPmSamplingObject))
		disableParams.PPmSamplingObject = handle
		_ = cupti.PmSamplingDisable(&disableParams)
	}

	// 2. Set configuration
	hwBufSize := cfg.HardwareBufferSize
	if hwBufSize == 0 {
		hwBufSize = 64 * 1024 * 1024
	}

	var setParams cupti.PmSamplingSetConfigParams
	setParams.StructSize = setConfigStructSize(v)
	setParams.PPmSamplingObject = handle
	setParams.ConfigSize = uint(len(cfg.ConfigImage))
	setParams.PConfig = uintptr(unsafe.Pointer(&cfg.ConfigImage[0]))
	setParams.HardwareBufferSize = hwBufSize
	setParams.SamplingInterval = cfg.SamplingInterval
	setParams.TriggerMode = cupti.PmSamplingTriggerMode(cfg.TriggerMode)
	setParams.HwBufferAppendMode = cupti.PmSamplingHardwareBufferAppendMode(cfg.AppendMode)

	if err := cupti.PmSamplingSetConfig(&setParams); err != nil {
		cleanupDisable()
		runtime.KeepAlive(cfg.ConfigImage)
		return nil, fmt.Errorf("cuptipmsampling: set config: %w", err)
	}
	runtime.KeepAlive(cfg.ConfigImage)

	// 3. Query required counter data image size
	cNames := make([]*byte, len(cfg.MetricNames))
	cBytes := make([][]byte, len(cfg.MetricNames))
	for i, name := range cfg.MetricNames {
		cBytes[i] = append([]byte(name), 0)
		cNames[i] = &cBytes[i][0]
	}

	var sizeParams cupti.PmSamplingGetCounterDataSizeParams
	sizeParams.StructSize = uint(unsafe.Offsetof(sizeParams.CounterDataSize) + unsafe.Sizeof(sizeParams.CounterDataSize))
	sizeParams.PPmSamplingObject = handle
	sizeParams.PMetricNames = uintptr(unsafe.Pointer(&cNames[0]))
	sizeParams.NumMetrics = uint(len(cNames))
	sizeParams.MaxSamples = cfg.MaxSamples

	if err := cupti.PmSamplingGetCounterDataSize(&sizeParams); err != nil {
		cleanupDisable()
		runtime.KeepAlive(cNames)
		runtime.KeepAlive(cBytes)
		return nil, fmt.Errorf("cuptipmsampling: get counter data size: %w", err)
	}
	runtime.KeepAlive(cNames)
	runtime.KeepAlive(cBytes)

	if sizeParams.CounterDataSize == 0 {
		cleanupDisable()
		return nil, errors.New("cuptipmsampling: counter data size returned zero")
	}

	// 4. Allocate and initialize counter data image
	counterData := make([]byte, sizeParams.CounterDataSize)
	var initParams cupti.PmSamplingCounterDataImageInitializeParams
	initParams.StructSize = uint(unsafe.Offsetof(initParams.PCounterData) + unsafe.Sizeof(initParams.PCounterData))
	initParams.PPmSamplingObject = handle
	initParams.CounterDataSize = sizeParams.CounterDataSize
	initParams.PCounterData = uintptr(unsafe.Pointer(&counterData[0]))

	if err := cupti.PmSamplingCounterDataImageInitialize(&initParams); err != nil {
		cleanupDisable()
		runtime.KeepAlive(counterData)
		return nil, fmt.Errorf("cuptipmsampling: initialize counter data image: %w", err)
	}

	sess := &Session{
		handle:      handle,
		device:      cfg.DeviceIndex,
		version:     v,
		cfg:         cfg,
		counterData: counterData,
	}
	sess.pin.Pin(&sess.counterData[0])
	return sess, nil
}

// Start begins periodic hardware performance monitoring on the GPU.
func (s *Session) Start() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return errors.New("cuptipmsampling: session is closed")
	}
	if s.started {
		return errors.New("cuptipmsampling: session is already started")
	}

	var startParams cupti.PmSamplingStartParams
	startParams.StructSize = uint(unsafe.Offsetof(startParams.PPmSamplingObject) + unsafe.Sizeof(startParams.PPmSamplingObject))
	startParams.PPmSamplingObject = s.handle

	if err := cupti.PmSamplingStart(&startParams); err != nil {
		return fmt.Errorf("cuptipmsampling: start: %w", err)
	}
	s.started = true
	return nil
}

// Stop halts periodic hardware performance monitoring.
func (s *Session) Stop() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.started {
		return nil
	}

	var stopParams cupti.PmSamplingStopParams
	stopParams.StructSize = uint(unsafe.Offsetof(stopParams.PPmSamplingObject) + unsafe.Sizeof(stopParams.PPmSamplingObject))
	stopParams.PPmSamplingObject = s.handle

	if err := cupti.PmSamplingStop(&stopParams); err != nil {
		return fmt.Errorf("cuptipmsampling: stop: %w", err)
	}
	s.started = false
	return nil
}

// Decode decodes data stored in the GPU hardware buffer into the counter data image.
func (s *Session) Decode() (DecodeResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return DecodeResult{}, errors.New("cuptipmsampling: session is closed")
	}

	var decodeParams cupti.PmSamplingDecodeDataParams
	decodeParams.StructSize = uint(unsafe.Offsetof(decodeParams.Overflow) + unsafe.Sizeof(decodeParams.Overflow))
	decodeParams.PPmSamplingObject = s.handle
	decodeParams.PCounterDataImage = uintptr(unsafe.Pointer(&s.counterData[0]))
	decodeParams.CounterDataImageSize = uint(len(s.counterData))

	if err := cupti.PmSamplingDecodeData(&decodeParams); err != nil {
		return DecodeResult{}, fmt.Errorf("cuptipmsampling: decode data: %w", err)
	}

	info, err := s.dataInfoLocked()
	if err != nil {
		return DecodeResult{}, err
	}

	return DecodeResult{
		StopReason: DecodeStopReason(decodeParams.DecodeStopReason),
		Overflow:   decodeParams.Overflow != 0,
		Info:       info,
	}, nil
}

// dataInfoLocked reads summary statistics from the counter data image.
func (s *Session) dataInfoLocked() (CounterDataInfo, error) {
	var infoParams cupti.PmSamplingGetCounterDataInfoParams
	infoParams.StructSize = uint(unsafe.Offsetof(infoParams.NumCompletedSamples) + unsafe.Sizeof(infoParams.NumCompletedSamples))
	infoParams.PCounterDataImage = uintptr(unsafe.Pointer(&s.counterData[0]))
	infoParams.CounterDataImageSize = uint(len(s.counterData))

	if err := cupti.PmSamplingGetCounterDataInfo(&infoParams); err != nil {
		return CounterDataInfo{}, fmt.Errorf("cuptipmsampling: get counter data info: %w", err)
	}

	return CounterDataInfo{
		NumTotalSamples:     infoParams.NumTotalSamples,
		NumPopulatedSamples: infoParams.NumPopulatedSamples,
		NumCompletedSamples: infoParams.NumCompletedSamples,
	}, nil
}

// DataInfo returns the current summary statistics for the counter data image.
func (s *Session) DataInfo() (CounterDataInfo, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return CounterDataInfo{}, errors.New("cuptipmsampling: session is closed")
	}
	return s.dataInfoLocked()
}

// SampleInfo retrieves the start and end timestamps (in nanoseconds) for a given sample index.
func (s *Session) SampleInfo(sampleIndex uint) (SampleInfo, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return SampleInfo{}, errors.New("cuptipmsampling: session is closed")
	}

	var params cupti.PmSamplingCounterDataGetSampleInfoParams
	params.StructSize = uint(unsafe.Offsetof(params.EndTimestamp) + unsafe.Sizeof(params.EndTimestamp))
	params.PPmSamplingObject = s.handle
	params.PCounterDataImage = uintptr(unsafe.Pointer(&s.counterData[0]))
	params.CounterDataImageSize = uint(len(s.counterData))
	params.SampleIndex = sampleIndex

	if err := cupti.PmSamplingCounterDataGetSampleInfo(&params); err != nil {
		return SampleInfo{}, fmt.Errorf("cuptipmsampling: get sample info for index %d: %w", sampleIndex, err)
	}

	return SampleInfo{
		Index:          sampleIndex,
		StartTimestamp: params.StartTimestamp,
		EndTimestamp:   params.EndTimestamp,
	}, nil
}

// CounterData returns a copy of the decoded counter data image.
func (s *Session) CounterData() []byte {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.counterData) == 0 {
		return nil
	}
	data := make([]byte, len(s.counterData))
	copy(data, s.counterData)
	return data
}

// Close stops sampling if running, disables PM sampling, and frees resources.
func (s *Session) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return nil
	}
	s.closed = true

	var firstErr error
	if s.started {
		var stopParams cupti.PmSamplingStopParams
		stopParams.StructSize = uint(unsafe.Offsetof(stopParams.PPmSamplingObject) + unsafe.Sizeof(stopParams.PPmSamplingObject))
		stopParams.PPmSamplingObject = s.handle
		if err := cupti.PmSamplingStop(&stopParams); err != nil {
			firstErr = fmt.Errorf("cuptipmsampling: close stop: %w", err)
		}
		s.started = false
	}

	if s.handle != 0 {
		var disableParams cupti.PmSamplingDisableParams
		disableParams.StructSize = uint(unsafe.Offsetof(disableParams.PPmSamplingObject) + unsafe.Sizeof(disableParams.PPmSamplingObject))
		disableParams.PPmSamplingObject = s.handle
		if err := cupti.PmSamplingDisable(&disableParams); err != nil && firstErr == nil {
			firstErr = fmt.Errorf("cuptipmsampling: close disable: %w", err)
		}
		s.handle = 0
	}

	if len(s.counterData) > 0 {
		s.pin.Unpin()
	}

	return firstErr
}
