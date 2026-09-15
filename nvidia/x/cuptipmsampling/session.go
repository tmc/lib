package cuptipmsampling

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/internal/cuptiowner"
)

type sessionAPI struct {
	version    func() (uint32, error)
	initialize func() error
	enable     func(*cupti.PmSamplingEnableParams) error
	config     func(*cupti.PmSamplingSetConfigParams) error
	size       func(*cupti.PmSamplingGetCounterDataSizeParams) error
	image      func(*cupti.PmSamplingCounterDataImageInitializeParams) error
	start      func(*cupti.PmSamplingStartParams) error
	stop       func(*cupti.PmSamplingStopParams) error
	disable    func(*cupti.PmSamplingDisableParams) error
}

// Session manages a CUPTI PM sampling lifecycle on a CUDA device. Its zero value is not usable.
// Do not discard the session until Close succeeds.
type Session struct {
	api          sessionAPI
	lease        *cuptiowner.Lease
	closing      bool
	initialized  bool
	unidentified bool
	mu           sync.Mutex
	handle       uintptr
	device       uint32
	version      uint32
	cfg          Config
	counterData  []byte
	pin          runtime.Pinner
	started      bool
	closed       bool
}

// NewSession creates, enables, and configures a PM sampling session. It excludes
// cooperating range and SASS collectors on the device until Close succeeds. A
// nonnil session returned with an error retains cleanup ownership; retry Close.
func NewSession(cfg Config) (*Session, error) {
	return newSession(cfg, sessionAPI{CheckVersion, initProfiler, cupti.PmSamplingEnable, cupti.PmSamplingSetConfig, cupti.PmSamplingGetCounterDataSize, cupti.PmSamplingCounterDataImageInitialize, cupti.PmSamplingStart, cupti.PmSamplingStop, cupti.PmSamplingDisable})
}
func newSession(cfg Config, api sessionAPI) (result *Session, err error) {
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

	v, err := api.version()
	if err != nil {
		return nil, err
	}
	lease, err := cuptiowner.AcquireDevice(cfg.DeviceIndex)
	if err != nil {
		return nil, err
	}
	cfg.ConfigImage = append([]byte(nil), cfg.ConfigImage...)
	cfg.MetricNames = append([]string(nil), cfg.MetricNames...)
	sess := &Session{api: api, lease: lease, device: cfg.DeviceIndex, version: v, cfg: cfg}
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

	// 1. Enable PM sampling on device
	var enableParams cupti.PmSamplingEnableParams
	enableParams.StructSize = uint(unsafe.Offsetof(enableParams.PPmSamplingObject) + unsafe.Sizeof(enableParams.PPmSamplingObject))
	enableParams.DeviceIndex = uint(cfg.DeviceIndex)

	err = api.enable(&enableParams)
	sess.handle = enableParams.PPmSamplingObject
	if err != nil {
		return nil, fmt.Errorf("cuptipmsampling: enable: %w", err)
	}
	if sess.handle == 0 {
		sess.unidentified = true
		return nil, errors.New("cuptipmsampling: enable returned no handle")
	}
	handle := sess.handle

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

	if err := api.config(&setParams); err != nil {
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

	if err := api.size(&sizeParams); err != nil {
		runtime.KeepAlive(cNames)
		runtime.KeepAlive(cBytes)
		return nil, fmt.Errorf("cuptipmsampling: get counter data size: %w", err)
	}
	runtime.KeepAlive(cNames)
	runtime.KeepAlive(cBytes)

	if sizeParams.CounterDataSize == 0 {
		return nil, errors.New("cuptipmsampling: counter data size returned zero")
	}

	// 4. Allocate and initialize counter data image
	counterData := make([]byte, sizeParams.CounterDataSize)
	sess.counterData = counterData
	sess.pin.Pin(&counterData[0])
	var initParams cupti.PmSamplingCounterDataImageInitializeParams
	initParams.StructSize = uint(unsafe.Offsetof(initParams.PCounterData) + unsafe.Sizeof(initParams.PCounterData))
	initParams.PPmSamplingObject = handle
	initParams.CounterDataSize = sizeParams.CounterDataSize
	initParams.PCounterData = uintptr(unsafe.Pointer(&counterData[0]))

	if err := api.image(&initParams); err != nil {
		runtime.KeepAlive(counterData)
		return nil, fmt.Errorf("cuptipmsampling: initialize counter data image: %w", err)
	}

	sess.initialized = true
	return sess, nil
}

// Start begins periodic hardware performance monitoring on the GPU.
func (s *Session) Start() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed || s.closing || !s.initialized {
		return errors.New("cuptipmsampling: session is closed")
	}
	if s.started {
		return errors.New("cuptipmsampling: session is already started")
	}

	var startParams cupti.PmSamplingStartParams
	startParams.StructSize = uint(unsafe.Offsetof(startParams.PPmSamplingObject) + unsafe.Sizeof(startParams.PPmSamplingObject))
	startParams.PPmSamplingObject = s.handle

	if err := s.api.start(&startParams); err != nil {
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

	if err := s.api.stop(&stopParams); err != nil {
		return fmt.Errorf("cuptipmsampling: stop: %w", err)
	}
	s.started = false
	return nil
}

// Decode decodes data stored in the GPU hardware buffer into the counter data image.
func (s *Session) Decode() (DecodeResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed || s.closing || !s.initialized {
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

	if s.closed || s.closing || !s.initialized {
		return CounterDataInfo{}, errors.New("cuptipmsampling: session is closed")
	}
	return s.dataInfoLocked()
}

// SampleInfo retrieves the start and end timestamps (in nanoseconds) for a given sample index.
func (s *Session) SampleInfo(sampleIndex uint) (SampleInfo, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed || s.closing || !s.initialized {
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

// Close stops sampling, disables PM sampling, and releases the device lease.
// Failed native cleanup retains handles and pinned storage for retry. After the
// first Close attempt, only cleanup methods may be used.
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
	if s.unidentified {
		return errors.New("cuptipmsampling: successful enable returned no handle; cleanup cannot be verified")
	}
	if s.started {
		p := cupti.PmSamplingStopParams{PPmSamplingObject: s.handle}
		p.StructSize = uint(unsafe.Offsetof(p.PPmSamplingObject) + unsafe.Sizeof(p.PPmSamplingObject))
		if err := s.api.stop(&p); err != nil {
			return fmt.Errorf("cuptipmsampling: close stop: %w", err)
		}
		s.started = false
	}
	if s.handle != 0 {
		p := cupti.PmSamplingDisableParams{PPmSamplingObject: s.handle}
		p.StructSize = uint(unsafe.Offsetof(p.PPmSamplingObject) + unsafe.Sizeof(p.PPmSamplingObject))
		if err := s.api.disable(&p); err != nil {
			return fmt.Errorf("cuptipmsampling: close disable: %w", err)
		}
		s.handle = 0
	}
	s.pin.Unpin()
	s.lease.Release()
	s.lease = nil
	s.closed = true
	return nil
}
