package cuptirange

import (
	"errors"
	"fmt"
	"runtime"
	"strings"
	"sync"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

// Plan owns a Range Profiler host object and its configuration. Do not copy it.
// Close must succeed before its pinned input storage can be released.
type Plan struct {
	api                        hostAPI
	mu                         sync.Mutex
	host                       uintptr
	names                      *metricNames
	pin                        runtime.Pinner
	chip, availability, config []byte
	passes                     uint
}

type hostAPI struct {
	version      func() (uint32, error)
	evaluate     func(*cupti.ProfilerHostEvaluateToGpuValuesParams) error
	initialize   func(*cupti.ProfilerHostInitializeParams) error
	deinitialize func(*cupti.ProfilerHostDeinitializeParams) error
	add          func(*cupti.ProfilerHostConfigAddMetricsParams) error
	size         func(*cupti.ProfilerHostGetConfigImageSizeParams) error
	config       func(*cupti.ProfilerHostGetConfigImageParams) error
	passes       func(*cupti.ProfilerHostGetNumOfPassesParams) error
}

// NewPlan builds a metric configuration for chip using an optional native
// counter availability image. maxConfigBytes bounds native-requested storage.
// It does not collect counters or replay application work. A nonnil Plan
// accompanying an error must be closed to retry failed native cleanup.
// Host initialization admits the 64-bit layouts for CUPTI 27, 130001 and
// 130200. Other versions fail before creating a host object.
func NewPlan(chip string, availability []byte, metrics []string, maxConfigBytes int) (plan *Plan, err error) {
	return newPlan(chip, availability, metrics, maxConfigBytes, hostAPI{
		version:    cupti.GetVersion,
		evaluate:   cupti.ProfilerHostEvaluateToGpuValues,
		initialize: cupti.ProfilerHostInitialize, deinitialize: cupti.ProfilerHostDeinitialize,
		add: cupti.ProfilerHostConfigAddMetrics, size: cupti.ProfilerHostGetConfigImageSize,
		config: cupti.ProfilerHostGetConfigImage, passes: cupti.ProfilerHostGetNumOfPasses,
	})
}

func newPlan(chip string, availability []byte, metrics []string, maxConfigBytes int, api hostAPI) (plan *Plan, err error) {
	if chip == "" || strings.ContainsRune(chip, 0) || maxConfigBytes <= 0 {
		return nil, errors.New("cuptirange: invalid chip or configuration limit")
	}
	names, err := newMetricNames(metrics)
	if err != nil {
		return nil, err
	}
	p := &Plan{api: api, names: names, chip: append([]byte(chip), 0), availability: append([]byte(nil), availability...)}
	defer func() {
		if err != nil {
			if cleanup := p.Close(); cleanup != nil {
				plan, err = p, errors.Join(err, cleanup)
			}
		}
	}()
	version, err := api.version()
	if err != nil {
		return nil, fmt.Errorf("cuptirange: query CUPTI version: %w", err)
	}
	if err := p.initializeHost(version); err != nil {
		return nil, err
	}
	add := cupti.ProfilerHostConfigAddMetricsParams{PHostObject: p.host, PpMetricNames: uintptr(unsafe.Pointer(&names.pointers[0])), NumMetrics: uint(len(names.pointers))}
	add.StructSize = uint(unsafe.Offsetof(add.NumMetrics) + unsafe.Sizeof(add.NumMetrics))
	if err := api.add(&add); err != nil {
		return nil, fmt.Errorf("cuptirange: add metrics: %w", err)
	}
	size := cupti.ProfilerHostGetConfigImageSizeParams{PHostObject: p.host}
	size.StructSize = uint(unsafe.Offsetof(size.ConfigImageSize) + unsafe.Sizeof(size.ConfigImageSize))
	if err := api.size(&size); err != nil {
		return nil, fmt.Errorf("cuptirange: configuration size: %w", err)
	}
	if size.ConfigImageSize == 0 || size.ConfigImageSize > uint(maxConfigBytes) {
		return nil, fmt.Errorf("cuptirange: configuration size %d exceeds bounds", size.ConfigImageSize)
	}
	p.config = make([]byte, int(size.ConfigImageSize))
	p.pin.Pin(&p.config[0])
	get := cupti.ProfilerHostGetConfigImageParams{PHostObject: p.host, ConfigImageSize: size.ConfigImageSize, PConfigImage: uintptr(unsafe.Pointer(&p.config[0]))}
	get.StructSize = uint(unsafe.Offsetof(get.PConfigImage) + unsafe.Sizeof(get.PConfigImage))
	if err := api.config(&get); err != nil {
		return nil, fmt.Errorf("cuptirange: configuration: %w", err)
	}
	passes := cupti.ProfilerHostGetNumOfPassesParams{ConfigImageSize: size.ConfigImageSize, PConfigImage: get.PConfigImage}
	passes.StructSize = uint(unsafe.Offsetof(passes.NumOfPasses) + unsafe.Sizeof(passes.NumOfPasses))
	if err := api.passes(&passes); err != nil {
		return nil, fmt.Errorf("cuptirange: passes: %w", err)
	}
	if passes.NumOfPasses == 0 {
		return nil, errors.New("cuptirange: configuration requires zero passes")
	}
	p.passes = passes.NumOfPasses
	return p, nil
}

// Close releases the native host object. Failed cleanup is retryable.
func (p *Plan) Close() error {
	if p == nil {
		return nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.host != 0 {
		params := cupti.ProfilerHostDeinitializeParams{PHostObject: p.host}
		params.StructSize = uint(unsafe.Offsetof(params.PHostObject) + unsafe.Sizeof(params.PHostObject))
		if err := p.api.deinitialize(&params); err != nil {
			return fmt.Errorf("cuptirange: close host: %w", err)
		}
		p.host = 0
	}
	p.pin.Unpin()
	if p.names != nil {
		p.names.close()
		p.names = nil
	}
	return nil
}

// Configuration returns a copy of the configuration image and required passes.
func (p *Plan) Configuration() ([]byte, uint) {
	if p == nil {
		return nil, 0
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]byte(nil), p.config...), p.passes
}
