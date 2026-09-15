package cuptirange

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

type counterAPI struct {
	enable     func(*cupti.RangeProfilerEnableParams) error
	disable    func(*cupti.RangeProfilerDisableParams) error
	size       func(*cupti.RangeProfilerGetCounterDataSizeParams) error
	initialize func(*cupti.RangeProfilerCounterDataImageInitializeParams) error
}

// counterStore owns the target object and all memory it can reference. Methods
// must execute with the owning CUDA context current. Failed disable retains pins.
type counterStore struct {
	mu          sync.Mutex
	api         counterAPI
	host        uintptr
	names       *metricNames
	image       []byte
	config      []byte
	rangeNames  map[string][]byte
	kernelNames *kernelNames
	quality     CollectionQuality
	pin         runtime.Pinner
}

func newCounterStore(context cupti.Context, metrics []string, maxRanges uint, maxNodes uint32, maxBytes int, api counterAPI) (store *counterStore, err error) {
	if context == 0 || maxRanges == 0 || uint64(maxNodes) < uint64(maxRanges) || maxBytes <= 0 {
		return nil, errors.New("cuptirange: invalid counter limits or context")
	}
	names, err := newMetricNames(metrics)
	if err != nil {
		return nil, err
	}
	c := &counterStore{api: api, names: names}
	enable := cupti.RangeProfilerEnableParams{Ctx: context}
	enable.StructSize = uint(unsafe.Offsetof(enable.PRangeProfilerObject) + unsafe.Sizeof(enable.PRangeProfilerObject))
	enableErr := api.enable(&enable)
	c.host = enable.PRangeProfilerObject
	defer func() {
		if err != nil {
			if cleanup := c.close(); cleanup != nil {
				store, err = c, errors.Join(err, cleanup)
			}
		}
	}()
	if enableErr != nil {
		return nil, fmt.Errorf("cuptirange: enable: %w", enableErr)
	}
	if c.host == 0 {
		return nil, errors.New("cuptirange: empty range profiler object")
	}
	size := cupti.RangeProfilerGetCounterDataSizeParams{
		PRangeProfilerObject: c.host, PMetricNames: uintptr(unsafe.Pointer(&names.pointers[0])),
		NumMetrics: uint(len(names.pointers)), MaxNumOfRanges: maxRanges, MaxNumRangeTreeNodes: maxNodes,
	}
	size.StructSize = uint(unsafe.Offsetof(size.CounterDataSize) + unsafe.Sizeof(size.CounterDataSize))
	if err := api.size(&size); err != nil {
		return nil, fmt.Errorf("cuptirange: counter size: %w", err)
	}
	if size.CounterDataSize == 0 || size.CounterDataSize > uint(maxBytes) {
		return nil, fmt.Errorf("cuptirange: counter size %d exceeds bounds", size.CounterDataSize)
	}
	c.image = make([]byte, int(size.CounterDataSize))
	c.pin.Pin(&c.image[0])
	init := cupti.RangeProfilerCounterDataImageInitializeParams{PRangeProfilerObject: c.host, CounterDataSize: size.CounterDataSize, PCounterData: uintptr(unsafe.Pointer(&c.image[0]))}
	init.StructSize = uint(unsafe.Offsetof(init.PCounterData) + unsafe.Sizeof(init.PCounterData))
	if err := api.initialize(&init); err != nil {
		return nil, fmt.Errorf("cuptirange: initialize counter image: %w", err)
	}
	return c, nil
}

func (c *counterStore) close() error {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.host != 0 {
		params := cupti.RangeProfilerDisableParams{PRangeProfilerObject: c.host}
		params.StructSize = uint(unsafe.Offsetof(params.PRangeProfilerObject) + unsafe.Sizeof(params.PRangeProfilerObject))
		if err := c.api.disable(&params); err != nil {
			return fmt.Errorf("cuptirange: disable: %w", err)
		}
		c.host = 0
	}
	c.pin.Unpin()
	c.rangeNames = nil
	if c.names != nil {
		c.names.close()
		c.names = nil
	}
	return nil
}
