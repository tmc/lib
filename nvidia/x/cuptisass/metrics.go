package cuptisass

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

// readCString reads a NUL-terminated C string from pointer p. The uintptr is a
// C pointer field, never the address of Go memory. Interpret its pointer
// representation once and index with [unsafe.Add]: holding a pointer value in a
// uintptr across statements is what makes the conversion unsound. An
// unterminated string is truncated rather than walked off the end of its
// mapping.
func readCString(p uintptr) string {
	if p == 0 {
		return ""
	}
	base := *(*unsafe.Pointer)(unsafe.Pointer(&p))
	var b []byte
	for i := 0; i < 1<<20; i++ {
		c := *(*byte)(unsafe.Add(base, i))
		if c == 0 {
			break
		}
		b = append(b, c)
	}
	return string(b)
}

var (
	profilerOnce sync.Once
	profilerErr  error
)

func initProfiler() error {
	profilerOnce.Do(func() {
		var params cupti.ProfilerInitializeParams
		params.StructSize = uint(unsafe.Offsetof(params.PPriv) + unsafe.Sizeof(params.PPriv))
		if err := cupti.ProfilerInitialize(&params); err != nil {
			profilerErr = fmt.Errorf("cuptisass: profiler initialize: %w", err)
		}
	})
	return profilerErr
}

// DeviceChipName queries the chip name (e.g. "GH100", "GB100", "AD102") for a device index.
func DeviceChipName(deviceIndex uint32) (string, error) {
	if err := initProfiler(); err != nil {
		return "", err
	}
	var params cupti.DeviceGetChipNameParams
	params.StructSize = uint(unsafe.Offsetof(params.PChipName) + unsafe.Sizeof(params.PChipName))
	params.DeviceIndex = uint(deviceIndex)

	if err := cupti.DeviceGetChipName(&params); err != nil {
		return "", fmt.Errorf("cuptisass: get chip name for device %d: %w", deviceIndex, err)
	}
	name := readCString(params.PChipName)
	if name == "" {
		return "", fmt.Errorf("cuptisass: empty chip name returned for device %d", deviceIndex)
	}
	return name, nil
}

// SupportedMetrics queries the list of supported SASS metrics for a chip name.
func SupportedMetrics(chip string) ([]Metric, error) {
	if err := initProfiler(); err != nil {
		return nil, err
	}
	if chip == "" {
		return nil, errors.New("cuptisass: chip name cannot be empty")
	}
	chipBytes := append([]byte(chip), 0)
	chipPtr := uintptr(unsafe.Pointer(&chipBytes[0]))

	var numParams cupti.SassMetricsGetNumOfMetricsParams
	numParams.StructSize = uint(unsafe.Offsetof(numParams.NumOfMetrics) + unsafe.Sizeof(numParams.NumOfMetrics))
	numParams.PChipName = chipPtr

	if err := cupti.SassMetricsGetNumOfMetrics(&numParams); err != nil {
		runtime.KeepAlive(chipBytes)
		return nil, fmt.Errorf("cuptisass: get num of metrics for %s: %w", chip, err)
	}

	count := numParams.NumOfMetrics
	if count == 0 {
		runtime.KeepAlive(chipBytes)
		return nil, nil
	}

	details := make([]cupti.SassMetricsMetricDetails, count)
	var getParams cupti.SassMetricsGetMetricsParams
	getParams.StructSize = uint(unsafe.Offsetof(getParams.PMetricsList) + unsafe.Sizeof(getParams.PMetricsList))
	getParams.PChipName = chipPtr
	getParams.NumOfMetrics = count
	getParams.PMetricsList = uintptr(unsafe.Pointer(&details[0]))

	if err := cupti.SassMetricsGetMetrics(&getParams); err != nil {
		runtime.KeepAlive(chipBytes)
		runtime.KeepAlive(details)
		return nil, fmt.Errorf("cuptisass: get metrics for %s: %w", chip, err)
	}
	runtime.KeepAlive(chipBytes)

	metrics := make([]Metric, count)
	for i := range details {
		metrics[i] = Metric{
			ID:          details[i].MetricId,
			Name:        readCString(details[i].PMetricName),
			Description: readCString(details[i].PMetricDescription),
		}
	}
	return metrics, nil
}

// MetricProperties queries metric details for a specific metric name on a chip.
func MetricProperties(chip, metricName string) (Metric, error) {
	if err := initProfiler(); err != nil {
		return Metric{}, err
	}
	if chip == "" {
		return Metric{}, errors.New("cuptisass: chip name cannot be empty")
	}
	if metricName == "" {
		return Metric{}, errors.New("cuptisass: metric name cannot be empty")
	}
	chipBytes := append([]byte(chip), 0)
	chipPtr := uintptr(unsafe.Pointer(&chipBytes[0]))

	metricBytes := append([]byte(metricName), 0)
	metricPtr := uintptr(unsafe.Pointer(&metricBytes[0]))

	var params cupti.SassMetricsGetPropertiesParams
	params.StructSize = uint(unsafe.Offsetof(params.Metric) + unsafe.Sizeof(params.Metric))
	params.PChipName = chipPtr
	params.PMetricName = metricPtr

	if err := cupti.SassMetricsGetProperties(&params); err != nil {
		runtime.KeepAlive(chipBytes)
		runtime.KeepAlive(metricBytes)
		return Metric{}, fmt.Errorf("cuptisass: get properties for %q on %s: %w", metricName, chip, err)
	}
	runtime.KeepAlive(chipBytes)
	runtime.KeepAlive(metricBytes)

	return Metric{
		ID:          params.Metric.MetricId,
		Name:        readCString(params.Metric.PMetricName),
		Description: readCString(params.Metric.PMetricDescription),
	}, nil
}
