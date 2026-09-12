package cuptirange

import (
	"errors"
	"fmt"
	"sort"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

// BaseMetrics returns an owned, sorted list of native base metric names for the
// plan's chip and counter availability image. kind selects counters, ratios or
// throughputs. Both limits must be positive; exceeding either fails without a
// partial inventory. This query does not collect measurements or replay work.
// A listed metric is not proof that its collection is permitted or compatible
// with other collectors. Close must not have succeeded before this call.
func (p *Plan) BaseMetrics(kind cupti.MetricType, maxNames, maxNameBytes int) ([]string, error) {
	return p.baseMetrics(kind, maxNames, maxNameBytes, cupti.ProfilerHostGetBaseMetrics)
}

func (p *Plan) baseMetrics(kind cupti.MetricType, maxNames, maxNameBytes int, query func(*cupti.ProfilerHostGetBaseMetricsParams) error) ([]string, error) {
	if p == nil || kind < cupti.MetricTypeCounter || kind >= cupti.MetricTypeCount || maxNames <= 0 || maxNameBytes <= 0 {
		return nil, errors.New("cuptirange: invalid metric inventory request")
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.host == 0 {
		return nil, errors.New("cuptirange: closed host")
	}
	params := cupti.ProfilerHostGetBaseMetricsParams{PHostObject: p.host, MetricType: kind}
	params.StructSize = uint(unsafe.Offsetof(params.NumMetrics) + unsafe.Sizeof(params.NumMetrics))
	if err := query(&params); err != nil {
		return nil, fmt.Errorf("cuptirange: list base metrics: %w", err)
	}
	return copyMetricNames(params.PpMetricNames, params.NumMetrics, maxNames, maxNameBytes)
}

// SubMetrics returns the native submetric names available for name and kind.
// It applies the same ownership and bounds rules as BaseMetrics. Returned names
// are copied verbatim; the caller must not invent suffixes or assume that every
// base metric supports the same rollups.
func (p *Plan) SubMetrics(name string, kind cupti.MetricType, maxNames, maxNameBytes int) ([]string, error) {
	return p.subMetrics(name, kind, maxNames, maxNameBytes, cupti.ProfilerHostGetSubMetrics)
}

func (p *Plan) subMetrics(name string, kind cupti.MetricType, maxNames, maxNameBytes int, query func(*cupti.ProfilerHostGetSubMetricsParams) error) ([]string, error) {
	if p == nil || kind < cupti.MetricTypeCounter || kind >= cupti.MetricTypeCount || maxNames <= 0 || maxNameBytes <= 0 {
		return nil, errors.New("cuptirange: invalid metric inventory request")
	}
	names, err := newMetricNames([]string{name})
	if err != nil {
		return nil, err
	}
	defer names.close()
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.host == 0 {
		return nil, errors.New("cuptirange: closed host")
	}
	params := cupti.ProfilerHostGetSubMetricsParams{PHostObject: p.host, MetricType: kind, PMetricName: names.pointers[0]}
	params.StructSize = uint(unsafe.Offsetof(params.PpSubMetrics) + unsafe.Sizeof(params.PpSubMetrics))
	if err := query(&params); err != nil {
		return nil, fmt.Errorf("cuptirange: list submetrics: %w", err)
	}
	return copyMetricNames(params.PpSubMetrics, params.NumOfSubmetrics, maxNames, maxNameBytes)
}

func copyMetricNames(address uintptr, count uint, maxNames, maxNameBytes int) ([]string, error) {
	if count > uint(maxNames) {
		return nil, fmt.Errorf("cuptirange: metric count %d exceeds limit %d", count, maxNames)
	}
	if count == 0 {
		return []string{}, nil
	}
	if address == 0 {
		return nil, errors.New("cuptirange: missing metric name table")
	}
	ptr := *(*unsafe.Pointer)(unsafe.Pointer(&address))
	table := unsafe.Slice((*uintptr)(ptr), int(count))
	names := make([]string, 0, len(table))
	seen := make(map[string]bool, len(table))
	for i, address := range table {
		name, err := copyMetricString(address, maxNameBytes)
		if err != nil {
			return nil, fmt.Errorf("cuptirange: metric %d: %w", i, err)
		}
		if name == "" || seen[name] {
			return nil, fmt.Errorf("cuptirange: empty or duplicate metric name %q", name)
		}
		seen[name] = true
		names = append(names, name)
	}
	sort.Strings(names)
	return names, nil
}

// Native output remains borrowed until copied under the plan's lifetime lock.
// Bounds limit copying but cannot make an invalid native pointer safe.
func copyMetricString(address uintptr, limit int) (string, error) {
	if address == 0 {
		return "", errors.New("nil metric name")
	}
	p := *(*unsafe.Pointer)(unsafe.Pointer(&address))
	var data []byte
	for i := 0; i < limit; i++ {
		b := *(*byte)(unsafe.Add(p, i))
		if b == 0 {
			return string(data), nil
		}
		data = append(data, b)
	}
	return "", errors.New("metric name exceeds byte limit")
}
