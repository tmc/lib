package cuptirange

import (
	"errors"
	"fmt"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

// MetricProperties retains CUPTI's description, hardware unit, dimensional
// unit, metric type and collection scope. These describe a metric, not an
// observed value. Units and descriptions are copied verbatim from CUPTI.
// Scope is MetricCollectionScopeInvalid when the native ABI has no scope field.
type MetricProperties struct {
	Name         string
	Description  string
	HardwareUnit string
	Dimension    string
	Type         cupti.MetricType
	Scope        cupti.MetricCollectionScope
	CUPTIVersion uint32
}

// Properties queries a metric's native definition. maxStringBytes bounds each
// returned string, including its terminator. CUPTI versions 27 and 130001 use
// the older layout without collection scope; 130101 and later include scope.
// Other versions are rejected. A query does not prove collection permission or compatible
// scheduling. The plan must remain open.
func (p *Plan) Properties(name string, maxStringBytes int) (MetricProperties, error) {
	if p == nil || maxStringBytes <= 0 {
		return MetricProperties{}, errors.New("cuptirange: invalid metric properties request")
	}
	version, err := cupti.GetVersion()
	if err != nil {
		return MetricProperties{}, fmt.Errorf("cuptirange: query CUPTI version: %w", err)
	}
	return p.properties(name, maxStringBytes, version, cupti.ProfilerHostGetMetricProperties)
}

func (p *Plan) properties(name string, maxStringBytes int, version uint32, query func(*cupti.ProfilerHostGetMetricPropertiesParams) error) (MetricProperties, error) {
	var zero MetricProperties
	if p == nil || maxStringBytes <= 0 {
		return zero, errors.New("cuptirange: invalid metric properties request")
	}
	if !qualifiedLayout(version) {
		return zero, fmt.Errorf("cuptirange: unqualified metric properties version %d", version)
	}
	names, err := newMetricNames([]string{name})
	if err != nil {
		return zero, err
	}
	defer names.close()
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.host == 0 {
		return zero, errors.New("cuptirange: closed host")
	}
	v := cupti.ProfilerHostGetMetricPropertiesParams{PHostObject: p.host, PMetricName: names.pointers[0], MetricCollectionScope: cupti.MetricCollectionScopeInvalid}
	v.StructSize = uint(unsafe.Offsetof(v.MetricCollectionScope) + unsafe.Sizeof(v.MetricCollectionScope))
	if !scopedProperties(version) {
		v.StructSize = uint(unsafe.Offsetof(v.MetricType) + unsafe.Sizeof(v.MetricType))
	}
	if err := query(&v); err != nil {
		return zero, fmt.Errorf("cuptirange: metric properties: %w", err)
	}
	if v.MetricType < cupti.MetricTypeCounter || v.MetricType >= cupti.MetricTypeCount {
		return zero, errors.New("cuptirange: invalid native metric type")
	}
	if !scopedProperties(version) {
		v.MetricCollectionScope = cupti.MetricCollectionScopeInvalid
	} else if v.MetricCollectionScope != cupti.MetricCollectionScopeContext && v.MetricCollectionScope != cupti.MetricCollectionScopeDevice {
		return zero, errors.New("cuptirange: native metric scope unavailable")
	}
	out := MetricProperties{Name: name, Type: v.MetricType, Scope: v.MetricCollectionScope, CUPTIVersion: version}
	for _, field := range []struct {
		name string
		ptr  uintptr
		dst  *string
	}{{"description", v.PDescription, &out.Description}, {"hardware unit", v.PHwUnit, &out.HardwareUnit}, {"dimension", v.PDimUnit, &out.Dimension}} {
		value, err := copyMetricString(field.ptr, maxStringBytes)
		if err != nil {
			return zero, fmt.Errorf("cuptirange: %s: %w", field.name, err)
		}
		*field.dst = value
	}
	return out, nil
}
