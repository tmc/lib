package cuptirange

import (
	"errors"
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
)

func TestProperties(t *testing.T) {
	strings, err := newMetricNames([]string{"native description", "dram", "byte"})
	if err != nil {
		t.Fatal(err)
	}
	defer strings.close()
	p := &Plan{host: 1}
	query := func(v *cupti.ProfilerHostGetMetricPropertiesParams) error {
		name, err := copyMetricString(v.PMetricName, 64)
		if err != nil || name != "dram__bytes_read.sum" || v.PHostObject != 1 || v.StructSize != 64 {
			t.Fatal("invalid query")
		}
		v.PDescription, v.PHwUnit, v.PDimUnit = strings.pointers[0], strings.pointers[1], strings.pointers[2]
		v.MetricType = cupti.MetricTypeCounter
		v.MetricCollectionScope = cupti.MetricCollectionScopeDevice
		return nil
	}
	got, err := p.properties("dram__bytes_read.sum", 64, 130200, query)
	if err != nil || got.Dimension != "byte" || got.Description != "native description" || got.Scope != cupti.MetricCollectionScopeDevice || got.CUPTIVersion != 130200 {
		t.Fatalf("properties=%+v err=%v", got, err)
	}
	strings.data[0][0] = 'x'
	if got.Description != "native description" {
		t.Fatal("borrowed description escaped")
	}
	if got, err := p.properties("dram__bytes_read.sum", 4, 130200, query); err == nil || got != (MetricProperties{}) {
		t.Fatalf("partial properties=%+v err=%v", got, err)
	}
}

func TestPropertiesRejectsUnknownABIAndScope(t *testing.T) {
	p := &Plan{host: 1}
	for _, version := range []uint32{0, 25, 29, 130000, 130201, 130300} {
		if _, err := p.properties("metric", 64, version, func(*cupti.ProfilerHostGetMetricPropertiesParams) error {
			t.Fatal("queried unqualified ABI")
			return nil
		}); err == nil {
			t.Fatal("unknown ABI accepted")
		}
	}
	if _, err := p.properties("metric", 64, 130200, func(*cupti.ProfilerHostGetMetricPropertiesParams) error { return nil }); err == nil {
		t.Fatal("unwritten scope became measured context scope")
	}
	want := errors.New("permission denied")
	if _, err := p.properties("metric", 64, 130200, func(*cupti.ProfilerHostGetMetricPropertiesParams) error { return want }); !errors.Is(err, want) {
		t.Fatalf("native error=%v", err)
	}
	p.host = 0
	if _, err := p.properties("metric", 64, 130200, func(*cupti.ProfilerHostGetMetricPropertiesParams) error { t.Fatal("queried closed host"); return nil }); err == nil {
		t.Fatal("closed host accepted")
	}
}

// TestPropertiesOlderABIHasNoScope covers the versions whose properties struct
// ends at metricType. The admitted CUDA 12 builds and 130001 do; 130101 carries
// metricCollectionScope even though its initialize struct does not grow.
func TestPropertiesOlderABIHasNoScope(t *testing.T) {
	strings, err := newMetricNames([]string{"description", "sm", "cycle"})
	if err != nil {
		t.Fatal(err)
	}
	defer strings.close()
	for _, version := range []uint32{24, 26, 27, 28, 130001} {
		p := &Plan{host: 1}
		got, err := p.properties("metric", 64, version, func(v *cupti.ProfilerHostGetMetricPropertiesParams) error {
			if v.StructSize != 60 {
				t.Fatalf("old struct size=%d", v.StructSize)
			}
			v.PDescription, v.PHwUnit, v.PDimUnit = strings.pointers[0], strings.pointers[1], strings.pointers[2]
			v.MetricType = cupti.MetricTypeRatio
			return nil
		})
		if err != nil || got.Scope != cupti.MetricCollectionScopeInvalid || got.Dimension != "cycle" || got.Type != cupti.MetricTypeRatio {
			t.Fatalf("properties=%+v err=%v", got, err)
		}
	}
}

// TestPropertiesScopedVersions covers every version whose properties struct
// reaches metricCollectionScope, which is the field CUPTI fills only when the
// declared size includes it.
func TestPropertiesScopedVersions(t *testing.T) {
	names, err := newMetricNames([]string{"description", "sm", "cycle"})
	if err != nil {
		t.Fatal(err)
	}
	defer names.close()
	for _, version := range []uint32{130101, 130200, 130202, 130301, 130400} {
		p := &Plan{host: 1}
		got, err := p.properties("metric", 64, version, func(v *cupti.ProfilerHostGetMetricPropertiesParams) error {
			if v.StructSize != 64 {
				t.Fatalf("scoped struct size=%d", v.StructSize)
			}
			v.PDescription, v.PHwUnit, v.PDimUnit = names.pointers[0], names.pointers[1], names.pointers[2]
			v.MetricType = cupti.MetricTypeRatio
			v.MetricCollectionScope = cupti.MetricCollectionScopeDevice
			return nil
		})
		if err != nil || got.Scope != cupti.MetricCollectionScopeDevice || got.CUPTIVersion != version {
			t.Fatalf("%d: properties=%+v err=%v", version, got, err)
		}
	}
}

func ExamplePlan_Properties() {
	var plan *Plan
	_, err := plan.Properties("dram__bytes_read.sum", 4096)
	fmt.Println(err)
	// Output: cuptirange: invalid metric properties request
}

func ExampleMetricProperties() {
	properties := MetricProperties{Name: "native_metric", Scope: cupti.MetricCollectionScopeDevice}
	fmt.Println(properties.Name, properties.Scope)
	// Output: native_metric MetricCollectionScopeDevice
}
