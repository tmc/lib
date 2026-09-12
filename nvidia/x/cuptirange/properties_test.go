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
	for _, version := range []uint32{0, 26, 130000, 130201, 130300} {
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

func TestPropertiesOlderABIHasNoScope(t *testing.T) {
	strings, err := newMetricNames([]string{"description", "sm", "cycle"})
	if err != nil {
		t.Fatal(err)
	}
	defer strings.close()
	for _, version := range []uint32{27, 130001} {
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
