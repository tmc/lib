package cuptirange

import (
	"errors"
	"fmt"
	"reflect"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

func TestBaseMetrics(t *testing.T) {
	names, err := newMetricNames([]string{"z__cycles", "a__bytes"})
	if err != nil {
		t.Fatal(err)
	}
	defer names.close()
	p := &Plan{host: 1}
	query := func(params *cupti.ProfilerHostGetBaseMetricsParams) error {
		if params.PHostObject != 1 || params.MetricType != cupti.MetricTypeCounter || params.StructSize != uint(unsafe.Offsetof(params.NumMetrics)+unsafe.Sizeof(params.NumMetrics)) {
			t.Fatal("wrong query")
		}
		params.NumMetrics = 2
		params.PpMetricNames = uintptr(unsafe.Pointer(&names.pointers[0]))
		return nil
	}
	got, err := p.baseMetrics(cupti.MetricTypeCounter, 2, 32, query)
	if err != nil || !reflect.DeepEqual(got, []string{"a__bytes", "z__cycles"}) {
		t.Fatalf("names=%v err=%v", got, err)
	}
	names.data[0][0] = 'x'
	if got[1] != "z__cycles" {
		t.Fatal("inventory aliases native name")
	}
	for _, tt := range []struct {
		name         string
		count, bytes int
	}{{"count limit", 1, 32}, {"name limit", 2, 3}, {"zero count", 0, 32}, {"zero bytes", 2, 0}} {
		t.Run(tt.name, func(t *testing.T) {
			if got, err := p.baseMetrics(cupti.MetricTypeCounter, tt.count, tt.bytes, query); err == nil || got != nil {
				t.Fatalf("partial inventory=%v err=%v", got, err)
			}
		})
	}
}

func TestBaseMetricsFailures(t *testing.T) {
	p := &Plan{host: 1}
	want := errors.New("native failure")
	if _, err := p.baseMetrics(cupti.MetricTypeRatio, 10, 32, func(*cupti.ProfilerHostGetBaseMetricsParams) error { return want }); !errors.Is(err, want) {
		t.Fatalf("err=%v", err)
	}
	for _, tt := range []struct {
		name  string
		count uint
		table uintptr
	}{{"nil table", 1, 0}, {"excess count", ^uint(0), 0}} {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := p.baseMetrics(cupti.MetricTypeCounter, 10, 32, func(v *cupti.ProfilerHostGetBaseMetricsParams) error {
				v.NumMetrics = tt.count
				v.PpMetricNames = tt.table
				return nil
			}); err == nil {
				t.Fatal("invalid native result accepted")
			}
		})
	}
	for _, list := range [][]string{{"same", "other"}} {
		names, err := newMetricNames(list)
		if err != nil {
			t.Fatal(err)
		}
		names.pointers[1] = names.pointers[0]
		_, err = p.baseMetrics(cupti.MetricTypeCounter, 10, 32, func(v *cupti.ProfilerHostGetBaseMetricsParams) error {
			v.NumMetrics = uint(len(list))
			v.PpMetricNames = uintptr(unsafe.Pointer(&names.pointers[0]))
			return nil
		})
		names.close()
		if err == nil {
			t.Fatal("duplicate inventory accepted")
		}
	}
	p.host = 0
	if _, err := p.baseMetrics(cupti.MetricTypeCounter, 10, 32, func(*cupti.ProfilerHostGetBaseMetricsParams) error { t.Fatal("queried closed host"); return nil }); err == nil {
		t.Fatal("closed host accepted")
	}
}

func ExamplePlan_BaseMetrics() {
	var plan *Plan
	_, err := plan.BaseMetrics(cupti.MetricTypeCounter, 10000, 1024)
	fmt.Println(err)
	// Output: cuptirange: invalid metric inventory request
}

func TestSubMetrics(t *testing.T) {
	names, err := newMetricNames([]string{"sum", "avg"})
	if err != nil {
		t.Fatal(err)
	}
	defer names.close()
	p := &Plan{host: 1}
	got, err := p.subMetrics("sm__cycles", cupti.MetricTypeCounter, 10, 32, func(v *cupti.ProfilerHostGetSubMetricsParams) error {
		name, err := copyMetricString(v.PMetricName, 32)
		if err != nil || name != "sm__cycles" || v.PHostObject != 1 {
			t.Fatal("invalid native query")
		}
		v.NumOfSubmetrics = 2
		v.PpSubMetrics = uintptr(unsafe.Pointer(&names.pointers[0]))
		return nil
	})
	if err != nil || !reflect.DeepEqual(got, []string{"avg", "sum"}) {
		t.Fatalf("names=%v err=%v", got, err)
	}
	if _, err := p.subMetrics("bad\x00name", cupti.MetricTypeCounter, 10, 32, func(*cupti.ProfilerHostGetSubMetricsParams) error { t.Fatal("queried invalid name"); return nil }); err == nil {
		t.Fatal("invalid name accepted")
	}
}

func ExamplePlan_SubMetrics() {
	var plan *Plan
	_, err := plan.SubMetrics("sm__cycles", cupti.MetricTypeCounter, 100, 1024)
	fmt.Println(err)
	// Output: cuptirange: invalid metric inventory request
}
