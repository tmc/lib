package cuptirange

import (
	"errors"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

func TestEvaluate(t *testing.T) {
	for _, mode := range []string{"success", "native error", "unwritten"} {
		t.Run(mode, func(t *testing.T) {
			names, err := newMetricNames([]string{"one", "two"})
			if err != nil {
				t.Fatal(err)
			}
			defer names.close()
			failure := errors.New("evaluate failed")
			p := Plan{host: 1, names: names, api: hostAPI{evaluate: func(params *cupti.ProfilerHostEvaluateToGpuValuesParams) error {
				if params.RangeIndex != 3 || params.NumMetrics != 2 || params.CounterDataImageSize != 4 {
					t.Error("incorrect evaluation parameters")
				}
				if mode == "native error" {
					return failure
				}
				if mode == "success" {
					ptr := *(*unsafe.Pointer)(unsafe.Pointer(&params.PMetricValues))
					values := unsafe.Slice((*float64)(ptr), 2)
					values[0], values[1] = 7, 11
				}
				return nil
			}}}
			values, err := p.Evaluate([]byte{1, 2, 3, 4}, 3)
			if mode == "success" {
				if err != nil || len(values) != 2 || values[0] != 7 || values[1] != 11 {
					t.Fatalf("values=%v error=%v", values, err)
				}
			} else if err == nil || values != nil {
				t.Fatal("invalid evaluation accepted")
			}
			if mode == "native error" && !errors.Is(err, failure) {
				t.Fatal("lost native error")
			}
		})
	}
}
