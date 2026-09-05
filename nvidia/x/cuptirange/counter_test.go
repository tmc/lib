package cuptirange

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
)

func TestCounterStoreCleanup(t *testing.T) {
	failure := errors.New("native failure")
	closed := 0
	api := counterAPI{
		enable: func(p *cupti.RangeProfilerEnableParams) error { p.PRangeProfilerObject = 7; return nil },
		size:   func(p *cupti.RangeProfilerGetCounterDataSizeParams) error { p.CounterDataSize = 32; return nil },
		initialize: func(p *cupti.RangeProfilerCounterDataImageInitializeParams) error {
			if p.CounterDataSize != 32 || p.PCounterData == 0 {
				t.Error("missing counter storage")
			}
			return failure
		},
		disable: func(*cupti.RangeProfilerDisableParams) error {
			closed++
			if closed == 1 {
				return failure
			}
			return nil
		},
	}
	c, err := newCounterStore(1, []string{"metric"}, 4, 8, 64, api)
	if c == nil || !errors.Is(err, failure) {
		t.Fatalf("store=%v error=%v", c, err)
	}
	if c.host != 7 || c.names == nil || len(c.image) != 32 {
		t.Fatal("lost retained resources")
	}
	if err := c.close(); err != nil {
		t.Fatal(err)
	}
	if err := c.close(); err != nil {
		t.Fatal(err)
	}
	if closed != 2 || c.host != 0 || c.names != nil {
		t.Fatal("cleanup not idempotent")
	}
}
