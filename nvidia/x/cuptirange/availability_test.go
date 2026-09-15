package cuptirange

import (
	"errors"
	"fmt"
	"github.com/tmc/lib/nvidia/cupti"
	"testing"
)

func TestCounterAvailabilityBounds(t *testing.T) {
	for _, size := range []uint{0, 4, 9} {
		calls := 0
		image, err := counterAvailability(1, 8, func(p *cupti.ProfilerGetCounterAvailabilityParams) error {
			calls++
			if p.StructSize != 40 || p.Ctx != 1 || p.BAllowDeviceLevelCounters {
				t.Fatalf("params %+v", p)
			}
			if calls == 1 {
				if p.PCounterAvailabilityImage != 0 {
					t.Fatal("size query has image")
				}
				p.CounterAvailabilityImageSize = size
			} else if p.PCounterAvailabilityImage == 0 {
				t.Fatal("image query missing image")
			}
			return nil
		})
		if size == 4 {
			if err != nil || len(image) != 4 || calls != 2 {
				t.Fatalf("image=%v calls=%d err=%v", image, calls, err)
			}
		} else if err == nil || calls != 1 {
			t.Fatalf("size=%d calls=%d err=%v", size, calls, err)
		}
	}
}
func TestCounterAvailabilityFailures(t *testing.T) {
	for _, fail := range []int{1, 2, 3} {
		calls := 0
		_, err := counterAvailability(1, 8, func(p *cupti.ProfilerGetCounterAvailabilityParams) error {
			calls++
			p.CounterAvailabilityImageSize = 4
			if fail == calls {
				return errors.New("query failed")
			}
			if fail == 3 && calls == 2 {
				p.CounterAvailabilityImageSize = 8
			}
			return nil
		})
		if err == nil {
			t.Fatalf("failure %d accepted", fail)
		}
	}
}
func ExampleCounterAvailability() {
	_, err := CounterAvailability(nil, 1024)
	fmt.Println(err)
	// Output: cuptirange: invalid counter availability arguments
}
