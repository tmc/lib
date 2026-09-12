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

func TestCounterStoreEnableFailureOwnership(t *testing.T) {
	enableFailure := errors.New("enable failed")
	cleanupFailure := errors.New("disable failed")
	for _, tt := range []struct {
		name        string
		handle      uintptr
		failCleanup bool
	}{
		{"no handle", 0, false},
		{"returned handle", 7, false},
		{"retry cleanup", 7, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			closed := 0
			api := counterAPI{
				enable: func(p *cupti.RangeProfilerEnableParams) error {
					p.PRangeProfilerObject = tt.handle
					return enableFailure
				},
				disable: func(p *cupti.RangeProfilerDisableParams) error {
					closed++
					if p.PRangeProfilerObject != tt.handle {
						t.Fatal("lost cleanup identity")
					}
					if tt.failCleanup && closed == 1 {
						return cleanupFailure
					}
					return nil
				},
				size: func(*cupti.RangeProfilerGetCounterDataSizeParams) error {
					t.Fatal("continued after enable failure")
					return nil
				},
			}
			c, err := newCounterStore(1, []string{"metric"}, 4, 8, 64, api)
			if !errors.Is(err, enableFailure) {
				t.Fatalf("lost enable error: %v", err)
			}
			if tt.failCleanup {
				if !errors.Is(err, cleanupFailure) || c == nil || c.host != tt.handle || c.names == nil {
					t.Fatalf("lost pending ownership: store=%v error=%v", c, err)
				}
				if err := c.close(); err != nil {
					t.Fatal(err)
				}
				if c.host != 0 || c.names != nil || closed != 2 {
					t.Fatal("retry did not release resources")
				}
			} else {
				wantClosed := 0
				if tt.handle != 0 {
					wantClosed = 1
				}
				if c != nil || closed != wantClosed {
					t.Fatalf("store=%v closes=%d want=%d", c, closed, wantClosed)
				}
			}
		})
	}
}
