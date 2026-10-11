package cuptirange

import (
	"errors"
	"slices"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

// recordingAPI returns a counterAPI whose calls append their names to calls.
func recordingAPI(t *testing.T, calls *[]string, profilerErr error) counterAPI {
	t.Helper()
	return counterAPI{
		profiler: func(p *cupti.ProfilerInitializeParams) error {
			*calls = append(*calls, "profiler")
			if want := uint(unsafe.Offsetof(p.PPriv) + unsafe.Sizeof(p.PPriv)); p.StructSize != want {
				t.Errorf("profiler StructSize = %d, want %d", p.StructSize, want)
			}
			return profilerErr
		},
		enable: func(p *cupti.RangeProfilerEnableParams) error {
			*calls = append(*calls, "enable")
			p.PRangeProfilerObject = 7
			return nil
		},
		size: func(p *cupti.RangeProfilerGetCounterDataSizeParams) error {
			*calls = append(*calls, "size")
			p.CounterDataSize = 32
			return nil
		},
		initialize: func(*cupti.RangeProfilerCounterDataImageInitializeParams) error {
			*calls = append(*calls, "initialize")
			return nil
		},
		disable: func(*cupti.RangeProfilerDisableParams) error {
			*calls = append(*calls, "disable")
			return nil
		},
	}
}

func TestCounterStoreInitializesProfilerBeforeEnable(t *testing.T) {
	var calls []string
	c, err := newCounterStore(1, []string{"metric"}, 4, 8, 64, recordingAPI(t, &calls, nil))
	if err != nil {
		t.Fatal(err)
	}
	if err := c.close(); err != nil {
		t.Fatal(err)
	}
	want := []string{"profiler", "enable", "size", "initialize", "disable"}
	if !slices.Equal(calls, want) {
		t.Fatalf("calls = %v, want %v", calls, want)
	}
}

func TestCounterStoreProfilerInitializeFailure(t *testing.T) {
	failure := errors.New("not initialized")
	var calls []string
	c, err := newCounterStore(1, []string{"metric"}, 4, 8, 64, recordingAPI(t, &calls, failure))
	if c != nil || !errors.Is(err, failure) {
		t.Fatalf("store=%v error=%v", c, err)
	}
	if !slices.Equal(calls, []string{"profiler"}) {
		t.Fatalf("calls = %v, want only profiler: nothing to enable or disable", calls)
	}
}

func TestCounterStoreRequiresProfilerInitializer(t *testing.T) {
	var calls []string
	api := recordingAPI(t, &calls, nil)
	api.profiler = nil
	c, err := newCounterStore(1, []string{"metric"}, 4, 8, 64, api)
	if c != nil || err == nil {
		t.Fatalf("store=%v error=%v", c, err)
	}
	if len(calls) != 0 {
		t.Fatalf("calls = %v, want none", calls)
	}
}
