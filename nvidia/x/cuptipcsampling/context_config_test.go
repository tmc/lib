package cuptipcsampling

import (
	"errors"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

// testRetainingControlAPI models CUPTI's handling of the sampling data buffer:
// the address set by the last configuration of a context outlives disable,
// later enables use it, and every disable writes the record header through it.
// Enable or disable through an address the package has released fails the way
// cuptiPCSamplingEnable does on GB10.
func testRetainingControlAPI(outputs map[uintptr]*samplingBuffer) samplingControlAPI {
	api := testPublicControlAPI()
	set := api.set
	var configured uintptr
	api.set = func(p *cupti.PCSamplingConfigurationInfoParams) error {
		for _, a := range unsafe.Slice((*cupti.PCSamplingConfigurationInfo)(nativePointer(p.PPCSamplingConfigurationInfo)), int(p.NumAttributes)) {
			if a.AttributeType == cupti.PcSamplingConfigurationAttrTypeSamplingDataBuffer {
				configured = uintptr(a.AttributeData[0])
			}
		}
		return set(p)
	}
	check := func(op string) error {
		if configured == 0 {
			return nil
		}
		if b := outputs[configured]; b == nil || b.released {
			return errors.New(op + " through released sampling data buffer: CUPTI_ERROR_INVALID_OPERATION")
		}
		return nil
	}
	api.enable = func(*cupti.PCSamplingEnableParams) error { return check("enable") }
	api.disable = func(*cupti.PCSamplingDisableParams) error { return check("disable") }
	return api
}

// The autotune collector enables PC sampling for its reason inventory and then
// starts a session, once per candidate, on one context. Each enable after the
// first runs with the previous session's configuration.
func TestSamplingSessionsShareContext(t *testing.T) {
	outputs := make(map[uintptr]*samplingBuffer)
	api := testRetainingControlAPI(outputs)
	var first *samplingBuffer
	for i := range 3 {
		disable, err := enableSampling(5, api)
		if err != nil {
			t.Fatalf("session %d: enable for inventory: %v", i, err)
		}
		if err := disable(); err != nil {
			t.Fatalf("session %d: disable after inventory: %v", i, err)
		}
		s, err := startSession(5, testPublicConfig(), api, testPublicInventory, testPublicNames, testPublicEmptyQuery)
		if err != nil {
			t.Fatalf("session %d: start: %v", i, err)
		}
		out := s.core.control.config.output
		outputs[uintptr(unsafe.Pointer(&out.data))] = out
		if first == nil {
			first = out
		}
		if err := s.Stop(); err != nil {
			t.Fatalf("session %d: stop: %v", i, err)
		}
		if err := s.Close(); err != nil {
			t.Fatalf("session %d: close: %v", i, err)
		}
	}
	if !first.released {
		t.Error("superseded configuration still held")
	}
}
