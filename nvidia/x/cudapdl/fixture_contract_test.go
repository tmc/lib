package cudapdl_test

import (
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudapdl"
)

type pdlFixturePointer driver.DevicePtr

func (p pdlFixturePointer) Ptr() driver.DevicePtr { return driver.DevicePtr(p) }

// These are the builder, descriptions, and packed arguments used by the
// gated native test. No graph is created and no driver/compiler call is made.
func TestPDLReferenceFixture(t *testing.T) {
	driverLoaded, compilerLoaded := driver.Loaded(), nvrtc.Loaded()
	f := newPDLReferenceFixture("sm_90")
	if f.producer.Name != "pdl_producer" || f.producer.Source != nativePDLProducerSource ||
		f.consumer.Name != "pdl_consumer" || f.consumer.Source != nativePDLConsumerSource {
		t.Fatal("fixture does not name the sources and entry points compiled by the native test")
	}
	if !f.producer.HasLaunchDependents || !f.consumer.HasWait || !f.consumer.HasIndependentWork {
		t.Fatal("fixture lost its explicit owner declarations")
	}
	wantArms := [2]pdlReferenceArm{
		{"ordinary", false, cudapdl.PolicyStandardFallback},
		{"programmatic", true, cudapdl.PolicyProgrammatic},
	}
	if f.arms != wantArms {
		t.Fatalf("native fixture arms=%v, want %v", f.arms, wantArms)
	}
	for _, arm := range f.arms {
		el := f.eligibility(arm.programmatic)
		if arm.programmatic {
			if !el.Eligible || el.Policy != cudapdl.PolicyProgrammatic || el.FallbackReason != "" {
				t.Fatalf("programmatic arm planning: %+v", el)
			}
		} else if el.Eligible || el.Policy != cudapdl.PolicyStandardFallback || el.FallbackReason != "ordinary reference arm" {
			t.Fatalf("ordinary arm planning: %+v", el)
		}
	}
	input, intermediate, output := pdlFixturePointer(0x1000), pdlFixturePointer(0x2000), pdlFixturePointer(0x3000)
	launches, err := f.launches(cudalaunch.Kernel(11), cudalaunch.Kernel(22), input, intermediate, output)
	if err != nil {
		t.Fatal(err)
	}
	for i, launch := range launches {
		if launch.desc.Function != []driver.Function{11, 22}[i] || launch.desc.Grid != driver.Dims(2, 1, 1) || launch.desc.Block != driver.Dims(128, 1, 1) {
			t.Fatalf("kernel %d description: %+v", i, launch.desc)
		}
		if len(launch.desc.Attrs) != 0 || launch.desc.SharedMem != 0 {
			t.Fatalf("kernel %d carries configuration Graph.AddLaunch cannot express: %+v", i, launch.desc)
		}
	}
	wantArgs := [2][]any{
		{input, intermediate, int32(256), int64(1000)},
		{intermediate, output, float32(3), int32(256), int64(1000)},
	}
	for i, launch := range launches {
		if !reflect.DeepEqual(launch.args, wantArgs[i]) {
			t.Fatalf("kernel %d args=%v, want %v", i, launch.args, wantArgs[i])
		}
		packed, err := cudalaunch.PackArgs(launch.args...)
		if err != nil {
			t.Fatal(err)
		}
		if len(packed) != len(wantArgs[i]) {
			t.Fatalf("kernel %d packed %d parameters, want %d", i, len(packed), len(wantArgs[i]))
		}
		for j, arg := range wantArgs[i] {
			var got any
			switch arg.(type) {
			case pdlFixturePointer:
				got = pdlFixturePointer(*(*driver.DevicePtr)(packed[j]))
			case int32:
				got = *(*int32)(packed[j])
			case int64:
				got = *(*int64)(packed[j])
			case float32:
				got = *(*float32)(packed[j])
			}
			if got != arg {
				t.Fatalf("kernel %d packed arg %d=%v, want %v", i, j, got, arg)
			}
		}
	}
	if driver.Loaded() != driverLoaded || nvrtc.Loaded() != compilerLoaded {
		t.Fatal("fixture construction loaded a vendor library")
	}
}

func TestPDLReferenceFixtureRejectsMissingFunction(t *testing.T) {
	for _, tt := range []struct {
		name               string
		producer, consumer *cudalaunch.Launcher
	}{
		{"nil producer", nil, cudalaunch.Kernel(22)},
		{"zero producer", cudalaunch.Kernel(0), cudalaunch.Kernel(22)},
		{"nil consumer", cudalaunch.Kernel(11), nil},
		{"zero consumer", cudalaunch.Kernel(11), cudalaunch.Kernel(0)},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := newPDLReferenceFixture("sm_90").launches(tt.producer, tt.consumer, 1, 2, 3); err == nil {
				t.Fatal("fixture accepted a missing function")
			}
		})
	}
}

func TestPDLReferencePredicates(t *testing.T) {
	for _, tt := range []struct {
		name                         string
		major, minor, driver         int
		typed                        bool
		compilerMajor, compilerMinor int32
		archs                        []int32
		want                         string
	}{
		{"supported SM90", 9, 0, 12030, true, 12, 3, []int32{80, 90}, ""},
		{"supported SM121", 12, 1, 13000, true, 13, 0, []int32{90, 100, 121}, ""},
		{"old device", 8, 9, 13000, true, 13, 0, []int32{89}, "device requires SM90 or higher"},
		{"old driver", 9, 0, 12020, true, 13, 0, []int32{90}, "driver requires CUDA 12.3 or higher and cuGraphAddDependencies_v2"},
		{"missing typed edge API", 9, 0, 13000, false, 13, 0, []int32{90}, "driver requires CUDA 12.3 or higher and cuGraphAddDependencies_v2"},
		{"old compiler", 9, 0, 13000, true, 12, 2, []int32{90}, "fixture requires NVRTC 12.3 or higher"},
		{"compiler lacks target", 12, 1, 13000, true, 13, 0, []int32{90, 120}, "NVRTC does not list the device's physical architecture"},
		{"empty target list", 9, 0, 13000, true, 13, 0, nil, "NVRTC does not list the device's physical architecture"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got := pdlReferenceUnsupported(tt.major, tt.minor, tt.driver, tt.typed, tt.compilerMajor, tt.compilerMinor, tt.archs)
			if got != tt.want {
				t.Fatalf("predicate=%q, want %q", got, tt.want)
			}
		})
	}
}

func TestPDLReferenceInputs(t *testing.T) {
	input, want := pdlReferenceInputs()
	if len(input) != 256 || len(want) != 256 {
		t.Fatalf("input/output lengths=%d/%d, want 256/256", len(input), len(want))
	}
	for _, tt := range []struct {
		i           int
		input, want float32
	}{{0, -32, -61}, {127, -0.25, 2.5}, {128, 0, 3}, {255, 31.75, 66.5}} {
		if input[tt.i] != tt.input || want[tt.i] != tt.want {
			t.Fatalf("reference[%d]=(%g,%g), want (%g,%g)", tt.i, input[tt.i], want[tt.i], tt.input, tt.want)
		}
	}
	for i := range input {
		if want[i] != input[i]*2+3 {
			t.Fatalf("reference[%d]=%g for input %g", i, want[i], input[i])
		}
	}
}
