//go:build linux

package cuptirange_test

import (
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cuptirange"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func TestNativeRangeCapture(t *testing.T) {
	if os.Getenv("CUPTI_RANGE_TEST") != "1" {
		t.Skip("CUPTI_RANGE_TEST=1 required")
	}
	chip := os.Getenv("CUPTI_CHIP")
	if chip == "" {
		t.Fatal("CUPTI_CHIP must identify the qualified device chip")
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
	}()
	cm, err := nvrtcx.Compile(`extern "C" __global__ void measured(unsigned* out) { out[0] = 7; }`, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatal(err)
	}
	module, err := cm.Load(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := dev.WithContext(module.Unload); err != nil {
			t.Error(err)
		}
	}()
	var fn driver.Function
	err = dev.WithContext(func() error {
		var err error
		fn, err = module.GetFunction("measured")
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	output, err := cudamem.Alloc[uint32](dev, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := output.Free(); err != nil {
			t.Error(err)
		}
	}()
	plan, err := cuptirange.NewPlan(chip, nil, []string{"sm__cycles_elapsed.avg"}, 16<<20)
	if plan != nil {
		defer func() {
			if err := plan.Close(); err != nil {
				t.Error(err)
			}
		}()
	}
	if err != nil {
		t.Fatal(err)
	}
	image, passes := plan.Configuration()
	if len(image) == 0 || passes == 0 {
		t.Fatal("empty profiling configuration")
	}
	launches := uint(0)
	capture, err := plan.Capture(dev, cuptirange.Limits{MaxRanges: 4, MaxRangeTreeNodes: 8, MaxCounterBytes: 16 << 20, MaxPasses: 16}, func(pass uint) error {
		if pass != launches {
			t.Errorf("pass=%d expected %d", pass, launches)
		}
		launches++
		return cudalaunch.Kernel(fn).Launch(output)
	})
	if capture != nil {
		defer func() {
			if err := capture.Close(); err != nil {
				t.Error(err)
			}
		}()
	}
	if err != nil {
		t.Fatal(err)
	}
	got := make([]uint32, 1)
	if err := output.CopyToHost(got); err != nil {
		t.Fatal(err)
	}
	if got[0] != 7 || launches == 0 || capture.Ranges != 1 {
		t.Fatalf("output=%d launches=%d ranges=%d", got[0], launches, capture.Ranges)
	}
	values, err := plan.Evaluate(capture.Image, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(values) != 1 || values[0] <= 0 {
		t.Fatalf("invalid cycle metric %v", values)
	}
	t.Logf("chip=%s passes=%d launches=%d ranges=%d cycles=%g; no timing comparison", chip, passes, launches, capture.Ranges, values[0])
}
