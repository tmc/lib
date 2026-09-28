//go:build linux

package cuptirange_test

import (
	"fmt"
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
		if err := dev.WithContext(module.Close); err != nil {
			t.Error(err)
		}
	}()
	var fn driver.Function
	err = dev.WithContext(func() error {
		var err error
		fn, err = module.Function("measured")
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

func TestNativeRangeIsolation(t *testing.T) {
	if os.Getenv("CUPTI_RANGE_TEST") != "1" || os.Getenv("GPU_LOCK_HELD") == "" {
		t.Skip("native qualification pending: CUPTI_RANGE_TEST=1 and GPU_LOCK_HELD required")
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

	const src = `
extern "C" __global__ void light_kernel(unsigned int *out) {
    out[0] = 42;
}

extern "C" __global__ void heavy_kernel(unsigned int *out, unsigned int trips) {
    unsigned int v = 1;
    #pragma unroll 1
    for (unsigned int i = 0; i < trips; i++) {
        v = v * 1664525u + 1013904223u;
    }
    out[0] = v;
}
`
	cm, err := nvrtcx.Compile(src, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatal(err)
	}
	module, err := cm.Load(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := dev.WithContext(module.Close); err != nil {
			t.Error(err)
		}
	}()

	var fnLight, fnHeavy driver.Function
	err = dev.WithContext(func() error {
		var err error
		fnLight, err = module.Function("light_kernel")
		if err != nil {
			return err
		}
		fnHeavy, err = module.Function("heavy_kernel")
		return err
	})
	if err != nil {
		t.Fatal(err)
	}

	outLight, err := cudamem.Alloc[uint32](dev, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := outLight.Free(); err != nil {
			t.Error(err)
		}
	}()

	outHeavy, err := cudamem.Alloc[uint32](dev, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := outHeavy.Free(); err != nil {
			t.Error(err)
		}
	}()

	metricName := os.Getenv("CUPTI_METRIC")
	if metricName == "" {
		metricName = "sm__inst_executed.sum"
	}

	plan, err := cuptirange.NewPlan(chip, nil, []string{metricName}, 16<<20)
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

	runCapture := func(trips uint32) (heavyVal, lightVal float64, err error) {
		options := cuptirange.CaptureOptions{
			Mode:     cuptirange.Regions,
			MaxDepth: 4,
		}
		limits := cuptirange.Limits{
			MaxRanges:         8,
			MaxRangeTreeNodes: 16,
			MaxCounterBytes:   16 << 20,
			MaxPasses:         16,
		}
		replay := cuptirange.Replay{
			Synchronize: driver.CtxSynchronize,
			Run: func(pass uint, s *cuptirange.Scope) error {
				if err := s.Region("heavy", func() error {
					return cudalaunch.Kernel(fnHeavy).Launch(outHeavy, trips)
				}); err != nil {
					return err
				}
				return s.Region("light", func() error {
					return cudalaunch.Kernel(fnLight).Launch(outLight)
				})
			},
		}

		capture, err := plan.CaptureReplay(dev, limits, options, replay)
		if capture != nil {
			defer func() {
				if cErr := capture.Close(); cErr != nil {
					t.Error(cErr)
				}
			}()
		}
		if err != nil {
			return 0, 0, err
		}

		var heavyInfo, lightInfo *cuptirange.RangeInfo
		for i := range capture.RangeInfo {
			switch capture.RangeInfo[i].Name {
			case "heavy":
				heavyInfo = &capture.RangeInfo[i]
			case "light":
				lightInfo = &capture.RangeInfo[i]
			}
		}
		if heavyInfo == nil || lightInfo == nil {
			return 0, 0, fmt.Errorf("missing heavy or light range in range info: %+v", capture.RangeInfo)
		}

		vHeavy, err := plan.Evaluate(capture.Image, heavyInfo.Index)
		if err != nil {
			return 0, 0, fmt.Errorf("evaluate heavy: %w", err)
		}
		vLight, err := plan.Evaluate(capture.Image, lightInfo.Index)
		if err != nil {
			return 0, 0, fmt.Errorf("evaluate light: %w", err)
		}
		return vHeavy[0], vLight[0], nil
	}

	trips1 := uint32(10000)
	h1, l1, err := runCapture(trips1)
	if err != nil {
		t.Fatalf("capture 1: %v", err)
	}

	trips2 := uint32(20000)
	h2, l2, err := runCapture(trips2)
	if err != nil {
		t.Fatalf("capture 2: %v", err)
	}

	// 1. Heavy exceeds light by at least the trip count times static instructions (at least 1 per trip).
	if h1-l1 < float64(trips1) {
		t.Fatalf("heavy (%g) did not exceed light (%g) by at least trip count %d", h1, l1, trips1)
	}

	// 2. Heavy work scales with the loop when trips double.
	if h2 <= h1 {
		t.Fatalf("heavy count did not increase when trips doubled: h1=%g, h2=%g", h1, h2)
	}

	// 3. Light work does not change when the heavy range's trip count doubles.
	// Range isolation guarantees that the heavy loop work does not leak into the light range.
	if l1 != l2 {
		t.Fatalf("range isolation violated: light count changed from %g to %g when heavy doubled", l1, l2)
	}

	t.Logf("range isolation confirmed on %s (%s): heavy(%d)=%g heavy(%d)=%g light=%g",
		chip, metricName, trips1, h1, trips2, h2, l1)
}
