package cudagreen

import (
	"fmt"
	"os"
	"runtime"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalock"
)

// TestNativeOpen creates the green contexts for the prefill-heavy and
// decode-heavy plans on device 0 and checks each reports its planned SM
// count. It runs only with CUDA_GREEN_NATIVE=1 and an inherited GPU lock.
func TestNativeOpen(t *testing.T) {
	if os.Getenv("CUDA_GREEN_NATIVE") != "1" {
		t.Skip("set CUDA_GREEN_NATIVE=1 to create green contexts on device 0")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatalf("native gate open without inherited gpu lock: %v", err)
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := driver.Init(0); err != nil {
		t.Fatal(err)
	}
	v, _ := driver.DriverGetVersion()
	t.Logf("driver version %d", v)
	if err := Supported(); err != nil {
		t.Fatal(err)
	}
	dev, err := driver.DeviceGet(0)
	if err != nil {
		t.Fatal(err)
	}
	major, err := dev.Attribute(driver.DevAttrComputeCapabilityMajor)
	if err != nil {
		t.Fatal(err)
	}
	minor, err := dev.Attribute(driver.DevAttrComputeCapabilityMinor)
	if err != nil {
		t.Fatal(err)
	}
	arch := fmt.Sprintf("sm_%d%d", major, minor)
	total, g, err := DeviceGranularity(dev)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%s: %d SMs, reported granularity %+v", arch, total, g)
	if g.Min == 0 || g.Alignment == 0 {
		if g, err = DefaultGranularity(arch); err != nil {
			t.Fatal(err)
		}
	}
	g.Alignment = roundUp(g.Min, g.Alignment)
	g.Min = g.Alignment
	for _, regime := range []struct {
		name  string
		specs []Spec
	}{
		{"prefill", PrefillHeavy()},
		{"decode", DecodeHeavy()},
	} {
		t.Run(regime.name, func(t *testing.T) {
			plan, err := ComputePlan(arch, total, g, regime.specs)
			if err != nil {
				t.Fatal(err)
			}
			p, err := Open(dev, plan)
			if err != nil {
				t.Fatal(err)
			}
			defer p.Close()
			for i, q := range plan.Partitions {
				t.Logf("%v: %d SMs", q.Spec.Role, q.SMs)
				if err := p.Stream(i).Synchronize(); err != nil {
					t.Errorf("%v stream: %v", q.Spec.Role, err)
				}
			}
			if err := p.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}
