//go:build linux && cupti_physical

package cuptisass_test

import (
	"runtime"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuptisass"
)

func withContext(t *testing.T) uintptr {
	t.Helper()
	runtime.LockOSThread()
	t.Cleanup(runtime.UnlockOSThread)
	if err := driver.Init(0); err != nil {
		t.Skipf("CUDA driver unavailable: %v", err)
	}
	dev, err := driver.NewDeviceGet(0)
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	ctx, err := dev.PrimaryCtxRetain()
	if err != nil {
		t.Fatalf("retain primary context: %v", err)
	}
	t.Cleanup(func() {
		if err := dev.PrimaryCtxRelease(); err != nil {
			t.Errorf("release context: %v", err)
		}
	})
	if err := ctx.SetCurrent(); err != nil {
		t.Fatalf("set current context: %v", err)
	}
	return uintptr(ctx)
}

func TestPhysicalDiscovery(t *testing.T) {
	_ = withContext(t)

	if err := cuptisass.CheckVersion(); err != nil {
		t.Skipf("CUPTI unavailable or unqualified version: %v", err)
	}

	chip, err := cuptisass.DeviceChipName(0)
	if err != nil {
		t.Fatalf("DeviceChipName: %v", err)
	}
	t.Logf("Device 0 chip: %s", chip)

	metrics, err := cuptisass.SupportedMetrics(chip)
	if err != nil {
		t.Fatalf("SupportedMetrics: %v", err)
	}
	if len(metrics) == 0 {
		t.Fatalf("SupportedMetrics returned 0 metrics for chip %s", chip)
	}
	t.Logf("Discovered %d SASS metrics on chip %s", len(metrics), chip)

	// Check a specific metric property
	metricName := metrics[0].Name
	prop, err := cuptisass.MetricProperties(chip, metricName)
	if err != nil {
		t.Fatalf("MetricProperties(%s): %v", metricName, err)
	}
	if prop.ID == 0 || prop.Name != metricName {
		t.Errorf("unexpected metric properties: %+v", prop)
	}
	t.Logf("Metric %s ID: %d, Description: %s", prop.Name, prop.ID, prop.Description)
}

func TestPhysicalSessionLifecycle(t *testing.T) {
	ctx := withContext(t)

	if err := cuptisass.CheckVersion(); err != nil {
		t.Skipf("CUPTI unavailable or unqualified version: %v", err)
	}

	chip, err := cuptisass.DeviceChipName(0)
	if err != nil {
		t.Fatalf("DeviceChipName: %v", err)
	}

	prop, err := cuptisass.MetricProperties(chip, "smsp__sass_inst_executed")
	if err != nil {
		t.Skipf("metric smsp__sass_inst_executed not available on %s: %v", chip, err)
	}

	cfg := cuptisass.Config{
		DeviceIndex:  0,
		LazyPatching: true,
		Metrics: []cuptisass.MetricConfig{
			{
				MetricID:    prop.ID,
				Granularity: cuptisass.GranularityGPU,
			},
		},
	}

	sess, err := cuptisass.NewSession(ctx, cfg)
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer func() {
		if err := sess.Close(); err != nil {
			t.Errorf("sess.Close: %v", err)
		}
	}()

	if err := sess.Enable(); err != nil {
		t.Fatalf("sess.Enable: %v", err)
	}

	// In an empty session with no kernel launches, flushed data should be empty
	records, err := sess.Flush()
	if err != nil {
		t.Fatalf("sess.Flush: %v", err)
	}
	if len(records) != 0 {
		t.Errorf("expected 0 records before kernel execution, got %d", len(records))
	}

	dropped, err := sess.Disable()
	if err != nil {
		t.Fatalf("sess.Disable: %v", err)
	}
	t.Logf("sess.Disable returned %d dropped records", dropped)
}
