//go:build linux

package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
)

func TestCUDAProbeDeviceCUBIN(t *testing.T) {
	if os.Getenv("DS4_PROBE_DEVICE_CUBIN") != "1" {
		t.Skip("explicit device CUBIN probe qualification required")
	}
	cache := os.Getenv("DS4_PROBE_CACHE")
	resolved, err := os.UserCacheDir()
	if err != nil || !filepath.IsAbs(cache) || filepath.Clean(cache) != cache || resolved != cache {
		t.Fatalf("fresh task cache admission: configured=%q resolved=%q error=%v", cache, resolved, err)
	}
	entries, err := os.ReadDir(cache)
	if err != nil || len(entries) != 0 {
		t.Fatalf("NVRTC cache must exist and be empty: entries=%d error=%v", len(entries), err)
	}
	device, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	f := &cudaFoundation{device: device}
	t.Cleanup(func() {
		if err := f.Close(); err != nil {
			t.Error(err)
		}
	})
	f.major, f.minor, err = device.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	if f.major != 12 || f.minor != 1 {
		t.Fatalf("requires sm121, got sm%d%d", f.major, f.minor)
	}
	if err := f.probe(); err != nil {
		t.Fatal(err)
	}
	maps, err := os.ReadFile("/proc/self/maps")
	if err != nil {
		t.Fatal(err)
	}
	path := os.Getenv("DS4_PROBE_MAPS")
	if path == "" {
		t.Fatal("DS4_PROBE_MAPS required")
	}
	if err := os.WriteFile(path, maps, 0600); err != nil {
		t.Fatal(err)
	}
	t.Logf("actual production probe CUBIN sm%d%d numeric output=4, maps=%s, fresh cache=%s", f.major, f.minor, path, cache)
}
