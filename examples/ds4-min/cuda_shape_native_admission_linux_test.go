//go:build linux

package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
)

const shapeNativeDriver = "/usr/lib/aarch64-linux-gnu/libcuda.so.580.173.02"
const shapeNativeNVRTC = "/usr/local/cuda-13.0/targets/sbsa-linux/lib/libnvrtc.so"

var shapeNativeLibraryPins = map[string]string{
	shapeNativeDriver: "a13f3101b105930ba0796b1ab99721cbee3e19360ced21253e2a09e1a102fb99",
	shapeNativeNVRTC:  "64db9c3fc128f4a57a22c314fda08831a8e3a725d439a70076a0fb98f7bea1b7",
	"/usr/local/cuda-13.0/targets/sbsa-linux/lib/libnvrtc-builtins.so": "84f1c5031bfe1146375b367b2c7fb8d31ea00430be7172177eeb5d04ab813b69",
	"/usr/local/cuda-13.0/include/nvrtc.h":                             "9e4415b5ff5a2c58ec2b1a02eefcc743187c94ce48b00743c68092efe8c0f86e",
}

func shapeNativeCheckFiles(t *testing.T) {
	t.Helper()
	for path, want := range shapeNativeLibraryPins {
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if got := fmt.Sprintf("%x", sha256.Sum256(b)); got != want {
			t.Fatalf("NATIVE_LIBRARY_SEAL path=%s got=%s want=%s", path, got, want)
		}
	}
}

// Checks are made in this process, rather than inferred from the earlier
// constructor-only receipt. NewDevice owns exactly one retained primary ref.
func shapeNativeDevice(t *testing.T, dir, prefix string, owners ...*shapeStateNativeOwner) *cuda.Device {
	t.Helper()
	shapeNativeCheckFiles(t)
	if os.Getenv("NVRTC_LIB_PATH") != shapeNativeNVRTC {
		t.Fatal("NATIVE_NVRTC_PATH")
	}
	if err := driver.Init(0); err != nil {
		t.Fatal(err)
	}
	if driver.Path() != shapeNativeDriver {
		t.Fatalf("NATIVE_DRIVER_PATH got=%s", driver.Path())
	}
	version, err := driver.DriverGetVersion()
	if err != nil {
		t.Fatal(err)
	}
	if version != 13000 {
		t.Fatalf("NATIVE_DRIVER_VERSION got=%d", version)
	}
	raw, err := driver.DeviceGet(0)
	if err != nil {
		t.Fatal(err)
	}
	uuid, err := raw.Uuid()
	if err != nil {
		t.Fatal(err)
	}
	pci, err := raw.PCIBusId()
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", uuid.Bytes) != "0b2649cdf5c3ee675b8eb18d60595e9d" || !strings.EqualFold(pci, "000f:01:00.0") {
		t.Fatal("NATIVE_DEVICE_IDENTITY")
	}
	flags, active, err := raw.PrimaryCtxGetState()
	if err != nil {
		t.Fatal(err)
	}
	if flags != 0 || active != 0 {
		t.Fatalf("NATIVE_CONTEXT_BEFORE flags=%d active=%d", flags, active)
	}
	d, err := cuda.NewDevice(0)
	var owner *shapeStateNativeOwner
	if len(owners) != 0 {
		owner = owners[0]
	}
	owner = shapeNativeDeviceOwner(d, err, owner)
	t.Cleanup(func() {
		if err := owner.Close(); err != nil {
			shapeStateNativeRetained = append(shapeStateNativeRetained, owner)
			t.Errorf("NATIVE_CONTEXT_RELEASE: %v", err)
			return
		}
		flags, active, err := raw.PrimaryCtxGetState()
		data, _ := json.MarshalIndent(struct {
			Flags, Active uint32
			Error         string
		}{flags, uint32(active), fmt.Sprint(err)}, "", "  ")
		if err := os.WriteFile(filepath.Join(dir, prefix+"-context-after.json"), data, 0600); err != nil {
			t.Errorf("context receipt: %v", err)
		}
		if err != nil || flags != 0 || active != 0 {
			t.Errorf("NATIVE_CONTEXT_AFTER flags=%d active=%d err=%v", flags, active, err)
		}
		shapeNativeCheckFiles(t)
	})
	if err != nil {
		t.Fatal(err)
	}
	if d == nil {
		t.Fatal("missing successful device owner")
	}
	major, minor, err := d.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	if major != 12 || minor != 1 {
		t.Fatalf("NATIVE_DEVICE_ARCH %d.%d", major, minor)
	}
	var free uint
	if err := d.WithContext(func() error { var err error; free, _, err = driver.MemGetInfo(); return err }); err != nil {
		t.Fatal(err)
	}
	if free < 64<<20 {
		t.Fatal("NATIVE_DEVICE_MEMORY below64MiB")
	}
	t.Logf("NATIVE_RESOURCE_PASS device_free_bytes=%d fixture_peak_bytes_below=2097152", free)
	data, _ := json.MarshalIndent(struct {
		Path          string
		Version       int
		UUID, PCI     string
		Flags, Active uint32
	}{driver.Path(), version, fmt.Sprintf("%x", uuid.Bytes), pci, flags, uint32(active)}, "", "  ")
	if err := os.WriteFile(filepath.Join(dir, prefix+"-context-before.json"), data, 0600); err != nil {
		t.Fatal(err)
	}
	t.Log("NATIVE_CONTEXT_PASS retain=1")
	return d
}

// This is called immediately after NewDevice, before any borrowed allocation.
// A nil device has no parent drain; an error-owned partial constructor drains
// its own resources through its io.Closer contract.
func shapeNativeDeviceOwner(d *cuda.Device, err error, owner *shapeStateNativeOwner) *shapeStateNativeOwner {
	if owner == nil {
		owner = new(shapeStateNativeOwner)
	}
	owner.retainError(err)
	owner.drain = func() error {
		if d == nil {
			return nil
		}
		return d.WithContext(driver.CtxSynchronize)
	}
	if d != nil {
		owner.parent = d.Close
	}
	return owner
}

func shapeNativeMappedCompiler(t *testing.T, dir, prefix string) {
	t.Helper()
	major, minor, err := nvrtc.Version()
	if err != nil {
		t.Fatal(err)
	}
	if major != 13 || minor != 0 || nvrtc.Path() != shapeNativeNVRTC {
		t.Fatal("NATIVE_LOADED_NVRTC")
	}
	maps, err := os.ReadFile("/proc/self/maps")
	if err != nil {
		t.Fatal(err)
	}
	qualified := map[string]bool{}
	for _, line := range strings.Split(string(maps), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 6 {
			continue
		}
		path := fields[5]
		if !strings.Contains(path, "libcuda") && !strings.Contains(path, "libnvrtc") {
			continue
		}
		switch path {
		case shapeNativeDriver, "/usr/local/cuda-13.0/targets/sbsa-linux/lib/libnvrtc.so.13.0.88", "/usr/local/cuda-13.0/targets/sbsa-linux/lib/libnvrtc-builtins.so.13.0.88":
			qualified[path] = true
		default:
			t.Fatalf("NATIVE_UNQUALIFIED_MAP %s", path)
		}
	}
	if len(qualified) != 3 {
		t.Fatalf("NATIVE_MAPPING_COUNT got=%d", len(qualified))
	}
	if err := os.WriteFile(filepath.Join(dir, prefix+"-maps.txt"), maps, 0600); err != nil {
		t.Fatal(err)
	}
	shapeNativeCheckFiles(t)
	t.Log("NATIVE_MAPPED_COMPILER_PASS")
}
