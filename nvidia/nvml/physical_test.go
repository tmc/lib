//go:build nvml_physical

// Package nvml_test cross-checks the generated NVML bindings against two
// independent oracles: the nvidia-smi command line tool running on the same
// host, and the CUDA driver API. NVML is unusually exposed to layout defects
// because most of its interesting results arrive through version-tagged
// structs, so several assertions here read a field whose offset would shift if
// the Go struct did not match the C one.
//
// Set NVML_PHYSICAL_PERTURB to the name of an assertion to run the negative
// control for it; that test must then fail.
package nvml_test

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvml"
)

// unsafeSizeofMemoryV2 reports the size of the Go struct, which must equal the
// size the NVML version stamp encodes for nvmlMemory_v2_t.
func unsafeSizeofMemoryV2() uintptr { return unsafe.Sizeof(nvml.MemoryV2{}) }

func perturb(name string) bool { return os.Getenv("NVML_PHYSICAL_PERTURB") == name }

// smi queries nvidia-smi for one property of device 0 and returns the trimmed
// value. It is the independent oracle: a separate process, using NVIDIA's own
// code path, with no dependency on the bindings under test.
func smi(t *testing.T, query string) string {
	t.Helper()
	cmd := exec.Command("nvidia-smi", "--id=0",
		"--query-gpu="+query, "--format=csv,noheader,nounits")
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		t.Skipf("nvidia-smi %s: %v: %s", query, err, stderr.String())
	}
	return strings.TrimSpace(string(out))
}

func smiInt(t *testing.T, query string) int64 {
	t.Helper()
	v, err := strconv.ParseInt(smi(t, query), 10, 64)
	if err != nil {
		t.Fatalf("parse nvidia-smi %s: %v", query, err)
	}
	return v
}

func smiFloat(t *testing.T, query string) float64 {
	t.Helper()
	v, err := strconv.ParseFloat(smi(t, query), 64)
	if err != nil {
		t.Fatalf("parse nvidia-smi %s: %v", query, err)
	}
	return v
}

// device0 initialises NVML and returns a handle to device 0.
func device0(t *testing.T) nvml.Device {
	t.Helper()
	if err := nvml.Init(); err != nil {
		t.Skipf("NVML unavailable: %v", err)
	}
	t.Cleanup(func() {
		if err := nvml.Shutdown(); err != nil {
			t.Errorf("shutdown: %v", err)
		}
	})
	dev, err := nvml.NewDeviceByIndex(0)
	if err != nil {
		t.Skipf("no NVML device 0: %v", err)
	}
	return dev
}

// cstring returns the NUL-terminated string held in a fixed-size C char array.
func cstring(b []byte) string {
	if i := bytes.IndexByte(b, 0); i >= 0 {
		b = b[:i]
	}
	return string(b)
}

// TestIdentityAgainstSmi compares the string-valued device identity that NVML
// reports with what nvidia-smi reports for the same device. These pass through
// caller-allocated char buffers whose length is part of the ABI, so a wrong
// buffer size shows up as truncation.
func TestIdentityAgainstSmi(t *testing.T) {
	dev := device0(t)

	count, err := nvml.DeviceGetCount()
	if err != nil {
		t.Fatalf("DeviceGetCount: %v", err)
	}
	if want := smiInt(t, "count"); int64(count) != want {
		t.Fatalf("DeviceGetCount = %d, want %d (nvidia-smi)", count, want)
	}
	t.Logf("device count %d matches nvidia-smi", count)

	for _, tc := range []struct {
		name  string
		query string
		get   func() (string, error)
	}{
		{"Name", "name", dev.GetName},
		{"UUID", "uuid", dev.GetUUID},
		{"Serial", "serial", dev.GetSerial},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tc.get()
			if err != nil {
				t.Skipf("%s unsupported: %v", tc.name, err)
			}
			want := smi(t, tc.query)
			if perturb("identity-" + tc.query) {
				want += "x"
			}
			if got != want {
				t.Fatalf("%s = %q, want %q (nvidia-smi)", tc.name, got, want)
			}
			t.Logf("%s = %q matches nvidia-smi", tc.name, got)
		})
	}

	t.Run("DriverVersion", func(t *testing.T) {
		got, err := nvml.SystemGetDriverVersion()
		if err != nil {
			t.Fatalf("SystemGetDriverVersion: %v", err)
		}
		want := smi(t, "driver_version")
		if perturb("driver-version") {
			want += "x"
		}
		if got != want {
			t.Fatalf("SystemGetDriverVersion = %q, want %q", got, want)
		}
		t.Logf("driver version %q matches nvidia-smi", got)
	})
}

// TestPciInfoLayout reads PciInfo.BusId, which sits at byte offset 36 of
// nvmlPciInfo_t, after a 16-byte legacy bus id and five 32-bit integers. If
// any preceding field had the wrong width the string read out of BusId would
// not match the bus id nvidia-smi prints. The numeric domain, bus and device
// fields are then cross-checked against the same string, so the test pins both
// the offset and the integer widths.
func TestPciInfoLayout(t *testing.T) {
	dev := device0(t)
	info, err := dev.GetPciInfo()
	if err != nil {
		t.Fatalf("GetPciInfo: %v", err)
	}
	busID := cstring(info.BusId[:])
	want := smi(t, "pci.bus_id")
	if perturb("pci-busid") {
		want = strings.Replace(want, ":00:", ":01:", 1)
	}
	if !strings.EqualFold(busID, want) {
		t.Fatalf("PciInfo.BusId = %q, want %q (nvidia-smi)", busID, want)
	}

	// nvidia-smi formats the bus id as DOMAIN:BUS:DEVICE.FUNCTION in hex.
	fields := strings.FieldsFunc(want, func(r rune) bool { return r == ':' || r == '.' })
	if len(fields) != 4 {
		t.Fatalf("unexpected bus id shape %q", want)
	}
	parse := func(s string) uint64 {
		v, err := strconv.ParseUint(s, 16, 64)
		if err != nil {
			t.Fatalf("parse %q: %v", s, err)
		}
		return v
	}
	for _, f := range []struct {
		name string
		got  uint32
		want uint64
	}{
		{"Domain", info.Domain, parse(fields[0])},
		{"Bus", info.Bus, parse(fields[1])},
		{"Device", info.Device, parse(fields[2])},
	} {
		if uint64(f.got) != f.want {
			t.Fatalf("PciInfo.%s = %d, want %d (from bus id %q)", f.name, f.got, f.want, want)
		}
	}
	legacy := cstring(info.BusIdLegacy[:])
	t.Logf("PciInfo BusId=%q legacy=%q domain=%d bus=%d device=%d pciDeviceId=%#08x",
		busID, legacy, info.Domain, info.Bus, info.Device, info.PciDeviceId)
	// The low half of pciDeviceId is the PCI vendor id; NVIDIA is 0x10de.
	if vendor := info.PciDeviceId & 0xffff; vendor != 0x10de {
		t.Fatalf("PciInfo.PciDeviceId vendor = %#04x, want 0x10de", vendor)
	}
}

// TestMemoryInfoLayout checks nvmlMemory_v2_t three ways: the version stamp the
// binding writes must survive the call, the four size fields must be
// internally consistent, and Total must agree both with nvidia-smi and with
// the CUDA driver's own total. Field order in this struct is
// version/total/reserved/free/used, which is not the order the v1 struct used,
// so a generator that reused the v1 layout would fail the consistency check.
func TestMemoryInfoLayout(t *testing.T) {
	dev := device0(t)
	mem, err := dev.GetMemoryInfo()
	if errors.Is(err, nvml.ErrNotSupported) {
		// Devices with unified host memory, such as GB10, report no
		// separate framebuffer; nvidia-smi prints "Not Supported" for
		// them too, so there is no oracle to check against here.
		t.Skipf("GetMemoryInfo unsupported on this device: %v", err)
	}
	if err != nil {
		t.Fatalf("GetMemoryInfo: %v", err)
	}
	t.Logf("MemoryV2 version=%#x total=%d reserved=%d free=%d used=%d",
		mem.Version, mem.Total, mem.Reserved, mem.Free, mem.Used)

	if mem.Version != nvml.MemoryV2Version {
		t.Fatalf("MemoryV2.Version = %#x, want %#x", mem.Version, nvml.MemoryV2Version)
	}
	// The version stamp encodes sizeof(nvmlMemory_v2_t) in its low bytes.
	if want := uint32(unsafeSizeofMemoryV2()); nvml.MemoryV2Version&0xffffff != want {
		t.Fatalf("sizeof(MemoryV2) = %d, but version stamp encodes %d",
			want, nvml.MemoryV2Version&0xffffff)
	}
	if got, want := mem.Free+mem.Used+mem.Reserved, mem.Total; got != want {
		t.Fatalf("free+used+reserved = %d, want total = %d (field order or width is wrong)", got, want)
	}

	totalMiB := mem.Total >> 20
	if perturb("memory-total") {
		totalMiB++
	}
	if want := uint64(smiInt(t, "memory.total")); totalMiB != want {
		t.Fatalf("NVML total = %d MiB, want %d MiB (nvidia-smi)", totalMiB, want)
	}
	t.Logf("NVML total %d MiB matches nvidia-smi", totalMiB)

	// Second oracle: the CUDA driver. cuDeviceTotalMem reports the memory
	// available to CUDA, which excludes the block NVML accounts as reserved,
	// so the exact identity across the two libraries is
	//	driver total == NVML total - NVML reserved.
	if err := driver.Init(0); err != nil {
		t.Skipf("CUDA driver unavailable: %v", err)
	}
	cudev, err := driver.NewDeviceGet(0)
	if err != nil {
		t.Skipf("no CUDA device 0: %v", err)
	}
	var driverTotal uint
	if err := cudev.TotalMem(&driverTotal); err != nil {
		t.Fatalf("driver TotalMem: %v", err)
	}
	if want := mem.Total - mem.Reserved; uint64(driverTotal) != want {
		t.Fatalf("driver total = %d bytes, NVML total-reserved = %d bytes", driverTotal, want)
	}
	t.Logf("driver total %d == NVML total %d - reserved %d, exactly",
		driverTotal, mem.Total, mem.Reserved)

	// On an idle GPU Used is zero, which makes Total-Reserved coincide with
	// Free. Allocating pins down which of the two the driver total tracks: after
	// the allocation Free must drop by at least the request while Total,
	// Reserved and the driver total are unchanged.
	ctx, err := cudev.PrimaryCtxRetain()
	if err != nil {
		t.Skipf("retain primary context: %v", err)
	}
	defer func() {
		if err := cudev.PrimaryCtxRelease(); err != nil {
			t.Errorf("release context: %v", err)
		}
	}()
	if err := ctx.SetCurrent(); err != nil {
		t.Fatalf("set current context: %v", err)
	}
	const request = 256 << 20
	ptr, err := driver.MemAlloc(request)
	if err != nil {
		t.Fatalf("MemAlloc: %v", err)
	}
	defer func() {
		if err := ptr.MemFree(); err != nil {
			t.Errorf("free device buffer: %v", err)
		}
	}()
	after, err := dev.GetMemoryInfo()
	if err != nil {
		t.Fatalf("GetMemoryInfo after alloc: %v", err)
	}
	t.Logf("after allocating %d MiB: total=%d reserved=%d free=%d used=%d",
		request>>20, after.Total, after.Reserved, after.Free, after.Used)
	if after.Total != mem.Total || after.Reserved != mem.Reserved {
		t.Fatalf("total/reserved moved across an allocation: %d/%d then %d/%d",
			mem.Total, mem.Reserved, after.Total, after.Reserved)
	}
	if after.Used < request {
		t.Fatalf("Used = %d after allocating %d bytes, want at least the request", after.Used, request)
	}
	if mem.Free-after.Free < request {
		t.Fatalf("Free fell by %d after allocating %d bytes", mem.Free-after.Free, request)
	}
	if got, want := after.Free+after.Used+after.Reserved, after.Total; got != want {
		t.Fatalf("free+used+reserved = %d, want total = %d, after allocation", got, want)
	}
	var driverTotalAfter uint
	if err := cudev.TotalMem(&driverTotalAfter); err != nil {
		t.Fatalf("driver TotalMem after alloc: %v", err)
	}
	if driverTotalAfter != driverTotal {
		t.Fatalf("driver total moved from %d to %d across an allocation", driverTotal, driverTotalAfter)
	}
	t.Logf("Used tracks the allocation and total/reserved do not; the identity is not a coincidence of an idle GPU")
}

// TestTelemetryAgainstSmi checks the scalar readings. Temperature and power
// move between the two samples, so these use a band rather than equality; the
// band is still tight enough that a wrong scale factor or a truncated 32-bit
// read would fall outside it.
func TestTelemetryAgainstSmi(t *testing.T) {
	dev := device0(t)

	t.Run("Temperature", func(t *testing.T) {
		got, err := dev.GetTemperature(nvml.TemperatureGpu)
		if err != nil {
			t.Fatalf("GetTemperature: %v", err)
		}
		want := smiFloat(t, "temperature.gpu")
		if perturb("temperature") {
			want += 50
		}
		if d := float64(got) - want; d < -5 || d > 5 {
			t.Fatalf("GetTemperature = %d C, nvidia-smi = %v C", got, want)
		}
		t.Logf("GetTemperature = %d C, nvidia-smi = %v C", got, want)
	})

	t.Run("PowerUsage", func(t *testing.T) {
		got, err := dev.GetPowerUsage()
		if err != nil {
			t.Skipf("GetPowerUsage unsupported: %v", err)
		}
		// NVML reports milliwatts; nvidia-smi reports watts.
		watts := float64(got) / 1000
		want := smiFloat(t, "power.draw")
		if perturb("power") {
			want *= 10
		}
		if d := watts - want; d < -25 || d > 25 {
			t.Fatalf("GetPowerUsage = %v W (%d mW), nvidia-smi = %v W", watts, got, want)
		}
		t.Logf("GetPowerUsage = %v W (%d mW), nvidia-smi = %v W", watts, got, want)
	})

	t.Run("MaxSmClock", func(t *testing.T) {
		got, err := dev.GetMaxClockInfo(nvml.ClockSm)
		if err != nil {
			t.Fatalf("GetMaxClockInfo: %v", err)
		}
		want := smiInt(t, "clocks.max.sm")
		if perturb("max-sm-clock") {
			want++
		}
		if int64(got) != want {
			t.Fatalf("GetMaxClockInfo(ClockSm) = %d MHz, want %d MHz (nvidia-smi)", got, want)
		}
		t.Logf("GetMaxClockInfo(ClockSm) = %d MHz matches nvidia-smi exactly", got)
	})

	t.Run("Utilization", func(t *testing.T) {
		util, err := dev.GetUtilizationRates()
		if err != nil {
			t.Fatalf("GetUtilizationRates: %v", err)
		}
		// Both fields are percentages; a swapped or misaligned field would
		// commonly land outside the range.
		if util.Gpu > 100 || util.Memory > 100 {
			t.Fatalf("Utilization = %+v, both fields must be percentages", util)
		}
		t.Logf("Utilization gpu=%d%% memory=%d%%", util.Gpu, util.Memory)
	})
}

// TestComputeCapability exercises a binding whose C function has two
// out-parameters but whose Go signature returns one and takes the other as a
// pointer. Both halves are checked against nvidia-smi and against the CUDA
// driver, so a swap between major and minor cannot pass.
func TestComputeCapability(t *testing.T) {
	dev := device0(t)
	var minor int32
	major, err := dev.GetCudaComputeCapability(&minor)
	if err != nil {
		t.Fatalf("GetCudaComputeCapability: %v", err)
	}
	got := fmt.Sprintf("%d.%d", major, minor)
	want := smi(t, "compute_cap")
	if perturb("compute-cap") {
		got = fmt.Sprintf("%d.%d", minor, major)
	}
	if got != want {
		t.Fatalf("compute capability = %s, want %s (nvidia-smi)", got, want)
	}
	t.Logf("compute capability %s matches nvidia-smi", got)

	if err := driver.Init(0); err != nil {
		t.Skipf("CUDA driver unavailable: %v", err)
	}
	cudev, err := driver.NewDeviceGet(0)
	if err != nil {
		t.Skipf("no CUDA device 0: %v", err)
	}
	var dmajor, dminor int32
	if err := cudev.ComputeCapability(&dmajor, &dminor); err != nil {
		t.Fatalf("driver ComputeCapability: %v", err)
	}
	if dmajor != major || dminor != minor {
		t.Fatalf("driver reports %d.%d, NVML reports %d.%d", dmajor, dminor, major, minor)
	}
	t.Logf("driver API agrees: %d.%d", dmajor, dminor)
}
