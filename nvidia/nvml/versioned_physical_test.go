//go:build linux

package nvml_test

import (
	"os"
	"testing"

	"github.com/ebitengine/purego"
	"github.com/tmc/lib/nvidia/nvml"
)

// This package is generated from cuda-headers nvml/nvml.h, which does not
// declare the nine symbols the pinned cuda-13.3 header adds. Whether to move
// the pin is a question about the driver, not about the header: a binding for
// nvmlDeviceGetAccountingStats_v2 is useless against a libnvidia-ml that does
// not export it, and actively misleading if the _v2 form disagrees with the
// _v1 form this package already binds.
//
// So this test asks the library rather than the header. It resolves the four
// symbols by name out of the same shared object the package loaded, and where
// a _v2 form exists it calls both forms and compares them.
//
// What it does not do is call nvmlDeviceGetBBXTimeData_v1 or
// nvmlSystemGetCPER_v1. Both take structs that carry an NVML_STRUCT_VERSION
// stamp the caller has to set, this package has no binding and no constant for
// either, and calling them with an unstamped struct measures nothing except
// what the library does with a malformed argument. Their presence is the
// question that was asked; their behavior is a separate piece of work.
//
// Requires a GPU and a driver. Run with NVML_VERSIONED_PHYSICAL_TEST=1.
func TestPhysicalVersionedSymbols(t *testing.T) {
	if os.Getenv("NVML_VERSIONED_PHYSICAL_TEST") != "1" {
		t.Skip("set NVML_VERSIONED_PHYSICAL_TEST=1")
	}
	if err := nvml.Init(); err != nil {
		t.Fatalf("initialize nvml: %v", err)
	}
	defer func() {
		if err := nvml.Shutdown(); err != nil {
			t.Errorf("shutdown nvml: %v", err)
		}
	}()
	path := nvml.Path()
	t.Logf("loaded library: %s", path)

	// The same object the package loaded, so the answer is about the library
	// under test and not about whatever else is on the search path.
	lib, err := purego.Dlopen(path, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		t.Fatalf("dlopen %s: %v", path, err)
	}
	defer purego.Dlclose(lib)

	present := make(map[string]bool)
	for _, name := range []string{
		"nvmlDeviceGetAccountingStats_v2",
		"nvmlDeviceGetRemappedRows_v2",
		"nvmlDeviceGetBBXTimeData_v1",
		"nvmlSystemGetCPER_v1",
		// Controls. The first is bound by this package and must resolve; the
		// second must not, or dlsym is answering yes to everything and the
		// four answers above are worthless.
		"nvmlDeviceGetAccountingStats",
		"nvmlDeviceGetNoSuchThing_v9",
	} {
		_, err := purego.Dlsym(lib, name)
		present[name] = err == nil
		t.Logf("symbol %-34s present=%t", name, present[name])
	}
	if !present["nvmlDeviceGetAccountingStats"] {
		t.Fatal("nvmlDeviceGetAccountingStats did not resolve; this package binds it," +
			" so the lookup is broken rather than the library incomplete")
	}
	if present["nvmlDeviceGetNoSuchThing_v9"] {
		t.Fatal("a symbol that cannot exist resolved; the presence answers above mean nothing")
	}

	count, err := nvml.DeviceGetCount()
	if err != nil {
		t.Fatalf("device count: %v", err)
	}
	if count == 0 {
		t.Skip("no devices")
	}
	device, err := nvml.NewDeviceByIndex(0)
	if err != nil {
		t.Fatalf("device 0: %v", err)
	}
	name, err := device.GetName()
	if err != nil {
		t.Fatalf("device name: %v", err)
	}
	t.Logf("device 0: %s", name)

	t.Run("RemappedRows", func(t *testing.T) {
		if !present["nvmlDeviceGetRemappedRows_v2"] {
			t.Skip("library does not export nvmlDeviceGetRemappedRows_v2")
		}
		var corr, unc, pending, failure uint32
		corr, err := device.GetRemappedRows(&unc, &pending, &failure)
		if err != nil {
			t.Fatalf("GetRemappedRows: %v", err)
		}
		var v2 remappedRowsInfoV2
		var getV2 func(nvml.Device, *remappedRowsInfoV2) nvml.Result
		purego.RegisterLibFunc(&getV2, lib, "nvmlDeviceGetRemappedRows_v2")
		if r := getV2(device, &v2); r != nvml.Success {
			t.Fatalf("nvmlDeviceGetRemappedRows_v2: %v", r)
		}
		t.Logf("v1: corr=%d unc=%d pending=%d failure=%d", corr, unc, pending, failure)
		t.Logf("v2: corrActive=%d corrInactive=%d uncActive=%d uncInactive=%d pending=%d failure=%d",
			v2.CorrActiveRemaps, v2.CorrInactiveRemaps, v2.UncActiveRemaps,
			v2.UncInactiveRemaps, v2.BPending, v2.BFailureOccurred)
		// These two fields are the same field in both structs, so they are the
		// part a disagreement would be a real disagreement about.
		if v2.BPending != pending {
			t.Errorf("pending: v1=%d v2=%d", pending, v2.BPending)
		}
		if v2.BFailureOccurred != failure {
			t.Errorf("failure occurred: v1=%d v2=%d", failure, v2.BFailureOccurred)
		}
		// Not asserted: the header does not say that v1's corrRows is v2's
		// active plus inactive counts. It is the obvious reading and it is
		// still a guess, so it is reported and left to whoever decides the pin.
		t.Logf("UNQUALIFIED: corr v1=%d vs v2 active+inactive=%d; unc v1=%d vs v2 active+inactive=%d."+
			" The header does not state this relation; these numbers are a measurement, not a check.",
			corr, v2.CorrActiveRemaps+v2.CorrInactiveRemaps,
			unc, v2.UncActiveRemaps+v2.UncInactiveRemaps)
	})

	t.Run("AccountingStats", func(t *testing.T) {
		if !present["nvmlDeviceGetAccountingStats_v2"] {
			t.Skip("library does not export nvmlDeviceGetAccountingStats_v2")
		}
		mode, err := device.GetAccountingMode()
		if err != nil {
			t.Skipf("accounting mode unavailable: %v", err)
		}
		if mode != nvml.FeatureEnabled {
			t.Skipf("accounting is %v; enabling it needs root, and without it there are no pids to compare", mode)
		}
		pids, err := device.GetAccountingPids(make([]uint32, 256))
		if err != nil {
			t.Skipf("accounting pids: %v", err)
		}
		if len(pids) == 0 {
			t.Skip("no accounted processes; nothing to compare")
		}
		var getV2 func(nvml.Device, *accountingStatsV2) nvml.Result
		purego.RegisterLibFunc(&getV2, lib, "nvmlDeviceGetAccountingStats_v2")
		compared := 0
		for _, pid := range pids {
			var v1 nvml.AccountingStats
			if err := device.GetAccountingStats(pid, &v1); err != nil {
				t.Logf("pid %d: v1 unavailable (%v); skipped", pid, err)
				continue
			}
			v2 := accountingStatsV2{Pid: pid}
			if r := getV2(device, &v2); r != nvml.Success {
				t.Errorf("pid %d: nvmlDeviceGetAccountingStats_v2: %v", pid, r)
				continue
			}
			compared++
			t.Logf("pid=%d v1{gpu=%d mem=%d maxMem=%d time=%d start=%d running=%d}"+
				" v2{gpu=%d mem=%d maxMem=%d time=%d start=%d running=%d samples=%d sumGpu=%d sumFb=%d}",
				pid, v1.GpuUtilization, v1.MemoryUtilization, v1.MaxMemoryUsage, v1.Time,
				v1.StartTime, v1.IsRunning,
				v2.GpuUtilization, v2.MemoryUtilization, v2.MaxMemoryUsage, v2.Time,
				v2.StartTime, v2.IsRunning, v2.SampleCount, v2.SumGpuUtil, v2.SumFbUtil)
			// Start time cannot move, whatever the process does between the two
			// calls, so it is the one field a mismatch cannot be excused.
			if v1.StartTime != v2.StartTime {
				t.Errorf("pid %d: start time v1=%d v2=%d", pid, v1.StartTime, v2.StartTime)
			}
			if v1.IsRunning != v2.IsRunning {
				t.Logf("pid %d: running flag changed between the two calls (v1=%d v2=%d);"+
					" not compared further", pid, v1.IsRunning, v2.IsRunning)
				continue
			}
			if v1.IsRunning != 0 {
				t.Logf("pid %d: still running, so the counters are free to move between"+
					" the two calls; only start time is checked", pid)
				continue
			}
			// Terminated: the counters are final and the two forms must agree.
			for _, f := range []struct {
				name   string
				v1, v2 uint64
			}{
				{"gpu utilization", uint64(v1.GpuUtilization), uint64(v2.GpuUtilization)},
				{"memory utilization", uint64(v1.MemoryUtilization), uint64(v2.MemoryUtilization)},
				{"max memory usage", v1.MaxMemoryUsage, v2.MaxMemoryUsage},
				{"time", v1.Time, v2.Time},
			} {
				if f.v1 != f.v2 {
					t.Errorf("pid %d terminated: %s v1=%d v2=%d", pid, f.name, f.v1, f.v2)
				}
			}
		}
		if compared == 0 {
			t.Skip("no pid produced both forms; nothing was compared")
		}
		t.Logf("compared %d accounted process(es)", compared)
	})
}
