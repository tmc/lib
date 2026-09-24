//go:build linux

package nccl_test

import (
	"os"
	"runtime"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nccl"
)

// TestPhysicalSimulateRanks measures whether GroupSimulateEnd populates
// SimInfo.EstimatedTime, at one rank and at two, in one process against one
// library. The single-rank arm is the control: on a one-GPU host the estimate
// came back as the initializer's own -1 seed, which is consistent both with
// the library declining to estimate a local copy and with it never writing the
// field at all. Those two readings differ only in what the two-rank arm does,
// so both arms run here or neither number means anything.
//
// Run with NCCL_SIMULATE_PHYSICAL_TEST=1 on a host with two GPUs.
func TestPhysicalSimulateRanks(t *testing.T) {
	if os.Getenv("NCCL_SIMULATE_PHYSICAL_TEST") != "1" {
		t.Skip("set NCCL_SIMULATE_PHYSICAL_TEST=1")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	check := func(what string, err error) {
		t.Helper()
		if err != nil {
			t.Fatalf("%s: %v", what, err)
		}
	}
	check("initialize driver", driver.Init(0))
	count, err := driver.DeviceGetCount()
	check("device count", err)
	t.Logf("devices present: %d", count)

	// simulate enqueues one all-reduce per communicator inside a group and
	// ends the group in simulation mode. It returns the estimate the library
	// wrote, and the seed it was given, so a caller can tell "untouched" from
	// "written as zero".
	simulate := func(t *testing.T, comms []nccl.Comm, ctxs []driver.Context, send, recv []driver.DevicePtr, streams []driver.Stream, n uint) (nccl.SimInfo, error) {
		info := nccl.NewSimInfo()
		if err := nccl.GroupStart(); err != nil {
			return info, err
		}
		for i, comm := range comms {
			if err := ctxs[i].SetCurrent(); err != nil {
				nccl.GroupEnd()
				return info, err
			}
			if err := comm.AllReduce(nccl.Buffer(send[i]), nccl.Buffer(recv[i]), n, nccl.Int32, nccl.Sum, nccl.Stream(streams[i])); err != nil {
				nccl.GroupEnd()
				return info, err
			}
		}
		return info, nccl.GroupSimulateEnd(&info)
	}

	const elems = 4096

	setup := func(t *testing.T, devices []int32) ([]nccl.Comm, []driver.Context, []driver.DevicePtr, []driver.DevicePtr, []driver.Stream, func()) {
		t.Helper()
		n := len(devices)
		comms := make([]nccl.Comm, n)
		ctxs := make([]driver.Context, n)
		send := make([]driver.DevicePtr, n)
		recv := make([]driver.DevicePtr, n)
		streams := make([]driver.Stream, n)
		var pin runtime.Pinner
		pin.Pin(&comms[0])
		pin.Pin(&devices[0])
		cleanup := func() {
			for i := range ctxs {
				if ctxs[i] == 0 {
					continue
				}
				if err := ctxs[i].SetCurrent(); err != nil {
					t.Errorf("cleanup context %d: %v", i, err)
					continue
				}
				if err := driver.CtxSynchronize(); err != nil {
					t.Errorf("cleanup synchronize %d: %v", i, err)
				}
				if comms[i] != 0 {
					if err := comms[i].Destroy(); err != nil {
						t.Errorf("cleanup comm %d: %v", i, err)
					}
				}
				if recv[i] != 0 {
					recv[i].Free()
				}
				if send[i] != 0 {
					send[i].Free()
				}
				if streams[i] != 0 {
					streams[i].Destroy()
				}
				driver.Device(int(devices[i])).PrimaryCtxRelease()
			}
			pin.Unpin()
		}
		for i, d := range devices {
			dev := driver.Device(int(d))
			var err error
			ctxs[i], err = dev.PrimaryCtxRetain()
			check("retain primary context", err)
			check("set current context", ctxs[i].SetCurrent())
			streams[i], err = driver.NewStream(0)
			check("create stream", err)
			send[i], err = driver.MemAlloc(elems * 4)
			check("allocate send", err)
			recv[i], err = driver.MemAlloc(elems * 4)
			check("allocate receive", err)
			host := make([]int32, elems)
			for k := range host {
				host[k] = int32(10000*i + k)
			}
			check("upload", send[i].CopyFromHost(unsafe.Pointer(&host[0]), elems*4))
		}
		check("init communicators", nccl.CommInitAll(&comms[0], int32(n), &devices[0]))
		return comms, ctxs, send, recv, streams, cleanup
	}

	report := func(t *testing.T, arm string, info nccl.SimInfo, err error) {
		t.Helper()
		if err != nil {
			t.Errorf("%s: GroupSimulateEnd: %v", arm, err)
			return
		}
		seed := nccl.NewSimInfo().EstimatedTime
		switch {
		case info.EstimatedTime == seed:
			t.Logf("UNQUALIFIED %s: estimatedTime is still the initializer seed %v."+
				" The library either declined to estimate or never wrote the field;"+
				" this arm alone cannot tell those apart.", arm, seed)
		case info.EstimatedTime < 0:
			t.Errorf("%s: estimatedTime = %v, negative but not the seed %v", arm, info.EstimatedTime, seed)
		default:
			t.Logf("%s: estimatedTime = %v (library wrote the field)", arm, info.EstimatedTime)
		}
	}

	t.Run("OneRank", func(t *testing.T) {
		comms, ctxs, send, recv, streams, cleanup := setup(t, []int32{0})
		defer cleanup()
		info, err := simulate(t, comms, ctxs, send, recv, streams, elems)
		report(t, "one rank", info, err)
	})

	t.Run("TwoRanks", func(t *testing.T) {
		if count < 2 {
			t.Skipf("need two devices, host has %d; the comparison this test exists for cannot be made", count)
		}
		comms, ctxs, send, recv, streams, cleanup := setup(t, []int32{0, 1})
		defer cleanup()
		info, err := simulate(t, comms, ctxs, send, recv, streams, elems)
		report(t, "two ranks", info, err)
	})
}
