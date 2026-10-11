//go:build linux

package nccl_test

import (
	"math"
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
	check := func(t *testing.T, what string, err error) {
		t.Helper()
		if err != nil {
			t.Fatalf("%s: %v", what, err)
		}
	}
	check(t, "initialize driver", driver.Init(0))
	count, err := driver.DeviceGetCount()
	check(t, "device count", err)
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
			if err := comm.AllReduce(send[i], recv[i], n, nccl.Int32, nccl.Sum, streams[i]); err != nil {
				nccl.GroupEnd()
				return info, err
			}
		}
		err := nccl.GroupSimulateEnd(&info)
		return info, err
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
					if err := recv[i].Free(); err != nil {
						t.Errorf("cleanup receive %d: %v", i, err)
					}
				}
				if send[i] != 0 {
					if err := send[i].Free(); err != nil {
						t.Errorf("cleanup send %d: %v", i, err)
					}
				}
				if streams[i] != 0 {
					if err := streams[i].Close(); err != nil {
						t.Errorf("cleanup stream %d: %v", i, err)
					}
				}
				driver.Device(int(devices[i])).PrimaryCtxRelease()
			}
			pin.Unpin()
		}
		ready := false
		defer func() {
			if !ready {
				cleanup()
			}
		}()
		for i, d := range devices {
			dev := driver.Device(int(d))
			var err error
			ctxs[i], err = dev.PrimaryCtxRetain()
			check(t, "retain primary context", err)
			check(t, "set current context", ctxs[i].SetCurrent())
			streams[i], err = driver.NewStream(0)
			check(t, "create stream", err)
			send[i], err = driver.MemAllocBytes(elems * 4)
			check(t, "allocate send", err)
			recv[i], err = driver.MemAllocBytes(elems * 4)
			check(t, "allocate receive", err)
			host := make([]int32, elems)
			for k := range host {
				host[k] = int32(10000*i + k)
			}
			check(t, "upload", send[i].CopyFromHostPtr(unsafe.Pointer(&host[0]), elems*4))
		}
		check(t, "init communicators", nccl.CommInitAll(&comms[0], int32(n), &devices[0]))
		ready = true
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
		case math.IsNaN(float64(info.EstimatedTime)) || math.IsInf(float64(info.EstimatedTime), 0):
			t.Errorf("%s: estimatedTime = %v, nonfinite estimate", arm, info.EstimatedTime)
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
		// CUDA current contexts belong to the calling OS thread.
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		comms, ctxs, send, recv, streams, cleanup := setup(t, []int32{0})
		defer cleanup()
		info, err := simulate(t, comms, ctxs, send, recv, streams, elems)
		report(t, "one rank", info, err)
	})

	t.Run("TwoRanks", func(t *testing.T) {
		// CUDA current contexts belong to the calling OS thread.
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		if count < 2 {
			t.Skipf("need two devices, host has %d; the comparison this test exists for cannot be made", count)
		}
		comms, ctxs, send, recv, streams, cleanup := setup(t, []int32{0, 1})
		defer cleanup()
		info, err := simulate(t, comms, ctxs, send, recv, streams, elems)
		report(t, "two ranks", info, err)
	})
}
