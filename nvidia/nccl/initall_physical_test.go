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

// TestPhysicalInitAll requires two GPUs and NCCL. Run with
// NCCL_INITALL_PHYSICAL_TEST=1 and an external process timeout.
func TestPhysicalInitAll(t *testing.T) {
	if os.Getenv("NCCL_INITALL_PHYSICAL_TEST") != "1" {
		t.Skip("set NCCL_INITALL_PHYSICAL_TEST=1")
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
	// Canaries surround the caller-owned two-slot output array.
	const canary = uint64(0x71e4c930fa268db5)
	storage := struct {
		before uint64
		comms  [2]nccl.Comm
		after  uint64
	}{before: canary, after: canary}
	devices := [2]int32{0, 1}
	var pin runtime.Pinner
	pin.Pin(&storage)
	pin.Pin(&devices)
	defer pin.Unpin()
	var contexts [2]driver.Context
	var streams [2]driver.Stream
	var send, recv [2]driver.DevicePtr
	var uuids [2]driver.Uuid
	defer func() {
		for i := range contexts {
			if contexts[i] == 0 {
				continue
			}
			if err := contexts[i].SetCurrent(); err != nil {
				t.Errorf("cleanup device %d context: %v", i, err)
				continue
			}
			for _, item := range []struct {
				name string
				run  func() error
			}{
				{"synchronize", driver.CtxSynchronize},
				{"communicator", func() error {
					if storage.comms[i] != 0 {
						return storage.comms[i].Destroy()
					}
					return nil
				}},
				{"receive buffer", func() error {
					if recv[i] != 0 {
						return recv[i].MemFree()
					}
					return nil
				}},
				{"send buffer", func() error {
					if send[i] != 0 {
						return send[i].MemFree()
					}
					return nil
				}},
				{"stream", func() error {
					if streams[i] != 0 {
						return streams[i].Destroy()
					}
					return nil
				}},
				{"primary context", driver.Device(i).PrimaryCtxRelease},
			} {
				if err := item.run(); err != nil {
					t.Errorf("cleanup device %d %s: %v", i, item.name, err)
				}
			}
		}
		if storage.before != canary || storage.after != canary {
			t.Error("CommInitAll overwrote canary")
		}
		if !t.Failed() {
			t.Log("checked cleanup complete; both communicator slots and canaries intact")
		}
	}()
	for i := range contexts {
		dev := driver.Device(i)
		uuid, err := dev.GetUuid()
		check("device UUID", err)
		uuids[i] = uuid
		contexts[i], err = dev.PrimaryCtxRetain()
		check("retain primary context", err)
		check("set current context", contexts[i].SetCurrent())
		streams[i], err = driver.NewStream(0)
		check("create stream", err)
	}
	if uuids[0] == uuids[1] {
		t.Fatal("device UUIDs must differ")
	}
	check("initialize two communicator slots", nccl.CommInitAll(&storage.comms[0], 2, &devices[0]))
	if storage.before != canary || storage.after != canary {
		t.Fatal("CommInitAll overwrote canary")
	}
	const count = 4096
	for i, comm := range storage.comms {
		if comm == 0 {
			t.Fatalf("communicator %d is null", i)
		}
		rank, err := comm.UserRank()
		check("rank", err)
		device, err := comm.CuDevice()
		check("communicator device", err)
		size, err := comm.Count()
		check("communicator size", err)
		if rank != int32(i) || device != devices[i] || size != 2 {
			t.Fatalf("slot %d: rank=%d device=%d count=%d", i, rank, device, size)
		}
		t.Logf("slot=%d rank=%d device=%d uuid=%x count=%d", i, rank, device, uuids[i], size)
		check("set allocation context", contexts[i].SetCurrent())
		send[i], err = driver.MemAlloc(count * 4)
		check("allocate send", err)
		recv[i], err = driver.MemAlloc(count * 4)
		check("allocate receive", err)
		host := make([]int32, count)
		for k := range host {
			host[k] = int32(10000*i + k)
		}
		check("upload", send[i].MemcpyHtoD(unsafe.Pointer(&host[0]), count*4))
		check("sentinel", recv[i].MemsetD32(0xdeadbeef, count))
	}
	check("group start", nccl.GroupStart())
	var enqueueErr error
	for i, comm := range storage.comms {
		if err := contexts[i].SetCurrent(); err != nil {
			enqueueErr = err
			break
		}
		if err := comm.AllReduce(nccl.Buffer(send[i]), nccl.Buffer(recv[i]), count, nccl.Int32, nccl.Sum, nccl.Stream(streams[i])); err != nil {
			enqueueErr = err
			break
		}
	}
	endErr := nccl.GroupEnd()
	check("enqueue all-reduce", enqueueErr)
	check("group end", endErr)
	for i := range contexts {
		check("set download context", contexts[i].SetCurrent())
		check("synchronize stream", streams[i].Synchronize())
		host := make([]int32, count)
		check("download", recv[i].MemcpyDtoH(unsafe.Pointer(&host[0]), count*4))
		for k, got := range host {
			want := int32(10000 + 2*k)
			if os.Getenv("NCCL_INITALL_BAD_ORACLE") == "1" {
				want++
			}
			if got != want {
				t.Fatalf("rank %d element %d: got %d want %d", i, k, got, want)
			}
		}
		t.Logf("all-reduce rank=%d verified=%d", i, count)
	}
}
