//go:build linux && cgo && nvidiampi

package custatevec_test

import (
	"os"
	"runtime"
	"testing"
	"time"
	"unsafe"

	"github.com/tmc/lib/nvidia/custatevec"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/internal/mpitest"
)

// TestDistributedSwap runs under mpiexec -n 2 with CUSTATEVEC_DISTRIBUTED_TEST=1.
// It swaps local bit 0 with global bit 2 and checks every complex amplitude.
func TestDistributedSwap(t *testing.T) {
	if os.Getenv("CUSTATEVEC_DISTRIBUTED_TEST") != "1" {
		t.Skip("set CUSTATEVEC_DISTRIBUTED_TEST=1")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	check := func(what string, err error) {
		t.Helper()
		if err != nil {
			t.Fatalf("%s: %v", what, err)
		}
	}
	clean := func(what string, err error) {
		t.Helper()
		if err != nil {
			t.Errorf("cleanup %s: %v", what, err)
		}
	}
	mpi, err := mpitest.Init()
	check("MPI init", err)
	defer func() { clean("MPI finalize", mpi.Close()) }()
	if mpi.Size != 2 {
		t.Fatalf("need 2 ranks, got %d", mpi.Size)
	}
	check("CUDA init", driver.Init(0))
	dev := driver.Device(mpi.Rank)
	uuid, err := dev.GetUuid()
	check("UUID", err)
	t.Logf("rank=%d device=%d uuid=%x", mpi.Rank, dev, uuid)
	ids := make([]byte, 32)
	check("gather UUID", mpi.AllGather(unsafe.Slice((*byte)(unsafe.Pointer(&uuid)), 16), ids))
	if string(ids[:16]) == string(ids[16:]) {
		t.Fatal("ranks must use distinct GPUs")
	}
	ctx, err := dev.PrimaryCtxRetain()
	check("retain context", err)
	defer func() { clean("release context", dev.PrimaryCtxRelease()) }()
	check("select context", ctx.SetCurrent())
	stream, err := driver.NewStream(0)
	check("stream", err)
	defer func() { clean("stream destroy", stream.Destroy()) }()
	event, err := driver.NewEvent(uint32(driver.EventInterprocess | driver.EventDisableTiming))
	check("event", err)
	defer func() { clean("event destroy", event.Destroy()) }()
	h, err := custatevec.Create()
	check("handle", err)
	defer func() { clean("handle destroy", h.Destroy()) }()
	comm, err := custatevec.NewCommunicator(h, custatevec.CommunicatorTypeOpenmpi, nil)
	check("communicator with NULL soname", err)
	defer func() { clean("communicator destroy", comm.Destroy(h)) }()
	var allocations []driver.DevicePtr
	defer func() {
		for _, p := range allocations {
			clean("device free", p.MemFree())
		}
	}()
	alloc := func(n uint) driver.DevicePtr {
		t.Helper()
		if n == 0 {
			n = 1
		}
		p, e := driver.MemAlloc(n)
		check("allocate", e)
		allocations = append(allocations, p)
		return p
	}
	state := alloc(4 * 16)
	host := make([]complex128, 4)
	for i := range host {
		g := mpi.Rank*4 + i
		host[i] = complex(float64(g+1), float64(-g))
	}
	check("upload state", state.MemcpyHtoD(unsafe.Pointer(&host[0]), 64))
	worker, extra, minTransfer, err := custatevec.NewSVSwapWorker(h, comm, uintptr(state), int32(mpi.Rank), custatevec.Event(event), custatevec.C64F, custatevec.Stream(stream))
	check("worker create", err)
	defer func() { clean("worker destroy", worker.Destroy(h)) }()
	if extra > 64<<20 || minTransfer > 64<<20 {
		t.Fatalf("workspace bound exceeded: extra=%d transfer=%d", extra, minTransfer)
	}
	extraPtr := alloc(extra)
	check("extra workspace", worker.SetExtraWorkspace(h, uintptr(extraPtr), extra))
	transfer := uint(1)
	for transfer < minTransfer {
		transfer *= 2
	}
	transferPtr := alloc(transfer)
	check("transfer workspace", worker.SetTransferWorkspace(h, uintptr(transferPtr), transfer))
	scheduler, err := custatevec.NewDistIndexBitSwapScheduler(h, 1, 2)
	check("scheduler", err)
	defer func() { clean("scheduler destroy", scheduler.Destroy(h)) }()
	pair := custatevec.Int2{X: 0, Y: 2}
	batches, err := scheduler.SetIndexBitSwaps(h, &pair, 1, nil, nil, 0)
	check("schedule", err)
	if batches == 0 || batches > 8 {
		t.Fatalf("invalid batch count %d", batches)
	}
	t.Logf("rank=%d batches=%d extra=%d transfer=%d", mpi.Rank, batches, extra, transfer)
	if os.Getenv("CUSTATEVEC_MISSING_RANK") == "1" && mpi.Rank == 1 {
		time.Sleep(60 * time.Second)
		t.Fatal("missing-rank control exceeded expected external timeout")
	}
	var total custatevec.Index
	for b := uint32(0); b < batches; b++ {
		var p custatevec.SVSwapParameters
		check("parameters", scheduler.GetParameters(h, int32(b), int32(mpi.Rank), &p))
		if p.TransferSize < 0 || p.TransferSize > 4 || p.OrgSubSVIndex != int32(mpi.Rank) || p.DstSubSVIndex < 0 || p.DstSubSVIndex >= 2 {
			t.Fatalf("invalid parameters %+v", p)
		}
		t.Logf("rank=%d batch=%d peer=%d transfer=%d type=%v", mpi.Rank, b, p.DstSubSVIndex, p.TransferSize, p.DataTransferType)
		check("set parameters", worker.SetParameters(h, &p, p.DstSubSVIndex))
		check("execute", worker.Execute(h, 0, p.TransferSize))
		total += p.TransferSize
	}
	if total == 0 {
		t.Fatal("no amplitudes transferred")
	}
	check("stream synchronize", stream.Synchronize())
	check("MPI barrier", mpi.Barrier())
	check("download state", state.MemcpyDtoH(unsafe.Pointer(&host[0]), 64))
	all := make([]complex128, 8)
	check("gather state", mpi.AllGather(unsafe.Slice((*byte)(unsafe.Pointer(&host[0])), 64), unsafe.Slice((*byte)(unsafe.Pointer(&all[0])), 128)))
	for i, got := range all {
		src := (i &^ 5) | ((i & 1) << 2) | ((i >> 2) & 1)
		want := complex(float64(src+1), float64(-src))
		if os.Getenv("CUSTATEVEC_BAD_ORACLE") == "1" {
			want++
		}
		if got != want {
			t.Fatalf("rank %d index %d got %v want %v", mpi.Rank, i, got, want)
		}
	}
	t.Logf("rank=%d verified=%d transferred=%d amplitudes=%v", mpi.Rank, len(all), total, all)
}
