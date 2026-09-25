//go:build linux

package nccl_test

import (
	"errors"
	"os"
	"runtime"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nccl"
)

// NCCL gates its caller-allocated descriptors on a protocol the caller has to
// honor: stamp size, magic and version, and the library both rejects a bad
// magic and reads the version to decide which fields it may write. The four
// constructors in defaults.gen.go were generated from the header's
// *_INITIALIZER macros, which means their values are transcribed from a header
// and have never been shown to be the values the library actually accepts.
//
// This test is that showing. Every descriptor is exercised twice: once zeroed,
// which must be refused, and once from its constructor, which must not be. The
// zeroed half is the half that carries the information. A test that only used
// the constructors would pass against a library that ignored the protocol
// entirely, and would therefore not be evidence that the stamps are load
// bearing.
//
// SimInfo is the reason to check rather than assume. Its magic is 0x74685283,
// not the 0xcafebeef the other three carry, so a generator that hardcoded one
// magic would produce three working descriptors and one that fails only at run
// time on hardware nobody has.
//
// Requires one GPU and NCCL. Run with NCCL_DESCRIPTOR_PHYSICAL_TEST=1.
func TestPhysicalVersionedDescriptors(t *testing.T) {
	if os.Getenv("NCCL_DESCRIPTOR_PHYSICAL_TEST") != "1" {
		t.Skip("set NCCL_DESCRIPTOR_PHYSICAL_TEST=1")
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
	ctx, err := driver.Device(0).PrimaryCtxRetain()
	check("retain primary context", err)
	defer func() {
		if err := driver.Device(0).PrimaryCtxRelease(); err != nil {
			t.Errorf("cleanup primary context: %v", err)
		}
	}()
	check("set current context", ctx.SetCurrent())

	id, err := nccl.GetUniqueId()
	check("unique id", err)

	// Config. A zeroed descriptor carries magic 0 and size 0; the library is
	// expected to refuse it rather than read the rest of the struct.
	var zeroConfig nccl.Config
	if comm, err := nccl.CommInitRankConfig(1, id, 0, &zeroConfig); err == nil {
		if comm != 0 {
			comm.Destroy()
		}
		t.Fatal("CommInitRankConfig accepted a zeroed Config; the magic is not load bearing" +
			" and this test proves nothing about the other three descriptors")
	} else {
		t.Logf("zeroed Config rejected with %v", err)
		requireRejection(t, "Config", err)
	}

	// A fresh id: the rejected attempt may have consumed the first one.
	id, err = nccl.GetUniqueId()
	check("unique id for the accepted config", err)
	config := nccl.NewConfig()
	comm, err := nccl.CommInitRankConfig(1, id, 0, &config)
	check("CommInitRankConfig with NewConfig", err)
	if comm == 0 {
		t.Fatal("CommInitRankConfig returned a null communicator and no error")
	}
	defer func() {
		if err := comm.Destroy(); err != nil {
			t.Errorf("cleanup communicator: %v", err)
		}
	}()
	t.Logf("NewConfig accepted: size=%d magic=%#x version=%d",
		config.Size, config.Magic, config.Version)

	t.Run("CommProperties", func(t *testing.T) {
		var zero nccl.CommProperties
		if err := comm.QueryProperties(&zero); err == nil {
			t.Error("QueryProperties accepted a zeroed CommProperties")
		} else {
			t.Logf("zeroed CommProperties rejected with %v", err)
			requireRejection(t, "CommProperties", err)
		}
		props := nccl.NewCommProperties()
		if err := comm.QueryProperties(&props); err != nil {
			t.Fatalf("QueryProperties with NewCommProperties: %v", err)
		}
		if props.Rank != 0 || props.NRanks != 1 {
			t.Errorf("rank=%d nranks=%d, want 0 and 1", props.Rank, props.NRanks)
		}
		t.Logf("properties: rank=%d nranks=%d cudaDev=%d deviceApi=%t multimem=%t"+
			" nLsaTeams=%d commHash=%#x devCommRuntimeVersionSize=%d",
			props.Rank, props.NRanks, props.CudaDev, props.DeviceApiSupport,
			props.MultimemSupport, props.NLsaTeams, props.CommHash,
			props.DevCommRuntimeVersionSize)
	})

	t.Run("DevCommRequirements", func(t *testing.T) {
		var devComm nccl.DevComm
		var zero nccl.DevCommRequirements
		if err := comm.DevCommCreate(&zero, &devComm); err == nil {
			comm.DevCommDestroy(&devComm)
			t.Error("DevCommCreate accepted a zeroed DevCommRequirements")
		} else {
			t.Logf("zeroed DevCommRequirements rejected with %v", err)
			requireRejection(t, "DevCommRequirements", err)
		}
		reqs := nccl.NewDevCommRequirements()
		devComm = nccl.DevComm{}
		switch err := comm.DevCommCreate(&reqs, &devComm); {
		case err == nil:
			t.Logf("NewDevCommRequirements accepted: size=%d magic=%#x version=%d",
				reqs.Size, reqs.Magic, reqs.Version)
			if err := comm.DevCommDestroy(&devComm); err != nil {
				t.Errorf("destroy device communicator: %v", err)
			}
		case isRejection(err):
			// The descriptor was refused on the same grounds as the zeroed one,
			// so the constructor's stamp is wrong or the library disagrees with
			// the header it was generated from. Either is a finding.
			t.Errorf("DevCommCreate refused NewDevCommRequirements as malformed: %v", err)
		default:
			// Unsupported here is not a defect in the descriptor. Say so
			// explicitly rather than counting it as a pass.
			t.Logf("UNQUALIFIED: DevCommCreate did not reject the descriptor but did not"+
				" succeed either (%v); the stamp is not disproved and not confirmed"+
				" on this machine", err)
		}
	})

	t.Run("SimInfo", func(t *testing.T) {
		// Earlier subtests create communicators, and NCCL sets the current
		// device while it does. On a host with one GPU that is invisible; on
		// a host with two it leaves a context this subtest never chose, and
		// the first allocation fails with CUDA_ERROR_INVALID_CONTEXT. Choose
		// the context here rather than inheriting whatever the library left.
		check("set current context", ctx.SetCurrent())
		const count = 1024
		send, err := driver.MemAllocBytes(count * 4)
		if err != nil {
			t.Fatalf("allocate send: %v", err)
		}
		defer send.Free()
		recv, err := driver.MemAllocBytes(count * 4)
		if err != nil {
			t.Fatalf("allocate receive: %v", err)
		}
		defer recv.Free()
		host := make([]int32, count)
		for i := range host {
			host[i] = int32(i)
		}
		if err := send.CopyFromHostPtr(unsafe.Pointer(&host[0]), count*4); err != nil {
			t.Fatalf("upload: %v", err)
		}
		stream, err := driver.NewStream(0)
		if err != nil {
			t.Fatalf("create stream: %v", err)
		}
		defer stream.Destroy()

		simulate := func(info *nccl.SimInfo) error {
			if err := nccl.GroupStart(); err != nil {
				return err
			}
			if err := comm.AllReduce(send, recv, count,
				nccl.Int32, nccl.Sum, stream); err != nil {
				nccl.GroupEnd()
				return err
			}
			return nccl.GroupSimulateEnd(info)
		}

		var zero nccl.SimInfo
		if err := simulate(&zero); err == nil {
			t.Error("GroupSimulateEnd accepted a zeroed SimInfo")
		} else {
			t.Logf("zeroed SimInfo rejected with %v", err)
			requireRejection(t, "SimInfo", err)
		}

		info := nccl.NewSimInfo()
		if info.Magic != 0x74685283 {
			t.Fatalf("NewSimInfo magic = %#x, want 0x74685283", info.Magic)
		}
		if err := simulate(&info); err != nil {
			t.Fatalf("GroupSimulateEnd with NewSimInfo: %v", err)
		}
		// This communicator has one rank, and a one-rank all-reduce is a
		// local copy with no transport to model. NCCL leaves the seed in
		// place rather than estimating, so the value here says nothing about
		// whether the library writes the field at all. That question needs
		// two ranks and is answered by TestPhysicalSimulateRanks, which runs
		// both arms against one library and measures 11.9 where this one
		// measures the seed.
		if seed := nccl.NewSimInfo().EstimatedTime; info.EstimatedTime == seed {
			t.Logf("UNQUALIFIED: estimated time is still the initializer seed %v"+
				" at one rank; see TestPhysicalSimulateRanks for the two-rank arm", seed)
		} else if info.EstimatedTime < 0 {
			t.Errorf("estimated time = %v: negative and not the seed %v",
				info.EstimatedTime, seed)
		}
		t.Logf("NewSimInfo accepted: size=%d magic=%#x version=%d estimatedTime=%v",
			info.Size, info.Magic, info.Version, info.EstimatedTime)
	})
}

// isRejection reports whether err is the library refusing a malformed
// descriptor. NCCL answers a bad magic or size with ErrInvalidUsage or
// ErrInvalidArgument; anything else means the call got past validation and
// failed for its own reasons.
func isRejection(err error) bool {
	var r nccl.Result
	if !errors.As(err, &r) {
		return false
	}
	return r == nccl.ErrInvalidUsage || r == nccl.ErrInvalidArgument
}

// requireRejection reports when a zeroed descriptor was refused for some reason
// other than being malformed. The refusal still happened, so the test is not
// wrong, but it was not evidence about the protocol.
func requireRejection(t *testing.T, what string, err error) {
	t.Helper()
	if !isRejection(err) {
		t.Errorf("zeroed %s was refused with %v, not ErrInvalidUsage or"+
			" ErrInvalidArgument; that is a refusal but not a demonstration"+
			" that the magic protocol is what refused it", what, err)
	}
}
