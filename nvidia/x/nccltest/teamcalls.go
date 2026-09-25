package nccltest

import (
	"errors"
	"fmt"
	"runtime"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nccl"
)

// TeamCallStatus reports how one host call behaved against the installed NCCL.
type TeamCallStatus string

const (
	// TeamCallQualified means the call succeeded and its outputs were checked.
	TeamCallQualified TeamCallStatus = "qualified"
	// TeamCallUnavailable means the symbol is absent from the loaded library.
	TeamCallUnavailable TeamCallStatus = "unavailable"
	// TeamCallUnqualified means the symbol is present but the call was rejected.
	TeamCallUnqualified TeamCallStatus = "unqualified"
)

// TeamCallResult records one host call's qualification. Detail carries the
// rejection or unavailability reason and is empty when the call qualified.
type TeamCallResult struct {
	Name   string
	Status TeamCallStatus
	Detail string
}

// TeamCallReport is the outcome of CheckTeamCalls. Version is the NCCL runtime
// version code; DevCommCftCaps records whether a CFT-capable device
// communicator could be created.
type TeamCallReport struct {
	Version        int32
	DevCommCftCaps TeamCallResult
	Calls          []TeamCallResult
}

// Qualified reports whether every call in the report qualified.
func (r TeamCallReport) Qualified() bool {
	for _, c := range r.Calls {
		if c.Status != TeamCallQualified {
			return false
		}
	}
	return true
}

// teamBarrierCounts are the barrier counts CheckTeamCalls requests.
var teamBarrierCounts = []int32{1, 3, 7}

// CheckTeamCalls exercises the six host calls that take an nccl.Team by value
// against a real NCCL communicator on one local rank, and reports each call's
// qualification rather than failing on the first rejection. A call that the
// installed library does not export is reported unavailable; a call that is
// exported but rejected is reported unqualified.
//
// It requires a CUDA device and an NCCL runtime. The single-rank communicator
// exercises neither inter-rank transport nor multi-GPU topology.
func CheckTeamCalls(device int32) (report TeamCallReport, err error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if err := driver.Init(0); err != nil {
		return report, fmt.Errorf("driver init: %w", err)
	}
	dev := driver.Device(device)
	ctx, err := dev.PrimaryCtxRetain()
	if err != nil {
		return report, fmt.Errorf("retain primary context: %w", err)
	}
	defer func() { err = errors.Join(err, dev.PrimaryCtxRelease()) }()
	if err := ctx.SetCurrent(); err != nil {
		return report, fmt.Errorf("set context: %w", err)
	}

	version, err := nccl.GetVersion()
	if err != nil {
		return report, fmt.Errorf("nccl version: %w", err)
	}
	report.Version = version

	id, err := nccl.GetUniqueId()
	if err != nil {
		return report, fmt.Errorf("unique id: %w", err)
	}
	comm, err := nccl.CommInitRank(1, id, 0)
	if err != nil {
		return report, fmt.Errorf("init rank: %w", err)
	}
	defer func() { err = errors.Join(err, comm.Destroy()) }()

	if failures := nccl.BindFailures(); len(failures) != 0 {
		return report, fmt.Errorf("binding failures: %+v", failures)
	}

	world, lsa := comm.TeamWorld(), comm.TeamLsa()
	cft := comm.TeamCft(nccl.CftTeamFlat)
	for _, team := range []nccl.Team{world, lsa, cft} {
		if team.NRanks != 1 || team.Rank != 0 || team.Stride != 1 {
			return report, fmt.Errorf("unexpected single-rank team %+v", team)
		}
	}

	report.Calls = append(report.Calls,
		checkRankConversion("TeamRankToWorld", comm.TeamRankToWorld, world, lsa, cft),
		checkRankConversion("TeamRankToLsa", comm.TeamRankToLsa, world, lsa, cft),
		checkLsaBarrier(lsa),
		checkGinBarrier(comm, world),
		checkCftBarrier(cft),
	)

	report.DevCommCftCaps = checkDevCommCft(comm)

	leInfo, err := checkCftDeviceLeInfo(comm, cft)
	if err != nil {
		return report, err
	}
	report.Calls = append(report.Calls, leInfo)
	return report, nil
}

// checkRankConversion qualifies a rank-conversion method across every team.
func checkRankConversion(name string, convert func(nccl.Team, int32) (int32, error), teams ...nccl.Team) TeamCallResult {
	for _, team := range teams {
		got, err := convert(team, 0)
		if err != nil {
			return rejected(name, err)
		}
		if got != 0 {
			return TeamCallResult{name, TeamCallUnqualified, fmt.Sprintf("rank 0 of %+v converted to %d", team, got)}
		}
	}
	return TeamCallResult{Name: name, Status: TeamCallQualified}
}

// checkLsaBarrier qualifies LsaBarrierCreateRequirement, checking that the
// requested barrier count and the output buffer pointer reach the handle.
func checkLsaBarrier(lsa nccl.Team) TeamCallResult {
	const name = "LsaBarrierCreateRequirement"
	for _, n := range teamBarrierCounts {
		var handle nccl.LsaBarrierHandle
		var req nccl.DevResourceRequirements
		var pin runtime.Pinner
		pin.Pin(&handle)
		pin.Pin(&req)
		err := nccl.LsaBarrierCreateRequirement(lsa, n, &handle, &req)
		switch {
		case err != nil:
			pin.Unpin()
			return rejected(name, err)
		case handle.NBarriers != n:
			pin.Unpin()
			return TeamCallResult{name, TeamCallUnqualified, fmt.Sprintf("requested %d barriers, handle reports %d", n, handle.NBarriers)}
		case req.OutBufferHandle != &handle.BufHandle:
			pin.Unpin()
			return TeamCallResult{name, TeamCallUnqualified, "output buffer handle does not point into the barrier handle"}
		}
		pin.Unpin()
	}
	return TeamCallResult{Name: name, Status: TeamCallQualified}
}

// checkGinBarrier qualifies GinBarrierCreateRequirement, checking the signal
// count and the output signal pointer.
func checkGinBarrier(comm nccl.Comm, world nccl.Team) TeamCallResult {
	const name = "GinBarrierCreateRequirement"
	for _, n := range teamBarrierCounts {
		var handle nccl.GinBarrierHandle
		var req nccl.DevResourceRequirements
		var pin runtime.Pinner
		pin.Pin(&handle)
		pin.Pin(&req)
		err := comm.GinBarrierCreateRequirement(world, n, &handle, &req)
		switch {
		case err != nil:
			pin.Unpin()
			return rejected(name, err)
		case req.GinSignalCount != n:
			pin.Unpin()
			return TeamCallResult{name, TeamCallUnqualified, fmt.Sprintf("requested %d signals, requirement reports %d", n, req.GinSignalCount)}
		case req.OutGinSignalStart != &handle.Signal0:
			pin.Unpin()
			return TeamCallResult{name, TeamCallUnqualified, "output signal start does not point into the barrier handle"}
		}
		pin.Unpin()
	}
	return TeamCallResult{Name: name, Status: TeamCallQualified}
}

// checkCftBarrier qualifies CftBarrierCreateRequirement. NCCL 2.31.2 declares
// the symbol in nccl_device/cft_barrier.h but does not export it.
func checkCftBarrier(cft nccl.Team) TeamCallResult {
	const name = "CftBarrierCreateRequirement"
	if !nccl.HasCftBarrierCreateRequirement() {
		return TeamCallResult{name, TeamCallUnavailable, "symbol absent from the loaded NCCL library"}
	}
	for _, n := range teamBarrierCounts {
		var handle nccl.CftBarrierHandle
		var req nccl.DevResourceRequirements
		var pin runtime.Pinner
		pin.Pin(&handle)
		pin.Pin(&req)
		err := nccl.CftBarrierCreateRequirement(cft, n, &handle, &req)
		switch {
		case err != nil:
			pin.Unpin()
			return rejected(name, err)
		case handle.NBarriers != n:
			pin.Unpin()
			return TeamCallResult{name, TeamCallUnqualified, fmt.Sprintf("requested %d barriers, handle reports %d", n, handle.NBarriers)}
		case req.OutBufferHandle != &handle.BufHandle:
			pin.Unpin()
			return TeamCallResult{name, TeamCallUnqualified, "output buffer handle does not point into the barrier handle"}
		}
		pin.Unpin()
	}
	return TeamCallResult{Name: name, Status: TeamCallQualified}
}

// checkDevCommCft records whether a CFT-capable device communicator can be
// created. GetCftDeviceLeInfo requires one; NCCL refuses the request when a
// rank in the communicator does not support CFT.
func checkDevCommCft(comm nccl.Comm) TeamCallResult {
	const name = "DevCommCreate(CftCaps=1)"
	req := nccl.NewDevCommRequirements()
	req.CftCaps = 1
	var devComm nccl.DevComm
	if err := comm.DevCommCreate(&req, &devComm); err != nil {
		return rejected(name, err)
	}
	if err := comm.DevCommDestroy(&devComm); err != nil {
		return TeamCallResult{name, TeamCallUnqualified, fmt.Sprintf("destroy: %v", err)}
	}
	return TeamCallResult{Name: name, Status: TeamCallQualified}
}

// checkCftDeviceLeInfo qualifies Window.GetCftDeviceLeInfo over a registered
// window. When the call succeeds its result is cross-checked against
// GetPeerDeviceLeInfo for the same offset.
func checkCftDeviceLeInfo(comm nccl.Comm, cft nccl.Team) (result TeamCallResult, err error) {
	const name = "GetCftDeviceLeInfo"
	const bytes = 4096

	ptr, err := nccl.MemAlloc(bytes)
	if err != nil {
		return result, fmt.Errorf("allocate window: %w", err)
	}
	defer func() { err = errors.Join(err, nccl.MemFree(ptr)) }()

	win, err := comm.WindowRegister(ptr, bytes, 0)
	if err != nil {
		return result, fmt.Errorf("register window: %w", err)
	}
	defer func() { err = errors.Join(err, comm.WindowDeregister(win)) }()

	for _, offset := range []uint{0, 64, 1024} {
		leID, leOffset, err := win.CftDeviceLeInfo(offset, 0, cft)
		if err != nil {
			var code nccl.Result
			if !errors.As(err, &code) {
				return result, fmt.Errorf("offset %d: %w", offset, err)
			}
			return rejected(name, err), nil
		}
		peerID, peerOffset, err := win.PeerDeviceLeInfo(offset, 0)
		if err != nil {
			return result, fmt.Errorf("peer LE control at offset %d: %w", offset, err)
		}
		if leID != peerID || leOffset != peerOffset {
			return TeamCallResult{name, TeamCallUnqualified,
				fmt.Sprintf("offset %d: cft=%d/%d peer=%d/%d", offset, leID, leOffset, peerID, peerOffset)}, nil
		}
	}
	return TeamCallResult{Name: name, Status: TeamCallQualified}, nil
}

// rejected classifies a call that the library exported but refused.
func rejected(name string, err error) TeamCallResult {
	return TeamCallResult{name, TeamCallUnqualified, err.Error()}
}
