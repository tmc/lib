package warpsynth_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/egraph/warpsynth"
)

func ExampleSynthesize() {
	s, err := warpsynth.Synthesize(warpsynth.Problem{
		Arch:          "sm_100a",
		SMEMCapBytes:  228 * 1024,
		TileSMEMBytes: 32 * 1024,
		TMALatency:    1000,
		Roles: []warpsynth.RoleDemand{
			{Kind: warpsynth.RoleTMAProducer, MinWarps: 4, MinRegs: 40, WorkCyclesTile: 400},
			{Kind: warpsynth.RoleMMAIssuer, MinWarps: 4, MinRegs: 64, WorkCyclesTile: 1000},
			{Kind: warpsynth.RoleEpilogueMath, MinWarps: 4, MinRegs: 200, WorkCyclesTile: 2400, ScalesWithWarps: true},
		},
		TMEM: []warpsynth.TMEMBufferReq{{Name: "acc", WidthCols: 256, DoubleBuf: true}},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, r := range s.Roles {
		fmt.Printf("%s: %d warps, setmaxnreg %d\n", r.Kind, r.Warps, r.Regs)
	}
	fmt.Printf("D=%d early=%v TMEM acc at %v of %d\n", s.PipelineStagesD, s.UseEarlySMEMRelease, s.TMEM[0].Cols, s.TotalTMEMCols)
	fmt.Println("deadlock-free over 1000 tiles:", warpsynth.VerifyDeadlockFreedom(s, 1000) == nil)
	// Output:
	// TMAProducer: 4 warps, setmaxnreg 40
	// MMAIssuer: 4 warps, setmaxnreg 72
	// EpilogueMath: 8 warps, setmaxnreg 200
	// D=7 early=true TMEM acc at [0 256] of 512
	// deadlock-free over 1000 tiles: true
}
