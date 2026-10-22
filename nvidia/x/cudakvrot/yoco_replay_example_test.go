package cudakvrot_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudakvrot"
)

func ExampleResolveSWABoundedReplay() {
	win, err := cudakvrot.ResolveSWABoundedReplay(300, 1, cudakvrot.SWAReplayConfig{
		WindowSize:           128,
		SpeculativeOvershoot: 6,
		PageBlockSize:        64,
	})
	if err != nil {
		panic(err)
	}
	fmt.Printf("start=%d end=%d active=%d pages=%d firstSlot=%d lastSlot=%d\n",
		win.StartPos, win.EndPos, win.ActiveTokens, win.RequiredPages,
		win.RingSlots[0], win.RingSlots[len(win.RingSlots)-1])
	// Output:
	// start=166 end=300 active=134 pages=3 firstSlot=32 lastSlot=31
}
