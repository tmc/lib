package cudashared_test

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudashared"
)

func ExampleStagedCopyFixtureSource() {
	src, err := cudashared.StagedCopyFixtureSource("sm_121a", 65)
	if err != nil {
		panic(err)
	}
	fmt.Println(strings.Contains(src, "__align__(4)"))
	// Output: true
}

func ExampleStagedCopyFixture() {
	limits, err := cudashared.LimitsFromDevice("sm_121a")
	if err != nil {
		panic(err)
	}
	p, err := cudashared.NewPlan(limits)
	if err != nil {
		panic(err)
	}
	if _, err := p.Add(cudashared.Request{Name: "stage0", Size: 260, Alignment: 32, Dynamic: true}); err != nil {
		panic(err)
	}
	src, launchBytes, err := cudashared.StagedCopyFixture(p, "sm_121a", 65)
	if err != nil {
		panic(err)
	}
	fmt.Println(launchBytes, strings.Contains(src, "__align__(32)"))
	// Output: 260 true
}
