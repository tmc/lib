package cudahotswap

import (
	"context"
	"fmt"
)

func ExampleService_Push() {
	s := settingsExampleService() // Host loader and lease, no GPU timing.
	defer s.Close(context.Background())
	_, err := s.SetCheck("kernel", CheckConfig{Checker: func(ctx context.Context, selected State) error {
		// A real consumer launches through Service here, then compares logits.
		return nil
	}}, "application check")
	if err != nil {
		fmt.Println(err)
		return
	}
	r, err := s.Push(context.Background(), "kernel", Source{Code: "new source", Entry: "entry"}, "rewrite")
	fmt.Println(r.Published, r.State.Generation, r.Report.Outcome, err)
	// Output: true 2 ok <nil>
}

func ExamplePushResult() {
	r := PushResult{Published: true, RolledBack: true, State: State{Version: 2}, Selected: State{Version: 1}}
	fmt.Println(r.State.Version, r.Selected.Version, r.RolledBack)
	// Output: 2 1 true
}

func ExampleReplayReport() {
	r := ReplayReport{Clock: "fake_event_interval", Native: false}
	fmt.Println(r.Clock, r.Native, r.SignP == nil)
	// Output: fake_event_interval false true
}

func ExampleReplayOutputReport() {
	r := ReplayOutputReport{Arg: 1, WriteOnly: true}
	fmt.Println(r.Arg, r.WriteOnly)
	// Output: 1 true
}

func ExampleReplayTrialReport() {
	r := ReplayTrialReport{Arm: "candidate", Round: 2, Complete: true}
	fmt.Println(r.Arm, r.Round, r.DeviceNanos == nil)
	// Output: candidate 2 true
}
