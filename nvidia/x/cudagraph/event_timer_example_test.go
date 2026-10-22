package cudagraph_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudagraph"
)

func ExamplePingPongTimer() {
	timer, err := cudagraph.NewPingPongTimer(cudagraph.PingPongTimerConfig{
		NumEventNodes: 2,
		HistoryCap:    4,
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	// Launch 1: arms bank 0; no prior replay exists yet, so LastDurationsMS is nil.
	if err := timer.RecordSimulatedReplay([]float64{1.25}); err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("launch=%d bank=%d prev_ms=%v\n", timer.ReplayCount(), timer.ActiveBank(), timer.LastDurationsMS())

	// Launch 2: flips to bank 1 and harvests bank 0's duration from launch 1.
	if err := timer.RecordSimulatedReplay([]float64{1.50}); err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("launch=%d bank=%d prev_ms=%v\n", timer.ReplayCount(), timer.ActiveBank(), timer.LastDurationsMS())

	// Launch 3: flips back to bank 0 and harvests bank 1's duration from launch 2.
	if err := timer.RecordSimulatedReplay([]float64{1.10}); err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("launch=%d bank=%d prev_ms=%v\n", timer.ReplayCount(), timer.ActiveBank(), timer.LastDurationsMS())

	// Output:
	// launch=1 bank=0 prev_ms=[]
	// launch=2 bank=1 prev_ms=[1.25]
	// launch=3 bank=0 prev_ms=[1.5]
}

func ExampleBucketRunner() {
	runner, err := cudagraph.NewBucketRunnerFromPolicy[string](cudagraph.BucketPolicy{
		MaxBsz: 24,
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	for _, b := range runner.WarmupOrder() {
		if err := runner.Register(b, fmt.Sprintf("exec_%d", b)); err != nil {
			fmt.Println("error:", err)
			return
		}
	}

	handle, pad, err := runner.Lookup(9)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("warmup=%v handle=%s bucket=%d pad=%d\n", runner.WarmupOrder(), handle, pad.BucketBsz, pad.PadCount)

	// Output:
	// warmup=[24 16 12 8 4 2 1] handle=exec_12 bucket=12 pad=3
}
